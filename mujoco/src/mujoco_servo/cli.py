from __future__ import annotations

import argparse
import json
import math

from .app import run_demo
from .config import (
    CameraConfig,
    ControllerConfig,
    DemoConfig,
    DepthConfig,
    EnvironmentSpec,
    available_actuator_modes,
    available_depth_backends,
    available_detectors,
    available_robots,
    available_tasks,
    available_trajectories,
    validate_config,
)
from .targets import TARGETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo visual-servo tracking demo")
    parser.add_argument(
        "--robot",
        default="panda",
        help=f"robot model or alias; built-ins: {', '.join(available_robots())}",
    )
    parser.add_argument(
        "--robot-file",
        default=None,
        help="strict JSON robot descriptor; relative asset paths resolve from this file",
    )
    parser.add_argument(
        "--target",
        default="cup",
        help="target object word or phrase, e.g. cup, capsule, hammer, red apple",
    )
    parser.add_argument(
        "--target-file", default=None, help="JSON file with additional target specs"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="perception prompt; defaults to the selected target name",
    )
    parser.add_argument(
        "--trajectory",
        default="circle",
        choices=available_trajectories(),
        help="target motion",
    )
    parser.add_argument(
        "--task",
        default="front-standoff",
        choices=available_tasks(),
        help="servo objective",
    )
    parser.add_argument(
        "--detector",
        default="color",
        choices=available_detectors(),
        help="perception backend; color works without optional model downloads",
    )
    parser.add_argument(
        "--depth-backend",
        default="mujoco",
        choices=available_depth_backends(),
        help="depth provider for 3D target anchors",
    )
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face model for --depth-backend depth-anything-v2",
    )
    parser.add_argument(
        "--depth-device",
        default="auto",
        type=_lower_text,
        choices=("auto", "cpu", "mps", "cuda"),
        help="device for optional learned depth backend",
    )
    parser.add_argument(
        "--no-depth-metric-hint",
        action="store_true",
        help="do not calibrate learned depth with MuJoCo metric depth",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="control steps to run; defaults to 1000000 with viewer, 1200 headless",
    )
    parser.add_argument(
        "--headless", action="store_true", help="run without the MuJoCo viewer"
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="do not sleep to match wall-clock time",
    )
    parser.add_argument(
        "--scripted-target",
        action="store_true",
        help="disable keyboard target offsets and use only the scripted trajectory",
    )
    parser.add_argument(
        "--key-speed-cm-s",
        type=float,
        default=18.0,
        help="continuous keyboard target speed in centimeters per second",
    )
    parser.add_argument(
        "--debug-perception",
        action="store_true",
        help="print semantic/depth detection diagnostics",
    )
    parser.add_argument(
        "--no-camera-overlay",
        action="store_true",
        help="hide the robot camera overlay in the MuJoCo viewer",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=6.0,
        help="robot camera processing rate in viewer mode",
    )
    parser.add_argument(
        "--detection-timeout",
        type=float,
        default=0.75,
        help="seconds before a stale visual target is discarded",
    )
    parser.add_argument(
        "--overlay-width-frac",
        type=float,
        default=0.42,
        help="fraction of viewer width used by the camera overlay",
    )
    parser.add_argument(
        "--standoff", type=float, default=None, help="standoff distance in meters"
    )
    parser.add_argument(
        "--standoff-cm",
        type=float,
        default=16.0,
        help="standoff distance in centimeters for standoff/front-standoff",
    )
    parser.add_argument(
        "--control-hz", type=float, default=120.0, help="controller update rate"
    )
    parser.add_argument(
        "--actuator-mode", choices=available_actuator_modes(), default="position"
    )
    parser.add_argument(
        "--max-joint-accel",
        type=float,
        default=8.0,
        help="joint acceleration limit in rad/s^2 or m/s^2",
    )
    parser.add_argument(
        "--torque-kp",
        type=float,
        default=80.0,
        help="joint-space torque controller proportional gain",
    )
    parser.add_argument(
        "--torque-kd",
        type=float,
        default=8.0,
        help="joint-space torque controller derivative gain",
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.05,
        help="joint-limit avoidance margin",
    )
    parser.add_argument(
        "--grasp-point",
        default=None,
        help="named target grasp point; defaults to the first descriptor entry",
    )
    parser.add_argument(
        "--grasp-approach",
        type=float,
        default=0.08,
        help="pre-grasp approach distance in meters",
    )
    parser.add_argument(
        "--grasp-attach-distance",
        type=float,
        default=0.065,
        help="maximum weld activation distance in meters",
    )
    parser.add_argument(
        "--grasp-lift",
        type=float,
        default=0.12,
        help="vertical lift distance after attachment in meters",
    )
    parser.add_argument(
        "--grasp-stage-tolerance",
        type=float,
        default=0.025,
        help="pre-grasp stage transition tolerance in meters",
    )
    parser.add_argument(
        "--reacquire-confirm-frames",
        type=int,
        default=3,
        help="consecutive detections required after target loss",
    )
    parser.add_argument(
        "--perception-latency",
        type=float,
        default=0.0,
        help="simulated perception latency in seconds",
    )
    parser.add_argument(
        "--perception-jitter",
        type=float,
        default=0.0,
        help="non-negative perception latency jitter in seconds",
    )
    parser.add_argument(
        "--perception-drop-probability",
        type=float,
        default=0.0,
        help="simulated detection drop probability",
    )
    parser.add_argument(
        "--settling-threshold",
        type=float,
        default=0.01,
        help="task settling threshold in meters",
    )
    parser.add_argument(
        "--no-default-floor",
        action="store_true",
        help="do not inject the default floor",
    )
    parser.add_argument(
        "--no-default-table",
        action="store_true",
        help="do not inject the default table",
    )
    parser.add_argument(
        "--no-default-lights",
        action="store_true",
        help="do not inject the default lights",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="print built-in target words and exit",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="random seed for random-walk trajectory"
    )
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=320)
    parser.add_argument(
        "--camera-mount-body",
        default=None,
        help="robot body for an eye-in-hand camera; default is world-fixed",
    )
    parser.add_argument(
        "--rgb-noise-std",
        type=float,
        default=0.0,
        help="rendered RGB Gaussian noise standard deviation",
    )
    parser.add_argument(
        "--depth-noise-std",
        type=float,
        default=0.0,
        help="rendered depth Gaussian noise standard deviation in meters",
    )
    parser.add_argument(
        "--camera-dropout-probability",
        type=float,
        default=0.0,
        help="camera frame dropout probability",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    if args.steps is not None and args.steps < 0:
        raise argparse.ArgumentTypeError("--steps must be non-negative")
    if args.camera_width < 32 or args.camera_height < 32:
        raise argparse.ArgumentTypeError(
            "--camera-width and --camera-height must be at least 32"
        )
    if not _is_finite(args.camera_fps) or args.camera_fps <= 0.0:
        raise argparse.ArgumentTypeError("--camera-fps must be positive")
    if not _is_finite(args.detection_timeout) or args.detection_timeout <= 0.0:
        raise argparse.ArgumentTypeError("--detection-timeout must be positive")
    if not _is_finite(args.key_speed_cm_s) or args.key_speed_cm_s < 0.0:
        raise argparse.ArgumentTypeError("--key-speed-cm-s must be non-negative")
    if args.standoff is not None and (
        not _is_finite(args.standoff) or args.standoff < 0.0
    ):
        raise argparse.ArgumentTypeError("--standoff must be non-negative")
    if not _is_finite(args.standoff_cm) or args.standoff_cm < 0.0:
        raise argparse.ArgumentTypeError("--standoff-cm must be non-negative")
    if (
        not _is_finite(args.overlay_width_frac)
        or not 0.05 <= float(args.overlay_width_frac) <= 0.95
    ):
        raise argparse.ArgumentTypeError("--overlay-width-frac must be in [0.05, 0.95]")
    positive_values = {
        "--control-hz": args.control_hz,
        "--max-joint-accel": args.max_joint_accel,
        "--torque-kp": args.torque_kp,
        "--settling-threshold": args.settling_threshold,
        "--grasp-approach": args.grasp_approach,
        "--grasp-attach-distance": args.grasp_attach_distance,
        "--grasp-lift": args.grasp_lift,
        "--grasp-stage-tolerance": args.grasp_stage_tolerance,
    }
    for option, value in positive_values.items():
        if not _is_finite(value) or value <= 0.0:
            raise argparse.ArgumentTypeError(f"{option} must be positive")
    nonnegative_values = {
        "--torque-kd": args.torque_kd,
        "--joint-limit-margin": args.joint_limit_margin,
        "--perception-latency": args.perception_latency,
        "--perception-jitter": args.perception_jitter,
    }
    for option, value in nonnegative_values.items():
        if not _is_finite(value) or value < 0.0:
            raise argparse.ArgumentTypeError(f"{option} must be non-negative")
    if args.reacquire_confirm_frames < 1:
        raise argparse.ArgumentTypeError(
            "--reacquire-confirm-frames must be at least 1"
        )
    if (
        not _is_finite(args.perception_drop_probability)
        or not 0.0 <= args.perception_drop_probability <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "--perception-drop-probability must be in [0, 1]"
        )
    for option, value in (
        ("--rgb-noise-std", args.rgb_noise_std),
        ("--depth-noise-std", args.depth_noise_std),
    ):
        if not _is_finite(value) or value < 0.0:
            raise argparse.ArgumentTypeError(f"{option} must be non-negative")
    if (
        not _is_finite(args.camera_dropout_probability)
        or not 0.0 <= args.camera_dropout_probability <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "--camera-dropout-probability must be in [0, 1]"
        )
    camera = CameraConfig(
        width=args.camera_width,
        height=args.camera_height,
        mount_body=None
        if args.camera_mount_body is None
        else args.camera_mount_body.strip(),
        rgb_noise_std=float(args.rgb_noise_std),
        depth_noise_std=float(args.depth_noise_std),
        dropout_probability=float(args.camera_dropout_probability),
    )
    standoff_m = (
        float(args.standoff)
        if args.standoff is not None
        else float(args.standoff_cm) / 100.0
    )
    controller = ControllerConfig(
        task=args.task,
        standoff_m=standoff_m,
        control_hz=float(args.control_hz),
        actuator_mode=args.actuator_mode,
        max_joint_accel=float(args.max_joint_accel),
        torque_kp=float(args.torque_kp),
        torque_kd=float(args.torque_kd),
        joint_limit_margin=float(args.joint_limit_margin),
        grasp_point=None if args.grasp_point is None else args.grasp_point.strip(),
        grasp_approach_m=float(args.grasp_approach),
        grasp_attach_distance_m=float(args.grasp_attach_distance),
        grasp_lift_m=float(args.grasp_lift),
        grasp_stage_tolerance_m=float(args.grasp_stage_tolerance),
    )
    depth = DepthConfig(
        backend=args.depth_backend,
        model=args.depth_model,
        device=args.depth_device,
        metric_hint=not args.no_depth_metric_hint,
    )
    config = DemoConfig(
        robot=args.robot,
        target=args.target,
        target_file=args.target_file,
        trajectory=args.trajectory,
        detector=args.detector,
        steps=int(args.steps)
        if args.steps is not None
        else (1200 if args.headless else 1_000_000),
        headless=args.headless,
        viewer=not args.headless,
        realtime=not args.no_realtime,
        manual_control=not args.scripted_target,
        key_speed_mps=float(args.key_speed_cm_s) / 100.0,
        camera_overlay=not args.no_camera_overlay,
        debug_perception=bool(args.debug_perception),
        camera_fps=float(args.camera_fps),
        overlay_width_fraction=float(args.overlay_width_frac),
        seed=args.seed,
        camera=camera,
        depth=depth,
        controller=controller,
        robot_file=args.robot_file,
        perception_prompt=None if args.prompt is None else args.prompt.strip(),
        detection_timeout_s=float(args.detection_timeout),
        reacquire_confirm_frames=int(args.reacquire_confirm_frames),
        perception_latency_s=float(args.perception_latency),
        perception_jitter_s=float(args.perception_jitter),
        perception_drop_probability=float(args.perception_drop_probability),
        settling_threshold_m=float(args.settling_threshold),
        environment=EnvironmentSpec(
            add_floor=not args.no_default_floor,
            add_table=not args.no_default_table,
            add_lights=not args.no_default_lights,
        ),
    )
    validate_config(config)
    return config


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _lower_text(value: str) -> str:
    return value.strip().lower()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_targets:
        print(json.dumps(sorted(TARGETS.keys()), indent=2))
        return 0
    try:
        config = config_from_args(args)
        summary = run_demo(config)
    except (
        ValueError,
        argparse.ArgumentTypeError,
        RuntimeError,
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
