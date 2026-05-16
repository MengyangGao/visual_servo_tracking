from __future__ import annotations

import argparse
import json
import math

from .app import run_demo
from .config import CameraConfig, ControllerConfig, DemoConfig, DepthConfig, available_depth_backends, available_robots, available_tasks, available_trajectories, validate_config
from .targets import TARGETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo visual-servo tracking demo")
    parser.add_argument("--robot", default="panda", choices=available_robots(), help="robot model")
    parser.add_argument("--target", default="cup", help="target object word or phrase, e.g. cup, capsule, hammer, red apple")
    parser.add_argument("--target-file", default=None, help="JSON file with additional target specs")
    parser.add_argument("--trajectory", default="circle", choices=available_trajectories(), help="target motion")
    parser.add_argument("--task", default="contact", choices=available_tasks(), help="servo objective")
    parser.add_argument("--detector", default="semantic", choices=("semantic", "oracle", "color"), help="perception backend; semantic is the primary path")
    parser.add_argument("--depth-backend", default="mujoco", choices=available_depth_backends(), help="depth provider for 3D target anchors")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf", help="Hugging Face model for --depth-backend depth-anything-v2")
    parser.add_argument("--depth-device", default="auto", help="device for optional learned depth backend: auto, cpu, mps, or cuda")
    parser.add_argument("--no-depth-metric-hint", action="store_true", help="do not calibrate learned depth with MuJoCo metric depth")
    parser.add_argument("--steps", type=int, default=None, help="control steps to run; defaults to 1000000 with viewer, 1200 headless")
    parser.add_argument("--headless", action="store_true", help="run without the MuJoCo viewer")
    parser.add_argument("--no-realtime", action="store_true", help="do not sleep to match wall-clock time")
    parser.add_argument("--scripted-target", action="store_true", help="disable keyboard target offsets and use only the scripted trajectory")
    parser.add_argument("--key-speed-cm-s", type=float, default=18.0, help="continuous keyboard target speed in centimeters per second")
    parser.add_argument("--semantic-interval", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--debug-perception", action="store_true", help="print semantic/depth detection diagnostics")
    parser.add_argument("--no-camera-overlay", action="store_true", help="hide the robot camera overlay in the MuJoCo viewer")
    parser.add_argument("--camera-fps", type=float, default=6.0, help="robot camera processing rate in viewer mode")
    parser.add_argument("--overlay-width-frac", type=float, default=0.42, help="fraction of viewer width used by the camera overlay")
    parser.add_argument("--standoff", type=float, default=None, help="standoff distance in meters")
    parser.add_argument("--standoff-cm", type=float, default=16.0, help="standoff distance in centimeters for standoff/front-standoff")
    parser.add_argument("--list-targets", action="store_true", help="print built-in target words and exit")
    parser.add_argument("--seed", type=int, default=7, help="random seed for random-walk trajectory")
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=320)
    return parser


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    if args.steps is not None and args.steps < 0:
        raise argparse.ArgumentTypeError("--steps must be non-negative")
    if args.camera_width < 32 or args.camera_height < 32:
        raise argparse.ArgumentTypeError("--camera-width and --camera-height must be at least 32")
    if not _is_finite(args.camera_fps) or args.camera_fps <= 0.0:
        raise argparse.ArgumentTypeError("--camera-fps must be positive")
    if not _is_finite(args.key_speed_cm_s) or args.key_speed_cm_s < 0.0:
        raise argparse.ArgumentTypeError("--key-speed-cm-s must be non-negative")
    if args.standoff is not None and (not _is_finite(args.standoff) or args.standoff < 0.0):
        raise argparse.ArgumentTypeError("--standoff must be non-negative")
    if not _is_finite(args.standoff_cm) or args.standoff_cm < 0.0:
        raise argparse.ArgumentTypeError("--standoff-cm must be non-negative")
    if not _is_finite(args.overlay_width_frac) or not 0.05 <= float(args.overlay_width_frac) <= 0.95:
        raise argparse.ArgumentTypeError("--overlay-width-frac must be in [0.05, 0.95]")
    camera = CameraConfig(width=args.camera_width, height=args.camera_height)
    standoff_m = float(args.standoff) if args.standoff is not None else float(args.standoff_cm) / 100.0
    controller = ControllerConfig(task=args.task, standoff_m=standoff_m)
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
        steps=int(args.steps) if args.steps is not None else (1200 if args.headless else 1_000_000),
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
    )
    validate_config(config)
    return config


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_targets:
        print(json.dumps(sorted(TARGETS.keys()), indent=2))
        return 0
    try:
        config = config_from_args(args)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    summary = run_demo(config)
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
