from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MENAGERIE_HOME = ROOT / "vendor" / "mujoco_menagerie"
MENAGERIE_HOME = Path(
    os.environ.get("MUJOCO_MENAGERIE_PATH", _DEFAULT_MENAGERIE_HOME)
).expanduser()
MENAGERIE_PANDA_XML = MENAGERIE_HOME / "franka_emika_panda" / "panda.xml"
MENAGERIE_PANDA_ASSETS = MENAGERIE_HOME / "franka_emika_panda" / "assets"


@dataclass(frozen=True)
class CameraConfig:
    name: str = "servo_camera"
    width: int = 424
    height: int = 320
    fovy_deg: float = 45.0
    position: tuple[float, float, float] = (0.85, -1.15, 0.85)
    lookat: tuple[float, float, float] = (0.45, 0.0, 0.35)
    mount_body: str | None = None
    rgb_noise_std: float = 0.0
    depth_noise_std: float = 0.0
    dropout_probability: float = 0.0


@dataclass(frozen=True)
class EnvironmentSpec:
    add_floor: bool = True
    add_table: bool = True
    add_lights: bool = True


@dataclass(frozen=True)
class GraspPoint:
    name: str = "center"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Direction of the final approach motion, expressed in the target frame.
    approach: tuple[float, float, float] = (0.0, 0.0, -1.0)
    width_m: float | None = None


@dataclass(frozen=True)
class TargetSpec:
    name: str
    shape: str
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    aliases: tuple[str, ...] = ()
    parts: tuple["TargetPart", ...] = ()
    base_position: tuple[float, float, float] | None = None
    mesh_path: Path | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.10
    friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)
    dynamics: str = "visual"
    grasp_points: tuple[GraspPoint, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.grasp_points:
            object.__setattr__(
                self,
                "grasp_points",
                (GraspPoint(width_m=float(min(self.size[0], self.size[1]))),),
            )

    @property
    def radius(self) -> float:
        return 0.5 * max(self.size)

    @property
    def grasp_width_m(self) -> float:
        return float(min(self.size[0], self.size[1]))


@dataclass(frozen=True)
class TargetPart:
    shape: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None
    mesh_path: Path | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ControllerConfig:
    task: str = "standoff"
    control_hz: float = 120.0
    position_gain: float = 7.0
    damping: float = 0.08
    max_ee_speed: float = 1.05
    max_joint_speed: float = 2.6
    standoff_m: float = 0.16
    align_offset_m: float = 0.0
    orientation_gain: float = 1.2
    max_angular_speed: float = 1.0
    smooth_target_alpha: float = 0.55
    actuator_mode: str = "position"
    max_joint_accel: float = 8.0
    torque_kp: float = 80.0
    torque_kd: float = 8.0
    joint_limit_margin: float = 0.05
    grasp_point: str | None = None
    grasp_approach_m: float = 0.08
    grasp_attach_distance_m: float = 0.065
    grasp_lift_m: float = 0.12
    grasp_stage_tolerance_m: float = 0.025


@dataclass(frozen=True)
class DepthConfig:
    backend: str = "mujoco"
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"
    metric_hint: bool = True


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path: Path
    asset_dir: Path
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    home_qpos: tuple[float, ...]
    ee_frame_name: str
    ee_frame_type: str
    ee_frame_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    passive_actuator_ctrl: tuple[tuple[str, float], ...] = ()
    max_gripper_width_m: float | None = None
    default_target_position: tuple[float, float, float] | None = None
    detection_bounds: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    aliases: tuple[str, ...] = ()
    tool_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_attachment_body: str | None = None
    gripper_actuator_names: tuple[str, ...] = ()
    gripper_open_ctrl: tuple[float, ...] = ()
    gripper_closed_ctrl: tuple[float, ...] = ()
    schema_version: int = 1

    @property
    def dof(self) -> int:
        return len(self.joint_names)


ROBOT_SPECS: dict[str, RobotSpec] = {
    "panda": RobotSpec(
        name="panda",
        xml_path=MENAGERIE_PANDA_XML,
        asset_dir=MENAGERIE_PANDA_ASSETS,
        joint_names=(
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ),
        actuator_names=(
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
        ),
        home_qpos=(0.0, -0.6, 0.0, -2.2, 0.0, 2.4, -0.7853),
        ee_frame_name="hand",
        ee_frame_type="body_point",
        ee_frame_offset=(0.0, 0.0, 0.10),
        passive_actuator_ctrl=(("actuator8", 255.0),),
        max_gripper_width_m=0.08,
        default_target_position=(0.55, 0.10, 0.40),
        detection_bounds=((0.05, -0.55, 0.05), (0.85, 0.55, 0.85)),
        aliases=("franka", "franka-panda", "franka_emika_panda"),
        grasp_attachment_body="hand",
        gripper_actuator_names=("actuator8",),
        gripper_open_ctrl=(255.0,),
        gripper_closed_ctrl=(0.0,),
    ),
    "ur5e": RobotSpec(
        name="ur5e",
        xml_path=MENAGERIE_HOME / "universal_robots_ur5e" / "ur5e.xml",
        asset_dir=MENAGERIE_HOME / "universal_robots_ur5e" / "assets",
        joint_names=(
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
        actuator_names=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ),
        home_qpos=(-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
        ee_frame_name="attachment_site",
        ee_frame_type="site",
        default_target_position=(-0.30, 0.45, 0.50),
        detection_bounds=((-0.80, -0.55, 0.05), (0.40, 0.75, 0.95)),
        aliases=("universal-robots-ur5e", "universal_robots_ur5e", "ur"),
        grasp_attachment_body="wrist_3_link",
    ),
    "lite6": RobotSpec(
        name="lite6",
        xml_path=MENAGERIE_HOME / "ufactory_lite6" / "lite6.xml",
        asset_dir=MENAGERIE_HOME / "ufactory_lite6" / "assets",
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        actuator_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        home_qpos=(0.0, 0.0, 1.57, 0.0, 1.57, 0.0),
        ee_frame_name="attachment_site",
        ee_frame_type="site",
        default_target_position=(0.32, 0.0, 0.38),
        detection_bounds=((-0.35, -0.55, 0.05), (0.85, 0.55, 0.85)),
        aliases=("ufactory-lite6", "ufactory_lite6", "xarm-lite6"),
        grasp_attachment_body="link6",
    ),
}


@dataclass(frozen=True)
class DemoConfig:
    robot: str = "panda"
    target: str = "cup"
    target_file: str | None = None
    trajectory: str = "circle"
    detector: str = "color"
    steps: int = 1200
    headless: bool = False
    viewer: bool = True
    realtime: bool = True
    manual_control: bool = True
    key_speed_mps: float = 0.18
    camera_overlay: bool = True
    debug_perception: bool = False
    camera_fps: float = 6.0
    overlay_width_fraction: float = 0.42
    seed: int = 7
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    robot_file: str | None = None
    perception_prompt: str | None = None
    detection_timeout_s: float = 0.75
    reacquire_confirm_frames: int = 3
    perception_latency_s: float = 0.0
    perception_jitter_s: float = 0.0
    perception_drop_probability: float = 0.0
    settling_threshold_m: float = 0.01
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)


@dataclass(frozen=True)
class ResolvedConfig:
    config: DemoConfig
    robot: RobotSpec
    target: TargetSpec
    extra_robots: dict[str, RobotSpec]
    extra_targets: dict[str, TargetSpec]


def available_tasks() -> tuple[str, ...]:
    return (
        "contact",
        "touch",
        "grasp",
        "standoff",
        "front-standoff",
        "align-x",
        "align-y",
        "align-z",
    )


def available_actuator_modes() -> tuple[str, ...]:
    return ("position", "velocity", "torque")


def available_trajectories() -> tuple[str, ...]:
    return ("static", "circle", "figure-eight", "random-walk", "waypoints")


def available_depth_backends() -> tuple[str, ...]:
    return ("mujoco", "depth-anything-v2", "none")


def available_detectors() -> tuple[str, ...]:
    return ("color", "semantic", "oracle")


def available_robots() -> tuple[str, ...]:
    return tuple(sorted(ROBOT_SPECS))


def validate_config(config: DemoConfig) -> None:
    resolve_config(config)


def resolve_config(config: DemoConfig) -> ResolvedConfig:
    _validate_config_fields(config)
    extra_robots = (
        load_robot_specs(config.robot_file) if config.robot_file is not None else {}
    )
    robot = resolve_robot(config.robot, extra_robots)
    # Keep target parsing local to avoid a module import cycle: targets owns the
    # JSON loader while TargetSpec remains part of the public config contract.
    from .targets import load_target_specs, resolve_target

    extra_targets = load_target_specs(config.target_file)
    target = resolve_target(config.target, extra_targets)
    return ResolvedConfig(config, robot, target, extra_robots, extra_targets)


def _validate_config_fields(config: DemoConfig) -> None:
    _validate_nonempty_text(config.robot, "robot")
    _validate_nonempty_text(config.target, "target")
    if config.robot_file is not None:
        _validate_nonempty_text(config.robot_file, "robot_file")
    if config.target_file is not None:
        _validate_nonempty_text(config.target_file, "target_file")
    if config.perception_prompt is not None:
        _validate_nonempty_text(config.perception_prompt, "perception_prompt")

    trajectory = _normalized_text(config.trajectory, "trajectory")
    if trajectory not in available_trajectories():
        raise ValueError(
            f"trajectory must be one of {', '.join(available_trajectories())}"
        )
    detector = _normalized_text(config.detector, "detector")
    if detector not in available_detectors():
        raise ValueError(f"detector must be one of {', '.join(available_detectors())}")

    _validate_integer(config.steps, "steps", minimum=0)
    _validate_integer(config.seed, "seed", minimum=0)
    camera_fps = _finite_number(config.camera_fps, "camera_fps")
    if camera_fps <= 0.0:
        raise ValueError("camera_fps must be positive")
    detection_timeout = _finite_number(
        config.detection_timeout_s, "detection_timeout_s"
    )
    if detection_timeout <= 0.0:
        raise ValueError("detection_timeout_s must be positive")
    _validate_integer(
        config.reacquire_confirm_frames, "reacquire_confirm_frames", minimum=1
    )
    latency = _finite_number(config.perception_latency_s, "perception_latency_s")
    jitter = _finite_number(config.perception_jitter_s, "perception_jitter_s")
    drop_probability = _finite_number(
        config.perception_drop_probability, "perception_drop_probability"
    )
    settling_threshold = _finite_number(
        config.settling_threshold_m, "settling_threshold_m"
    )
    if latency < 0.0:
        raise ValueError("perception_latency_s must be non-negative")
    if jitter < 0.0:
        raise ValueError("perception_jitter_s must be non-negative")
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("perception_drop_probability must be in [0, 1]")
    if settling_threshold <= 0.0:
        raise ValueError("settling_threshold_m must be positive")
    overlay_fraction = _finite_number(
        config.overlay_width_fraction, "overlay_width_fraction"
    )
    if not 0.05 <= overlay_fraction <= 0.95:
        raise ValueError("overlay_width_fraction must be in [0.05, 0.95]")
    key_speed = _finite_number(config.key_speed_mps, "key_speed_mps")
    if key_speed < 0.0:
        raise ValueError("key_speed_mps must be non-negative")
    for name in (
        "headless",
        "viewer",
        "realtime",
        "manual_control",
        "camera_overlay",
        "debug_perception",
    ):
        if not isinstance(getattr(config, name), (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")

    _validate_controller_config(config.controller)
    _validate_camera_config(config.camera)
    _validate_depth_config(config.depth)
    _validate_environment_spec(config.environment)


def _validate_controller_config(config: ControllerConfig) -> None:
    task = _normalized_text(config.task, "task")
    if task not in available_tasks():
        raise ValueError(f"task must be one of {', '.join(available_tasks())}")
    values = {
        "control_hz": _finite_number(config.control_hz, "control_hz"),
        "position_gain": _finite_number(config.position_gain, "position_gain"),
        "damping": _finite_number(config.damping, "damping"),
        "max_ee_speed": _finite_number(config.max_ee_speed, "max_ee_speed"),
        "max_joint_speed": _finite_number(config.max_joint_speed, "max_joint_speed"),
        "standoff_m": _finite_number(config.standoff_m, "standoff_m"),
        "align_offset_m": _finite_number(config.align_offset_m, "align_offset_m"),
        "orientation_gain": _finite_number(config.orientation_gain, "orientation_gain"),
        "max_angular_speed": _finite_number(
            config.max_angular_speed, "max_angular_speed"
        ),
        "smooth_target_alpha": _finite_number(
            config.smooth_target_alpha, "smooth_target_alpha"
        ),
        "max_joint_accel": _finite_number(config.max_joint_accel, "max_joint_accel"),
        "torque_kp": _finite_number(config.torque_kp, "torque_kp"),
        "torque_kd": _finite_number(config.torque_kd, "torque_kd"),
        "joint_limit_margin": _finite_number(
            config.joint_limit_margin, "joint_limit_margin"
        ),
        "grasp_approach_m": _finite_number(config.grasp_approach_m, "grasp_approach_m"),
        "grasp_attach_distance_m": _finite_number(
            config.grasp_attach_distance_m, "grasp_attach_distance_m"
        ),
        "grasp_lift_m": _finite_number(config.grasp_lift_m, "grasp_lift_m"),
        "grasp_stage_tolerance_m": _finite_number(
            config.grasp_stage_tolerance_m, "grasp_stage_tolerance_m"
        ),
    }
    if config.grasp_point is not None:
        _validate_nonempty_text(config.grasp_point, "grasp_point")
    actuator_mode = _normalized_text(config.actuator_mode, "actuator_mode")
    if actuator_mode not in available_actuator_modes():
        raise ValueError(
            f"actuator_mode must be one of {', '.join(available_actuator_modes())}"
        )
    if config.actuator_mode != actuator_mode:
        raise ValueError(
            "actuator_mode must be normalized lowercase without surrounding whitespace"
        )
    if values["control_hz"] <= 0.0:
        raise ValueError("control_hz must be positive")
    if values["position_gain"] < 0.0:
        raise ValueError("position_gain must be non-negative")
    if values["damping"] <= 0.0:
        raise ValueError("damping must be positive")
    if values["max_ee_speed"] <= 0.0 or values["max_joint_speed"] <= 0.0:
        raise ValueError("max speeds must be positive")
    if values["standoff_m"] < 0.0:
        raise ValueError("standoff_m must be non-negative")
    if values["orientation_gain"] < 0.0:
        raise ValueError("orientation_gain must be non-negative")
    if values["max_angular_speed"] <= 0.0:
        raise ValueError("max_angular_speed must be positive")
    if not 0.0 <= values["smooth_target_alpha"] <= 1.0:
        raise ValueError("smooth_target_alpha must be in [0, 1]")
    if values["max_joint_accel"] <= 0.0:
        raise ValueError("max_joint_accel must be positive")
    if values["torque_kp"] <= 0.0:
        raise ValueError("torque_kp must be positive")
    if values["torque_kd"] < 0.0:
        raise ValueError("torque_kd must be non-negative")
    if values["joint_limit_margin"] < 0.0:
        raise ValueError("joint_limit_margin must be non-negative")
    if values["grasp_approach_m"] <= 0.0:
        raise ValueError("grasp_approach_m must be positive")
    if values["grasp_attach_distance_m"] <= 0.0:
        raise ValueError("grasp_attach_distance_m must be positive")
    if values["grasp_lift_m"] <= 0.0:
        raise ValueError("grasp_lift_m must be positive")
    if values["grasp_stage_tolerance_m"] <= 0.0:
        raise ValueError("grasp_stage_tolerance_m must be positive")


def _validate_environment_spec(config: EnvironmentSpec) -> None:
    if not isinstance(config, EnvironmentSpec):
        raise ValueError("environment must be an EnvironmentSpec")
    for name in ("add_floor", "add_table", "add_lights"):
        if not isinstance(getattr(config, name), (bool, np.bool_)):
            raise ValueError(f"environment {name} must be a boolean")


def _validate_camera_config(config: CameraConfig) -> None:
    _validate_nonempty_text(config.name, "camera name")
    _validate_integer(config.width, "camera width", minimum=32)
    _validate_integer(config.height, "camera height", minimum=32)
    fovy = _finite_number(config.fovy_deg, "camera fovy_deg")
    if not 1.0 <= fovy <= 179.0:
        raise ValueError("camera fovy_deg must be in [1, 179]")
    for name, values in {"position": config.position, "lookat": config.lookat}.items():
        _finite_vector(values, 3, f"camera {name}")
    if (
        np.linalg.norm(
            np.asarray(config.position, dtype=float)
            - np.asarray(config.lookat, dtype=float)
        )
        < 1e-6
    ):
        raise ValueError("camera position and lookat must be distinct")
    if config.mount_body is not None:
        _validate_nonempty_text(config.mount_body, "camera mount_body")
    rgb_noise = _finite_number(config.rgb_noise_std, "camera rgb_noise_std")
    depth_noise = _finite_number(config.depth_noise_std, "camera depth_noise_std")
    dropout = _finite_number(config.dropout_probability, "camera dropout_probability")
    if rgb_noise < 0.0 or depth_noise < 0.0:
        raise ValueError("camera noise standard deviations must be non-negative")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("camera dropout_probability must be in [0, 1]")


def _validate_depth_config(config: DepthConfig) -> None:
    backend = _normalized_text(config.backend, "depth backend")
    if backend not in available_depth_backends():
        raise ValueError(
            f"depth backend must be one of {', '.join(available_depth_backends())}"
        )
    _validate_nonempty_text(config.model, "depth model")
    device = _normalized_text(config.device, "depth device")
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("depth device must be one of auto, cpu, mps, cuda")
    if config.device != device:
        raise ValueError(
            "depth device must be normalized lowercase without surrounding whitespace"
        )
    if not isinstance(config.metric_hint, (bool, np.bool_)):
        raise ValueError("depth metric_hint must be a boolean")


def _normalized_text(value: Any, field_name: str) -> str:
    _validate_nonempty_text(value, field_name)
    return value.strip().lower()


def _validate_nonempty_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _validate_integer(value: Any, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{field_name} must be {qualifier}")
    return number


def _finite_vector(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise ValueError(f"{field_name} must contain {length} finite numbers")
    result = tuple(_finite_number(item, field_name) for item in value)
    return result


_ROBOT_REQUIRED_FIELDS = {
    "name",
    "xml_path",
    "asset_dir",
    "joint_names",
    "actuator_names",
    "home_qpos",
    "ee_frame",
}
_ROBOT_OPTIONAL_FIELDS = {
    "schema_version",
    "passive_actuator_ctrl",
    "max_gripper_width_m",
    "default_target_position",
    "detection_bounds",
    "aliases",
    "tool_axis",
    "base_position",
    "grasp_attachment_body",
    "gripper_actuator_names",
    "gripper_open_ctrl",
    "gripper_closed_ctrl",
}


def load_robot_specs(path: str | Path) -> dict[str, RobotSpec]:
    source = Path(path).expanduser()
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    entries = _robot_entries(payload)
    base_dir = source.resolve().parent
    specs: dict[str, RobotSpec] = {}
    tokens: dict[str, str] = {}
    for index, entry in enumerate(entries):
        spec = _robot_from_mapping(entry, base_dir, index)
        for token in (spec.name, *spec.aliases):
            owner = tokens.get(token)
            if owner is not None:
                raise ValueError(
                    f"duplicate robot name or alias '{token}' used by '{owner}' and '{spec.name}'"
                )
            tokens[token] = spec.name
        specs[spec.name] = spec
    return specs


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite JSON number '{value}'")


def _robot_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "robots" in payload:
        unknown = set(payload) - {"robots", "schema_version"}
        if unknown:
            raise ValueError(
                f"robot file wrapper contains unknown fields: {', '.join(sorted(unknown))}"
            )
        _schema_version(payload.get("schema_version", 1), "robot file")
        entries = payload["robots"]
    elif isinstance(payload, dict):
        entries = [payload]
    else:
        raise ValueError(
            "robot file must contain a robot object, a list, or a {'robots': [...]} object"
        )
    if not isinstance(entries, list) or not entries:
        raise ValueError("robot file must contain at least one robot object")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("robot entries must be objects")
    return entries


def _robot_from_mapping(entry: dict[str, Any], base_dir: Path, index: int) -> RobotSpec:
    unknown = set(entry) - _ROBOT_REQUIRED_FIELDS - _ROBOT_OPTIONAL_FIELDS
    missing = _ROBOT_REQUIRED_FIELDS - set(entry)
    if unknown:
        raise ValueError(
            f"robot entry {index} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"robot entry {index} is missing fields: {', '.join(sorted(missing))}"
        )

    schema_version = _schema_version(
        entry.get("schema_version", 1), f"robot entry {index}"
    )
    name = _json_text(entry["name"], f"robot entry {index}.name", normalize=True)
    xml_path = _descriptor_path(entry["xml_path"], base_dir, f"robot '{name}'.xml_path")
    asset_dir = _descriptor_path(
        entry["asset_dir"], base_dir, f"robot '{name}'.asset_dir"
    )
    joint_names = _json_text_list(entry["joint_names"], f"robot '{name}'.joint_names")
    actuator_names = _json_text_list(
        entry["actuator_names"], f"robot '{name}'.actuator_names"
    )
    if len(actuator_names) != len(joint_names):
        raise ValueError(
            f"robot '{name}' actuator_names must have the same length as joint_names"
        )
    home_qpos = _json_number_vector(
        entry["home_qpos"], len(joint_names), f"robot '{name}'.home_qpos"
    )
    ee_name, ee_type, ee_offset = _parse_ee_frame(entry["ee_frame"], name)
    passive_ctrl = _parse_passive_actuator_ctrl(
        entry.get("passive_actuator_ctrl", {}), name
    )
    overlap = set(actuator_names) & {actuator_name for actuator_name, _ in passive_ctrl}
    if overlap:
        raise ValueError(
            f"robot '{name}' passive actuators overlap controlled actuators: {', '.join(sorted(overlap))}"
        )
    aliases = _json_text_list(
        entry.get("aliases", []),
        f"robot '{name}'.aliases",
        allow_empty=True,
        normalize=True,
    )
    max_gripper_width = _optional_positive_number(
        entry.get("max_gripper_width_m"), f"robot '{name}'.max_gripper_width_m"
    )
    default_target = _optional_json_vector(
        entry.get("default_target_position"),
        3,
        f"robot '{name}'.default_target_position",
    )
    bounds = _parse_detection_bounds(entry.get("detection_bounds"), name)
    tool_axis = _parse_tool_axis(entry.get("tool_axis", [0.0, 0.0, 1.0]), name)
    robot_base_position = _json_number_vector(
        entry.get("base_position", [0.0, 0.0, 0.0]),
        3,
        f"robot '{name}'.base_position",
    )
    grasp_attachment_body = entry.get("grasp_attachment_body")
    if grasp_attachment_body is not None:
        grasp_attachment_body = _json_text(
            grasp_attachment_body, f"robot '{name}'.grasp_attachment_body"
        )
    gripper_actuator_names = _json_text_list(
        entry.get("gripper_actuator_names", []),
        f"robot '{name}'.gripper_actuator_names",
        allow_empty=True,
    )
    gripper_open_ctrl = _json_optional_number_list(
        entry.get("gripper_open_ctrl", []),
        len(gripper_actuator_names),
        f"robot '{name}'.gripper_open_ctrl",
    )
    gripper_closed_ctrl = _json_optional_number_list(
        entry.get("gripper_closed_ctrl", []),
        len(gripper_actuator_names),
        f"robot '{name}'.gripper_closed_ctrl",
    )

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"robot '{name}' xml_path does not exist or is not a file: {xml_path}"
        )
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"robot '{name}' asset_dir does not exist or is not a directory: {asset_dir}"
        )
    return RobotSpec(
        name=name,
        xml_path=xml_path,
        asset_dir=asset_dir,
        joint_names=joint_names,
        actuator_names=actuator_names,
        home_qpos=home_qpos,
        ee_frame_name=ee_name,
        ee_frame_type=ee_type,
        ee_frame_offset=ee_offset,
        passive_actuator_ctrl=passive_ctrl,
        max_gripper_width_m=max_gripper_width,
        default_target_position=default_target,
        detection_bounds=bounds,
        aliases=aliases,
        tool_axis=tool_axis,
        base_position=robot_base_position,
        grasp_attachment_body=grasp_attachment_body,
        gripper_actuator_names=gripper_actuator_names,
        gripper_open_ctrl=gripper_open_ctrl,
        gripper_closed_ctrl=gripper_closed_ctrl,
        schema_version=schema_version,
    )


def _descriptor_path(value: Any, base_dir: Path, field_name: str) -> Path:
    text = _json_text(value, field_name)
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _json_text(value: Any, field_name: str, normalize: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    text = value.strip()
    return text.lower() if normalize else text


def _json_text_list(
    value: Any, field_name: str, allow_empty: bool = False, normalize: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field_name} must be {qualifier} of non-empty strings")
    result = tuple(_json_text(item, field_name, normalize=normalize) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _json_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _json_number_vector(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must contain {length} finite numbers")
    return tuple(_json_number(item, field_name) for item in value)


def _json_optional_number_list(
    value: Any, length: int, field_name: str
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of {length} finite numbers")
    if length == 0:
        if value:
            raise ValueError(f"{field_name} requires gripper_actuator_names")
        return ()
    return _json_number_vector(value, length, field_name)


def _schema_version(value: Any, field_name: str) -> int:
    version = _validate_integer(value, f"{field_name}.schema_version", minimum=1)
    if version != 1:
        raise ValueError(
            f"{field_name}.schema_version {version} is unsupported; supported versions: 1"
        )
    return version


def _optional_json_vector(
    value: Any, length: int, field_name: str
) -> tuple[float, ...] | None:
    return None if value is None else _json_number_vector(value, length, field_name)


def _optional_positive_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = _json_number(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _parse_ee_frame(
    value: Any, robot_name: str
) -> tuple[str, str, tuple[float, float, float]]:
    field_name = f"robot '{robot_name}'.ee_frame"
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    unknown = set(value) - {"name", "type", "offset"}
    missing = {"name", "type"} - set(value)
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"{field_name} is missing fields: {', '.join(sorted(missing))}"
        )
    name = _json_text(value["name"], f"{field_name}.name")
    frame_type = _json_text(value["type"], f"{field_name}.type", normalize=True)
    if frame_type not in {"site", "body", "body_point"}:
        raise ValueError(f"{field_name}.type must be one of site, body, body_point")
    offset = _json_number_vector(
        value.get("offset", [0.0, 0.0, 0.0]), 3, f"{field_name}.offset"
    )
    return name, frame_type, offset


def _parse_passive_actuator_ctrl(
    value: Any, robot_name: str
) -> tuple[tuple[str, float], ...]:
    field_name = f"robot '{robot_name}'.passive_actuator_ctrl"
    pairs: list[tuple[str, float]] = []
    if isinstance(value, dict):
        pairs = [
            (_json_text(name, field_name), _json_number(ctrl, f"{field_name}.{name}"))
            for name, ctrl in value.items()
        ]
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict) or set(item) != {"name", "value"}:
                raise ValueError(
                    f"{field_name}[{index}] must contain exactly name and value"
                )
            pairs.append(
                (
                    _json_text(item["name"], f"{field_name}[{index}].name"),
                    _json_number(item["value"], f"{field_name}[{index}].value"),
                )
            )
    else:
        raise ValueError(
            f"{field_name} must be an object map or a list of name/value objects"
        )
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicate actuator names")
    return tuple(pairs)


def _parse_detection_bounds(
    value: Any, robot_name: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if value is None:
        return None
    field_name = f"robot '{robot_name}'.detection_bounds"
    if isinstance(value, dict):
        if set(value) != {"min", "max"}:
            raise ValueError(f"{field_name} object must contain exactly min and max")
        lower_value, upper_value = value["min"], value["max"]
    elif isinstance(value, list) and len(value) == 2:
        lower_value, upper_value = value
    else:
        raise ValueError(
            f"{field_name} must be [[min...], [max...]] or a min/max object"
        )
    lower = _json_number_vector(lower_value, 3, f"{field_name}.min")
    upper = _json_number_vector(upper_value, 3, f"{field_name}.max")
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise ValueError(
            f"{field_name} min values must be strictly less than max values"
        )
    return lower, upper


def _parse_tool_axis(value: Any, robot_name: str) -> tuple[float, float, float]:
    field_name = f"robot '{robot_name}'.tool_axis"
    axis = np.asarray(_json_number_vector(value, 3, field_name), dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError(f"{field_name} must be non-zero")
    normalized = axis / norm
    return tuple(float(component) for component in normalized)


def resolve_robot(name: str, extra: dict[str, RobotSpec] | None = None) -> RobotSpec:
    normalized = _normalized_text(name, "robot")
    collections = (extra or {}, ROBOT_SPECS)
    for collection in collections:
        for spec in collection.values():
            if not isinstance(spec, RobotSpec):
                raise ValueError("extra robot mappings must contain RobotSpec values")
            if normalized == spec.name or normalized in spec.aliases:
                return spec
    known = sorted(
        {
            spec.name
            for collection in collections
            for spec in collection.values()
            if isinstance(spec, RobotSpec)
        }
    )
    raise ValueError(f"unknown robot '{name}'; available robots: {', '.join(known)}")


def project_root() -> Path:
    return ROOT


def default_home_qpos() -> np.ndarray:
    raise RuntimeError(
        "procedural robot fallback was removed; use MuJoCo Menagerie Panda"
    )


def menagerie_home_qpos() -> np.ndarray:
    return np.array(ROBOT_SPECS["panda"].home_qpos, dtype=float)
