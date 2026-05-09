from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_HOME = ROOT / "vendor" / "mujoco_menagerie"
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


@dataclass(frozen=True)
class TargetSpec:
    name: str
    shape: str
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    aliases: tuple[str, ...] = ()
    parts: tuple["TargetPart", ...] = ()
    base_position: tuple[float, float, float] | None = None

    @property
    def radius(self) -> float:
        return 0.5 * max(self.size)


@dataclass(frozen=True)
class TargetPart:
    shape: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class ControllerConfig:
    task: str = "contact"
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
    default_target_position: tuple[float, float, float] | None = None
    detection_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    aliases: tuple[str, ...] = ()

    @property
    def dof(self) -> int:
        return len(self.joint_names)


ROBOT_SPECS: dict[str, RobotSpec] = {
    "panda": RobotSpec(
        name="panda",
        xml_path=MENAGERIE_PANDA_XML,
        asset_dir=MENAGERIE_PANDA_ASSETS,
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        actuator_names=("actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7"),
        home_qpos=(0.0, -0.6, 0.0, -2.2, 0.0, 2.4, -0.7853),
        ee_frame_name="hand",
        ee_frame_type="body_point",
        ee_frame_offset=(0.0, 0.0, 0.10),
        passive_actuator_ctrl=(("actuator8", 255.0),),
        default_target_position=(0.44, 0.13, 0.33),
        detection_bounds=((0.05, -0.55, 0.05), (0.85, 0.55, 0.85)),
        aliases=("franka", "franka-panda", "franka_emika_panda"),
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
        actuator_names=("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
        home_qpos=(-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
        ee_frame_name="attachment_site",
        ee_frame_type="site",
        default_target_position=(-0.30, 0.30, 0.33),
        detection_bounds=((-0.80, -0.55, 0.05), (0.40, 0.75, 0.95)),
        aliases=("universal-robots-ur5e", "universal_robots_ur5e", "ur"),
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
        default_target_position=(0.32, 0.10, 0.28),
        detection_bounds=((-0.35, -0.55, 0.05), (0.85, 0.55, 0.85)),
        aliases=("ufactory-lite6", "ufactory_lite6", "xarm-lite6"),
    ),
}


@dataclass(frozen=True)
class DemoConfig:
    robot: str = "panda"
    target: str = "cup"
    target_file: str | None = None
    trajectory: str = "circle"
    detector: str = "semantic"
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


def validate_config(config: DemoConfig) -> None:
    if config.steps < 0:
        raise ValueError("steps must be non-negative")
    if config.camera.width < 32 or config.camera.height < 32:
        raise ValueError("camera width and height must be at least 32 pixels")
    if not np.isfinite(config.camera_fps) or config.camera_fps <= 0.0:
        raise ValueError("camera_fps must be positive")
    if not np.isfinite(config.overlay_width_fraction) or not 0.05 <= config.overlay_width_fraction <= 0.95:
        raise ValueError("overlay_width_fraction must be in [0.05, 0.95]")
    if not np.isfinite(config.key_speed_mps) or config.key_speed_mps < 0.0:
        raise ValueError("key_speed_mps must be non-negative")
    _validate_controller_config(config.controller)
    _validate_camera_config(config.camera)
    _validate_depth_config(config.depth)


def _validate_controller_config(config: ControllerConfig) -> None:
    checks = {
        "control_hz": config.control_hz,
        "position_gain": config.position_gain,
        "damping": config.damping,
        "max_ee_speed": config.max_ee_speed,
        "max_joint_speed": config.max_joint_speed,
        "standoff_m": config.standoff_m,
        "max_angular_speed": config.max_angular_speed,
        "smooth_target_alpha": config.smooth_target_alpha,
    }
    for name, value in checks.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if config.damping <= 0.0:
        raise ValueError("damping must be positive")
    if config.max_ee_speed <= 0.0 or config.max_joint_speed <= 0.0:
        raise ValueError("max speeds must be positive")
    if config.standoff_m < 0.0:
        raise ValueError("standoff_m must be non-negative")
    if not 0.0 <= config.smooth_target_alpha <= 1.0:
        raise ValueError("smooth_target_alpha must be in [0, 1]")


def _validate_camera_config(config: CameraConfig) -> None:
    if config.width < 32 or config.height < 32:
        raise ValueError("camera width and height must be at least 32 pixels")
    if not np.isfinite(config.fovy_deg) or not 1.0 <= config.fovy_deg <= 179.0:
        raise ValueError("camera fovy_deg must be in [1, 179]")
    for name, values in {"position": config.position, "lookat": config.lookat}.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"camera {name} must contain three finite values")


def _validate_depth_config(config: DepthConfig) -> None:
    if not config.backend.strip():
        raise ValueError("depth backend must be non-empty")
    if not config.model.strip():
        raise ValueError("depth model must be non-empty")


def project_root() -> Path:
    return ROOT


def default_home_qpos() -> np.ndarray:
    raise RuntimeError("procedural robot fallback was removed; use MuJoCo Menagerie Panda")


def menagerie_home_qpos() -> np.ndarray:
    return np.array(ROBOT_SPECS["panda"].home_qpos, dtype=float)


def available_robots() -> tuple[str, ...]:
    return tuple(sorted(ROBOT_SPECS))


def resolve_robot(name: str) -> RobotSpec:
    normalized = name.strip().lower()
    for key, spec in ROBOT_SPECS.items():
        if normalized == key or normalized in spec.aliases:
            return spec
    raise ValueError(f"unknown robot '{name}'")


def available_tasks() -> tuple[str, ...]:
    return ("contact", "standoff", "front-standoff", "align-x", "align-y", "align-z")


def available_trajectories() -> tuple[str, ...]:
    return ("static", "circle", "figure-eight", "random-walk", "waypoints")


def available_depth_backends() -> tuple[str, ...]:
    return ("mujoco", "depth-anything-v2", "none")
