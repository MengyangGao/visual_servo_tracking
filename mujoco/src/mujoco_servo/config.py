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
    width: int = 640
    height: int = 480
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
    camera_fps: float = 6.0
    overlay_width_fraction: float = 0.42
    seed: int = 7
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


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
