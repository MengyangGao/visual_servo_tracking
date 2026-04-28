from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


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

    @property
    def radius(self) -> float:
        return 0.5 * max(self.size)


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
    smooth_target_alpha: float = 0.55


@dataclass(frozen=True)
class DemoConfig:
    target: str = "cup"
    trajectory: str = "circle"
    detector: str = "oracle"
    steps: int = 1200
    headless: bool = False
    viewer: bool = True
    realtime: bool = True
    seed: int = 7
    camera: CameraConfig = field(default_factory=CameraConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


def project_root() -> Path:
    return ROOT


def default_home_qpos() -> np.ndarray:
    return np.array([0.05, 1.2, 0.0, -1.35, 0.0, 1.75, 0.0], dtype=float)


def available_tasks() -> tuple[str, ...]:
    return ("contact", "standoff", "align-x", "align-y", "align-z")


def available_trajectories() -> tuple[str, ...]:
    return ("static", "circle", "figure-eight", "random-walk", "waypoints")
