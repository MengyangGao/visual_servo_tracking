from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(slots=True)
class TargetPrototype:
    name: str
    primitive: str
    size_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    nominal_standoff_m: float = 0.30
    aliases: tuple[str, ...] = ()


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(slots=True)
class CameraPose:
    translation_m: np.ndarray
    rotation_world_from_cam: np.ndarray

    @staticmethod
    def identity() -> "CameraPose":
        return CameraPose(
            translation_m=np.zeros(3, dtype=float),
            rotation_world_from_cam=np.eye(3, dtype=float),
        )


@dataclass(slots=True)
class Detection:
    success: bool
    prompt: str
    label: str
    score: float
    bbox_xyxy: np.ndarray
    centroid_px: np.ndarray
    mask: Optional[np.ndarray] = None
    mask_area_px: int = 0
    estimated_distance_m: Optional[float] = None
    backend: str = ""
    track_id: Optional[str] = None
    target_position_world: Optional[np.ndarray] = None


@dataclass(slots=True)
class ServoTelemetry:
    step: int
    prompt: str
    backend: str
    qpos: np.ndarray
    qvel: np.ndarray
    ee_position_m: np.ndarray
    ee_orientation_wxyz: np.ndarray
    target_position_m: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    detection_score: float


@dataclass(slots=True)
class CameraFrame:
    image_bgr: np.ndarray
    timestamp_s: float
    device_index: int
    backend_name: str


@dataclass(slots=True)
class AppSettings:
    prompt: str = "cup"
    backend: str = "auto"
    mode: str = "sim"
    run_mode: str = "auto"
    vision_preset: str = "default"
    max_steps: int = 240
    control_rate_hz: float = 30.0
    camera_index: Optional[int] = None
    camera_width: int = 1280
    camera_height: int = 720
    robot_view_width: int = 640
    robot_view_height: int = 480
    show_view: bool = True
    record: bool = False
    output_dir: Path = Path("outputs")
    robot_scene_path: Optional[Path] = None
    use_reference_robot: bool = True


@dataclass(slots=True)
class SceneBundle:
    model: object
    data: object
    target_proto: TargetPrototype
    ee_body_name: str
    actuator_names: tuple[str, ...]
    camera_intrinsics: CameraIntrinsics
    camera_pose: CameraPose
