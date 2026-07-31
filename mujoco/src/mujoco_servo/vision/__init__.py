"""Camera geometry and visual feature utilities."""

from .camera_rig import CameraRig
from .geometry import (
    camera_point,
    normalized_pixel,
    point_interaction_matrix,
    project_world_point,
)
from .pose import PoseEstimate6D, align_rotation_to_reference, estimate_pose_6d
from .tracking import LabeledMeasurement, MultiTargetTracker, TargetTrack

__all__ = [
    "CameraRig",
    "LabeledMeasurement",
    "MultiTargetTracker",
    "PoseEstimate6D",
    "TargetTrack",
    "align_rotation_to_reference",
    "camera_point",
    "estimate_pose_6d",
    "normalized_pixel",
    "point_interaction_matrix",
    "project_world_point",
]
