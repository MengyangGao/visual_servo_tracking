from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import FeatureObservation, ServoMode, ServoObjective
from ..math_utils import clamp_norm
from ..perception import CameraObservation
from ..vision.geometry import (
    normalized_pixel,
    point_interaction_matrix,
    project_world_point,
)


@dataclass(slots=True)
class VisualServoObjective:
    """Outer loop supporting PBVS, true point-feature IBVS, and a smooth hybrid."""

    mode: ServoMode | str = ServoMode.PBVS
    gain: float = 1.8
    depth_gain: float = 1.2
    max_speed_mps: float = 0.45
    hybrid_switch_m: float = 0.10

    def __post_init__(self) -> None:
        self.mode = ServoMode(self.mode)
        for name in ("gain", "depth_gain", "max_speed_mps", "hybrid_switch_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

    def compute(
        self,
        *,
        ee_position_world: np.ndarray,
        desired_position_world: np.ndarray,
        camera: CameraObservation | None = None,
        target_feature: FeatureObservation | None = None,
        camera_role: str = "external",
        desired_feature_pixel: np.ndarray | None = None,
        desired_feature_depth_m: float | None = None,
    ) -> ServoObjective:
        ee = np.asarray(ee_position_world, dtype=float).reshape(3)
        desired = np.asarray(desired_position_world, dtype=float).reshape(3)
        pbvs = self._pbvs(ee, desired)
        if self.mode is ServoMode.PBVS:
            return pbvs
        if camera is None or target_feature is None:
            raise ValueError(
                f"{self.mode.value} requires a camera observation and feature"
            )
        ibvs = self._ibvs(
            ee=ee,
            desired=desired,
            camera=camera,
            feature=target_feature,
            eye_in_hand=camera_role == "eye-in-hand",
            desired_feature_pixel=desired_feature_pixel,
            desired_feature_depth_m=desired_feature_depth_m,
        )
        if self.mode is ServoMode.IBVS:
            return ibvs
        distance = float(np.linalg.norm(desired - ee))
        # PBVS handles large displacements; IBVS takes over continuously near the goal.
        ibvs_weight = float(np.clip(1.0 - distance / self.hybrid_switch_m, 0.0, 1.0))
        velocity = (
            1.0 - ibvs_weight
        ) * pbvs.linear_velocity_world + ibvs_weight * ibvs.linear_velocity_world
        return ServoObjective(
            linear_velocity_world=clamp_norm(velocity, self.max_speed_mps),
            error=np.concatenate([pbvs.error, ibvs.error]),
            mode=ServoMode.HYBRID,
            image_error_px=ibvs.image_error_px,
            position_error_m=pbvs.position_error_m,
        )

    def _pbvs(self, ee: np.ndarray, desired: np.ndarray) -> ServoObjective:
        error = desired - ee
        return ServoObjective(
            linear_velocity_world=clamp_norm(self.gain * error, self.max_speed_mps),
            error=error,
            mode=ServoMode.PBVS,
            position_error_m=float(np.linalg.norm(error)),
        )

    def _ibvs(
        self,
        *,
        ee: np.ndarray,
        desired: np.ndarray,
        camera: CameraObservation,
        feature: FeatureObservation,
        eye_in_hand: bool,
        desired_feature_pixel: np.ndarray | None,
        desired_feature_depth_m: float | None,
    ) -> ServoObjective:
        if eye_in_hand:
            desired_pixel = (
                np.array([camera.intrinsics.cx, camera.intrinsics.cy], dtype=float)
                if desired_feature_pixel is None
                else np.asarray(desired_feature_pixel, dtype=float).reshape(2)
            )
            desired_depth = (
                feature.depth_m
                if desired_feature_depth_m is None
                else float(desired_feature_depth_m)
            )
            actual_pixel = feature.pixel
            actual_depth = feature.depth_m
        else:
            desired_pixel, desired_depth = project_world_point(desired, camera)
            actual_pixel, actual_depth = project_world_point(ee, camera)
        actual_xy = normalized_pixel(actual_pixel, camera.intrinsics)
        desired_xy = normalized_pixel(desired_pixel, camera.intrinsics)
        image_error = actual_xy - desired_xy
        depth_error = actual_depth - desired_depth
        if eye_in_hand:
            interaction = point_interaction_matrix(
                float(actual_xy[0]), float(actual_xy[1]), feature.depth_m
            )[:, :3]
            camera_velocity = (
                -self.gain * np.linalg.pinv(interaction, rcond=1e-5) @ image_error
            )
            camera_velocity[2] += self.depth_gain * depth_error
            # MuJoCo camera axes use +z backwards; the interaction matrix uses +z forward.
            camera_to_world = np.asarray(camera.camera_xmat, dtype=float).reshape(3, 3)
            camera_velocity[1] *= -1.0
            camera_velocity[2] *= -1.0
            velocity_world = camera_to_world @ camera_velocity
        else:
            # For an external camera the controlled feature is the projected EE.
            z = actual_depth
            jacobian = np.array(
                [[1.0 / z, 0.0, -actual_xy[0] / z], [0.0, 1.0 / z, -actual_xy[1] / z]],
                dtype=float,
            )
            camera_velocity = (
                -self.gain * np.linalg.pinv(jacobian, rcond=1e-5) @ image_error
            )
            camera_velocity[2] += self.depth_gain * (desired_depth - actual_depth)
            camera_to_world = np.asarray(camera.camera_xmat, dtype=float).reshape(3, 3)
            camera_velocity[1] *= -1.0
            camera_velocity[2] *= -1.0
            velocity_world = camera_to_world @ camera_velocity
        return ServoObjective(
            linear_velocity_world=clamp_norm(velocity_world, self.max_speed_mps),
            error=np.array([image_error[0], image_error[1], depth_error]),
            mode=ServoMode.IBVS,
            image_error_px=float(np.linalg.norm(actual_pixel - desired_pixel)),
            position_error_m=float(np.linalg.norm(desired - ee)),
        )
