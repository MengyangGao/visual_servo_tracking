from __future__ import annotations

import numpy as np

from ..perception import CameraIntrinsics, CameraObservation


def camera_point(
    point_world: np.ndarray,
    camera_position: np.ndarray,
    camera_xmat: np.ndarray,
) -> np.ndarray:
    """Transform a world point to MuJoCo camera coordinates (x right, y up, z back)."""

    point = np.asarray(point_world, dtype=float).reshape(3)
    position = np.asarray(camera_position, dtype=float).reshape(3)
    rotation_world_from_camera = np.asarray(camera_xmat, dtype=float).reshape(3, 3)
    return rotation_world_from_camera.T @ (point - position)


def normalized_pixel(pixel: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    uv = np.asarray(pixel, dtype=float).reshape(2)
    return np.array(
        [
            (uv[0] - intrinsics.cx) / intrinsics.fx,
            (uv[1] - intrinsics.cy) / intrinsics.fy,
        ],
        dtype=float,
    )


def project_world_point(
    point_world: np.ndarray, observation: CameraObservation
) -> tuple[np.ndarray, float]:
    """Project a world point using the exact convention used by MuJoCo cameras."""

    point_camera = camera_point(
        point_world, observation.camera_position, observation.camera_xmat
    )
    depth = -float(point_camera[2])
    if not np.isfinite(depth) or depth <= 1e-8:
        raise ValueError("world point is behind the camera")
    x = float(point_camera[0]) / depth
    y_down = -float(point_camera[1]) / depth
    intr = observation.intrinsics
    return np.array([intr.fx * x + intr.cx, intr.fy * y_down + intr.cy]), depth


def point_interaction_matrix(x: float, y: float, depth_m: float) -> np.ndarray:
    """Classical 2x6 point-feature interaction matrix for an eye-in-hand camera."""

    z = float(depth_m)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("interaction-matrix depth must be positive and finite")
    return np.array(
        [
            [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
            [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
        ],
        dtype=float,
    )
