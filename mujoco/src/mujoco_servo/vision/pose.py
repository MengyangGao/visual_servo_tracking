from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..perception import CameraObservation


@dataclass(slots=True, frozen=True)
class PoseEstimate6D:
    position_world: np.ndarray
    rotation_world: np.ndarray
    extents_m: np.ndarray
    point_count: int
    quality: float


def estimate_pose_6d(
    observation: CameraObservation,
    mask: np.ndarray,
    *,
    minimum_points: int = 40,
) -> PoseEstimate6D | None:
    """Estimate an oriented 3D box from a segmented metric-depth point cloud."""
    binary = np.asarray(mask) > 0
    depth = np.asarray(observation.depth_m, dtype=float)
    valid = binary & np.isfinite(depth) & (depth > 0.0)
    rows, cols = np.nonzero(valid)
    if rows.size < minimum_points:
        return None
    z = depth[rows, cols]
    intr = observation.intrinsics
    x = (cols - intr.cx) * z / intr.fx
    y = -(rows - intr.cy) * z / intr.fy
    points_camera = np.column_stack([x, y, -z])
    rotation_wc = np.asarray(observation.camera_xmat, dtype=float).reshape(3, 3)
    points_world = (
        np.asarray(observation.camera_position, dtype=float).reshape(1, 3)
        + points_camera @ rotation_wc.T
    )
    center = np.median(points_world, axis=0)
    centered = points_world - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    local = centered @ rotation
    lower, upper = np.percentile(local, [2.0, 98.0], axis=0)
    extents = np.maximum(upper - lower, 1e-6)
    coverage = rows.size / max(1, int(binary.sum()))
    anisotropy = 1.0 - float(
        np.clip(eigenvalues[order][-1] / max(eigenvalues[order][0], 1e-12), 0.0, 1.0)
    )
    return PoseEstimate6D(
        position_world=center,
        rotation_world=rotation,
        extents_m=extents,
        point_count=int(rows.size),
        quality=float(np.clip(coverage * (0.5 + 0.5 * anisotropy), 0.0, 1.0)),
    )
