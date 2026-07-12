from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class RobustPointEstimate:
    """A robust surface-point estimate and its measurement uncertainty."""

    position: np.ndarray
    covariance: np.ndarray
    world_bbox_min: np.ndarray
    world_bbox_max: np.ndarray
    support_mask: np.ndarray
    valid_fraction: float
    quality: float


def select_near_surface_support(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select the nearest statistically meaningful, connected depth layer.

    A segmentation mask can cover the visible front surface, a rear surface, and
    holes exposing the background.  Selecting a coherent near layer before
    back-projection prevents those layers from being averaged into a point which
    does not lie on any physical surface.
    """

    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask) > 0
    valid = selected & np.isfinite(depth) & (depth > 0.0)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return np.zeros(depth.shape, dtype=bool)
    if count < 12:
        return valid

    values = np.sort(depth[valid])
    typical_depth = max(float(np.median(values)), 1e-3)
    # A real discontinuity is at least 8 mm and at least 1.5% of range.  The
    # threshold intentionally exceeds ordinary MuJoCo z-buffer quantisation.
    gap_threshold = max(0.008, 0.015 * typical_depth)
    split_indices = np.nonzero(np.diff(values) > gap_threshold)[0] + 1
    clusters = np.split(values, split_indices)
    min_support = max(8, int(np.ceil(0.06 * count)))
    meaningful = [cluster for cluster in clusters if cluster.size >= min_support]
    layer = (
        min(meaningful, key=lambda cluster: float(np.median(cluster)))
        if meaningful
        else values
    )

    lo = float(np.min(layer)) - 1e-7
    hi = float(np.max(layer)) + 1e-7
    support = valid & (depth >= lo) & (depth <= hi)
    if not np.any(support):
        return valid

    # Depth layers may contain isolated pixels at the same range.  Keep one
    # spatially coherent component instead of combining unrelated surfaces.
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    if component_count <= 2:
        return support
    source_y, source_x = np.nonzero(selected)
    source_center = np.array([np.mean(source_x), np.mean(source_y)], dtype=float)
    candidates: list[tuple[float, int]] = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_support:
            continue
        distance = float(np.linalg.norm(centroids[component] - source_center))
        # Area dominates; distance deterministically breaks close ties.
        candidates.append((-float(area) + 0.01 * distance, component))
    if not candidates:
        return support
    component = min(candidates, key=lambda item: item[0])[1]
    return labels == component


def robust_point_estimate(
    points: np.ndarray, support_mask: np.ndarray, mask_area: int
) -> RobustPointEstimate:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        raise ValueError("at least one finite point is required")

    position = geometric_median(points)
    distances = np.linalg.norm(points - position, axis=1)
    if points.shape[0] >= 16:
        cutoff = float(np.percentile(distances, 95.0))
        inliers = distances <= max(cutoff, 1e-9)
        robust_points = points[inliers]
        position = geometric_median(robust_points)
    else:
        robust_points = points

    if robust_points.shape[0] >= 2:
        scatter = np.cov(robust_points, rowvar=False)
        scatter = np.asarray(scatter, dtype=np.float64).reshape(3, 3)
        covariance = scatter / max(float(robust_points.shape[0]), 1.0)
    else:
        covariance = np.zeros((3, 3), dtype=np.float64)
    # Two millimetres is a conservative floor for rendered depth and pixel
    # discretisation.  It also keeps covariance positive definite downstream.
    covariance = 0.5 * (covariance + covariance.T) + np.eye(3) * 4e-6

    support_count = int(np.count_nonzero(support_mask))
    valid_fraction = min(1.0, float(support_count) / max(float(mask_area), 1.0))
    radial_mad = (
        float(np.median(np.abs(distances - np.median(distances))))
        if distances.size
        else 0.0
    )
    scale = max(float(np.linalg.norm(position)), 0.1)
    compactness = float(np.exp(-radial_mad / max(0.006, 0.025 * scale)))
    quality = float(np.clip(valid_fraction * compactness, 0.0, 1.0))
    return RobustPointEstimate(
        position=position,
        covariance=covariance,
        world_bbox_min=np.min(robust_points, axis=0),
        world_bbox_max=np.max(robust_points, axis=0),
        support_mask=(np.asarray(support_mask) > 0).astype(np.uint8) * 255,
        valid_fraction=valid_fraction,
        quality=quality,
    )


def geometric_median(
    points: np.ndarray, *, tolerance: float = 1e-6, max_iterations: int = 64
) -> np.ndarray:
    """Return the Weiszfeld geometric median of a finite point cloud."""

    samples = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError("points must contain at least one finite 3D point")
    estimate = np.median(samples, axis=0)
    for _ in range(max_iterations):
        distances = np.linalg.norm(samples - estimate, axis=1)
        coincident = distances <= tolerance
        if np.any(coincident):
            return samples[int(np.argmax(coincident))].copy()
        weights = 1.0 / np.maximum(distances, tolerance)
        updated = np.sum(samples * weights[:, None], axis=0) / np.sum(weights)
        if float(np.linalg.norm(updated - estimate)) <= tolerance:
            return updated
        estimate = updated
    return estimate
