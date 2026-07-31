from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WorkSurface:
    center_xy: tuple[float, float]
    half_size_xy: tuple[float, float]
    top_z: float

    def contains(self, point: np.ndarray, *, inset_m: float = 0.0) -> bool:
        value = np.asarray(point, dtype=float).reshape(3)
        center = np.asarray(self.center_xy, dtype=float)
        half = np.asarray(self.half_size_xy, dtype=float) - float(inset_m)
        return bool(np.all(half >= 0.0) and np.all(np.abs(value[:2] - center) <= half))

    def clamp_xy(self, point: np.ndarray, *, inset_m: float) -> np.ndarray:
        value = np.asarray(point, dtype=float).reshape(3).copy()
        center = np.asarray(self.center_xy, dtype=float)
        half = np.asarray(self.half_size_xy, dtype=float) - float(inset_m)
        if np.any(half < 0.0):
            raise ValueError("object and placement margin do not fit on work surface")
        value[:2] = np.clip(value[:2], center - half, center + half)
        return value


@dataclass(frozen=True, slots=True)
class CartesianPathCheck:
    valid: bool
    reason: str = ""


class CartesianPathValidator:
    """Conservative sampled validation for policy-level Cartesian segments."""

    def __init__(
        self,
        workspace_min: np.ndarray,
        workspace_max: np.ndarray,
        *,
        support_z: float,
        minimum_clearance_m: float = 0.005,
        sample_spacing_m: float = 0.02,
    ) -> None:
        self.minimum = np.asarray(workspace_min, dtype=float).reshape(3)
        self.maximum = np.asarray(workspace_max, dtype=float).reshape(3)
        self.support_z = float(support_z)
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.sample_spacing_m = float(sample_spacing_m)
        if np.any(self.maximum <= self.minimum):
            raise ValueError("workspace maximum must exceed minimum")

    def check(self, start: np.ndarray, end: np.ndarray) -> CartesianPathCheck:
        start_value = np.asarray(start, dtype=float).reshape(3)
        end_value = np.asarray(end, dtype=float).reshape(3)
        if not np.isfinite(start_value).all() or not np.isfinite(end_value).all():
            return CartesianPathCheck(False, "path contains non-finite coordinates")
        distance = float(np.linalg.norm(end_value - start_value))
        count = max(2, int(np.ceil(distance / self.sample_spacing_m)) + 1)
        samples = np.linspace(start_value, end_value, count)
        if np.any(samples < self.minimum) or np.any(samples > self.maximum):
            return CartesianPathCheck(False, "path leaves robot workspace")
        if float(np.min(samples[:, 2])) < self.support_z + self.minimum_clearance_m:
            return CartesianPathCheck(False, "path violates work-surface clearance")
        return CartesianPathCheck(True)
