from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ServoMode(str, Enum):
    """The visual quantity closed by the outer-loop controller."""

    IBVS = "ibvs"
    PBVS = "pbvs"
    HYBRID = "hybrid"


class CameraRole(str, Enum):
    EXTERNAL = "external"
    EYE_IN_HAND = "eye-in-hand"
    OBSERVER = "observer"


@dataclass(slots=True)
class FeatureObservation:
    """A detected point feature and its metric depth in one camera."""

    pixel: np.ndarray
    depth_m: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.pixel = np.asarray(self.pixel, dtype=float).reshape(2)
        if not np.isfinite(self.pixel).all():
            raise ValueError("feature pixel must be finite")
        if not np.isfinite(self.depth_m) or self.depth_m <= 0.0:
            raise ValueError("feature depth must be positive and finite")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("feature confidence must be in [0, 1]")


@dataclass(slots=True)
class ServoObjective:
    """Cartesian outer-loop command expressed in the world frame."""

    linear_velocity_world: np.ndarray
    error: np.ndarray
    mode: ServoMode
    image_error_px: float = 0.0
    position_error_m: float = 0.0

    def __post_init__(self) -> None:
        self.linear_velocity_world = np.asarray(
            self.linear_velocity_world, dtype=float
        ).reshape(3)
        self.error = np.asarray(self.error, dtype=float).reshape(-1)
        if not np.isfinite(self.linear_velocity_world).all():
            raise ValueError("servo velocity must be finite")
