from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TargetSpec


TARGETS: dict[str, TargetSpec] = {
    "cup": TargetSpec("cup", "cylinder", (0.075, 0.075, 0.105), (0.95, 0.32, 0.18, 1.0), ("mug", "red cup")),
    "apple": TargetSpec("apple", "sphere", (0.075, 0.075, 0.075), (0.9, 0.08, 0.10, 1.0), ("red apple",)),
    "box": TargetSpec("box", "box", (0.11, 0.085, 0.09), (0.18, 0.45, 0.92, 1.0), ("blue box", "cube")),
    "bottle": TargetSpec("bottle", "cylinder", (0.052, 0.052, 0.22), (0.10, 0.55, 0.85, 1.0), ("blue bottle",)),
    "phone": TargetSpec("phone", "box", (0.075, 0.014, 0.145), (0.08, 0.08, 0.09, 1.0), ("mobile", "black phone")),
}

BASE_POSITIONS: dict[str, np.ndarray] = {
    "cup": np.array([0.48, 0.02, 0.34], dtype=float),
    "apple": np.array([0.44, 0.13, 0.33], dtype=float),
    "box": np.array([0.50, -0.10, 0.34], dtype=float),
    "bottle": np.array([0.43, -0.18, 0.42], dtype=float),
    "phone": np.array([0.52, 0.08, 0.33], dtype=float),
}


def resolve_target(name_or_prompt: str) -> TargetSpec:
    text = " ".join(name_or_prompt.lower().strip().split())
    for key, spec in TARGETS.items():
        if key in text or any(alias in text for alias in spec.aliases):
            return spec
    return TARGETS["cup"]


def base_position(target: TargetSpec) -> np.ndarray:
    return BASE_POSITIONS.get(target.name, BASE_POSITIONS["cup"]).copy()


@dataclass
class TargetMotion:
    target: TargetSpec
    mode: str
    seed: int = 7

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._random_velocity = np.array([0.035, -0.025, 0.018], dtype=float)
        self._random_pos = base_position(self.target)
        self._last_time = 0.0
        self._waypoints = np.array(
            [
                base_position(self.target) + np.array([0.00, 0.00, 0.00]),
                base_position(self.target) + np.array([0.10, 0.05, 0.02]),
                base_position(self.target) + np.array([0.04, -0.12, -0.015]),
                base_position(self.target) + np.array([-0.08, -0.04, 0.025]),
            ],
            dtype=float,
        )

    def position(self, time_s: float) -> np.ndarray:
        mode = self.mode.strip().lower()
        base = base_position(self.target)
        t = float(max(time_s, 0.0))
        phase = (sum(ord(ch) for ch in self.target.name) % 360) * np.pi / 180.0
        if mode == "static":
            return base
        if mode == "circle":
            return base + np.array([0.075 * np.cos(0.58 * t + phase), 0.055 * np.sin(0.58 * t + phase), 0.020 * np.sin(0.30 * t)], dtype=float)
        if mode == "figure-eight":
            return base + np.array([0.085 * np.sin(0.54 * t), 0.060 * np.sin(1.08 * t + 0.4), 0.024 * np.sin(0.38 * t + phase)], dtype=float)
        if mode == "random-walk":
            return self._random_walk(t)
        if mode == "waypoints":
            return self._waypoint_position(t)
        raise ValueError(f"unknown target trajectory '{self.mode}'")

    def _random_walk(self, time_s: float) -> np.ndarray:
        dt = max(0.0, min(0.05, time_s - self._last_time))
        self._last_time = time_s
        jitter = self._rng.normal(0.0, 0.06, size=3)
        jitter[2] *= 0.35
        self._random_velocity = 0.985 * self._random_velocity + 0.015 * jitter
        self._random_pos = self._random_pos + self._random_velocity * dt
        low = base_position(self.target) + np.array([-0.13, -0.16, -0.05])
        high = base_position(self.target) + np.array([0.13, 0.16, 0.06])
        for i in range(3):
            if self._random_pos[i] < low[i] or self._random_pos[i] > high[i]:
                self._random_velocity[i] *= -0.65
        self._random_pos = np.clip(self._random_pos, low, high)
        return self._random_pos.copy()

    def _waypoint_position(self, time_s: float) -> np.ndarray:
        segment_s = 2.4
        scaled = time_s / segment_s
        index = int(np.floor(scaled)) % len(self._waypoints)
        nxt = (index + 1) % len(self._waypoints)
        local = scaled - np.floor(scaled)
        blend = 0.5 - 0.5 * np.cos(np.pi * local)
        return (1.0 - blend) * self._waypoints[index] + blend * self._waypoints[nxt]
