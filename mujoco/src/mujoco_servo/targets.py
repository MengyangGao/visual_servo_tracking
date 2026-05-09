from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TargetPart, TargetSpec


TARGETS: dict[str, TargetSpec] = {
    "cup": TargetSpec(
        "cup",
        "compound",
        (0.095, 0.075, 0.105),
        (0.95, 0.32, 0.18, 1.0),
        ("mug", "red cup"),
        parts=(
            TargetPart("cylinder", (0.075, 0.075, 0.105), rgba=(0.95, 0.32, 0.18, 1.0)),
            TargetPart("capsule", (0.014, 0.014, 0.070), pos=(0.048, 0.0, 0.005), rgba=(0.95, 0.32, 0.18, 1.0), quat=(0.7071, 0.0, 0.7071, 0.0)),
        ),
    ),
    "apple": TargetSpec(
        "apple",
        "compound",
        (0.085, 0.085, 0.105),
        (0.9, 0.08, 0.10, 1.0),
        ("red apple", "fruit"),
        parts=(
            TargetPart("sphere", (0.080, 0.080, 0.080), rgba=(0.9, 0.08, 0.10, 1.0)),
            TargetPart("capsule", (0.010, 0.010, 0.038), pos=(0.0, 0.0, 0.052), rgba=(0.34, 0.18, 0.08, 1.0)),
            TargetPart("box", (0.035, 0.014, 0.006), pos=(0.020, 0.0, 0.066), rgba=(0.12, 0.55, 0.16, 1.0), quat=(0.9239, 0.0, 0.3827, 0.0)),
        ),
    ),
    "box": TargetSpec("box", "box", (0.11, 0.085, 0.09), (0.18, 0.45, 0.92, 1.0), ("blue box", "cube", "block")),
    "bottle": TargetSpec("bottle", "cylinder", (0.052, 0.052, 0.22), (0.10, 0.55, 0.85, 1.0), ("blue bottle",)),
    "phone": TargetSpec("phone", "box", (0.075, 0.014, 0.145), (0.08, 0.08, 0.09, 1.0), ("mobile", "black phone")),
    "capsule": TargetSpec("capsule", "capsule", (0.045, 0.045, 0.16), (0.55, 0.85, 0.25, 1.0), ("pill", "green capsule")),
    "sphere": TargetSpec("sphere", "sphere", (0.085, 0.085, 0.085), (0.96, 0.85, 0.18, 1.0), ("ball", "yellow sphere")),
    "cylinder": TargetSpec("cylinder", "cylinder", (0.075, 0.075, 0.13), (0.55, 0.25, 0.9, 1.0), ("can", "purple cylinder")),
    "hammer": TargetSpec(
        "hammer",
        "compound",
        (0.18, 0.05, 0.12),
        (0.45, 0.26, 0.12, 1.0),
        ("tool", "mallet"),
        parts=(
            TargetPart("capsule", (0.018, 0.018, 0.18), pos=(0.0, 0.0, 0.0), rgba=(0.45, 0.26, 0.12, 1.0), quat=(0.7071, 0.0, 0.7071, 0.0)),
            TargetPart("box", (0.095, 0.040, 0.040), pos=(0.075, 0.0, 0.0), rgba=(0.15, 0.15, 0.16, 1.0)),
        ),
    ),
    "dumbbell": TargetSpec(
        "dumbbell",
        "compound",
        (0.18, 0.055, 0.055),
        (0.10, 0.70, 0.62, 1.0),
        ("barbell", "weight"),
        parts=(
            TargetPart("capsule", (0.015, 0.015, 0.16), rgba=(0.10, 0.70, 0.62, 1.0), quat=(0.7071, 0.0, 0.7071, 0.0)),
            TargetPart("sphere", (0.052, 0.052, 0.052), pos=(-0.085, 0.0, 0.0), rgba=(0.08, 0.45, 0.40, 1.0)),
            TargetPart("sphere", (0.052, 0.052, 0.052), pos=(0.085, 0.0, 0.0), rgba=(0.08, 0.45, 0.40, 1.0)),
        ),
    ),
    "tower": TargetSpec(
        "tower",
        "compound",
        (0.08, 0.08, 0.18),
        (0.92, 0.55, 0.12, 1.0),
        ("stack", "stacked blocks"),
        parts=(
            TargetPart("box", (0.090, 0.090, 0.045), pos=(0.0, 0.0, -0.045), rgba=(0.90, 0.30, 0.18, 1.0)),
            TargetPart("box", (0.070, 0.070, 0.045), pos=(0.0, 0.0, 0.000), rgba=(0.18, 0.48, 0.90, 1.0)),
            TargetPart("box", (0.052, 0.052, 0.045), pos=(0.0, 0.0, 0.045), rgba=(0.95, 0.82, 0.20, 1.0)),
        ),
    ),
}

BASE_POSITIONS: dict[str, np.ndarray] = {
    "cup": np.array([0.48, 0.02, 0.34], dtype=float),
    "apple": np.array([0.44, 0.13, 0.33], dtype=float),
    "box": np.array([0.50, -0.10, 0.34], dtype=float),
    "bottle": np.array([0.43, -0.18, 0.42], dtype=float),
    "phone": np.array([0.52, 0.08, 0.33], dtype=float),
    "capsule": np.array([0.48, 0.12, 0.35], dtype=float),
    "sphere": np.array([0.46, -0.12, 0.34], dtype=float),
    "cylinder": np.array([0.51, 0.00, 0.36], dtype=float),
    "hammer": np.array([0.50, -0.10, 0.35], dtype=float),
    "dumbbell": np.array([0.48, 0.12, 0.34], dtype=float),
    "tower": np.array([0.44, 0.02, 0.40], dtype=float),
}


def load_target_specs(path: str | Path | None) -> dict[str, TargetSpec]:
    if path is None:
        return {}
    source = Path(path)
    payload = json.loads(source.read_text())
    entries = payload.get("targets", payload if isinstance(payload, list) else None)
    if not isinstance(entries, list):
        raise ValueError("target file must contain a list or a {'targets': [...]} object")
    specs: dict[str, TargetSpec] = {}
    for entry in entries:
        spec = _target_from_mapping(entry)
        specs[spec.name] = spec
    return specs


def _target_from_mapping(entry: Any) -> TargetSpec:
    if not isinstance(entry, dict):
        raise ValueError("target entries must be objects")
    name = _required_text(entry, "name")
    shape = str(entry.get("shape", "box")).strip().lower()
    size = _float_tuple(entry.get("size", (0.10, 0.10, 0.10)), 3, "size")
    rgba = _float_tuple(entry.get("rgba", (0.85, 0.25, 0.25, 1.0)), 4, "rgba")
    aliases = tuple(str(value).strip().lower() for value in entry.get("aliases", ()) if str(value).strip())
    base = entry.get("base_position")
    base_position = None if base is None else _float_tuple(base, 3, "base_position")
    parts = tuple(_target_part_from_mapping(part) for part in entry.get("parts", ()))
    return TargetSpec(name=name, shape=shape, size=size, rgba=rgba, aliases=aliases, parts=parts, base_position=base_position)


def _target_part_from_mapping(entry: Any) -> TargetPart:
    if not isinstance(entry, dict):
        raise ValueError("target parts must be objects")
    rgba = entry.get("rgba")
    quat = entry.get("quat")
    pos_value = entry.get("pos", entry.get("offset", (0.0, 0.0, 0.0)))
    return TargetPart(
        shape=str(entry.get("shape", "box")).strip().lower(),
        size=_float_tuple(entry.get("size", (0.05, 0.05, 0.05)), 3, "part.size"),
        pos=_float_tuple(pos_value, 3, "part.pos"),
        rgba=None if rgba is None else _float_tuple(rgba, 4, "part.rgba"),
        quat=None if quat is None else _float_tuple(quat, 4, "part.quat"),
    )


def _required_text(entry: dict[str, Any], key: str) -> str:
    value = str(entry.get(key, "")).strip().lower()
    if not value:
        raise ValueError(f"target entry missing '{key}'")
    return value


def _float_tuple(value: Any, expected: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"{field} must contain {expected} numbers")
    return tuple(float(item) for item in value)


def resolve_target(name_or_prompt: str, extra_targets: dict[str, TargetSpec] | None = None) -> TargetSpec:
    text = " ".join(name_or_prompt.lower().strip().split())
    extra = extra_targets or {}
    targets = {**TARGETS, **extra}
    for collection in (extra, TARGETS):
        if text in collection:
            return collection[text]
    for collection in (extra, TARGETS):
        for spec in collection.values():
            if any(text == alias for alias in spec.aliases):
                return spec
    words = set(text.split())
    for key, spec in targets.items():
        if key in words or any(alias in text for alias in spec.aliases):
            return spec
    safe_name = "_".join(part for part in text.split() if part.isalnum()) or "object"
    return TargetSpec(safe_name[:32], "box", (0.10, 0.10, 0.10), (0.85, 0.25, 0.25, 1.0), (text,))


def base_position(target: TargetSpec) -> np.ndarray:
    if target.base_position is not None:
        return np.array(target.base_position, dtype=float)
    return BASE_POSITIONS.get(target.name, np.array([0.48, 0.02, 0.35], dtype=float)).copy()


@dataclass
class TargetMotion:
    target: TargetSpec
    mode: str
    seed: int = 7
    base_override: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._base = np.array(self.base_override, dtype=float).reshape(3) if self.base_override is not None else base_position(self.target)
        self._rng = np.random.default_rng(self.seed)
        self._random_velocity = np.array([0.035, -0.025, 0.018], dtype=float)
        self._random_pos = self._base.copy()
        self._last_time = 0.0
        self._waypoints = np.array(
            [
                self._base + np.array([0.00, 0.00, 0.00]),
                self._base + np.array([0.10, 0.05, 0.02]),
                self._base + np.array([0.04, -0.12, -0.015]),
                self._base + np.array([-0.08, -0.04, 0.025]),
            ],
            dtype=float,
        )

    def position(self, time_s: float) -> np.ndarray:
        mode = self.mode.strip().lower()
        base = self._base
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
        low = self._base + np.array([-0.13, -0.16, -0.05])
        high = self._base + np.array([0.13, 0.16, 0.06])
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
