from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class LabeledMeasurement:
    label: str
    position_world: np.ndarray
    confidence: float


@dataclass(slots=True)
class TargetTrack:
    track_id: int
    label: str
    position_world: np.ndarray
    velocity_world: np.ndarray
    confidence: float
    age_s: float
    missed_s: float


class MultiTargetTracker:
    """Label-aware nearest-neighbour tracker with bounded occlusion prediction."""

    def __init__(
        self, *, max_match_distance_m: float = 0.20, occlusion_timeout_s: float = 0.75
    ) -> None:
        self.max_match_distance_m = float(max_match_distance_m)
        self.occlusion_timeout_s = float(occlusion_timeout_s)
        self._tracks: list[TargetTrack] = []
        self._next_id = 1

    @property
    def tracks(self) -> tuple[TargetTrack, ...]:
        return tuple(self._tracks)

    def update(
        self, measurements: list[LabeledMeasurement], dt_s: float
    ) -> tuple[TargetTrack, ...]:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("tracker dt must be positive and finite")
        unmatched = set(range(len(measurements)))
        kept: list[TargetTrack] = []
        for track in self._tracks:
            predicted = track.position_world + track.velocity_world * dt
            candidates = [
                (
                    index,
                    float(
                        np.linalg.norm(
                            np.asarray(measurements[index].position_world) - predicted
                        )
                    ),
                )
                for index in unmatched
                if measurements[index].label == track.label
            ]
            if candidates:
                index, distance = min(candidates, key=lambda item: item[1])
            else:
                index, distance = -1, float("inf")
            if distance <= self.max_match_distance_m:
                measurement = measurements[index]
                position = np.asarray(measurement.position_world, dtype=float).reshape(
                    3
                )
                velocity = 0.65 * track.velocity_world + 0.35 * (
                    (position - track.position_world) / dt
                )
                track.position_world = position
                track.velocity_world = velocity
                track.confidence = float(measurement.confidence)
                track.age_s += dt
                track.missed_s = 0.0
                unmatched.remove(index)
                kept.append(track)
            else:
                track.position_world = predicted
                track.age_s += dt
                track.missed_s += dt
                track.confidence *= 0.92
                if track.missed_s <= self.occlusion_timeout_s:
                    kept.append(track)
        for index in sorted(unmatched):
            measurement = measurements[index]
            kept.append(
                TargetTrack(
                    track_id=self._next_id,
                    label=measurement.label,
                    position_world=np.asarray(measurement.position_world, dtype=float)
                    .reshape(3)
                    .copy(),
                    velocity_world=np.zeros(3, dtype=float),
                    confidence=float(measurement.confidence),
                    age_s=0.0,
                    missed_s=0.0,
                )
            )
            self._next_id += 1
        self._tracks = kept
        return self.tracks
