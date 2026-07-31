from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """Executable Cartesian grasp candidate with explainable score terms."""

    name: str
    grasp_position: np.ndarray
    pregrasp_position: np.ndarray
    approach: np.ndarray
    width_m: float | None
    score: float
    reachability_score: float
    visibility_score: float
    width_score: float
    clearance_score: float


@dataclass(frozen=True, slots=True)
class GraspRejection:
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class GraspPlanningContext:
    ee_position: np.ndarray
    camera_position: np.ndarray
    support_z: float
    approach_distance_m: float
    max_reach_m: float = 1.0
    max_gripper_width_m: float | None = None
    excluded_names: frozenset[str] = frozenset()
    reachable: Callable[[np.ndarray], bool] | None = None
    path_is_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None


class GraspPlanner:
    """Rank target-provided grasp points using geometry available at runtime."""

    def __init__(self) -> None:
        self.last_rejections: tuple[GraspRejection, ...] = ()

    def plan(
        self,
        grasp_points: Iterable[object],
        context: GraspPlanningContext,
    ) -> tuple[GraspCandidate, ...]:
        ee = _vector3(context.ee_position, "end-effector position")
        camera = _vector3(context.camera_position, "camera position")
        if not np.isfinite(context.support_z):
            raise ValueError("support_z must be finite")
        if (
            not np.isfinite(context.approach_distance_m)
            or context.approach_distance_m <= 0
        ):
            raise ValueError("approach_distance_m must be positive and finite")
        if not np.isfinite(context.max_reach_m) or context.max_reach_m <= 0:
            raise ValueError("max_reach_m must be positive and finite")

        candidates: list[GraspCandidate] = []
        rejections: list[GraspRejection] = []
        for index, point in enumerate(grasp_points):
            name = str(getattr(point, "name", f"candidate-{index}"))
            if name in context.excluded_names:
                rejections.append(GraspRejection(name, "failed in an earlier attempt"))
                continue
            position = _vector3(point.position, "grasp position")
            approach = _unit_vector(point.approach, "grasp approach")
            width = getattr(point, "width_m", None)
            if width is not None:
                width = float(width)
                if not np.isfinite(width) or width <= 0.0:
                    rejections.append(GraspRejection(name, "invalid gripper width"))
                    continue
                if (
                    context.max_gripper_width_m is not None
                    and width > context.max_gripper_width_m + 1e-9
                ):
                    rejections.append(
                        GraspRejection(name, "wider than gripper aperture")
                    )
                    continue

            pregrasp = position - approach * float(context.approach_distance_m)
            distance = float(np.linalg.norm(pregrasp - ee))
            if distance > context.max_reach_m:
                rejections.append(GraspRejection(name, "outside coarse reach radius"))
                continue
            if min(position[2], pregrasp[2]) < context.support_z + 0.005:
                rejections.append(
                    GraspRejection(name, "insufficient surface clearance")
                )
                continue
            if context.reachable is not None and not (
                context.reachable(pregrasp) and context.reachable(position)
            ):
                rejections.append(GraspRejection(name, "numerical IK did not converge"))
                continue
            if context.path_is_valid is not None and not (
                context.path_is_valid(ee, pregrasp)
                and context.path_is_valid(pregrasp, position)
            ):
                rejections.append(GraspRejection(name, "Cartesian path is unsafe"))
                continue

            reachability = float(
                np.clip(1.0 - distance / context.max_reach_m, 0.0, 1.0)
            )
            view = _unit_vector(position - camera, "camera-to-grasp ray")
            visibility = float(0.5 + 0.5 * abs(np.dot(approach, view)))
            if width is None or context.max_gripper_width_m is None:
                width_score = 0.75
            else:
                width_score = float(
                    np.clip(1.0 - width / context.max_gripper_width_m, 0.0, 1.0)
                )
            clearance = float(
                np.clip((pregrasp[2] - context.support_z) / 0.20, 0.0, 1.0)
            )
            score = (
                0.45 * reachability
                + 0.20 * visibility
                + 0.20 * width_score
                + 0.15 * clearance
            )
            candidates.append(
                GraspCandidate(
                    name=name,
                    grasp_position=position.copy(),
                    pregrasp_position=pregrasp,
                    approach=approach,
                    width_m=width,
                    score=float(score),
                    reachability_score=reachability,
                    visibility_score=visibility,
                    width_score=width_score,
                    clearance_score=clearance,
                )
            )
        self.last_rejections = tuple(rejections)
        return tuple(
            sorted(candidates, key=lambda candidate: (-candidate.score, candidate.name))
        )

    def select(
        self,
        grasp_points: Iterable[object],
        context: GraspPlanningContext,
    ) -> GraspCandidate:
        candidates = self.plan(grasp_points, context)
        if not candidates:
            raise RuntimeError(
                "no executable grasp candidate: "
                + (
                    "; ".join(
                        f"{item.name}: {item.reason}" for item in self.last_rejections
                    )
                    or "no candidates were provided"
                )
            )
        return candidates[0]


def _vector3(value: object, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain finite values")
    return vector


def _unit_vector(value: object, label: str) -> np.ndarray:
    vector = _vector3(value, label)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm
