from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .reactive import GripperCommand, PolicyCommand, PolicyObservation, PolicyPhase


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    workspace_min: tuple[float, float, float] = (-1.25, -1.25, 0.03)
    workspace_max: tuple[float, float, float] = (1.25, 1.25, 1.90)
    max_goal_distance_m: float = 0.65
    max_normal_force_n: float = 80.0


class SafetySupervisor:
    """Fail-closed validation for task-policy Cartesian commands."""

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        path_is_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    ) -> None:
        limits = SafetyLimits() if limits is None else limits
        self.limits = limits
        self._minimum = np.asarray(limits.workspace_min, dtype=float).reshape(3)
        self._maximum = np.asarray(limits.workspace_max, dtype=float).reshape(3)
        if not np.isfinite(self._minimum).all() or not np.isfinite(self._maximum).all():
            raise ValueError("workspace limits must be finite")
        if np.any(self._maximum <= self._minimum):
            raise ValueError("workspace maximum must exceed minimum")
        self._path_is_valid = path_is_valid

    def supervise(
        self, command: PolicyCommand, observation: PolicyObservation
    ) -> PolicyCommand:
        ee = np.asarray(observation.ee_position, dtype=float).reshape(3)
        goal = np.asarray(command.goal_position, dtype=float).reshape(3)
        if not np.isfinite(ee).all() or not np.isfinite(goal).all():
            safe_ee = np.nan_to_num(ee, nan=0.0, posinf=0.0, neginf=0.0)
            return PolicyCommand(
                PolicyPhase.FAILED,
                safe_ee,
                GripperCommand.OPEN,
                hold=True,
                reason="non-finite policy command",
            )
        if observation.normal_force_n > self.limits.max_normal_force_n:
            return PolicyCommand(
                PolicyPhase.RECOVER,
                ee.copy(),
                GripperCommand.OPEN,
                hold=True,
                reason="contact force exceeded safety limit",
            )
        if self._path_is_valid is not None and not self._path_is_valid(ee, goal):
            return PolicyCommand(
                PolicyPhase.RECOVER,
                ee.copy(),
                GripperCommand.OPEN,
                hold=True,
                reason="Cartesian path failed workspace/clearance validation",
            )
        goal = np.clip(goal, self._minimum, self._maximum)
        delta = goal - ee
        distance = float(np.linalg.norm(delta))
        if distance > self.limits.max_goal_distance_m:
            goal = ee + delta * (self.limits.max_goal_distance_m / distance)
        return replace(command, goal_position=goal)
