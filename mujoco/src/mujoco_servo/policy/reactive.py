from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .grasp_planner import GraspCandidate


class PolicyPhase(str, Enum):
    ACQUIRE = "ACQUIRE"
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    CLOSE = "CLOSE"
    VERIFY = "VERIFY"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    RECOVER = "RECOVER"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class GripperCommand(str, Enum):
    HOLD = "hold"
    OPEN = "open"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class ReactivePolicyConfig:
    stage_tolerance_m: float = 0.025
    lift_distance_m: float = 0.12
    transfer_clearance_m: float = 0.14
    retreat_distance_m: float = 0.10
    place_tolerance_m: float = 0.035
    verification_frames: int = 5
    close_timeout_s: float = 2.5
    motion_timeout_s: float = 8.0
    max_attempts: int = 2


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    time_s: float
    ee_position: np.ndarray
    target_position: np.ndarray | None
    tracking_valid: bool
    grasped: bool
    contact_stable_frames: int = 0
    normal_force_n: float = 0.0
    place_error_m: float | None = None


@dataclass(frozen=True, slots=True)
class PolicyCommand:
    phase: PolicyPhase
    goal_position: np.ndarray
    gripper: GripperCommand
    hold: bool = False
    reason: str = ""


class ReactivePickPlacePolicy:
    """Deterministic, contact-verified pick/place policy with bounded recovery."""

    def __init__(
        self,
        config: ReactivePolicyConfig,
        place_position: np.ndarray,
    ) -> None:
        self.config = config
        self.place_position = _vector3(place_position, "place position")
        self.phase = PolicyPhase.ACQUIRE
        self.attempts = 0
        self.failure_reason: str | None = None
        self.selected_grasp: GraspCandidate | None = None
        self._phase_enter_time_s = 0.0
        self._target_to_grasp = np.zeros(3, dtype=float)
        self._lift_goal: np.ndarray | None = None
        self._recovery_goal: np.ndarray | None = None

    def reset(self, time_s: float = 0.0) -> None:
        self.phase = PolicyPhase.ACQUIRE
        self.attempts = 0
        self.failure_reason = None
        self.selected_grasp = None
        self._phase_enter_time_s = float(time_s)
        self._target_to_grasp.fill(0.0)
        self._lift_goal = None
        self._recovery_goal = None

    def set_grasp(self, candidate: GraspCandidate, target_position: np.ndarray) -> None:
        self.selected_grasp = candidate
        target = _vector3(target_position, "target position")
        self._target_to_grasp = candidate.grasp_position - target

    def request_recovery(
        self, ee_position: np.ndarray, time_s: float, reason: str
    ) -> PolicyCommand:
        return self._begin_recovery(
            _vector3(ee_position, "end-effector position"), float(time_s), reason
        )

    def abort(self, time_s: float, reason: str) -> None:
        self.failure_reason = reason
        self._transition(PolicyPhase.FAILED, float(time_s))

    @property
    def succeeded(self) -> bool:
        return self.phase is PolicyPhase.SUCCEEDED

    @property
    def failed(self) -> bool:
        return self.phase is PolicyPhase.FAILED

    def step(self, observation: PolicyObservation) -> PolicyCommand:
        now = float(observation.time_s)
        ee = _vector3(observation.ee_position, "end-effector position")
        target = (
            None
            if observation.target_position is None
            else _vector3(observation.target_position, "target position")
        )
        if self.phase in {PolicyPhase.SUCCEEDED, PolicyPhase.FAILED}:
            return PolicyCommand(self.phase, ee.copy(), GripperCommand.OPEN, hold=True)

        if self.phase is PolicyPhase.ACQUIRE:
            if (
                not observation.tracking_valid
                or target is None
                or self.selected_grasp is None
            ):
                return PolicyCommand(
                    self.phase,
                    ee.copy(),
                    GripperCommand.OPEN,
                    hold=True,
                    reason="waiting for a valid target and grasp",
                )
            self._transition(PolicyPhase.PREGRASP, now)

        candidate = self.selected_grasp
        if candidate is None:
            return self._begin_recovery(ee, now, "selected grasp became unavailable")

        if self.phase is PolicyPhase.PREGRASP:
            if not observation.tracking_valid or target is None:
                return PolicyCommand(
                    self.phase,
                    ee.copy(),
                    GripperCommand.OPEN,
                    hold=True,
                    reason="target lost before grasp",
                )
            goal = (
                target
                + (candidate.pregrasp_position - candidate.grasp_position)
                + self._target_to_grasp
            )
            if _near(ee, goal, self.config.stage_tolerance_m):
                self._transition(PolicyPhase.APPROACH, now)
            else:
                return PolicyCommand(self.phase, goal, GripperCommand.OPEN)

        if self.phase is PolicyPhase.APPROACH:
            if not observation.tracking_valid or target is None:
                return PolicyCommand(
                    self.phase,
                    ee.copy(),
                    GripperCommand.OPEN,
                    hold=True,
                    reason="target lost during approach",
                )
            goal = target + self._target_to_grasp
            if _near(ee, goal, self.config.stage_tolerance_m):
                self._transition(PolicyPhase.CLOSE, now)
            else:
                return PolicyCommand(self.phase, goal, GripperCommand.OPEN)

        if self.phase is PolicyPhase.CLOSE:
            goal = ee.copy() if target is None else target + self._target_to_grasp
            if observation.grasped:
                self._transition(PolicyPhase.VERIFY, now)
            elif now - self._phase_enter_time_s > self.config.close_timeout_s:
                return self._begin_recovery(ee, now, "contact verification timed out")
            else:
                return PolicyCommand(self.phase, goal, GripperCommand.CLOSE)

        if self.phase is PolicyPhase.VERIFY:
            if (
                observation.grasped
                and observation.contact_stable_frames >= self.config.verification_frames
            ):
                self._lift_goal = ee + np.array([0.0, 0.0, self.config.lift_distance_m])
                self._transition(PolicyPhase.LIFT, now)
            elif now - self._phase_enter_time_s > self.config.close_timeout_s:
                return self._begin_recovery(ee, now, "grasp was not stable")
            else:
                return PolicyCommand(self.phase, ee.copy(), GripperCommand.CLOSE)

        if self.phase is PolicyPhase.LIFT:
            if not observation.grasped:
                return self._begin_recovery(ee, now, "object slipped during lift")
            assert self._lift_goal is not None
            if _near(ee, self._lift_goal, self.config.stage_tolerance_m):
                self._transition(PolicyPhase.TRANSFER, now)
            elif self._motion_timed_out(now):
                return self._begin_recovery(ee, now, "lift motion timed out")
            else:
                return PolicyCommand(
                    self.phase, self._lift_goal.copy(), GripperCommand.CLOSE
                )

        transfer_goal = self.place_position + self._target_to_grasp
        transfer_goal[2] = max(
            transfer_goal[2] + self.config.transfer_clearance_m,
            self.place_position[2] + self.config.transfer_clearance_m,
        )
        if self.phase is PolicyPhase.TRANSFER:
            if not observation.grasped:
                return self._begin_recovery(ee, now, "object slipped during transfer")
            if _near(ee, transfer_goal, self.config.stage_tolerance_m):
                self._transition(PolicyPhase.PLACE, now)
            elif self._motion_timed_out(now):
                return self._begin_recovery(ee, now, "transfer motion timed out")
            else:
                return PolicyCommand(self.phase, transfer_goal, GripperCommand.CLOSE)

        place_goal = self.place_position + self._target_to_grasp
        if self.phase is PolicyPhase.PLACE:
            if not observation.grasped:
                return self._begin_recovery(ee, now, "object slipped before placement")
            if _near(ee, place_goal, self.config.stage_tolerance_m):
                self._transition(PolicyPhase.RELEASE, now)
            elif self._motion_timed_out(now):
                return self._begin_recovery(ee, now, "place motion timed out")
            else:
                return PolicyCommand(self.phase, place_goal, GripperCommand.CLOSE)

        if self.phase is PolicyPhase.RELEASE:
            self._transition(PolicyPhase.RETREAT, now)
            return PolicyCommand(self.phase, place_goal, GripperCommand.OPEN)

        if self.phase is PolicyPhase.RETREAT:
            retreat_goal = place_goal + np.array(
                [0.0, 0.0, self.config.retreat_distance_m]
            )
            if _near(ee, retreat_goal, self.config.stage_tolerance_m):
                self._transition(PolicyPhase.SUCCEEDED, now)
                return PolicyCommand(
                    self.phase,
                    ee.copy(),
                    GripperCommand.OPEN,
                    hold=True,
                    reason="place sequence completed",
                )
            if self._motion_timed_out(now):
                self.failure_reason = "retreat motion timed out"
                self._transition(PolicyPhase.FAILED, now)
                return PolicyCommand(
                    self.phase,
                    ee.copy(),
                    GripperCommand.OPEN,
                    hold=True,
                    reason=self.failure_reason,
                )
            return PolicyCommand(self.phase, retreat_goal, GripperCommand.OPEN)

        if self.phase is PolicyPhase.RECOVER:
            assert self._recovery_goal is not None
            if _near(ee, self._recovery_goal, self.config.stage_tolerance_m):
                if self.attempts >= self.config.max_attempts:
                    self._transition(PolicyPhase.FAILED, now)
                    return PolicyCommand(
                        self.phase,
                        ee.copy(),
                        GripperCommand.OPEN,
                        hold=True,
                        reason=self.failure_reason or "attempt budget exhausted",
                    )
                self.selected_grasp = None
                self._transition(PolicyPhase.ACQUIRE, now)
                return PolicyCommand(
                    self.phase,
                    ee.copy(),
                    GripperCommand.OPEN,
                    hold=True,
                    reason="ready to replan",
                )
            return PolicyCommand(
                self.phase, self._recovery_goal.copy(), GripperCommand.OPEN
            )

        return PolicyCommand(self.phase, ee.copy(), GripperCommand.HOLD, hold=True)

    def _begin_recovery(self, ee: np.ndarray, now: float, reason: str) -> PolicyCommand:
        self.attempts += 1
        self.failure_reason = reason
        self._recovery_goal = ee + np.array([0.0, 0.0, self.config.retreat_distance_m])
        self._transition(PolicyPhase.RECOVER, now)
        return PolicyCommand(
            self.phase, self._recovery_goal.copy(), GripperCommand.OPEN, reason=reason
        )

    def _transition(self, phase: PolicyPhase, now: float) -> None:
        self.phase = phase
        self._phase_enter_time_s = float(now)

    def _motion_timed_out(self, now: float) -> bool:
        return now - self._phase_enter_time_s > self.config.motion_timeout_s


def _vector3(value: object, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain finite values")
    return vector


def _near(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    return float(np.linalg.norm(left - right)) <= float(tolerance)
