from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .config import ControllerConfig, ROBOT_SPECS
from .control import ResolvedRateController, ServoState


@dataclass(frozen=True, slots=True)
class BimanualGoals:
    """Cartesian hand goals derived from one visually estimated target pose."""

    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True, slots=True)
class BimanualServoState:
    left: ServoState
    right: ServoState
    safety_limited: bool = False
    safety_reason: str = ""


@dataclass(frozen=True, slots=True)
class BimanualSafetyConfig:
    minimum_hand_separation_m: float = 0.12
    maximum_goal_speed_mps: float = 1.5
    path_samples: int = 16


def symmetric_handover_goals(
    target_position: np.ndarray,
    *,
    hand_separation_m: float = 0.24,
    height_offset_m: float = 0.0,
) -> BimanualGoals:
    """Create symmetric left/right hand goals around a visual target."""
    target = np.asarray(target_position, dtype=float).reshape(3)
    if not np.isfinite(target).all():
        raise ValueError("target_position must contain finite values")
    if not np.isfinite(hand_separation_m) or hand_separation_m <= 0.0:
        raise ValueError("hand_separation_m must be positive and finite")
    if not np.isfinite(height_offset_m):
        raise ValueError("height_offset_m must be finite")
    half = 0.5 * float(hand_separation_m)
    offset_z = float(height_offset_m)
    return BimanualGoals(
        left=target + np.array([0.0, half, offset_z]),
        right=target + np.array([0.0, -half, offset_z]),
    )


class G1BimanualController:
    """Coordinate both fixed-base Unitree G1 arms over disjoint actuators.

    The class intentionally controls only upper limbs. Whole-body balance and
    locomotion are outside this fixed-base visual-manipulation primitive.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        config: ControllerConfig = ControllerConfig(task="contact"),
        safety: BimanualSafetyConfig = BimanualSafetyConfig(),
    ) -> None:
        arm_config = replace(config, task="contact")
        left = ROBOT_SPECS["g1-left-arm"]
        right = ROBOT_SPECS["g1-right-arm"]
        self.left = ResolvedRateController(
            model,
            left.ee_frame_name,
            left.ee_frame_type,
            left.ee_frame_offset,
            left,
            arm_config,
        )
        self.right = ResolvedRateController(
            model,
            right.ee_frame_name,
            right.ee_frame_type,
            right.ee_frame_offset,
            right,
            arm_config,
        )
        self.safety = safety
        if safety.minimum_hand_separation_m <= 0.0:
            raise ValueError("minimum_hand_separation_m must be positive")
        if safety.maximum_goal_speed_mps <= 0.0:
            raise ValueError("maximum_goal_speed_mps must be positive")
        if safety.path_samples < 2:
            raise ValueError("path_samples must be at least two")
        self._previous_goals: BimanualGoals | None = None

    def reset(self, data: mujoco.MjData) -> None:
        self.left.reset(data)
        self.right.reset(data)
        self._previous_goals = BimanualGoals(
            self.left.frame_position(data), self.right.frame_position(data)
        )

    def step(
        self,
        data: mujoco.MjData,
        goals: BimanualGoals,
        time_s: float,
        step_index: int,
        dt: float | None = None,
    ) -> BimanualServoState:
        left_goal = np.asarray(goals.left, dtype=float).reshape(3)
        right_goal = np.asarray(goals.right, dtype=float).reshape(3)
        if not np.isfinite(left_goal).all() or not np.isfinite(right_goal).all():
            raise ValueError("bimanual goals must contain finite values")
        previous = self._previous_goals
        left_current = self.left.frame_position(data)
        right_current = self.right.frame_position(data)
        if previous is None:
            previous = BimanualGoals(left_current, right_current)
        dt_s = float(dt) if dt is not None else 1.0 / self.left.config.control_hz
        maximum_step = self.safety.maximum_goal_speed_mps * max(dt_s, 1e-6)
        left_goal, left_limited = _limit_goal_step(
            previous.left, left_goal, maximum_step
        )
        right_goal, right_limited = _limit_goal_step(
            previous.right, right_goal, maximum_step
        )
        if not _separated_paths(
            left_current,
            left_goal,
            right_current,
            right_goal,
            self.safety.minimum_hand_separation_m,
            self.safety.path_samples,
        ):
            left_state = self.left.hold(data, time_s, step_index, dt)
            right_state = self.right.hold(data, time_s, step_index, dt)
            return BimanualServoState(
                left_state,
                right_state,
                safety_limited=True,
                safety_reason="inter-hand path violates minimum separation",
            )
        # Controllers address disjoint actuator sets, so sequential writes
        # compose into one MuJoCo control vector without overwriting each arm.
        left_state = self.left.step(data, left_goal, time_s, step_index, dt)
        right_state = self.right.step(data, right_goal, time_s, step_index, dt)
        self._previous_goals = BimanualGoals(left_goal.copy(), right_goal.copy())
        return BimanualServoState(
            left_state,
            right_state,
            safety_limited=left_limited or right_limited,
            safety_reason=(
                "goal velocity limited" if left_limited or right_limited else ""
            ),
        )


def _limit_goal_step(
    previous: np.ndarray, requested: np.ndarray, maximum_step: float
) -> tuple[np.ndarray, bool]:
    delta = np.asarray(requested, dtype=float) - np.asarray(previous, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= maximum_step:
        return np.asarray(requested, dtype=float).copy(), False
    return np.asarray(previous, dtype=float) + delta * (maximum_step / distance), True


def _separated_paths(
    left_start: np.ndarray,
    left_end: np.ndarray,
    right_start: np.ndarray,
    right_end: np.ndarray,
    minimum_separation: float,
    samples: int,
) -> bool:
    phase = np.linspace(0.0, 1.0, int(samples))[:, None]
    left = left_start + phase * (left_end - left_start)
    right = right_start + phase * (right_end - right_start)
    return bool(np.all(np.linalg.norm(left - right, axis=1) >= minimum_separation))
