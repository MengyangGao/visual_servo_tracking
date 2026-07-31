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


def symmetric_handover_goals(
    target_position: np.ndarray,
    *,
    hand_separation_m: float = 0.24,
    height_offset_m: float = 0.0,
) -> BimanualGoals:
    """Create collision-aware left/right hand goals around a visual target."""
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

    def reset(self, data: mujoco.MjData) -> None:
        self.left.reset(data)
        self.right.reset(data)

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
        # Controllers address disjoint actuator sets, so sequential writes
        # compose into one MuJoCo control vector without overwriting each arm.
        left_state = self.left.step(data, left_goal, time_s, step_index, dt)
        right_state = self.right.step(data, right_goal, time_s, step_index, dt)
        return BimanualServoState(left_state, right_state)
