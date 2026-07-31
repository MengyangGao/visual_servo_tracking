from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import ControllerConfig, RobotSpec
from .math_utils import (
    clamp_norm,
    damped_pseudo_inverse,
    normalize,
    tool_z_facing_rotation,
    vector_alignment_error,
)
from .scene import frame_position, resolve_joint_actuator, resolve_passive_actuator


@dataclass(slots=True)
class ServoState:
    step: int
    time_s: float
    ee_position: np.ndarray
    target_position: np.ndarray
    desired_position: np.ndarray
    position_error_m: float
    target_distance_m: float
    orientation_error_rad: float
    qpos_command: np.ndarray
    qvel_command: np.ndarray | None = None
    actuator_mode: str = "position"
    saturated_joints: int = 0
    adaptive_damping: float = 0.0
    servo_mode: str = "pbvs"
    image_error_px: float = 0.0


def desired_ee_position(
    task: str,
    target_position: np.ndarray,
    ee_position: np.ndarray,
    config: ControllerConfig,
    front_origin: np.ndarray | tuple[float, float, float] | None = None,
) -> np.ndarray:
    target = np.asarray(target_position, dtype=float).reshape(3)
    ee = np.asarray(ee_position, dtype=float).reshape(3)
    mode = task.strip().lower()
    if mode in {"contact", "touch", "grasp", "pick-place"}:
        return target.copy()
    if mode == "standoff":
        direction = normalize(ee - target, np.array([-1.0, 0.0, 0.0]))
        return target + direction * float(config.standoff_m)
    if mode == "front-standoff":
        origin = (
            np.zeros(3, dtype=float)
            if front_origin is None
            else np.asarray(front_origin, dtype=float).reshape(3)
        )
        horizontal = np.array(
            [target[0] - origin[0], target[1] - origin[1], 0.0], dtype=float
        )
        direction = normalize(horizontal, np.array([1.0, 0.0, 0.0]))
        desired = target - direction * float(config.standoff_m)
        desired[2] = target[2]
        return desired
    if mode == "align-x":
        return np.array([target[0] + config.align_offset_m, ee[1], ee[2]], dtype=float)
    if mode == "align-y":
        return np.array([ee[0], target[1] + config.align_offset_m, ee[2]], dtype=float)
    if mode == "align-z":
        return np.array([ee[0], ee[1], target[2] + config.align_offset_m], dtype=float)
    raise ValueError(f"unknown servo task '{task}'")


def desired_ee_orientation(
    task: str, target_position: np.ndarray, desired_position: np.ndarray
) -> np.ndarray | None:
    if task.strip().lower() != "front-standoff":
        return None
    forward = np.asarray(target_position, dtype=float).reshape(3) - np.asarray(
        desired_position, dtype=float
    ).reshape(3)
    forward[2] = 0.0
    return tool_z_facing_rotation(forward)


class ResolvedRateController:
    def __init__(
        self,
        model: mujoco.MjModel,
        ee_frame_name: str,
        ee_frame_type: str,
        ee_frame_offset: tuple[float, float, float] | np.ndarray,
        robot: RobotSpec,
        config: ControllerConfig,
    ) -> None:
        self.model = model
        self.robot = robot
        self.ee_frame_name = ee_frame_name
        self.ee_frame_type = ee_frame_type
        self.ee_frame_offset = np.asarray(ee_frame_offset, dtype=float).reshape(3)
        self.config = config
        self._filtered_target: np.ndarray | None = None
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in robot.joint_names
        ]
        if any(joint_id < 0 for joint_id in self._joint_ids):
            missing = [
                name
                for name, joint_id in zip(robot.joint_names, self._joint_ids)
                if joint_id < 0
            ]
            raise RuntimeError(
                f"robot '{robot.name}' missing joints: {', '.join(missing)}"
            )
        self._actuator_ids = [
            self._actuator_id_for_joint(name, i)
            for i, name in enumerate(robot.joint_names)
        ]
        self._actuator_gears = np.array(
            [model.actuator_gear[actuator_id, 0] for actuator_id in self._actuator_ids],
            dtype=float,
        )
        self._passive_actuator_ids = tuple(
            (resolve_passive_actuator(model, robot, name, value), float(value))
            for name, value in robot.passive_actuator_ctrl
        )
        self._passive_actuator_overrides: dict[int, float] = {}
        self._qpos_adr = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in self._joint_ids], dtype=int
        )
        self._dof_adr = np.array(
            [model.jnt_dofadr[joint_id] for joint_id in self._joint_ids], dtype=int
        )
        self._qpos_home = np.array(robot.home_qpos, dtype=float)
        if self._qpos_home.shape != (len(self._joint_ids),):
            raise RuntimeError(
                f"robot '{robot.name}' home_qpos must have {len(self._joint_ids)} values"
            )
        self._qpos_command = self._qpos_home.copy()
        self._last_qvel_command = np.zeros(len(self._joint_ids), dtype=float)
        self._hold_qpos: np.ndarray | None = None
        self._last_saturated_joints = 0
        self._frame_id = self._resolve_frame_id(model, ee_frame_type, ee_frame_name)

    @staticmethod
    def _resolve_frame_id(
        model: mujoco.MjModel, frame_type: str, frame_name: str
    ) -> int:
        if frame_type == "site":
            frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
        elif frame_type in {"body", "body_point"}:
            frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        else:
            raise ValueError(f"unknown end-effector frame type '{frame_type}'")
        if frame_id < 0:
            raise RuntimeError(f"{frame_type} '{frame_name}' not found")
        return frame_id

    def reset(self, data: mujoco.MjData) -> None:
        self._qpos_command = np.array(data.qpos[self._qpos_adr], dtype=float)
        self._last_qvel_command = np.zeros(len(self._joint_ids), dtype=float)
        self._filtered_target = None
        self._hold_qpos = None
        self._last_saturated_joints = 0
        self._passive_actuator_overrides.clear()

    def set_gripper_closed(self, closed: bool) -> None:
        """Latch gripper controls so subsequent arm updates cannot reopen it."""
        values = (
            self.robot.gripper_closed_ctrl if closed else self.robot.gripper_open_ctrl
        )
        self._passive_actuator_overrides.clear()
        for name, value in zip(self.robot.gripper_actuator_names, values):
            actuator_id = resolve_passive_actuator(self.model, self.robot, name, value)
            self._passive_actuator_overrides[actuator_id] = float(value)

    @property
    def actuator_mode(self) -> str:
        return str(getattr(self.config, "actuator_mode", "position")).strip().lower()

    @property
    def last_saturated_joints(self) -> int:
        return self._last_saturated_joints

    def reset_target_filter(self, target_position: np.ndarray | None = None) -> None:
        """Reset target smoothing after acquisition or a discontinuous re-acquisition."""
        self._filtered_target = (
            None
            if target_position is None
            else np.asarray(target_position, dtype=float).reshape(3).copy()
        )

    def begin_hold(self, data: mujoco.MjData) -> None:
        """Latch the loss-time joint pose exactly once."""
        self._hold_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float).copy()
        self._qpos_command = self._hold_qpos.copy()
        self._last_qvel_command.fill(0.0)

    def end_hold(self, target_position: np.ndarray | None = None) -> None:
        self._hold_qpos = None
        self.reset_target_filter(target_position)

    def hold(
        self,
        data: mujoco.MjData,
        time_s: float,
        step_index: int,
        dt: float | None = None,
    ) -> ServoState:
        ee_pos = frame_position(
            self.model,
            data,
            self.ee_frame_type,
            self.ee_frame_name,
            self.ee_frame_offset,
        )
        if self._hold_qpos is None:
            self.begin_hold(data)
        assert self._hold_qpos is not None
        dt_s = self._valid_dt(dt)
        current_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float)
        # A velocity servo still needs a small position loop to hold a fixed
        # loss-time pose.  Position and torque modes use the same latched pose
        # directly, so no moving-reference drift can accumulate.
        hold_qvel = np.clip(
            6.0 * (self._hold_qpos - current_qpos),
            -float(self.config.max_joint_speed),
            float(self.config.max_joint_speed),
        )
        if self.actuator_mode != "velocity":
            hold_qvel.fill(0.0)
        saturated = self._apply_actuation(
            data, hold_qvel, dt_s, position_reference=self._hold_qpos
        )
        return ServoState(
            step=step_index,
            time_s=time_s,
            ee_position=ee_pos,
            target_position=ee_pos.copy(),
            desired_position=ee_pos.copy(),
            position_error_m=0.0,
            target_distance_m=0.0,
            orientation_error_rad=0.0,
            qpos_command=self._hold_qpos.copy(),
            qvel_command=hold_qvel.copy(),
            actuator_mode=self.actuator_mode,
            saturated_joints=saturated,
        )

    def step(
        self,
        data: mujoco.MjData,
        target_position: np.ndarray,
        time_s: float,
        step_index: int,
        dt: float | None = None,
        *,
        cartesian_velocity_world: np.ndarray | None = None,
        desired_rotation_world: np.ndarray | None = None,
    ) -> ServoState:
        target = np.asarray(target_position, dtype=float).reshape(3)
        if self._filtered_target is None:
            self._filtered_target = target.copy()
        else:
            alpha = float(self.config.smooth_target_alpha)
            self._filtered_target = (
                1.0 - alpha
            ) * self._filtered_target + alpha * target

        ee_pos = frame_position(
            self.model,
            data,
            self.ee_frame_type,
            self.ee_frame_name,
            self.ee_frame_offset,
        )
        desired = desired_ee_position(
            self.config.task,
            self._filtered_target,
            ee_pos,
            self.config,
            self.robot.base_position,
        )
        error = desired - ee_pos
        ee_velocity = (
            clamp_norm(
                float(self.config.position_gain) * error, self.config.max_ee_speed
            )
            if cartesian_velocity_world is None
            else clamp_norm(
                np.asarray(cartesian_velocity_world, dtype=float).reshape(3),
                self.config.max_ee_speed,
            )
        )
        desired_rotation = (
            desired_ee_orientation(self.config.task, self._filtered_target, desired)
            if desired_rotation_world is None
            else np.asarray(desired_rotation_world, dtype=float).reshape(3, 3)
        )

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        if self.ee_frame_type == "site":
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self._frame_id)
        elif self.ee_frame_type == "body_point":
            mujoco.mj_jac(self.model, data, jacp, jacr, ee_pos, self._frame_id)
        else:
            mujoco.mj_jacBody(self.model, data, jacp, jacr, self._frame_id)
        orientation_error = np.zeros(3, dtype=float)
        position_jac = jacp[:, self._dof_adr]
        current_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float)
        joint_weights = self._joint_limit_weights(current_qpos)
        primary_inverse, adaptive_damping = self._weighted_adaptive_inverse(
            position_jac, joint_weights
        )
        qvel = primary_inverse @ ee_velocity
        # Use an undamped Moore-Penrose projector for task hierarchy.  A
        # projector built from the damped inverse is not a true null-space
        # projector; secondary orientation/posture commands then leak into the
        # position task and make an otherwise converged end effector drift.
        primary_nullspace = (
            np.eye(len(self._joint_ids))
            - np.linalg.pinv(position_jac, rcond=1e-5) @ position_jac
        )
        posture_nullspace = primary_nullspace
        if desired_rotation is not None:
            current_rotation = self.frame_rotation(data)
            # Front tracking constrains the tool approach axis, not roll about
            # that axis.  Treating it as a full 3-DoF orientation task wastes a
            # degree of freedom and can pull the Cartesian position away from
            # an already converged standoff pose.
            current_tool_axis = current_rotation @ np.asarray(
                self.robot.tool_axis, dtype=float
            )
            orientation_error = vector_alignment_error(
                current_tool_axis, desired_rotation[:, 2]
            )
            angular_velocity = clamp_norm(
                self.config.orientation_gain * orientation_error,
                self.config.max_angular_speed,
            )
            # Rotation around the tool axis does not change the facing
            # direction, so remove that unobservable row-space component.
            axis_projector = np.eye(3) - np.outer(current_tool_axis, current_tool_axis)
            orientation_jac = axis_projector @ jacr[:, self._dof_adr]
            correction_jac = orientation_jac @ primary_nullspace
            correction = damped_pseudo_inverse(correction_jac, adaptive_damping) @ (
                angular_velocity - orientation_jac @ qvel
            )
            position_gate = (
                1.0
                if desired_rotation_world is not None
                else float(np.clip(1.0 - np.linalg.norm(error) / 0.025, 0.0, 1.0))
            )
            qvel = qvel + (0.30 * position_gate) * (primary_nullspace @ correction)
            stacked_jac = np.vstack([position_jac, orientation_jac])
            posture_nullspace = (
                np.eye(len(self._joint_ids))
                - np.linalg.pinv(stacked_jac, rcond=1e-5) @ stacked_jac
            )

        home_error = self._qpos_home - current_qpos
        home_gain = 0.03 if desired_rotation is not None else 0.18
        qvel = qvel + home_gain * (posture_nullspace @ home_error)
        qvel = qvel + 0.20 * (
            posture_nullspace @ self._joint_limit_avoidance(current_qpos)
        )
        if not np.isfinite(qvel).all():
            qvel = np.zeros_like(qvel)

        dt_s = self._valid_dt(dt)
        raw_qvel = qvel.copy()
        max_speed = float(self.config.max_joint_speed)
        qvel = np.clip(qvel, -max_speed, max_speed)
        max_accel = float(getattr(self.config, "max_joint_accel", 8.0))
        max_delta = max_accel * dt_s
        qvel = np.clip(
            qvel,
            self._last_qvel_command - max_delta,
            self._last_qvel_command + max_delta,
        )
        constraint_saturated = np.abs(qvel - raw_qvel) > 1e-10
        self._qpos_command = self._qpos_command + qvel * dt_s
        command_lead = 0.08
        unclipped_qpos_command = self._qpos_command.copy()
        self._qpos_command = np.clip(
            self._qpos_command,
            current_qpos - command_lead,
            current_qpos + command_lead,
        )
        constraint_saturated |= (
            np.abs(self._qpos_command - unclipped_qpos_command) > 1e-10
        )
        margin = float(getattr(self.config, "joint_limit_margin", 0.05))
        for i, joint_id in enumerate(self._joint_ids):
            if self.model.jnt_limited[joint_id]:
                lo, hi = self.model.jnt_range[joint_id]
                safe_margin = min(margin, max(1e-4, 0.45 * float(hi - lo)))
                before = self._qpos_command[i]
                self._qpos_command[i] = np.clip(
                    before, lo + safe_margin, hi - safe_margin
                )
                constraint_saturated[i] |= not np.isclose(before, self._qpos_command[i])
        actuation_saturated = self._apply_actuation(
            data, qvel, dt_s, position_reference=self._qpos_command
        )
        saturated = max(
            int(np.count_nonzero(constraint_saturated)), actuation_saturated
        )
        self._last_saturated_joints = saturated
        self._last_qvel_command = qvel.copy()
        self._hold_qpos = None

        return ServoState(
            step=step_index,
            time_s=time_s,
            ee_position=ee_pos,
            target_position=self._filtered_target.copy(),
            desired_position=desired,
            position_error_m=float(np.linalg.norm(error)),
            target_distance_m=float(np.linalg.norm(self._filtered_target - ee_pos)),
            orientation_error_rad=float(np.linalg.norm(orientation_error)),
            qpos_command=self._qpos_command.copy(),
            qvel_command=qvel.copy(),
            actuator_mode=self.actuator_mode,
            saturated_joints=saturated,
            adaptive_damping=adaptive_damping,
        )

    def _valid_dt(self, dt: float | None) -> float:
        dt_s = float(dt) if dt is not None else 1.0 / float(self.config.control_hz)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("controller dt must be positive and finite")
        return dt_s

    def _weighted_adaptive_inverse(
        self,
        jacobian: np.ndarray,
        joint_weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        jac = np.asarray(jacobian, dtype=float)
        weights = np.asarray(joint_weights, dtype=float).reshape(jac.shape[1])
        inv_weights = 1.0 / np.maximum(weights, 1.0)
        weighted_jac = jac * np.sqrt(inv_weights)[None, :]
        singular_values = np.linalg.svd(weighted_jac, compute_uv=False)
        sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
        threshold = 0.08
        base = float(self.config.damping)
        proximity = float(np.clip((threshold - sigma_min) / threshold, 0.0, 1.0))
        damping = base * (1.0 + 4.0 * proximity * proximity)
        inverse = (inv_weights[:, None] * jac.T) @ np.linalg.inv(
            jac @ (inv_weights[:, None] * jac.T)
            + damping * damping * np.eye(jac.shape[0])
        )
        return inverse, damping

    def _joint_limit_weights(self, qpos: np.ndarray) -> np.ndarray:
        margin = float(getattr(self.config, "joint_limit_margin", 0.05))
        weights = np.ones(len(self._joint_ids), dtype=float)
        if margin <= 0.0:
            return weights
        for i, joint_id in enumerate(self._joint_ids):
            if not self.model.jnt_limited[joint_id]:
                continue
            lo, hi = self.model.jnt_range[joint_id]
            distance = max(1e-6, min(float(qpos[i] - lo), float(hi - qpos[i])))
            if distance < margin:
                weights[i] += 24.0 * (1.0 - distance / margin) ** 2
        return weights

    def _joint_limit_avoidance(self, qpos: np.ndarray) -> np.ndarray:
        margin = float(getattr(self.config, "joint_limit_margin", 0.05))
        avoidance = np.zeros(len(self._joint_ids), dtype=float)
        if margin <= 0.0:
            return avoidance
        for i, joint_id in enumerate(self._joint_ids):
            if not self.model.jnt_limited[joint_id]:
                continue
            lo, hi = self.model.jnt_range[joint_id]
            lower_distance = float(qpos[i] - lo)
            upper_distance = float(hi - qpos[i])
            if lower_distance < margin:
                avoidance[i] += (margin - max(lower_distance, 0.0)) / margin
            if upper_distance < margin:
                avoidance[i] -= (margin - max(upper_distance, 0.0)) / margin
        return avoidance

    def _apply_actuation(
        self,
        data: mujoco.MjData,
        qvel_command: np.ndarray,
        dt_s: float,
        *,
        position_reference: np.ndarray,
    ) -> int:
        del dt_s  # Kept explicit in this boundary for future discrete actuator models.
        mode = self.actuator_mode
        if mode not in {"position", "velocity", "torque", "impedance"}:
            raise ValueError(f"unknown actuator mode '{mode}'")
        current_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float)
        current_qvel = np.asarray(data.qvel[self._dof_adr], dtype=float)
        if mode == "position":
            controls = self._actuator_gears * np.asarray(
                position_reference, dtype=float
            )
        elif mode == "velocity":
            # A MuJoCo velocity actuator produces no force at zero tracking
            # error.  Add the exact inverse-transmission bias feed-forward so
            # gravity does not create a permanent Cartesian offset, while the
            # commanded motion remains a direct joint-velocity reference.
            gains = np.array(
                [
                    self.model.actuator_gainprm[actuator_id, 0]
                    for actuator_id in self._actuator_ids
                ],
                dtype=float,
            )
            bias = np.asarray(data.qfrc_bias[self._dof_adr], dtype=float)
            controls = self._actuator_gears * np.asarray(
                qvel_command, dtype=float
            ) + bias / (self._actuator_gears * gains)
        else:
            if mode == "impedance":
                kp_scale, kd_scale = self.robot.impedance_gain_scale
                kp = float(getattr(self.config, "impedance_kp", 35.0)) * kp_scale
                kd = float(getattr(self.config, "impedance_kd", 6.0)) * kd_scale
            else:
                kp_scale, kd_scale = self.robot.torque_gain_scale
                kp = float(getattr(self.config, "torque_kp", 80.0)) * kp_scale
                kd = float(getattr(self.config, "torque_kd", 8.0)) * kd_scale
            generalized_torque = (
                np.asarray(data.qfrc_bias[self._dof_adr], dtype=float)
                + kp * (np.asarray(position_reference, dtype=float) - current_qpos)
                + kd * (np.asarray(qvel_command, dtype=float) - current_qvel)
            )
            controls = generalized_torque / self._actuator_gears
        saturated = 0
        for actuator_id, value in zip(self._actuator_ids, controls):
            saturated += int(self._write_ctrl(data, actuator_id, float(value)))
        for actuator_id, value in self._passive_actuator_ids:
            self._write_ctrl(
                data,
                actuator_id,
                self._passive_actuator_overrides.get(actuator_id, value),
            )
        self._last_saturated_joints = saturated
        return saturated

    def frame_rotation(self, data: mujoco.MjData) -> np.ndarray:
        if self.ee_frame_type == "site":
            return np.array(data.site_xmat[self._frame_id], dtype=float).reshape(3, 3)
        return np.array(data.xmat[self._frame_id], dtype=float).reshape(3, 3)

    def _actuator_id_for_joint(self, joint_name: str, index: int) -> int:
        return resolve_joint_actuator(self.model, self.robot, joint_name, index)

    def _write_ctrl(self, data: mujoco.MjData, actuator_id: int, value: float) -> bool:
        ctrl = float(value)
        saturated = False
        if self.model.actuator_ctrllimited[actuator_id]:
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            clipped = float(np.clip(ctrl, lo, hi))
            saturated = not np.isclose(clipped, ctrl)
            ctrl = clipped
        data.ctrl[actuator_id] = ctrl
        return saturated
