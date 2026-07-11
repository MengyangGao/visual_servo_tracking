from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import ControllerConfig, RobotSpec
from .math_utils import clamp_norm, damped_pseudo_inverse, normalize, tool_z_facing_rotation, vector_alignment_error
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
    if mode == "contact":
        return target.copy()
    if mode == "standoff":
        direction = normalize(ee - target, np.array([-1.0, 0.0, 0.0]))
        return target + direction * float(config.standoff_m)
    if mode == "front-standoff":
        origin = np.zeros(3, dtype=float) if front_origin is None else np.asarray(front_origin, dtype=float).reshape(3)
        horizontal = np.array([target[0] - origin[0], target[1] - origin[1], 0.0], dtype=float)
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


def desired_ee_orientation(task: str, target_position: np.ndarray, desired_position: np.ndarray) -> np.ndarray | None:
    if task.strip().lower() != "front-standoff":
        return None
    forward = np.asarray(target_position, dtype=float).reshape(3) - np.asarray(desired_position, dtype=float).reshape(3)
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
        self._joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot.joint_names]
        if any(joint_id < 0 for joint_id in self._joint_ids):
            missing = [name for name, joint_id in zip(robot.joint_names, self._joint_ids) if joint_id < 0]
            raise RuntimeError(f"robot '{robot.name}' missing joints: {', '.join(missing)}")
        self._actuator_ids = [self._actuator_id_for_joint(name, i) for i, name in enumerate(robot.joint_names)]
        self._actuator_gears = np.array([model.actuator_gear[actuator_id, 0] for actuator_id in self._actuator_ids], dtype=float)
        self._passive_actuator_ids = tuple(
            (resolve_passive_actuator(model, robot, name, value), float(value))
            for name, value in robot.passive_actuator_ctrl
        )
        self._qpos_adr = np.array([model.jnt_qposadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._dof_adr = np.array([model.jnt_dofadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._qpos_home = np.array(robot.home_qpos, dtype=float)
        if self._qpos_home.shape != (len(self._joint_ids),):
            raise RuntimeError(f"robot '{robot.name}' home_qpos must have {len(self._joint_ids)} values")
        self._qpos_command = self._qpos_home.copy()
        self._frame_id = self._resolve_frame_id(model, ee_frame_type, ee_frame_name)

    @staticmethod
    def _resolve_frame_id(model: mujoco.MjModel, frame_type: str, frame_name: str) -> int:
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
        self._filtered_target = None

    def hold(self, data: mujoco.MjData, time_s: float, step_index: int) -> ServoState:
        ee_pos = frame_position(self.model, data, self.ee_frame_type, self.ee_frame_name, self.ee_frame_offset)
        current_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float)
        self._qpos_command = current_qpos.copy()
        for i, actuator_id in enumerate(self._actuator_ids):
            self._write_ctrl(data, actuator_id, self._actuator_gears[i] * self._qpos_command[i])
        for actuator_id, value in self._passive_actuator_ids:
            self._write_ctrl(data, actuator_id, value)
        return ServoState(
            step=step_index,
            time_s=time_s,
            ee_position=ee_pos,
            target_position=ee_pos.copy(),
            desired_position=ee_pos.copy(),
            position_error_m=0.0,
            target_distance_m=0.0,
            orientation_error_rad=0.0,
            qpos_command=self._qpos_command.copy(),
        )

    def step(self, data: mujoco.MjData, target_position: np.ndarray, time_s: float, step_index: int, dt: float | None = None) -> ServoState:
        target = np.asarray(target_position, dtype=float).reshape(3)
        if self._filtered_target is None:
            self._filtered_target = target.copy()
        else:
            alpha = float(self.config.smooth_target_alpha)
            self._filtered_target = (1.0 - alpha) * self._filtered_target + alpha * target

        ee_pos = frame_position(self.model, data, self.ee_frame_type, self.ee_frame_name, self.ee_frame_offset)
        desired = desired_ee_position(
            self.config.task,
            self._filtered_target,
            ee_pos,
            self.config,
            self.robot.base_position,
        )
        error = desired - ee_pos
        ee_velocity = clamp_norm(float(self.config.position_gain) * error, self.config.max_ee_speed)
        desired_rotation = desired_ee_orientation(self.config.task, self._filtered_target, desired)

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
        primary_inverse = damped_pseudo_inverse(position_jac, self.config.damping)
        qvel = primary_inverse @ ee_velocity
        # Use an undamped Moore-Penrose projector for task hierarchy.  A
        # projector built from the damped inverse is not a true null-space
        # projector; secondary orientation/posture commands then leak into the
        # position task and make an otherwise converged end effector drift.
        primary_nullspace = np.eye(len(self._joint_ids)) - np.linalg.pinv(position_jac, rcond=1e-5) @ position_jac
        posture_nullspace = primary_nullspace
        if desired_rotation is not None:
            current_rotation = self.frame_rotation(data)
            # Front tracking constrains the tool approach axis, not roll about
            # that axis.  Treating it as a full 3-DoF orientation task wastes a
            # degree of freedom and can pull the Cartesian position away from
            # an already converged standoff pose.
            current_tool_axis = current_rotation @ np.asarray(self.robot.tool_axis, dtype=float)
            orientation_error = vector_alignment_error(current_tool_axis, desired_rotation[:, 2])
            angular_velocity = clamp_norm(self.config.orientation_gain * orientation_error, self.config.max_angular_speed)
            # Rotation around the tool axis does not change the facing
            # direction, so remove that unobservable row-space component.
            axis_projector = np.eye(3) - np.outer(current_tool_axis, current_tool_axis)
            orientation_jac = axis_projector @ jacr[:, self._dof_adr]
            correction_jac = orientation_jac @ primary_nullspace
            correction = damped_pseudo_inverse(correction_jac, self.config.damping) @ (angular_velocity - orientation_jac @ qvel)
            position_gate = float(np.clip(1.0 - np.linalg.norm(error) / 0.025, 0.0, 1.0))
            qvel = qvel + (0.30 * position_gate) * (primary_nullspace @ correction)
            stacked_jac = np.vstack([position_jac, orientation_jac])
            posture_nullspace = np.eye(len(self._joint_ids)) - np.linalg.pinv(stacked_jac, rcond=1e-5) @ stacked_jac

        home_error = self._qpos_home - np.asarray(data.qpos[self._qpos_adr], dtype=float)
        home_gain = 0.03 if desired_rotation is not None else 0.18
        qvel = qvel + home_gain * (posture_nullspace @ home_error)
        qvel = clamp_norm(qvel, self.config.max_joint_speed)
        if not np.isfinite(qvel).all():
            qvel = np.zeros_like(qvel)

        dt_s = float(dt) if dt is not None else 1.0 / float(self.config.control_hz)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("controller dt must be positive and finite")
        current_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float)
        self._qpos_command = self._qpos_command + qvel * dt_s
        self._qpos_command = np.clip(self._qpos_command, current_qpos - 0.08, current_qpos + 0.08)
        for i, joint_id in enumerate(self._joint_ids):
            if self.model.jnt_limited[joint_id]:
                lo, hi = self.model.jnt_range[joint_id]
                self._qpos_command[i] = np.clip(self._qpos_command[i], lo + 1e-4, hi - 1e-4)
        for i, actuator_id in enumerate(self._actuator_ids):
            self._write_ctrl(data, actuator_id, self._actuator_gears[i] * self._qpos_command[i])
        for actuator_id, value in self._passive_actuator_ids:
            self._write_ctrl(data, actuator_id, value)

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
        )

    def frame_rotation(self, data: mujoco.MjData) -> np.ndarray:
        if self.ee_frame_type == "site":
            return np.array(data.site_xmat[self._frame_id], dtype=float).reshape(3, 3)
        return np.array(data.xmat[self._frame_id], dtype=float).reshape(3, 3)

    def _actuator_id_for_joint(self, joint_name: str, index: int) -> int:
        return resolve_joint_actuator(self.model, self.robot, joint_name, index)

    def _write_ctrl(self, data: mujoco.MjData, actuator_id: int, value: float) -> None:
        ctrl = float(value)
        if self.model.actuator_ctrllimited[actuator_id]:
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            ctrl = float(np.clip(ctrl, lo, hi))
        data.ctrl[actuator_id] = ctrl
