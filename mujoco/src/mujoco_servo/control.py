from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import ControllerConfig, default_home_qpos
from .math_utils import clamp_norm, damped_pseudo_inverse, normalize
from .scene import site_position


@dataclass(slots=True)
class ServoState:
    step: int
    time_s: float
    ee_position: np.ndarray
    target_position: np.ndarray
    desired_position: np.ndarray
    position_error_m: float
    target_distance_m: float
    qpos_command: np.ndarray


def desired_ee_position(task: str, target_position: np.ndarray, ee_position: np.ndarray, config: ControllerConfig) -> np.ndarray:
    target = np.asarray(target_position, dtype=float).reshape(3)
    ee = np.asarray(ee_position, dtype=float).reshape(3)
    mode = task.strip().lower()
    if mode == "contact":
        return target.copy()
    if mode == "standoff":
        direction = normalize(ee - target, np.array([-1.0, 0.0, 0.0]))
        return target + direction * float(config.standoff_m)
    if mode == "align-x":
        return np.array([target[0] + config.align_offset_m, ee[1], ee[2]], dtype=float)
    if mode == "align-y":
        return np.array([ee[0], target[1] + config.align_offset_m, ee[2]], dtype=float)
    if mode == "align-z":
        return np.array([ee[0], ee[1], target[2] + config.align_offset_m], dtype=float)
    raise ValueError(f"unknown servo task '{task}'")


class ResolvedRateController:
    def __init__(self, model: mujoco.MjModel, ee_site_name: str, config: ControllerConfig) -> None:
        self.model = model
        self.ee_site_name = ee_site_name
        self.config = config
        self._qpos_command = default_home_qpos()
        self._filtered_target: np.ndarray | None = None
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ]
        if any(joint_id < 0 for joint_id in self._joint_ids):
            raise RuntimeError("expected joints joint1..joint7")
        self._qpos_adr = np.array([model.jnt_qposadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._dof_adr = np.array([model.jnt_dofadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self._site_id < 0:
            raise RuntimeError(f"site '{ee_site_name}' not found")

    def reset(self, data: mujoco.MjData) -> None:
        self._qpos_command = np.array(data.qpos[self._qpos_adr], dtype=float)
        self._filtered_target = None

    def step(self, data: mujoco.MjData, target_position: np.ndarray, time_s: float, step_index: int) -> ServoState:
        target = np.asarray(target_position, dtype=float).reshape(3)
        if self._filtered_target is None:
            self._filtered_target = target.copy()
        else:
            alpha = float(self.config.smooth_target_alpha)
            self._filtered_target = (1.0 - alpha) * self._filtered_target + alpha * target

        ee_pos = site_position(self.model, data, self.ee_site_name)
        desired = desired_ee_position(self.config.task, self._filtered_target, ee_pos, self.config)
        error = desired - ee_pos
        ee_velocity = clamp_norm(float(self.config.position_gain) * error, self.config.max_ee_speed)

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self._site_id)
        arm_jac = jacp[:, self._dof_adr]
        qvel = damped_pseudo_inverse(arm_jac, self.config.damping) @ ee_velocity

        home_error = default_home_qpos() - np.asarray(data.qpos[self._qpos_adr], dtype=float)
        nullspace = np.eye(7) - damped_pseudo_inverse(arm_jac, self.config.damping) @ arm_jac
        qvel = qvel + 0.18 * (nullspace @ home_error)
        qvel = clamp_norm(qvel, self.config.max_joint_speed)

        dt = 1.0 / float(self.config.control_hz)
        current_qpos = np.asarray(data.qpos[self._qpos_adr], dtype=float)
        self._qpos_command = self._qpos_command + qvel * dt
        self._qpos_command = np.clip(self._qpos_command, current_qpos - 0.22, current_qpos + 0.22)
        for i, joint_id in enumerate(self._joint_ids):
            lo, hi = self.model.jnt_range[joint_id]
            self._qpos_command[i] = np.clip(self._qpos_command[i], lo + 1e-4, hi - 1e-4)
        data.ctrl[:7] = self._qpos_command

        return ServoState(
            step=step_index,
            time_s=time_s,
            ee_position=ee_pos,
            target_position=self._filtered_target.copy(),
            desired_position=desired,
            position_error_m=float(np.linalg.norm(error)),
            target_distance_m=float(np.linalg.norm(self._filtered_target - ee_pos)),
            qpos_command=self._qpos_command.copy(),
        )
