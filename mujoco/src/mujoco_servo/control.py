from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .geometry import (
    damped_pseudo_inverse,
    look_at_rotation,
    normalize,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_quaternion_wxyz,
)
from .types import CameraPose, Detection, ServoTelemetry, TargetPrototype
from .scene import body_pose_world


@dataclass(slots=True)
class ServoGains:
    position: float = 2.0
    orientation: float = 1.0
    damping: float = 0.08
    max_joint_delta: float = 0.10


def _ee_jacobian(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"end-effector body '{body_name}' not found")
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return np.vstack([jacp, jacr])


def _desired_target_position(
    detection: Detection,
    prototype: TargetPrototype,
    camera_intrinsics,
    camera_pose: CameraPose,
    ee_position: np.ndarray,
) -> np.ndarray:
    if detection.target_position_world is not None:
        return np.asarray(detection.target_position_world, dtype=float)
    if detection.estimated_distance_m is None:
        depth = prototype.nominal_standoff_m
    else:
        depth = float(detection.estimated_distance_m)
    u, v = np.asarray(detection.centroid_px, dtype=float).reshape(2)
    ray_cam = np.array(
        [
            (u - camera_intrinsics.cx) / camera_intrinsics.fx,
            (v - camera_intrinsics.cy) / camera_intrinsics.fy,
            1.0,
        ],
        dtype=float,
    )
    ray_cam = normalize(ray_cam)
    point_cam = ray_cam * depth
    point_world = camera_pose.rotation_world_from_cam @ point_cam + camera_pose.translation_m
    return point_world


def compute_servo_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    detection: Detection,
    prototype: TargetPrototype,
    camera_intrinsics,
    camera_pose: CameraPose,
    ee_body_name: str,
    gains: ServoGains,
    dt: float,
) -> tuple[np.ndarray, ServoTelemetry]:
    ee_pos, ee_rot = body_pose_world(model, data, ee_body_name)
    target_pos = _desired_target_position(detection, prototype, camera_intrinsics, camera_pose, ee_pos)
    offset = target_pos - ee_pos
    distance = float(np.linalg.norm(offset))
    if distance > 1e-9:
        desired_pos = target_pos - normalize(offset) * prototype.nominal_standoff_m
    else:
        desired_pos = target_pos.copy()
    forward = normalize(target_pos - ee_pos)
    if np.any(forward):
        desired_rot = look_at_rotation(forward, np.array([0.0, 0.0, 1.0], dtype=float))
    else:
        desired_rot = ee_rot
    pos_error = desired_pos - ee_pos
    ori_error = rotation_matrix_to_axis_angle(desired_rot @ ee_rot.T)
    twist = np.concatenate([gains.position * pos_error, gains.orientation * ori_error])
    jacobian = _ee_jacobian(model, data, ee_body_name)
    qvel = damped_pseudo_inverse(jacobian, damping=gains.damping) @ twist
    qvel = np.asarray(qvel, dtype=float)
    if qvel.shape[0] > 0:
        norm = float(np.linalg.norm(qvel))
        if norm > gains.max_joint_delta / max(dt, 1e-9):
            qvel = qvel / norm * (gains.max_joint_delta / max(dt, 1e-9))
    qpos = np.array(data.qpos.copy(), dtype=float)
    if model.nu >= 7:
        qpos[:7] = qpos[:7] + qvel[:7] * dt
    telemetry = ServoTelemetry(
        step=0,
        prompt=detection.prompt,
        backend=detection.backend,
        qpos=qpos.copy(),
        qvel=qvel.copy(),
        ee_position_m=ee_pos.copy(),
        ee_orientation_wxyz=rotation_matrix_to_quaternion_wxyz(ee_rot),
        target_position_m=target_pos.copy(),
        position_error_m=float(np.linalg.norm(pos_error)),
        orientation_error_rad=float(np.linalg.norm(ori_error)),
        detection_score=float(detection.score),
    )
    return qpos, telemetry
