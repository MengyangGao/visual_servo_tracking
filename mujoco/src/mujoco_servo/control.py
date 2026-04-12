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
from .image_features import bbox_corners_xyxy, order_corners_clockwise
from .types import CameraPose, Detection, ServoTelemetry, TargetPrototype
from .scene import body_pose_world


@dataclass(slots=True)
class ServoGains:
    feature: float = 2.4
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


def _feature_points_from_detection(detection: Detection) -> np.ndarray:
    if detection.corners_px is not None:
        return order_corners_clockwise(detection.corners_px)
    return bbox_corners_xyxy(detection.bbox_xyxy)


def _desired_feature_corners(prototype: TargetPrototype, camera_intrinsics) -> np.ndarray:
    extent = max(float(prototype.size_m[0]), float(prototype.size_m[1]), float(prototype.size_m[2]), 1e-3)
    depth = max(float(prototype.nominal_standoff_m), 1e-3)
    half_extent_px_x = max(16.0, 0.5 * camera_intrinsics.fx * extent / depth)
    half_extent_px_y = max(16.0, 0.5 * camera_intrinsics.fy * extent / depth)
    center_x = float(camera_intrinsics.cx)
    center_y = float(camera_intrinsics.cy)
    return order_corners_clockwise(
        np.array(
            [
                [center_x - half_extent_px_x, center_y - half_extent_px_y],
                [center_x + half_extent_px_x, center_y - half_extent_px_y],
                [center_x + half_extent_px_x, center_y + half_extent_px_y],
                [center_x - half_extent_px_x, center_y + half_extent_px_y],
            ],
            dtype=np.float32,
        )
    )


def _interaction_matrix(points_px: np.ndarray, depth_m: float, intrinsics) -> np.ndarray:
    depth = max(float(depth_m), 1e-6)
    matrix = np.zeros((8, 6), dtype=np.float64)
    for i, (u, v) in enumerate(np.asarray(points_px, dtype=float).reshape(4, 2)):
        x = (u - intrinsics.cx) / intrinsics.fx
        y = (v - intrinsics.cy) / intrinsics.fy
        matrix[2 * i : 2 * i + 2] = np.array(
            [
                [-1.0 / depth, 0.0, x / depth, x * y, -(1.0 + x * x), y],
                [0.0, -1.0 / depth, y / depth, 1.0 + y * y, -x * y, -x],
            ],
            dtype=np.float64,
        )
    return matrix


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
    current_points = _feature_points_from_detection(detection)
    desired_points = _desired_feature_corners(prototype, camera_intrinsics)
    depth = float(detection.estimated_distance_m or prototype.nominal_standoff_m)
    feature_error = np.zeros((8, 1), dtype=np.float64)
    for i in range(4):
        feature_error[2 * i] = (current_points[i, 0] - desired_points[i, 0]) / camera_intrinsics.fx
        feature_error[2 * i + 1] = (current_points[i, 1] - desired_points[i, 1]) / camera_intrinsics.fy
    interaction = _interaction_matrix(current_points, depth, camera_intrinsics)
    camera_twist_cam = -gains.feature * damped_pseudo_inverse(interaction, damping=gains.damping) @ feature_error
    camera_twist_cam = np.asarray(camera_twist_cam, dtype=float).reshape(6)
    camera_twist_world = np.concatenate(
        [
            camera_pose.rotation_world_from_cam @ camera_twist_cam[:3],
            camera_pose.rotation_world_from_cam @ camera_twist_cam[3:],
        ]
    )
    forward = normalize(target_pos - ee_pos)
    if np.any(forward):
        desired_rot = look_at_rotation(forward, np.array([0.0, 0.0, 1.0], dtype=float))
    else:
        desired_rot = ee_rot
    pos_error = target_pos - ee_pos
    ori_error = rotation_matrix_to_axis_angle(desired_rot @ ee_rot.T)
    jacobian = _ee_jacobian(model, data, ee_body_name)
    qvel = damped_pseudo_inverse(jacobian, damping=gains.damping) @ camera_twist_world
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
        feature_error_px=float(np.linalg.norm(feature_error)),
    )
    return qpos, telemetry
