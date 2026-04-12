from __future__ import annotations

import numpy as np


def normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm < eps:
        return arr * 0.0
    return arr / norm


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vec, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def rotation_matrix_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(trace))
    if angle < 1e-8:
        return np.zeros(3, dtype=float)
    denom = 2.0 * np.sin(angle)
    axis = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=float
    ) / denom
    return axis * angle


def rotation_matrix_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    quat = np.zeros(4, dtype=float)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat[0] = 0.25 * s
        quat[1] = (r[2, 1] - r[1, 2]) / s
        quat[2] = (r[0, 2] - r[2, 0]) / s
        quat[3] = (r[1, 0] - r[0, 1]) / s
        return quat
    diag = np.diag(r)
    idx = int(np.argmax(diag))
    if idx == 0:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        quat[0] = (r[2, 1] - r[1, 2]) / s
        quat[1] = 0.25 * s
        quat[2] = (r[0, 1] + r[1, 0]) / s
        quat[3] = (r[0, 2] + r[2, 0]) / s
    elif idx == 1:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        quat[0] = (r[0, 2] - r[2, 0]) / s
        quat[1] = (r[0, 1] + r[1, 0]) / s
        quat[2] = 0.25 * s
        quat[3] = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        quat[0] = (r[1, 0] - r[0, 1]) / s
        quat[1] = (r[0, 2] + r[2, 0]) / s
        quat[2] = (r[1, 2] + r[2, 1]) / s
        quat[3] = 0.25 * s
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat /= norm
    return quat


def look_at_rotation(forward_world: np.ndarray, up_hint_world: np.ndarray | None = None) -> np.ndarray:
    forward = normalize(forward_world)
    if not np.any(forward):
        return np.eye(3, dtype=float)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=float) if up_hint_world is None else normalize(up_hint_world)
    x_axis = np.cross(up_hint, forward)
    if np.linalg.norm(x_axis) < 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = np.cross(up_hint, forward)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(forward, x_axis))
    return np.column_stack([x_axis, y_axis, forward])


def damped_pseudo_inverse(jacobian: np.ndarray, damping: float = 1e-2) -> np.ndarray:
    j = np.asarray(jacobian, dtype=float)
    m, n = j.shape
    if m <= n:
        return j.T @ np.linalg.inv(j @ j.T + (damping**2) * np.eye(m))
    return np.linalg.inv(j.T @ j + (damping**2) * np.eye(n)) @ j.T
