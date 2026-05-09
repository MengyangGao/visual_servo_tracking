from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .config import DepthConfig


@dataclass(slots=True)
class DepthEstimate:
    depth_m: np.ndarray
    backend: str
    metric: bool


class DepthBackend(Protocol):
    name: str

    def estimate(self, frame_bgr: np.ndarray, metric_hint_m: np.ndarray | None = None) -> DepthEstimate:
        ...


class NoDepthBackend:
    name = "none"

    def estimate(self, frame_bgr: np.ndarray, metric_hint_m: np.ndarray | None = None) -> DepthEstimate:
        _validate_frame(frame_bgr)
        h, w = frame_bgr.shape[:2]
        return DepthEstimate(np.full((h, w), np.nan, dtype=np.float32), self.name, False)


class MujocoDepthBackend:
    name = "mujoco"

    def estimate(self, frame_bgr: np.ndarray, metric_hint_m: np.ndarray | None = None) -> DepthEstimate:
        _validate_frame(frame_bgr)
        if metric_hint_m is None:
            raise ValueError("mujoco depth backend requires a metric depth hint")
        return DepthEstimate(_validate_depth(metric_hint_m, frame_bgr.shape[:2], "metric depth hint"), self.name, True)


class DepthAnythingV2Backend:
    name = "depth-anything-v2"

    def __init__(self, config: DepthConfig) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything V2 requires optional semantic dependencies. Install with "
                "`conda run -n visual_servo python -m pip install -e 'mujoco[semantic]'`."
            ) from exc
        device_name = config.device if config.device != "auto" else os.getenv("MUJOCO_SERVO_DEVICE", "auto")
        if device_name == "auto":
            device_name = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        device_arg = 0 if device_name == "cuda" else device_name
        self._image_cls = Image
        self._pipe = pipeline(task="depth-estimation", model=config.model, device=device_arg)
        self._use_metric_hint = bool(config.metric_hint)

    def estimate(self, frame_bgr: np.ndarray, metric_hint_m: np.ndarray | None = None) -> DepthEstimate:
        _validate_frame(frame_bgr)
        image = self._image_cls.fromarray(frame_bgr[:, :, ::-1])
        result = self._pipe(image)
        raw_depth = np.asarray(result["depth"], dtype=np.float32)
        if raw_depth.ndim == 3:
            raw_depth = raw_depth[:, :, 0]
        if raw_depth.shape != frame_bgr.shape[:2]:
            raw_depth = cv2.resize(raw_depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth = self._calibrate_depth(raw_depth, metric_hint_m)
        depth = _validate_depth(depth, frame_bgr.shape[:2], "learned depth")
        return DepthEstimate(depth.astype(np.float32), self.name, metric_hint_m is not None and self._use_metric_hint)

    def _calibrate_depth(self, relative_depth: np.ndarray, metric_hint_m: np.ndarray | None) -> np.ndarray:
        depth = np.asarray(relative_depth, dtype=np.float32)
        valid = np.isfinite(depth)
        if not np.any(valid):
            return np.full(depth.shape, np.nan, dtype=np.float32)
        depth = depth - float(np.min(depth[valid]))
        max_value = float(np.max(depth[valid]))
        if max_value > 1e-6:
            depth = depth / max_value
        if metric_hint_m is None or not self._use_metric_hint:
            return np.maximum(depth, 1e-3)
        hint = _validate_depth(metric_hint_m, depth.shape, "metric depth hint")
        hint_valid = valid & np.isfinite(hint) & (hint > 0.0)
        if not np.any(hint_valid):
            return np.maximum(depth, 1e-3)
        scale = float(np.median(hint[hint_valid]) / max(float(np.median(depth[hint_valid])), 1e-6))
        return np.maximum(depth * scale, 1e-3)


def build_depth_backend(config: DepthConfig) -> DepthBackend:
    backend = config.backend.strip().lower()
    if backend in {"none", "off", "disabled"}:
        return NoDepthBackend()
    if backend in {"mujoco", "sim", "simulation", "metric"}:
        return MujocoDepthBackend()
    if backend in {"depth-anything-v2", "depth_anything_v2", "depth-anything", "depthanything"}:
        return DepthAnythingV2Backend(config)
    raise ValueError(f"unknown depth backend '{config.backend}'")


def _validate_frame(frame_bgr: np.ndarray) -> None:
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame_bgr must have shape (height, width, 3)")


def _validate_depth(depth_m: np.ndarray, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != expected_shape:
        raise ValueError(f"{name} shape {depth.shape} does not match frame shape {expected_shape}")
    if np.isinf(depth).any():
        raise ValueError(f"{name} must not contain infinite values")
    return depth.copy()
