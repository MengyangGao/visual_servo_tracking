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
        depth = _validate_depth(metric_hint_m, frame_bgr.shape[:2], "metric depth hint")
        valid = np.isfinite(depth) & (depth > 0.0)
        depth[~valid] = np.nan
        return DepthEstimate(depth, self.name, bool(np.any(valid)))


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
                "`python -m pip install -e 'mujoco[semantic]'`."
            ) from exc
        requested_device = config.device if config.device.strip().lower() != "auto" else os.getenv("MUJOCO_SERVO_DEVICE", "auto")
        device_name = _resolve_torch_device_name(torch, requested_device)
        device_arg = _pipeline_device_arg(device_name)
        self._image_cls = Image
        self._pipe = pipeline(task="depth-estimation", model=config.model, device=device_arg)
        self._use_metric_hint = bool(config.metric_hint)

    def estimate(self, frame_bgr: np.ndarray, metric_hint_m: np.ndarray | None = None) -> DepthEstimate:
        _validate_frame(frame_bgr)
        image = self._image_cls.fromarray(frame_bgr[:, :, ::-1])
        result = self._pipe(image)
        if not isinstance(result, dict):
            raise RuntimeError("depth-estimation pipeline must return a mapping")
        numeric_output = result.get("predicted_depth") is not None
        raw_value = result.get("predicted_depth") if numeric_output else result.get("depth")
        if raw_value is None:
            raise RuntimeError("depth-estimation pipeline returned neither predicted_depth nor depth")
        raw_depth = _depth_array(raw_value)
        if raw_depth.shape != frame_bgr.shape[:2]:
            raw_depth = cv2.resize(raw_depth, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth, metric = self._calibrate_depth(raw_depth, metric_hint_m, numeric_output=numeric_output)
        depth = _validate_depth(depth, frame_bgr.shape[:2], "learned depth")
        return DepthEstimate(depth.astype(np.float32), self.name, metric)

    def _calibrate_depth(
        self,
        relative_depth: np.ndarray,
        metric_hint_m: np.ndarray | None,
        *,
        numeric_output: bool = True,
    ) -> tuple[np.ndarray, bool]:
        raw = np.asarray(relative_depth, dtype=np.float32)
        if metric_hint_m is not None and self._use_metric_hint:
            hint = _validate_depth(metric_hint_m, raw.shape, "metric depth hint")
            calibration = _fit_metric_calibration(raw, hint) if numeric_output else None
            if calibration is not None:
                kind, slope, intercept = calibration
                source = _calibration_source(raw, kind)
                calibrated = slope * source + intercept
                valid = np.isfinite(calibrated) & (calibrated > 0.0)
                calibrated = np.where(valid, calibrated, np.nan).astype(np.float32)
                return calibrated, True
            return _relative_depth_map(raw), False
        return _relative_depth_map(raw), False


def _resolve_torch_device_name(torch_module, requested: str) -> str:
    device = str(requested).strip().lower()
    device = {"gpu": "cuda", "metal": "mps"}.get(device, device)
    if device and device != "auto":
        return device
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _pipeline_device_arg(device_name: str):
    if device_name == "cuda":
        return 0
    if device_name.startswith("cuda:"):
        try:
            return int(device_name.split(":", 1)[1])
        except ValueError:
            return device_name
    return device_name


def _depth_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    depth = np.asarray(value, dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim == 3 and depth.shape[-1] in {1, 3, 4}:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise RuntimeError(f"depth-estimation pipeline returned invalid depth shape {depth.shape}")
    return depth


def _relative_depth_map(raw_depth: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_depth, dtype=np.float32)
    valid = np.isfinite(raw)
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return result
    values = raw[valid]
    lo, hi = np.percentile(values, [1.0, 99.0])
    span = float(hi - lo)
    if not np.isfinite(span) or span <= max(1e-6, 1e-6 * max(abs(float(lo)), abs(float(hi)), 1.0)):
        result[valid] = 1e-3
        return result
    normalized = np.clip((raw[valid] - float(lo)) / span, 0.0, 1.0)
    result[valid] = 1e-3 + 0.999 * normalized
    return result


def _calibration_source(raw_depth: np.ndarray, kind: str) -> np.ndarray:
    raw = np.asarray(raw_depth, dtype=np.float64)
    if kind == "raw":
        return raw
    source = np.full(raw.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(raw) & (np.abs(raw) > 1e-8)
    source[valid] = 1.0 / raw[valid]
    return source


def _fit_metric_calibration(raw_depth: np.ndarray, metric_hint_m: np.ndarray) -> tuple[str, float, float] | None:
    raw = np.asarray(raw_depth, dtype=np.float64)
    hint = np.asarray(metric_hint_m, dtype=np.float64)
    common = np.isfinite(raw) & np.isfinite(hint) & (hint > 0.0)
    if np.count_nonzero(common) < 16:
        return None
    hint_values = hint[common]
    hint_lo, hint_hi = np.percentile(hint_values, [5.0, 95.0])
    hint_span = float(hint_hi - hint_lo)
    hint_scale = max(float(np.median(hint_values)), 1e-6)
    if not np.isfinite(hint_span) or hint_span <= max(1e-5, 0.01 * hint_scale):
        return None
    candidates: list[tuple[float, str, float, float]] = []
    for kind in ("raw", "inverse"):
        source = _calibration_source(raw, kind)
        valid = common & np.isfinite(source)
        if np.count_nonzero(valid) < 16:
            continue
        x = source[valid]
        y = hint[valid]
        x_lo, x_hi = np.percentile(x, [5.0, 95.0])
        if float(x_hi - x_lo) <= max(1e-8, 1e-6 * max(abs(float(x_lo)), abs(float(x_hi)), 1.0)):
            continue
        fitted = _robust_affine_fit(x, y)
        if fitted is None:
            continue
        slope, intercept = fitted
        prediction = slope * x + intercept
        if np.mean(np.isfinite(prediction) & (prediction > 0.0)) < 0.98:
            continue
        relative_error = np.abs(prediction - y) / np.maximum(y, 1e-3)
        median_error = float(np.median(relative_error))
        p90_error = float(np.percentile(relative_error, 90.0))
        predicted_span = abs(slope) * float(x_hi - x_lo)
        if predicted_span < 0.25 * hint_span:
            continue
        if median_error <= 0.08 and p90_error <= 0.20:
            candidates.append((median_error + 0.25 * p90_error, kind, float(slope), float(intercept)))
    if not candidates:
        return None
    _, kind, slope, intercept = min(candidates, key=lambda item: item[0])
    return kind, slope, intercept


def _robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    active = np.ones(x.shape, dtype=bool)
    slope = intercept = 0.0
    for _ in range(4):
        if np.count_nonzero(active) < 16 or float(np.std(x[active])) <= 1e-10:
            return None
        design = np.column_stack([x[active], np.ones(np.count_nonzero(active), dtype=np.float64)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[active], rcond=None)
        if rank < 2 or not np.isfinite(coefficients).all():
            return None
        slope, intercept = (float(coefficients[0]), float(coefficients[1]))
        residual = y - (slope * x + intercept)
        center = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - center)))
        threshold = max(3.5 * 1.4826 * mad, 0.01 * max(float(np.median(y[active])), 1e-6), 1e-5)
        updated = np.abs(residual - center) <= threshold
        if np.count_nonzero(updated) < 16 or np.array_equal(updated, active):
            break
        active = updated
    return slope, intercept


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
    if frame.dtype != np.uint8:
        raise ValueError("frame_bgr must use uint8 BGR pixels")


def _validate_depth(depth_m: np.ndarray, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != expected_shape:
        raise ValueError(f"{name} shape {depth.shape} does not match frame shape {expected_shape}")
    if np.isinf(depth).any():
        raise ValueError(f"{name} must not contain infinite values")
    return depth.copy()
