from __future__ import annotations

import inspect
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .estimation import robust_point_estimate, select_near_surface_support
from .targets import TargetSpec


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(slots=True)
class CameraObservation:
    frame_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    camera_position: np.ndarray
    camera_xmat: np.ndarray
    wall_time_s: float = 0.0
    sim_time_s: float = 0.0
    depth_backend: str = "unknown"
    depth_metric: bool = False


@dataclass(slots=True)
class Detection:
    success: bool
    backend: str
    target_position: np.ndarray | None
    score: float = 0.0
    bbox_xyxy: np.ndarray | None = None
    centroid_px: np.ndarray | None = None
    mask: np.ndarray | None = None
    anchor_type: str = "unknown"
    world_bbox_min: np.ndarray | None = None
    world_bbox_max: np.ndarray | None = None
    # capture_time_s is the wall-clock timestamp attached to the source frame;
    # measurement_time_s is its simulation timestamp.  Defaults preserve the
    # original three-argument construction used by third-party backends.
    capture_time_s: float = 0.0
    measurement_time_s: float = 0.0
    covariance: np.ndarray | None = None
    valid_fraction: float = 0.0
    quality: float = 0.0
    inference_time_s: float = 0.0


@dataclass(slots=True)
class _WorldAnchorEstimate:
    position: np.ndarray | None
    mask: np.ndarray
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    anchor_type: str
    covariance: np.ndarray | None = None
    valid_fraction: float = 0.0
    quality: float = 0.0


class PerceptionBackend(Protocol):
    name: str

    def detect(
        self,
        observation: CameraObservation | None,
        truth_position: np.ndarray,
        target: TargetSpec,
        prompt: str,
    ) -> Detection: ...


def _bbox_mask(frame_shape: tuple[int, int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    bbox = _valid_bbox(bbox_xyxy)
    if bbox is None:
        return mask
    x1, y1, x2, y2 = bbox
    if x2 <= 0.0 or y2 <= 0.0 or x1 >= w or y1 >= h:
        return mask
    left = max(0, min(w - 1, int(np.floor(x1))))
    top = max(0, min(h - 1, int(np.floor(y1))))
    right = max(left + 1, min(w, int(np.ceil(x2))))
    bottom = max(top + 1, min(h, int(np.ceil(y2))))
    mask[top:bottom, left:right] = 255
    return mask


def _estimate_world_position(
    observation: CameraObservation,
    bbox_xyxy: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    estimate = _estimate_world_anchor(observation, bbox_xyxy, mask)
    return (
        estimate.position,
        estimate.mask,
        estimate.bbox_min,
        estimate.bbox_max,
        estimate.anchor_type,
    )


def _estimate_world_anchor(
    observation: CameraObservation,
    bbox_xyxy: np.ndarray,
    mask: np.ndarray | None,
) -> _WorldAnchorEstimate:
    _validate_observation(observation)
    bbox = _valid_bbox(bbox_xyxy)
    frame_shape = observation.frame_bgr.shape
    empty_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if bbox is None:
        return _WorldAnchorEstimate(None, empty_mask, None, None, "invalid_bbox")
    if mask is None:
        mask = _bbox_mask(frame_shape, bbox)
    else:
        mask = np.asarray(mask)
        if mask.shape != frame_shape[:2]:
            raise ValueError("mask shape must match observation frame shape")
        mask = (mask > 0).astype(np.uint8) * 255
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    if x2 <= 0.0 or y2 <= 0.0 or x1 >= w or y1 >= h:
        return _WorldAnchorEstimate(None, empty_mask, None, None, "invalid_bbox")
    clipped_bbox = np.array(
        [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)],
        dtype=float,
    )
    bbox_support = _bbox_mask(frame_shape, clipped_bbox)
    mask = cv2.bitwise_and(mask, bbox_support)
    if not observation.depth_metric:
        return _WorldAnchorEstimate(None, mask, None, None, "non_metric_depth")
    valid = mask > 0
    depth = np.asarray(observation.depth_m, dtype=float)
    valid &= np.isfinite(depth)
    valid &= depth > 0.0
    valid = select_near_surface_support(depth, valid)
    if not np.any(valid):
        return _WorldAnchorEstimate(None, mask, None, None, "no_valid_depth")

    ys, xs = np.nonzero(valid)
    if xs.size > 4096:
        indices = np.linspace(0, xs.size - 1, 4096, dtype=int)
        xs_sample = xs[indices]
        ys_sample = ys[indices]
    else:
        xs_sample = xs
        ys_sample = ys
    world_points = _pixels_depth_to_world(
        observation, xs_sample, ys_sample, depth[ys_sample, xs_sample]
    )
    support_mask = valid.astype(np.uint8) * 255
    estimate = robust_point_estimate(
        world_points, support_mask, int(np.count_nonzero(mask))
    )
    return _WorldAnchorEstimate(
        estimate.position,
        estimate.support_mask,
        estimate.world_bbox_min,
        estimate.world_bbox_max,
        "surface_depth_point_cluster",
        covariance=estimate.covariance,
        valid_fraction=estimate.valid_fraction,
        quality=estimate.quality,
    )


def _trim_depth_outliers(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = depth[valid]
    if values.size < 16:
        return valid
    lo, hi = np.percentile(values, [2.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return valid
    trimmed = valid & (depth >= lo) & (depth <= hi)
    return trimmed if np.any(trimmed) else valid


def _pixel_depth_to_world(
    observation: CameraObservation, u: float, v: float, z: float
) -> np.ndarray:
    intr = observation.intrinsics
    # MuJoCo/OpenGL camera coordinates: +x right, +y up, camera looks along -z.
    point_cam = np.array(
        [(u - intr.cx) * z / intr.fx, -(v - intr.cy) * z / intr.fy, -z], dtype=float
    )
    return observation.camera_position + observation.camera_xmat @ point_cam


def _pixels_depth_to_world(
    observation: CameraObservation, us: np.ndarray, vs: np.ndarray, zs: np.ndarray
) -> np.ndarray:
    intr = observation.intrinsics
    us = np.asarray(us, dtype=float)
    vs = np.asarray(vs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    points_cam = np.column_stack(
        [
            (us - intr.cx) * zs / intr.fx,
            -(vs - intr.cy) * zs / intr.fy,
            -zs,
        ]
    )
    camera_position = np.asarray(observation.camera_position, dtype=float).reshape(3)
    camera_xmat = np.asarray(observation.camera_xmat, dtype=float).reshape(3, 3)
    return camera_position + np.einsum("ij,nj->ni", camera_xmat, points_cam)


def _target_color_mask(
    frame_bgr: np.ndarray,
    frame_hsv: np.ndarray,
    target_bgr: np.ndarray,
    target_hsv: np.ndarray,
) -> np.ndarray:
    target_sat = int(target_hsv[1])
    target_val = int(target_hsv[2])
    if target_sat < 55 or target_val < 50:
        target_pixel = np.clip(np.rint(target_bgr), 0, 255).astype(np.uint8)
        frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_pixel.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[
            0, 0
        ].astype(np.float32)
        distance = np.linalg.norm(frame_lab - target_lab, axis=2)
        tolerance = float(max(8, min(32, round(7.0 + 0.12 * target_val))))
        selected = distance <= tolerance
        if target_val < 50:
            value = frame_hsv[:, :, 2]
            saturation = frame_hsv[:, :, 1]
            value_low = max(3, int(round(0.25 * target_val)))
            value_high = min(255, max(target_val + 24, 2 * target_val))
            dark_band = (value >= value_low) & (value <= value_high)
            if target_sat < 55:
                dark_band &= saturation <= min(255, max(90, target_sat + 70))
            else:
                chroma_distance = np.linalg.norm(
                    frame_lab[:, :, 1:] - target_lab[1:], axis=2
                )
                dark_band &= chroma_distance <= 18.0
            selected |= dark_band
        return selected.astype(np.uint8) * 255
    hue = int(target_hsv[0])
    min_sat = max(35, target_sat - 150)
    min_val = max(20, int(round(0.22 * target_val)))
    return _hue_range_mask(
        frame_hsv, hue, tolerance=12, min_sat=min_sat, min_val=min_val
    )


def _is_boundary_background_contour(
    contour: np.ndarray, frame_shape: tuple[int, int, int]
) -> bool:
    h, w = frame_shape[:2]
    area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    touches_boundary = x <= 0 or y <= 0 or x + width >= w or y + height >= h
    area_fraction = area / max(1.0, float(h * w))
    spans_frame = width >= 0.90 * w or height >= 0.90 * h
    return area_fraction >= 0.92 or (
        touches_boundary and (area_fraction >= 0.35 or spans_frame)
    )


class OraclePerception:
    name = "oracle"

    def detect(
        self,
        observation: CameraObservation | None,
        truth_position: np.ndarray,
        target: TargetSpec,
        prompt: str,
    ) -> Detection:
        position = np.asarray(truth_position, dtype=float).reshape(3).copy()
        return Detection(
            success=True,
            backend=self.name,
            target_position=position,
            score=1.0,
            anchor_type="truth_center",
            capture_time_s=0.0 if observation is None else observation.wall_time_s,
            measurement_time_s=0.0 if observation is None else observation.sim_time_s,
            covariance=np.eye(3, dtype=float) * 1e-12,
            valid_fraction=1.0,
            quality=1.0,
        )


class ColorSegmentationPerception:
    name = "color"

    def __init__(self) -> None:
        self._last_bbox: np.ndarray | None = None
        self._last_area: float | None = None
        self._last_depth_median: float | None = None
        self._last_aspect: float | None = None
        self._last_solidity: float | None = None
        self._last_centroid: np.ndarray | None = None
        self._velocity_px = np.zeros(2, dtype=float)
        self._lost_frames = 0
        self._confidence = 0.0
        self._candidate_quality = 0.0

    def detect(
        self,
        observation: CameraObservation | None,
        truth_position: np.ndarray,
        target: TargetSpec,
        prompt: str,
    ) -> Detection:
        if observation is None:
            return Detection(False, self.name, None)
        _validate_observation(observation)
        frame_bgr = observation.frame_bgr
        rgba = np.array(target.rgba[:3], dtype=float)
        target_bgr = np.array([rgba[2], rgba[1], rgba[0]], dtype=float) * 255.0
        hsv_color = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = _target_color_mask(frame_bgr, frame_hsv, target_bgr, hsv_color)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [
            contour
            for contour in contours
            if not _is_boundary_background_contour(contour, frame_bgr.shape)
        ]
        if not contours:
            return self._mark_lost(observation, mask)
        contour = self._select_contour(contours, observation, target)
        if contour is None:
            return self._mark_lost(observation, mask)
        area = float(cv2.contourArea(contour))
        if area < 20.0:
            return self._mark_lost(observation, mask)
        selected_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(selected_mask, [contour], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = np.array([x, y, x + w, y + h], dtype=float)
        moments = cv2.moments(contour)
        cx = (
            x + 0.5 * w
            if abs(moments["m00"]) < 1e-9
            else moments["m10"] / moments["m00"]
        )
        cy = (
            y + 0.5 * h
            if abs(moments["m00"]) < 1e-9
            else moments["m01"] / moments["m00"]
        )
        anchor = _estimate_world_anchor(observation, bbox, selected_mask)
        position = anchor.position
        selected_mask = anchor.mask
        self._last_bbox = bbox.copy()
        self._last_area = area
        aspect = float(w) / max(float(h), 1.0)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        centroid = np.array([cx, cy], dtype=float)
        if self._last_centroid is not None:
            observed_velocity = centroid - self._last_centroid
            self._velocity_px = 0.65 * self._velocity_px + 0.35 * observed_velocity
        self._last_centroid = centroid
        self._last_aspect = aspect
        self._last_solidity = solidity
        depth_values = observation.depth_m[
            (selected_mask > 0)
            & np.isfinite(observation.depth_m)
            & (observation.depth_m > 0.0)
        ]
        self._last_depth_median = (
            float(np.median(depth_values)) if depth_values.size else None
        )
        self._lost_frames = 0
        anchor_quality = anchor.quality if position is not None else 0.0
        self._confidence = float(
            np.clip(
                0.45 * self._confidence
                + 0.55 * min(self._candidate_quality, anchor_quality),
                0.0,
                1.0,
            )
        )
        return Detection(
            success=position is not None,
            backend=self.name,
            target_position=position,
            score=self._confidence,
            bbox_xyxy=bbox,
            centroid_px=centroid,
            mask=selected_mask,
            anchor_type=anchor.anchor_type,
            world_bbox_min=anchor.bbox_min,
            world_bbox_max=anchor.bbox_max,
            capture_time_s=observation.wall_time_s,
            measurement_time_s=observation.sim_time_s,
            covariance=anchor.covariance,
            valid_fraction=anchor.valid_fraction,
            quality=min(self._confidence, anchor_quality),
        )

    def _mark_lost(
        self, observation: CameraObservation, mask: np.ndarray | None
    ) -> Detection:
        self._lost_frames += 1
        self._confidence *= 0.55
        self._velocity_px *= 0.5
        return Detection(
            False,
            self.name,
            None,
            score=self._confidence,
            mask=mask,
            capture_time_s=observation.wall_time_s,
            measurement_time_s=observation.sim_time_s,
            quality=self._confidence,
        )

    def _select_contour(
        self,
        contours: list[np.ndarray],
        observation: CameraObservation,
        target: TargetSpec,
    ) -> np.ndarray | None:
        h, w = observation.frame_bgr.shape[:2]
        frame_center = np.array([0.5 * (w - 1), 0.5 * (h - 1)], dtype=float)
        frame_diagonal = max(float(np.hypot(w, h)), 1.0)
        previous_center = None
        previous_diagonal = 1.0
        if self._last_bbox is not None:
            x1, y1, x2, y2 = self._last_bbox
            previous_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=float)
            previous_diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)

        ranked: list[tuple[float, np.ndarray]] = []
        approximate_face_area = float(
            np.prod(np.sort(np.asarray(target.size, dtype=float))[-2:])
        )
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 20.0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                x, y, width, height = cv2.boundingRect(contour)
                center = np.array([x + 0.5 * width, y + 0.5 * height], dtype=float)
            else:
                center = np.array(
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                    dtype=float,
                )
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            valid_depth = (
                (contour_mask > 0)
                & np.isfinite(observation.depth_m)
                & (observation.depth_m > 0.0)
            )
            depth_median = (
                float(np.median(observation.depth_m[valid_depth]))
                if np.any(valid_depth)
                else None
            )
            x, y, width, height = cv2.boundingRect(contour)
            aspect = float(width) / max(float(height), 1.0)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(hull_area, 1.0)

            if previous_center is not None and self._last_area is not None:
                expected_center = previous_center + self._velocity_px
                motion_px = float(np.linalg.norm(center - expected_center))
                motion_limit = (
                    max(8.0, 0.45 * previous_diagonal) + min(self._lost_frames, 3) * 1.5
                )
                area_ratio = area / max(self._last_area, 1.0)
                if motion_px > motion_limit or not 0.45 <= area_ratio <= 2.0:
                    continue
                if self._last_aspect is not None:
                    aspect_ratio = aspect / max(self._last_aspect, 1e-3)
                    if not 0.55 <= aspect_ratio <= 1.8:
                        continue
                if (
                    self._last_solidity is not None
                    and abs(solidity - self._last_solidity) > 0.28
                ):
                    continue
                if depth_median is not None and self._last_depth_median is not None:
                    depth_limit = max(0.035, 0.08 * self._last_depth_median)
                    if abs(depth_median - self._last_depth_median) > depth_limit:
                        continue
                    delta_x_m = (
                        (center[0] - expected_center[0])
                        * depth_median
                        / observation.intrinsics.fx
                    )
                    delta_y_m = (
                        (center[1] - expected_center[1])
                        * depth_median
                        / observation.intrinsics.fy
                    )
                    displacement_m = float(
                        np.linalg.norm(
                            [
                                delta_x_m,
                                delta_y_m,
                                depth_median - self._last_depth_median,
                            ]
                        )
                    )
                    displacement_limit_m = min(0.15, max(0.06, 1.25 * max(target.size)))
                    if displacement_m > displacement_limit_m:
                        continue
                motion_cost = motion_px / max(motion_limit, 1.0)
                area_cost = abs(float(np.log(area_ratio)))
                depth_cost = 0.0
                if depth_median is not None and self._last_depth_median is not None:
                    depth_cost = abs(depth_median - self._last_depth_median) / max(
                        0.04, 0.12 * self._last_depth_median
                    )
                shape_cost = (
                    0.0
                    if self._last_aspect is None
                    else abs(float(np.log(aspect / max(self._last_aspect, 1e-3))))
                )
                score = (
                    motion_cost
                    + 0.40 * area_cost
                    + 0.55 * depth_cost
                    + 0.25 * shape_cost
                )
            else:
                center_cost = (
                    float(np.linalg.norm(center - frame_center)) / frame_diagonal
                )
                size_cost = 0.0
                if observation.depth_metric and depth_median is not None:
                    expected_area = (
                        observation.intrinsics.fx
                        * observation.intrinsics.fy
                        * approximate_face_area
                        / max(depth_median * depth_median, 1e-6)
                    )
                    size_cost = abs(
                        float(np.log(max(area, 1.0) / max(expected_area, 1.0)))
                    )
                score = 2.5 * center_cost + 0.35 * size_cost
            ranked.append((score, contour))
        if not ranked:
            return None
        best_score, best_contour = min(ranked, key=lambda item: item[0])
        self._candidate_quality = float(np.exp(-best_score))
        if previous_center is None:
            self._candidate_quality = max(self._candidate_quality, 0.65)
        return best_contour


def _resolve_torch_device_name(torch_module, requested: str) -> str:
    device = str(requested).strip().lower()
    aliases = {"gpu": "cuda", "metal": "mps"}
    device = aliases.get(device, device)
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


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sanitize_instance_mask(mask: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.ndim != 2:
        raise ValueError("instance mask must be two-dimensional")
    support = _bbox_mask((binary.shape[0], binary.shape[1], 1), bbox_xyxy)
    binary = cv2.bitwise_and(binary, support)
    if not np.any(binary):
        return binary
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count <= 1:
        return np.zeros_like(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    component = 1 + int(np.argmax(areas))
    return (labels == component).astype(np.uint8) * 255


def _mask_centroid(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=float)


class SemanticPerception:
    name = "semantic"

    def __init__(
        self,
        *,
        grounding_model: str | None = None,
        sam_model: str | None = None,
        grounding_revision: str | None = None,
        sam_revision: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import (
                GroundingDinoForObjectDetection,
                GroundingDinoProcessor,
                SamModel,
                SamProcessor,
            )
        except Exception as exc:
            raise RuntimeError(
                "semantic perception requires optional dependencies. Install with "
                "`python -m pip install -e 'mujoco[semantic]'`."
            ) from exc
        self._torch = torch
        self._image_cls = Image
        gdino_id = grounding_model or os.getenv(
            "MUJOCO_SERVO_GDINO_MODEL", "IDEA-Research/grounding-dino-tiny"
        )
        sam_id = sam_model or os.getenv(
            "MUJOCO_SERVO_SAM_MODEL", "facebook/sam-vit-base"
        )
        gdino_revision = grounding_revision or os.getenv("MUJOCO_SERVO_GDINO_REVISION")
        selected_sam_revision = sam_revision or os.getenv("MUJOCO_SERVO_SAM_REVISION")
        gdino_kwargs = {} if not gdino_revision else {"revision": gdino_revision}
        sam_kwargs = (
            {} if not selected_sam_revision else {"revision": selected_sam_revision}
        )
        self._gdino_processor = GroundingDinoProcessor.from_pretrained(
            gdino_id, **gdino_kwargs
        )
        self._gdino_model = GroundingDinoForObjectDetection.from_pretrained(
            gdino_id, **gdino_kwargs
        )
        self._sam_processor = SamProcessor.from_pretrained(sam_id, **sam_kwargs)
        self._sam_model = SamModel.from_pretrained(sam_id, **sam_kwargs)
        device_name = _resolve_torch_device_name(
            torch, device or os.getenv("MUJOCO_SERVO_DEVICE", "auto")
        )
        self._device = torch.device(device_name)
        self._gdino_model.to(self._device).eval()
        self._sam_model.to(self._device).eval()
        self._box_threshold = float(
            os.getenv("MUJOCO_SERVO_GDINO_BOX_THRESHOLD", "0.25")
        )
        self._text_threshold = float(
            os.getenv("MUJOCO_SERVO_GDINO_TEXT_THRESHOLD", "0.25")
        )
        self._initialized = False
        self._last_bbox: np.ndarray | None = None
        self._last_mask: np.ndarray | None = None
        self._last_detection: Detection | None = None
        self._hsv_center: np.ndarray | None = None
        self._last_depth_median: float | None = None
        self._frames_since_redetect = 0
        self._track_failures = 0
        self._redetect_interval = int(os.getenv("MUJOCO_SERVO_REDETECT_INTERVAL", "45"))
        self._max_track_failures = int(
            os.getenv("MUJOCO_SERVO_MAX_TRACK_FAILURES", "3")
        )
        self._last_sam_inference_s = 0.0

    def detect(
        self,
        observation: CameraObservation | None,
        truth_position: np.ndarray,
        target: TargetSpec,
        prompt: str,
    ) -> Detection:
        if observation is None:
            return Detection(False, self.name, None)
        _validate_observation(observation)
        if self._initialized:
            self._frames_since_redetect = getattr(self, "_frames_since_redetect", 0) + 1
            redetect_interval = max(1, int(getattr(self, "_redetect_interval", 45)))
            should_redetect = self._frames_since_redetect >= redetect_interval
            if not should_redetect:
                tracked = self._track_from_last_mask(observation)
                if tracked.success:
                    self._track_failures = 0
                    self._last_detection = tracked
                    return tracked
                self._track_failures = getattr(self, "_track_failures", 0) + 1
                # A failed local gate means there is no trustworthy measurement;
                # invoke the semantic detector immediately instead of emitting a
                # stale bbox for several control cycles.
        image = self._image_cls.fromarray(observation.frame_bgr[:, :, ::-1])
        text = prompt.strip().lower()
        if not text.endswith("."):
            text = f"{text}."
        inference_started = time.perf_counter()
        with self._inference_context():
            inputs = self._to_device(
                self._gdino_processor(images=image, text=text, return_tensors="pt")
            )
            outputs = self._gdino_model(**inputs)
            post_process = self._gdino_processor.post_process_grounded_object_detection
            threshold_name = (
                "threshold"
                if "threshold" in inspect.signature(post_process).parameters
                else "box_threshold"
            )
            results = post_process(
                outputs,
                inputs["input_ids"],
                **{
                    threshold_name: self._box_threshold,
                    "text_threshold": self._text_threshold,
                    "target_sizes": [image.size[::-1]],
                },
            )[0]
        gdino_inference_s = time.perf_counter() - inference_started
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        if len(boxes) == 0:
            self._track_failures = getattr(self, "_track_failures", 0) + 1
            if self._track_failures >= int(getattr(self, "_max_track_failures", 3)):
                self._initialized = False
            return Detection(
                False,
                self.name,
                None,
                capture_time_s=observation.wall_time_s,
                measurement_time_s=observation.sim_time_s,
                inference_time_s=gdino_inference_s,
            )
        best = int(self._torch.argmax(scores).item())
        bbox = boxes[best].detach().cpu().numpy().astype(float)
        score = float(scores[best].detach().cpu().item())
        mask = self._sam_mask(image, bbox)
        anchor = _estimate_world_anchor(observation, bbox, mask)
        position = anchor.position
        mask = anchor.mask
        x1, y1, x2, y2 = bbox
        centroid = _mask_centroid(mask)
        if centroid is None:
            centroid = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=float)
        detection = Detection(
            success=position is not None,
            backend=self.name,
            target_position=position,
            score=score,
            bbox_xyxy=bbox,
            centroid_px=centroid,
            mask=mask,
            anchor_type=anchor.anchor_type,
            world_bbox_min=anchor.bbox_min,
            world_bbox_max=anchor.bbox_max,
            capture_time_s=observation.wall_time_s,
            measurement_time_s=observation.sim_time_s,
            covariance=anchor.covariance,
            valid_fraction=anchor.valid_fraction,
            quality=float(np.clip(score * anchor.quality, 0.0, 1.0)),
            inference_time_s=gdino_inference_s + self._last_sam_inference_s,
        )
        if detection.success:
            self._initialized = True
            self._frames_since_redetect = 0
            self._track_failures = 0
            self._last_bbox = bbox.copy()
            self._last_mask = mask.copy()
            self._last_detection = detection
            self._hsv_center = self._mask_hsv_center(observation.frame_bgr, mask)
            self._last_depth_median = self._mask_depth_median(observation.depth_m, mask)
        return detection

    def _sam_mask(self, image, bbox: np.ndarray) -> np.ndarray:
        box = np.asarray(bbox, dtype=float).reshape(4).tolist()
        started = time.perf_counter()
        with self._inference_context():
            inputs = self._to_device(
                self._sam_processor(image, input_boxes=[[box]], return_tensors="pt")
            )
            outputs = self._sam_model(**inputs)
            masks = self._sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
            )[0]
        self._last_sam_inference_s = time.perf_counter() - started
        candidates = _tensor_to_numpy(masks)
        if candidates.ndim < 2:
            raise RuntimeError(f"SAM returned invalid mask shape {candidates.shape}")
        height, width = candidates.shape[-2:]
        candidates = candidates.reshape(-1, height, width)
        scores = _tensor_to_numpy(outputs.iou_scores).reshape(-1)
        best = (
            int(np.argmax(scores[: len(candidates)]))
            if scores.size >= len(candidates)
            else 0
        )
        mask = (candidates[best] > 0).astype(np.uint8) * 255
        return _sanitize_instance_mask(mask, np.asarray(bbox, dtype=float))

    def _track_from_last_mask(self, observation: CameraObservation) -> Detection:
        if self._last_bbox is None:
            return Detection(
                False,
                self.name,
                None,
                capture_time_s=observation.wall_time_s,
                measurement_time_s=observation.sim_time_s,
            )
        depth_mask = self._local_depth_mask(observation.depth_m)
        color_mask = self._local_color_mask(observation.frame_bgr)
        if depth_mask is not None and color_mask is not None:
            candidate = cv2.bitwise_and(depth_mask, color_mask)
        elif color_mask is not None:
            # A depth jump commonly means temporary occlusion or a depth-model
            # failure.  Color may bridge a small motion, but it remains seed-
            # connected and is subject to the same area/edge hard gates below.
            candidate = color_mask
        elif depth_mask is not None:
            # Textureless objects may be tracked by depth alone.  Restrict depth
            # evidence to a modest dilation of the previous mask so a coplanar
            # wall cannot become the new target.
            seed = self._tracking_seed_mask(depth_mask.shape)
            candidate = cv2.bitwise_and(depth_mask, seed)
        else:
            candidate = None
        if candidate is None or not np.any(candidate):
            return Detection(
                False,
                self.name,
                None,
                mask=candidate,
                capture_time_s=observation.wall_time_s,
                measurement_time_s=observation.sim_time_s,
            )
        candidate = self._seed_connected_component(candidate)
        if candidate is None or not np.any(candidate):
            return Detection(
                False,
                self.name,
                None,
                mask=candidate,
                capture_time_s=observation.wall_time_s,
                measurement_time_s=observation.sim_time_s,
            )
        self._active_search_bounds = self._expanded_bbox(
            observation.frame_bgr.shape, self._last_bbox, 0.6
        )
        detection = self._tracking_detection_from_mask(observation, candidate)
        self._active_search_bounds = None
        if detection.success:
            self._last_bbox = detection.bbox_xyxy.copy()
            self._last_mask = detection.mask.copy()
            self._last_detection = detection
            self._last_depth_median = self._mask_depth_median(
                observation.depth_m, detection.mask
            )
        return detection

    def _tracking_detection_from_mask(
        self, observation: CameraObservation, mask: np.ndarray
    ) -> Detection:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(
                False,
                self.name,
                None,
                mask=mask,
                capture_time_s=observation.wall_time_s,
                measurement_time_s=observation.sim_time_s,
            )
        contour = None
        selected_mask = None
        for candidate in sorted(contours, key=cv2.contourArea, reverse=True):
            candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(candidate_mask, [candidate], -1, 255, thickness=cv2.FILLED)
            if self._tracking_mask_is_consistent(candidate_mask):
                contour = candidate
                selected_mask = candidate_mask
                break
        if contour is None or selected_mask is None:
            return Detection(
                False,
                self.name,
                None,
                mask=mask,
                capture_time_s=observation.wall_time_s,
                measurement_time_s=observation.sim_time_s,
            )
        x, y, w, h = cv2.boundingRect(contour)
        bbox = np.array([x, y, x + w, y + h], dtype=float)
        anchor = _estimate_world_anchor(observation, bbox, selected_mask)
        position = anchor.position
        selected_mask = anchor.mask
        moments = cv2.moments(contour)
        cx = (
            x + 0.5 * w
            if abs(moments["m00"]) < 1e-9
            else moments["m10"] / moments["m00"]
        )
        cy = (
            y + 0.5 * h
            if abs(moments["m00"]) < 1e-9
            else moments["m01"] / moments["m00"]
        )
        detection = Detection(
            success=position is not None,
            backend=f"{self.name}-track",
            target_position=position,
            score=float(
                0.92 * self._last_detection.score
                if self._last_detection is not None
                else 0.5
            ),
            bbox_xyxy=bbox,
            centroid_px=np.array([cx, cy], dtype=float),
            mask=selected_mask,
            anchor_type=anchor.anchor_type,
            world_bbox_min=anchor.bbox_min,
            world_bbox_max=anchor.bbox_max,
            capture_time_s=observation.wall_time_s,
            measurement_time_s=observation.sim_time_s,
            covariance=anchor.covariance,
            valid_fraction=anchor.valid_fraction,
            quality=float(
                np.clip(
                    anchor.quality
                    * (
                        self._last_detection.quality
                        if self._last_detection is not None
                        else 0.6
                    ),
                    0.0,
                    1.0,
                )
            ),
        )
        return detection

    def _tracking_mask_is_consistent(self, mask: np.ndarray) -> bool:
        area = float(np.count_nonzero(mask))
        if area < 16.0:
            return False
        if self._last_mask is None or not np.any(self._last_mask):
            return True
        last = np.asarray(self._last_mask) > 0
        current = np.asarray(mask) > 0
        if last.shape != current.shape:
            return False
        last_area = float(np.count_nonzero(last))
        ratio = area / max(last_area, 1.0)
        if ratio < 0.35 or ratio > 1.55:
            return False
        intersection = float(np.count_nonzero(last & current))
        union = float(np.count_nonzero(last | current))
        iou = intersection / max(union, 1.0)
        previous_center = _mask_centroid(last)
        current_center = _mask_centroid(current)
        if previous_center is None or current_center is None:
            return False
        shift = float(np.linalg.norm(current_center - previous_center))
        bbox = _valid_bbox(self._last_bbox)
        if bbox is None:
            return False
        diagonal = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        if shift > max(24.0, 1.5 * diagonal):
            return False
        if iou < 0.03 and shift > max(10.0, 0.65 * diagonal):
            return False
        search_bounds = getattr(self, "_active_search_bounds", None)
        if search_bounds is not None:
            x1, y1, x2, y2 = search_bounds
            ys, xs = np.nonzero(current)
            if xs.size == 0:
                return False
            # Reaching a local search boundary means the component continues
            # beyond observed support, the signature of same-plane flooding.
            if (
                np.min(xs) <= x1
                or np.max(xs) >= x2 - 1
                or np.min(ys) <= y1
                or np.max(ys) >= y2 - 1
            ):
                return False
        return True

    def _tracking_seed_mask(self, shape: tuple[int, int]) -> np.ndarray:
        if self._last_mask is None or np.asarray(self._last_mask).shape != shape:
            return np.zeros(shape, dtype=np.uint8)
        bbox = _valid_bbox(self._last_bbox)
        diagonal = (
            16.0
            if bbox is None
            else float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        )
        radius = max(3, min(13, int(round(0.16 * diagonal))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(
            (np.asarray(self._last_mask) > 0).astype(np.uint8) * 255, kernel
        )

    def _seed_connected_component(self, candidate: np.ndarray) -> np.ndarray | None:
        binary = (np.asarray(candidate) > 0).astype(np.uint8)
        if not np.any(binary):
            return None
        seed = self._tracking_seed_mask(binary.shape) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        ranked: list[tuple[int, int]] = []
        for component in range(1, count):
            component_mask = labels == component
            overlap = int(np.count_nonzero(component_mask & seed))
            if overlap <= 0:
                continue
            ranked.append((overlap, component))
        if not ranked:
            return None
        component = max(ranked, key=lambda item: item[0])[1]
        result = (labels == component).astype(np.uint8) * 255
        return result if int(stats[component, cv2.CC_STAT_AREA]) >= 16 else None

    def _local_depth_mask(self, depth_m: np.ndarray) -> np.ndarray | None:
        if self._last_bbox is None or self._last_depth_median is None:
            return None
        depth = np.asarray(depth_m, dtype=float)
        if not np.isfinite(self._last_depth_median) or self._last_depth_median <= 0.0:
            return None
        x1, y1, x2, y2 = self._expanded_bbox((*depth.shape, 1), self._last_bbox, 0.35)
        roi = depth[y1:y2, x1:x2]
        valid = np.isfinite(roi) & (roi > 0.0)
        if not np.any(valid):
            return None
        tolerance = max(0.025, 0.07 * float(self._last_depth_median))
        roi_mask = valid & (np.abs(roi - float(self._last_depth_median)) <= tolerance)
        mask = np.zeros(depth.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = roi_mask.astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask if np.any(mask) else None

    def _local_color_mask(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self._last_bbox is None or self._hsv_center is None:
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        x1, y1, x2, y2 = self._expanded_bbox(frame_bgr.shape, self._last_bbox, 0.6)
        hue = int(self._hsv_center[0])
        sat = int(self._hsv_center[1])
        val = int(self._hsv_center[2])
        roi_hsv = hsv[y1:y2, x1:x2]
        if sat < 55 or val < 50:
            sat_delta = np.abs(roi_hsv[:, :, 1].astype(np.int16) - sat)
            val_delta = np.abs(roi_hsv[:, :, 2].astype(np.int16) - val)
            roi = (
                (sat_delta <= max(30, 90 - sat))
                & (val_delta <= max(10, int(round(0.30 * max(val, 20)))))
            ).astype(np.uint8) * 255
        else:
            roi = _hue_range_mask(
                roi_hsv,
                hue,
                tolerance=14,
                min_sat=max(35, sat - 70),
                min_val=max(25, val - 90),
            )
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = roi
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask if np.any(mask) else None

    @staticmethod
    def _expanded_bbox(
        frame_shape: tuple[int, int, int], bbox: np.ndarray, scale: float
    ) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        valid_bbox = _valid_bbox(bbox)
        if valid_bbox is None:
            return 0, 0, w, h
        x1, y1, x2, y2 = valid_bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = max(8.0, 0.5 * (x2 - x1) * (1.0 + scale))
        half_h = max(8.0, 0.5 * (y2 - y1) * (1.0 + scale))
        left = max(0, min(w - 1, int(cx - half_w)))
        right = max(left + 1, min(w, int(cx + half_w)))
        top = max(0, min(h - 1, int(cy - half_h)))
        bottom = max(top + 1, min(h, int(cy + half_h)))
        return left, top, right, bottom

    @staticmethod
    def _mask_hsv_center(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        valid = mask > 0
        if not np.any(valid):
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        samples = hsv[valid].astype(float)
        saturation = float(np.median(samples[:, 1]))
        value = float(np.median(samples[:, 2]))
        if saturation < 10.0:
            hue = float(np.median(samples[:, 0]))
        else:
            angles = samples[:, 0] * (2.0 * np.pi / 180.0)
            weights = np.maximum(samples[:, 1], 1.0)
            angle = np.arctan2(
                np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles))
            )
            hue = float((angle * 180.0 / (2.0 * np.pi)) % 180.0)
        return np.array([hue, saturation, value], dtype=float)

    @staticmethod
    def _mask_depth_median(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
        valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid):
            return None
        return float(np.median(depth_m[valid]))

    def _to_device(self, inputs):
        converted = {}
        for key, value in inputs.items():
            if self._torch.is_tensor(value):
                if value.is_floating_point():
                    value = value.to(dtype=self._torch.float32)
                converted[key] = value.to(self._device)
            else:
                converted[key] = value
        return converted

    def _inference_context(self) -> ExitStack:
        """Build a safe inference-only context on CPU, CUDA, or MPS."""

        stack = ExitStack()
        inference_mode = getattr(self._torch, "inference_mode", None)
        no_grad = getattr(self._torch, "no_grad", None)
        if callable(inference_mode):
            stack.enter_context(inference_mode())
        elif callable(no_grad):
            stack.enter_context(no_grad())
        device_type = str(getattr(self._device, "type", self._device)).split(":", 1)[0]
        autocast = getattr(self._torch, "autocast", None)
        if device_type == "cuda" and callable(autocast):
            try:
                stack.enter_context(
                    autocast(device_type="cuda", dtype=self._torch.float16)
                )
            except (RuntimeError, TypeError):
                # Older torch builds and unusual accelerators may not expose a
                # usable autocast context. Inference mode still remains active.
                pass
        return stack


def build_perception(name: str) -> PerceptionBackend:
    normalized = name.strip().lower()
    if normalized in {"oracle", "sim", "simulation"}:
        return OraclePerception()
    if normalized in {"color", "segmentation", "mask"}:
        return ColorSegmentationPerception()
    if normalized in {"semantic", "grounding-dino", "grounded-sam"}:
        return SemanticPerception()
    raise ValueError(f"unknown detector '{name}'")


def _hue_range_mask(
    hsv: np.ndarray, hue: int, tolerance: int, min_sat: int, min_val: int
) -> np.ndarray:
    hue = int(hue) % 180
    tolerance = max(0, int(tolerance))
    low_h = hue - tolerance
    high_h = hue + tolerance
    sat = max(0, min(255, int(min_sat)))
    val = max(0, min(255, int(min_val)))
    if low_h < 0:
        left = cv2.inRange(
            hsv,
            np.array([0, sat, val], dtype=np.uint8),
            np.array([high_h, 255, 255], dtype=np.uint8),
        )
        right = cv2.inRange(
            hsv,
            np.array([180 + low_h, sat, val], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        return cv2.bitwise_or(left, right)
    if high_h > 179:
        left = cv2.inRange(
            hsv,
            np.array([0, sat, val], dtype=np.uint8),
            np.array([high_h - 180, 255, 255], dtype=np.uint8),
        )
        right = cv2.inRange(
            hsv,
            np.array([low_h, sat, val], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        return cv2.bitwise_or(left, right)
    return cv2.inRange(
        hsv,
        np.array([low_h, sat, val], dtype=np.uint8),
        np.array([high_h, 255, 255], dtype=np.uint8),
    )


def _valid_bbox(bbox_xyxy: np.ndarray) -> np.ndarray | None:
    try:
        bbox = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(bbox).all():
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return bbox


def _validate_observation(observation: CameraObservation) -> None:
    frame = np.asarray(observation.frame_bgr)
    depth = np.asarray(observation.depth_m)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("observation frame_bgr must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("observation frame_bgr must use uint8 BGR pixels")
    if depth.shape != frame.shape[:2]:
        raise ValueError("observation depth_m shape must match frame height/width")
    if (
        observation.intrinsics.width != frame.shape[1]
        or observation.intrinsics.height != frame.shape[0]
    ):
        raise ValueError("observation intrinsics size must match frame shape")
    intrinsic_values = np.array(
        [
            observation.intrinsics.fx,
            observation.intrinsics.fy,
            observation.intrinsics.cx,
            observation.intrinsics.cy,
        ],
        dtype=float,
    )
    if not np.isfinite(intrinsic_values).all():
        raise ValueError("observation intrinsics must contain finite values")
    if observation.intrinsics.fx <= 0.0 or observation.intrinsics.fy <= 0.0:
        raise ValueError("observation focal lengths must be positive")
    if not np.isfinite(observation.sim_time_s) or observation.sim_time_s < 0.0:
        raise ValueError("observation sim_time_s must be finite and non-negative")
    if np.isinf(depth).any():
        raise ValueError("observation depth_m must not contain infinite values")
    camera_position = np.asarray(observation.camera_position, dtype=float)
    if camera_position.shape != (3,) or not np.isfinite(camera_position).all():
        raise ValueError("observation camera_position must have shape (3,)")
    camera_xmat = np.asarray(observation.camera_xmat, dtype=float)
    if camera_xmat.shape != (3, 3) or not np.isfinite(camera_xmat).all():
        raise ValueError("observation camera_xmat must have shape (3, 3)")
