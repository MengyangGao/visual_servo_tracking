from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .config import TARGET_LIBRARY, canonical_prompt, default_camera_intrinsics, lookup_target_prototype
from .geometry import normalize
from .types import CameraIntrinsics, CameraPose, Detection, TargetPrototype


def estimate_distance_from_bbox(
    prototype: TargetPrototype,
    bbox_xyxy: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> float | None:
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    real_width = max(float(prototype.size_m[0]), float(prototype.size_m[1]), 1e-3)
    pixel_extent = max(w, h)
    return float(intrinsics.fx * real_width / pixel_extent)


class PerceptionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
    ) -> Detection:
        raise NotImplementedError


class OracleBackend(PerceptionBackend):
    name = "oracle"

    def __init__(self, target_pose_provider: Callable[[str], np.ndarray]):
        self._target_pose_provider = target_pose_provider

    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
    ) -> Detection:
        prototype = lookup_target_prototype(prompt)
        target_position = self._target_pose_provider(prompt)
        point_cam = camera_pose.rotation_world_from_cam.T @ (target_position - camera_pose.translation_m)
        z = float(max(point_cam[2], 1e-3))
        u = intrinsics.fx * point_cam[0] / z + intrinsics.cx
        v = intrinsics.fy * point_cam[1] / z + intrinsics.cy
        nominal_half_extent = max(prototype.size_m[0], prototype.size_m[1]) * 0.5
        pixel_half = max(12.0, intrinsics.fx * nominal_half_extent / z)
        bbox = np.array([u - pixel_half, v - pixel_half, u + pixel_half, v + pixel_half], dtype=float)
        return Detection(
            success=True,
            prompt=prompt,
            label=prototype.name,
            score=0.99,
            bbox_xyxy=bbox,
            centroid_px=np.array([u, v], dtype=float),
            mask=None,
            mask_area_px=int((2.0 * pixel_half) ** 2),
            estimated_distance_m=z,
            backend=self.name,
            target_position_world=target_position.copy(),
        )


class PromptGuidedVisionBackend(PerceptionBackend):
    name = "heuristic"

    _COLOR_RANGES = {
        "red": [((0, 70, 40), (10, 255, 255)), ((170, 70, 40), (180, 255, 255))],
        "green": [((35, 50, 40), (85, 255, 255))],
        "blue": [((90, 50, 40), (135, 255, 255))],
        "yellow": [((20, 50, 60), (35, 255, 255))],
        "orange": [((10, 60, 50), (22, 255, 255))],
        "purple": [((135, 40, 40), (165, 255, 255))],
        "black": [((0, 0, 0), (180, 255, 80))],
        "white": [((0, 0, 180), (180, 40, 255))],
    }

    def _infer_color_mask(self, frame_bgr: np.ndarray, prompt: str) -> np.ndarray | None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        prompt_text = canonical_prompt(prompt)
        for color_name, ranges in self._COLOR_RANGES.items():
            if color_name in prompt_text:
                mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                for lower, upper in ranges:
                    mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                return mask
        return None

    def _saliency_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
    ) -> Detection:
        prototype = lookup_target_prototype(prompt)
        mask = self._infer_color_mask(frame_bgr, prompt)
        if mask is None or int(mask.sum()) == 0:
            mask = self._saliency_mask(frame_bgr)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(
                success=False,
                prompt=prompt,
                label=prototype.name,
                score=0.0,
                bbox_xyxy=np.zeros(4, dtype=float),
                centroid_px=np.zeros(2, dtype=float),
                mask=mask,
                mask_area_px=0,
                estimated_distance_m=None,
                backend=self.name,
            )

        h, w = frame_bgr.shape[:2]
        best = None
        best_score = -1.0
        image_center = np.array([w / 2.0, h / 2.0], dtype=float)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200.0:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            centroid = np.array([x + bw / 2.0, y + bh / 2.0], dtype=float)
            center_score = 1.0 / (1.0 + np.linalg.norm((centroid - image_center) / np.array([w, h], dtype=float)))
            rectangularity = area / max(float(bw * bh), 1.0)
            score = area * center_score * (0.5 + 0.5 * rectangularity)
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh, centroid, area)
        if best is None:
            return Detection(
                success=False,
                prompt=prompt,
                label=prototype.name,
                score=0.0,
                bbox_xyxy=np.zeros(4, dtype=float),
                centroid_px=np.zeros(2, dtype=float),
                mask=mask,
                mask_area_px=0,
                estimated_distance_m=None,
                backend=self.name,
            )

        x, y, bw, bh, centroid, area = best
        bbox = np.array([x, y, x + bw, y + bh], dtype=float)
        distance = estimate_distance_from_bbox(prototype, bbox, intrinsics)
        return Detection(
            success=True,
            prompt=prompt,
            label=prototype.name,
            score=min(1.0, best_score / max(float(w * h), 1.0)),
            bbox_xyxy=bbox,
            centroid_px=centroid,
            mask=mask,
            mask_area_px=int(area),
            estimated_distance_m=distance,
            backend=self.name,
        )


class GroundedSam2Backend(PerceptionBackend):
    name = "grounded-sam2"

    @classmethod
    def available(cls) -> bool:
        try:
            import importlib

            importlib.import_module("transformers")
            importlib.import_module("sam2")
            return True
        except Exception:
            return False

    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
    ) -> Detection:
        raise RuntimeError(
            "Grounded SAM 2 backend is scaffolded but not wired to weights in this workspace. "
            "Use heuristic or oracle backend, or install/configure the model path."
        )


def build_backend(name: str, prompt: str, target_pose_provider: Callable[[str], np.ndarray]) -> PerceptionBackend:
    normalized = canonical_prompt(name)
    if normalized in {"oracle", "simulation", "sim"}:
        return OracleBackend(target_pose_provider)
    if normalized in {"heuristic", "fallback", "opencv"}:
        return PromptGuidedVisionBackend()
    if normalized in {"grounded-sam2", "grounded_sam2", "open-vocab", "open_vocab"}:
        if GroundedSam2Backend.available():
            return GroundedSam2Backend()
        return PromptGuidedVisionBackend()
    if normalized == "auto":
        if GroundedSam2Backend.available():
            return GroundedSam2Backend()
        return PromptGuidedVisionBackend()
    return PromptGuidedVisionBackend()

