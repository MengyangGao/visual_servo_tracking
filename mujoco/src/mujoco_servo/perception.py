from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .targets import TargetSpec


@dataclass(slots=True)
class Detection:
    success: bool
    backend: str
    target_position: np.ndarray | None
    score: float = 0.0
    bbox_xyxy: np.ndarray | None = None
    centroid_px: np.ndarray | None = None
    mask: np.ndarray | None = None


class PerceptionBackend(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray | None, target_position: np.ndarray, target: TargetSpec) -> Detection:
        ...


class OraclePerception:
    name = "oracle"

    def detect(self, frame_bgr: np.ndarray | None, target_position: np.ndarray, target: TargetSpec) -> Detection:
        return Detection(success=True, backend=self.name, target_position=np.asarray(target_position, dtype=float).reshape(3).copy(), score=1.0)


class ColorSegmentationPerception:
    name = "color"

    def detect(self, frame_bgr: np.ndarray | None, target_position: np.ndarray, target: TargetSpec) -> Detection:
        if frame_bgr is None:
            return Detection(False, self.name, None)
        rgba = np.array(target.rgba[:3], dtype=float)
        target_bgr = np.array([rgba[2], rgba[1], rgba[0]], dtype=float) * 255.0
        hsv_color = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue = int(hsv_color[0])
        lower = np.array([max(0, hue - 12), 45, 35], dtype=np.uint8)
        upper = np.array([min(179, hue + 12), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(frame_hsv, lower, upper)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(False, self.name, None, mask=mask)
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 20.0:
            return Detection(False, self.name, None, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        cx = x + 0.5 * w if abs(moments["m00"]) < 1e-9 else moments["m10"] / moments["m00"]
        cy = y + 0.5 * h if abs(moments["m00"]) < 1e-9 else moments["m01"] / moments["m00"]
        return Detection(
            success=True,
            backend=self.name,
            target_position=np.asarray(target_position, dtype=float).reshape(3).copy(),
            score=min(1.0, area / 2500.0),
            bbox_xyxy=np.array([x, y, x + w, y + h], dtype=float),
            centroid_px=np.array([cx, cy], dtype=float),
            mask=mask,
        )


class SemanticPerception:
    name = "semantic"

    def __init__(self) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "semantic perception is optional. Install with `pip install -e .[semantic]` "
                "and configure a Grounding DINO / SAM2 implementation before using --detector semantic."
            ) from exc

    def detect(self, frame_bgr: np.ndarray | None, target_position: np.ndarray, target: TargetSpec) -> Detection:
        raise NotImplementedError("semantic detector adapter is intentionally pluggable; oracle/color are ready now")


def build_perception(name: str) -> PerceptionBackend:
    normalized = name.strip().lower()
    if normalized in {"oracle", "sim", "simulation"}:
        return OraclePerception()
    if normalized in {"color", "segmentation", "mask"}:
        return ColorSegmentationPerception()
    if normalized in {"semantic", "grounding-dino", "grounded-sam2", "sam2"}:
        return SemanticPerception()
    raise ValueError(f"unknown detector '{name}'")
