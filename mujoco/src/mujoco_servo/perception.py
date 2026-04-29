from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

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

    def detect(self, observation: CameraObservation | None, truth_position: np.ndarray, target: TargetSpec, prompt: str) -> Detection:
        ...


def _bbox_mask(frame_shape: tuple[int, int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    mask = np.zeros((h, w), dtype=np.uint8)
    left = max(0, min(w - 1, int(np.floor(x1))))
    top = max(0, min(h - 1, int(np.floor(y1))))
    right = max(left + 1, min(w, int(np.ceil(x2))))
    bottom = max(top + 1, min(h, int(np.ceil(y2))))
    mask[top:bottom, left:right] = 255
    return mask


def _estimate_world_position(observation: CameraObservation, bbox_xyxy: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray]:
    if mask is None:
        mask = _bbox_mask(observation.frame_bgr.shape, bbox_xyxy)
    valid = mask > 0
    depth = np.asarray(observation.depth_m, dtype=float)
    valid &= np.isfinite(depth)
    valid &= depth > 0.0
    if not np.any(valid):
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=float).reshape(4)
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        sample = depth[max(0, min(depth.shape[0] - 1, int(v))), max(0, min(depth.shape[1] - 1, int(u)))]
        if not np.isfinite(sample) or sample <= 0.0:
            return None, mask
        z = float(sample)
    else:
        ys, xs = np.nonzero(valid)
        u = float(np.mean(xs))
        v = float(np.mean(ys))
        z = float(np.median(depth[valid]))
    intr = observation.intrinsics
    # MuJoCo/OpenGL camera coordinates: +x right, +y up, camera looks along -z.
    point_cam = np.array([(u - intr.cx) * z / intr.fx, -(v - intr.cy) * z / intr.fy, -z], dtype=float)
    return observation.camera_position + observation.camera_xmat @ point_cam, mask


class OraclePerception:
    name = "oracle"

    def detect(self, observation: CameraObservation | None, truth_position: np.ndarray, target: TargetSpec, prompt: str) -> Detection:
        return Detection(success=True, backend=self.name, target_position=np.asarray(truth_position, dtype=float).reshape(3).copy(), score=1.0)


class ColorSegmentationPerception:
    name = "color"

    def detect(self, observation: CameraObservation | None, truth_position: np.ndarray, target: TargetSpec, prompt: str) -> Detection:
        if observation is None:
            return Detection(False, self.name, None)
        frame_bgr = observation.frame_bgr
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
        bbox = np.array([x, y, x + w, y + h], dtype=float)
        moments = cv2.moments(contour)
        cx = x + 0.5 * w if abs(moments["m00"]) < 1e-9 else moments["m10"] / moments["m00"]
        cy = y + 0.5 * h if abs(moments["m00"]) < 1e-9 else moments["m01"] / moments["m00"]
        position, mask = _estimate_world_position(observation, bbox, mask)
        return Detection(
            success=position is not None,
            backend=self.name,
            target_position=position,
            score=min(1.0, area / 2500.0),
            bbox_xyxy=bbox,
            centroid_px=np.array([cx, cy], dtype=float),
            mask=mask,
        )


class SemanticPerception:
    name = "semantic"

    def __init__(self) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor, SamModel, SamProcessor
        except Exception as exc:
            raise RuntimeError(
                "semantic perception requires optional dependencies. Install with "
                "`conda run -n visual_servo python -m pip install -e 'mujoco[semantic]'`."
            ) from exc
        self._torch = torch
        self._image_cls = Image
        self._gdino_processor = GroundingDinoProcessor.from_pretrained(os.getenv("MUJOCO_SERVO_GDINO_MODEL", "IDEA-Research/grounding-dino-tiny"))
        self._gdino_model = GroundingDinoForObjectDetection.from_pretrained(os.getenv("MUJOCO_SERVO_GDINO_MODEL", "IDEA-Research/grounding-dino-tiny"))
        self._sam_processor = SamProcessor.from_pretrained(os.getenv("MUJOCO_SERVO_SAM_MODEL", "facebook/sam-vit-base"))
        self._sam_model = SamModel.from_pretrained(os.getenv("MUJOCO_SERVO_SAM_MODEL", "facebook/sam-vit-base"))
        device = os.getenv("MUJOCO_SERVO_DEVICE", "auto")
        if device == "auto":
            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self._device = torch.device(device)
        self._gdino_model.to(self._device).eval()
        self._sam_model.to(self._device).eval()
        self._box_threshold = float(os.getenv("MUJOCO_SERVO_GDINO_BOX_THRESHOLD", "0.25"))
        self._text_threshold = float(os.getenv("MUJOCO_SERVO_GDINO_TEXT_THRESHOLD", "0.25"))
        self._initialized = False
        self._last_bbox: np.ndarray | None = None
        self._last_mask: np.ndarray | None = None
        self._last_detection: Detection | None = None
        self._hsv_center: np.ndarray | None = None

    def detect(self, observation: CameraObservation | None, truth_position: np.ndarray, target: TargetSpec, prompt: str) -> Detection:
        if observation is None:
            return Detection(False, self.name, None)
        if self._initialized:
            tracked = self._track_from_last_mask(observation)
            if tracked.success:
                self._last_detection = tracked
                return tracked
        image = self._image_cls.fromarray(observation.frame_bgr[:, :, ::-1])
        text = prompt.strip().lower()
        if not text.endswith("."):
            text = f"{text}."
        with self._torch.no_grad():
            inputs = self._to_device(self._gdino_processor(images=image, text=text, return_tensors="pt"))
            outputs = self._gdino_model(**inputs)
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self._box_threshold,
                text_threshold=self._text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        if len(boxes) == 0:
            return Detection(False, self.name, None)
        best = int(self._torch.argmax(scores).item())
        bbox = boxes[best].detach().cpu().numpy().astype(float)
        score = float(scores[best].detach().cpu().item())
        mask = self._sam_mask(image, bbox)
        position, mask = _estimate_world_position(observation, bbox, mask)
        x1, y1, x2, y2 = bbox
        detection = Detection(
            success=position is not None,
            backend=self.name,
            target_position=position,
            score=score,
            bbox_xyxy=bbox,
            centroid_px=np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=float),
            mask=mask,
        )
        if detection.success:
            self._initialized = True
            self._last_bbox = bbox.copy()
            self._last_mask = mask.copy()
            self._last_detection = detection
            self._hsv_center = self._mask_hsv_center(observation.frame_bgr, mask)
        return detection

    def _sam_mask(self, image, bbox: np.ndarray) -> np.ndarray:
        box = np.asarray(bbox, dtype=float).reshape(4).tolist()
        with self._torch.no_grad():
            inputs = self._to_device(self._sam_processor(image, input_boxes=[[[box]]], return_tensors="pt"))
            outputs = self._sam_model(**inputs)
            masks = self._sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
            )[0]
        mask = masks[0, 0].numpy().astype(np.uint8) * 255
        return mask

    def _track_from_last_mask(self, observation: CameraObservation) -> Detection:
        if self._last_bbox is None:
            return Detection(False, self.name, None)
        mask = self._local_color_mask(observation.frame_bgr)
        if mask is None:
            mask = self._bbox_roi_mask(observation.frame_bgr.shape, self._last_bbox)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(False, self.name, None, mask=mask)
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 16.0:
            return Detection(False, self.name, None, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = np.array([x, y, x + w, y + h], dtype=float)
        position, mask = _estimate_world_position(observation, bbox, mask)
        moments = cv2.moments(contour)
        cx = x + 0.5 * w if abs(moments["m00"]) < 1e-9 else moments["m10"] / moments["m00"]
        cy = y + 0.5 * h if abs(moments["m00"]) < 1e-9 else moments["m01"] / moments["m00"]
        detection = Detection(
            success=position is not None,
            backend=f"{self.name}-track",
            target_position=position,
            score=float(self._last_detection.score if self._last_detection is not None else 0.5),
            bbox_xyxy=bbox,
            centroid_px=np.array([cx, cy], dtype=float),
            mask=mask,
        )
        if detection.success:
            self._last_bbox = bbox.copy()
            self._last_mask = mask.copy()
            self._last_detection = detection
        return detection

    def _local_color_mask(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self._last_bbox is None or self._hsv_center is None:
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        x1, y1, x2, y2 = self._expanded_bbox(frame_bgr.shape, self._last_bbox, 1.2)
        hue = int(self._hsv_center[0])
        sat = int(self._hsv_center[1])
        val = int(self._hsv_center[2])
        lower = np.array([max(0, hue - 14), max(35, sat - 70), max(25, val - 90)], dtype=np.uint8)
        upper = np.array([min(179, hue + 14), 255, 255], dtype=np.uint8)
        roi = cv2.inRange(hsv[y1:y2, x1:x2], lower, upper)
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = roi
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _bbox_roi_mask(self, frame_shape: tuple[int, int, int], bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self._expanded_bbox(frame_shape, bbox, 0.35)
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def _expanded_bbox(frame_shape: tuple[int, int, int], bbox: np.ndarray, scale: float) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = np.asarray(bbox, dtype=float).reshape(4)
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
        return np.median(hsv[valid], axis=0)

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


def build_perception(name: str) -> PerceptionBackend:
    normalized = name.strip().lower()
    if normalized in {"oracle", "sim", "simulation"}:
        return OraclePerception()
    if normalized in {"color", "segmentation", "mask"}:
        return ColorSegmentationPerception()
    if normalized in {"semantic", "grounding-dino", "grounded-sam"}:
        return SemanticPerception()
    raise ValueError(f"unknown detector '{name}'")
