from __future__ import annotations

import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .config import canonical_prompt, lookup_target_prototype, project_root
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


@dataclass(slots=True)
class GroundedSam2Config:
    grounding_model_id: str = os.getenv("MUJOCO_SERVO_GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
    grounding_box_threshold: float = float(os.getenv("MUJOCO_SERVO_GDINO_BOX_THRESHOLD", "0.35"))
    grounding_text_threshold: float = float(os.getenv("MUJOCO_SERVO_GDINO_TEXT_THRESHOLD", "0.25"))
    sam2_repo_id: str = os.getenv("MUJOCO_SERVO_SAM2_REPO", "facebook/sam2.1-hiera-base-plus")
    sam2_model_cfg_name: str = os.getenv("MUJOCO_SERVO_SAM2_MODEL_CFG_NAME", "configs/sam2.1/sam2.1_hiera_b+.yaml")
    sam2_checkpoint_name: str = os.getenv("MUJOCO_SERVO_SAM2_CHECKPOINT_NAME", "sam2.1_hiera_base_plus.pt")
    sam2_checkpoint: str = os.getenv("MUJOCO_SERVO_SAM2_CHECKPOINT", "")
    device: str = os.getenv("MUJOCO_SERVO_DEVICE", "auto")
    allow_bbox_fallback: bool = os.getenv("MUJOCO_SERVO_ALLOW_BBOX_FALLBACK", "1") != "0"

    @classmethod
    def from_preset(cls, preset: str) -> "GroundedSam2Config":
        normalized = canonical_prompt(preset)
        if normalized in {"lite", "fast", "tiny"}:
            return cls(
                grounding_model_id="IDEA-Research/grounding-dino-tiny",
                sam2_repo_id="facebook/sam2.1-hiera-tiny",
                sam2_model_cfg_name="configs/sam2.1/sam2.1_hiera_t.yaml",
                sam2_checkpoint_name="sam2.1_hiera_tiny.pt",
            )
        if normalized in {"small"}:
            return cls(
                grounding_model_id="IDEA-Research/grounding-dino-tiny",
                sam2_repo_id="facebook/sam2.1-hiera-small",
                sam2_model_cfg_name="configs/sam2.1/sam2.1_hiera_s.yaml",
                sam2_checkpoint_name="sam2.1_hiera_small.pt",
            )
        if normalized in {"base", "default", "balanced", "default-base"}:
            return cls()
        return cls()


def _ensure_path_on_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _default_reference_repo() -> Path | None:
    env_root = os.getenv("MUJOCO_SERVO_SAM2_REPO", "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "reference" / "Grounded-SAM-2")
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _normalize_grounding_prompt(prompt: str) -> str:
    text = canonical_prompt(prompt)
    return text if text.endswith(".") else f"{text}."


def _mask_from_bbox(frame_shape: tuple[int, int, int] | tuple[int, int], bbox_xyxy: np.ndarray) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _best_detection_index(scores: np.ndarray) -> int:
    if scores.size == 0:
        return -1
    return int(np.argmax(scores))


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

    def track(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        previous_detection: Detection | None = None,
    ) -> Detection:
        return self.detect(frame_bgr, prompt, intrinsics, camera_pose)


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

    def track(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        previous_detection: Detection | None = None,
    ) -> Detection:
        return self.detect(frame_bgr, prompt, intrinsics, camera_pose)


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

    @staticmethod
    def _roi_from_previous(frame_shape: tuple[int, int, int], previous_detection: Detection) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = previous_detection.bbox_xyxy.astype(float)
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        pad_x = int(max(24.0, bw * 0.75))
        pad_y = int(max(24.0, bh * 0.75))
        left = max(0, int(x1) - pad_x)
        top = max(0, int(y1) - pad_y)
        right = min(w, int(x2) + pad_x)
        bottom = min(h, int(y2) + pad_y)
        return left, top, right, bottom

    @staticmethod
    def _shift_detection(
        detection: Detection,
        offset_x: int,
        offset_y: int,
        frame_shape: tuple[int, int, int] | tuple[int, int],
    ) -> Detection:
        shifted = Detection(
            success=detection.success,
            prompt=detection.prompt,
            label=detection.label,
            score=detection.score,
            bbox_xyxy=detection.bbox_xyxy.copy(),
            centroid_px=detection.centroid_px.copy(),
            mask=detection.mask,
            mask_area_px=detection.mask_area_px,
            estimated_distance_m=detection.estimated_distance_m,
            backend=detection.backend,
            track_id=detection.track_id,
            target_position_world=detection.target_position_world,
        )
        shifted.bbox_xyxy[[0, 2]] += offset_x
        shifted.bbox_xyxy[[1, 3]] += offset_y
        shifted.centroid_px[[0]] += offset_x
        shifted.centroid_px[[1]] += offset_y
        if shifted.mask is not None:
            height, width = frame_shape[:2]
            full_mask = np.zeros((height, width), dtype=detection.mask.dtype)
            mask_h, mask_w = detection.mask.shape[:2]
            x_end = min(width, offset_x + mask_w)
            y_end = min(height, offset_y + mask_h)
            full_mask[offset_y:y_end, offset_x:x_end] = detection.mask[: y_end - offset_y, : x_end - offset_x]
            shifted.mask = full_mask
        return shifted

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

    def track(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        previous_detection: Detection | None = None,
    ) -> Detection:
        if previous_detection is not None and previous_detection.success:
            left, top, right, bottom = self._roi_from_previous(frame_bgr.shape, previous_detection)
            roi = frame_bgr[top:bottom, left:right]
            if roi.size > 0:
                local_detection = self.detect(roi, prompt, intrinsics, camera_pose)
                if local_detection.success:
                    return self._shift_detection(local_detection, left, top, frame_bgr.shape)
        return self.detect(frame_bgr, prompt, intrinsics, camera_pose)


class GroundedSam2Backend(PerceptionBackend):
    name = "grounded-sam2"

    @classmethod
    def available(cls) -> bool:
        try:
            import importlib

            importlib.import_module("transformers")
            return True
        except Exception:
            return False

    def __init__(self, config: GroundedSam2Config | None = None):
        if not self.available():
            raise RuntimeError(
                "Grounded SAM 2 backend requires the 'transformers' package. "
                "Install the open-vocab extras or choose the heuristic backend."
            )
        self.config = config or GroundedSam2Config()
        self.device = self._resolve_device(self.config.device)
        self._processor = None
        self._grounding_model = None
        self._sam2_predictor = None
        self._sam2_ready = False
        self._sam2_cfg_path: str | None = None
        self._sam2_checkpoint_path: str | None = None
        self._load_models()

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = canonical_prompt(device)
        if requested in {"auto", ""}:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return requested

    def _load_models(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        reference_repo = _default_reference_repo()
        if reference_repo is not None:
            _ensure_path_on_sys_path(reference_repo)

        cache_dir = project_root() / "outputs" / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        self._processor = AutoProcessor.from_pretrained(self.config.grounding_model_id, cache_dir=str(cache_dir))
        self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.grounding_model_id,
            cache_dir=str(cache_dir),
        ).to(self.device)

        sam2_checkpoint: Path | None = None
        if self.config.sam2_checkpoint:
            candidate = Path(self.config.sam2_checkpoint).expanduser()
            if candidate.exists():
                sam2_checkpoint = candidate
        if sam2_checkpoint is None:
            sam2_checkpoint = Path(
                hf_hub_download(
                    repo_id=self.config.sam2_repo_id,
                    filename=self.config.sam2_checkpoint_name,
                    cache_dir=str(cache_dir),
                )
            )
        self._sam2_checkpoint_path = str(sam2_checkpoint)
        self._sam2_cfg_path = self.config.sam2_model_cfg_name

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(self.config.sam2_model_cfg_name, str(sam2_checkpoint), device=self.device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            self._sam2_ready = True
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"SAM2 could not be initialized, falling back to box masks: {exc}", RuntimeWarning)
            self._sam2_predictor = None
            self._sam2_ready = False

    def _predict_boxes(self, frame_bgr: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self._processor is None or self._grounding_model is None:
            raise RuntimeError("GroundingDINO model is not initialized")
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        prompt_text = _normalize_grounding_prompt(prompt)
        inputs = self._processor(images=image_pil, text=prompt_text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._grounding_model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.grounding_box_threshold,
            text_threshold=self.config.grounding_text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        if not results:
            return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=float), []
        boxes = results[0]["boxes"].detach().cpu().numpy()
        scores = results[0]["scores"].detach().cpu().numpy()
        labels = list(results[0]["labels"])
        return boxes, scores, labels

    def _segment_boxes(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        if boxes_xyxy.size == 0:
            return np.zeros((0, frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
        if self._sam2_ready and self._sam2_predictor is not None:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._sam2_predictor.set_image(image_rgb)
            masks, _, _ = self._sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.asarray(boxes_xyxy, dtype=float),
                multimask_output=False,
            )
            if masks.ndim == 2:
                masks = masks[None, ...]
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            return masks.astype(np.uint8)
        return np.stack([_mask_from_bbox(frame_bgr.shape, box) for box in boxes_xyxy], axis=0)

    def _detect_best(self, frame_bgr: np.ndarray, prompt: str, intrinsics: CameraIntrinsics) -> Detection:
        prototype = lookup_target_prototype(prompt)
        boxes, scores, labels = self._predict_boxes(frame_bgr, prompt)
        if boxes.size == 0:
            return Detection(
                success=False,
                prompt=prompt,
                label=prototype.name,
                score=0.0,
                bbox_xyxy=np.zeros(4, dtype=float),
                centroid_px=np.zeros(2, dtype=float),
                mask=None,
                mask_area_px=0,
                estimated_distance_m=None,
                backend=self.name,
            )
        idx = _best_detection_index(scores)
        if idx < 0:
            return Detection(
                success=False,
                prompt=prompt,
                label=prototype.name,
                score=0.0,
                bbox_xyxy=np.zeros(4, dtype=float),
                centroid_px=np.zeros(2, dtype=float),
                mask=None,
                mask_area_px=0,
                estimated_distance_m=None,
                backend=self.name,
            )
        box = boxes[idx]
        mask = self._segment_boxes(frame_bgr, boxes[[idx]])[0]
        centroid = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=float)
        distance = estimate_distance_from_bbox(prototype, box, intrinsics)
        return Detection(
            success=True,
            prompt=prompt,
            label=str(labels[idx]) if idx < len(labels) else prototype.name,
            score=float(scores[idx]),
            bbox_xyxy=np.asarray(box, dtype=float),
            centroid_px=centroid,
            mask=mask,
            mask_area_px=int(np.count_nonzero(mask)),
            estimated_distance_m=distance,
            backend=self.name,
        )

    def track(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        previous_detection: Detection | None = None,
    ) -> Detection:
        if previous_detection is not None and previous_detection.success:
            left, top, right, bottom = PromptGuidedVisionBackend._roi_from_previous(frame_bgr.shape, previous_detection)
            roi = frame_bgr[top:bottom, left:right]
            if roi.size > 0:
                local = self._detect_best(roi, prompt, intrinsics)
                if local.success:
                    return PromptGuidedVisionBackend._shift_detection(local, left, top, frame_bgr.shape)
        return self._detect_best(frame_bgr, prompt, intrinsics)

    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
    ) -> Detection:
        return self._detect_best(frame_bgr, prompt, intrinsics)


def build_backend(
    name: str,
    prompt: str,
    target_pose_provider: Callable[[str], np.ndarray],
    config: GroundedSam2Config | None = None,
) -> PerceptionBackend:
    normalized = canonical_prompt(name)
    if normalized in {"oracle", "simulation", "sim"}:
        return OracleBackend(target_pose_provider)
    if normalized in {"heuristic", "fallback", "opencv"}:
        return PromptGuidedVisionBackend()
    if normalized in {"grounded-sam2", "grounded_sam2", "open-vocab", "open_vocab"}:
        try:
            return GroundedSam2Backend(config=config)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Falling back to heuristic backend: {exc}", RuntimeWarning)
            return PromptGuidedVisionBackend()
    if normalized == "auto":
        try:
            return GroundedSam2Backend(config=config)
        except Exception:
            return PromptGuidedVisionBackend()
    return PromptGuidedVisionBackend()


@dataclass(slots=True)
class PerceptionSession:
    backend: PerceptionBackend
    prompt: str
    previous_detection: Detection | None = None

    def update(
        self,
        frame_bgr: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
    ) -> Detection:
        if self.previous_detection is None:
            detection = self.backend.detect(frame_bgr, self.prompt, intrinsics, camera_pose)
        else:
            detection = self.backend.track(frame_bgr, self.prompt, intrinsics, camera_pose, self.previous_detection)
        self.previous_detection = detection if detection.success else self.previous_detection
        return detection
