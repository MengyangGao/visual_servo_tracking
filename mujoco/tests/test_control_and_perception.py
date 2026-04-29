from __future__ import annotations

import cv2
import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import ControllerConfig
from mujoco_servo.control import desired_ee_position
from mujoco_servo.perception import CameraIntrinsics, CameraObservation, ColorSegmentationPerception, OraclePerception, SemanticPerception
from mujoco_servo.targets import resolve_target


def test_task_goal_modes() -> None:
    target = np.array([0.5, 0.1, 0.35], dtype=float)
    ee = np.array([0.2, -0.2, 0.5], dtype=float)
    cfg = ControllerConfig(standoff_m=0.2, align_offset_m=0.03)
    assert np.allclose(desired_ee_position("contact", target, ee, cfg), target)
    standoff = desired_ee_position("standoff", target, ee, cfg)
    assert np.isclose(np.linalg.norm(standoff - target), 0.2)
    front = desired_ee_position("front-standoff", target, ee, cfg)
    assert np.isclose(np.linalg.norm((front - target)[:2]), 0.2)
    assert np.isclose(front[2], target[2])
    assert np.allclose(desired_ee_position("align-x", target, ee, cfg), [0.53, -0.2, 0.5])
    assert np.allclose(desired_ee_position("align-y", target, ee, cfg), [0.2, 0.13, 0.5])
    assert np.allclose(desired_ee_position("align-z", target, ee, cfg), [0.2, -0.2, 0.38])


def test_oracle_perception_returns_world_target() -> None:
    target = resolve_target("cup")
    position = np.array([0.42, -0.1, 0.33], dtype=float)
    detection = OraclePerception().detect(None, position, target, "cup")
    assert detection.success
    assert detection.score == 1.0
    assert np.allclose(detection.target_position, position)


def test_color_segmentation_detects_render_like_blob() -> None:
    target = resolve_target("box")
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    depth = np.ones((180, 240), dtype=np.float32)
    observation = CameraObservation(
        frame_bgr=image,
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=180.0, fy=180.0, cx=120.0, cy=90.0, width=240, height=180),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
    )
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    cv2.rectangle(image, (80, 50), (145, 125), bgr, -1)
    detection = ColorSegmentationPerception().detect(observation, np.array([0.4, 0.0, 0.3]), target, "box")
    assert detection.success
    assert detection.bbox_xyxy is not None
    assert detection.centroid_px is not None
    assert 105 < detection.centroid_px[0] < 120
    assert 80 < detection.centroid_px[1] < 100


def test_semantic_detect_reuses_initialized_local_tracker_without_models() -> None:
    target = resolve_target("apple")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 60), 18, (0, 0, 230), -1)
    observation = CameraObservation(
        frame_bgr=image,
        depth_m=np.ones((120, 160), dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=120.0, fy=120.0, cx=80.0, cy=60.0, width=160, height=120),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
    )
    semantic = object.__new__(SemanticPerception)
    semantic._initialized = True
    semantic._last_bbox = np.array([60.0, 40.0, 100.0, 80.0])
    semantic._last_mask = np.zeros((120, 160), dtype=np.uint8)
    semantic._last_mask[42:78, 62:98] = 255
    semantic._last_detection = None
    semantic._hsv_center = np.array([0, 255, 230], dtype=float)
    detection = semantic.detect(observation, np.zeros(3, dtype=float), target, "apple")
    assert detection.success
    assert detection.backend == "semantic-track"
    assert detection.bbox_xyxy is not None
    assert detection.mask is not None
