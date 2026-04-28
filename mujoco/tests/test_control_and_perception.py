from __future__ import annotations

import cv2
import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import ControllerConfig
from mujoco_servo.control import desired_ee_position
from mujoco_servo.perception import ColorSegmentationPerception, OraclePerception
from mujoco_servo.targets import resolve_target


def test_task_goal_modes() -> None:
    target = np.array([0.5, 0.1, 0.35], dtype=float)
    ee = np.array([0.2, -0.2, 0.5], dtype=float)
    cfg = ControllerConfig(standoff_m=0.2, align_offset_m=0.03)
    assert np.allclose(desired_ee_position("contact", target, ee, cfg), target)
    standoff = desired_ee_position("standoff", target, ee, cfg)
    assert np.isclose(np.linalg.norm(standoff - target), 0.2)
    assert np.allclose(desired_ee_position("align-x", target, ee, cfg), [0.53, -0.2, 0.5])
    assert np.allclose(desired_ee_position("align-y", target, ee, cfg), [0.2, 0.13, 0.5])
    assert np.allclose(desired_ee_position("align-z", target, ee, cfg), [0.2, -0.2, 0.38])


def test_oracle_perception_returns_world_target() -> None:
    target = resolve_target("cup")
    position = np.array([0.42, -0.1, 0.33], dtype=float)
    detection = OraclePerception().detect(None, position, target)
    assert detection.success
    assert detection.score == 1.0
    assert np.allclose(detection.target_position, position)


def test_color_segmentation_detects_render_like_blob() -> None:
    target = resolve_target("box")
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    cv2.rectangle(image, (80, 50), (145, 125), bgr, -1)
    detection = ColorSegmentationPerception().detect(image, np.array([0.4, 0.0, 0.3]), target)
    assert detection.success
    assert detection.bbox_xyxy is not None
    assert detection.centroid_px is not None
    assert 105 < detection.centroid_px[0] < 120
    assert 80 < detection.centroid_px[1] < 100
