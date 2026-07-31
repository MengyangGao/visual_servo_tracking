from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mujoco_servo.config import ControllerConfig, TargetSpec
from mujoco_servo.control import ResolvedRateController, desired_ee_position
from mujoco_servo.math_utils import rotation_error_vector, vector_alignment_error
from mujoco_servo.perception import (
    CameraIntrinsics,
    CameraObservation,
    ColorSegmentationPerception,
    Detection,
    OraclePerception,
    SemanticPerception,
    _estimate_world_anchor,
    _estimate_world_position,
    _resolve_torch_device_name,
)
from mujoco_servo.scene import build_scene
from mujoco_servo.targets import resolve_target


def test_task_goal_modes() -> None:
    target = np.array([0.5, 0.1, 0.35], dtype=float)
    ee = np.array([0.2, -0.2, 0.5], dtype=float)
    cfg = ControllerConfig(standoff_m=0.2, align_offset_m=0.03)
    assert np.allclose(desired_ee_position("contact", target, ee, cfg), target)
    assert np.allclose(desired_ee_position("touch", target, ee, cfg), target)
    assert np.allclose(desired_ee_position("grasp", target, ee, cfg), target)
    assert np.allclose(desired_ee_position("pick-place", target, ee, cfg), target)
    standoff = desired_ee_position("standoff", target, ee, cfg)
    assert np.isclose(np.linalg.norm(standoff - target), 0.2)
    fixed_standoff = desired_ee_position(
        "standoff",
        target,
        ee,
        cfg,
        standoff_direction_world=np.array([0.0, -1.0, 0.0]),
    )
    assert np.allclose(fixed_standoff, target + np.array([0.0, -0.2, 0.0]))
    front = desired_ee_position("front-standoff", target, ee, cfg)
    assert np.isclose(np.linalg.norm((front - target)[:2]), 0.2)
    assert np.isclose(front[2], target[2])
    assert np.allclose(
        desired_ee_position("align-x", target, ee, cfg), [0.53, -0.2, 0.5]
    )
    assert np.allclose(
        desired_ee_position("align-y", target, ee, cfg), [0.2, 0.13, 0.5]
    )
    assert np.allclose(
        desired_ee_position("align-z", target, ee, cfg), [0.2, -0.2, 0.38]
    )
    translated_front = desired_ee_position(
        "front-standoff",
        np.array([2.0, 1.0, 0.4]),
        ee,
        cfg,
        front_origin=np.array([1.5, 1.0, 0.0]),
    )
    assert np.allclose(translated_front, [1.8, 1.0, 0.4])


def test_rotation_errors_remain_defined_at_180_degrees() -> None:
    rotations = (
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )
    for rotation in rotations:
        error = rotation_error_vector(rotation, np.eye(3))
        assert np.isclose(np.linalg.norm(error), np.pi)
    alignment = vector_alignment_error(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    )
    assert np.isclose(np.linalg.norm(alignment), np.pi)


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
        intrinsics=CameraIntrinsics(
            fx=180.0, fy=180.0, cx=120.0, cy=90.0, width=240, height=180
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    cv2.rectangle(image, (80, 50), (145, 125), bgr, -1)
    detection = ColorSegmentationPerception().detect(
        observation, np.array([0.4, 0.0, 0.3]), target, "box"
    )
    assert detection.success
    assert detection.bbox_xyxy is not None
    assert detection.centroid_px is not None
    assert detection.anchor_type == "model_center_from_surface"
    assert detection.world_bbox_min is not None
    assert detection.world_bbox_max is not None
    assert 105 < detection.centroid_px[0] < 120
    assert 80 < detection.centroid_px[1] < 100


def test_color_depth_surface_is_corrected_to_known_object_center() -> None:
    target = TargetSpec(
        "red-ball",
        "sphere",
        (0.10, 0.10, 0.10),
        (0.95, 0.10, 0.10, 1.0),
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    cv2.circle(image, (80, 60), 16, bgr, -1)
    cv2.circle(depth, (80, 60), 16, 1.0 - (2.0 / 3.0) * 0.05, -1)
    observation = CameraObservation(
        frame_bgr=image,
        depth_m=depth,
        intrinsics=CameraIntrinsics(120.0, 120.0, 80.0, 60.0, 160, 120),
        camera_position=np.zeros(3),
        camera_xmat=np.eye(3),
        depth_metric=True,
    )

    detection = ColorSegmentationPerception().detect(
        observation, np.zeros(3), target, "red ball"
    )

    assert detection.success
    assert detection.anchor_type == "model_center_from_surface"
    assert np.allclose(detection.target_position, [0.0, 0.0, -1.0], atol=0.004)


def test_color_segmentation_handles_red_hue_wraparound() -> None:
    target = resolve_target("apple")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.ones((120, 160), dtype=np.float32)
    observation = CameraObservation(
        frame_bgr=image,
        depth_m=depth,
        intrinsics=CameraIntrinsics(
            fx=120.0, fy=120.0, cx=80.0, cy=60.0, width=160, height=120
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    cv2.circle(image, (80, 60), 18, (0, 0, 230), -1)
    detection = ColorSegmentationPerception().detect(
        observation, np.array([0.4, 0.0, 0.3]), target, "apple"
    )
    assert detection.success
    assert detection.bbox_xyxy is not None


def test_color_segmentation_uses_only_selected_same_color_contour_for_3d() -> None:
    target = resolve_target("box")
    image = np.zeros((100, 240, 3), dtype=np.uint8)
    depth = np.ones((100, 240), dtype=np.float32)
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    cv2.rectangle(image, (80, 30), (160, 70), bgr, -1)
    cv2.rectangle(image, (200, 40), (225, 60), bgr, -1)
    observation = CameraObservation(
        image,
        depth,
        CameraIntrinsics(100.0, 100.0, 120.0, 50.0, 240, 100),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    detection = ColorSegmentationPerception().detect(
        observation, np.zeros(3), target, "box"
    )
    assert detection.success
    assert detection.mask[50, 210] == 0
    assert detection.mask[50, 120] == 255
    assert np.isclose(detection.target_position[0], 0.0, atol=0.03)


@pytest.mark.parametrize(
    ("target", "bgr"),
    [
        (resolve_target("phone"), (34, 31, 31)),
        (
            TargetSpec("gray", "box", (0.1, 0.1, 0.1), (0.5, 0.5, 0.5, 1.0)),
            (128, 128, 128),
        ),
    ],
)
def test_color_segmentation_supports_neutral_dark_and_gray_targets(target, bgr) -> None:
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    cv2.rectangle(image, (45, 30), (95, 75), bgr, -1)
    observation = CameraObservation(
        image,
        np.ones((100, 140), dtype=np.float32),
        CameraIntrinsics(100.0, 100.0, 70.0, 50.0, 140, 100),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    detection = ColorSegmentationPerception().detect(
        observation, np.zeros(3), target, target.name
    )
    assert detection.success
    assert detection.mask[50, 70] == 255
    assert detection.mask[0, 0] == 0


def test_color_segmentation_rejects_giant_boundary_background() -> None:
    target = resolve_target("box")
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    image = np.full((100, 140, 3), bgr, dtype=np.uint8)
    observation = CameraObservation(
        image,
        np.ones((100, 140), dtype=np.float32),
        CameraIntrinsics(100.0, 100.0, 70.0, 50.0, 140, 100),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    detection = ColorSegmentationPerception().detect(
        observation, np.zeros(3), target, "box"
    )
    assert not detection.success


def test_observation_depth_shape_mismatch_is_rejected() -> None:
    target = resolve_target("box")
    observation = CameraObservation(
        frame_bgr=np.zeros((12, 16, 3), dtype=np.uint8),
        depth_m=np.ones((10, 16), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=12.0, fy=12.0, cx=8.0, cy=6.0, width=16, height=12
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    with pytest.raises(ValueError, match="depth_m shape"):
        ColorSegmentationPerception().detect(observation, np.zeros(3), target, "box")


def test_depth_anchor_rejects_mask_depth_outliers() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    depth[40, 40] = 10.0
    observation = CameraObservation(
        frame_bgr=image,
        depth_m=depth,
        intrinsics=CameraIntrinsics(
            fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    position, _, _, bbox_max, anchor_type = _estimate_world_position(
        observation, np.array([40, 40, 60, 60], dtype=float), mask
    )
    assert anchor_type == "surface_depth_point_cluster"
    assert np.linalg.norm(position - np.array([-0.005, 0.005, -1.0])) < 0.02
    assert bbox_max[2] < -0.9


def test_depth_anchor_intersects_mask_with_clipped_bbox() -> None:
    mask = np.zeros((80, 160), dtype=np.uint8)
    mask[25:55, 20:60] = 255
    mask[25:55, 120:150] = 255
    observation = CameraObservation(
        np.zeros((80, 160, 3), dtype=np.uint8),
        np.ones((80, 160), dtype=np.float32),
        CameraIntrinsics(100.0, 100.0, 80.0, 40.0, 160, 80),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    position, selected, _, _, anchor_type = _estimate_world_position(
        observation, np.array([-10, 20, 70, 60], dtype=float), mask
    )
    assert anchor_type == "surface_depth_point_cluster"
    assert selected[40, 40] == 255
    assert selected[40, 130] == 0
    assert position[0] < -0.30


def test_depth_anchor_selects_near_layer_and_reports_uncertainty() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:60, 20:100] = 255
    depth = np.full((80, 120), np.nan, dtype=np.float32)
    depth[20:60, 20:60] = 1.0
    depth[20:60, 60:100] = 1.35
    observation = CameraObservation(
        np.zeros((80, 120, 3), dtype=np.uint8),
        depth,
        CameraIntrinsics(100.0, 100.0, 60.0, 40.0, 120, 80),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
        wall_time_s=12.5,
        sim_time_s=3.25,
    )
    estimate = _estimate_world_anchor(
        observation, np.array([20, 20, 100, 60], dtype=float), mask
    )
    assert estimate.position is not None
    assert np.isclose(estimate.position[2], -1.0, atol=1e-3)
    assert estimate.position[0] < -0.15
    assert estimate.covariance is not None
    assert np.linalg.eigvalsh(estimate.covariance).min() > 0.0
    assert 0.45 <= estimate.valid_fraction <= 0.55
    assert np.count_nonzero(estimate.mask[:, 60:]) == 0


def test_color_tracker_does_not_switch_to_remote_same_color_distractor() -> None:
    target = resolve_target("box")
    bgr = tuple(int(v * 255) for v in target.rgba[2::-1])
    depth = np.ones((100, 200), dtype=np.float32)
    observation = CameraObservation(
        np.zeros((100, 200, 3), dtype=np.uint8),
        depth,
        CameraIntrinsics(100.0, 100.0, 100.0, 50.0, 200, 100),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    cv2.rectangle(observation.frame_bgr, (65, 35), (95, 65), bgr, -1)
    tracker = ColorSegmentationPerception()
    acquired = tracker.detect(observation, np.zeros(3), target, "box")
    assert acquired.success
    initial_quality = acquired.quality

    observation.frame_bgr.fill(0)
    # At z=1 m and fx=100, this 20-pixel displacement corresponds to 0.2 m;
    # it must be treated as another object, not a continuation of the track.
    cv2.rectangle(observation.frame_bgr, (85, 35), (115, 65), bgr, -1)
    lost = tracker.detect(observation, np.zeros(3), target, "box")
    assert not lost.success
    assert lost.target_position is None
    assert lost.quality < initial_quality

    observation.frame_bgr.fill(0)
    cv2.rectangle(observation.frame_bgr, (67, 35), (97, 65), bgr, -1)
    reacquired = tracker.detect(observation, np.zeros(3), target, "box")
    assert reacquired.success
    assert reacquired.centroid_px[0] < 90.0


def test_color_detection_exposes_capture_time_covariance_and_quality() -> None:
    target = resolve_target("apple")
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 40), 14, (0, 0, 230), -1)
    observation = CameraObservation(
        image,
        np.ones((80, 100), dtype=np.float32),
        CameraIntrinsics(100.0, 100.0, 50.0, 40.0, 100, 80),
        np.zeros(3),
        np.eye(3),
        wall_time_s=8.5,
        sim_time_s=2.75,
        depth_metric=True,
    )
    detection = ColorSegmentationPerception().detect(
        observation, np.zeros(3), target, "apple"
    )
    assert detection.success
    assert detection.capture_time_s == 8.5
    assert detection.measurement_time_s == 2.75
    assert detection.covariance is not None and detection.covariance.shape == (3, 3)
    assert 0.0 < detection.valid_fraction <= 1.0
    assert 0.0 < detection.quality <= 1.0


def test_depth_anchor_never_uses_relative_depth_as_metres() -> None:
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:30, 20:40] = 255
    observation = CameraObservation(
        np.zeros((40, 60, 3), dtype=np.uint8),
        np.full((40, 60), 0.25, dtype=np.float32),
        CameraIntrinsics(60.0, 60.0, 30.0, 20.0, 60, 40),
        np.zeros(3),
        np.eye(3),
        depth_backend="relative-test",
        depth_metric=False,
    )
    position, selected, bbox_min, bbox_max, anchor_type = _estimate_world_position(
        observation, np.array([20, 10, 40, 30], dtype=float), mask
    )
    assert position is None
    assert bbox_min is None and bbox_max is None
    assert anchor_type == "non_metric_depth"
    assert np.array_equal(selected, mask)


def test_perception_rejects_non_uint8_frames() -> None:
    target = resolve_target("box")
    observation = CameraObservation(
        np.zeros((40, 60, 3), dtype=np.float32),
        np.ones((40, 60), dtype=np.float32),
        CameraIntrinsics(60.0, 60.0, 30.0, 20.0, 60, 40),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    with pytest.raises(ValueError, match="uint8"):
        ColorSegmentationPerception().detect(observation, np.zeros(3), target, "box")


def test_observation_rejects_invalid_sim_time() -> None:
    target = resolve_target("box")
    observation = CameraObservation(
        np.zeros((40, 60, 3), dtype=np.uint8),
        np.ones((40, 60), dtype=np.float32),
        CameraIntrinsics(60.0, 60.0, 30.0, 20.0, 60, 40),
        np.zeros(3),
        np.eye(3),
        sim_time_s=-0.1,
        depth_metric=True,
    )
    with pytest.raises(ValueError, match="sim_time_s"):
        ColorSegmentationPerception().detect(observation, np.zeros(3), target, "box")


def test_depth_anchor_rejects_invalid_bbox_without_crashing() -> None:
    observation = CameraObservation(
        frame_bgr=np.zeros((20, 30, 3), dtype=np.uint8),
        depth_m=np.ones((20, 30), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=30.0, fy=30.0, cx=15.0, cy=10.0, width=30, height=20
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    position, mask, bbox_min, bbox_max, anchor_type = _estimate_world_position(
        observation,
        np.array([5.0, np.nan, 2.0, 9.0]),
        None,
    )
    assert position is None
    assert anchor_type == "invalid_bbox"
    assert bbox_min is None
    assert bbox_max is None
    assert mask.shape == (20, 30)


def test_observation_rejects_nonfinite_camera_pose() -> None:
    target = resolve_target("box")
    observation = CameraObservation(
        frame_bgr=np.zeros((12, 16, 3), dtype=np.uint8),
        depth_m=np.ones((12, 16), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=12.0, fy=12.0, cx=8.0, cy=6.0, width=16, height=12
        ),
        camera_position=np.array([0.0, np.nan, 0.0]),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    with pytest.raises(ValueError, match="camera_position"):
        ColorSegmentationPerception().detect(observation, np.zeros(3), target, "box")


def test_observation_rejects_nonfinite_intrinsics() -> None:
    target = resolve_target("box")
    observation = CameraObservation(
        frame_bgr=np.zeros((12, 16, 3), dtype=np.uint8),
        depth_m=np.ones((12, 16), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=np.nan, fy=12.0, cx=8.0, cy=6.0, width=16, height=12
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    with pytest.raises(ValueError, match="intrinsics"):
        ColorSegmentationPerception().detect(observation, np.zeros(3), target, "box")


def test_semantic_detect_reuses_initialized_local_tracker_without_models() -> None:
    target = resolve_target("apple")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (80, 60), 18, (0, 0, 230), -1)
    observation = CameraObservation(
        frame_bgr=image,
        depth_m=np.ones((120, 160), dtype=np.float32),
        intrinsics=CameraIntrinsics(
            fx=120.0, fy=120.0, cx=80.0, cy=60.0, width=160, height=120
        ),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_metric=True,
    )
    semantic = object.__new__(SemanticPerception)
    semantic._initialized = True
    semantic._last_bbox = np.array([60.0, 40.0, 100.0, 80.0])
    semantic._last_mask = np.zeros((120, 160), dtype=np.uint8)
    semantic._last_mask[42:78, 62:98] = 255
    semantic._last_detection = None
    semantic._hsv_center = np.array([0, 255, 230], dtype=float)
    semantic._last_depth_median = 1.0
    detection = semantic.detect(observation, np.zeros(3, dtype=float), target, "apple")
    assert detection.success
    assert detection.backend == "semantic-track"
    assert semantic._track_failures == 0
    assert detection.bbox_xyxy is not None
    assert detection.mask is not None


def test_semantic_tracker_falls_back_to_color_when_depth_mask_is_empty() -> None:
    target = resolve_target("apple")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(image, (84, 60), 18, (0, 0, 230), -1)
    observation = CameraObservation(
        image,
        np.full((120, 160), 2.0, dtype=np.float32),
        CameraIntrinsics(120.0, 120.0, 80.0, 60.0, 160, 120),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    semantic = object.__new__(SemanticPerception)
    semantic._initialized = True
    semantic._last_bbox = np.array([60.0, 40.0, 100.0, 80.0])
    semantic._last_mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(semantic._last_mask, (80, 60), 18, 255, -1)
    semantic._last_detection = None
    semantic._hsv_center = np.array([0.0, 255.0, 230.0])
    semantic._last_depth_median = 1.0
    detection = semantic.detect(observation, np.zeros(3), target, "apple")
    assert detection.success
    assert detection.backend == "semantic-track"
    assert detection.centroid_px[0] > 80.0


def test_semantic_tracker_does_not_treat_old_bbox_as_fresh_evidence() -> None:
    observation = CameraObservation(
        np.zeros((120, 160, 3), dtype=np.uint8),
        np.full((120, 160), 2.0, dtype=np.float32),
        CameraIntrinsics(120.0, 120.0, 80.0, 60.0, 160, 120),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    semantic = object.__new__(SemanticPerception)
    semantic._last_bbox = np.array([60.0, 40.0, 100.0, 80.0])
    semantic._last_mask = np.zeros((120, 160), dtype=np.uint8)
    semantic._last_mask[42:78, 62:98] = 255
    semantic._last_detection = None
    semantic._hsv_center = np.array([0.0, 255.0, 230.0])
    semantic._last_depth_median = 1.0
    detection = semantic._track_from_last_mask(observation)
    assert not detection.success
    assert detection.target_position is None


def test_semantic_tracker_rejects_same_depth_plane_mask_expansion() -> None:
    image = np.full((120, 160, 3), (0, 0, 230), dtype=np.uint8)
    observation = CameraObservation(
        image,
        np.ones((120, 160), dtype=np.float32),
        CameraIntrinsics(120.0, 120.0, 80.0, 60.0, 160, 120),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    semantic = object.__new__(SemanticPerception)
    semantic._last_bbox = np.array([60.0, 40.0, 100.0, 80.0])
    semantic._last_mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(semantic._last_mask, (80, 60), 16, 255, -1)
    semantic._last_detection = None
    semantic._hsv_center = np.array([0.0, 255.0, 230.0])
    semantic._last_depth_median = 1.0
    detection = semantic._track_from_last_mask(observation)
    assert not detection.success
    assert detection.target_position is None
    assert np.count_nonzero(semantic._last_mask) < 1_000


def test_semantic_tracking_failure_immediately_triggers_redetection() -> None:
    calls = {"processor": 0}

    class Context:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return Context()

        @staticmethod
        def is_tensor(value):
            return False

    class FakeProcessor:
        def __call__(self, *, images, text, return_tensors):
            calls["processor"] += 1
            return {"input_ids": np.array([[1]], dtype=np.int64)}

        @staticmethod
        def post_process_grounded_object_detection(
            outputs,
            input_ids,
            *,
            box_threshold,
            text_threshold,
            target_sizes,
        ):
            return [{"boxes": [], "scores": []}]

    observation = CameraObservation(
        np.zeros((30, 40, 3), dtype=np.uint8),
        np.ones((30, 40), dtype=np.float32),
        CameraIntrinsics(40.0, 40.0, 20.0, 15.0, 40, 30),
        np.zeros(3),
        np.eye(3),
        depth_metric=True,
    )
    semantic = object.__new__(SemanticPerception)
    semantic._initialized = True
    semantic._frames_since_redetect = 0
    semantic._redetect_interval = 45
    semantic._track_failures = 0
    semantic._max_track_failures = 3
    semantic._track_from_last_mask = lambda unused: Detection(
        False, "semantic-track", None
    )
    semantic._image_cls = SimpleNamespace(
        fromarray=lambda array: SimpleNamespace(size=(40, 30))
    )
    semantic._torch = FakeTorch()
    semantic._device = "cpu"
    semantic._gdino_processor = FakeProcessor()
    semantic._gdino_model = lambda **inputs: SimpleNamespace()
    semantic._box_threshold = 0.25
    semantic._text_threshold = 0.25
    result = semantic.detect(observation, np.zeros(3), resolve_target("box"), "box")
    assert not result.success
    assert calls["processor"] == 1


def test_semantic_tracking_gate_rejects_large_shrink_and_remote_swap() -> None:
    semantic = object.__new__(SemanticPerception)
    semantic._last_bbox = np.array([40.0, 30.0, 80.0, 70.0])
    semantic._last_mask = np.zeros((100, 140), dtype=np.uint8)
    semantic._last_mask[30:70, 40:80] = 255
    tiny = np.zeros_like(semantic._last_mask)
    tiny[45:52, 55:62] = 255
    remote = np.zeros_like(semantic._last_mask)
    remote[30:70, 95:135] = 255
    assert not semantic._tracking_mask_is_consistent(tiny)
    assert not semantic._tracking_mask_is_consistent(remote)


def test_semantic_hsv_center_uses_circular_hue_statistics() -> None:
    hsv = np.zeros((10, 20, 3), dtype=np.uint8)
    hsv[:, :10] = (179, 240, 220)
    hsv[:, 10:] = (1, 240, 220)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    center = SemanticPerception._mask_hsv_center(
        bgr, np.full((10, 20), 255, dtype=np.uint8)
    )
    assert center is not None
    assert center[0] < 5.0 or center[0] > 175.0


def test_sam_uses_standard_box_shape_and_highest_iou_mask() -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        float32 = np.float32

        @staticmethod
        def no_grad():
            return NoGrad()

        @staticmethod
        def is_tensor(value):
            return False

    candidates = np.zeros((1, 3, 20, 30), dtype=np.uint8)
    candidates[0, 0, 6:12, 9:14] = 1
    candidates[0, 1, 6:12, 14:19] = 1
    candidates[0, 2, 7:14, 12:20] = 1
    candidates[0, 2, 0:3, 0:3] = 1
    captured = {}

    class FakeImageProcessor:
        @staticmethod
        def post_process_masks(pred_masks, original_sizes, reshaped_sizes):
            return [FakeTensor(candidates)]

    class FakeProcessor:
        image_processor = FakeImageProcessor()

        def __call__(self, image, input_boxes, return_tensors):
            captured["input_boxes"] = input_boxes
            return {
                "original_sizes": FakeTensor([[20, 30]]),
                "reshaped_input_sizes": FakeTensor([[20, 30]]),
            }

    semantic = object.__new__(SemanticPerception)
    semantic._torch = FakeTorch()
    semantic._device = "cpu"
    semantic._sam_processor = FakeProcessor()
    semantic._sam_model = lambda **inputs: SimpleNamespace(
        pred_masks=FakeTensor(np.zeros((1, 1, 3, 4, 4))),
        iou_scores=FakeTensor([[[0.1, 0.2, 0.9]]]),
    )
    bbox = np.array([8.0, 5.0, 22.0, 16.0])
    mask = semantic._sam_mask(object(), bbox)
    assert captured["input_boxes"] == [[bbox.tolist()]]
    assert mask[10, 16] == 255
    assert mask[1, 1] == 0
    assert mask[8, 10] == 0


def test_semantic_auto_device_prefers_cuda_over_mps() -> None:
    available = SimpleNamespace(is_available=lambda: True)
    fake_torch = SimpleNamespace(
        cuda=available, backends=SimpleNamespace(mps=available)
    )
    assert _resolve_torch_device_name(fake_torch, " AUTO ") == "cuda"
    assert _resolve_torch_device_name(fake_torch, " METAL ") == "mps"


def test_controller_uses_robot_spec_dimensions_for_lite6() -> None:
    scene = build_scene(resolve_target("cup"), robot="lite6")
    controller = ResolvedRateController(
        scene.model,
        scene.ee_frame_name,
        scene.ee_frame_type,
        scene.ee_frame_offset,
        scene.robot,
        ControllerConfig(),
    )
    assert len(controller._joint_ids) == 6
    assert len(controller._actuator_ids) == 6


def test_standoff_controller_latches_relative_tracking_offset() -> None:
    scene = build_scene(resolve_target("cup"), robot="panda")
    controller = ResolvedRateController(
        scene.model,
        scene.ee_frame_name,
        scene.ee_frame_type,
        scene.ee_frame_offset,
        scene.robot,
        ControllerConfig(standoff_m=0.10),
    )
    controller.reset(scene.data)
    first_target = np.array([0.50, 0.05, 0.40])
    first_goal = controller.desired_position(scene.data, first_target)
    target_translation = np.array([0.06, -0.04, 0.02])
    second_goal = controller.desired_position(
        scene.data, first_target + target_translation
    )

    assert np.allclose(second_goal - first_goal, target_translation)
    assert np.isclose(np.linalg.norm(first_goal - first_target), 0.10)
