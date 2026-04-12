from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import default_camera_intrinsics, target_world_position
from mujoco_servo.image_features import bbox_corners_xyxy
from mujoco_servo.perception import (
    OracleBackend,
    GroundedSam2Config,
    PerceptionBackend,
    PerceptionSession,
    PromptGuidedVisionBackend,
    build_backend,
)
from mujoco_servo.types import CameraPose, Detection


class PerceptionTest(unittest.TestCase):
    def test_session_uses_track_after_first_frame(self) -> None:
        class CountingBackend(PerceptionBackend):
            name = "counting"

            def __init__(self) -> None:
                self.detect_calls = 0
                self.track_calls = 0

            def detect(self, frame_bgr, prompt, intrinsics, camera_pose) -> Detection:  # type: ignore[override]
                self.detect_calls += 1
                return Detection(
                    success=True,
                    prompt=prompt,
                    label="first",
                    score=0.8,
                    bbox_xyxy=np.array([10.0, 10.0, 30.0, 30.0]),
                    centroid_px=np.array([20.0, 20.0]),
                    backend=self.name,
                )

            def track(self, frame_bgr, prompt, intrinsics, camera_pose, previous_detection=None) -> Detection:  # type: ignore[override]
                self.track_calls += 1
                return Detection(
                    success=True,
                    prompt=prompt,
                    label="tracked",
                    score=0.9,
                    bbox_xyxy=np.array([30.0, 30.0, 50.0, 50.0]),
                    centroid_px=np.array([40.0, 40.0]),
                    backend=self.name,
                )

        backend = CountingBackend()
        session = PerceptionSession(backend=backend, prompt="cup")
        intr = default_camera_intrinsics(160, 120)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        first = session.update(frame, intr, CameraPose.identity())
        second = session.update(frame, intr, CameraPose.identity())
        self.assertEqual(backend.detect_calls, 1)
        self.assertEqual(backend.track_calls, 1)
        self.assertEqual(first.label, "first")
        self.assertEqual(second.label, "tracked")

    def test_oracle_returns_world_pose(self) -> None:
        intr = default_camera_intrinsics(640, 480)
        backend = OracleBackend(target_world_position)
        detection = backend.detect(np.zeros((480, 640, 3), dtype=np.uint8), "cup", intr, CameraPose.identity())
        self.assertTrue(detection.success)
        self.assertEqual(detection.backend, "oracle")
        self.assertIsNotNone(detection.target_position_world)
        self.assertGreater(detection.estimated_distance_m or 0.0, 0.0)
        self.assertIsNotNone(detection.corners_px)
        self.assertEqual(np.asarray(detection.corners_px).shape, (4, 2))

    def test_heuristic_color_detection(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[80:160, 100:200] = (0, 0, 255)
        intr = default_camera_intrinsics(320, 240)
        backend = PromptGuidedVisionBackend()
        detection = backend.detect(image, "red cup", intr, CameraPose.identity())
        self.assertTrue(detection.success)
        self.assertEqual(detection.backend, "heuristic")
        self.assertGreater(detection.mask_area_px, 0)
        self.assertIsNotNone(detection.corners_px)
        self.assertEqual(np.asarray(detection.corners_px).shape, (4, 2))
        x1, y1, x2, y2 = detection.bbox_xyxy
        self.assertGreater(x2 - x1, 10)
        self.assertGreater(y2 - y1, 10)

    def test_heuristic_track_remaps_roi_to_full_frame(self) -> None:
        frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_b = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_a[80:150, 40:110] = (0, 0, 255)
        frame_b[80:150, 160:230] = (0, 0, 255)
        intr = default_camera_intrinsics(320, 240)
        backend = PromptGuidedVisionBackend()
        first = backend.detect(frame_a, "red cup", intr, CameraPose.identity())
        self.assertTrue(first.success)
        tracked = backend.track(frame_b, "red cup", intr, CameraPose.identity(), first)
        self.assertTrue(tracked.success)
        self.assertEqual(tracked.mask.shape, frame_b.shape[:2])
        self.assertIsNotNone(tracked.corners_px)
        self.assertEqual(np.asarray(tracked.corners_px).shape, (4, 2))
        x1, y1, x2, y2 = tracked.bbox_xyxy
        self.assertAlmostEqual(float(x1), 160.0, delta=20.0)
        self.assertAlmostEqual(float(y1), 80.0, delta=20.0)
        self.assertGreater(tracked.mask_area_px, 0)

    def test_grounded_sam2_backend_falls_back_when_constructor_fails(self) -> None:
        with patch("mujoco_servo.perception.GroundedSam2Backend", side_effect=RuntimeError("no model")):
            backend = build_backend("grounded-sam2", "cup", target_world_position)
        self.assertEqual(backend.name, "heuristic")

    def test_grounded_sam2_preset_mapping(self) -> None:
        default = GroundedSam2Config.from_preset("default")
        small = GroundedSam2Config.from_preset("small")
        lite = GroundedSam2Config.from_preset("lite")
        self.assertEqual(default.grounding_model_id, "IDEA-Research/grounding-dino-base")
        self.assertEqual(default.sam2_checkpoint_name, "sam2.1_hiera_base_plus.pt")
        self.assertEqual(small.grounding_model_id, "IDEA-Research/grounding-dino-tiny")
        self.assertEqual(small.sam2_checkpoint_name, "sam2.1_hiera_small.pt")
        self.assertEqual(lite.grounding_model_id, "IDEA-Research/grounding-dino-tiny")
        self.assertEqual(lite.sam2_checkpoint_name, "sam2.1_hiera_tiny.pt")

    def test_bbox_corners_are_ordered(self) -> None:
        corners = bbox_corners_xyxy(np.array([10.0, 20.0, 30.0, 40.0]))
        self.assertEqual(corners.shape, (4, 2))
        self.assertAlmostEqual(float(corners[0, 0]), 10.0)
        self.assertAlmostEqual(float(corners[0, 1]), 20.0)


if __name__ == "__main__":
    unittest.main()
