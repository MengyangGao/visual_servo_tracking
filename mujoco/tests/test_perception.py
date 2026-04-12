from __future__ import annotations

import unittest

import cv2
import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import default_camera_intrinsics, target_world_position
from mujoco_servo.perception import OracleBackend, PromptGuidedVisionBackend
from mujoco_servo.types import CameraPose


class PerceptionTest(unittest.TestCase):
    def test_oracle_returns_world_pose(self) -> None:
        intr = default_camera_intrinsics(640, 480)
        backend = OracleBackend(target_world_position)
        detection = backend.detect(np.zeros((480, 640, 3), dtype=np.uint8), "cup", intr, CameraPose.identity())
        self.assertTrue(detection.success)
        self.assertEqual(detection.backend, "oracle")
        self.assertIsNotNone(detection.target_position_world)
        self.assertGreater(detection.estimated_distance_m or 0.0, 0.0)

    def test_heuristic_color_detection(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[80:160, 100:200] = (0, 0, 255)
        intr = default_camera_intrinsics(320, 240)
        backend = PromptGuidedVisionBackend()
        detection = backend.detect(image, "red cup", intr, CameraPose.identity())
        self.assertTrue(detection.success)
        self.assertEqual(detection.backend, "heuristic")
        self.assertGreater(detection.mask_area_px, 0)
        x1, y1, x2, y2 = detection.bbox_xyxy
        self.assertGreater(x2 - x1, 10)
        self.assertGreater(y2 - y1, 10)


if __name__ == "__main__":
    unittest.main()

