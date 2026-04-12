from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.camera import discover_cameras


class CameraDiscoveryTest(unittest.TestCase):
    def test_discovery_returns_list(self) -> None:
        class FakeCapture:
            def __init__(self, *args, **kwargs) -> None:
                self._opened = True

            def isOpened(self) -> bool:
                return self._opened

            def read(self):
                return True, np.zeros((12, 16, 3), dtype=np.uint8)

            def release(self) -> None:
                self._opened = False

        with patch("cv2.VideoCapture", side_effect=lambda *args, **kwargs: FakeCapture()):
            cameras = discover_cameras(max_devices=2)
        self.assertIsInstance(cameras, list)
        self.assertGreaterEqual(len(cameras), 1)
        for info in cameras:
            self.assertIsInstance(info.index, int)
            self.assertIsInstance(info.backend_name, str)


if __name__ == "__main__":
    unittest.main()
