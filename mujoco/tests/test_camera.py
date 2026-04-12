from __future__ import annotations

import unittest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.camera import discover_cameras


class CameraDiscoveryTest(unittest.TestCase):
    def test_discovery_returns_list(self) -> None:
        cameras = discover_cameras(max_devices=2)
        self.assertIsInstance(cameras, list)
        for info in cameras:
            self.assertIsInstance(info.index, int)
            self.assertIsInstance(info.backend_name, str)


if __name__ == "__main__":
    unittest.main()

