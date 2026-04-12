from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import build_settings
from mujoco_servo.rendering import side_by_side_view
from mujoco_servo.runtime import run_camera, run_simulation


class RuntimeSmokeTest(unittest.TestCase):
    def test_simulation_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = build_settings(
                prompt="apple",
                backend="oracle",
                mode="sim",
                run_mode="auto",
                max_steps=8,
                show_view=False,
                record=False,
            )
            settings.output_dir = Path(tmpdir)
            summary = run_simulation(settings)
            self.assertEqual(summary["mode"], "sim")
            self.assertEqual(summary["steps"], 8)
            self.assertIsNotNone(summary["final_position_error_m"])

    def test_camera_heuristic_smoke(self) -> None:
        # This test is intentionally light: it validates backend wiring without requiring
        # the open-vocabulary model to be installed.
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = build_settings(
                prompt="red cup",
                backend="heuristic",
                mode="camera",
                run_mode="auto",
                max_steps=1,
                camera_index=None,
                show_view=False,
                record=False,
            )
            settings.output_dir = Path(tmpdir)
            try:
                summary = run_camera(settings)
            except RuntimeError as exc:
                self.skipTest(f"camera unavailable in this environment: {exc}")
            self.assertEqual(summary["mode"], "camera")

    def test_side_by_side_view_shapes(self) -> None:
        robot = np.zeros((120, 160, 3), dtype=np.uint8)
        camera = np.zeros((180, 240, 3), dtype=np.uint8)
        robot[:] = (10, 20, 30)
        camera[:] = (40, 50, 60)
        combined = side_by_side_view(robot, camera)
        self.assertEqual(combined.ndim, 3)
        self.assertEqual(combined.shape[2], 3)
        self.assertGreater(combined.shape[1], robot.shape[1] + camera.shape[1])
        self.assertEqual(combined.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
