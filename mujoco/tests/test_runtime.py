from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import build_settings
from mujoco_servo.control import ServoGains, compute_servo_command
from mujoco_servo.types import Detection
from mujoco_servo.robot import build_robot_spec
from mujoco_servo.scene import build_scene_bundle, set_mocap_body_pose
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

    def test_scene_contains_visible_mocap_markers(self) -> None:
        spec = build_robot_spec(prefer_reference=False)
        bundle = build_scene_bundle(spec, "apple", 640, 480)
        for body_name in ("vision_camera", "vision_target", "vision_ee"):
            body_id = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.assertGreaterEqual(body_id, 0)
            self.assertTrue(set_mocap_body_pose(bundle.model, bundle.data, body_name, np.array([0.1, 0.2, 0.3]), np.eye(3)))

    def test_feature_servo_command_uses_corners(self) -> None:
        spec = build_robot_spec(prefer_reference=False)
        bundle = build_scene_bundle(spec, "phone", 640, 480)
        det = Detection(
            success=True,
            prompt="phone",
            label="phone",
            score=0.92,
            bbox_xyxy=np.array([220.0, 140.0, 340.0, 280.0]),
            centroid_px=np.array([280.0, 210.0]),
            corners_px=np.array([[220.0, 140.0], [340.0, 140.0], [340.0, 280.0], [220.0, 280.0]]),
            estimated_distance_m=0.42,
            backend="heuristic",
        )
        qpos_cmd, telemetry = compute_servo_command(
            model=bundle.model,
            data=bundle.data,
            detection=det,
            prototype=bundle.target_proto,
            camera_intrinsics=bundle.camera_intrinsics,
            camera_pose=bundle.camera_pose,
            ee_body_name=bundle.ee_body_name,
            gains=ServoGains(),
            dt=1.0 / 30.0,
        )
        self.assertEqual(qpos_cmd.shape[0], bundle.model.nq)
        self.assertTrue(np.isfinite(qpos_cmd).all())
        self.assertGreater(telemetry.feature_error_px, 0.0)


if __name__ == "__main__":
    unittest.main()
