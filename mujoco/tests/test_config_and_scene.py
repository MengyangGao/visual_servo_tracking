from __future__ import annotations

import unittest

import numpy as np

from ._bootstrap import SRC  # noqa: F401

import mujoco

from mujoco_servo.config import build_settings, lookup_target_prototype, target_world_position
from mujoco_servo.robot import build_robot_spec
from mujoco_servo.scene import body_pose_world, build_scene_bundle


class SceneSmokeTest(unittest.TestCase):
    def test_prompt_catalog(self) -> None:
        proto = lookup_target_prototype("red apple")
        self.assertEqual(proto.name, "apple")
        self.assertGreater(proto.nominal_standoff_m, 0.0)

    def test_scene_bundle_loads_and_steps(self) -> None:
        settings = build_settings(prompt="cup", backend="oracle", mode="sim", max_steps=3, show_view=False)
        robot_spec = build_robot_spec(prefer_reference=False)
        bundle = build_scene_bundle(robot_spec, settings.prompt, settings.camera_width, settings.camera_height)
        self.assertGreaterEqual(bundle.model.nq, 7)
        self.assertGreaterEqual(bundle.model.nu, 7)
        self.assertEqual(bundle.target_proto.name, "cup")

        target = target_world_position(settings.prompt)
        self.assertEqual(target.shape, (3,))

        hand_pos, hand_rot = body_pose_world(bundle.model, bundle.data, bundle.ee_body_name)
        self.assertEqual(hand_pos.shape, (3,))
        self.assertEqual(hand_rot.shape, (3, 3))

        for _ in range(2):
            mujoco.mj_step(bundle.model, bundle.data)

        self.assertTrue(np.isfinite(bundle.data.qpos).all())


if __name__ == "__main__":
    unittest.main()

