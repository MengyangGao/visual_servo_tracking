from __future__ import annotations

import numpy as np
import mujoco

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import CameraConfig
from mujoco_servo.scene import build_scene, site_position
from mujoco_servo.targets import TargetMotion, resolve_target


def test_scene_contains_robot_target_and_camera() -> None:
    scene = build_scene(resolve_target("cup"), CameraConfig())
    assert scene.model.nu == 7
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "target_site") >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "servo_camera") >= 0
    ee = site_position(scene.model, scene.data, "ee_site")
    target = site_position(scene.model, scene.data, "target_site")
    assert ee.shape == (3,)
    assert target.shape == (3,)
    assert np.isfinite(ee).all()
    assert np.isfinite(target).all()


def test_target_trajectories_are_smooth_and_moving() -> None:
    target = resolve_target("bottle")
    motion = TargetMotion(target, "figure-eight")
    p0 = motion.position(0.0)
    p1 = motion.position(1.0)
    p2 = motion.position(2.0)
    assert not np.allclose(p0, p1)
    assert not np.allclose(p1, p2)
    assert np.linalg.norm(p2 - p1) < 0.25


def test_random_walk_stays_inside_workspace() -> None:
    motion = TargetMotion(resolve_target("apple"), "random-walk", seed=4)
    samples = np.array([motion.position(i / 30.0) for i in range(180)])
    assert np.ptp(samples[:, 0]) > 0.005
    assert samples[:, 0].min() > 0.25
    assert samples[:, 0].max() < 0.70
    assert samples[:, 2].min() > 0.25
