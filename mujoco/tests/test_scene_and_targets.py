from __future__ import annotations

import json

import numpy as np
import mujoco

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import CameraConfig
from mujoco_servo.scene import build_scene, frame_position, site_position
from mujoco_servo.targets import TargetMotion, base_position, load_target_specs, resolve_target


def test_scene_contains_robot_target_and_camera() -> None:
    scene = build_scene(resolve_target("hammer"), CameraConfig())
    assert scene.source == "menagerie"
    assert scene.model.nu >= 7
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "hand") >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "target_site") >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "servo_camera") >= 0
    assert scene.ee_frame_type == "body_point"
    ee = frame_position(scene.model, scene.data, scene.ee_frame_type, scene.ee_frame_name, scene.ee_frame_offset)
    target = site_position(scene.model, scene.data, "target_site")
    assert ee.shape == (3,)
    assert target.shape == (3,)
    assert np.isfinite(ee).all()
    assert np.isfinite(target).all()
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom_0") >= 0


def test_scene_can_load_alternate_robot_spec() -> None:
    scene = build_scene(resolve_target("box"), CameraConfig(), robot="ur5e")
    assert scene.robot.name == "ur5e"
    assert scene.ee_frame_type == "site"
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site") >= 0
    assert scene.robot.dof == 6


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


def test_custom_target_file_adds_replaceable_target(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "banana",
                        "shape": "capsule",
                        "size": [0.04, 0.04, 0.16],
                        "rgba": [0.95, 0.78, 0.12, 1.0],
                        "aliases": ["yellow banana"],
                        "base_position": [0.42, -0.05, 0.36],
                    }
                ]
            }
        )
    )
    extra = load_target_specs(target_file)
    target = resolve_target("track the yellow banana", extra)
    assert target.name == "banana"
    assert target.shape == "capsule"
    assert np.allclose(base_position(target), [0.42, -0.05, 0.36])
