from __future__ import annotations

import json

import numpy as np
import mujoco
import pytest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import CameraConfig
from mujoco_servo.app import VisualServoSimulation
from mujoco_servo.config import DemoConfig
from mujoco_servo.scene import build_scene, frame_position, set_target_position, site_position
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


def test_scene_accepts_robot_specific_target_position() -> None:
    scene = build_scene(resolve_target("apple"), CameraConfig(), robot="ur5e", target_position=np.array([-0.3, 0.3, 0.33]))
    target = site_position(scene.model, scene.data, "target_site")
    assert np.allclose(target, [-0.3, 0.3, 0.33], atol=1e-6)


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


def test_custom_target_exact_match_beats_builtin_substring(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "urbox",
                        "shape": "box",
                        "size": [0.07, 0.07, 0.07],
                        "rgba": [0.2, 0.7, 0.2, 1.0],
                    }
                ]
            }
        )
    )
    extra = load_target_specs(target_file)
    target = resolve_target("urbox", extra)
    assert target.name == "urbox"


def test_custom_target_part_offset_alias_is_supported(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "offset-object",
                        "parts": [
                            {
                                "shape": "box",
                                "size": [0.04, 0.04, 0.04],
                                "offset": [0.02, -0.01, 0.03],
                            }
                        ],
                    }
                ]
            }
        )
    )
    extra = load_target_specs(target_file)
    target = resolve_target("offset-object", extra)
    assert target.parts
    assert np.allclose(target.parts[0].pos, [0.02, -0.01, 0.03], atol=1e-9)


def test_custom_target_base_position_is_respected_by_runtime(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "far-custom",
                        "shape": "box",
                        "size": [0.04, 0.04, 0.04],
                        "rgba": [0.2, 0.7, 0.2, 1.0],
                        "base_position": [0.25, -0.20, 0.31],
                    }
                ]
            }
        )
    )
    app = VisualServoSimulation(
        DemoConfig(
            target="far-custom",
            target_file=str(target_file),
            detector="oracle",
            trajectory="static",
            steps=1,
            headless=True,
            viewer=False,
            realtime=False,
        )
    )
    assert np.allclose(app.motion.position(0.0), [0.25, -0.20, 0.31])


def test_target_file_rejects_invalid_geometry(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "bad",
                        "shape": "mesh",
                        "size": [-0.04, 0.04, 0.04],
                        "rgba": [1.2, 0.7, 0.2, 1.0],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="shape"):
        load_target_specs(target_file)


def test_target_file_rejects_duplicate_names(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {"name": "dup", "shape": "box"},
                    {"name": "dup", "shape": "sphere"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_target_specs(target_file)


def test_target_file_rejects_duplicate_aliases(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {"name": "first", "shape": "box", "aliases": ["shared object"]},
                    {"name": "second", "shape": "sphere", "aliases": ["shared object"]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate target name or alias"):
        load_target_specs(target_file)


def test_set_target_position_rejects_nonfinite_values() -> None:
    scene = build_scene(resolve_target("cup"), CameraConfig())
    with pytest.raises(ValueError, match="target position"):
        set_target_position(scene.model, scene.data, np.array([0.4, np.nan, 0.3]))
