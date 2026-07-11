from __future__ import annotations

from dataclasses import replace
import json

import cv2
import numpy as np
import mujoco
import pytest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import CameraConfig, RobotSpec
from mujoco_servo.app import VisualServoSimulation
from mujoco_servo.config import DemoConfig
from mujoco_servo.scene import build_scene, camera_position, frame_position, joint_positions, set_target_position, site_position
from mujoco_servo.targets import TargetMotion, base_position, load_target_specs, resolve_target


def _write_minimal_robot(tmp_path, *, include: bool = False) -> RobotSpec:
    asset_dir = tmp_path / "robot assets & meshes"
    asset_dir.mkdir(exist_ok=True)
    robot_xml = tmp_path / "minimal_robot.xml"
    if include:
        robot_xml.write_text('<mujoco model="included"><include file="robot_part.xml"/></mujoco>')
    else:
        robot_xml.write_text(
            """
            <mujoco model="minimal">
              <compiler angle="radian" meshdir="some-other-relative-directory" autolimits="true"/>
              <worldbody>
                <body name="minimal_link" pos="0 0 0.1">
                  <joint name="minimal_joint" type="hinge" axis="0 0 1" range="-1 1"/>
                  <geom type="capsule" size="0.025 0.10" mass="1"/>
                  <site name="minimal_tool" pos="0 0 0.22" size="0.01"/>
                </body>
              </worldbody>
              <actuator>
                <position name="minimal_actuator" joint="minimal_joint" kp="100"/>
              </actuator>
            </mujoco>
            """
        )
    return RobotSpec(
        name="minimal",
        xml_path=robot_xml,
        asset_dir=asset_dir,
        joint_names=("minimal_joint",),
        actuator_names=("minimal_actuator",),
        home_qpos=(0.15,),
        ee_frame_name="minimal_tool",
        ee_frame_type="site",
        default_target_position=(0.24, -0.08, 0.31),
    )


def _write_tetrahedron_obj(path) -> None:
    path.write_text(
        """
        v 0 0 0
        v 0.02 0 0
        v 0 0.02 0
        v 0 0 0.02
        f 1 3 2
        f 1 2 4
        f 1 4 3
        f 2 3 4
        """
    )


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


def test_scene_uses_robot_default_target_position_and_supports_no_keyframe(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    scene = build_scene(resolve_target("box"), CameraConfig(), robot=robot)
    assert scene.source == "external"
    assert scene.model.nkey == 0
    assert scene.ee_site_name == "minimal_tool"
    assert scene.ee_body_name is None
    assert np.allclose(site_position(scene.model, scene.data, scene.target_site_name), robot.default_target_position)
    assert np.allclose(joint_positions(scene.model, scene.data, robot.joint_names), robot.home_qpos)
    assert np.allclose(camera_position(scene.model, scene.data, scene.camera_name), scene.data.cam_xpos[0])


def test_scene_rejects_robot_mjcf_include_with_clear_error(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path, include=True)
    with pytest.raises(RuntimeError, match="uses <include>.*self-contained"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_rejects_torque_actuator_robot_adapter(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text()
    text = text.replace(
        '<position name="minimal_actuator" joint="minimal_joint" kp="100"/>',
        '<motor name="minimal_actuator" joint="minimal_joint" gear="1"/>',
    )
    robot.xml_path.write_text(text)
    with pytest.raises(RuntimeError, match="must be a non-degenerate MuJoCo position servo"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_rejects_named_actuator_mapped_to_wrong_joint(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text()
    text = text.replace(
        '<joint name="minimal_joint" type="hinge" axis="0 0 1" range="-1 1"/>',
        '<joint name="minimal_joint" type="hinge" axis="0 0 1" range="-1 1"/>'
        '<joint name="other_joint" type="slide" axis="1 0 0" range="-0.1 0.1"/>',
    )
    text = text.replace('joint="minimal_joint" kp="100"', 'joint="other_joint" kp="100"')
    robot.xml_path.write_text(text)
    with pytest.raises(RuntimeError, match="does not transmit joint 'minimal_joint'"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_rejects_missing_declared_actuator_instead_of_falling_back(tmp_path) -> None:
    robot = replace(_write_minimal_robot(tmp_path), actuator_names=("misspelled_actuator",))
    with pytest.raises(RuntimeError, match="declared actuator 'misspelled_actuator' not found"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_names_single_unnamed_joint_actuator_from_descriptor(tmp_path) -> None:
    robot = replace(_write_minimal_robot(tmp_path), actuator_names=("declared_actuator",))
    robot.xml_path.write_text(robot.xml_path.read_text().replace(' name="minimal_actuator"', ""))
    scene = build_scene(resolve_target("box"), robot=robot)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "declared_actuator") >= 0


def test_scene_rejects_non_joint_actuator_transmission_with_colliding_id(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text()
    text = text.replace(
        "<actuator>",
        '<tendon><fixed name="minimal_tendon"><joint joint="minimal_joint" coef="1"/></fixed></tendon><actuator>',
    )
    text = text.replace('joint="minimal_joint" kp="100"', 'tendon="minimal_tendon" kp="100"')
    robot.xml_path.write_text(text)
    with pytest.raises(RuntimeError, match="must use a joint transmission"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_rejects_home_position_outside_joint_range(tmp_path) -> None:
    robot = replace(_write_minimal_robot(tmp_path), home_qpos=(10.0,))
    with pytest.raises(RuntimeError, match="home_qpos.*must be within"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_rejects_home_position_outside_actuator_control_range(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text().replace(
        'joint="minimal_joint" kp="100"',
        'joint="minimal_joint" kp="100" ctrlrange="-0.1 0.1"',
    )
    robot.xml_path.write_text(text)
    with pytest.raises(RuntimeError, match="maps to control.*outside actuator"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_rejects_passive_actuator_constant_outside_control_range(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text().replace(
        "</actuator>",
        '<motor name="passive_actuator" joint="minimal_joint" ctrlrange="-1 1"/></actuator>',
    )
    robot.xml_path.write_text(text)
    robot = replace(robot, passive_actuator_ctrl=(("passive_actuator", 2.0),))
    with pytest.raises(RuntimeError, match="passive actuator 'passive_actuator'.*outside range"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_scales_position_command_by_scalar_transmission_gear(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text().replace('joint="minimal_joint" kp="100"', 'joint="minimal_joint" kp="100" gear="2"')
    robot.xml_path.write_text(text)
    scene = build_scene(resolve_target("box"), robot=robot)
    actuator_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "minimal_actuator")
    assert np.isclose(scene.data.ctrl[actuator_id], 2.0 * robot.home_qpos[0])


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


def test_custom_target_file_accepts_top_level_list(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            [
                {
                    "name": "banana",
                    "shape": "capsule",
                    "size": [0.04, 0.04, 0.16],
                    "rgba": [0.95, 0.78, 0.12, 1.0],
                    "aliases": ["yellow banana"],
                }
            ]
        )
    )
    targets = load_target_specs(target_file)
    assert targets["banana"].shape == "capsule"
    assert resolve_target("track the yellow banana", targets).name == "banana"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("aliases", "yellow banana", "aliases must be a list"),
        ("parts", {"shape": "box"}, "parts must be a list"),
    ],
)
def test_target_file_rejects_non_list_collection_fields(tmp_path, field, value, message) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(json.dumps({"targets": [{"name": "bad", field: value}]}))
    with pytest.raises(ValueError, match=message):
        load_target_specs(target_file)


def test_target_file_normalizes_quaternion_and_rejects_zero(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    payload = {
        "targets": [
            {
                "name": "oriented",
                "shape": "compound",
                "parts": [{"shape": "box", "quat": [2.0, 0.0, 0.0, 0.0]}],
            }
        ]
    }
    target_file.write_text(json.dumps(payload))
    target = load_target_specs(target_file)["oriented"]
    assert np.allclose(target.parts[0].quat, [1.0, 0.0, 0.0, 0.0])
    payload["targets"][0]["parts"][0]["quat"] = [0.0, 0.0, 0.0, 0.0]
    target_file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="quaternion|quat.*non-zero"):
        load_target_specs(target_file)


def test_unknown_target_is_not_silently_replaced_by_box() -> None:
    with pytest.raises(ValueError, match="unknown target"):
        resolve_target("a completely unknown model")
    with pytest.raises(ValueError, match="unknown target"):
        resolve_target("candy")


def test_capsule_size_is_full_outer_size_and_target_is_noncolliding() -> None:
    scene = build_scene(resolve_target("capsule"), CameraConfig())
    geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    assert np.allclose(scene.model.geom_size[geom_id, :2], [0.0225, 0.0575], atol=1e-9)
    assert scene.model.geom_contype[geom_id] == 0
    assert scene.model.geom_conaffinity[geom_id] == 0


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ("sphere", [0.04, 0.05, 0.04]),
        ("cylinder", [0.04, 0.05, 0.10]),
        ("capsule", [0.06, 0.06, 0.04]),
    ],
)
def test_target_file_rejects_inconsistent_round_geometry_sizes(tmp_path, shape, size) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(json.dumps([{"name": "bad-round", "shape": shape, "size": size}]))
    with pytest.raises(ValueError, match="equal|diameter|height"):
        load_target_specs(target_file)


def test_scene_loads_relative_mesh_target_with_scale_and_safe_path(tmp_path) -> None:
    mesh_path = tmp_path / "target mesh & model.obj"
    _write_tetrahedron_obj(mesh_path)
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "mesh-object",
                        "shape": "mesh",
                        "mesh_file": mesh_path.name,
                        "scale": [2.0, 3.0, 4.0],
                        "size": [0.04, 0.06, 0.08],
                        "rgba": [0.2, 0.7, 0.3, 1.0],
                    }
                ]
            }
        )
    )
    target = load_target_specs(target_file)["mesh-object"]
    assert target.mesh_path == mesh_path.resolve()
    assert target.mesh_scale == (2.0, 3.0, 4.0)
    scene = build_scene(target, robot=_write_minimal_robot(tmp_path))
    geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    mesh_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH, "target_mesh")
    assert scene.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert np.allclose(scene.model.mesh_scale[mesh_id], [2.0, 3.0, 4.0])


def test_scene_loads_open_visual_mesh_as_shell(tmp_path) -> None:
    mesh_path = tmp_path / "open-plane.obj"
    mesh_path.write_text("v 0 0 0\nv 0.1 0 0\nv 0.1 0.1 0\nv 0 0.1 0\nf 1 2 3\nf 1 3 4\n")
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            [
                {
                    "name": "open-mesh",
                    "shape": "mesh",
                    "mesh_file": mesh_path.name,
                    "size": [0.1, 0.1, 0.01],
                }
            ]
        )
    )
    target = load_target_specs(target_file)["open-mesh"]
    scene = build_scene(target, robot=_write_minimal_robot(tmp_path))
    mesh_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH, "target_mesh")
    assert mesh_id >= 0


def test_scene_resolves_texture_without_texturedir_relative_to_robot_xml(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    texture_path = tmp_path / "root_texture.png"
    assert cv2.imwrite(str(texture_path), np.full((4, 4, 3), 180, dtype=np.uint8))
    text = robot.xml_path.read_text()
    text = text.replace(
        "<worldbody>",
        '<asset><texture name="root_texture" type="2d" file="root_texture.png"/>'
        '<material name="root_material" texture="root_texture"/></asset><worldbody>',
    )
    text = text.replace('mass="1"', 'mass="1" material="root_material"')
    robot.xml_path.write_text(text)
    scene = build_scene(resolve_target("box"), robot=robot)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_TEXTURE, "root_texture") >= 0


def test_scene_keeps_injected_visual_target_when_source_discards_visuals(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    text = robot.xml_path.read_text().replace('autolimits="true"', 'autolimits="true" discardvisual="true"')
    robot.xml_path.write_text(text)
    scene = build_scene(resolve_target("box"), robot=robot)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom") >= 0


def test_scene_disables_source_strippath_for_absolute_target_mesh(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    robot.xml_path.write_text(robot.xml_path.read_text().replace('autolimits="true"', 'autolimits="true" strippath="true"'))
    target_dir = tmp_path / "separate target assets"
    target_dir.mkdir()
    mesh_path = target_dir / "target.obj"
    _write_tetrahedron_obj(mesh_path)
    target_file = target_dir / "targets.json"
    target_file.write_text(
        json.dumps([{"name": "mesh", "shape": "mesh", "mesh_file": mesh_path.name, "size": [0.02, 0.02, 0.02]}])
    )
    target = load_target_specs(target_file)["mesh"]
    scene = build_scene(target, robot=robot)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH, "target_mesh") >= 0


def test_scene_preserves_robot_assets_that_rely_on_strippath(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    mesh_path = robot.asset_dir / "link.obj"
    _write_tetrahedron_obj(mesh_path)
    text = robot.xml_path.read_text().replace('autolimits="true"', 'autolimits="true" strippath="true"')
    text = text.replace(
        "<worldbody>",
        '<asset><mesh name="link_mesh" file="exported/stale/path/link.obj" inertia="shell"/></asset><worldbody>',
    )
    text = text.replace('<geom type="capsule" size="0.025 0.10" mass="1"/>', '<geom type="mesh" mesh="link_mesh" mass="1"/>')
    robot.xml_path.write_text(text)
    scene = build_scene(resolve_target("box"), robot=robot)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH, "link_mesh") >= 0


def test_scene_rejects_reserved_injected_name_collision_clearly(tmp_path) -> None:
    robot = _write_minimal_robot(tmp_path)
    robot.xml_path.write_text(robot.xml_path.read_text().replace('name="minimal_link"', 'name="target"'))
    with pytest.raises(RuntimeError, match="reserved injected body name 'target'"):
        build_scene(resolve_target("box"), robot=robot)


def test_scene_loads_mesh_part_in_compound_target(tmp_path) -> None:
    mesh_path = tmp_path / "part.obj"
    _write_tetrahedron_obj(mesh_path)
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            [
                {
                    "name": "compound-mesh",
                    "shape": "compound",
                    "size": [0.08, 0.08, 0.08],
                    "parts": [
                        {
                            "shape": "mesh",
                            "mesh_path": mesh_path.name,
                            "mesh_scale": [1.5, 1.5, 1.5],
                            "size": [0.03, 0.03, 0.03],
                            "offset": [0.01, 0.0, 0.0],
                        }
                    ],
                }
            ]
        )
    )
    scene = build_scene(load_target_specs(target_file)["compound-mesh"], robot=_write_minimal_robot(tmp_path))
    geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom_0")
    mesh_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_MESH, "target_part_mesh_0")
    assert scene.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert np.allclose(scene.model.mesh_scale[mesh_id], [1.5, 1.5, 1.5])


def test_target_file_rejects_invalid_mesh_scale_and_missing_file(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    _write_tetrahedron_obj(mesh_path)
    target_file = tmp_path / "targets.json"
    target = {
        "name": "bad-mesh",
        "shape": "mesh",
        "mesh_file": mesh_path.name,
        "scale": [1.0, 0.0, 1.0],
        "size": [0.1, 0.1, 0.1],
    }
    target_file.write_text(json.dumps({"targets": [target]}))
    with pytest.raises(ValueError, match="scale.*positive"):
        load_target_specs(target_file)
    target["mesh_file"] = "missing.obj"
    target["scale"] = [1.0, 1.0, 1.0]
    target_file.write_text(json.dumps({"targets": [target]}))
    with pytest.raises(ValueError, match="mesh file does not exist"):
        load_target_specs(target_file)


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


def test_custom_target_phrase_match_has_priority_over_builtin(tmp_path) -> None:
    target_file = tmp_path / "targets.json"
    target_file.write_text(
        json.dumps(
            [
                {
                    "name": "custom-cup",
                    "shape": "cylinder",
                    "size": [0.06, 0.06, 0.10],
                    "aliases": ["cup"],
                }
            ]
        )
    )
    assert resolve_target("please track the cup.", load_target_specs(target_file)).name == "custom-cup"


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
                        "shape": "torus",
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
