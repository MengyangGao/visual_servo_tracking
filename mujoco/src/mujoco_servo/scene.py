from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from textwrap import dedent

import mujoco
import numpy as np

from .config import canonical_camera_pose, default_camera_intrinsics, lookup_target_prototype, target_world_position
from .geometry import normalize
from .robot import RobotSpec
from .types import CameraIntrinsics, CameraPose, SceneBundle, TargetPrototype


def _target_body_xml(proto: TargetPrototype, target_name: str = "target") -> str:
    size = proto.size_m
    r, g, b, a = proto.rgba
    if proto.primitive == "sphere":
        geom = f'<geom name="{target_name}_geom" type="sphere" size="{size[0] / 2.0:.4f}" rgba="{r} {g} {b} {a}"/>'
    elif proto.primitive == "cylinder":
        geom = f'<geom name="{target_name}_geom" type="cylinder" size="{size[0] / 2.0:.4f} {size[2] / 2.0:.4f}" rgba="{r} {g} {b} {a}"/>'
    else:
        geom = f'<geom name="{target_name}_geom" type="box" size="{size[0] / 2.0:.4f} {size[1] / 2.0:.4f} {size[2] / 2.0:.4f}" rgba="{r} {g} {b} {a}"/>'
    return dedent(
        f"""
        <body name="{target_name}" mocap="true" pos="{target_world_position(proto.name)[0]:.4f} {target_world_position(proto.name)[1]:.4f} {target_world_position(proto.name)[2]:.4f}">
          {geom}
          <site name="{target_name}_site" pos="0 0 0" size="0.01" rgba="1 1 1 1"/>
        </body>
        """
    ).strip()


def _camera_marker_xml() -> str:
    return dedent(
        """
        <body name="vision_camera" mocap="true" pos="0.18 -0.95 0.82">
          <geom type="box" size="0.058 0.038 0.038" rgba="0.20 0.62 1.00 0.90"/>
          <site name="vision_camera_site" pos="0 0 0" size="0.01" rgba="1 1 1 1"/>
        </body>
        """
    ).strip()


def _target_marker_xml(proto: TargetPrototype) -> str:
    size = proto.size_m
    return dedent(
        f"""
        <body name="vision_target" mocap="true" pos="{target_world_position(proto.name)[0]:.4f} {target_world_position(proto.name)[1]:.4f} {target_world_position(proto.name)[2]:.4f}">
          <geom type="sphere" size="{max(size) * 0.30:.4f}" rgba="1.00 0.20 0.20 0.94"/>
          <site name="vision_target_site" pos="0 0 0" size="0.008" rgba="1 1 1 1"/>
        </body>
        """
    ).strip()


def _ee_marker_xml() -> str:
    return dedent(
        """
        <body name="vision_ee" mocap="true" pos="0.30 -0.10 0.45">
          <geom type="sphere" size="0.030" rgba="0.10 0.80 1.00 0.98"/>
          <geom type="box" size="0.020 0.014 0.008" pos="0.032 0 0" rgba="0.05 0.35 1.00 0.96"/>
          <site name="vision_ee_site" pos="0 0 0" size="0.008" rgba="1 1 1 1"/>
        </body>
        """
    ).strip()


def _inject_target(xml_text: str, proto: TargetPrototype, target_name: str = "target") -> str:
    marker = "</worldbody>"
    if marker not in xml_text:
        raise ValueError("scene XML is missing a worldbody closing tag")
    marker_block = "\n  ".join([_target_body_xml(proto, target_name), _camera_marker_xml(), _target_marker_xml(proto), _ee_marker_xml()])
    return xml_text.replace(marker, f"{marker_block}\n  {marker}", 1)


def _fallback_scene_xml(proto: TargetPrototype) -> str:
    size = proto.size_m
    target = _target_body_xml(proto)
    return dedent(
        f"""
        <mujoco model="panda_lite">
          <compiler angle="radian" autolimits="true"/>
          <option integrator="implicitfast" timestep="0.002"/>
          <default>
            <default class="arm">
              <joint type="hinge" axis="0 0 1" damping="1" armature="0.05" range="-2.8973 2.8973"/>
              <geom contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
              <position kp="40"/>
            </default>
          </default>
          <worldbody>
            <light pos="0 0 2" mode="trackcom"/>
            <geom name="floor" type="plane" size="0 0 0.05" rgba="0.15 0.15 0.15 1"/>
            <body name="base" pos="0 0 0.0">
              <geom type="cylinder" size="0.08 0.05" rgba="0.2 0.2 0.2 1"/>
              <body name="link1" pos="0 0 0.15">
                <joint name="joint1" class="arm"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.03" rgba="0.85 0.85 0.9 1"/>
                <body name="link2" pos="0 0 0.18">
                  <joint name="joint2" class="arm" axis="0 1 0"/>
                  <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.028" rgba="0.8 0.8 0.85 1"/>
                  <body name="link3" pos="0.18 0 0">
                    <joint name="joint3" class="arm"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.025" rgba="0.82 0.82 0.88 1"/>
                    <body name="link4" pos="0 0 0.16">
                      <joint name="joint4" class="arm" axis="0 1 0"/>
                      <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.023" rgba="0.8 0.8 0.85 1"/>
                      <body name="link5" pos="0.14 0 0">
                        <joint name="joint5" class="arm"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.02" rgba="0.83 0.83 0.88 1"/>
                        <body name="link6" pos="0 0 0.14">
                          <joint name="joint6" class="arm" axis="0 1 0"/>
                          <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.018" rgba="0.8 0.8 0.85 1"/>
                          <body name="link7" pos="0.12 0 0">
                            <joint name="joint7" class="arm"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.10" size="0.016" rgba="0.82 0.82 0.88 1"/>
                            <body name="hand" pos="0 0 0.10">
                              <geom type="box" size="0.03 0.03 0.05" rgba="0.2 0.2 0.2 1"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="actuator1" joint="joint1" kp="40"/>
            <position name="actuator2" joint="joint2" kp="40"/>
            <position name="actuator3" joint="joint3" kp="35"/>
            <position name="actuator4" joint="joint4" kp="30"/>
            <position name="actuator5" joint="joint5" kp="25"/>
            <position name="actuator6" joint="joint6" kp="20"/>
            <position name="actuator7" joint="joint7" kp="18"/>
          </actuator>
          <keyframe>
            <key name="home" qpos="0 0 0 -1.57 0 1.57 -0.78"/>
          </keyframe>
        </mujoco>
        """
    ).replace(
        "</worldbody>",
        f"{target}\n          {_camera_marker_xml()}\n          {_target_marker_xml(proto)}\n          {_ee_marker_xml()}\n          </worldbody>",
    )


def build_scene_xml(robot_spec: RobotSpec, prompt: str) -> str:
    proto = lookup_target_prototype(prompt)
    if robot_spec.scene_xml_path is not None and robot_spec.scene_xml_path.exists():
        text = robot_spec.scene_xml_path.read_text()
        return _inject_target(text, proto)
    return _fallback_scene_xml(proto)


def build_scene_bundle(robot_spec: RobotSpec, prompt: str, image_width: int, image_height: int) -> SceneBundle:
    xml_text = build_scene_xml(robot_spec, prompt)
    if robot_spec.scene_xml_path is not None and robot_spec.scene_xml_path.exists():
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_tracking_scene.xml",
            dir=robot_spec.scene_xml_path.parent,
            delete=False,
        ) as handle:
            handle.write(xml_text)
            temp_path = Path(handle.name)
        try:
            model = mujoco.MjModel.from_xml_path(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    intrinsics = default_camera_intrinsics(image_width, image_height)
    camera_pose = canonical_camera_pose()
    return SceneBundle(
        model=model,
        data=data,
        target_proto=lookup_target_prototype(prompt),
        ee_body_name=robot_spec.ee_body_name,
        actuator_names=robot_spec.actuator_names,
        camera_intrinsics=intrinsics,
        camera_pose=camera_pose,
    )


def set_mocap_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    position: np.ndarray,
    rotation_world_from_body: np.ndarray | None = None,
) -> bool:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        return False
    data.mocap_pos[mocap_id] = np.asarray(position, dtype=float).reshape(3)
    if rotation_world_from_body is not None:
        from .geometry import rotation_matrix_to_quaternion_wxyz

        data.mocap_quat[mocap_id] = rotation_matrix_to_quaternion_wxyz(np.asarray(rotation_world_from_body, dtype=float).reshape(3, 3))
    return True


def target_body_id(model: mujoco.MjModel, body_name: str = "target") -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"body '{body_name}' not found")
    return body_id


def body_pose_world(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"body '{body_name}' not found")
    pos = np.array(data.xpos[body_id], dtype=float)
    rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    return pos, rot
