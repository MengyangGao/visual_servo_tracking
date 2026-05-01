from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

import mujoco
import numpy as np

from .config import CameraConfig, RobotSpec, TargetPart, resolve_robot
from .math_utils import look_at_xyaxes
from .targets import TargetSpec, base_position


@dataclass(slots=True)
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    target: TargetSpec
    robot: RobotSpec
    source: str
    ee_frame_name: str
    ee_frame_type: str
    ee_frame_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_site_name: str = "ee_site"
    ee_body_name: str = "hand"
    target_body_name: str = "target"
    target_site_name: str = "target_site"
    camera_name: str = "servo_camera"


def _target_geom_xml(target: TargetSpec) -> str:
    if target.parts:
        return "\n".join(_target_part_geom_xml(part, index, target.rgba) for index, part in enumerate(target.parts))
    return _primitive_geom_xml("target_geom", target.shape, target.size, (0.0, 0.0, 0.0), target.rgba, None)


def _target_part_geom_xml(part: TargetPart, index: int, fallback_rgba: tuple[float, float, float, float]) -> str:
    rgba = part.rgba or fallback_rgba
    return _primitive_geom_xml(f"target_geom_{index}", part.shape, part.size, part.pos, rgba, part.quat)


def _primitive_geom_xml(
    name: str,
    shape: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    rgba_value: tuple[float, float, float, float],
    quat: tuple[float, float, float, float] | None,
) -> str:
    sx, sy, sz = size
    r, g, b, a = rgba_value
    rgba = f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}"
    px, py, pz = pos
    attrs = [f'name="{name}"', f'rgba="{rgba}"', f'pos="{px:.5f} {py:.5f} {pz:.5f}"']
    if quat is not None:
        attrs.append('quat="' + " ".join(f"{v:.5f}" for v in quat) + '"')
    if shape == "sphere":
        attrs.extend(['type="sphere"', f'size="{0.5 * max(size):.5f}"'])
    elif shape == "cylinder":
        attrs.extend(['type="cylinder"', f'size="{0.5 * sx:.5f} {0.5 * sz:.5f}"'])
    elif shape == "capsule":
        attrs.extend(['type="capsule"', f'size="{0.5 * sx:.5f} {0.5 * sz:.5f}"'])
    else:
        attrs.extend(['type="box"', f'size="{0.5 * sx:.5f} {0.5 * sy:.5f} {0.5 * sz:.5f}"'])
    return "<geom " + " ".join(attrs) + "/>"


def _tracking_worldbody_xml(target: TargetSpec, camera: CameraConfig) -> str:
    target_pos = base_position(target)
    camera_pos = np.array(camera.position, dtype=float)
    camera_lookat = np.array(camera.lookat, dtype=float)
    x_axis, y_axis = look_at_xyaxes(camera_pos, camera_lookat)
    xyaxes = " ".join(f"{v:.6f}" for v in np.r_[x_axis, y_axis])
    target_geom = _target_geom_xml(target)
    return dedent(
        f"""
        <visual>
          <headlight diffuse="0.35 0.35 0.35" ambient="0.18 0.18 0.18" specular="0.25 0.25 0.25"/>
          <rgba haze="0.58 0.66 0.76 1"/>
          <global azimuth="120" elevation="-20"/>
        </visual>

        <asset>
          <texture type="skybox" builtin="gradient" rgb1="0.55 0.68 0.86" rgb2="0.08 0.11 0.16" width="512" height="3072"/>
          <texture type="2d" name="servo_groundplane_tex" builtin="checker" mark="edge" rgb1="0.33 0.36 0.38" rgb2="0.19 0.21 0.23" markrgb="0.75 0.75 0.72" width="300" height="300"/>
          <material name="servo_groundplane" texture="servo_groundplane_tex" texuniform="true" texrepeat="5 5" reflectance="0.18"/>
          <material name="servo_table_mat" rgba="0.46 0.42 0.34 1" specular="0.25" shininess="0.35"/>
        </asset>

        <worldbody>
          <light name="servo_key" pos="0.15 -0.8 1.8" dir="-0.2 0.5 -1" directional="true" diffuse="0.85 0.82 0.74" specular="0.25 0.25 0.22"/>
          <light name="servo_fill" pos="-0.8 0.55 1.25" dir="0.6 -0.25 -1" directional="true" diffuse="0.35 0.43 0.55" specular="0.08 0.10 0.12"/>
          <light name="servo_rim" pos="0.9 0.65 1.1" dir="-0.7 -0.35 -0.8" directional="true" diffuse="0.28 0.24 0.20" specular="0.15 0.12 0.10"/>
          <geom name="servo_floor" size="0 0 0.05" type="plane" material="servo_groundplane"/>
          <geom name="servo_table" type="box" pos="0.48 0 0.18" size="0.45 0.38 0.035" material="servo_table_mat"/>

          <body name="camera_marker" pos="{camera_pos[0]:.5f} {camera_pos[1]:.5f} {camera_pos[2]:.5f}">
            <geom type="box" size="0.045 0.030 0.025" rgba="0.15 0.55 0.95 0.9" contype="0" conaffinity="0"/>
            <camera name="{camera.name}" pos="0 0 0" xyaxes="{xyaxes}" fovy="{camera.fovy_deg:.3f}"/>
          </body>

          <body name="target" mocap="true" pos="{target_pos[0]:.5f} {target_pos[1]:.5f} {target_pos[2]:.5f}">
            {target_geom}
            <site name="target_site" pos="0 0 0" size="0.012" rgba="1 1 1 1"/>
          </body>
        </worldbody>
        """
    ).strip()


def build_menagerie_mjcf(target: TargetSpec, camera: CameraConfig, robot: RobotSpec) -> str:
    text = robot.xml_path.read_text()
    text = text.replace('meshdir="assets"', f'meshdir="{robot.asset_dir}"')
    insertion = "\n" + _tracking_worldbody_xml(target, camera) + "\n"
    return text.replace("</mujoco>", f"{insertion}</mujoco>", 1)


def build_scene(target: TargetSpec, camera: CameraConfig | None = None, robot: RobotSpec | str = "panda") -> Scene:
    cam = camera or CameraConfig()
    robot_spec = resolve_robot(robot) if isinstance(robot, str) else robot
    if not robot_spec.xml_path.exists() or not robot_spec.asset_dir.exists():
        raise FileNotFoundError(
            f"MuJoCo Menagerie assets for robot '{robot_spec.name}' are required. Run "
            "`git submodule update --init --recursive mujoco/vendor/mujoco_menagerie`."
        )
    source = "menagerie"
    model = mujoco.MjModel.from_xml_string(build_menagerie_mjcf(target, cam, robot_spec))
    home = np.array(robot_spec.home_qpos, dtype=float)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot_spec.joint_names]
    for i, joint_id in enumerate(joint_ids):
        if joint_id < 0:
            raise RuntimeError(f"robot '{robot_spec.name}' joint '{robot_spec.joint_names[i]}' not found")
        data.qpos[model.jnt_qposadr[joint_id]] = home[i]
    _set_robot_actuator_ctrl(model, data, robot_spec, home)
    set_target_position(model, data, base_position(target))
    mujoco.mj_forward(model, data)
    return Scene(
        model=model,
        data=data,
        target=target,
        robot=robot_spec,
        source=source,
        ee_frame_name=robot_spec.ee_frame_name,
        ee_frame_type=robot_spec.ee_frame_type,
        ee_frame_offset=robot_spec.ee_frame_offset,
        camera_name=cam.name,
    )


def _set_robot_actuator_ctrl(model: mujoco.MjModel, data: mujoco.MjData, robot: RobotSpec, qpos_command: np.ndarray) -> None:
    for i, joint_name in enumerate(robot.joint_names):
        actuator_id = _actuator_id_for_joint(model, robot, joint_name, i)
        data.ctrl[actuator_id] = qpos_command[i]
    for actuator_name, value in robot.passive_actuator_ctrl:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise RuntimeError(f"robot '{robot.name}' passive actuator '{actuator_name}' not found")
        data.ctrl[actuator_id] = float(value)


def _actuator_id_for_joint(model: mujoco.MjModel, robot: RobotSpec, joint_name: str, index: int) -> int:
    if index < len(robot.actuator_names):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, robot.actuator_names[index])
        if actuator_id >= 0:
            return int(actuator_id)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id:
            return actuator_id
    raise RuntimeError(f"robot '{robot.name}' actuator for joint '{joint_name}' not found")


def set_target_position(model: mujoco.MjModel, data: mujoco.MjData, position: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if body_id < 0:
        raise KeyError("target body missing")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError("target body is not mocap-controlled")
    data.mocap_pos[mocap_id] = np.asarray(position, dtype=float).reshape(3)


def site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise KeyError(f"site '{site_name}' missing")
    return np.array(data.site_xpos[site_id], dtype=float)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"body '{body_name}' missing")
    return np.array(data.xpos[body_id], dtype=float)


def frame_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_type: str,
    frame_name: str,
    frame_offset: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    if frame_type == "site":
        return site_position(model, data, frame_name)
    if frame_type in {"body", "body_point"}:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if body_id < 0:
            raise KeyError(f"body '{frame_name}' missing")
        position = np.array(data.xpos[body_id], dtype=float)
        if frame_type == "body_point":
            offset = np.asarray(frame_offset if frame_offset is not None else (0.0, 0.0, 0.0), dtype=float).reshape(3)
            rotation = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
            position = position + rotation @ offset
        return position
    raise ValueError(f"unknown frame type '{frame_type}'")
