from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

import mujoco
import numpy as np

from .config import CameraConfig, default_home_qpos
from .math_utils import look_at_xyaxes
from .targets import TargetSpec, base_position


@dataclass(slots=True)
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    target: TargetSpec
    ee_site_name: str = "ee_site"
    target_body_name: str = "target"
    target_site_name: str = "target_site"
    camera_name: str = "servo_camera"


def _target_geom_xml(target: TargetSpec) -> str:
    sx, sy, sz = target.size
    r, g, b, a = target.rgba
    rgba = f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}"
    if target.shape == "sphere":
        return f'<geom name="target_geom" type="sphere" size="{0.5 * max(target.size):.5f}" rgba="{rgba}"/>'
    if target.shape == "cylinder":
        return f'<geom name="target_geom" type="cylinder" size="{0.5 * sx:.5f} {0.5 * sz:.5f}" rgba="{rgba}"/>'
    return f'<geom name="target_geom" type="box" size="{0.5 * sx:.5f} {0.5 * sy:.5f} {0.5 * sz:.5f}" rgba="{rgba}"/>'


def build_mjcf(target: TargetSpec, camera: CameraConfig) -> str:
    target_pos = base_position(target)
    camera_pos = np.array(camera.position, dtype=float)
    camera_lookat = np.array(camera.lookat, dtype=float)
    x_axis, y_axis = look_at_xyaxes(camera_pos, camera_lookat)
    xyaxes = " ".join(f"{v:.6f}" for v in np.r_[x_axis, y_axis])
    home = " ".join(f"{v:.6f}" for v in default_home_qpos())
    target_geom = _target_geom_xml(target)
    return dedent(
        f"""
        <mujoco model="visual_servo_tracking">
          <compiler angle="radian" autolimits="true"/>
          <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

          <default>
            <default class="arm">
              <joint type="hinge" damping="1.2" armature="0.04" limited="true"/>
              <geom contype="0" conaffinity="0" density="300" rgba="0.72 0.74 0.78 1"/>
              <position kp="140" kv="24" ctrllimited="true"/>
            </default>
          </default>

          <asset>
            <material name="mat_floor" rgba="0.17 0.18 0.19 1"/>
            <material name="mat_dark" rgba="0.08 0.09 0.10 1"/>
            <material name="mat_arm" rgba="0.82 0.84 0.88 1"/>
          </asset>

          <worldbody>
            <light name="key" pos="0.1 -0.8 1.8" dir="-0.2 0.5 -1" diffuse="0.9 0.9 0.86"/>
            <light name="fill" pos="-0.8 0.4 1.2" dir="0.4 -0.2 -1" diffuse="0.35 0.42 0.55"/>
            <geom name="floor" type="plane" size="2.0 2.0 0.05" material="mat_floor"/>
            <geom name="table" type="box" pos="0.45 0 0.18" size="0.45 0.38 0.035" rgba="0.34 0.32 0.28 1"/>

            <body name="camera_marker" pos="{camera_pos[0]:.5f} {camera_pos[1]:.5f} {camera_pos[2]:.5f}">
              <geom type="box" size="0.045 0.030 0.025" rgba="0.15 0.55 0.95 0.9"/>
              <camera name="{camera.name}" pos="0 0 0" xyaxes="{xyaxes}" fovy="{camera.fovy_deg:.3f}"/>
            </body>

            <body name="target" mocap="true" pos="{target_pos[0]:.5f} {target_pos[1]:.5f} {target_pos[2]:.5f}">
              {target_geom}
              <site name="target_site" pos="0 0 0" size="0.012" rgba="1 1 1 1"/>
            </body>

            <body name="panda_like_base" pos="0 0 0.22">
              <geom type="cylinder" size="0.09 0.055" material="mat_dark"/>
              <body name="link1" pos="0 0 0.08">
                <joint name="joint1" class="arm" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.24" size="0.038" material="mat_arm"/>
                <body name="link2" pos="0 0 0.24">
                  <joint name="joint2" class="arm" axis="0 1 0" range="-1.7628 1.7628"/>
                  <geom type="capsule" fromto="0 0 0 0.20 0 0" size="0.034" material="mat_arm"/>
                  <body name="link3" pos="0.20 0 0">
                    <joint name="joint3" class="arm" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.20" size="0.032" material="mat_arm"/>
                    <body name="link4" pos="0 0 0.20">
                      <joint name="joint4" class="arm" axis="0 1 0" range="-3.0718 -0.0698"/>
                      <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.030" material="mat_arm"/>
                      <body name="link5" pos="0.22 0 0">
                        <joint name="joint5" class="arm" axis="0 0 1" range="-2.8973 2.8973"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.17" size="0.027" material="mat_arm"/>
                        <body name="link6" pos="0 0 0.17">
                          <joint name="joint6" class="arm" axis="0 1 0" range="-0.0175 3.7525"/>
                          <geom type="capsule" fromto="0 0 0 0.16 0 0" size="0.024" material="mat_arm"/>
                          <body name="link7" pos="0.16 0 0">
                            <joint name="joint7" class="arm" axis="0 0 1" range="-2.8973 2.8973"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.10" size="0.020" material="mat_arm"/>
                            <body name="hand" pos="0 0 0.10">
                              <geom type="box" size="0.035 0.028 0.030" material="mat_dark"/>
                              <geom type="capsule" fromto="0.035 -0.026 0 0.095 -0.026 0" size="0.009" rgba="0.08 0.08 0.08 1"/>
                              <geom type="capsule" fromto="0.035 0.026 0 0.095 0.026 0" size="0.009" rgba="0.08 0.08 0.08 1"/>
                              <site name="ee_site" pos="0.105 0 0" size="0.014" rgba="0.05 0.9 1.0 1"/>
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
            <position name="act1" joint="joint1" class="arm" ctrlrange="-2.8973 2.8973"/>
            <position name="act2" joint="joint2" class="arm" ctrlrange="-1.7628 1.7628"/>
            <position name="act3" joint="joint3" class="arm" ctrlrange="-2.8973 2.8973"/>
            <position name="act4" joint="joint4" class="arm" ctrlrange="-3.0718 -0.0698"/>
            <position name="act5" joint="joint5" class="arm" ctrlrange="-2.8973 2.8973"/>
            <position name="act6" joint="joint6" class="arm" ctrlrange="-0.0175 3.7525"/>
            <position name="act7" joint="joint7" class="arm" ctrlrange="-2.8973 2.8973"/>
          </actuator>

          <keyframe>
            <key name="home" qpos="{home}" ctrl="{home}"/>
          </keyframe>
        </mujoco>
        """
    ).strip()


def build_scene(target: TargetSpec, camera: CameraConfig | None = None) -> Scene:
    cam = camera or CameraConfig()
    model = mujoco.MjModel.from_xml_string(build_mjcf(target, cam))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = default_home_qpos()
    set_target_position(model, data, base_position(target))
    mujoco.mj_forward(model, data)
    return Scene(model=model, data=data, target=target, camera_name=cam.name)


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
