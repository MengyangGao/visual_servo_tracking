from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from textwrap import dedent
import xml.etree.ElementTree as ET

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
    ee_site_name: str | None = None
    ee_body_name: str | None = None
    target_body_name: str = "target"
    target_site_name: str = "target_site"
    camera_name: str = "servo_camera"


def _target_geom_xml(target: TargetSpec) -> str:
    shape = target.shape.strip().lower()
    if shape == "compound":
        if not target.parts:
            raise ValueError("compound target must contain at least one part")
        return "\n".join(_target_part_geom_xml(part, index, target.rgba) for index, part in enumerate(target.parts))
    if target.parts:
        raise ValueError(f"target shape '{shape}' cannot also contain parts; use shape 'compound'")
    mesh_name = "target_mesh" if shape == "mesh" else None
    if shape == "mesh" and target.mesh_path is None:
        raise ValueError("mesh target requires mesh_path")
    if shape != "mesh" and target.mesh_path is not None:
        raise ValueError(f"target shape '{shape}' cannot define mesh_path")
    return _primitive_geom_xml("target_geom", shape, target.size, (0.0, 0.0, 0.0), target.rgba, None, mesh_name)


def _target_part_geom_xml(part: TargetPart, index: int, fallback_rgba: tuple[float, float, float, float]) -> str:
    rgba = part.rgba or fallback_rgba
    shape = part.shape.strip().lower()
    mesh_name = f"target_part_mesh_{index}" if shape == "mesh" else None
    if shape == "mesh" and part.mesh_path is None:
        raise ValueError(f"mesh target part {index} requires mesh_path")
    if shape != "mesh" and part.mesh_path is not None:
        raise ValueError(f"target part {index} shape '{shape}' cannot define mesh_path")
    return _primitive_geom_xml(f"target_geom_{index}", shape, part.size, part.pos, rgba, part.quat, mesh_name)


def _primitive_geom_xml(
    name: str,
    shape: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    rgba_value: tuple[float, float, float, float],
    quat: tuple[float, float, float, float] | None,
    mesh_name: str | None = None,
) -> str:
    shape = shape.strip().lower()
    sx, sy, sz = _finite_tuple(size, 3, "target geometry size", positive=True)
    r, g, b, a = _finite_tuple(rgba_value, 4, "target geometry rgba")
    if any(value < 0.0 or value > 1.0 for value in (r, g, b, a)):
        raise ValueError("target geometry rgba values must be in [0, 1]")
    rgba = f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}"
    px, py, pz = _finite_tuple(pos, 3, "target geometry position")
    attrs = [
        f'name="{_xml_attr(name)}"',
        f'rgba="{rgba}"',
        f'pos="{px:.5f} {py:.5f} {pz:.5f}"',
        'contype="0"',
        'conaffinity="0"',
    ]
    if quat is not None:
        quat_values = np.asarray(_finite_tuple(quat, 4, "target geometry quaternion"), dtype=float)
        norm = float(np.linalg.norm(quat_values))
        if norm <= 1e-12:
            raise ValueError("target geometry quaternion must be non-zero")
        attrs.append('quat="' + " ".join(f"{v:.8f}" for v in quat_values / norm) + '"')
    if shape == "sphere":
        _require_equal_diameters((sx, sy, sz), "sphere")
        attrs.extend(['type="sphere"', f'size="{0.5 * sx:.5f}"'])
    elif shape == "cylinder":
        _require_equal_diameters((sx, sy), "cylinder")
        attrs.extend(['type="cylinder"', f'size="{0.5 * sx:.5f} {0.5 * sz:.5f}"'])
    elif shape == "capsule":
        _require_equal_diameters((sx, sy), "capsule")
        if sz <= sx:
            raise ValueError("capsule full height must be greater than its diameter")
        cylinder_half_length = 0.5 * (sz - sx)
        attrs.extend(['type="capsule"', f'size="{0.5 * sx:.5f} {cylinder_half_length:.5f}"'])
    elif shape == "box":
        attrs.extend(['type="box"', f'size="{0.5 * sx:.5f} {0.5 * sy:.5f} {0.5 * sz:.5f}"'])
    elif shape == "mesh":
        if not mesh_name:
            raise ValueError("mesh target geometry requires a mesh asset name")
        attrs.extend(['type="mesh"', f'mesh="{_xml_attr(mesh_name)}"'])
    else:
        raise ValueError(f"unsupported target geometry shape '{shape}'")
    return "<geom " + " ".join(attrs) + "/>"


def _target_mesh_assets_xml(target: TargetSpec) -> str:
    assets: list[str] = []
    shape = target.shape.strip().lower()
    if shape == "mesh":
        assets.append(_mesh_asset_xml("target_mesh", target.mesh_path, target.mesh_scale))
    for index, part in enumerate(target.parts):
        if part.shape.strip().lower() == "mesh":
            assets.append(_mesh_asset_xml(f"target_part_mesh_{index}", part.mesh_path, part.mesh_scale))
    return "\n".join(assets)


def _mesh_asset_xml(name: str, mesh_path: Path | None, scale: tuple[float, float, float]) -> str:
    path = _validated_mesh_path(mesh_path)
    sx, sy, sz = _finite_tuple(scale, 3, "mesh scale", positive=True)
    return (
        f'<mesh name="{_xml_attr(name)}" file="{_xml_attr(str(path))}" '
        f'scale="{sx:.8g} {sy:.8g} {sz:.8g}" inertia="shell"/>'
    )


def _validated_mesh_path(mesh_path: Path | None) -> Path:
    if mesh_path is None:
        raise ValueError("mesh target requires mesh_path")
    path = Path(mesh_path).expanduser().resolve()
    if path.suffix.lower() not in {".obj", ".stl"}:
        raise ValueError("target mesh must be an OBJ or STL file")
    if not path.is_file():
        raise ValueError(f"target mesh file does not exist: {path}")
    return path


def _finite_tuple(value, expected: int, field: str, positive: bool = False) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain {expected} numbers") from exc
    if len(values) != expected or not np.isfinite(values).all():
        raise ValueError(f"{field} must contain {expected} finite numbers")
    if positive and any(item <= 0.0 for item in values):
        raise ValueError(f"{field} values must be positive")
    return values


def _require_equal_diameters(values: tuple[float, ...], shape: str) -> None:
    if not np.allclose(values, values[0], rtol=1e-6, atol=1e-9):
        raise ValueError(f"{shape} requires equal circular diameters")


def _tracking_worldbody_xml(target: TargetSpec, camera: CameraConfig, target_pos: np.ndarray) -> str:
    camera_pos = np.array(camera.position, dtype=float)
    camera_lookat = np.array(camera.lookat, dtype=float)
    x_axis, y_axis = look_at_xyaxes(camera_pos, camera_lookat)
    xyaxes = " ".join(f"{v:.6f}" for v in np.r_[x_axis, y_axis])
    target_geom = _target_geom_xml(target)
    target_mesh_assets = _target_mesh_assets_xml(target)
    camera_name = _xml_attr(camera.name)
    return dedent(
        f"""
        <visual>
          <headlight diffuse="0.35 0.35 0.35" ambient="0.18 0.18 0.18" specular="0.25 0.25 0.25"/>
          <rgba haze="0.58 0.66 0.76 1"/>
          <global azimuth="120" elevation="-20" offwidth="{camera.width}" offheight="{camera.height}"/>
        </visual>

        <asset>
          {target_mesh_assets}
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
            <camera name="{camera_name}" pos="0 0 0" xyaxes="{xyaxes}" fovy="{camera.fovy_deg:.3f}"/>
          </body>

          <body name="target" mocap="true" pos="{target_pos[0]:.5f} {target_pos[1]:.5f} {target_pos[2]:.5f}">
            {target_geom}
            <site name="target_site" pos="0 0 0" size="0.012" rgba="1 1 1 1"/>
          </body>
        </worldbody>
        """
    ).strip()


def build_menagerie_mjcf(target: TargetSpec, camera: CameraConfig, robot: RobotSpec, target_pos: np.ndarray) -> str:
    try:
        root = ET.fromstring(robot.xml_path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise RuntimeError(f"robot '{robot.name}' MJCF is not valid XML: {exc}") from exc
    if root.tag != "mujoco":
        raise RuntimeError(f"robot '{robot.name}' MJCF root must be <mujoco>")
    if any(element.tag == "include" for element in root.iter()):
        raise RuntimeError(
            f"robot '{robot.name}' MJCF uses <include>, which cannot be resolved safely after in-memory scene injection; "
            "provide a self-contained MJCF file"
        )
    _name_declared_unnamed_actuators(root, robot)
    _reject_injected_name_collisions(root, target, camera, robot)

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    xml_directory = robot.xml_path.expanduser().resolve().parent
    absolute_asset_dir = robot.asset_dir.expanduser().resolve()
    original_asset_dir = compiler.get("assetdir")
    effective_texture_dir = compiler.get("texturedir") or original_asset_dir
    if effective_texture_dir:
        texture_path = Path(effective_texture_dir).expanduser()
        absolute_texture_dir = texture_path if texture_path.is_absolute() else xml_directory / texture_path
    else:
        # With no texturedir/assetdir, MuJoCo resolves texture files relative
        # to the main MJCF file, not relative to meshdir.
        absolute_texture_dir = xml_directory
    if compiler.get("strippath", "false").strip().lower() == "true":
        _absolutize_stripped_robot_assets(root, absolute_asset_dir, absolute_texture_dir)
    compiler.attrib.pop("assetdir", None)
    compiler.set("meshdir", str(absolute_asset_dir))
    compiler.set("texturedir", str(absolute_texture_dir.resolve()))
    # Scene injection adds absolute mesh paths and non-colliding visual target
    # geoms.  Preserving either source flag can strip those paths or delete the
    # target before rendering.
    compiler.set("strippath", "false")
    compiler.set("discardvisual", "false")

    fragment_text = f"<fragment>{_tracking_worldbody_xml(target, camera, target_pos)}</fragment>"
    try:
        fragment = ET.fromstring(fragment_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"generated tracking scene is not valid XML: {exc}") from exc
    root.extend(list(fragment))
    return ET.tostring(root, encoding="unicode")


def _absolutize_stripped_robot_assets(root: ET.Element, mesh_dir: Path, texture_dir: Path) -> None:
    """Preserve source strippath semantics before disabling it for injected assets."""

    mesh_file_attributes = {"file"}
    texture_file_attributes = {"file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"mesh", "hfield", "skin"}:
            base, attributes = mesh_dir, mesh_file_attributes
        elif tag == "texture":
            base, attributes = texture_dir, texture_file_attributes
        else:
            continue
        for attribute in attributes:
            value = element.get(attribute)
            if not value:
                continue
            # MuJoCo strippath removes both POSIX and exported Windows-style
            # directory components.  Recreate that result as an absolute path.
            basename = Path(value.replace("\\", "/")).name
            element.set(attribute, str((base / basename).expanduser().resolve()))


def _name_declared_unnamed_actuators(root: ET.Element, robot: RobotSpec) -> None:
    """Give deterministic descriptor names to otherwise unnamed joint actuators."""

    actuator_section = root.find("actuator")
    if actuator_section is None:
        return
    actuators = list(actuator_section)
    existing_names = {element.get("name") for element in actuators if element.get("name")}
    for joint_name, actuator_name in zip(robot.joint_names, robot.actuator_names):
        if actuator_name in existing_names:
            continue
        candidates = [
            element
            for element in actuators
            if not element.get("name")
            and (element.get("joint") == joint_name or element.get("jointinparent") == joint_name)
        ]
        if len(candidates) == 1:
            candidates[0].set("name", actuator_name)
            existing_names.add(actuator_name)


def _reject_injected_name_collisions(
    root: ET.Element,
    target: TargetSpec,
    camera: CameraConfig,
    robot: RobotSpec,
) -> None:
    geom_names = {"servo_floor", "servo_table"}
    mesh_names: set[str] = set()
    if target.shape.strip().lower() == "compound":
        geom_names.update(f"target_geom_{index}" for index in range(len(target.parts)))
    else:
        geom_names.add("target_geom")
    if target.shape.strip().lower() == "mesh":
        mesh_names.add("target_mesh")
    mesh_names.update(
        f"target_part_mesh_{index}"
        for index, part in enumerate(target.parts)
        if part.shape.strip().lower() == "mesh"
    )
    reserved = {
        "body": {"camera_marker", "target"},
        "site": {"target_site"},
        "camera": {camera.name},
        "geom": geom_names,
        "light": {"servo_key", "servo_fill", "servo_rim"},
        "texture": {"servo_groundplane_tex"},
        "material": {"servo_groundplane", "servo_table_mat"},
        "mesh": mesh_names,
    }
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        name = element.get("name")
        if name and name in reserved.get(tag, set()):
            raise RuntimeError(
                f"robot '{robot.name}' MJCF uses reserved injected {tag} name '{name}'; rename it in the robot model"
            )


def _xml_attr(value: str) -> str:
    return escape(value, quote=True)


def build_scene(
    target: TargetSpec,
    camera: CameraConfig | None = None,
    robot: RobotSpec | str = "panda",
    target_position: np.ndarray | None = None,
) -> Scene:
    cam = camera or CameraConfig()
    robot_spec = resolve_robot(robot) if isinstance(robot, str) else robot
    if target_position is None and robot_spec.default_target_position is not None:
        initial_target_position = robot_spec.default_target_position
    else:
        initial_target_position = base_position(target) if target_position is None else target_position
    target_pos = np.asarray(initial_target_position, dtype=float).reshape(3)
    if not np.isfinite(target_pos).all():
        raise ValueError("target_position must contain three finite values")
    if not robot_spec.xml_path.is_file():
        raise FileNotFoundError(f"robot '{robot_spec.name}' MJCF file not found: {robot_spec.xml_path}")
    if not robot_spec.asset_dir.is_dir():
        raise FileNotFoundError(f"robot '{robot_spec.name}' asset directory not found: {robot_spec.asset_dir}")
    source = "menagerie" if "mujoco_menagerie" in robot_spec.xml_path.parts else "external"
    model = mujoco.MjModel.from_xml_string(build_menagerie_mjcf(target, cam, robot_spec, target_pos))
    home = np.array(robot_spec.home_qpos, dtype=float)
    if home.shape != (len(robot_spec.joint_names),) or not np.isfinite(home).all():
        raise RuntimeError(f"robot '{robot_spec.name}' home_qpos must have {len(robot_spec.joint_names)} finite values")
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot_spec.joint_names]
    for i, joint_id in enumerate(joint_ids):
        if joint_id < 0:
            raise RuntimeError(f"robot '{robot_spec.name}' joint '{robot_spec.joint_names[i]}' not found")
        if model.jnt_type[joint_id] not in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            raise RuntimeError(f"robot '{robot_spec.name}' joint '{robot_spec.joint_names[i]}' must be a scalar hinge or slide joint")
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if home[i] < lower or home[i] > upper:
                raise RuntimeError(
                    f"robot '{robot_spec.name}' home_qpos for joint '{robot_spec.joint_names[i]}' "
                    f"must be within [{lower:g}, {upper:g}]"
                )
        data.qpos[model.jnt_qposadr[joint_id]] = home[i]
    _set_robot_actuator_ctrl(model, data, robot_spec, home)
    set_target_position(model, data, target_pos)
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
        ee_site_name=robot_spec.ee_frame_name if robot_spec.ee_frame_type == "site" else None,
        ee_body_name=robot_spec.ee_frame_name if robot_spec.ee_frame_type in {"body", "body_point"} else None,
        camera_name=cam.name,
    )


def _set_robot_actuator_ctrl(model: mujoco.MjModel, data: mujoco.MjData, robot: RobotSpec, qpos_command: np.ndarray) -> None:
    for i, joint_name in enumerate(robot.joint_names):
        actuator_id = resolve_joint_actuator(model, robot, joint_name, i)
        _validate_position_actuator(model, actuator_id, robot.name, joint_name)
        gear = float(model.actuator_gear[actuator_id, 0])
        control = gear * qpos_command[i]
        if model.actuator_ctrllimited[actuator_id]:
            lower, upper = model.actuator_ctrlrange[actuator_id]
            if control < lower or control > upper:
                actuator_name = robot.actuator_names[i]
                raise RuntimeError(
                    f"robot '{robot.name}' home_qpos for joint '{joint_name}' maps to control {control:g}, "
                    f"outside actuator '{actuator_name}' range [{lower:g}, {upper:g}]"
                )
        _write_ctrl(model, data, actuator_id, control)
    for actuator_name, value in robot.passive_actuator_ctrl:
        actuator_id = resolve_passive_actuator(model, robot, actuator_name, value)
        _write_ctrl(model, data, actuator_id, value)


def resolve_joint_actuator(model: mujoco.MjModel, robot: RobotSpec, joint_name: str, index: int) -> int:
    if index >= len(robot.actuator_names):
        raise RuntimeError(f"robot '{robot.name}' has no declared actuator for joint '{joint_name}'")
    actuator_name = robot.actuator_names[index]
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise RuntimeError(f"robot '{robot.name}' declared actuator '{actuator_name}' not found")
    supported_transmissions = {mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT}
    if model.actuator_trntype[actuator_id] not in supported_transmissions:
        raise RuntimeError(
            f"robot '{robot.name}' actuator '{actuator_name}' for joint '{joint_name}' must use a joint transmission"
        )
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0 or int(model.actuator_trnid[actuator_id, 0]) != joint_id:
        raise RuntimeError(
            f"robot '{robot.name}' actuator '{actuator_name}' does not transmit joint '{joint_name}'"
        )
    return int(actuator_id)


def resolve_passive_actuator(model: mujoco.MjModel, robot: RobotSpec, actuator_name: str, value: float) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise RuntimeError(f"robot '{robot.name}' passive actuator '{actuator_name}' not found")
    control = float(value)
    if not np.isfinite(control):
        raise RuntimeError(f"robot '{robot.name}' passive actuator '{actuator_name}' control must be finite")
    if model.actuator_ctrllimited[actuator_id]:
        lower, upper = model.actuator_ctrlrange[actuator_id]
        if control < lower or control > upper:
            raise RuntimeError(
                f"robot '{robot.name}' passive actuator '{actuator_name}' control {control:g} "
                f"is outside range [{lower:g}, {upper:g}]"
            )
    return int(actuator_id)


def _validate_position_actuator(
    model: mujoco.MjModel,
    actuator_id: int,
    robot_name: str,
    joint_name: str,
) -> None:
    gain = float(model.actuator_gainprm[actuator_id, 0])
    position_bias = float(model.actuator_biasprm[actuator_id, 1])
    gear = float(model.actuator_gear[actuator_id, 0])
    is_affine = model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
    is_position_servo = is_affine and gain > 0.0 and np.isclose(position_bias, -gain, rtol=1e-4, atol=1e-8)
    if not is_position_servo or not np.isfinite(gear) or abs(gear) < 1e-12:
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or f"#{actuator_id}"
        raise RuntimeError(
            f"robot '{robot_name}' actuator '{actuator_name}' for joint '{joint_name}' must be a "
            "non-degenerate MuJoCo position servo; torque and velocity actuators are not supported"
        )


def _write_ctrl(model: mujoco.MjModel, data: mujoco.MjData, actuator_id: int, value: float) -> None:
    ctrl = float(value)
    if model.actuator_ctrllimited[actuator_id]:
        lo, hi = model.actuator_ctrlrange[actuator_id]
        ctrl = float(np.clip(ctrl, lo, hi))
    data.ctrl[actuator_id] = ctrl


def set_target_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    body_name: str = "target",
) -> None:
    target_position = np.asarray(position, dtype=float).reshape(3)
    if not np.isfinite(target_position).all():
        raise ValueError("target position must contain three finite values")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"target body '{body_name}' missing")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError("target body is not mocap-controlled")
    data.mocap_pos[mocap_id] = target_position


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


def camera_position(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise KeyError(f"camera '{camera_name}' missing")
    return np.array(data.cam_xpos[camera_id], dtype=float)


def joint_positions(model: mujoco.MjModel, data: mujoco.MjData, joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    positions: list[float] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"joint '{joint_name}' missing")
        if model.jnt_type[joint_id] not in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            raise ValueError(f"joint '{joint_name}' is not a scalar hinge or slide joint")
        positions.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return np.array(positions, dtype=float)


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
