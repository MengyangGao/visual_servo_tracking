# MuJoCo assets and descriptors

## Menagerie robots

The repository pins Google DeepMind MuJoCo Menagerie as `mujoco/vendor/mujoco_menagerie`.

| Robot | MJCF | Asset directory | Grasp attachment |
| --- | --- | --- | --- |
| Franka Panda (`panda`) | `franka_emika_panda/panda.xml` | `franka_emika_panda/assets/` | `hand`, with official coupled-finger actuator metadata |
| Universal Robots UR5e (`ur5e`) | `universal_robots_ur5e/ur5e.xml` | `universal_robots_ur5e/assets/` | `wrist_3_link` weld/suction abstraction |
| UFactory Lite6 (`lite6`) | `ufactory_lite6/lite6.xml` | `ufactory_lite6/assets/` | `link6` weld/suction abstraction |

Initialize and inspect the pinned checkout with:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie
git submodule status mujoco/vendor/mujoco_menagerie
```

The source checkout finds that submodule automatically. Wheels contain code only, so an installed wheel needs `MUJOCO_MENAGERIE_PATH` set before `mujoco_servo` is imported.

Preserve Menagerie's top-level `LICENSE`, `CITATION.cff`, and model-specific attribution when redistributing selected assets.

## Robot descriptor version 1

`--robot-file` accepts one object, a list, or a wrapper:

```json
{
  "schema_version": 1,
  "robots": [
    {
      "schema_version": 1,
      "name": "my-arm",
      "xml_path": "robot.xml",
      "asset_dir": "assets",
      "joint_names": ["joint1", "joint2"],
      "actuator_names": ["joint1_position", "joint2_position"],
      "home_qpos": [0.0, 0.5],
      "ee_frame": {
        "name": "tool_site",
        "type": "site",
        "offset": [0.0, 0.0, 0.0]
      },
      "tool_axis": [0.0, 0.0, 1.0],
      "base_position": [0.0, 0.0, 0.0],
      "default_target_position": [0.45, 0.0, 0.35],
      "detection_bounds": {
        "min": [-0.5, -0.6, 0.05],
        "max": [0.9, 0.6, 1.0]
      },
      "grasp_attachment_body": "tool_link",
      "max_gripper_width_m": 0.08,
      "gripper_actuator_names": ["gripper"],
      "gripper_open_ctrl": [1.0],
      "gripper_closed_ctrl": [0.0],
      "passive_actuator_ctrl": {},
      "aliases": ["arm-alias"]
    }
  ]
}
```

Omitting `schema_version` selects version 1 for backward compatibility. Unknown fields and unsupported versions are rejected.

Paths resolve relative to the descriptor. The asset directory supplies the effective mesh directory; source texture/asset directory semantics are converted to absolute paths before in-memory compilation. The runtime forces `strippath=false` and `discardvisual=false` so injected target assets remain renderable.

Controlled joints must be named scalar hinge/slide joints, and each named actuator must transmit its corresponding joint. Scene construction supports:

- original compatible position servos;
- controlled actuator rewriting to MuJoCo velocity servos;
- controlled actuator rewriting to torque motors.

Passive actuators, such as a gripper, are not rewritten. A custom MJCF must currently be a self-contained `<mujoco>` document; `<include>` is rejected. Reachability, collision safety, actuator tuning, and controller stability cannot be inferred from the descriptor alone.

## Target descriptor version 1

Built-ins are `apple`, `bottle`, `box`, `capsule`, `cup`, `cylinder`, `dumbbell`, `hammer`, `phone`, `sphere`, and `tower`.

Target files accept one list or a versioned wrapper. Custom targets support:

- `box`, `sphere`, `cylinder`, and `capsule` primitives;
- compound targets assembled from primitive and/or mesh parts;
- OBJ and binary/ASCII STL meshes with positive XYZ scale;
- a normalized WXYZ body quaternion;
- visual or physical dynamics;
- named executable grasp points.

Example physical mesh target:

```json
{
  "schema_version": 1,
  "targets": [
    {
      "schema_version": 1,
      "name": "mesh-part",
      "shape": "mesh",
      "mesh_file": "meshes/part.obj",
      "scale": [1.0, 1.0, 1.0],
      "rgba": [0.7, 0.2, 0.1, 1.0],
      "base_position": [0.48, 0.0, 0.40],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "dynamics": "physical",
      "mass": 0.20,
      "friction": [0.9, 0.01, 0.001],
      "grasp_points": [
        {
          "name": "side",
          "position": [0.04, 0.0, 0.0],
          "approach": [-1.0, 0.0, 0.0],
          "width_m": 0.05
        }
      ]
    }
  ]
}
```

Mesh paths resolve relative to the descriptor and remain external runtime assets. If `size` is omitted, scaled AABB extents are computed from mesh vertices; an explicit `size` remains useful as perception metadata. Users remain responsible for mesh units, origin, provenance, and redistribution rights.

### Visual versus physical

- `"dynamics": "visual"` is the default and preserves the mocap-controlled, non-colliding visual reference behavior.
- `"dynamics": "physical"` creates a colliding free body using the configured mass and MuJoCo friction triplet. It falls under gravity and can contact the injected table/floor.

Physical targets receive a disabled weld equality when the selected robot declares `grasp_attachment_body`. The runtime can activate or deactivate it through `activate_grasp()` and `deactivate_grasp()`. This provides deterministic grasp-task attachment; it is not a physics claim about force closure.

Grasp-point `position` and `approach` use target-local coordinates. `approach` points along the final motion from pregrasp to grasp and is normalized. Optional widths are checked against robot gripper metadata when available. A default center top-down grasp point is generated when none is specified.

## Environment and camera assets

`EnvironmentSpec` independently enables the injected floor, table, and lights. A world-fixed camera creates `camera_marker`; a `CameraConfig.mount_body` camera is injected directly beneath the named robot body and uses body-local position/look-at coordinates.

Injected `servo_*`, `target*`, camera, and weld names are reserved. Conflicts fail early with a rename error.
