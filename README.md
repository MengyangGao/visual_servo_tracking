# Visual Servo Tracking

This repository contains visual-servo experiments for robotic manipulators.

- `mujoco/` is the active project: a Python and MuJoCo RGB-D visual-servo simulator.
- `matlab/` is frozen historical work. It remains in the repository for reference, but it is not in the active implementation or validation scope.

## MuJoCo capabilities

The simulator currently provides:

- a world-fixed camera named `servo_camera` that is automatically side-framed around the selected robot workspace and renders RGB plus depth;
- `color`, `semantic`, and simulator-truth `oracle` perception backends;
- public simulator-state and named body/site position access;
- replaceable robot models through three built-in MuJoCo Menagerie descriptors or a strict JSON robot descriptor;
- replaceable primitive, compound, OBJ, and STL visual target models through strict JSON;
- static, circular, figure-eight, random-walk, and waypoint target motion;
- contact, standoff, front-standoff, and single-axis alignment tasks;
- a passive viewer with a top-right camera overlay and keyboard target motion.

The default configuration is deliberately lightweight: `panda`, `cup`, `color`, MuJoCo metric depth, and `front-standoff` at 16 cm. Semantic models are optional.

### Position semantics and safety behavior

The three perception paths do not report the same physical point:

- `oracle` reads `target_site` from MuJoCo and reports the simulated target-body origin with anchor `truth_center`.
- `color` segments pixels near the target's top-level `rgba`, then unprojects the visible RGB-D mask surface. Its 3D anchor is normally `surface_depth_mask_centroid`.
- `semantic` uses Grounding DINO for a box, SAM for the initial mask, and a local mask/color/depth tracker afterward. It also reports a visible RGB-D surface anchor, not simulator truth.

Color and semantic detections drive the controller only when depth is metric. Relative monocular depth, `--depth-backend none`, or an unsuccessful learned-depth calibration can still be displayed for diagnostics, but produces a `non_metric_depth` anchor and does not command motion. If a valid visual observation becomes older than `--detection-timeout` (default `0.75` simulation seconds), the controller holds the current joints instead of pursuing stale data. Non-oracle modes never fall back to simulator truth.

Target geoms are visual and have collision disabled. Consequently, `contact` means "place the configured end-effector control point at `target_site`"; it is not a physical contact, grasp, or force-control task and may visually pass into the target. The default `front-standoff` mode avoids that behavior. For color and semantic modes, the controller tracks the detected visible surface anchor, while summary truth-error fields are evaluated against the simulated center; those values need not coincide for a large object.

## Installation

Python 3.10 and MuJoCo 3.3.1 or newer are required. The repository does not assume a pre-existing Conda environment.

From the repository root:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "mujoco[test]"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

For semantic perception and learned depth, install both the semantic and test extras:

```bash
python -m pip install -e "mujoco[semantic,test]"
```

The editable install uses the checked-out Menagerie submodule. A built wheel contains the Python package but not the external Menagerie git submodule or user-provided meshes. To use built-in robots from a wheel, point the process at a Menagerie checkout before importing `mujoco_servo`:

```bash
export MUJOCO_MENAGERIE_PATH=/absolute/path/to/mujoco_menagerie
```

The directory must directly contain `franka_emika_panda/`, `universal_robots_ur5e/`, and `ufactory_lite6/`. Alternatively, use `--robot-file` with your own assets.

## Running the simulator

Default interactive run on Linux or Windows:

```bash
python mujoco/scripts/demo.py \
  --robot panda \
  --target cup \
  --trajectory circle
```

On macOS, the native passive viewer must be launched through `mjpython`:

```bash
mjpython mujoco/scripts/demo.py \
  --robot panda \
  --target cup \
  --trajectory circle
```

The installed console entry point and module entry point are also available for headless runs:

```bash
mujoco-servo --headless --detector oracle --steps 240 --trajectory static --no-realtime
python -m mujoco_servo --headless --detector oracle --steps 240 --trajectory static --no-realtime
```

`--headless` disables the passive viewer, but color and semantic perception still render RGB-D offscreen and therefore still require a working OpenGL backend. Linux CI commonly uses an EGL or OSMesa-capable MuJoCo setup. A macOS background session without a CoreGraphics connection may not support offscreen rendering. `oracle` does not render and is the appropriate display-independent smoke test.

### Semantic example

`--target` chooses the simulated model. `--prompt` independently tells Grounding DINO what to find:

```bash
mjpython mujoco/scripts/demo.py \
  --robot panda \
  --target cup \
  --prompt "red mug" \
  --trajectory static \
  --detector semantic \
  --depth-backend mujoco \
  --camera-fps 3 \
  --detection-timeout 0.75
```

The semantic backend loads models lazily on first use. The first run downloads model weights through Hugging Face unless they are already cached. Viewer inference uses a background worker where supported; macOS keeps viewer perception on the main thread to avoid AppKit thread violations. Headless perception is synchronous and sampled against simulation time at `--camera-fps`.

Semantic environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MUJOCO_SERVO_GDINO_MODEL` | `IDEA-Research/grounding-dino-tiny` | Grounding DINO model id or local path |
| `MUJOCO_SERVO_SAM_MODEL` | `facebook/sam-vit-base` | SAM model id or local path |
| `MUJOCO_SERVO_DEVICE` | `auto` | Semantic device and learned-depth fallback device: `auto`, `cpu`, `mps`, or `cuda` |
| `MUJOCO_SERVO_GDINO_BOX_THRESHOLD` | `0.25` | Grounding DINO box threshold |
| `MUJOCO_SERVO_GDINO_TEXT_THRESHOLD` | `0.25` | Grounding DINO text threshold |
| `MUJOCO_SERVO_REDETECT_INTERVAL` | `45` | Local-tracker frames before semantic redetection |
| `MUJOCO_SERVO_MAX_TRACK_FAILURES` | `3` | Tracker failures before reinitialization |

`--depth-device` controls the optional learned-depth pipeline; semantic device selection uses `MUJOCO_SERVO_DEVICE`.

### Learned depth

Depth Anything V2 can replace the direct MuJoCo depth provider:

```bash
python mujoco/scripts/demo.py \
  --headless \
  --robot panda \
  --target cup \
  --detector color \
  --depth-backend depth-anything-v2 \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf \
  --camera-fps 2 \
  --steps 240 \
  --no-realtime
```

By default, the backend attempts to calibrate learned output with MuJoCo metric depth. `--no-depth-metric-hint` removes that simulator hint. Learned output without a successful calibration is treated as relative depth and is intentionally rejected for control; a model name containing words such as `metric` is not trusted as a unit contract.

## Reading simulator positions

`VisualServoSimulation` exposes copies of the current MuJoCo state in world coordinates. Linear positions are in meters; hinge joint values are radians and slide-joint values are meters.

```python
from mujoco_servo import DemoConfig, VisualServoSimulation

config = DemoConfig(
    detector="oracle",
    trajectory="static",
    steps=120,
    headless=True,
    viewer=False,
    realtime=False,
)
simulation = VisualServoSimulation(config)

initial = simulation.get_state()
print(initial.as_dict())
print(simulation.get_body_position("target"))
print(simulation.get_site_position("target_site"))

summary = simulation.run()
print(summary.final_target_position)
print(summary.final_end_effector_position)
print(summary.final_detected_position)
print(summary.final_detection_anchor)
```

`get_state()` returns `target_position`, `end_effector_position`, `camera_position`, ordered `joint_positions`, the last detected position, detection backend/anchor, and detection age. `get_body_position(name)` and `get_site_position(name)` raise `KeyError` for unknown MuJoCo names.

The CLI prints the complete `RunSummary` as JSON. Position-related summary fields are:

- `final_target_position`: simulator-truth `target_site` center;
- `final_end_effector_position`: configured EE frame or body-point position;
- `final_detected_position`: latest valid oracle center or RGB-D surface anchor, if any;
- `final_detection_anchor`: `truth_center`, a surface-anchor label, or the last failure label;
- `final_target_distance_m`: EE-to-truth-center distance;
- `final_error_m`: final task-space error evaluated against simulator truth;
- `final_orientation_error_rad`: front-standoff tool-axis alignment error.

## Replacing the robot

### Built-in robots

All built-ins come from the pinned MuJoCo Menagerie submodule:

| CLI name | Aliases | MJCF path below Menagerie root | Controlled joints | EE frame | Default target position |
| --- | --- | --- | ---: | --- | --- |
| `panda` | `franka`, `franka-panda`, `franka_emika_panda` | `franka_emika_panda/panda.xml` | 7 | body point `hand` + `[0, 0, 0.10]` | `[0.55, 0.10, 0.40]` |
| `ur5e` | `universal-robots-ur5e`, `universal_robots_ur5e`, `ur` | `universal_robots_ur5e/ur5e.xml` | 6 | site `attachment_site` | `[-0.30, 0.45, 0.50]` |
| `lite6` | `ufactory-lite6`, `ufactory_lite6`, `xarm-lite6` | `ufactory_lite6/lite6.xml` | 6 | site `attachment_site` | `[0.32, 0.00, 0.38]` |

Their local tool approach axis is `+Z`. `front-standoff` aligns the configured local `tool_axis` with the horizontal direction toward the detected target.

### Custom robot descriptor

Pass a strict JSON descriptor together with its robot name or alias:

```bash
python mujoco/scripts/demo.py \
  --robot my-arm \
  --robot-file /path/to/my-arm/robot.json \
  --target cup \
  --detector oracle \
  --headless \
  --steps 120 \
  --no-realtime
```

Complete descriptor example:

```json
{
  "name": "my-arm",
  "xml_path": "robot.xml",
  "asset_dir": "assets",
  "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
  "actuator_names": ["joint1_pos", "joint2_pos", "joint3_pos", "joint4_pos", "joint5_pos", "joint6_pos"],
  "home_qpos": [0.0, -0.7, 1.2, 0.0, 0.8, 0.0],
  "ee_frame": {
    "name": "tool_site",
    "type": "site",
    "offset": [0.0, 0.0, 0.0]
  },
  "tool_axis": [0.0, 0.0, 1.0],
  "base_position": [0.0, 0.0, 0.0],
  "passive_actuator_ctrl": {
    "gripper_position": 0.04
  },
  "max_gripper_width_m": 0.08,
  "default_target_position": [0.45, 0.0, 0.35],
  "detection_bounds": {
    "min": [-0.5, -0.7, 0.05],
    "max": [0.9, 0.7, 1.0]
  },
  "aliases": ["my-robot"]
}
```

The file may instead contain a list of robot objects or `{"robots": [...]}`. Custom names and aliases take priority over built-ins. `xml_path` and `asset_dir` are resolved relative to the descriptor file. `passive_actuator_ctrl` may be the object map shown above or a list such as `[{"name": "gripper_position", "value": 0.04}]`. `detection_bounds` may be the `min`/`max` object shown above or `[[min_x, min_y, min_z], [max_x, max_y, max_z]]`. Optional `tool_axis` is normalized and must be non-zero; its coordinates are local to the EE frame. Optional `base_position` defaults to world `[0, 0, 0]` and defines the horizontal approach direction and automatic camera side view for a translated robot installation.

Custom robot requirements:

- The MJCF must be a self-contained XML file whose root is `<mujoco>`. `<include>` is rejected because the scene is injected in memory.
- Every controlled joint must be a named scalar hinge or slide joint.
- `joint_names`, `actuator_names`, and `home_qpos` must have equal non-zero lengths and contain no duplicates.
- Each controlled actuator must transmit the corresponding joint and behave as a non-degenerate position servo. Scalar transmission gear is applied automatically; generic torque/motor and velocity semantics are rejected.
- A single unnamed joint actuator may receive its descriptor name deterministically; an incorrect name for an already named actuator is rejected. Every home position must fit both the joint range and its actuator control range.
- `ee_frame.type` must be `site`, `body`, or `body_point`. `offset` is applied only for `body_point` and is expressed in that body's local frame.
- The asset directory must exist and supplies the effective MJCF `meshdir`. Texture lookup preserves the source compiler's `texturedir`/`assetdir` semantics (or the XML directory when neither is set), while all effective directories are made absolute before in-memory compilation. The injected visual scene forces `strippath=false` and `discardvisual=false` so custom targets are not stripped or discarded.
- Injected object names are reserved within their MuJoCo namespaces: bodies `camera_marker`/`target`, site `target_site`, the configured camera name, and the documented `servo_`/`target_geom`/`target_mesh` names. A collision fails early with a clear rename error instead of a MuJoCo duplicate-name failure.
- `passive_actuator_ctrl` entries must not overlap controlled actuators and are held at their configured constants.
- `detection_bounds` is strongly recommended. Without it, detections are only bounded to the broad world cube `[-5, 5]` meters; with it, target keyboard motion is also clipped to the configured workspace.
- The loader validates structure, finite values, lengths, duplicates, paths, and bound ordering, but it cannot prove reachability, actuator tuning, collision safety, or control stability for an arbitrary model.

## Replacing the target

`--target` selects a built-in or custom target model. Unknown target names are rejected rather than silently replaced with a generic box. A target JSON file may contain a list or `{"targets": [...]}`; custom exact names and aliases take priority.

Supported size conventions:

| Shape | `size` meaning |
| --- | --- |
| `box` | full X, Y, Z extents |
| `sphere` | three equal full diameters |
| `cylinder` | equal X/Y full diameter and full Z height |
| `capsule` | equal X/Y full diameter and full end-to-end Z height; height must exceed diameter |
| `mesh` | required approximate full bounding box used as target metadata; actual geometry comes from mesh plus scale |
| `compound` | approximate full bounding box for the assembled object; geometry comes from `parts` |

Primitive target:

```json
{
  "targets": [
    {
      "name": "banana-proxy",
      "shape": "capsule",
      "size": [0.04, 0.04, 0.16],
      "rgba": [0.95, 0.78, 0.12, 1.0],
      "aliases": ["yellow banana"],
      "base_position": [0.48, 0.0, 0.38]
    }
  ]
}
```

Compound target:

```json
{
  "targets": [
    {
      "name": "marker-tool",
      "shape": "compound",
      "size": [0.18, 0.06, 0.08],
      "rgba": [0.85, 0.20, 0.15, 1.0],
      "base_position": [0.50, 0.0, 0.40],
      "parts": [
        {
          "shape": "capsule",
          "size": [0.025, 0.025, 0.16],
          "pos": [0.0, 0.0, 0.0],
          "quat": [0.7071, 0.0, 0.7071, 0.0]
        },
        {
          "shape": "box",
          "size": [0.06, 0.06, 0.04],
          "offset": [0.08, 0.0, 0.0],
          "rgba": [0.20, 0.35, 0.90, 1.0]
        }
      ]
    }
  ]
}
```

OBJ or STL mesh target:

```json
{
  "targets": [
    {
      "name": "mesh-cup",
      "shape": "mesh",
      "size": [0.10, 0.08, 0.12],
      "mesh_file": "meshes/cup.obj",
      "scale": [1.0, 1.0, 1.0],
      "rgba": [0.90, 0.18, 0.12, 1.0],
      "aliases": ["custom red cup"],
      "base_position": [0.50, 0.0, 0.40]
    }
  ]
}
```

Mesh paths are resolved relative to the target JSON file, must exist, and must end in `.obj` or `.stl`. `mesh_path` is accepted as an alias for `mesh_file`; `mesh_scale` is accepted as an alias for `scale`, but each pair is mutually exclusive. Mesh parts are also allowed inside `compound` and use the same `mesh_file`/`scale` fields. A part accepts `shape`, `size`, exactly one of `pos` or `offset`, optional `rgba`, and optional normalized WXYZ `quat`.

All numeric values must be finite; dimensions and mesh scales must be positive; RGBA values must be in `[0, 1]`; names and aliases must be unique; unknown fields are rejected. For reliable color perception, set the target's top-level `rgba` to the dominant rendered surface color even when individual compound parts override it.

Select custom geometry independently from the semantic phrase:

```bash
python mujoco/scripts/demo.py \
  --target mesh-cup \
  --target-file /path/to/targets.json \
  --prompt "red ceramic mug" \
  --detector semantic
```

## Tasks and controls

Tasks:

- `front-standoff` (default): hold the requested horizontal distance and align the robot's local `tool_axis` toward the target;
- `standoff`: hold the requested distance along the current EE-to-target direction;
- `contact`: command the EE control point to the target center/surface anchor, without physical collision semantics;
- `align-x`, `align-y`, `align-z`: adjust one Cartesian coordinate using `align_offset_m` from the Python API.

Use `--standoff-cm` for CLI distances or `--standoff` in meters. If both are supplied, `--standoff` takes precedence.

Viewer target controls:

- Arrow keys move the target horizontally.
- `,` moves it down and `.` moves it up.
- Space or Backspace resets the manual offset.
- `--scripted-target` disables keyboard offsets.
- Standard MuJoCo mouse orbit, pan, and zoom remain available.

Useful options include `--camera-fps`, `--camera-width`, `--camera-height`, `--detection-timeout`, `--debug-perception`, `--no-camera-overlay`, `--overlay-width-frac`, `--seed`, and `--list-targets`. Run `python mujoco/scripts/demo.py --help` for the authoritative CLI list.

When `CameraConfig` keeps its default pose, `VisualServoSimulation` places the camera to the side of the base-to-target approach line so a swapped robot is less likely to occlude the target. Supplying a non-default `CameraConfig(position=..., lookat=...)` preserves that absolute world pose. The camera remains fixed after scene construction; this is not eye-in-hand control.

## Validation

After installing `mujoco[test]`:

```bash
python -m pytest mujoco/tests

python mujoco/scripts/demo.py \
  --headless \
  --detector oracle \
  --robot panda \
  --target cup \
  --trajectory static \
  --steps 240 \
  --no-realtime
```

With a working offscreen OpenGL context:

```bash
python mujoco/scripts/demo.py \
  --headless \
  --detector color \
  --robot panda \
  --target cup \
  --trajectory static \
  --steps 240 \
  --camera-fps 6 \
  --no-realtime
```

After installing `mujoco[semantic,test]` and allowing model downloads, replace `--detector color` with `--detector semantic` and add an appropriate `--prompt` for an integration smoke test.

## Current limitations

- The automatically framed camera is fixed in the world after scene construction; eye-in-hand camera mounting and camera calibration/noise models are not implemented.
- Color segmentation assumes a controlled rendering and a representative top-level target color. It is not a general real-world color recognizer.
- Semantic quality depends on the selected models, prompt, scene, cache/network availability, and hardware. Base tests mock model inference rather than downloading weights.
- RGB-D visual anchors describe the visible surface, while oracle describes the target origin. The system does not estimate an occluded object center from a complete 3D model.
- Relative monocular depth is intentionally diagnostic-only. There is no scale-free controller for non-metric depth.
- Targets are mocap-controlled, visual, and non-colliding. Physical grasping, contact forces, target dynamics, and obstacle avoidance are out of scope.
- Custom robot support is descriptor-driven and intentionally strict; MJCF files with `<include>`, multi-DoF free/ball controlled joints, torque-only actuation, or incompatible EE conventions require adaptation.
- A Python wheel does not bundle Menagerie or user assets. Built-in robots need `MUJOCO_MENAGERIE_PATH`; custom assets need `--robot-file`/`--target-file` paths that remain available at runtime.
- GUI layout and input still need a local viewer smoke test, especially on macOS.

## MATLAB archive

The MATLAB directory contains earlier calibration, fixed-camera, eye-in-hand, and real-camera experiments, plus `matlab/report.md`. It is frozen and intentionally excluded from the active MuJoCo implementation, bug-fix, and acceptance scope.
