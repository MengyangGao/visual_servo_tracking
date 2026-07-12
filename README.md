# Visual Servo Tracking

This repository contains a configurable MuJoCo RGB-D visual-servo simulator. The active implementation is under `mujoco/`; `matlab/` is a frozen historical archive and is not part of current validation.

The MuJoCo implementation supports:

- simulator-truth, color, and open-vocabulary semantic target detection;
- metric MuJoCo depth and optional Depth Anything V2;
- world-fixed or robot-mounted RGB-D cameras, with optional sensor noise and dropout;
- Franka Panda, Universal Robots UR5e, and UFactory Lite6 models from MuJoCo Menagerie;
- position-, velocity-, and torque-actuated joint control;
- visual reference targets and free-body physical targets;
- versioned custom robot and target descriptors, including executable grasp points;
- reusable `reset()`, `step()`, `observe()`, and `close()` simulation APIs;
- a benchmark matrix, branch-coverage checks, and Linux EGL CI.

The default remains intentionally lightweight: Panda, cup, color segmentation, MuJoCo metric depth, position actuation, and a stable 16 cm standoff task. Select `--task front-standoff` when tool-axis facing is part of the experiment.

## Install

Initialize the pinned Menagerie assets first:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie
```

### Conda environment

The checked-in environment installs semantic, test, and development dependencies:

```bash
conda env create -f environment.yml
conda activate visual_servo
```

Recreate it after dependency changes with:

```bash
conda env remove -n visual_servo
conda env create -f environment.yml
```

### Python virtual environment

For the lightweight color/oracle simulator:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "mujoco[test,dev]"
```

Install optional semantic and learned-depth dependencies with:

```bash
python -m pip install -e "mujoco[semantic,test,dev]"
```

Python 3.10 or newer and MuJoCo 3.3.1 or newer are required.

## Stable smoke commands

These oracle commands do not render and are suitable for a display-independent controller check:

```bash
python -m mujoco_servo \
  --headless --no-realtime \
  --detector oracle --trajectory static \
  --robot panda --target cup \
  --actuator-mode position --steps 240

python -m mujoco_servo \
  --headless --no-realtime \
  --detector oracle --trajectory static \
  --robot panda --target cup \
  --actuator-mode velocity --steps 240

python -m mujoco_servo \
  --headless --no-realtime \
  --detector oracle --trajectory static \
  --robot panda --target cup \
  --actuator-mode torque --steps 240
```

The installed `mujoco-servo` command and `python mujoco/scripts/demo.py` accept the same options. Run `python -m mujoco_servo --help` for the authoritative list.

### Interactive viewer

Linux and Windows:

```bash
python -m mujoco_servo --robot panda --target cup --trajectory circle
```

macOS must launch the native passive viewer through MuJoCo's `mjpython`:

```bash
mjpython mujoco/scripts/demo.py --robot panda --target cup --trajectory circle
```

Viewer target controls are the arrow keys for horizontal motion, `,` and `.` for down/up, and Space or Backspace to reset the manual offset. `--scripted-target` disables keyboard offsets.

### Three built-in robots

```bash
python -m mujoco_servo --headless --no-realtime --detector oracle --trajectory static --robot panda --target cup --steps 240
python -m mujoco_servo --headless --no-realtime --detector oracle --trajectory static --robot ur5e --target box --steps 240
python -m mujoco_servo --headless --no-realtime --detector oracle --trajectory static --robot lite6 --target apple --steps 240
```

| Robot | Controlled joints | End-effector frame | Grasp attachment |
| --- | ---: | --- | --- |
| `panda` | 7 | `hand` body point + 0.10 m local offset | `hand`; official coupled finger actuator metadata is retained |
| `ur5e` | 6 | `attachment_site` | `wrist_3_link` weld/suction abstraction |
| `lite6` | 6 | `attachment_site` | `link6` weld/suction abstraction |

All three compile in every supported actuator mode. UR5e and Lite6 do not include a modeled gripper in these Menagerie files, so their stable grasp abstraction is an explicit runtime weld rather than a simulated finger closure.

## Perception

### Color

Color perception segments the selected target's configured dominant RGBA and unprojects the visible mask with metric depth:

```bash
python -m mujoco_servo \
  --headless --no-realtime \
  --detector color --depth-backend mujoco \
  --robot panda --target cup --trajectory static \
  --camera-fps 6 --steps 240
```

Color and semantic modes still render RGB-D when `--headless` is used, so they require a working OpenGL context. Linux CI uses `MUJOCO_GL=egl`; macOS background sessions without CoreGraphics may not support offscreen rendering.

### Semantic recognition

Semantic perception uses Grounding DINO for open-vocabulary boxes, SAM for masks, and a local color/depth tracker between redetections:

```bash
MUJOCO_SERVO_DEVICE=auto mjpython mujoco/scripts/demo.py \
  --robot panda --target cup \
  --detector semantic --prompt "red drinking mug" \
  --depth-backend mujoco --trajectory static --camera-fps 3
```

Device selection in `auto` mode is CUDA, then Apple MPS when available, then CPU. On Apple Silicon no explicit MPS flag is needed; `--depth-device auto` and `MUJOCO_SERVO_DEVICE=auto` select it automatically. macOS viewer perception stays on the main thread to respect AppKit restrictions.

The JSON run summary records the effective `perception_device` and `depth_device`. MuJoCo rigid-body dynamics itself runs on CPU; rendering uses the platform OpenGL backend, while learned perception/depth can use CUDA or MPS.

The first semantic run downloads model weights unless they are already cached. Override model/device settings with:

| Variable | Default |
| --- | --- |
| `MUJOCO_SERVO_GDINO_MODEL` | `IDEA-Research/grounding-dino-tiny` |
| `MUJOCO_SERVO_SAM_MODEL` | `facebook/sam-vit-base` |
| `MUJOCO_SERVO_DEVICE` | `auto` |
| `MUJOCO_SERVO_GDINO_BOX_THRESHOLD` | `0.25` |
| `MUJOCO_SERVO_GDINO_TEXT_THRESHOLD` | `0.25` |
| `MUJOCO_SERVO_REDETECT_INTERVAL` | `45` |
| `MUJOCO_SERVO_MAX_TRACK_FAILURES` | `3` |

### Depth and observation timing

MuJoCo metric depth is the default. Optional learned depth is selected with `--depth-backend depth-anything-v2`; relative monocular output is diagnostic-only unless metric calibration succeeds. Non-metric anchors never drive the Cartesian controller.

The runtime models target loss and reacquisition explicitly. Useful robustness controls include:

- `--detection-timeout` and `--reacquire-confirm-frames`;
- `--perception-latency`, `--perception-jitter`, and `--perception-drop-probability`;
- `--rgb-noise-std`, `--depth-noise-std`, and `--camera-dropout-probability`.

Color and semantic detections are visible surface anchors. Oracle detection reads the simulated `target_site` center. Their reported 3D positions therefore need not match for a large object.

## Controller and actuator modes

The Cartesian resolved-rate controller uses weighted adaptive damping, joint-limit avoidance, acceleration and speed limits, null-space posture control, and exact hold behavior during perception loss.

| Mode | CLI | Meaning |
| --- | --- | --- |
| Position | `--actuator-mode position` | Sends joint-position references to MuJoCo position servos. |
| Velocity | `--actuator-mode velocity` | Rewrites controlled actuators as velocity servos and sends velocity references with bias-force feed-forward. |
| Torque | `--actuator-mode torque` | Rewrites controlled actuators as motors and applies bias compensation plus joint-space PD torque. |

Relevant tuning flags are `--control-hz`, `--max-joint-accel`, `--joint-limit-margin`, `--torque-kp`, and `--torque-kd`. Torque mode is still a simulation controller; its gains are not safe commands for a real robot.

Tasks include `front-standoff`, `standoff`, `contact`, `touch`, `grasp`, and single-axis alignment. `touch` and `grasp` use a staged pregrasp/approach/lift workflow with physical targets. `--grasp-point`, `--grasp-approach`, `--grasp-attach-distance`, `--grasp-lift`, and `--grasp-stage-tolerance` expose task tuning without changing the descriptor. Grasp attachment is explicit and reversible; it is not a claim of force-closure grasp synthesis.

## Physical targets and grasp points

Targets default to `"dynamics": "visual"`, preserving the old mocap-controlled, non-colliding reference behavior. `"dynamics": "physical"` creates a colliding free body with mass and friction; it falls under MuJoCo gravity and can contact the default table.

Target files accept a top-level list or a versioned wrapper. Missing `schema_version` means version 1 for backward compatibility:

```json
{
  "schema_version": 1,
  "targets": [
    {
      "schema_version": 1,
      "name": "grasp-box",
      "shape": "box",
      "size": [0.06, 0.05, 0.10],
      "rgba": [0.18, 0.52, 0.92, 1.0],
      "base_position": [0.48, 0.0, 0.40],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "dynamics": "physical",
      "mass": 0.20,
      "friction": [0.9, 0.01, 0.001],
      "grasp_points": [
        {
          "name": "top",
          "position": [0.0, 0.0, 0.05],
          "approach": [0.0, 0.0, -1.0],
          "width_m": 0.05
        }
      ]
    }
  ]
}
```

`position` and `approach` are expressed in the target body's local coordinates; `approach` points along the final motion from pregrasp to grasp and is normalized. The optional width is checked against robot gripper metadata where available. If no grasp point is supplied, a center top-down grasp point with an inferred XY width is created.

Select the target with:

```bash
python -m mujoco_servo \
  --target grasp-box --target-file /path/to/targets.json \
  --task grasp --detector oracle --trajectory static
```

A built-in target is automatically promoted to physical dynamics when `--task touch` or `--task grasp` is selected. A stable display-independent grasp smoke is:

```bash
python -m mujoco_servo \
  --headless --no-realtime \
  --robot panda --target cup \
  --task grasp --detector oracle --trajectory static \
  --steps 1200
```

OBJ/STL mesh targets and mesh parts are supported. When mesh `size` is omitted, the loader computes its scaled AABB from OBJ or binary/ASCII STL vertices. Paths resolve relative to the target descriptor.

`VisualServoSimulation.get_grasp_point()`, `activate_grasp()`, and `release_grasp()` provide public explicit attachment control. The lower-level scene functions are `grasp_point_world()`, `activate_grasp()`, and `deactivate_grasp()`.

## Camera and environment

The default camera is world-fixed and automatically side-framed for the selected robot workspace. A non-default Python `CameraConfig(position=..., lookat=...)` preserves an explicit world pose.

For an eye-in-hand camera, set a robot body name; position and look-at coordinates then use that body's local frame. Use Python when an explicit local pose is required:

```python
from mujoco_servo import CameraConfig, DemoConfig, VisualServoSimulation

config = DemoConfig(
    camera=CameraConfig(
        mount_body="hand",
        position=(0.0, 0.0, 0.05),
        lookat=(0.0, 0.0, 0.30),
    )
)
with VisualServoSimulation(config) as simulation:
    print(simulation.observe().camera_position)
```

`--camera-mount-body hand` exposes the mounting choice from the CLI; the complete local camera pose is currently configured through `CameraConfig`.

The built-in floor, table, and lights can be disabled independently with `--no-default-floor`, `--no-default-table`, and `--no-default-lights`, or with `EnvironmentSpec` from Python.

## Python episode API

`VisualServoSimulation` can be used without its long-running viewer loop:

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

with VisualServoSimulation(config) as simulation:
    initial = simulation.reset()
    for _ in range(120):
        state = simulation.step()
    observed = simulation.observe()
    print(observed.as_dict())
    print(simulation.get_body_position("target"))
    print(simulation.get_site_position("target_site"))
```

- `reset()` restores the initial MuJoCo and runtime state and returns a `SimulationState`.
- `step()` advances one requested controller interval and returns a `SimulationState`.
- `observe()`/`get_state()` return copies of simulator and accepted perception state.
- `close()` releases renderer and perception workers and is safe to call repeatedly.
- The context manager calls `close()` automatically.
- `get_grasp_point()`, `activate_grasp()`, and `release_grasp()` expose explicit physical-target attachment control.

World positions are meters. Hinge positions are radians and slide positions are meters. Detection covariance/timestamps, tracking state, manipulation state, contact/grasp/lift status, target/EE/camera positions, and ordered robot joint positions are included in the public state.

## Robot replacement

`--robot-file` accepts a robot object, list, or `{"schema_version": 1, "robots": [...]}`. Version 1 is assumed for old descriptors. At minimum each robot declares:

- `name`, `xml_path`, and `asset_dir`;
- ordered `joint_names`, `actuator_names`, and `home_qpos`;
- `ee_frame` with `name`, `type`, and optional local `offset`.

Optional fields include aliases, tool axis, base position, workspace detection bounds, passive actuator controls, grasp attachment body, and gripper open/close metadata. Controlled joints must be scalar hinge/slide joints with joint transmissions. Official Menagerie position actuators can be rewritten into velocity or torque mode at scene construction.

Custom MJCF currently must be a self-contained `<mujoco>` document; `<include>` is rejected. Asset paths remain external and are not copied into wheels. Full descriptor and asset rules are in [`mujoco/ASSETS.md`](mujoco/ASSETS.md).

## Benchmark, tests, and CI

Run the full local suite:

```bash
python -m pytest -q mujoco/tests
ruff check mujoco
ruff format --check mujoco/src mujoco/tests mujoco/scripts
coverage run --branch -m pytest -q mujoco/tests
coverage report --fail-under=75
```

Run a reproducible oracle matrix and write its JSON report:

```bash
mujoco-servo-benchmark \
  --robots panda ur5e lite6 \
  --actuator-modes position velocity torque \
  --trajectories static circle \
  --seeds 7 \
  --steps 1200 \
  --output /tmp/mujoco-servo-benchmark.json
```

`--enforce` returns nonzero if any configured position/orientation threshold is missed. Static scenarios use final error; moving scenarios use RMS over the second half of the episode so startup transients do not masquerade as tracking error. Orientation is gated only for `front-standoff`. Inspect each scenario's `acceptance` object when changing tasks or thresholds.

GitHub Actions runs Python 3.10–3.13 on Ubuntu with EGL, Ruff formatting/linting, branch coverage of at least 75%, wheel construction, and an installed-wheel oracle smoke test.

## Wheel and external assets

A wheel contains simulator code, not Menagerie or user meshes. Point an installed wheel at a Menagerie checkout before importing the package:

```bash
export MUJOCO_MENAGERIE_PATH=/absolute/path/to/mujoco_menagerie
python -m mujoco_servo --headless --detector oracle --steps 2 --no-realtime
```

## Current limits

- There is no real-camera/real-robot transport, ROS/ROS 2 integration, hardware safety layer, or hardware calibration workflow in the MuJoCo implementation.
- Multi-object detection, association, and simultaneous multi-arm/multi-target control are intentionally deferred; one selected target is controlled per simulation.
- Semantic quality and latency depend on model weights, prompt, hardware, and cache/network availability. Routine tests mock large-model inference.
- Eye-in-hand mounting and synthetic noise are implemented, but camera intrinsic/extrinsic calibration estimation and distortion simulation are not.
- Grasp attachment is a deterministic weld/suction abstraction. General grasp synthesis, tactile sensing, force closure, slip, and contact-force control are not implemented.
- The controller does not perform global collision-aware motion planning or obstacle avoidance.
- Monocular relative depth remains diagnostic-only unless metric calibration succeeds.
- Custom robots require compatible scalar joints, actuator transmissions, EE conventions, and a self-contained MJCF. Descriptor validation cannot prove reachability or stability.
- Custom target geometry is limited to primitives, compounds, OBJ, and STL; textured semantic realism depends on user-provided scene assets.
- Wheels do not embed Menagerie or user assets.

## MATLAB archive

`matlab/` contains earlier calibration, fixed-camera, eye-in-hand, and real-camera experiments. It is frozen and intentionally excluded from current implementation and acceptance scope.
