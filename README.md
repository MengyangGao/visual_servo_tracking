# Visual Servo Tracking

This repository contains visual-servo tracking demos for robotic manipulators.

- `matlab/`: MATLAB experiments for calibration, fixed-camera tracking, eye-in-hand tracking, and real-camera demos.
- `mujoco/`: Python + MuJoCo visual-servo simulator using a Franka Emika Panda arm from Google DeepMind MuJoCo Menagerie.

## MuJoCo

The MuJoCo project is the current active simulator. It builds a scene with:

- a selectable Menagerie robot arm, defaulting to Franka Emika Panda,
- a modular target object selected by word, such as `apple`, `cup`, `box`, `sphere`, or `capsule`, with optional JSON target files,
- a fixed robot perception camera named `servo_camera`,
- a passive MuJoCo viewer with a top-right robot-camera overlay showing semantic detections, masks, depth source, boxes, and tracker state,
- task modes including direct contact and front standoff tracking at a requested distance.

Semantic perception is the primary path. It uses `GroundingDINO` for open-vocabulary box detection, `SAM` for the initial mask, then a local mask/depth tracker for subsequent frames. Depth is explicit and modular: simulation defaults to fast metric MuJoCo depth, while `--depth-backend depth-anything-v2` enables an optional learned monocular depth backend through Hugging Face Transformers. The viewer loop is decoupled from semantic and learned-depth inference so the main MuJoCo view keeps running while model inference is pending.

Detection backends:

- `semantic` (default): open-vocabulary visual servo with depth-assisted 3D anchor estimation.
- `color`: HSV mask debug backend for controlled color targets.
- `oracle`: simulator truth debug backend (`target_site`) for controller smoke tests.

### Setup

Use the project conda environment:

```bash
conda activate visual_servo
```

Initialize the MuJoCo Menagerie submodule:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie
```

Install the MuJoCo package:

```bash
conda run -n visual_servo python -m pip install -e mujoco
```

Install optional semantic perception dependencies:

```bash
conda run -n visual_servo python -m pip install -e "mujoco[semantic]"
```

### Run

On macOS, use `mjpython` for the native MuJoCo viewer:

```bash
conda run -n visual_servo mjpython mujoco/scripts/demo.py \
  --robot panda \
  --target apple \
  --trajectory static \
  --task front-standoff \
  --standoff-cm 10 \
  --detector semantic \
  --depth-backend mujoco \
  --camera-fps 3
```

The first semantic run can pause while Hugging Face weights load. On macOS, viewer mode keeps perception on the main thread to avoid AppKit thread crashes (`NSScreen reconfig must only happen on the main thread`). Viewer runs default to a long session, so `--steps` is usually unnecessary.

Use learned monocular depth when you want to test camera-only depth behavior. This is slower than MuJoCo metric depth, so keep `--camera-fps` modest:

```bash
conda run -n visual_servo mjpython mujoco/scripts/demo.py \
  --robot panda \
  --target "red cup" \
  --trajectory static \
  --task front-standoff \
  --detector semantic \
  --depth-backend depth-anything-v2 \
  --depth-model depth-anything/Depth-Anything-V2-Small-hf \
  --camera-fps 1
```

Debug-only oracle smoke run, useful when validating robot/control changes without model downloads:

```bash
conda run -n visual_servo python mujoco/scripts/demo.py \
  --headless \
  --steps 120 \
  --target cup \
  --trajectory static \
  --task contact \
  --detector oracle \
  --no-realtime
```

Debug-only color segmentation is still available with `--detector color`, but it is not the primary path.

Headless acceptance matrix examples:

```bash
# semantic (primary)
conda run -n visual_servo python mujoco/scripts/demo.py \
  --headless --detector semantic --robot panda --target apple \
  --trajectory static --task front-standoff --steps 180 --camera-fps 6 --no-realtime

# color (debug)
conda run -n visual_servo python mujoco/scripts/demo.py \
  --headless --detector color --robot panda --target apple \
  --trajectory static --task front-standoff --steps 180 --no-realtime

# oracle (debug)
conda run -n visual_servo python mujoco/scripts/demo.py \
  --headless --detector oracle --robot panda --target apple \
  --trajectory static --task front-standoff --steps 180 --no-realtime
```

### Controls

The MuJoCo mouse controls are left to the standard viewer, so orbit/pan/zoom should behave normally.

Target keyboard controls:

- Arrow keys: horizontal target movement.
- `,`: move target down.
- `.`: move target up.
- Space or Backspace: reset manual target offset.
- `--scripted-target`: disable keyboard offsets and use only the selected trajectory.

Target offsets are keyboard-controlled.

### Useful Options

- `--target`: target word or phrase.
- `--target-file`: JSON file with additional primitive or compound target specs.
- `--robot`: `panda`, `ur5e`, or `lite6`.
- `--trajectory`: `static`, `circle`, `figure-eight`, `random-walk`, or `waypoints`.
- `--task`: `contact`, `standoff`, `front-standoff`, `align-x`, `align-y`, or `align-z`.
- `--standoff-cm`: distance for standoff modes.
- `--detector`: `semantic`, `oracle`, or `color`; default is `semantic`.
- `--depth-backend`: `mujoco`, `depth-anything-v2`, or `none`; default is metric `mujoco`.
- `--depth-model`: Hugging Face model id for learned depth, default `depth-anything/Depth-Anything-V2-Small-hf`.
- `--depth-device`: `auto`, `cpu`, `mps`, or `cuda`.
- `--no-depth-metric-hint`: disables MuJoCo metric-depth calibration for learned depth.
- `--camera-fps`: robot-camera processing rate in viewer mode.
- `--camera-width`, `--camera-height`: robot-camera render size; defaults are `424x320` to keep semantic runs responsive.
- `--steps`: control steps; defaults to `1000000` with the viewer and `1200` in headless mode.
- `--debug-perception`: print bbox, mask area, depth backend, 3D target estimate, and simulation-only truth error for diagnosing bad visual servo targets.
- `--overlay-width-frac`: top-right overlay width as a fraction of viewer width.
- `--no-camera-overlay`: hide the robot-camera overlay.
- `--list-targets`: print built-in target names.

Robot/target swapping notes:

- Each robot has its own default target workspace center so `oracle` control is reachable out-of-the-box (`panda`, `lite6`, `ur5e`).
- `--target-file` custom targets take exact-name priority over built-in targets (example: `urbox` will not be mistaken for built-in `box`).
- Custom target part entries accept both `pos` and `offset` keys. Target files are validated for finite positive sizes, supported primitive shapes, RGBA range, duplicate names, and finite base positions.
- Runtime summaries report task error against the simulated target for evaluation; non-oracle modes still never command from simulator truth.
- Camera and depth inputs are shape-checked before 3D anchor estimation, and viewer perception is throttled by `--camera-fps` in both asynchronous and macOS main-thread modes.

### Validation

```bash
conda run -n visual_servo pytest mujoco/tests
```

Minimal custom target file:

```json
{
  "targets": [
    {
      "name": "banana",
      "shape": "capsule",
      "size": [0.04, 0.04, 0.16],
      "rgba": [0.95, 0.78, 0.12, 1.0],
      "aliases": ["yellow banana"],
      "base_position": [0.42, -0.05, 0.36]
    }
  ]
}
```

## MATLAB

The MATLAB project covers:

- `T1`: ChArUco-based camera calibration.
- `T2`: position-based tracking with fixed-camera and eye-in-hand simulation modes.
- `T3`: feature-based tracking with an IBVS control loop.
- Real-camera follow and IBVS demos that reuse saved calibration parameters.

Technical report:

- [`matlab/report.md`](matlab/report.md)

Main MATLAB entry point:

```matlab
addpath(genpath(pwd));
results = run_demo();
```

Refresh real-camera calibration parameters:

```matlab
addpath(genpath(pwd));
results = run_live_camera_calibration();
```

Run real-camera follow or IBVS demos:

```matlab
addpath(genpath(pwd));
follow = run_real_camera_follow();
ibvs = run_real_camera_ibvs();
```

Public MATLAB assets:

- Printable ChArUco board: [`matlab/assets/charuco_board_printable.png`](matlab/assets/charuco_board_printable.png), [`matlab/assets/charuco_board_printable.pdf`](matlab/assets/charuco_board_printable.pdf)
- Saved camera parameters: [`matlab/assets/cameraParams.mat`](matlab/assets/cameraParams.mat)
- Technical report assets: [`matlab/assets/report/`](matlab/assets/report/)

Generated MATLAB results and logs are written locally under `matlab/results/`.
