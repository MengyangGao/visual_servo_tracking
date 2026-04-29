# Visual Servo Tracking

This repository contains visual-servo tracking demos for robotic manipulators.

- `matlab/`: MATLAB experiments for calibration, fixed-camera tracking, eye-in-hand tracking, and real-camera demos.
- `mujoco/`: Python + MuJoCo visual-servo simulator using a Franka Emika Panda arm from Google DeepMind MuJoCo Menagerie.

## MuJoCo

The MuJoCo project is the current active simulator. It builds a scene with:

- a Menagerie Franka Emika Panda robot arm,
- a modular target object selected by word, such as `apple`, `cup`, `box`, `sphere`, or `capsule`,
- a fixed robot perception camera named `servo_camera`,
- a passive MuJoCo viewer with a top-right robot-camera overlay showing image detections, masks, boxes, and tracker state,
- task modes including direct contact and front standoff tracking at a requested distance.

The semantic perception path uses `GroundingDINO` for the first open-vocabulary box detection, `SAM` for the initial mask, then a local mask/color/depth tracker for subsequent frames. The viewer loop is decoupled from semantic inference so the main MuJoCo view keeps running while model inference is pending.

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
  --target apple \
  --trajectory static \
  --task front-standoff \
  --standoff-cm 10 \
  --detector semantic \
  --steps 1000000
```

For a lighter detector:

```bash
conda run -n visual_servo mjpython mujoco/scripts/demo.py \
  --target cup \
  --trajectory circle \
  --task contact \
  --detector color \
  --steps 1000000
```

Headless smoke run:

```bash
conda run -n visual_servo python mujoco/scripts/demo.py \
  --headless \
  --steps 120 \
  --target cup \
  --trajectory static \
  --task contact \
  --detector color \
  --no-realtime
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
- `--trajectory`: `static`, `circle`, `figure-eight`, `random-walk`, or `waypoints`.
- `--task`: `contact`, `standoff`, `front-standoff`, `align-x`, `align-y`, or `align-z`.
- `--standoff-cm`: distance for standoff modes.
- `--detector`: `oracle`, `color`, or `semantic`.
- `--camera-fps`: robot-camera processing rate in viewer mode.
- `--overlay-width-frac`: top-right overlay width as a fraction of viewer width.
- `--no-camera-overlay`: hide the robot-camera overlay.
- `--list-targets`: print built-in target names.

### Validation

```bash
conda run -n visual_servo pytest mujoco/tests
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
