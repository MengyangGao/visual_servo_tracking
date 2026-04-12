# MuJoCo Vision Servo

This branch is the Python + MuJoCo replacement for the MATLAB deliverable.

## What it does

- Loads Franka Panda from `reference/mujoco_menagerie` when available.
- Uses a fixed MuJoCo world viewer so robot motion stays visible.
- Renders a second follow-view for the robot-side output frame.
- Uses image-space feature corners, not a centroid-only controller.
- Supports simulation and real-camera modes.
- Supports `oracle`, `heuristic`, and `grounded-sam2` perception backends.

## Perception backends

- `oracle`: deterministic backend for simulation and tests.
- `heuristic`: prompt-guided classical fallback.
- `grounded-sam2`: Grounding DINO + SAM 2 when open-vocabulary weights are available.

Vision presets:

- `default`: `IDEA-Research/grounding-dino-base` + `facebook/sam2.1-hiera-base-plus`
- `small`: `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-small`
- `lite`: `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-tiny`

If SAM 2 is not available, the backend falls back to box-mask mode so the controller still runs.

## Install

```bash
conda run -n mujoco python -m pip install -e .
```

To enable the open-vocabulary backend:

```bash
conda run -n mujoco python -m pip install -e ".[open-vocab]"
```

If you have a local checkout of `reference/Grounded-SAM-2`, the backend uses it automatically. Otherwise set `MUJOCO_SERVO_SAM2_REPO` and `MUJOCO_SERVO_SAM2_CHECKPOINT`.

The first `grounded-sam2` run downloads the Grounding DINO weights and the SAM 2 checkpoint into `mujoco/outputs/hf_cache`.

## Run

Simulation:

```bash
conda run -n mujoco python -m mujoco_servo sim --prompt "red apple"
```

Real camera:

```bash
mjpython -m mujoco_servo camera --prompt "cup" --backend grounded-sam2 --vision-preset lite --run-mode manual
```

The camera mode uses the system camera and the fixed MuJoCo world viewer. The recorded/preview frame shows the follow-view plus the camera frame.

GUI:

```bash
conda run -n mujoco python -m mujoco_servo gui
```

List available cameras:

```bash
conda run -n mujoco python -m mujoco_servo cameras
```

## Notes

- The controller works from feature corners derived from detections.
- The world viewer is intentionally static so motion stays visible.
- On macOS, camera access still depends on system permission prompts.
- The open-vocabulary backend is optional; the smoke tests do not require it.
