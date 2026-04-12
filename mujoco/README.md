# MuJoCo Vision Servo

This project is the new Python + MuJoCo vision-servo branch of the repo.

## What is included

- a modular robot loader with Franka Panda as the first target
- a shared control loop for simulation and real camera input
- a perception interface that can swap between an oracle backend, a classical fallback, and an open-vocabulary backend
- CLI and GUI entry points

## Current backend policy

- `oracle`: deterministic backend for simulation and tests
- `heuristic`: classical prompt-guided fallback for real-camera development
- `grounded-sam2`: open-vocabulary backend that uses Grounding DINO for text grounding and SAM 2 for segmentation when SAM 2 is available
- `default` vision preset: `IDEA-Research/grounding-dino-base` + `facebook/sam2.1-hiera-base-plus`
- `small` vision preset: `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-small`
- `lite` vision preset: `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-tiny`

If SAM 2 is not installed or the checkpoint is not configured, the backend still runs in box-mask mode so the controller remains usable.

The code is written so the controller does not depend on backend-specific output shapes.

## Run

Install the package first:

```bash
conda run -n mujoco python -m pip install -e .
```

To enable the open-vocabulary backend:

```bash
conda run -n mujoco python -m pip install -e ".[open-vocab]"
```

If you have a local checkout of the SAM 2 reference repository, the backend will use it automatically when it finds `reference/Grounded-SAM-2`. Otherwise, point `MUJOCO_SERVO_SAM2_REPO` to the checkout path and set `MUJOCO_SERVO_SAM2_CHECKPOINT` to a SAM 2 checkpoint.
If you prefer an editable install, run `conda run -n mujoco python -m pip install -e /path/to/Grounded-SAM-2` before launching the project.
The first `grounded-sam2` run downloads the Grounding DINO base weights and the SAM 2 checkpoint into `mujoco/outputs/hf_cache`.

Then run the CLI:

```bash
conda run -n mujoco python -m mujoco_servo sim --prompt "red apple"
```

For the GUI:

```bash
conda run -n mujoco python -m mujoco_servo gui
```

For camera input:

```bash
conda run -n mujoco python -m mujoco_servo camera --prompt "red cup"
```

To reduce latency, use a lighter preset:

```bash
conda run -n mujoco python -m mujoco_servo camera --prompt "cup" --backend grounded-sam2 --vision-preset lite --run-mode manual
```

On macOS, run camera mode with `mjpython` so the official MuJoCo viewer can launch.
The robot scene uses the MuJoCo viewer. The separate live camera preview is disabled by default on macOS because Tk and `mjpython` can clash; enable it with `MUJOCO_SERVO_ENABLE_TK_PREVIEW=1` if your setup is stable.

Use `--backend heuristic` if you want a lightweight local vision fallback without loading open-vocabulary weights.

## Notes

- The Panda asset is loaded from the local `reference/` checkout when present.
- If the reference asset is missing, the loader falls back to a small built-in arm scene so tests still run.
- The open-vocabulary backend is optional and is not required for the smoke tests.
- On macOS, camera access still depends on the terminal/process permission prompt from the operating system.
