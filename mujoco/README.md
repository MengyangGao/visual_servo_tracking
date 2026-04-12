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
- `grounded-sam2`: optional open-vocabulary backend scaffold

The code is written so the controller does not depend on backend-specific output shapes.

## Run

Install the package first:

```bash
conda run -n mujoco python -m pip install -e .
```

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

## Notes

- The Panda asset is loaded from the local `reference/` checkout when present.
- If the reference asset is missing, the loader falls back to a small built-in arm scene so tests still run.
- The open-vocabulary backend is scaffolded but not required for the smoke tests.
- On macOS, camera access still depends on the terminal/process permission prompt from the operating system.
