# MuJoCo visual-servo status and roadmap

## Scope

`mujoco/` is the active project. `matlab/` is frozen historical work and excluded from implementation and acceptance.

The current simulator provides target position access, color and semantic RGB-D tracking, replaceable robot and target assets, three actuator modes, physical target contact, executable grasp points, reusable episode APIs, benchmarks, and CI.

## Implemented baseline

### Simulation API

- `VisualServoSimulation.reset()`, `step()`, `observe()`/`get_state()`, and idempotent `close()`.
- Context-manager cleanup.
- Named body/site world-position reads.
- State and summary metrics for truth/detected positions, tracking loss/reacquisition, timing, depth quality, saturation, RMS/P95 error, and settling time.

### Perception and depth

- Default controlled-scene color segmentation.
- Optional Grounding DINO + SAM semantic detection with local tracking.
- MuJoCo metric depth and optional Depth Anything V2.
- No truth fallback in visual modes; stale/lost observations produce a latched hold.
- Explicit reacquisition confirmation, observation latency/jitter/drop simulation, and camera RGB/depth noise/dropout.
- CUDA, then Apple MPS, then CPU automatic model-device selection.

### Camera and environment

- Robot-workspace-aware fixed camera.
- Optional body-mounted eye-in-hand camera.
- Independently configurable default floor, table, and lights.
- Passive viewer and camera overlay; macOS viewer runs through `mjpython`.

### Robots and control

- Pinned Menagerie Panda, UR5e, and Lite6.
- Version-1 strict custom robot descriptors with legacy unversioned compatibility.
- Resolved-rate Cartesian tracking with adaptive damping, joint-limit avoidance, null-space posture, speed/acceleration limits, and saturation metrics.
- Position, velocity, and torque actuation for all three built-ins.
- Torque bias compensation and joint-space PD; velocity bias-force feed-forward.
- Touch/grasp staged task workflow and deterministic weld/suction attachment abstraction.
- Panda finger actuator metadata plus attachment metadata for gripperless UR5e/Lite6 models.

### Targets

- Primitive, compound, OBJ, and STL target geometry.
- Version-1 strict descriptors with legacy compatibility.
- Visual mocap reference mode and colliding physical free-body mode.
- Body orientation, mass, friction, automatic mesh AABB, and target-local named grasp points.
- Runtime grasp-point world transforms and attachment activation/deactivation.

### Engineering

- Conda and venv installation paths.
- Python 3.10–3.13 EGL CI.
- Ruff lint/format checks, at least 75% branch coverage, wheel build, and installed-wheel smoke test.
- Reproducible multi-robot/multi-trajectory benchmark JSON reports.

## Acceptance

From the repository root:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie
conda env create -f environment.yml
conda activate visual_servo

ruff check mujoco
ruff format --check mujoco/src mujoco/tests mujoco/scripts
coverage run --branch -m pytest -q mujoco/tests
coverage report --fail-under=75
```

Controller smoke:

```bash
python -m mujoco_servo \
  --headless --no-realtime \
  --detector oracle --trajectory static \
  --robot panda --target cup \
  --actuator-mode position --steps 240
```

Benchmark report:

```bash
mujoco-servo-benchmark \
  --robots panda ur5e lite6 \
  --actuator-modes position velocity torque \
  --trajectories static circle \
  --seeds 7 --steps 1200 \
  --output /tmp/mujoco-servo-benchmark.json
```

Benchmark thresholds are configurable gates. Static runs use final error and moving runs use second-half steady-state RMS; `front-standoff` additionally checks orientation. Use `--enforce` after choosing limits appropriate to the selected robot, trajectory, task, and actuator mode.

## Known boundaries

- No real-camera or real-robot driver, ROS/ROS 2 transport, hardware safety layer, or real-system calibration is implemented.
- One selected target is controlled per simulation. Multi-object detection/association and simultaneous multi-target control are deferred.
- Semantic production inference depends on large external weights, prompt quality, cache/network availability, and hardware; routine tests mock the heavy models.
- Eye-in-hand mounting and noise injection do not replace a calibrated lens/distortion and hand-eye calibration model.
- Weld/suction attachment is deterministic task scaffolding, not general grasp synthesis, tactile feedback, slip, or force-closure validation.
- No global collision-aware motion planner or obstacle avoidance layer is present.
- Learned relative monocular depth cannot control metric Cartesian motion unless calibration succeeds.
- Custom robot MJCF must remain self-contained and use compatible scalar joint transmissions; `<include>`, ball/free controlled joints, and arbitrary actuator semantics require adaptation.
- Wheels contain code only. Menagerie and user assets remain external.

## Deferred priorities

1. Define and validate a real hardware/ROS boundary only after selecting an actual robot, camera, and safety architecture.
2. Add calibrated camera distortion, hand-eye uncertainty, and sim-to-real observation models.
3. Add collision-aware planning and physically evaluated grasp quality if manipulation, rather than visual tracking, becomes the primary objective.
4. Add opt-in cached-weight semantic integration jobs on suitable GPU hardware.
5. Add multi-object association only when a concrete multi-target task and metric are defined.
