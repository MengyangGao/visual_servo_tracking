# MuJoCo visual-servo status and roadmap

## Scope

`mujoco/` is the active project. `matlab/` is frozen historical work and excluded from implementation and acceptance.

The current platform covers RGB-D visual servo, replaceable Menagerie robots and target assets, physical contact grasping, reactive pick-and-place, fixed-base humanoid upper-limb primitives, reusable episode APIs, benchmarks, media recording and CI.

## Implemented system

### Simulation and observability

- `VisualServoSimulation.reset()`, `step()`, `observe()`/`get_state()`, context-manager cleanup and idempotent `close()`.
- Named body/site/grasp-point world-position reads.
- Truth and detected positions, tracking loss/reacquisition, timing, depth quality, saturation, RMS/P95 error, contact evidence, policy attempts, placement error and explicit success/failure metrics.

### Perception and visual control

- Controlled-scene color segmentation and optional Grounding DINO + SAM open-vocabulary detection.
- MuJoCo metric RGB-D and optional Depth Anything V2 with CUDA, Apple MPS, then CPU model-device selection.
- IBVS, PBVS and continuous Hybrid switching; external and eye-in-hand camera roles.
- Segmented point-cloud 6D principal-axis pose and labeled multi-target tracking with bounded occlusion prediction.
- No truth fallback in visual modes. Before grasp, stale observations latch the current joint pose; after verified closure, only the bounded manipulation state may bridge visual occlusion.

### Manipulation policy

- Target-local named executable grasp points transformed into world coordinates.
- Candidate rejection for gripper-width, reach and support-surface clearance violations.
- Explainable ranking by reachability, camera visibility, gripper margin and pregrasp clearance.
- Reactive phases: acquire, pregrasp, approach, close, contact verification, lift, transfer, place, release, retreat, recover and terminal success/failure.
- Bounded close/motion timeouts and attempt budget; slip or failed contact triggers open-gripper upward recovery and replanning.
- Cartesian safety supervisor rejects non-finite commands, clamps workspace and goal distance, and opens/holds on excessive normal force.
- Physical success uses bilateral opposing finger contacts, minimum normal force, bounded relative slip and consecutive stable frames. No target weld or mocap attachment is used.

### Robots and control

- Eleven selectable Menagerie profiles: Panda, FR3, UR5e, UR10e, Lite6, xArm7, iiwa14, Kinova Gen3, Sawyer, and Unitree G1 left/right arms.
- Strict version-1 custom robot descriptors with legacy unversioned compatibility.
- Resolved-rate Cartesian control with adaptive damping, joint-limit avoidance, null-space posture, speed/acceleration limits and saturation metrics.
- Position, velocity, torque and impedance actuator modes.
- `G1BimanualController` composes disjoint left/right 7-DoF arm commands from symmetric visual handover goals. G1 remains fixed-base; balance and locomotion are not implemented.

### Targets and presentation

- Primitive, compound, OBJ and STL geometry; visual mocap or colliding physical free-body dynamics.
- Orientation, mass, friction, mesh AABB and multiple grasp candidates.
- Free MuJoCo viewer camera, robot camera overlay, independent overview camera and 16:9 MP4/MOV/AVI dashboard recording.
- Repository GIF/MP4/PNG evidence for moving visual servo, real-contact grasp and color-vision pick-and-place.

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
python -m build mujoco
```

Physical policy smoke:

```bash
mujoco-servo \
  --headless --no-realtime --scripted-target \
  --robot panda --target grasp-cube --trajectory static \
  --detector oracle --servo-mode pbvs --task pick-place --steps 3200
```

Visual policy evidence (requires a working MuJoCo rendering context):

```bash
mujoco-servo \
  --headless --no-realtime --scripted-target \
  --robot panda --target grasp-cube --trajectory static \
  --detector color --servo-mode pbvs --task pick-place \
  --steps 2400 --camera-fps 24 --record /tmp/pick-place.mp4
```

Benchmark thresholds remain configurable because actuator semantics and workspaces differ by robot. Static runs use final Cartesian error and moving runs use second-half steady-state RMS.

## Known boundaries

- No real-camera or real-robot driver, ROS 2 transport, hardware emergency stop, real-system calibration or certified collision monitor is implemented.
- Reactive Cartesian phases do not replace global collision-aware motion planning in clutter.
- G1 is a fixed-base upper-limb example. There is no gait, balance, foot-contact policy or fall recovery.
- Semantic production inference depends on external weights, prompt quality, cache/network availability and hardware; routine tests mock heavy models.
- Point-cloud PCA orientation is ambiguous for symmetric objects.
- Learned relative monocular depth cannot safely drive metric Cartesian motion unless calibration succeeds.
- New physical objects and grippers require contact/friction/controller retuning and fresh measured validation.
- Custom robot MJCF must be self-contained and use compatible scalar joint transmissions; includes, ball/free controlled joints and arbitrary actuator semantics require adaptation.
- Wheels contain code only. Menagerie and user assets remain external.

## Next priorities

1. Add a global collision-aware planner and scene obstacle representation for cluttered pick-and-place.
2. Add multi-object task allocation and explicit semantic target selection to the policy layer.
3. Add calibrated distortion, hand-eye uncertainty and sim-to-real observation models.
4. Define a ROS 2/hardware boundary only after selecting an actual robot, camera and safety architecture.
5. Extend the fixed-base G1 primitive to whole-body manipulation only with explicit balance, contact and fall-recovery acceptance tests.
