# MuJoCo visual-servo implementation status

## Scope

`mujoco/` is the active project. The MATLAB implementation is frozen historical work and is not part of this plan's implementation or acceptance scope.

The current objective is a configurable MuJoCo RGB-D visual-servo simulator that can:

1. expose simulator positions;
2. track controlled targets by color;
3. track open-vocabulary targets with optional semantic models;
4. replace the robot through validated model descriptors;
5. replace the visual target through validated primitive, compound, or mesh descriptors.

The default demo is `panda + cup + color + MuJoCo metric depth + front-standoff`.

## Implemented

### Simulator state

- `VisualServoSimulation.get_state()` returns simulator time, target, end-effector, camera, joints, and last detection state.
- `get_body_position(name)` and `get_site_position(name)` expose named MuJoCo world positions.
- `RunSummary` includes final truth target, EE, detected position, detection anchor, target distance, task error, and orientation error.
- Oracle truth is read back from `target_site`; it is not copied directly from the trajectory command.

### Perception and depth

- `color` is the default lightweight backend and segments the configured target color from rendered RGB.
- `semantic` lazily loads Grounding DINO and SAM, then reuses a local mask/color/depth tracker between redetections.
- `oracle` reports the simulated target center and is reserved for controller/debug acceptance.
- Color and semantic backends estimate a visible RGB-D surface anchor.
- MuJoCo metric depth is the default. Depth Anything V2 is optional and can attempt metric calibration from a MuJoCo hint.
- Non-metric depth does not produce a control target.
- Camera sampling is throttled in viewer and headless modes.
- A configurable detection timeout discards stale visual targets and causes joint hold; non-oracle modes do not fall back to simulator truth.

### Controller and tasks

- Resolved-rate Cartesian position control uses damped pseudoinverses, joint/EE speed limits, joint-limit clamping, and a null-space home posture.
- `front-standoff` is the default task and aligns the robot descriptor's local `tool_axis` toward the target while maintaining horizontal distance.
- `contact`, general `standoff`, and `align-x/y/z` remain available.
- The controller accepts named scalar hinge/slide joints and joint-position actuator commands.

### Robot replacement

- Built-ins: Menagerie Panda, UR5e, and Lite6.
- `MUJOCO_MENAGERIE_PATH` overrides the source-tree Menagerie root.
- `--robot-file` accepts a strict JSON robot object, list, or `{"robots": [...]}` wrapper.
- Relative XML and asset paths resolve from the descriptor.
- The loader validates required/unknown fields, lengths, duplicates, finite values, paths, EE frame type, passive controls, workspace bounds, optional world base position, and non-zero normalized tool axis.
- Custom names and aliases take priority over built-ins.
- Scene injection rejects MJCF `<include>` and requires a self-contained `<mujoco>` root.

### Target replacement

- Built-in primitive and compound target library with strict name/phrase resolution.
- `--target-file` accepts a strict list or `{"targets": [...]}` wrapper.
- Supported target geometry: box, sphere, cylinder, capsule, compound parts, and external OBJ/STL mesh targets or parts.
- Relative mesh paths resolve from the target descriptor; dimensions, scales, colors, quaternions, duplicates, and unknown fields are validated.
- `--prompt` separates semantic wording from selected simulated geometry.

### Runtime and viewer

- Robot-workspace-aware side framing for a world-fixed RGB-D camera, plus a passive MuJoCo viewer; an explicit non-default `CameraConfig` pose is preserved.
- Top-right overlay displays RGB, mask, box, centroid, backend, anchor, score, prompt label, and depth status.
- Standard viewer camera mouse controls remain intact.
- Arrow/comma/period keyboard target motion is workspace-clamped when robot bounds are available.
- macOS viewer launch is supported through `mjpython`; other platforms use standard Python.

### Packaging and tests

- Package version: `0.3.0`, Python `>=3.10`.
- Base dependencies contain MuJoCo, NumPy, and OpenCV.
- `test` extra installs pytest; `semantic` adds PyTorch, Pillow, and Transformers 4.x.
- Tests cover configuration validation, descriptor parsing, scene/model replacement, target schemas, position APIs, control behavior, perception/depth units, CLI errors, and runtime smoke paths.

## Acceptance commands

From the repository root:

```bash
git submodule update --init --recursive mujoco/vendor/mujoco_menagerie

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e "mujoco[test]"
python -m pytest mujoco/tests
```

Display-independent controller and position smoke test:

```bash
python mujoco/scripts/demo.py \
  --headless \
  --detector oracle \
  --robot panda \
  --target cup \
  --trajectory static \
  --steps 240 \
  --no-realtime
```

Color integration test, only where offscreen OpenGL is available:

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

Semantic integration requires:

```bash
python -m pip install -e "mujoco[semantic,test]"
```

Then run the color command with `--detector semantic --prompt "red mug"`, allowing Hugging Face model access or providing cached/local model paths.

## Honest limitations

- The camera is automatically framed but remains fixed in the world; eye-in-hand mounting, calibration error, sensor noise, and real-camera input are not implemented.
- RGB-D detections estimate a visible surface, not the occluded object center. Oracle and image-based target positions therefore differ by design.
- Non-metric monocular depth is diagnostic-only and cannot drive the Cartesian controller.
- Target geoms are mocap-controlled and non-colliding. `contact` is center-point tracking, not physical contact, grasping, or force control.
- Color recognition assumes a controlled synthetic scene and representative configured RGBA.
- Semantic recognition depends on model quality, prompt, downloads/cache, hardware, and scene composition; base tests do not download production weights.
- Custom robots must use self-contained MJCF, scalar controlled joints, compatible position actuators, and a meaningful EE descriptor. Reachability and stability cannot be proven from JSON alone.
- Custom target meshes are limited to OBJ/STL and require correct user-provided scale/origin and licensing.
- Wheels contain code only. Menagerie and user assets must remain externally accessible at runtime.
- RGB-D headless runs still need an offscreen OpenGL context; oracle runs do not.
- Physical obstacle avoidance, self-collision-aware planning, grasp synthesis, occlusion reasoning, and real-world transfer are not implemented.

## Next work, if the scope expands

1. Add eye-in-hand camera descriptors and calibrated camera/noise models.
2. Add physical target bodies and contact/force-aware grasp tasks separate from the current visual reference target.
3. Add collision-aware Cartesian planning and singularity/reachability diagnostics for arbitrary robot descriptors.
4. Add opt-in cached-model semantic integration tests on suitable GPU/GL infrastructure.
5. Add model-based object-center estimation when full object geometry and pose are available.
