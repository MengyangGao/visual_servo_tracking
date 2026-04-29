# Objective

Rebuild the `mujoco/` project into a clean MuJoCo visual-servo simulation: a virtual camera observes a modular target object, and a Panda-like 7-DoF robot end-effector smoothly tracks the moving target.

Update: replace the procedural fallback robot as the default with Google DeepMind's official MuJoCo Menagerie Franka Emika Panda model, kept as a git submodule.

Update 2: remove the procedural arm fallback entirely, make Menagerie Panda mandatory, add camera-derived semantic perception, expand word-addressable target objects, and add a horizontal front standoff task with CLI distance control.

Update 3: refine viewer interaction by unlocking the camera, hiding perturb visualization by default, adding toggleable mouse target drag, making keyboard target motion continuous, improving initial Panda pose, and throttling semantic perception.

# User Value

- Run a direct demo with `python scripts/demo.py` or `mjpython scripts/demo.py`.
- See a convincing, smooth visual-servo tracking loop in MuJoCo.
- Switch target objects, target motion, perception backend, and servo task mode without rewriting code.
- Keep the core implementation independent from macOS viewer details.

# Constraints

- Only use and modify content under `mujoco/`.
- Use the existing conda environment named `visual_servo`.
- The user explicitly allowed deleting/replacing `src`, `tests`, and `pyproject.toml`.
- Must support Python 3.10 because `visual_servo` is Python 3.10.
- Use MuJoCo Menagerie as the required robot model source.
- Remove the procedural arm fallback so there is a single robot source of truth.
- [ASSUMPTION] Adding `.gitmodules` at the repository root is acceptable because git submodules require it, even though runtime code remains under `mujoco/`.
- Keep push/PR out of scope unless the user explicitly approves later.

# Assumptions

- [ASSUMPTION] The submodule path should be `mujoco/vendor/mujoco_menagerie` so all model assets stay under `mujoco/`.
- [ASSUMPTION] For camera-based semantic perception, the command loop may use rendered RGB-D from the fixed MuJoCo camera to estimate 3D target position from masks/bounding boxes.
- [ASSUMPTION] "EE horizontal and facing the object at x cm" means the gripper tool axis points horizontally toward the target, with the EE placed `x` centimeters away from the target along the horizontal line from robot base to object.
- [ASSUMPTION] MuJoCo viewer mouse camera control should remain the default; target mouse dragging should be explicitly toggled so it does not steal camera interaction.
- [ASSUMPTION] The default high-quality demo should use oracle/simulation perception for stable closed-loop behavior, with image-based color segmentation available as a lightweight detector.
- [ASSUMPTION] Advanced semantic backends such as Grounding DINO / SAM2 should be represented by a pluggable interface and optional dependency path, not required for the base demo.
- [ASSUMPTION] The first acceptance target is contact tracking: the EE center converges to the target center with a small configurable radius.

# Affected Files

- `mujoco/pyproject.toml`
- `mujoco/scripts/demo.py`
- `mujoco/src/mujoco_servo/*`
- `mujoco/tests/*`
- `mujoco/plan.md`
- `mujoco/vendor/mujoco_menagerie`
- `.gitmodules`

# Steps

1. Replace the current package with a compact modular architecture:
   - config/dataclasses
   - target library and trajectories
   - scene generation
   - perception backends
   - resolved-rate controller
   - simulation app/runtime
2. Build a self-contained MJCF scene:
   - procedural 7-DoF arm with position actuators
   - mocap target object
   - fixed virtual camera and visible camera marker
   - EE and target sites for measurements
3. Implement task modes:
   - `contact` default
   - `standoff`
   - `align-x`
   - `align-y`
   - `align-z`
4. Implement target motion modes:
   - static
   - circle
   - figure-eight
   - random walk
   - scripted waypoints
5. Implement perception:
   - oracle pose backend for stable simulation truth
   - color segmentation backend from rendered camera images
   - optional semantic backend stub with clear dependency error
6. Add `scripts/demo.py` CLI:
   - runs headless for tests
   - launches MuJoCo passive viewer when requested
   - prints final metrics
7. Add tests for scene construction, trajectories, task goals, controller convergence smoke, and detector behavior.
8. Validate with the `visual_servo` conda env.
9. Add MuJoCo Menagerie as a submodule at `mujoco/vendor/mujoco_menagerie`.
10. Load `franka_emika_panda/scene.xml` by default and inject the target, camera, and tracking sites.
11. Keep the procedural scene builder as fallback for missing submodule/assets.
12. Extend tests to assert the Menagerie Panda path is used when present.
13. Remove all procedural arm generation code and fail fast if the Menagerie submodule is missing.
14. Expand target library to include primitive and compound target bodies addressed by free-form words.
15. Add camera observation plumbing: rendered RGB, rendered depth, intrinsics, and camera pose.
16. Implement semantic perception using Grounding DINO for open-vocabulary boxes and SAM/SAM2-compatible mask extraction when optional dependencies are installed.
17. Add `front-standoff` task mode with CLI distance in centimeters and 6D pose control.
18. Disable per-frame viewer camera rewrites and only set initial camera once.
19. Disable perturb/select visualization by default; expose a key toggle for target mouse dragging.
20. Replace discrete WASD/QE target nudges with continuous arrow/PageUp/PageDown velocity control.
21. Move the Panda start pose higher and away from the object approach path.
22. Set semantic device selection to auto and throttle expensive Grounding DINO + SAM inference.

# Validation

- `conda run -n visual_servo python --version`
- `conda run -n visual_servo python -m pip install -e mujoco`
- `conda run -n visual_servo pytest mujoco/tests`
- `conda run -n visual_servo python mujoco/scripts/demo.py --headless --steps 240 --target cup --trajectory circle --task contact`
- `git submodule status mujoco/vendor/mujoco_menagerie`

# Overlooked Risks Or Edge Cases

1. macOS MuJoCo rendering may require `mjpython`; the runtime must still work headless without viewer rendering.
2. A procedural arm may have singularities or unreachable target poses; the controller needs damping, velocity limits, and workspace clamping.
3. Color segmentation can fail when lighting/background changes; oracle must remain the default for a reliable demo and tests.
4. Menagerie Panda joint/body/site names differ from the procedural model, so controller discovery must be name-based and tested.
5. Injecting custom world objects into an included Menagerie scene can conflict with duplicate worldbody/default/asset declarations; use a wrapper MJCF include instead of editing upstream XML.
6. Menagerie actuator types/ranges may differ from the fallback model, so commands must map to named Panda actuators rather than assuming actuator order.
7. Grounding DINO/SAM downloads are large and may be slow on first run; tests should validate wiring without requiring model weights.
8. Depth unprojection depends on MuJoCo camera conventions; validate with color detector smoke tests against oracle behavior.
9. Passive viewer key callbacks do not expose key release events, so continuous target motion uses short velocity holds refreshed by key repeat.
10. Mouse target dragging shares MuJoCo's perturb mechanism; keeping it off by default preserves normal camera orbit behavior.

# Risks

- Menagerie is a larger git submodule and increases checkout/update time.
- The robot model now requires the Menagerie submodule to be initialized.
- Semantic perception quality depends on optional model availability and local hardware.
- Installing MuJoCo/OpenCV into `visual_servo` may require network access.
- Passive viewer validation may need a local GUI run with `mjpython` on macOS.

# Rollback Notes

- All changes are under `mujoco/`.
- If needed, restore prior content from git history.
- No git push or PR will be created without explicit approval.
