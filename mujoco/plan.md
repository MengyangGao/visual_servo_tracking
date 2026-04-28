# Objective

Rebuild the `mujoco/` project into a clean MuJoCo visual-servo simulation: a virtual camera observes a modular target object, and a Panda-like 7-DoF robot end-effector smoothly tracks the moving target.

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
- Do not rely on unavailable `reference/` assets.
- Keep push/PR out of scope unless the user explicitly approves later.

# Assumptions

- [ASSUMPTION] A self-contained procedural Panda-like arm is acceptable for this stage because no local MuJoCo Menagerie assets exist.
- [ASSUMPTION] The default high-quality demo should use oracle/simulation perception for stable closed-loop behavior, with image-based color segmentation available as a lightweight detector.
- [ASSUMPTION] Advanced semantic backends such as Grounding DINO / SAM2 should be represented by a pluggable interface and optional dependency path, not required for the base demo.
- [ASSUMPTION] The first acceptance target is contact tracking: the EE center converges to the target center with a small configurable radius.

# Affected Files

- `mujoco/pyproject.toml`
- `mujoco/scripts/demo.py`
- `mujoco/src/mujoco_servo/*`
- `mujoco/tests/*`
- `mujoco/plan.md`

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

# Validation

- `conda run -n visual_servo python --version`
- `conda run -n visual_servo python -m pip install -e mujoco`
- `conda run -n visual_servo pytest mujoco/tests`
- `conda run -n visual_servo python mujoco/scripts/demo.py --headless --steps 240 --target cup --trajectory circle --task contact`

# Overlooked Risks Or Edge Cases

1. macOS MuJoCo rendering may require `mjpython`; the runtime must still work headless without viewer rendering.
2. A procedural arm may have singularities or unreachable target poses; the controller needs damping, velocity limits, and workspace clamping.
3. Color segmentation can fail when lighting/background changes; oracle must remain the default for a reliable demo and tests.

# Risks

- The self-contained robot is Panda-like, not a mesh-accurate Franka Panda.
- Installing MuJoCo/OpenCV into `visual_servo` may require network access.
- Passive viewer validation may need a local GUI run with `mjpython` on macOS.

# Rollback Notes

- All changes are under `mujoco/`.
- If needed, restore prior content from git history.
- No git push or PR will be created without explicit approval.
