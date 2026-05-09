# Objective

Implement the MuJoCo review findings so the simulator moves toward smooth, general semantic visual servoing with replaceable target objects and replaceable robot arms.
Extend that implementation so semantic perception is the primary path, depth is modeled as an explicit backend, and the user-facing commands document what each mode does.
Complete final acceptance for detector modes, robot/target swapping, and stability regressions; then deliver a clean checkpoint commit.

# User Value

- Robot models can be selected through a small robot specification layer instead of being hard-coded to Franka Panda.
- Visual servo behavior reflects perception availability instead of silently using simulation truth in semantic/color modes.
- Perception reports richer target anchors and state, making future semantic tracking and grasp/contact tasks less brittle.
- Tests cover the new extension points and the most important regressions.
- Semantic-first demos exercise open-vocabulary perception by default while still keeping oracle/color available only as debug tools.
- Depth can come from MuJoCo ground-truth depth in simulation or an optional monocular model backend for camera-only paths.

# Constraints

- Keep changes tightly scoped to `mujoco/` plus this plan.
- Preserve current Panda demo behavior and existing CLI defaults.
- Avoid adding heavyweight runtime dependencies beyond existing optional semantic dependencies.
- Keep tests fast by mocking heavy semantic/depth model calls instead of downloading or running large models in the normal suite.
- Use the existing `visual_servo` Conda environment for validation.
- Do not push or open a PR.
- Keep viewer-related behavior safe on macOS where AppKit calls must stay on the main thread.

# Assumptions

- [ASSUMPTION] The first robot-swap implementation should support the current Panda and at least one additional Menagerie manipulator that can be loaded in tests.
- [ASSUMPTION] External target assets can start as JSON-configured primitive/compound targets; mesh libraries can be layered onto the same interface later.
- [ASSUMPTION] In non-oracle perception modes, holding the last observation or holding the end-effector position is preferable to using truth while detection is pending.
- [ASSUMPTION] Depth Anything V2 via Hugging Face Transformers is the right optional monocular depth backend because the project already uses Transformers for semantic perception.
- [ASSUMPTION] MuJoCo rendered depth remains the best default in simulation because it is metric, fast, deterministic, and testable.

# Affected Files

- `mujoco/src/mujoco_servo/config.py`
- `mujoco/src/mujoco_servo/scene.py`
- `mujoco/src/mujoco_servo/control.py`
- `mujoco/src/mujoco_servo/app.py`
- `mujoco/src/mujoco_servo/perception.py`
- `mujoco/src/mujoco_servo/depth.py`
- `mujoco/src/mujoco_servo/targets.py`
- `mujoco/src/mujoco_servo/cli.py`
- `mujoco/tests/*.py`
- `README.md`

# Steps

1. Add `RobotSpec`, robot registry, CLI robot selection, and scene construction from a selected robot spec.
2. Replace fixed 7-joint/ctrl assumptions in the resolved-rate controller with spec-provided joints and actuators.
3. Remove truth fallback for async semantic/color perception and add command source / perception age fields to summaries.
4. Align controller integration with actual control `dt` derived from MuJoCo substeps.
5. Add target loading from JSON while preserving built-in primitive/compound targets.
6. Extend detection with anchor metadata and improve semantic tracker state with periodic redetection and loss recovery.
7. Add tests for robot specs, non-oracle fallback behavior, delayed perception, custom targets, anchor estimation, and control dimensions.
8. Run full MuJoCo tests and inspect the final diff/status.
9. Make semantic the default detector while preserving explicit `--detector oracle` and `--detector color` debug modes.
10. Add a depth backend module with `mujoco`, `none`, and optional `depth-anything-v2` providers, plus tests using a fake model pipeline.
11. Update README commands to explain semantic/depth combinations and performance controls.
12. Run final acceptance matrix for `oracle`, `color`, `semantic`, robot swaps, and custom target swaps.
13. Fix any discovered regressions in target resolution, control stability, and runtime crash paths.
14. Re-run full tests and create a final local commit.

# Overlooked Risks / Edge Cases

1. Target-name substring collisions (example: `urbox` accidentally matching built-in `box`) can silently select the wrong object.
2. Semantic detector warmup can return zero updates for many steps; control may appear frozen unless hold logic and diagnostics are explicit.
3. Viewer mode plus background model loading on macOS can trigger AppKit thread violations and crash (`NSScreen reconfig must only happen on the main thread`).

# Validation

- `conda run -n visual_servo pytest mujoco/tests`
- Focused smoke checks for Panda default and at least one alternate robot spec through tests.
- Tests for semantic default configuration, mocked semantic detection, mocked monocular depth, and fallback to MuJoCo metric depth.

# Risks

- Alternate Menagerie robots may use different actuator models or missing keyframes, so initial support may need conservative specs.
- Semantic model tests cannot load real GroundingDINO/SAM in routine CI; tracker logic needs mock/object-new tests.
- Removing truth fallback can make non-oracle runs visibly less aggressive before first detection, which is correct but changes demo feel.
- Monocular depth models predict relative depth unless calibrated; for servo control, metric MuJoCo depth should remain preferred in simulation and real metric depth should come from RGB-D hardware when available.
- Semantic-first defaults can fail without optional model dependencies; headless tests should use `--detector oracle` or mocked semantic backends where no model download is intended.

# Rollback Notes

- Revert this commit or restore the changed `mujoco/src/mujoco_servo` files and tests.
- The default CLI should remain `--robot panda`, so rollback should not require data migration.
