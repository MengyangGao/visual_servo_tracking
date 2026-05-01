# Objective

Implement the MuJoCo review findings so the simulator moves toward smooth, general semantic visual servoing with replaceable target objects and replaceable robot arms.

# User Value

- Robot models can be selected through a small robot specification layer instead of being hard-coded to Franka Panda.
- Visual servo behavior reflects perception availability instead of silently using simulation truth in semantic/color modes.
- Perception reports richer target anchors and state, making future semantic tracking and grasp/contact tasks less brittle.
- Tests cover the new extension points and the most important regressions.

# Constraints

- Keep changes tightly scoped to `mujoco/` plus this plan.
- Preserve current Panda demo behavior and existing CLI defaults.
- Avoid adding heavyweight runtime dependencies beyond existing optional semantic dependencies.
- Use the existing `visual_servo` Conda environment for validation.
- Do not push or open a PR.

# Assumptions

- [ASSUMPTION] The first robot-swap implementation should support the current Panda and at least one additional Menagerie manipulator that can be loaded in tests.
- [ASSUMPTION] External target assets can start as JSON-configured primitive/compound targets; mesh libraries can be layered onto the same interface later.
- [ASSUMPTION] In non-oracle perception modes, holding the last observation or holding the end-effector position is preferable to using truth while detection is pending.

# Affected Files

- `mujoco/src/mujoco_servo/config.py`
- `mujoco/src/mujoco_servo/scene.py`
- `mujoco/src/mujoco_servo/control.py`
- `mujoco/src/mujoco_servo/app.py`
- `mujoco/src/mujoco_servo/perception.py`
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

# Validation

- `conda run -n visual_servo pytest mujoco/tests`
- Focused smoke checks for Panda default and at least one alternate robot spec through tests.

# Risks

- Alternate Menagerie robots may use different actuator models or missing keyframes, so initial support may need conservative specs.
- Semantic model tests cannot load real GroundingDINO/SAM in routine CI; tracker logic needs mock/object-new tests.
- Removing truth fallback can make non-oracle runs visibly less aggressive before first detection, which is correct but changes demo feel.

# Rollback Notes

- Revert this commit or restore the changed `mujoco/src/mujoco_servo` files and tests.
- The default CLI should remain `--robot panda`, so rollback should not require data migration.
