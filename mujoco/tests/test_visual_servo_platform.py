from __future__ import annotations

import numpy as np

from mujoco_servo.app import VisualServoSimulation
from mujoco_servo.config import ROBOT_SPECS, CameraConfig, ControllerConfig, DemoConfig
from mujoco_servo.core import FeatureObservation, ServoMode
from mujoco_servo.humanoid import (
    BimanualGoals,
    BimanualSafetyConfig,
    G1BimanualController,
    symmetric_handover_goals,
)
from mujoco_servo.perception import CameraIntrinsics, CameraObservation
from mujoco_servo.scene import build_scene
from mujoco_servo.servo import VisualServoObjective
from mujoco_servo.targets import TARGETS
from mujoco_servo.vision import (
    LabeledMeasurement,
    MultiTargetTracker,
    align_rotation_to_reference,
    estimate_pose_6d,
    point_interaction_matrix,
    project_world_point,
)


def _observation() -> CameraObservation:
    return CameraObservation(
        frame_bgr=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_m=np.ones((80, 100), dtype=np.float32),
        intrinsics=CameraIntrinsics(100.0, 100.0, 49.5, 39.5, 100, 80),
        camera_position=np.zeros(3),
        camera_xmat=np.eye(3),
        depth_backend="mujoco",
        depth_metric=True,
    )


def test_projection_and_interaction_matrix_are_metric_and_finite() -> None:
    observation = _observation()
    pixel, depth = project_world_point(np.array([0.1, -0.2, -1.0]), observation)
    assert np.allclose(pixel, [59.5, 59.5])
    assert depth == 1.0
    assert np.isfinite(point_interaction_matrix(0.1, -0.2, depth)).all()


def test_ibvs_and_hybrid_close_image_features() -> None:
    observation = _observation()
    feature = FeatureObservation(np.array([70.0, 40.0]), 1.0, 0.9)
    for mode in (ServoMode.IBVS, ServoMode.HYBRID):
        objective = VisualServoObjective(mode=mode).compute(
            ee_position_world=np.array([0.25, 0.0, -1.0]),
            desired_position_world=np.array([0.0, 0.0, -1.0]),
            camera=observation,
            target_feature=feature,
            camera_role="external",
        )
        assert objective.image_error_px > 0.0
        assert np.isfinite(objective.linear_velocity_world).all()
        assert np.linalg.norm(objective.linear_velocity_world) <= 0.45 + 1e-9


def test_six_d_pose_uses_segmented_metric_depth() -> None:
    observation = _observation()
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[25:55, 35:65] = 255
    pose = estimate_pose_6d(observation, mask)
    assert pose is not None
    assert pose.point_count == 900
    assert np.allclose(
        pose.rotation_world.T @ pose.rotation_world, np.eye(3), atol=1e-6
    )
    assert pose.extents_m.min() > 0.0


def test_six_d_pose_axis_ambiguity_is_aligned_to_prior() -> None:
    reference = np.eye(3)
    ambiguous = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])[
        :, [1, 0, 2]
    ]
    ambiguous[:, 2] *= -1.0
    aligned = align_rotation_to_reference(ambiguous, reference)
    assert np.allclose(aligned.T @ aligned, np.eye(3), atol=1e-9)
    assert np.linalg.det(aligned) > 0.0
    assert np.trace(reference.T @ aligned) >= 1.0


def test_multi_target_tracker_predicts_through_short_occlusion() -> None:
    tracker = MultiTargetTracker(occlusion_timeout_s=0.3)
    tracks = tracker.update([LabeledMeasurement("cup", np.zeros(3), 0.9)], 0.1)
    track_id = tracks[0].track_id
    tracker.update([LabeledMeasurement("cup", np.array([0.01, 0.0, 0.0]), 0.9)], 0.1)
    predicted = tracker.update([], 0.1)
    assert predicted[0].track_id == track_id
    assert predicted[0].missed_s == 0.1
    assert tracker.update([], 0.3) == ()


def test_all_menagerie_profiles_compile_into_the_platform_scene() -> None:
    for robot in ROBOT_SPECS.values():
        scene = build_scene(TARGETS["cup"], CameraConfig(), robot)
        assert scene.source == "menagerie"
        assert scene.model.nv >= robot.dof
        assert set(scene.camera_names) == {"servo_camera", "servo_overview"}


def test_g1_bimanual_controller_composes_disjoint_arm_commands() -> None:
    scene = build_scene(TARGETS["cup"], CameraConfig(), ROBOT_SPECS["g1-right-arm"])
    controller = G1BimanualController(
        scene.model,
        ControllerConfig(task="contact", smooth_target_alpha=1.0),
    )
    controller.reset(scene.data)
    goals = symmetric_handover_goals(np.array([0.42, 0.0, 1.05]))
    state = controller.step(scene.data, goals, 0.0, 0, 1.0 / 120.0)
    assert np.isfinite(state.left.qpos_command).all()
    assert np.isfinite(state.right.qpos_command).all()
    assert state.left.qpos_command.shape == (7,)
    assert state.right.qpos_command.shape == (7,)
    left_ids = {
        scene.model.actuator(name).id
        for name in ROBOT_SPECS["g1-left-arm"].actuator_names
    }
    right_ids = {
        scene.model.actuator(name).id
        for name in ROBOT_SPECS["g1-right-arm"].actuator_names
    }
    assert left_ids.isdisjoint(right_ids)
    assert np.isfinite(scene.data.ctrl[list(left_ids | right_ids)]).all()


def test_g1_bimanual_controller_holds_both_arms_on_crossing_path() -> None:
    scene = build_scene(TARGETS["cup"], CameraConfig(), ROBOT_SPECS["g1-right-arm"])
    controller = G1BimanualController(
        scene.model,
        ControllerConfig(task="contact", smooth_target_alpha=1.0),
        BimanualSafetyConfig(
            minimum_hand_separation_m=0.20,
            maximum_goal_speed_mps=100.0,
        ),
    )
    controller.reset(scene.data)
    previous = controller._previous_goals
    assert previous is not None
    crossing = BimanualGoals(previous.right.copy(), previous.left.copy())
    state = controller.step(scene.data, crossing, 0.0, 0, 1.0)
    assert state.safety_limited
    assert "separation" in state.safety_reason


def test_g1_robot_specific_effort_gains_converge_for_both_arms() -> None:
    for robot in ("g1-left-arm", "g1-right-arm"):
        for actuator_mode in ("torque", "impedance"):
            summary = VisualServoSimulation(
                DemoConfig(
                    robot=robot,
                    target="cup",
                    detector="oracle",
                    trajectory="static",
                    steps=600,
                    headless=True,
                    viewer=False,
                    realtime=False,
                    manual_control=False,
                    controller=ControllerConfig(
                        task="standoff", actuator_mode=actuator_mode
                    ),
                )
            ).run()
            assert summary.final_error_m < 0.001
            assert summary.task_succeeded
