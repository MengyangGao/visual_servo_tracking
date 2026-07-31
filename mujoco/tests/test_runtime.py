from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from mujoco_servo import app as app_module
from mujoco_servo.app import ManipulationState, TrackingState, VisualServoSimulation
from mujoco_servo.clock import PhaseAccumulatorClock
from mujoco_servo.config import CameraConfig, ControllerConfig, DemoConfig, DepthConfig
from mujoco_servo.perception import Detection
from mujoco_servo.scene import frame_position


def test_headless_demo_reduces_contact_error() -> None:
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="oracle",
        steps=240,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
        camera_fps=1000.0,
    )
    app = VisualServoSimulation(cfg)
    start_target = app.motion.position(0.0)
    start_ee = frame_position(
        app.scene.model,
        app.scene.data,
        app.scene.ee_frame_type,
        app.scene.ee_frame_name,
        app.scene.ee_frame_offset,
    )
    start_error = float(((start_target - start_ee) ** 2).sum() ** 0.5)
    summary = app.run()
    assert summary.steps == 240
    assert summary.robot == "panda"
    assert summary.oracle_truth_steps == 240
    assert summary.truth_fallback_steps == 0
    assert summary.final_error_m < start_error
    assert summary.final_error_m < 0.22


def test_headless_circle_smoke() -> None:
    cfg = DemoConfig(
        target="cup",
        trajectory="circle",
        detector="oracle",
        steps=120,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
    )
    summary = VisualServoSimulation(cfg).run()
    assert summary.steps == 120
    assert summary.final_error_m < summary.max_error_m


def test_contact_task_center_tracking_accepts_wide_visual_target() -> None:
    cfg = DemoConfig(
        robot="panda",
        target="sphere",
        trajectory="static",
        detector="oracle",
        steps=600,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
    )
    summary = VisualServoSimulation(cfg).run()
    assert summary.final_error_m < 0.03


def test_standoff_task_accepts_target_too_wide_for_panda_gripper() -> None:
    cfg = DemoConfig(
        robot="panda",
        target="sphere",
        trajectory="static",
        detector="oracle",
        steps=1,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(
            task="front-standoff", standoff_m=0.12, control_hz=120.0
        ),
    )
    summary = VisualServoSimulation(cfg).run()
    assert summary.steps == 1


def test_front_standoff_tracks_requested_distance() -> None:
    cfg = DemoConfig(
        target="box",
        trajectory="static",
        detector="oracle",
        steps=240,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(
            task="front-standoff", standoff_m=0.12, control_hz=120.0
        ),
    )
    summary = VisualServoSimulation(cfg).run()
    assert summary.steps == 240
    assert summary.final_error_m < 0.02
    assert abs(summary.final_target_distance_m - 0.12) < 0.02


def test_non_oracle_without_detection_holds_end_effector(monkeypatch) -> None:
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="color",
        steps=8,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
        camera_fps=1000.0,
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_update_perception", lambda viewer, truth_position: None)
    summary = app.run()
    assert summary.hold_steps == 8
    assert summary.truth_fallback_steps == 0
    assert summary.oracle_truth_steps == 0
    assert summary.perception_updates == 0
    assert summary.final_error_m > 0.05


def test_semantic_mock_detection_drives_without_truth_fallback(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(
                True,
                "semantic-test",
                np.array([0.48, 0.02, 0.34], dtype=float),
                score=0.9,
            )

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="semantic",
        steps=12,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
        camera_fps=1000.0,
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.detector == "semantic"
    assert summary.perception_updates == 12
    assert summary.rejected_detections == 0
    assert summary.truth_fallback_steps == 0
    assert summary.oracle_truth_steps == 0


@pytest.mark.parametrize(
    ("robot", "target"), [("panda", "cup"), ("ur5e", "box"), ("lite6", "apple")]
)
def test_color_detector_drives_moving_visible_target(robot, target) -> None:
    cfg = DemoConfig(
        robot=robot,
        target=target,
        trajectory="circle",
        detector="color",
        steps=12,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
    )
    try:
        summary = VisualServoSimulation(cfg).run()
    except RuntimeError as exc:
        if "camera rendering is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise
    assert summary.perception_updates > 0
    assert summary.hold_steps == 0
    assert summary.truth_fallback_steps == 0


def test_implausible_detection_is_rejected(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(
                True, "semantic-test", np.array([0.2, 1.2, 0.3], dtype=float), score=0.9
            )

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="semantic",
        steps=4,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
        camera_fps=1000.0,
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.perception_updates == 0
    assert summary.rejected_detections == 4
    assert summary.hold_steps == 4


def test_malformed_detection_position_is_rejected(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(True, "semantic-test", np.array([0.4, 0.2]), score=0.9)

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="semantic",
        steps=3,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
        camera_fps=1000.0,
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.perception_updates == 0
    assert summary.rejected_detections == 3
    assert summary.hold_steps == 3


def test_alternate_robot_headless_smoke() -> None:
    cfg = DemoConfig(
        robot="ur5e",
        target="box",
        trajectory="static",
        detector="oracle",
        steps=20,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="contact", control_hz=120.0),
    )
    summary = VisualServoSimulation(cfg).run()
    assert summary.robot == "ur5e"
    assert summary.steps == 20


def test_oracle_moving_target_moves_each_robot() -> None:
    for robot in ("panda", "ur5e", "lite6"):
        cfg = DemoConfig(
            robot=robot,
            target="cup",
            trajectory="circle",
            detector="oracle",
            steps=90,
            headless=True,
            viewer=False,
            realtime=False,
            controller=ControllerConfig(task="contact", control_hz=120.0),
        )
        app = VisualServoSimulation(cfg)
        start_qpos = np.array(app.scene.data.qpos[:], dtype=float)
        start_ee = frame_position(
            app.scene.model,
            app.scene.data,
            app.scene.ee_frame_type,
            app.scene.ee_frame_name,
            app.scene.ee_frame_offset,
        )
        summary = app.run()
        end_qpos = np.array(app.scene.data.qpos[:], dtype=float)
        end_ee = frame_position(
            app.scene.model,
            app.scene.data,
            app.scene.ee_frame_type,
            app.scene.ee_frame_name,
            app.scene.ee_frame_offset,
        )
        assert summary.steps == 90
        assert np.linalg.norm(end_qpos - start_qpos) > 1e-3
        assert np.linalg.norm(end_ee - start_ee) > 1e-3


def test_front_standoff_converges_and_faces_target_for_each_robot() -> None:
    for robot in ("panda", "ur5e", "lite6"):
        summary = VisualServoSimulation(
            DemoConfig(
                robot=robot,
                target="box",
                trajectory="static",
                detector="oracle",
                steps=1200,
                headless=True,
                viewer=False,
                realtime=False,
                controller=ControllerConfig(task="front-standoff", standoff_m=0.16),
            )
        ).run()
        assert summary.final_error_m < 0.035, robot
        assert abs(summary.final_target_distance_m - 0.16) < 0.02, robot
        assert summary.final_orientation_error_rad < 0.60, robot


def test_semantic_viewer_mode_lazily_loads_backend(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(False, self.name, None)

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="semantic",
        steps=1,
        headless=False,
        viewer=True,
        realtime=False,
    )
    app = VisualServoSimulation(cfg)
    assert app.perception is None
    assert app._ensure_perception() is not None
    assert app.detector_name == "semantic"


def test_sync_viewer_perception_is_camera_fps_throttled(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(True, self.name, truth_position.copy(), score=1.0)

    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="semantic",
        steps=1,
        headless=True,
        viewer=False,
        realtime=False,
        camera_fps=1.0,
    )
    app = VisualServoSimulation(cfg)
    calls = 0

    def render():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(app, "_render_camera_observation", render)
    assert (
        app._update_perception(object(), np.array([0.4, 0.0, 0.3], dtype=float))
        is not None
    )
    assert (
        app._update_perception(object(), np.array([0.4, 0.0, 0.3], dtype=float)) is None
    )
    assert calls == 1


def test_headless_perception_uses_simulation_time_camera_rate(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(True, self.name, truth_position.copy(), score=1.0)

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="semantic",
        steps=50,
        headless=True,
        viewer=False,
        realtime=False,
        camera_fps=6.0,
        controller=ControllerConfig(task="contact"),
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.perception_updates == 3


def test_semantic_prompt_is_independent_from_target_model(monkeypatch) -> None:
    prompts: list[str] = []

    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            prompts.append(prompt)
            return Detection(True, self.name, truth_position.copy(), score=1.0)

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    app = VisualServoSimulation(
        DemoConfig(
            target="cup",
            perception_prompt="red drinking vessel",
            trajectory="static",
            detector="semantic",
            steps=1,
            headless=True,
            viewer=False,
            realtime=False,
            camera_fps=1000.0,
        )
    )
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    app.run()
    assert prompts == ["red drinking vessel"]


def test_stale_visual_detection_expires_and_holds(monkeypatch) -> None:
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="color",
        steps=12,
        headless=True,
        viewer=False,
        realtime=False,
        detection_timeout_s=0.02,
        controller=ControllerConfig(task="contact"),
    )
    app = VisualServoSimulation(cfg)
    calls = 0

    def detect_once(viewer, truth_position):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Detection(True, "color-test", truth_position.copy(), score=1.0)
        return None

    monkeypatch.setattr(app, "_update_perception", detect_once)
    summary = app.run()
    assert summary.perception_updates == 1
    assert summary.hold_steps >= 8
    assert summary.truth_fallback_steps == 0


def test_public_state_age_uses_last_accepted_detection_and_hides_stale_position() -> (
    None
):
    app = VisualServoSimulation(
        DemoConfig(
            detector="color",
            steps=0,
            headless=True,
            viewer=False,
            realtime=False,
            detection_timeout_s=0.1,
        )
    )
    accepted = Detection(True, "color-test", np.array([0.5, 0.1, 0.4]), score=1.0)
    app._last_accepted_detection = accepted
    app._last_accepted_detection_position = accepted.target_position.copy()
    app._last_accepted_detection_sim_time = 0.0
    app._last_detection_wall_time = (
        1.0e20  # A later rejected raw detection must not refresh accepted age.
    )
    app.scene.data.time = 1.0
    state = app.get_state()
    assert np.isclose(state.detection_age_s, 1.0)
    assert state.detected_position is None
    assert state.detection_backend is None


def test_public_state_reads_positions_from_mujoco() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            target="cup",
            trajectory="static",
            detector="oracle",
            steps=2,
            headless=True,
            viewer=False,
            realtime=False,
        )
    )
    summary = app.run()
    state = app.get_state()
    assert np.allclose(state.target_position, app.get_site_position("target_site"))
    assert state.target_position.shape == (3,)
    assert state.end_effector_position.shape == (3,)
    assert state.camera_position.shape == (3,)
    assert state.joint_positions.shape == (app.robot.dof,)
    assert np.allclose(summary.final_target_position, state.target_position)
    assert np.allclose(summary.final_end_effector_position, state.end_effector_position)


def test_default_camera_frames_each_robot_workspace_and_custom_pose_is_preserved() -> (
    None
):
    ur_app = VisualServoSimulation(
        DemoConfig(
            robot="ur5e",
            trajectory="static",
            detector="oracle",
            steps=0,
            headless=True,
            viewer=False,
            realtime=False,
        )
    )
    assert np.allclose(ur_app.camera.lookat, ur_app.motion.position(0.0))
    offset = np.asarray(ur_app.camera.position) - np.asarray(ur_app.camera.lookat)
    approach = np.asarray(ur_app.camera.lookat)[:2]
    assert np.isclose(np.linalg.norm(offset[:2]), 1.2)
    assert np.isclose(np.dot(offset[:2], approach), 0.0, atol=1e-9)
    assert offset[1] <= 0.0
    assert np.isclose(offset[2], 0.7)

    custom = CameraConfig(position=(2.0, -2.0, 1.5), lookat=(0.0, 0.0, 0.2))
    custom_app = VisualServoSimulation(
        DemoConfig(
            camera=custom,
            detector="oracle",
            steps=0,
            headless=True,
            viewer=False,
            realtime=False,
        )
    )
    assert custom_app.camera == custom


def test_runtime_loads_custom_robot_descriptor(tmp_path) -> None:
    robot_xml = tmp_path / "one_link.xml"
    robot_xml.write_text(
        """
        <mujoco model="one-link">
          <worldbody>
            <body name="base">
              <body name="link" pos="0 0 0.2">
                <joint name="joint" type="hinge" axis="0 0 1" range="-2 2"/>
                <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02"/>
                <site name="tool" pos="0.25 0 0"/>
              </body>
            </body>
          </worldbody>
          <actuator><position name="joint_position" joint="joint" kp="50"/></actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    descriptor = tmp_path / "robots.json"
    descriptor.write_text(
        json.dumps(
            {
                "name": "one-link",
                "xml_path": "one_link.xml",
                "asset_dir": ".",
                "joint_names": ["joint"],
                "actuator_names": ["joint_position"],
                "home_qpos": [0.0],
                "ee_frame": {"name": "tool", "type": "site"},
                "default_target_position": [0.35, 0.0, 0.25],
                "detection_bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    app = VisualServoSimulation(
        DemoConfig(
            robot="one-link",
            robot_file=str(descriptor),
            target="cup",
            trajectory="static",
            detector="oracle",
            steps=1,
            headless=True,
            viewer=False,
            realtime=False,
            controller=ControllerConfig(task="standoff"),
        )
    )
    summary = app.run()
    assert summary.robot == "one-link"
    assert state_is_finite(app.get_state())


def state_is_finite(state) -> bool:
    return bool(
        np.isfinite(state.target_position).all()
        and np.isfinite(state.end_effector_position).all()
        and np.isfinite(state.joint_positions).all()
    )


def test_config_validation_rejects_bad_camera_size() -> None:
    cfg = DemoConfig(
        camera=CameraConfig(width=16, height=320), headless=True, viewer=False
    )
    with pytest.raises(ValueError, match="camera width"):
        VisualServoSimulation(cfg)


def test_config_validation_rejects_unknown_depth_backend() -> None:
    cfg = DemoConfig(depth=DepthConfig(backend="made-up"), headless=True, viewer=False)
    with pytest.raises(ValueError, match="depth backend"):
        VisualServoSimulation(cfg)


def test_config_validation_rejects_unknown_depth_device() -> None:
    cfg = DemoConfig(depth=DepthConfig(device="gpu"), headless=True, viewer=False)
    with pytest.raises(ValueError, match="depth device"):
        VisualServoSimulation(cfg)


def test_zero_step_run_reports_zero_completed_steps() -> None:
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="oracle",
        steps=0,
        headless=True,
        viewer=False,
        realtime=False,
    )
    summary = VisualServoSimulation(cfg).run()
    assert summary.steps == 0
    assert np.isfinite(summary.final_error_m)


def test_default_path_uses_color_visual_front_standoff() -> None:
    assert DemoConfig().detector == "color"
    assert DemoConfig().controller.task == "standoff"


def test_viewer_key_controls_use_requested_shortcuts() -> None:
    glfw = pytest.importorskip("glfw")
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="oracle",
        steps=1,
        headless=True,
        viewer=False,
        realtime=False,
    )
    app = VisualServoSimulation(cfg)
    app._handle_key(glfw.KEY_PERIOD)
    assert app._manual_target_velocity[2] > 0.0
    app._handle_key(glfw.KEY_COMMA)
    assert app._manual_target_velocity[2] < 0.0


def test_scripted_target_disables_keyboard_offsets() -> None:
    glfw = pytest.importorskip("glfw")
    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="oracle",
        steps=1,
        headless=True,
        viewer=False,
        realtime=False,
        manual_control=False,
    )
    app = VisualServoSimulation(cfg)
    app._handle_key(glfw.KEY_UP)
    assert np.allclose(app._manual_target_velocity, 0.0)
    assert np.allclose(app._target_position(0.0), app.motion.position(0.0))


def test_camera_overlay_uses_top_right_viewport_origin() -> None:
    @dataclass
    class Viewport:
        width: int
        height: int

    class FakeViewer:
        viewport = Viewport(width=1000, height=800)

        def __init__(self) -> None:
            self.rect = None
            self.image = None

        def set_images(self, viewport_image) -> None:
            self.rect, self.image = viewport_image

    cfg = DemoConfig(
        target="cup",
        trajectory="static",
        detector="oracle",
        steps=1,
        headless=True,
        viewer=False,
        realtime=False,
    )
    app = VisualServoSimulation(cfg)
    app._latest_overlay_bgr = np.zeros(
        (cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8
    )
    viewer = FakeViewer()
    app._update_viewer_overlay(viewer)
    assert viewer.rect is not None
    assert viewer.rect.left == 568
    assert viewer.rect.bottom == 472
    assert viewer.rect.width == 420
    assert viewer.rect.height == 316


def test_camera_overlay_degrades_gracefully_for_legacy_viewer_api() -> None:
    cfg = DemoConfig(
        detector="oracle", steps=0, headless=True, viewer=False, realtime=False
    )
    app = VisualServoSimulation(cfg)
    app._latest_overlay_bgr = np.zeros(
        (cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8
    )
    app._update_viewer_overlay(object())


def test_phase_accumulator_preserves_noninteger_control_rate() -> None:
    clock = PhaseAccumulatorClock(physics_dt_s=0.002, control_hz=120.0)
    ticks = [clock.next_tick() for _ in range(1200)]
    assert {tick.substeps for tick in ticks} == {4, 5}
    assert sum(tick.substeps for tick in ticks) == 5000
    assert np.isclose(clock.effective_hz(), 120.0)


def test_public_lifecycle_supports_step_reset_repeat_and_context_manager() -> None:
    cfg = DemoConfig(
        detector="oracle",
        trajectory="static",
        steps=4,
        headless=True,
        viewer=False,
        realtime=False,
        manual_control=False,
    )
    with VisualServoSimulation(cfg) as app:
        initial = app.observe()
        stepped = app.step()
        assert stepped.time_s > initial.time_s
        reset = app.reset()
        assert np.isclose(reset.time_s, initial.time_s)
        first = app.run()
        second = app.run()
        assert first.steps == second.steps == 4
        assert second.simulated_duration_s > 0.0
        assert state_is_finite(app.observe())


def test_lost_target_requires_confirmed_reacquisition_and_accepts_large_jump(
    monkeypatch,
) -> None:
    cfg = DemoConfig(
        detector="color",
        trajectory="static",
        steps=10,
        headless=True,
        viewer=False,
        realtime=False,
        manual_control=False,
        detection_timeout_s=0.01,
        reacquire_confirm_frames=3,
    )
    app = VisualServoSimulation(cfg)
    calls = 0

    def scripted_detection(viewer, truth_position):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Detection(True, "scripted", truth_position.copy(), score=1.0)
        if calls <= 3:
            return None
        moved = truth_position + np.array([0.0, -0.25, 0.0])
        return Detection(True, "scripted", moved, score=1.0)

    monkeypatch.setattr(app, "_update_perception", scripted_detection)
    summary = app.run()
    assert summary.lost_events == 1
    assert summary.reacquire_events == 1
    assert summary.hold_steps >= 3
    assert summary.tracking_state == TrackingState.TRACKING.value


def test_hold_reference_is_latched_once() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            detector="color", steps=0, headless=True, viewer=False, realtime=False
        )
    )
    app.controller.begin_hold(app.scene.data)
    reference = app.controller._hold_qpos.copy()
    app.scene.data.qpos[app.controller._qpos_adr] += 0.01
    app.controller.hold(app.scene.data, 0.0, 0, 0.01)
    assert np.array_equal(app.controller._hold_qpos, reference)
    assert np.array_equal(app.controller._qpos_command, reference)


@pytest.mark.parametrize("actuator_mode", ["position", "velocity", "torque"])
def test_all_actuator_modes_produce_finite_commands(actuator_mode) -> None:
    app = VisualServoSimulation(
        DemoConfig(
            detector="oracle",
            trajectory="static",
            steps=60,
            headless=True,
            viewer=False,
            realtime=False,
            manual_control=False,
            controller=ControllerConfig(task="contact", actuator_mode=actuator_mode),
        )
    )
    summary = app.run()
    assert summary.tracking_state == TrackingState.TRACKING.value
    assert np.isfinite(app.scene.data.ctrl).all()
    assert np.isfinite(summary.rms_error_m)


def test_summary_reports_requested_and_long_run_effective_rate() -> None:
    summary = VisualServoSimulation(
        DemoConfig(
            detector="oracle",
            trajectory="static",
            steps=1200,
            headless=True,
            viewer=False,
            realtime=False,
            manual_control=False,
            controller=ControllerConfig(control_hz=120.0),
        )
    ).run()
    assert summary.requested_control_hz == 120.0
    assert np.isclose(summary.effective_control_hz, 120.0)
    assert np.isclose(summary.simulated_duration_s, 10.0)
    assert summary.wall_duration_s > 0.0
    assert summary.perception_device == "simulator"
    assert summary.depth_device == "simulator"


def test_configured_perception_latency_delays_availability(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(True, self.name, truth_position.copy(), score=1.0)

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    app = VisualServoSimulation(
        DemoConfig(
            detector="semantic",
            trajectory="static",
            steps=8,
            headless=True,
            viewer=False,
            realtime=False,
            manual_control=False,
            camera_fps=1000.0,
            perception_latency_s=0.02,
        )
    )
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.hold_steps > 0
    assert summary.perception_updates > 0
    assert summary.mean_perception_latency_ms >= 19.9


def test_real_contact_grasp_executes_and_lifts_without_weld() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            robot="panda",
            target="grasp-cube",
            detector="oracle",
            trajectory="static",
            steps=1800,
            headless=True,
            viewer=False,
            realtime=False,
            manual_control=False,
            controller=ControllerConfig(task="grasp"),
        )
    )
    assert app.target.dynamics == "physical"
    summary = app.run()
    assert summary.manipulation_state == ManipulationState.COMPLETE.value
    assert summary.grasped
    assert summary.target_lift_m >= 0.09
    assert summary.contact_steps > 0
    assert app.scene.model.neq == 1  # Panda finger coupling only; no target weld.


def test_panda_gripper_command_remains_closed_after_grasp() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            robot="panda",
            target="grasp-cube",
            detector="oracle",
            trajectory="static",
            steps=1800,
            headless=True,
            viewer=False,
            realtime=False,
            manual_control=False,
            controller=ControllerConfig(task="grasp"),
        )
    )
    summary = app.run()
    assert summary.grasped
    actuator_id = app.scene.model.actuator("actuator8").id
    assert app.scene.data.ctrl[actuator_id] == 0.0


def test_reactive_pick_place_completes_with_physical_contact_and_release() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            robot="panda",
            target="grasp-cube",
            detector="oracle",
            trajectory="static",
            steps=3200,
            headless=True,
            viewer=False,
            realtime=False,
            manual_control=False,
            controller=ControllerConfig(task="pick-place", servo_mode="pbvs"),
        )
    )
    summary = app.run()
    assert summary.manipulation_state == ManipulationState.COMPLETE.value
    assert summary.task_succeeded
    assert not summary.grasped
    assert summary.policy_name == "reactive-pick-place"
    assert summary.policy_phase == "SUCCEEDED"
    assert summary.policy_attempts == 0
    assert summary.selected_grasp == "center"
    assert summary.place_error_m is not None
    assert summary.place_error_m <= 0.035
    assert summary.contact_steps > 0
    assert summary.steps < 3200
    assert summary.termination_reason == "policy_succeeded"
    assert summary.grasp_pose_source == "oracle-6d"
    assert app._grasp_initial_target_z is None
    assert summary.target_lift_m > 0.05
    assert summary.grasp_normal_force_n > 0.0
    assert summary.grasp_relative_slip_m > 0.0


def test_pick_place_rejects_destination_outside_work_surface() -> None:
    with pytest.raises(ValueError, match="work surface"):
        VisualServoSimulation(
            DemoConfig(
                robot="panda",
                target="grasp-cube",
                detector="oracle",
                headless=True,
                viewer=False,
                realtime=False,
                controller=ControllerConfig(
                    task="pick-place", place_position=(0.9, 0.9, 0.25)
                ),
            )
        )


def test_numerical_ik_reachability_uses_live_robot_limits() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            detector="oracle",
            headless=True,
            viewer=False,
            realtime=False,
        )
    )
    current = app.controller.frame_position(app.scene.data)
    assert app.controller.is_position_reachable(app.scene.data, current)
    assert not app.controller.is_position_reachable(
        app.scene.data, np.array([5.0, 5.0, 5.0])
    )
    app.close()


def test_contact_acceptance_thresholds_are_runtime_configuration() -> None:
    app = VisualServoSimulation(
        DemoConfig(
            target="grasp-cube",
            detector="oracle",
            headless=True,
            viewer=False,
            realtime=False,
            controller=ControllerConfig(
                task="grasp",
                grasp_min_normal_force_n=0.4,
                grasp_max_relative_slip_m=0.002,
                grasp_confirmation_frames=12,
            ),
        )
    )
    evaluator = app._grasp_evaluator
    assert evaluator is not None
    assert evaluator.min_normal_force_n == 0.4
    assert evaluator.max_relative_slip_m == 0.002
    assert evaluator.confirmation_frames == 12
    app.close()
