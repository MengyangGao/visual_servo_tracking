from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo import app as app_module
from mujoco_servo.app import VisualServoSimulation
from mujoco_servo.config import CameraConfig, ControllerConfig, DemoConfig
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
    )
    app = VisualServoSimulation(cfg)
    start_target = app.motion.position(0.0)
    start_ee = frame_position(app.scene.model, app.scene.data, app.scene.ee_frame_type, app.scene.ee_frame_name, app.scene.ee_frame_offset)
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
        target="apple",
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


def test_front_standoff_tracks_requested_distance() -> None:
    cfg = DemoConfig(
        target="box",
        trajectory="static",
        detector="oracle",
        steps=240,
        headless=True,
        viewer=False,
        realtime=False,
        controller=ControllerConfig(task="front-standoff", standoff_m=0.12, control_hz=120.0),
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
            return Detection(True, "semantic-test", np.array([0.48, 0.02, 0.34], dtype=float), score=0.9)

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
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.detector == "semantic"
    assert summary.perception_updates == 12
    assert summary.rejected_detections == 0
    assert summary.truth_fallback_steps == 0
    assert summary.oracle_truth_steps == 0


def test_implausible_detection_is_rejected(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(True, "semantic-test", np.array([0.2, 1.2, 0.3], dtype=float), score=0.9)

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
    )
    app = VisualServoSimulation(cfg)
    monkeypatch.setattr(app, "_render_camera_observation", lambda: None)
    summary = app.run()
    assert summary.perception_updates == 0
    assert summary.rejected_detections == 4
    assert summary.hold_steps == 4


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


def test_semantic_viewer_mode_loads_backend_on_main_thread(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(False, self.name, None)

    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(target="apple", trajectory="static", detector="semantic", steps=1, headless=False, viewer=True, realtime=False)
    app = VisualServoSimulation(cfg)
    assert app.perception is not None
    assert app.detector_name == "semantic"


def test_sync_viewer_perception_is_camera_fps_throttled(monkeypatch) -> None:
    class FakeSemantic:
        name = "semantic"

        def detect(self, observation, truth_position, target, prompt):
            return Detection(True, self.name, truth_position.copy(), score=1.0)

    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module, "build_perception", lambda name: FakeSemantic())
    cfg = DemoConfig(target="apple", trajectory="static", detector="semantic", steps=1, headless=True, viewer=False, realtime=False, camera_fps=1.0)
    app = VisualServoSimulation(cfg)
    calls = 0

    def render():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(app, "_render_camera_observation", render)
    assert app._update_perception(object(), np.array([0.4, 0.0, 0.3], dtype=float)) is not None
    assert app._update_perception(object(), np.array([0.4, 0.0, 0.3], dtype=float)) is None
    assert calls == 1


def test_config_validation_rejects_bad_camera_size() -> None:
    cfg = DemoConfig(camera=CameraConfig(width=16, height=320), headless=True, viewer=False)
    with pytest.raises(ValueError, match="camera width"):
        VisualServoSimulation(cfg)


def test_default_detector_is_semantic() -> None:
    assert DemoConfig().detector == "semantic"


def test_viewer_key_controls_use_requested_shortcuts() -> None:
    glfw = pytest.importorskip("glfw")
    cfg = DemoConfig(target="cup", trajectory="static", detector="oracle", steps=1, headless=True, viewer=False, realtime=False)
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

    cfg = DemoConfig(target="cup", trajectory="static", detector="oracle", steps=1, headless=True, viewer=False, realtime=False)
    app = VisualServoSimulation(cfg)
    app._latest_overlay_bgr = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)
    viewer = FakeViewer()
    app._update_viewer_overlay(viewer)
    assert viewer.rect is not None
    assert viewer.rect.left == 568
    assert viewer.rect.bottom == 472
    assert viewer.rect.width == 420
    assert viewer.rect.height == 316
