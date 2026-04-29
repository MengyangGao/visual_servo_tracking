from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.app import VisualServoSimulation
from mujoco_servo.config import ControllerConfig, DemoConfig
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


def test_semantic_viewer_mode_lazily_loads_backend() -> None:
    cfg = DemoConfig(target="apple", trajectory="static", detector="semantic", steps=1, headless=False, viewer=True, realtime=False)
    app = VisualServoSimulation(cfg)
    assert app.perception is None
    assert app.detector_name == "semantic"


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
    app._latest_overlay_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    viewer = FakeViewer()
    app._update_viewer_overlay(viewer)
    assert viewer.rect is not None
    assert viewer.rect.left == 568
    assert viewer.rect.bottom == 473
    assert viewer.rect.width == 420
    assert viewer.rect.height == 315
