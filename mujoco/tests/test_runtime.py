from __future__ import annotations

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.app import VisualServoSimulation
from mujoco_servo.config import ControllerConfig, DemoConfig


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
    start_ee = app.scene.data.site_xpos[app.scene.model.site("ee_site").id].copy()
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
