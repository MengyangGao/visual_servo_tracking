from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.config import DepthConfig
from mujoco_servo.depth import DepthAnythingV2Backend, DepthEstimate, build_depth_backend
from mujoco_servo.perception import CameraIntrinsics, CameraObservation


def test_mujoco_depth_backend_returns_metric_hint() -> None:
    backend = build_depth_backend(DepthConfig(backend="mujoco"))
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    metric = np.full((8, 10), 1.25, dtype=np.float32)
    estimate = backend.estimate(image, metric)
    assert estimate.backend == "mujoco"
    assert estimate.metric
    assert np.allclose(estimate.depth_m, 1.25)


def test_mujoco_depth_backend_rejects_shape_mismatch() -> None:
    backend = build_depth_backend(DepthConfig(backend="mujoco"))
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    metric = np.full((7, 10), 1.25, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        backend.estimate(image, metric)


def test_depth_backend_rejects_nonfinite_float_frame() -> None:
    backend = build_depth_backend(DepthConfig(backend="none"))
    image = np.zeros((8, 10, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="frame_bgr"):
        backend.estimate(image)


def test_depth_anything_backend_can_be_mocked_and_metric_calibrated(monkeypatch) -> None:
    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(mps=FakeMps()))
    fake_image = types.SimpleNamespace(fromarray=lambda array: array)

    def fake_pipeline(task, model, device):
        assert task == "depth-estimation"
        assert model == "fake-depth-anything"

        def run(image):
            return {"depth": np.tile(np.linspace(0.2, 1.0, 6, dtype=np.float32), (4, 1))}

        return run

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))

    backend = DepthAnythingV2Backend(DepthConfig(backend="depth-anything-v2", model="fake-depth-anything"))
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    metric_hint = np.full((4, 6), 2.0, dtype=np.float32)
    estimate = backend.estimate(image, metric_hint)
    assert estimate.backend == "depth-anything-v2"
    assert estimate.metric
    assert estimate.depth_m.shape == (4, 6)
    assert np.isclose(np.median(estimate.depth_m), 2.0, atol=1e-5)


def test_app_resolves_pending_learned_depth_off_main_thread_path() -> None:
    from mujoco_servo.app import VisualServoSimulation
    from mujoco_servo.config import DemoConfig

    class FakeDepthBackend:
        name = "depth-anything-v2"

        def estimate(self, frame_bgr, metric_hint_m):
            return DepthEstimate(np.asarray(metric_hint_m, dtype=np.float32) + 0.5, self.name, True)

    app = VisualServoSimulation(DemoConfig(detector="semantic", headless=False, viewer=True, steps=1))
    app.depth_backend = FakeDepthBackend()
    observation = CameraObservation(
        frame_bgr=np.zeros((4, 6, 3), dtype=np.uint8),
        depth_m=np.ones((4, 6), dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=6.0, fy=6.0, cx=3.0, cy=2.0, width=6, height=4),
        camera_position=np.zeros(3, dtype=float),
        camera_xmat=np.eye(3, dtype=float),
        depth_backend="mujoco-hint",
        depth_metric=True,
    )
    resolved = app._resolve_observation_depth(observation)
    assert resolved.depth_backend == "depth-anything-v2"
    assert resolved.depth_metric
    assert np.allclose(resolved.depth_m, 1.5)
