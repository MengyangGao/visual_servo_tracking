from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mujoco_servo.config import DepthConfig
from mujoco_servo.depth import (
    DepthAnythingV2Backend,
    DepthEstimate,
    _spatial_calibration_split,
    build_depth_backend,
)
from mujoco_servo.perception import CameraIntrinsics, CameraObservation


def _fake_learned_backend(
    result: dict, *, use_hint: bool = True
) -> DepthAnythingV2Backend:
    backend = object.__new__(DepthAnythingV2Backend)
    backend._image_cls = types.SimpleNamespace(fromarray=lambda array: array)
    backend._pipe = lambda image: result
    backend._use_metric_hint = use_hint
    return backend


def test_mujoco_depth_backend_returns_metric_hint() -> None:
    backend = build_depth_backend(DepthConfig(backend="mujoco"))
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    metric = np.full((8, 10), 1.25, dtype=np.float32)
    estimate = backend.estimate(image, metric)
    assert estimate.backend == "mujoco"
    assert estimate.metric
    assert np.allclose(estimate.depth_m, 1.25)
    assert estimate.confidence == 1.0
    assert estimate.valid_fraction == 1.0
    assert estimate.validity == "metric"


def test_mujoco_depth_backend_rejects_shape_mismatch() -> None:
    backend = build_depth_backend(DepthConfig(backend="mujoco"))
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    metric = np.full((7, 10), 1.25, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        backend.estimate(image, metric)


def test_mujoco_depth_backend_does_not_mark_all_nan_hint_metric() -> None:
    backend = build_depth_backend(DepthConfig(backend="mujoco"))
    estimate = backend.estimate(
        np.zeros((8, 10, 3), dtype=np.uint8), np.full((8, 10), np.nan, dtype=np.float32)
    )
    assert not estimate.metric
    assert np.isnan(estimate.depth_m).all()


def test_depth_backend_rejects_non_uint8_frame() -> None:
    backend = build_depth_backend(DepthConfig(backend="none"))
    image = np.zeros((8, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        backend.estimate(image)


def test_depth_anything_backend_can_be_mocked_and_metric_calibrated(
    monkeypatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(), backends=types.SimpleNamespace(mps=FakeMps())
    )
    fake_image = types.SimpleNamespace(fromarray=lambda array: array)
    raw = np.tile(np.linspace(0.2, 1.0, 6, dtype=np.float32), (4, 1))

    def fake_pipeline(task, model, device):
        assert task == "depth-estimation"
        assert model == "fake-metric-depth-anything"
        assert device == 0

        def run(image):
            return {
                "predicted_depth": raw[None, :, :],
                "depth": np.full((4, 6), 255, dtype=np.uint8),
            }

        return run

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline)
    )

    backend = DepthAnythingV2Backend(
        DepthConfig(backend="depth-anything-v2", model="fake-metric-depth-anything")
    )
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    metric_hint = 0.6 + 1.8 * raw
    estimate = backend.estimate(image, metric_hint)
    assert estimate.backend == "depth-anything-v2"
    assert estimate.metric
    assert estimate.depth_m.shape == (4, 6)
    assert np.allclose(estimate.depth_m, metric_hint, atol=1e-5)
    without_hint = backend.estimate(image)
    assert not without_hint.metric


def test_depth_anything_relative_without_hint_is_never_metric() -> None:
    raw = np.tile(np.linspace(0.2, 1.0, 6, dtype=np.float32), (4, 1))
    backend = _fake_learned_backend({"predicted_depth": raw})
    estimate = backend.estimate(np.zeros((4, 6, 3), dtype=np.uint8))
    assert not estimate.metric
    assert np.nanmin(estimate.depth_m) >= 1e-3
    assert np.nanmax(estimate.depth_m) <= 1.0


def test_depth_anything_all_nan_hint_does_not_claim_metric() -> None:
    raw = np.tile(np.linspace(0.2, 1.0, 6, dtype=np.float32), (4, 1))
    backend = _fake_learned_backend({"predicted_depth": raw})
    estimate = backend.estimate(
        np.zeros((4, 6, 3), dtype=np.uint8), np.full((4, 6), np.nan, dtype=np.float32)
    )
    assert not estimate.metric


def test_depth_anything_constant_prediction_fails_metric_calibration() -> None:
    raw = np.ones((4, 6), dtype=np.float32)
    hint = np.tile(np.linspace(0.8, 2.0, 6, dtype=np.float32), (4, 1))
    backend = _fake_learned_backend({"predicted_depth": raw})
    estimate = backend.estimate(np.zeros((4, 6, 3), dtype=np.uint8), hint)
    assert not estimate.metric
    assert np.allclose(estimate.depth_m, 1e-3)


def test_depth_anything_bad_calibration_residual_stays_relative() -> None:
    raw = np.tile(np.linspace(0.2, 1.4, 12, dtype=np.float32), (4, 1))
    hint = 1.4 + 0.5 * np.sin(raw * 13.0)
    backend = _fake_learned_backend({"predicted_depth": raw})
    estimate = backend.estimate(np.zeros((4, 12, 3), dtype=np.uint8), hint)
    assert not estimate.metric


def test_depth_anything_can_fit_inverse_relative_depth() -> None:
    hint = np.tile(np.linspace(0.8, 2.4, 12, dtype=np.float32), (4, 1))
    raw = 1.0 / hint
    backend = _fake_learned_backend({"predicted_depth": raw})
    estimate = backend.estimate(np.zeros((4, 12, 3), dtype=np.uint8), hint)
    assert estimate.metric
    assert np.allclose(estimate.depth_m, hint, atol=1e-4)


def test_depth_calibration_rejects_bad_requested_roi_despite_good_global_fit() -> None:
    raw = np.tile(np.linspace(0.2, 1.2, 20, dtype=np.float32), (8, 1))
    hint = 0.7 + 1.5 * raw
    roi = np.zeros(raw.shape, dtype=bool)
    roi[:, 7] = True
    hint[:, 7] += 0.65
    backend = _fake_learned_backend({"predicted_depth": raw})

    global_estimate = backend.estimate(np.zeros((*raw.shape, 3), dtype=np.uint8), hint)
    roi_estimate = backend.estimate(
        np.zeros((*raw.shape, 3), dtype=np.uint8),
        hint,
        validation_roi=roi,
    )
    assert global_estimate.metric
    assert global_estimate.confidence > 0.0
    assert global_estimate.validity == "metric-calibrated"
    assert not roi_estimate.metric
    assert roi_estimate.confidence == 0.0
    assert roi_estimate.validity == "relative-calibration-rejected"


def test_depth_calibration_validation_roi_shape_is_checked() -> None:
    raw = np.tile(np.linspace(0.2, 1.0, 8, dtype=np.float32), (4, 1))
    backend = _fake_learned_backend({"predicted_depth": raw})
    with pytest.raises(ValueError, match="validation_roi shape"):
        backend.estimate(
            np.zeros((4, 8, 3), dtype=np.uint8),
            0.5 + raw,
            validation_roi=np.ones((3, 8), dtype=bool),
        )


def test_depth_calibration_spatial_holdout_catches_local_failure() -> None:
    raw = np.tile(np.linspace(0.2, 1.2, 20, dtype=np.float32), (20, 1))
    hint = 0.7 + 1.5 * raw
    _, holdout = _spatial_calibration_split(np.ones(raw.shape, dtype=bool))
    corrupt_indices = np.flatnonzero(holdout)[:20]
    hint.flat[corrupt_indices] += 0.8
    backend = _fake_learned_backend({"predicted_depth": raw})
    estimate = backend.estimate(np.zeros((*raw.shape, 3), dtype=np.uint8), hint)
    assert not estimate.metric
    assert estimate.validity == "relative-calibration-rejected"


def test_depth_anything_visualization_output_is_not_treated_as_metric() -> None:
    visualization = np.tile(np.arange(6, dtype=np.uint8), (4, 1))
    hint = 1.0 + visualization.astype(np.float32)
    backend = _fake_learned_backend({"depth": visualization})
    estimate = backend.estimate(np.zeros((4, 6, 3), dtype=np.uint8), hint)
    assert not estimate.metric


def test_app_resolves_pending_learned_depth_off_main_thread_path() -> None:
    from mujoco_servo.app import VisualServoSimulation
    from mujoco_servo.config import DemoConfig

    class FakeDepthBackend:
        name = "depth-anything-v2"

        def estimate(self, frame_bgr, metric_hint_m):
            return DepthEstimate(
                np.asarray(metric_hint_m, dtype=np.float32) + 0.5, self.name, True
            )

    app = VisualServoSimulation(
        DemoConfig(detector="semantic", headless=False, viewer=True, steps=1)
    )
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
