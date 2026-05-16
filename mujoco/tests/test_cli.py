from __future__ import annotations

import argparse

import pytest

from ._bootstrap import SRC  # noqa: F401

from mujoco_servo.cli import build_parser, config_from_args


def test_cli_config_rejects_invalid_camera_fps() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--camera-fps", "0"])
    with pytest.raises(argparse.ArgumentTypeError, match="camera-fps"):
        config_from_args(args)


def test_cli_config_preserves_small_positive_camera_fps() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--camera-fps", "0.25"])
    config = config_from_args(args)
    assert config.camera_fps == 0.25


def test_cli_config_rejects_nonfinite_camera_fps() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--camera-fps", "nan"])
    with pytest.raises(argparse.ArgumentTypeError, match="camera-fps"):
        config_from_args(args)


def test_cli_config_rejects_invalid_overlay_fraction() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--overlay-width-frac", "2.0"])
    with pytest.raises(argparse.ArgumentTypeError, match="overlay-width-frac"):
        config_from_args(args)


def test_cli_config_preserves_valid_large_overlay_fraction() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--overlay-width-frac", "0.9"])
    config = config_from_args(args)
    assert config.overlay_width_fraction == 0.9


def test_cli_config_rejects_nonfinite_standoff() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--standoff-cm", "inf"])
    with pytest.raises(argparse.ArgumentTypeError, match="standoff-cm"):
        config_from_args(args)


def test_cli_config_rejects_negative_standoff() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--standoff-cm", "-1"])
    with pytest.raises(argparse.ArgumentTypeError, match="standoff-cm"):
        config_from_args(args)
