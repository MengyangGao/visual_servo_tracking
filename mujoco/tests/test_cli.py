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


def test_cli_config_rejects_negative_standoff() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--standoff-cm", "-1"])
    with pytest.raises(argparse.ArgumentTypeError, match="standoff-cm"):
        config_from_args(args)
