from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from mujoco_servo import cli as cli_module
from mujoco_servo.cli import build_parser, config_from_args
from mujoco_servo.config import (
    CameraConfig,
    ControllerConfig,
    DemoConfig,
    DepthConfig,
    EnvironmentSpec,
    TargetPart,
    TargetSpec,
    load_robot_specs,
    resolve_config,
    resolve_robot,
    validate_config,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _write_robot_descriptor(
    tmp_path: Path, payload: dict | list | None = None
) -> tuple[Path, dict]:
    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    (tmp_path / "robot.xml").write_text("<mujoco/>", encoding="utf-8")
    descriptor = {
        "name": "testbot",
        "xml_path": "robot.xml",
        "asset_dir": "assets",
        "joint_names": ["joint_a", "joint_b"],
        "actuator_names": ["actuator_a", "actuator_b"],
        "home_qpos": [0.1, -0.2],
        "ee_frame": {"name": "tool_site", "type": "site", "offset": [0.0, 0.0, 0.03]},
        "passive_actuator_ctrl": {"gripper": 0.5},
        "max_gripper_width_m": 0.07,
        "default_target_position": [0.3, 0.1, 0.4],
        "detection_bounds": {"min": [-0.5, -0.4, 0.0], "max": [0.8, 0.6, 1.0]},
        "aliases": ["test-bot"],
        "tool_axis": [0.0, 2.0, 0.0],
        "base_position": [1.0, -0.5, 0.0],
    }
    path = tmp_path / "robot.json"
    path.write_text(
        json.dumps(descriptor if payload is None else payload), encoding="utf-8"
    )
    return path, descriptor


def test_cli_config_rejects_invalid_camera_fps() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--camera-fps", "0"])
    with pytest.raises(argparse.ArgumentTypeError, match="camera-fps"):
        config_from_args(args)


def test_config_rejects_camera_with_zero_view_direction() -> None:
    with pytest.raises(ValueError, match="position and lookat"):
        validate_config(
            DemoConfig(
                camera=CameraConfig(position=(0.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.0))
            )
        )


def test_cli_config_preserves_small_positive_camera_fps() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--camera-fps", "0.25"])
    config = config_from_args(args)
    assert config.camera_fps == 0.25


def test_cli_config_preserves_detection_timeout() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--detection-timeout", "1.25"])
    config = config_from_args(args)
    assert config.detection_timeout_s == 1.25


def test_cli_config_rejects_invalid_detection_timeout() -> None:
    parser = build_parser()
    args = parser.parse_args(["--headless", "--detection-timeout", "nan"])
    with pytest.raises(argparse.ArgumentTypeError, match="detection-timeout"):
        config_from_args(args)


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


def test_cli_defaults_to_color_and_preserves_prompt() -> None:
    config = config_from_args(
        build_parser().parse_args(["--headless", "--prompt", "  red mug  "])
    )
    assert config.detector == "color"
    assert config.controller.task == "standoff"
    assert config.detection_timeout_s == 0.75
    assert config.perception_prompt == "red mug"
    assert config.robot_file is None


def test_cli_accepts_builtin_robot_alias() -> None:
    config = config_from_args(
        build_parser().parse_args(["--headless", "--robot", "franka"])
    )
    assert config.robot == "franka"
    assert resolve_robot(config.robot).name == "panda"


def test_cli_normalizes_depth_device() -> None:
    config = config_from_args(
        build_parser().parse_args(["--headless", "--depth-device", " CPU "])
    )
    assert config.depth.device == "cpu"


def test_removed_semantic_interval_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--semantic-interval", "3"])
    assert exc.value.code == 2


def test_target_mesh_config_fields_keep_existing_constructor_compatible(
    tmp_path: Path,
) -> None:
    target = TargetSpec(
        "mesh",
        "mesh",
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0, 1.0),
        mesh_path=tmp_path / "object.obj",
    )
    part = TargetPart(
        "mesh",
        (0.1, 0.1, 0.1),
        mesh_path=tmp_path / "part.obj",
        mesh_scale=(2.0, 2.0, 2.0),
    )
    assert target.mesh_scale == (1.0, 1.0, 1.0)
    assert part.mesh_scale == (2.0, 2.0, 2.0)


def test_load_robot_specs_resolves_relative_paths_and_all_fields(
    tmp_path: Path,
) -> None:
    path, _ = _write_robot_descriptor(tmp_path)
    spec = load_robot_specs(path)["testbot"]
    assert spec.xml_path == (tmp_path / "robot.xml").resolve()
    assert spec.asset_dir == (tmp_path / "assets").resolve()
    assert spec.joint_names == ("joint_a", "joint_b")
    assert spec.actuator_names == ("actuator_a", "actuator_b")
    assert spec.home_qpos == (0.1, -0.2)
    assert spec.ee_frame_name == "tool_site"
    assert spec.ee_frame_type == "site"
    assert spec.ee_frame_offset == (0.0, 0.0, 0.03)
    assert spec.passive_actuator_ctrl == (("gripper", 0.5),)
    assert spec.max_gripper_width_m == 0.07
    assert spec.default_target_position == (0.3, 0.1, 0.4)
    assert spec.detection_bounds == ((-0.5, -0.4, 0.0), (0.8, 0.6, 1.0))
    assert spec.aliases == ("test-bot",)
    assert spec.tool_axis == (0.0, 1.0, 0.0)
    assert spec.base_position == (1.0, -0.5, 0.0)


def test_builtin_robot_defaults_use_safe_workspaces_and_z_tool_axis() -> None:
    assert resolve_robot("panda").default_target_position == (0.55, 0.10, 0.40)
    assert resolve_robot("lite6").default_target_position == (0.32, 0.0, 0.38)
    assert resolve_robot("ur5e").default_target_position == (-0.30, 0.45, 0.50)
    assert all(
        resolve_robot(name).tool_axis == (0.0, 0.0, 1.0)
        for name in ("panda", "lite6", "ur5e")
    )


def test_load_robot_specs_accepts_wrapper_and_passive_name_value_list(
    tmp_path: Path,
) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    descriptor["passive_actuator_ctrl"] = [{"name": "gripper", "value": 0.25}]
    path, _ = _write_robot_descriptor(tmp_path, {"robots": [descriptor]})
    assert load_robot_specs(path)["testbot"].passive_actuator_ctrl == (
        ("gripper", 0.25),
    )


def test_custom_robot_alias_has_priority_over_builtin(tmp_path: Path) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    descriptor["aliases"] = ["panda"]
    path, _ = _write_robot_descriptor(tmp_path, descriptor)
    custom = load_robot_specs(path)
    assert resolve_robot("panda", custom).name == "testbot"


def test_cli_loads_custom_robot_file_and_prompt(tmp_path: Path) -> None:
    path, _ = _write_robot_descriptor(tmp_path)
    config = config_from_args(
        build_parser().parse_args(
            [
                "--headless",
                "--robot",
                "test-bot",
                "--robot-file",
                str(path),
                "--prompt",
                "blue fixture",
            ]
        )
    )
    assert config.robot == "test-bot"
    assert config.robot_file == str(path)
    assert config.perception_prompt == "blue fixture"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update({"unexpected": 1}), "unknown fields"),
        (
            lambda payload: payload.update({"actuator_names": ["actuator_a"]}),
            "same length",
        ),
        (lambda payload: payload.update({"home_qpos": [0.0]}), "home_qpos"),
        (
            lambda payload: payload.update({"joint_names": ["joint_a", "joint_a"]}),
            "duplicates",
        ),
        (lambda payload: payload.update({"aliases": ["same", "same"]}), "duplicates"),
        (
            lambda payload: payload.update({"max_gripper_width_m": -0.1}),
            "must be positive",
        ),
        (
            lambda payload: payload.update(
                {"detection_bounds": [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]}
            ),
            "strictly less",
        ),
        (
            lambda payload: payload.update(
                {"ee_frame": {"name": "tool", "type": "invalid"}}
            ),
            "ee_frame.type",
        ),
        (
            lambda payload: payload.update(
                {"passive_actuator_ctrl": {"actuator_a": 0.0}}
            ),
            "overlap",
        ),
        (lambda payload: payload.update({"tool_axis": [0.0, 0.0, 0.0]}), "non-zero"),
        (
            lambda payload: payload.update({"base_position": [0.0, 0.0]}),
            "base_position",
        ),
    ],
)
def test_load_robot_specs_rejects_invalid_schema(
    tmp_path: Path, mutator, message: str
) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    mutator(descriptor)
    path, _ = _write_robot_descriptor(tmp_path, descriptor)
    with pytest.raises(ValueError, match=message):
        load_robot_specs(path)


def test_load_robot_specs_rejects_duplicate_tokens_across_robots(
    tmp_path: Path,
) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    other = deepcopy(descriptor)
    other["name"] = "otherbot"
    other["aliases"] = ["test-bot"]
    path, _ = _write_robot_descriptor(tmp_path, [descriptor, other])
    with pytest.raises(ValueError, match="duplicate robot name or alias"):
        load_robot_specs(path)


def test_load_robot_specs_rejects_nonfinite_json_constant(tmp_path: Path) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    descriptor["home_qpos"][0] = float("nan")
    path, _ = _write_robot_descriptor(tmp_path, descriptor)
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_robot_specs(path)


def test_load_robot_specs_rejects_missing_asset_paths(tmp_path: Path) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    descriptor["xml_path"] = "missing.xml"
    path, _ = _write_robot_descriptor(tmp_path, descriptor)
    with pytest.raises(FileNotFoundError, match="xml_path"):
        load_robot_specs(path)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (DemoConfig(robot="unknown"), "unknown robot"),
        (DemoConfig(detector="unknown"), "detector"),
        (DemoConfig(trajectory="unknown"), "trajectory"),
        (DemoConfig(controller=ControllerConfig(task="unknown")), "task"),
        (DemoConfig(steps=-1), "steps"),
        (DemoConfig(steps=1.5), "integer"),
        (DemoConfig(seed=-1), "seed"),
        (DemoConfig(camera_fps=float("nan")), "camera_fps"),
        (DemoConfig(detection_timeout_s=0.0), "detection_timeout_s"),
        (DemoConfig(key_speed_mps=-0.1), "key_speed_mps"),
        (DemoConfig(overlay_width_fraction=float("inf")), "overlay_width_fraction"),
        (DemoConfig(controller=ControllerConfig(position_gain=-1.0)), "position_gain"),
        (
            DemoConfig(controller=ControllerConfig(orientation_gain=float("nan"))),
            "orientation_gain",
        ),
        (
            DemoConfig(controller=ControllerConfig(max_angular_speed=-1.0)),
            "max_angular_speed",
        ),
        (
            DemoConfig(controller=ControllerConfig(align_offset_m=float("nan"))),
            "align_offset_m",
        ),
        (DemoConfig(camera=CameraConfig(fovy_deg=float("nan"))), "fovy_deg"),
        (DemoConfig(depth=DepthConfig(device="CPU")), "normalized lowercase"),
    ],
)
def test_validate_config_rejects_invalid_programmatic_values(
    config: DemoConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_menagerie_home_honors_environment_override(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["MUJOCO_MENAGERIE_PATH"] = str(tmp_path)
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mujoco_servo.config import MENAGERIE_HOME; print(MENAGERIE_HOME)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert Path(result.stdout.strip()) == tmp_path


@pytest.mark.parametrize(
    "error",
    [
        ValueError("runtime config problem"),
        RuntimeError("runtime dependency problem"),
        FileNotFoundError("missing robot asset"),
        OSError("asset read problem"),
        json.JSONDecodeError("invalid descriptor JSON", "{", 0),
    ],
)
def test_cli_main_reports_expected_errors_without_traceback(
    monkeypatch, capsys, error: Exception
) -> None:
    def fail_run(config):
        raise error

    monkeypatch.setattr(cli_module, "run_demo", fail_run)
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["--headless", "--detector", "color"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert "Traceback" not in captured.err


def test_cli_main_reports_robot_file_errors_without_traceback(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as exc:
        cli_module.main(
            ["--headless", "--robot", "custom", "--robot-file", str(missing)]
        )
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err


def test_cli_exposes_actuation_latency_and_environment_controls() -> None:
    args = build_parser().parse_args(
        [
            "--actuator-mode",
            "torque",
            "--control-hz",
            "200",
            "--max-joint-accel",
            "12",
            "--torque-kp",
            "90",
            "--torque-kd",
            "9",
            "--joint-limit-margin",
            "0.03",
            "--grasp-point",
            "top",
            "--grasp-approach",
            "0.09",
            "--grasp-attach-distance",
            "0.07",
            "--grasp-lift",
            "0.15",
            "--grasp-stage-tolerance",
            "0.02",
            "--place-position",
            "0.5",
            "-0.1",
            "0.25",
            "--policy-max-attempts",
            "3",
            "--policy-close-timeout",
            "3.0",
            "--policy-motion-timeout",
            "9.0",
            "--policy-max-force",
            "60.0",
            "--grasp-min-force",
            "0.4",
            "--grasp-max-slip",
            "0.002",
            "--grasp-confirmation-frames",
            "12",
            "--grasp-lost-frames",
            "18",
            "--policy-place-tolerance",
            "0.025",
            "--no-stop-on-terminal",
            "--reacquire-confirm-frames",
            "4",
            "--perception-latency",
            "0.08",
            "--perception-jitter",
            "0.01",
            "--perception-drop-probability",
            "0.2",
            "--settling-threshold",
            "0.005",
            "--no-default-table",
        ]
    )
    config = config_from_args(args)
    assert config.controller.actuator_mode == "torque"
    assert config.controller.control_hz == 200.0
    assert config.controller.max_joint_accel == 12.0
    assert config.controller.torque_kp == 90.0
    assert config.controller.torque_kd == 9.0
    assert config.controller.joint_limit_margin == 0.03
    assert config.controller.grasp_point == "top"
    assert config.controller.grasp_approach_m == 0.09
    assert config.controller.grasp_attach_distance_m == 0.07
    assert config.controller.grasp_lift_m == 0.15
    assert config.controller.grasp_stage_tolerance_m == 0.02
    assert config.controller.place_position == (0.5, -0.1, 0.25)
    assert config.controller.policy_max_attempts == 3
    assert config.controller.policy_close_timeout_s == 3.0
    assert config.controller.policy_motion_timeout_s == 9.0
    assert config.controller.policy_max_normal_force_n == 60.0
    assert config.controller.grasp_min_normal_force_n == 0.4
    assert config.controller.grasp_max_relative_slip_m == 0.002
    assert config.controller.grasp_confirmation_frames == 12
    assert config.controller.grasp_lost_frames == 18
    assert config.controller.policy_place_tolerance_m == 0.025
    assert not config.stop_on_terminal
    assert config.reacquire_confirm_frames == 4
    assert config.perception_latency_s == 0.08
    assert config.perception_jitter_s == 0.01
    assert config.perception_drop_probability == 0.2
    assert config.settling_threshold_m == 0.005
    assert config.environment == EnvironmentSpec(add_table=False)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            DemoConfig(controller=ControllerConfig(actuator_mode="invalid")),
            "actuator_mode",
        ),
        (
            DemoConfig(controller=ControllerConfig(max_joint_accel=0.0)),
            "max_joint_accel",
        ),
        (DemoConfig(controller=ControllerConfig(torque_kp=0.0)), "torque_kp"),
        (DemoConfig(controller=ControllerConfig(torque_kd=-1.0)), "torque_kd"),
        (
            DemoConfig(controller=ControllerConfig(joint_limit_margin=-0.1)),
            "joint_limit_margin",
        ),
        (
            DemoConfig(controller=ControllerConfig(grasp_approach_m=0.0)),
            "grasp_approach_m",
        ),
        (
            DemoConfig(controller=ControllerConfig(grasp_attach_distance_m=0.0)),
            "grasp_attach_distance_m",
        ),
        (DemoConfig(controller=ControllerConfig(grasp_lift_m=0.0)), "grasp_lift_m"),
        (
            DemoConfig(controller=ControllerConfig(grasp_stage_tolerance_m=0.0)),
            "grasp_stage_tolerance_m",
        ),
        (DemoConfig(reacquire_confirm_frames=0), "reacquire_confirm_frames"),
        (DemoConfig(perception_latency_s=-0.1), "perception_latency_s"),
        (DemoConfig(perception_jitter_s=-0.1), "perception_jitter_s"),
        (DemoConfig(perception_drop_probability=1.1), "perception_drop_probability"),
        (DemoConfig(settling_threshold_m=0.0), "settling_threshold_m"),
        (DemoConfig(camera=CameraConfig(rgb_noise_std=-1.0)), "noise"),
        (
            DemoConfig(camera=CameraConfig(dropout_probability=1.1)),
            "dropout_probability",
        ),
    ],
)
def test_new_runtime_config_fields_are_strictly_validated(
    config: DemoConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_validate_config_preflights_target_and_resolve_config_returns_assets(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-targets.json"
    with pytest.raises(FileNotFoundError, match="missing-targets"):
        validate_config(DemoConfig(target_file=str(missing)))
    with pytest.raises(ValueError, match="unknown target"):
        validate_config(DemoConfig(target="not-a-target"))
    resolved = resolve_config(DemoConfig(target="cup", robot="panda"))
    assert resolved.robot.name == "panda"
    assert resolved.target.name == "cup"


def test_robot_descriptor_schema_version_and_grasp_metadata(tmp_path: Path) -> None:
    _, descriptor = _write_robot_descriptor(tmp_path)
    descriptor.update(
        {
            "schema_version": 1,
            "grasp_attachment_body": "tool_body",
            "gripper_actuator_names": ["gripper"],
            "gripper_open_ctrl": [0.5],
            "gripper_closed_ctrl": [0.0],
            "torque_gain_scale": [0.5, 0.75],
            "impedance_gain_scale": [0.4, 0.6],
        }
    )
    path, _ = _write_robot_descriptor(
        tmp_path, {"schema_version": 1, "robots": [descriptor]}
    )
    spec = load_robot_specs(path)["testbot"]
    assert spec.schema_version == 1
    assert spec.grasp_attachment_body == "tool_body"
    assert spec.gripper_actuator_names == ("gripper",)
    assert spec.torque_gain_scale == (0.5, 0.75)
    assert spec.impedance_gain_scale == (0.4, 0.6)
    descriptor["schema_version"] = 2
    path, _ = _write_robot_descriptor(tmp_path, descriptor)
    with pytest.raises(ValueError, match="unsupported"):
        load_robot_specs(path)
