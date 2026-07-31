from __future__ import annotations

from mujoco_servo.benchmark import build_parser, run_matrix
from mujoco_servo.config import available_actuator_modes, available_robots


def test_benchmark_defaults_cover_all_robots_modes_and_core_trajectories() -> None:
    args = build_parser().parse_args([])
    assert set(args.robots) == set(available_robots())
    assert set(args.actuator_modes) == set(available_actuator_modes())
    assert args.trajectories == ["static", "circle"]
    assert args.task == "standoff"
    assert args.moving_path_ratio_min == 0.8
    assert args.moving_path_ratio_max == 1.2


def test_benchmark_single_scenario_reports_acceptance() -> None:
    args = build_parser().parse_args(
        [
            "--robots",
            "panda",
            "--actuator-modes",
            "position",
            "--trajectories",
            "static",
            "--seeds",
            "11",
            "--steps",
            "2",
        ]
    )
    report = run_matrix(args)
    assert report["schema_version"] == 1
    assert len(report["scenarios"]) == 1
    scenario = report["scenarios"][0]
    assert scenario["robot"] == "panda"
    assert scenario["actuator_mode"] == "position"
    assert scenario["seed"] == 11
    assert scenario["acceptance"]["orientation_checked"] is False
    assert scenario["acceptance"]["path_ratio_checked"] is False


def test_moving_benchmark_requires_end_effector_to_follow_target_path() -> None:
    args = build_parser().parse_args(
        [
            "--robots",
            "panda",
            "--actuator-modes",
            "position",
            "--trajectories",
            "circle",
            "--steps",
            "1200",
        ]
    )

    scenario = run_matrix(args)["scenarios"][0]

    assert scenario["acceptance"]["path_ratio_checked"] is True
    assert 0.9 <= scenario["tracking_path_ratio"] <= 1.1
    assert scenario["acceptance"]["passed"] is True
