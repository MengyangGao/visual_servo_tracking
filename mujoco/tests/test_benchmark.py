from __future__ import annotations

from mujoco_servo.benchmark import build_parser, run_matrix
from mujoco_servo.config import available_actuator_modes, available_robots


def test_benchmark_defaults_cover_all_robots_modes_and_core_trajectories() -> None:
    args = build_parser().parse_args([])
    assert set(args.robots) == set(available_robots())
    assert set(args.actuator_modes) == set(available_actuator_modes())
    assert args.trajectories == ["static", "circle"]
    assert args.task == "standoff"


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
