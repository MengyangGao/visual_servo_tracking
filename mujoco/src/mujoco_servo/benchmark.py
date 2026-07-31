from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .app import VisualServoSimulation
from .config import (
    ControllerConfig,
    DemoConfig,
    available_actuator_modes,
    available_detectors,
    available_robots,
    available_tasks,
    available_trajectories,
)

DEFAULT_TARGETS = {"panda": "cup", "ur5e": "box", "lite6": "box"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible MuJoCo visual-servo benchmark matrix"
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=list(available_robots()),
        choices=available_robots(),
    )
    parser.add_argument(
        "--actuator-modes",
        nargs="+",
        default=list(available_actuator_modes()),
        choices=available_actuator_modes(),
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        default=["static", "circle"],
        choices=available_trajectories(),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--detector", choices=available_detectors(), default="oracle")
    parser.add_argument("--task", default="standoff", choices=available_tasks())
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--static-position-mm", type=float, default=10.0)
    parser.add_argument("--moving-rms-mm", type=float, default=20.0)
    parser.add_argument("--orientation-deg", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero if any scenario misses a threshold",
    )
    return parser


def run_matrix(args: argparse.Namespace) -> dict:
    scenarios: list[dict] = []
    started = time.perf_counter()
    for robot in args.robots:
        for actuator_mode in args.actuator_modes:
            for trajectory in args.trajectories:
                for seed in args.seeds:
                    config = DemoConfig(
                        robot=robot,
                        target=DEFAULT_TARGETS.get(robot, "cup"),
                        trajectory=trajectory,
                        detector=args.detector,
                        steps=args.steps,
                        headless=True,
                        viewer=False,
                        realtime=False,
                        manual_control=False,
                        seed=seed,
                        controller=ControllerConfig(
                            task=args.task, actuator_mode=actuator_mode
                        ),
                    )
                    run_started = time.perf_counter()
                    with VisualServoSimulation(config) as simulation:
                        summary = simulation.run()
                    result = summary.as_dict()
                    result["seed"] = seed
                    result["wall_duration_s"] = time.perf_counter() - run_started
                    position_metric = (
                        result.get("steady_state_rms_error_m", result["rms_error_m"])
                        if trajectory != "static"
                        else result["final_error_m"]
                    )
                    position_limit = (
                        args.moving_rms_mm / 1000.0
                        if trajectory != "static"
                        else args.static_position_mm / 1000.0
                    )
                    orientation_limit = args.orientation_deg * 3.141592653589793 / 180.0
                    orientation_required = args.task == "front-standoff"
                    orientation_passed = (
                        result.get("final_orientation_error_rad", 0.0)
                        <= orientation_limit
                        if orientation_required
                        else True
                    )
                    result["acceptance"] = {
                        "position_metric_m": position_metric,
                        "position_limit_m": position_limit,
                        "orientation_limit_rad": orientation_limit,
                        "passed": bool(
                            position_metric <= position_limit and orientation_passed
                        ),
                        "orientation_checked": orientation_required,
                    }
                    scenarios.append(result)
    return {
        "schema_version": 1,
        "thresholds": {
            "static_position_mm": args.static_position_mm,
            "moving_rms_mm": args.moving_rms_mm,
            "orientation_deg": args.orientation_deg,
        },
        "wall_duration_s": time.perf_counter() - started,
        "passed": all(item["acceptance"]["passed"] for item in scenarios),
        "scenarios": scenarios,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    report = run_matrix(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if args.enforce and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
