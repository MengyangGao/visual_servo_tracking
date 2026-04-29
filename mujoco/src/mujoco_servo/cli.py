from __future__ import annotations

import argparse
import json

from .app import run_demo
from .config import CameraConfig, ControllerConfig, DemoConfig, available_tasks, available_trajectories
from .targets import TARGETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo visual-servo tracking demo")
    parser.add_argument("--target", default="cup", help="target object word or phrase, e.g. cup, capsule, hammer, red apple")
    parser.add_argument("--trajectory", default="circle", choices=available_trajectories(), help="target motion")
    parser.add_argument("--task", default="contact", choices=available_tasks(), help="servo objective")
    parser.add_argument("--detector", default="oracle", choices=("oracle", "color", "semantic"), help="perception backend")
    parser.add_argument("--steps", type=int, default=1200, help="control steps to run")
    parser.add_argument("--headless", action="store_true", help="run without the MuJoCo viewer")
    parser.add_argument("--no-realtime", action="store_true", help="do not sleep to match wall-clock time")
    parser.add_argument("--scripted-target", action="store_true", help="ignore interactive viewer target edits and use only the scripted trajectory")
    parser.add_argument("--key-speed-cm-s", type=float, default=18.0, help="continuous keyboard target speed in centimeters per second")
    parser.add_argument("--semantic-interval", type=int, default=10, help="run semantic Grounding DINO + SAM every N control frames")
    parser.add_argument("--standoff", type=float, default=None, help="standoff distance in meters")
    parser.add_argument("--standoff-cm", type=float, default=16.0, help="standoff distance in centimeters for standoff/front-standoff")
    parser.add_argument("--list-targets", action="store_true", help="print built-in target words and exit")
    parser.add_argument("--seed", type=int, default=7, help="random seed for random-walk trajectory")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    return parser


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    camera = CameraConfig(width=args.camera_width, height=args.camera_height)
    standoff_m = float(args.standoff) if args.standoff is not None else float(args.standoff_cm) / 100.0
    controller = ControllerConfig(task=args.task, standoff_m=standoff_m)
    return DemoConfig(
        target=args.target,
        trajectory=args.trajectory,
        detector=args.detector,
        steps=args.steps,
        headless=args.headless,
        viewer=not args.headless,
        realtime=not args.no_realtime,
        interactive_target=not args.scripted_target,
        key_speed_mps=float(args.key_speed_cm_s) / 100.0,
        semantic_interval=max(1, int(args.semantic_interval)),
        seed=args.seed,
        camera=camera,
        controller=controller,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_targets:
        print(json.dumps(sorted(TARGETS.keys()), indent=2))
        return 0
    summary = run_demo(config_from_args(args))
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
