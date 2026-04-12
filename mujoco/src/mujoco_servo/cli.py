from __future__ import annotations

import argparse

from .camera import discover_cameras
from .config import build_settings
from .runtime import run_camera, run_gui, run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mujoco-servo", description="MuJoCo vision servo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("sim", help="run simulation")
    sim.add_argument("--prompt", default="cup")
    sim.add_argument("--backend", default="oracle")
    sim.add_argument("--steps", type=int, default=240)
    sim.add_argument("--no-view", action="store_true")
    sim.add_argument("--record", action="store_true")

    cam = subparsers.add_parser("camera", help="run camera loop")
    cam.add_argument("--prompt", default="cup")
    cam.add_argument("--backend", default="auto")
    cam.add_argument("--steps", type=int, default=240)
    cam.add_argument("--camera-index", type=int, default=None)
    cam.add_argument("--mode", default="camera")
    cam.add_argument("--run-mode", default="manual")
    cam.add_argument("--no-view", action="store_true")
    cam.add_argument("--record", action="store_true")

    subparsers.add_parser("gui", help="launch GUI")
    subparsers.add_parser("cameras", help="list cameras")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "cameras":
        cameras = discover_cameras()
        if not cameras:
            print("No camera devices found or the process does not have camera permission.")
        for info in cameras:
            print(f"{info.index}: backend={info.backend_name} size={info.frame_size}")
        return 0
    if args.command == "gui":
        run_gui()
        return 0
    if args.command == "sim":
        settings = build_settings(
            prompt=args.prompt,
            backend=args.backend,
            mode="sim",
            run_mode="auto",
            max_steps=args.steps,
            show_view=not args.no_view,
            record=args.record,
        )
        summary = run_simulation(settings)
        print(summary)
        return 0
    if args.command == "camera":
        settings = build_settings(
            prompt=args.prompt,
            backend=args.backend,
            mode=args.mode,
            run_mode=args.run_mode,
            max_steps=args.steps,
            camera_index=args.camera_index,
            show_view=not args.no_view,
            record=args.record,
        )
        summary = run_camera(settings)
        print(summary)
        return 0
    parser.error("unknown command")
    return 1
