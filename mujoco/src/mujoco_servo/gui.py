from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from .config import build_settings
from .camera import discover_cameras
from .runtime import run_camera, run_simulation


def launch_gui() -> None:
    root = tk.Tk()
    root.title("MuJoCo Vision Servo")
    root.geometry("520x320")

    prompt_var = tk.StringVar(value="cup")
    mode_var = tk.StringVar(value="camera")
    backend_var = tk.StringVar(value="auto")
    vision_preset_var = tk.StringVar(value="default")
    run_mode_var = tk.StringVar(value="manual")
    camera_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Idle")
    stop_event = threading.Event()
    worker = {"thread": None}

    def refresh_cameras() -> None:
        cams = discover_cameras()
        camera_var.set(", ".join(f"{c.index}" for c in cams) or "none")

    def start_task() -> None:
        if worker["thread"] is not None and worker["thread"].is_alive():
            return
        stop_event.clear()
        settings = build_settings(
            prompt=prompt_var.get(),
            backend=backend_var.get(),
            mode=mode_var.get(),
            run_mode=run_mode_var.get(),
            vision_preset=vision_preset_var.get(),
            max_steps=300,
            camera_index=None,
            show_view=True,
            record=False,
        )

        def _runner() -> None:
            status_var.set("Running")
            try:
                if settings.mode == "sim":
                    summary = run_simulation(settings, stop_event=stop_event)
                else:
                    summary = run_camera(settings, stop_event=stop_event)
                status_var.set(str(summary))
            except Exception as exc:  # noqa: BLE001
                status_var.set(f"Error: {exc}")

        worker["thread"] = threading.Thread(target=_runner, daemon=True)
        worker["thread"].start()

    def stop_task() -> None:
        stop_event.set()
        status_var.set("Stopping")

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Prompt").grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=prompt_var, width=28).grid(row=0, column=1, sticky="ew")

    ttk.Label(frame, text="Mode").grid(row=1, column=0, sticky="w")
    ttk.Combobox(frame, textvariable=mode_var, values=["sim", "camera"], width=24, state="readonly").grid(row=1, column=1, sticky="ew")

    ttk.Label(frame, text="Backend").grid(row=2, column=0, sticky="w")
    ttk.Combobox(frame, textvariable=backend_var, values=["auto", "oracle", "heuristic", "grounded-sam2"], width=24, state="readonly").grid(row=2, column=1, sticky="ew")

    ttk.Label(frame, text="Vision preset").grid(row=3, column=0, sticky="w")
    ttk.Combobox(frame, textvariable=vision_preset_var, values=["default", "small", "lite"], width=24, state="readonly").grid(row=3, column=1, sticky="ew")

    ttk.Label(frame, text="Run mode").grid(row=4, column=0, sticky="w")
    ttk.Combobox(frame, textvariable=run_mode_var, values=["auto", "manual"], width=24, state="readonly").grid(row=4, column=1, sticky="ew")

    ttk.Label(frame, text="Camera").grid(row=5, column=0, sticky="w")
    ttk.Label(frame, textvariable=camera_var).grid(row=5, column=1, sticky="w")

    button_row = ttk.Frame(frame)
    button_row.grid(row=6, column=0, columnspan=2, pady=(10, 6), sticky="w")
    ttk.Button(button_row, text="Refresh cameras", command=refresh_cameras).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(button_row, text="Start", command=start_task).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(button_row, text="Stop", command=stop_task).pack(side=tk.LEFT)

    ttk.Label(frame, textvariable=status_var, wraplength=460).grid(row=7, column=0, columnspan=2, sticky="w")
    frame.columnconfigure(1, weight=1)
    refresh_cameras()
    root.mainloop()
