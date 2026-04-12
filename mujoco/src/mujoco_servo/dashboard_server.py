from __future__ import annotations

import argparse
from multiprocessing.connection import Client

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mujoco-servo-dashboard", description="Helper dashboard process")
    parser.add_argument("--title", default="MuJoCo Vision Servo Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey", required=True)
    return parser


def _decode_frame(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = Client((args.host, args.port), authkey=bytes.fromhex(args.authkey))

    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(args.title)
    root.geometry("1600x980")
    root.minsize(1200, 760)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    label = tk.Label(root, bd=0, highlightthickness=0)
    label.pack(fill=tk.BOTH, expand=True)
    photo = None

    try:
        while True:
            while conn.poll():
                kind, payload = conn.recv()
                if kind == "quit":
                    return 0
                if kind == "frame" and isinstance(payload, (bytes, bytearray)):
                    image = _decode_frame(bytes(payload))
                    if image is None:
                        continue
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                    label.configure(image=photo)
            root.update_idletasks()
            root.update()
    except tk.TclError:
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
