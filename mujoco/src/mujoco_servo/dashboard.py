from __future__ import annotations

import os
import secrets
import subprocess
import sys
import socket
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from pathlib import Path

import cv2
import numpy as np


def _default_python_executable() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _needs_helper_process() -> bool:
    return sys.platform == "darwin" and Path(sys.executable).name == "mjpython"


class _LocalTkDashboardWindow:
    def __init__(self, title: str = "MuJoCo Vision Servo Dashboard") -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry("1600x980")
        self._root.minsize(1200, 760)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label = tk.Label(self._root, bd=0, highlightthickness=0)
        self._label.pack(fill=tk.BOTH, expand=True)
        self._photo = None
        self._closed = False

    def update(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self._photo = self._ImageTk.PhotoImage(image=image)
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()

    def is_open(self) -> bool:
        return (not self._closed) and bool(self._root.winfo_exists())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except Exception:
            pass


@dataclass(slots=True)
class _RemoteDashboardBridge:
    listener: Listener
    connection: object
    process: subprocess.Popen


class ThreePanelDashboardWindow:
    def __init__(self, title: str = "MuJoCo Vision Servo Dashboard") -> None:
        self._title = title
        self._closed = False
        self._local: _LocalTkDashboardWindow | None = None
        self._remote: _RemoteDashboardBridge | None = None
        if _needs_helper_process():
            self._start_remote()
        else:
            try:
                self._local = _LocalTkDashboardWindow(title=title)
            except Exception:
                self._start_remote()

    def _start_remote(self) -> None:
        authkey = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        try:
            try:
                listener._listener._socket.settimeout(0.25)  # type: ignore[attr-defined]
            except Exception:
                pass
            host, port = listener.address
            helper = subprocess.Popen(
                [
                    _default_python_executable(),
                    "-m",
                    "mujoco_servo.dashboard_server",
                    "--title",
                    self._title,
                    "--host",
                    str(host),
                    "--port",
                    str(port),
                    "--authkey",
                    authkey.hex(),
                ],
                env=os.environ.copy(),
            )
            connection = None
            for _ in range(80):
                if helper.poll() is not None:
                    break
                try:
                    connection = listener.accept()
                    break
                except socket.timeout:
                    continue
            if connection is None:
                try:
                    helper.terminate()
                except Exception:
                    pass
                try:
                    helper.wait(timeout=0.5)
                except Exception:
                    pass
                raise RuntimeError("dashboard helper failed to connect")
            self._remote = _RemoteDashboardBridge(listener=listener, connection=connection, process=helper)
        except Exception:
            try:
                listener.close()
            except Exception:
                pass
            self._remote = None

    def update(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        if self._local is not None:
            self._local.update(frame_bgr)
            return
        if self._remote is None:
            return
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        try:
            self._remote.connection.send(("frame", encoded.tobytes()))
        except Exception:
            self.close()

    def is_open(self) -> bool:
        if self._closed:
            return False
        if self._local is not None:
            return self._local.is_open()
        if self._remote is None:
            return False
        return self._remote.process.poll() is None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._local is not None:
            self._local.close()
            self._local = None
        if self._remote is not None:
            try:
                self._remote.connection.send(("quit", None))
            except Exception:
                pass
            try:
                self._remote.connection.close()
            except Exception:
                pass
            try:
                self._remote.listener.close()
            except Exception:
                pass
            try:
                self._remote.process.terminate()
            except Exception:
                pass
            self._remote = None
