from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class ThreePanelDashboardWindow:
    def __init__(self, title: str = "MuJoCo Vision Servo Dashboard") -> None:
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry("1600x980")
        self._root.minsize(1200, 760)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label = tk.Label(self._root, bd=0, highlightthickness=0)
        self._label.pack(fill=tk.BOTH, expand=True)
        self._photo: ImageTk.PhotoImage | None = None
        self._closed = False

    def update(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(image=image)
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
