from __future__ import annotations

import cv2
import numpy as np


class CameraPreviewWindow:
    def __init__(self, title: str = "Camera view") -> None:
        import tkinter as tk
        from PIL import ImageTk

        self._tk = tk
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title(title)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label = tk.Label(self._root)
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
