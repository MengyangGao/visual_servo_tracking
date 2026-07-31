from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(
        self, path: str | Path, *, fps: float, frame_size: tuple[int, int]
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = tuple(int(value) for value in frame_size)
        suffix = self.path.suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi"}:
            raise ValueError("record path must end in .mp4, .mov, or .avi")
        codec = "mp4v" if suffix in {".mp4", ".mov"} else "MJPG"
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*codec), float(fps), self.frame_size
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video recorder: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        frame = np.asarray(frame_bgr, dtype=np.uint8)
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
