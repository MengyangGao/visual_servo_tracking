from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .types import CameraFrame


def _backend_candidates() -> list[int]:
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


@dataclass(slots=True)
class CameraInfo:
    index: int
    backend_name: str
    frame_size: tuple[int, int] | None = None


def discover_cameras(max_devices: int | None = None) -> list[CameraInfo]:
    if max_devices is None:
        max_devices = 1 if sys.platform == "darwin" else 4
    found: list[CameraInfo] = []
    for index in range(max_devices):
        for backend in _backend_candidates():
            cap = cv2.VideoCapture(index, backend)
            try:
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append(CameraInfo(index=index, backend_name=str(backend), frame_size=(frame.shape[1], frame.shape[0])))
                    break
            finally:
                cap.release()
    return found


class CameraStream:
    def __init__(self, capture: cv2.VideoCapture, index: int, backend_name: str):
        self._capture = capture
        self.index = index
        self.backend_name = backend_name

    def read(self) -> CameraFrame:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("camera frame read failed")
        return CameraFrame(
            image_bgr=frame,
            timestamp_s=time.time(),
            device_index=self.index,
            backend_name=self.backend_name,
        )

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def open_camera(index: int | None = None, width: int = 1280, height: int = 720) -> CameraStream:
    candidates = [index] if index is not None else [info.index for info in discover_cameras()]
    if not candidates:
        candidates = list(range(1 if sys.platform == "darwin" else 4))
    for camera_index in candidates:
        for backend in _backend_candidates():
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            ok, frame = capture.read()
            if ok and frame is not None:
                return CameraStream(capture, camera_index, str(backend))
            capture.release()
    raise RuntimeError("no usable camera found")
