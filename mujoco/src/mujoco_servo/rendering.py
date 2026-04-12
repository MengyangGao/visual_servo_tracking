from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np


@dataclass(slots=True)
class ViewLayout:
    robot_title: str = "MuJoCo view"
    camera_title: str = "Camera view"
    label_scale: float = 0.62
    label_thickness: int = 2
    label_color: tuple[int, int, int] = (255, 255, 255)
    robot_label_color: tuple[int, int, int] = (0, 176, 255)
    camera_label_color: tuple[int, int, int] = (120, 220, 120)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def _label_panel(image: np.ndarray, label: str, color: tuple[int, int, int], layout: ViewLayout) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 230), 34), color, thickness=-1)
    cv2.putText(
        canvas,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        layout.label_scale,
        layout.label_color,
        layout.label_thickness,
        cv2.LINE_AA,
    )
    return canvas


def side_by_side_view(robot_bgr: np.ndarray, camera_bgr: np.ndarray | None, layout: ViewLayout | None = None) -> np.ndarray:
    layout = layout or ViewLayout()
    robot = _label_panel(robot_bgr, layout.robot_title, layout.robot_label_color, layout)
    if camera_bgr is None:
        return robot
    camera = _label_panel(camera_bgr, layout.camera_title, layout.camera_label_color, layout)
    target_h = max(robot.shape[0], camera.shape[0])
    if robot.shape[0] != target_h:
        new_w = int(round(robot.shape[1] * target_h / robot.shape[0]))
        robot = cv2.resize(robot, (new_w, target_h), interpolation=cv2.INTER_AREA)
    if camera.shape[0] != target_h:
        new_w = int(round(camera.shape[1] * target_h / camera.shape[0]))
        camera = cv2.resize(camera, (new_w, target_h), interpolation=cv2.INTER_AREA)
    gap = np.full((target_h, 12, 3), 235, dtype=np.uint8)
    return np.hstack([robot, gap, camera])


class MujocoSceneRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        lookat: tuple[float, float, float] = (0.55, 0.0, 0.25),
        distance: float = 1.6,
        azimuth: float = 135.0,
        elevation: float = -20.0,
    ) -> None:
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = np.array(lookat, dtype=float)
        self._camera.distance = float(distance)
        self._camera.azimuth = float(azimuth)
        self._camera.elevation = float(elevation)
        self._camera.fixedcamid = -1
        self._camera.trackbodyid = -1

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.update_scene(data, camera=self._camera)
        frame = self._renderer.render()
        return _ensure_bgr(frame)

    def set_lookat(self, lookat: tuple[float, float, float]) -> None:
        self._camera.lookat[:] = np.array(lookat, dtype=float)

    def close(self) -> None:
        self._renderer.close()


class MujocoViewerSession:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        lookat: tuple[float, float, float] = (0.55, 0.0, 0.25),
        distance: float = 1.6,
        azimuth: float = 135.0,
        elevation: float = -20.0,
    ) -> None:
        self._handle = mujoco_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        self.set_lookat(lookat)
        self._handle.cam.distance = float(distance)
        self._handle.cam.azimuth = float(azimuth)
        self._handle.cam.elevation = float(elevation)

    def set_lookat(self, lookat: tuple[float, float, float]) -> None:
        self._handle.cam.lookat[:] = np.array(lookat, dtype=float)

    def sync(self) -> None:
        self._handle.sync()

    def is_running(self) -> bool:
        return self._handle.is_running()

    def close(self) -> None:
        self._handle.close()
