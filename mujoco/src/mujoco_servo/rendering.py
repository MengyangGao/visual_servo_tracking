from __future__ import annotations

from dataclasses import dataclass

import cv2
import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np


@dataclass(slots=True)
class ViewLayout:
    robot_title: str = "MuJoCo follow"
    camera_title: str = "Real camera"
    world_title: str = "MuJoCo world"
    label_scale: float = 0.62
    label_thickness: int = 2
    label_color: tuple[int, int, int] = (255, 255, 255)
    robot_label_color: tuple[int, int, int] = (0, 176, 255)
    camera_label_color: tuple[int, int, int] = (120, 220, 120)
    world_label_color: tuple[int, int, int] = (70, 150, 255)
    panel_height: int = 320
    gap_width: int = 12
    footer_height: int = 92


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def _label_panel(image: np.ndarray, label: str, color: tuple[int, int, int], layout: ViewLayout) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 270), 36), color, thickness=-1)
    cv2.putText(
        canvas,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        layout.label_scale,
        layout.label_color,
        layout.label_thickness,
        cv2.LINE_AA,
    )
    return canvas


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image.copy()
    scale = target_height / max(image.shape[0], 1)
    target_width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _panel_placeholder(label: str, size: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), color, thickness=2)
    cv2.putText(
        canvas,
        label,
        (20, max(42, height // 2 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "renderer unavailable",
        (20, max(74, height // 2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (215, 215, 215),
        1,
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


def three_panel_view(
    world_bgr: np.ndarray | None,
    follow_bgr: np.ndarray | None,
    camera_bgr: np.ndarray | None,
    layout: ViewLayout | None = None,
    footer_lines: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    layout = layout or ViewLayout()
    target_h = layout.panel_height

    world = _panel_placeholder(layout.world_title, (640, target_h), layout.world_label_color) if world_bgr is None else _label_panel(_resize_to_height(world_bgr, target_h), layout.world_title, layout.world_label_color, layout)
    follow = _panel_placeholder(layout.robot_title, (640, target_h), layout.robot_label_color) if follow_bgr is None else _label_panel(_resize_to_height(follow_bgr, target_h), layout.robot_title, layout.robot_label_color, layout)
    camera = _panel_placeholder(layout.camera_title, (640, target_h), layout.camera_label_color) if camera_bgr is None else _label_panel(_resize_to_height(camera_bgr, target_h), layout.camera_title, layout.camera_label_color, layout)

    target_h = max(world.shape[0], follow.shape[0], camera.shape[0])
    if world.shape[0] != target_h:
        world = _resize_to_height(world, target_h)
    if follow.shape[0] != target_h:
        follow = _resize_to_height(follow, target_h)
    if camera.shape[0] != target_h:
        camera = _resize_to_height(camera, target_h)

    gap = np.full((target_h, layout.gap_width, 3), 232, dtype=np.uint8)
    row = np.hstack([world, gap, follow, gap, camera])
    if not footer_lines:
        return row

    footer = np.full((layout.footer_height, row.shape[1], 3), 22, dtype=np.uint8)
    y = 24
    for idx, line in enumerate(footer_lines):
        if y >= footer.shape[0] - 8:
            break
        cv2.putText(
            footer,
            str(line),
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return np.vstack([row, footer])


class MujocoSceneRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        lookat: tuple[float, float, float] = (0.55, 0.0, 0.25),
        distance: float = 2.4,
        azimuth: float = 135.0,
        elevation: float = -20.0,
        follow_body_name: str | None = None,
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
        self._follow_body_id = -1
        if follow_body_name is not None:
            self._follow_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, follow_body_name)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        return self.render_with_lookat(data, None)

    def render_with_lookat(self, data: mujoco.MjData, lookat: tuple[float, float, float] | None) -> np.ndarray:
        if lookat is not None:
            self._camera.lookat[:] = np.array(lookat, dtype=float)
        elif self._follow_body_id >= 0:
            self._camera.lookat[:] = np.array(data.xpos[self._follow_body_id], dtype=float)
        self._renderer.update_scene(data, camera=self._camera)
        frame = self._renderer.render()
        return _ensure_bgr(frame)

    def set_lookat(self, lookat: tuple[float, float, float]) -> None:
        self._camera.lookat[:] = np.array(lookat, dtype=float)

    def set_distance(self, distance: float) -> None:
        self._camera.distance = float(distance)

    def close(self) -> None:
        self._renderer.close()


class MujocoViewerSession:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        lookat: tuple[float, float, float] = (0.55, 0.0, 0.25),
        distance: float = 2.4,
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
