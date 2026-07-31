from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..perception import CameraIntrinsics, CameraObservation


@dataclass(slots=True)
class CameraRig:
    """Render synchronized RGB-D observations from any named MuJoCo cameras."""

    model: mujoco.MjModel
    width: int = 424
    height: int = 320
    _renderer: mujoco.Renderer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._renderer = mujoco.Renderer(
            self.model, width=self.width, height=self.height
        )

    def close(self) -> None:
        self._renderer.close()

    def render(
        self, data: mujoco.MjData, camera_names: tuple[str, ...]
    ) -> dict[str, CameraObservation]:
        observations: dict[str, CameraObservation] = {}
        for name in camera_names:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                raise KeyError(f"camera '{name}' not found")
            self._renderer.update_scene(data, camera=name)
            rgb = self._renderer.render().copy()
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(data, camera=name)
            depth = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()
            fovy = float(self.model.cam_fovy[camera_id])
            fy = 0.5 * self.height / np.tan(np.deg2rad(fovy) * 0.5)
            observations[name] = CameraObservation(
                frame_bgr=rgb[:, :, ::-1].copy(),
                depth_m=depth,
                intrinsics=CameraIntrinsics(
                    fx=fy,
                    fy=fy,
                    cx=0.5 * (self.width - 1),
                    cy=0.5 * (self.height - 1),
                    width=self.width,
                    height=self.height,
                ),
                camera_position=np.asarray(
                    data.cam_xpos[camera_id], dtype=float
                ).copy(),
                camera_xmat=np.asarray(data.cam_xmat[camera_id], dtype=float)
                .reshape(3, 3)
                .copy(),
                sim_time_s=float(data.time),
                depth_backend="mujoco",
                depth_metric=True,
            )
        return observations

    def __enter__(self) -> CameraRig:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
