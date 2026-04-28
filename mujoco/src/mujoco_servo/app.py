from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .config import DemoConfig
from .control import ResolvedRateController, ServoState
from .perception import build_perception
from .scene import build_scene, set_target_position, site_position
from .targets import TargetMotion, resolve_target


@dataclass(slots=True)
class RunSummary:
    steps: int
    task: str
    target: str
    trajectory: str
    detector: str
    final_error_m: float
    final_target_distance_m: float
    mean_error_m: float
    min_error_m: float
    max_error_m: float

    def as_dict(self) -> dict:
        return asdict(self)


class VisualServoSimulation:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.target = resolve_target(config.target)
        self.scene = build_scene(self.target, config.camera)
        self.motion = TargetMotion(self.target, config.trajectory, config.seed)
        self.perception = build_perception(config.detector)
        self.controller = ResolvedRateController(self.scene.model, self.scene.ee_site_name, config.controller)
        self.controller.reset(self.scene.data)
        self._substeps = max(1, int(round((1.0 / config.controller.control_hz) / self.scene.model.opt.timestep)))
        self._renderer = None

    def run(self) -> RunSummary:
        viewer = None
        if self.config.viewer and not self.config.headless:
            viewer = self._try_open_viewer()
        try:
            summary = self._run_loop(viewer)
        finally:
            if viewer is not None:
                viewer.close()
            if self._renderer is not None:
                self._renderer.close()
        return summary

    def _try_open_viewer(self):
        try:
            import mujoco.viewer

            return mujoco.viewer.launch_passive(self.scene.model, self.scene.data)
        except Exception as exc:
            if sys.platform == "darwin":
                print(f"viewer unavailable ({exc}); retry with `mjpython scripts/demo.py` for the native macOS viewer")
            else:
                print(f"viewer unavailable ({exc}); continuing headless")
            return None

    def _render_camera_frame(self) -> np.ndarray | None:
        if self.perception.name == "oracle":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.scene.model, width=self.config.camera.width, height=self.config.camera.height)
        self._renderer.update_scene(self.scene.data, camera=self.scene.camera_name)
        rgb = self._renderer.render()
        return rgb[:, :, ::-1].copy()

    def _run_loop(self, viewer) -> RunSummary:
        model = self.scene.model
        data = self.scene.data
        errors: list[float] = []
        last_state: ServoState | None = None
        wall_start = time.perf_counter()
        for step in range(self.config.steps):
            if viewer is not None and not viewer.is_running():
                break
            time_s = float(data.time)
            target_pos = self.motion.position(time_s)
            set_target_position(model, data, target_pos)
            mujoco.mj_forward(model, data)

            frame = self._render_camera_frame()
            detection = self.perception.detect(frame, target_pos, self.target)
            command_target = detection.target_position if detection.success and detection.target_position is not None else target_pos
            last_state = self.controller.step(data, command_target, time_s, step)
            errors.append(last_state.position_error_m)

            for _ in range(self._substeps):
                set_target_position(model, data, self.motion.position(float(data.time)))
                mujoco.mj_step(model, data)

            if viewer is not None:
                self._update_viewer_camera(viewer)
                viewer.sync()
            if self.config.realtime and viewer is not None:
                expected = (step + 1) / float(self.config.controller.control_hz)
                remaining = expected - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(min(remaining, 0.02))

        if last_state is None:
            ee = site_position(model, data, self.scene.ee_site_name)
            target = site_position(model, data, self.scene.target_site_name)
            final_error = float(np.linalg.norm(target - ee))
            errors = [final_error]
        else:
            final_error = last_state.position_error_m
        return RunSummary(
            steps=len(errors),
            task=self.config.controller.task,
            target=self.target.name,
            trajectory=self.config.trajectory,
            detector=self.perception.name,
            final_error_m=float(final_error),
            final_target_distance_m=float(last_state.target_distance_m if last_state is not None else errors[-1]),
            mean_error_m=float(np.mean(errors)),
            min_error_m=float(np.min(errors)),
            max_error_m=float(np.max(errors)),
        )

    def _update_viewer_camera(self, viewer) -> None:
        target = site_position(self.scene.model, self.scene.data, self.scene.target_site_name)
        ee = site_position(self.scene.model, self.scene.data, self.scene.ee_site_name)
        midpoint = 0.55 * target + 0.45 * ee
        viewer.cam.lookat[:] = midpoint
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 132.0
        viewer.cam.elevation = -24.0


def run_demo(config: DemoConfig) -> RunSummary:
    return VisualServoSimulation(config).run()
