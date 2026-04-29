from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .config import DemoConfig
from .control import ResolvedRateController, ServoState
from .perception import CameraIntrinsics, CameraObservation, build_perception
from .scene import build_scene, frame_position, set_target_position, site_position
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
        self.controller = ResolvedRateController(
            self.scene.model,
            self.scene.ee_frame_name,
            self.scene.ee_frame_type,
            self.scene.ee_frame_offset,
            config.controller,
        )
        self.controller.reset(self.scene.data)
        self._substeps = max(1, int(round((1.0 / config.controller.control_hz) / self.scene.model.opt.timestep)))
        self._renderer = None
        self._manual_target_offset = np.zeros(3, dtype=float)
        self._manual_target_velocity = np.zeros(3, dtype=float)
        self._manual_velocity_until = 0.0
        self._last_target_update_time: float | None = None
        self._mouse_drag_enabled = False
        self._viewer_camera_initialized = False

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

            viewer = mujoco.viewer.launch_passive(self.scene.model, self.scene.data, key_callback=self._handle_key)
            self._initialize_viewer(viewer)
            return viewer
        except Exception as exc:
            if sys.platform == "darwin":
                print(f"viewer unavailable ({exc}); retry with `mjpython scripts/demo.py` for the native macOS viewer")
            else:
                print(f"viewer unavailable ({exc}); continuing headless")
            return None

    def _render_camera_observation(self) -> CameraObservation | None:
        if self.perception.name == "oracle":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.scene.model, width=self.config.camera.width, height=self.config.camera.height)
        self._renderer.update_scene(self.scene.data, camera=self.scene.camera_name)
        rgb = self._renderer.render()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.scene.data, camera=self.scene.camera_name)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        cam_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_CAMERA, self.scene.camera_name)
        fovy = float(self.scene.model.cam_fovy[cam_id])
        fy = 0.5 * self.config.camera.height / np.tan(np.deg2rad(fovy) * 0.5)
        intrinsics = CameraIntrinsics(
            fx=fy,
            fy=fy,
            cx=0.5 * self.config.camera.width,
            cy=0.5 * self.config.camera.height,
            width=self.config.camera.width,
            height=self.config.camera.height,
        )
        return CameraObservation(
            frame_bgr=rgb[:, :, ::-1].copy(),
            depth_m=depth,
            intrinsics=intrinsics,
            camera_position=np.array(self.scene.data.cam_xpos[cam_id], dtype=float),
            camera_xmat=np.array(self.scene.data.cam_xmat[cam_id], dtype=float).reshape(3, 3),
        )

    def _run_loop(self, viewer) -> RunSummary:
        model = self.scene.model
        data = self.scene.data
        errors: list[float] = []
        last_state: ServoState | None = None
        last_observed_target: np.ndarray | None = None
        wall_start = time.perf_counter()
        for step in range(self.config.steps):
            if viewer is not None and not viewer.is_running():
                break
            time_s = float(data.time)
            target_pos = self._target_position(time_s)
            set_target_position(model, data, target_pos)
            mujoco.mj_forward(model, data)
            if viewer is not None:
                self._apply_viewer_target_drag(viewer, target_pos)
                target_pos = site_position(model, data, self.scene.target_site_name)

            should_detect = self._should_run_detector(step, last_observed_target)
            observation = self._render_camera_observation() if should_detect else None
            detection = self.perception.detect(observation, target_pos, self.target, self.config.target) if should_detect else None
            if detection is not None and detection.success and detection.target_position is not None:
                last_observed_target = detection.target_position.copy()
            if last_observed_target is not None:
                command_target = last_observed_target
            elif self.perception.name == "oracle":
                command_target = target_pos
            else:
                command_target = frame_position(model, data, self.scene.ee_frame_type, self.scene.ee_frame_name, self.scene.ee_frame_offset)
            last_state = self.controller.step(data, command_target, time_s, step)
            errors.append(last_state.position_error_m)

            for _ in range(self._substeps):
                if self.config.interactive_target and viewer is not None:
                    set_target_position(model, data, target_pos)
                else:
                    set_target_position(model, data, self._target_position(float(data.time)))
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
            if self.config.realtime and viewer is not None:
                expected = (step + 1) / float(self.config.controller.control_hz)
                remaining = expected - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(min(remaining, 0.02))

        if last_state is None:
            ee = frame_position(model, data, self.scene.ee_frame_type, self.scene.ee_frame_name, self.scene.ee_frame_offset)
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

    def _target_position(self, time_s: float) -> np.ndarray:
        self._integrate_manual_target_velocity(time_s)
        return self.motion.position(time_s) + self._manual_target_offset

    def _integrate_manual_target_velocity(self, time_s: float) -> None:
        if self._last_target_update_time is None:
            self._last_target_update_time = time_s
            return
        dt = max(0.0, min(0.05, time_s - self._last_target_update_time))
        self._last_target_update_time = time_s
        if time_s > self._manual_velocity_until:
            self._manual_target_velocity[:] = 0.0
            return
        self._manual_target_offset += self._manual_target_velocity * dt

    def _should_run_detector(self, step: int, last_observed_target: np.ndarray | None) -> bool:
        if self.perception.name != "semantic":
            return True
        if last_observed_target is None:
            return True
        return step % max(1, int(self.config.semantic_interval)) == 0

    def _initialize_viewer(self, viewer) -> None:
        if self._viewer_camera_initialized:
            return
        target = site_position(self.scene.model, self.scene.data, self.scene.target_site_name)
        ee = frame_position(self.scene.model, self.scene.data, self.scene.ee_frame_type, self.scene.ee_frame_name, self.scene.ee_frame_offset)
        midpoint = 0.55 * target + 0.45 * ee
        viewer.cam.lookat[:] = midpoint
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 132.0
        viewer.cam.elevation = -24.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
        self._clear_target_perturb(viewer)
        self._viewer_camera_initialized = True

    def _select_target_for_perturb(self, viewer) -> None:
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, self.scene.target_body_name)
        if body_id < 0:
            return
        viewer.perturb.select = body_id
        viewer.perturb.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
        viewer.perturb.active2 = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
        viewer.perturb.refpos[:] = site_position(self.scene.model, self.scene.data, self.scene.target_site_name)
        viewer.perturb.localpos[:] = 0.0
        viewer.perturb.scale = 0.25

    def _clear_target_perturb(self, viewer) -> None:
        viewer.perturb.select = 0
        viewer.perturb.active = 0
        viewer.perturb.active2 = 0

    def _apply_viewer_target_drag(self, viewer, scripted_target: np.ndarray) -> None:
        if not self.config.interactive_target or not self._mouse_drag_enabled:
            self._clear_target_perturb(viewer)
            return
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, self.scene.target_body_name)
        if body_id < 0:
            return
        mocap_id = int(self.scene.model.body_mocapid[body_id])
        if mocap_id < 0:
            return
        dragged = np.array(self.scene.data.mocap_pos[mocap_id], dtype=float)
        if np.linalg.norm(dragged - scripted_target) > 1e-6:
            self._manual_target_offset = dragged - self.motion.position(float(self.scene.data.time))
        self._select_target_for_perturb(viewer)

    def _handle_key(self, keycode: int) -> None:
        try:
            import glfw
        except Exception:
            return
        if keycode == glfw.KEY_F2:
            self._mouse_drag_enabled = not self._mouse_drag_enabled
            return
        speed = float(self.config.key_speed_mps)
        mapping = {
            glfw.KEY_LEFT: np.array([0.0, speed, 0.0], dtype=float),
            glfw.KEY_RIGHT: np.array([0.0, -speed, 0.0], dtype=float),
            glfw.KEY_UP: np.array([speed, 0.0, 0.0], dtype=float),
            glfw.KEY_DOWN: np.array([-speed, 0.0, 0.0], dtype=float),
            glfw.KEY_PAGE_UP: np.array([0.0, 0.0, speed], dtype=float),
            glfw.KEY_PAGE_DOWN: np.array([0.0, 0.0, -speed], dtype=float),
        }
        if keycode in mapping:
            self._manual_target_velocity = mapping[keycode]
            self._manual_velocity_until = float(self.scene.data.time) + 0.22
        elif keycode in {glfw.KEY_SPACE, getattr(glfw, "KEY_BACKSPACE", -1)}:
            self._manual_target_offset[:] = 0.0
            self._manual_target_velocity[:] = 0.0
            self._manual_velocity_until = 0.0


def run_demo(config: DemoConfig) -> RunSummary:
    return VisualServoSimulation(config).run()
