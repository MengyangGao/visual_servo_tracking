from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass

import cv2
import mujoco
import numpy as np

from .config import DemoConfig
from .control import ResolvedRateController, ServoState
from .perception import CameraIntrinsics, CameraObservation, Detection, PerceptionBackend, build_perception
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
        self.detector_name = config.detector.strip().lower()
        self.perception: PerceptionBackend | None = None
        if not self._should_lazy_load_perception():
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
        self._latest_overlay_bgr: np.ndarray | None = None
        self._latest_overlay_rgb: np.ndarray | None = None
        self._overlay_rect_key: tuple[int, int, int, int] | None = None
        self._last_camera_wall_time = -1.0e9
        self._perception_executor: ThreadPoolExecutor | None = None
        self._perception_future: Future[tuple[CameraObservation, Detection]] | None = None
        self._last_detection: Detection | None = None
        self._perception_disabled = False
        self._target_drag_perturb_initialized = False
        self._target_drag_clear_requested = False

    def run(self) -> RunSummary:
        viewer = None
        if self.config.viewer and not self.config.headless:
            viewer = self._try_open_viewer()
        try:
            if self._uses_async_perception(viewer):
                self._perception_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mujoco-servo-perception")
            summary = self._run_loop(viewer)
        finally:
            if self._perception_executor is not None:
                self._perception_executor.shutdown(wait=False, cancel_futures=True)
                self._perception_executor = None
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
        if self.detector_name == "oracle":
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
            if viewer is not None and self.config.interactive_target and self._mouse_drag_enabled:
                target_pos = self._apply_viewer_target_drag(viewer, target_pos)
            set_target_position(model, data, target_pos)
            mujoco.mj_forward(model, data)

            detection = self._update_perception(viewer, target_pos)
            if detection is not None and detection.success and detection.target_position is not None:
                last_observed_target = detection.target_position.copy()
            if last_observed_target is not None:
                command_target = last_observed_target
            elif self.detector_name == "oracle":
                command_target = target_pos
            elif self._uses_async_perception(viewer):
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
                self._keep_viewer_camera_free(viewer)
                self._update_viewer_overlay(viewer)
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
            detector=self.detector_name,
            final_error_m=float(final_error),
            final_target_distance_m=float(last_state.target_distance_m if last_state is not None else errors[-1]),
            mean_error_m=float(np.mean(errors)),
            min_error_m=float(np.min(errors)),
            max_error_m=float(np.max(errors)),
        )

    def _uses_async_perception(self, viewer) -> bool:
        return viewer is not None and self.detector_name != "oracle"

    def _should_lazy_load_perception(self) -> bool:
        return self.detector_name == "semantic" and self.config.viewer and not self.config.headless

    def _ensure_perception(self) -> PerceptionBackend:
        if self.perception is None:
            self.perception = build_perception(self.config.detector)
        return self.perception

    def _update_perception(self, viewer, truth_position: np.ndarray) -> Detection | None:
        perception = self._ensure_perception() if not self._uses_async_perception(viewer) else None
        if self.detector_name == "oracle":
            return perception.detect(None, truth_position, self.target, self.config.target)
        if self._uses_async_perception(viewer):
            return self._update_async_perception(truth_position)
        observation = self._render_camera_observation()
        detection = perception.detect(observation, truth_position, self.target, self.config.target)
        self._last_detection = detection
        self._latest_overlay_bgr = self._draw_camera_overlay(observation, detection, False)
        self._latest_overlay_rgb = None
        return detection

    def _update_async_perception(self, truth_position: np.ndarray) -> Detection | None:
        if self._perception_future is not None and self._perception_future.done():
            try:
                observation, detection = self._perception_future.result()
                self._last_detection = detection
                self._latest_overlay_bgr = self._draw_camera_overlay(observation, detection, False)
                self._latest_overlay_rgb = None
            except Exception as exc:
                self._perception_disabled = True
                self._last_detection = Detection(False, self.detector_name, None)
                print(f"perception worker failed: {exc}", file=sys.stderr)
            finally:
                self._perception_future = None
        if self._perception_disabled:
            return self._last_detection
        now = time.perf_counter()
        camera_period = 1.0 / max(0.5, float(self.config.camera_fps))
        should_sample = self._perception_future is None and now - self._last_camera_wall_time >= camera_period
        if should_sample:
            observation = self._render_camera_observation()
            self._last_camera_wall_time = now
            self._latest_overlay_bgr = self._draw_camera_overlay(observation, None, True)
            self._latest_overlay_rgb = None
            if self._perception_executor is not None:
                prompt = self.config.target
                target = self.target
                truth = truth_position.copy()
                self._perception_future = self._perception_executor.submit(self._detect_in_worker, observation, truth, target, prompt)
        return self._last_detection

    def _detect_in_worker(self, observation: CameraObservation, truth_position: np.ndarray, target, prompt: str) -> tuple[CameraObservation, Detection]:
        perception = self._ensure_perception()
        return observation, perception.detect(observation, truth_position, target, prompt)

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

    def _initialize_viewer(self, viewer) -> None:
        if self._viewer_camera_initialized:
            return
        target = site_position(self.scene.model, self.scene.data, self.scene.target_site_name)
        ee = frame_position(self.scene.model, self.scene.data, self.scene.ee_frame_type, self.scene.ee_frame_name, self.scene.ee_frame_offset)
        midpoint = 0.55 * target + 0.45 * ee
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = midpoint
            viewer.cam.distance = 1.35
            viewer.cam.azimuth = 132.0
            viewer.cam.elevation = -24.0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
            self._clear_target_perturb(viewer)
        self._viewer_camera_initialized = True

    def _keep_viewer_camera_free(self, viewer) -> None:
        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            return
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.fixedcamid = -1

    def _initialize_target_perturb(self, viewer) -> None:
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, self.scene.target_body_name)
        if body_id < 0:
            return
        with viewer.lock():
            viewer.perturb.select = body_id
            viewer.perturb.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
            viewer.perturb.active2 = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
            if viewer.user_scn is not None:
                mujoco.mjv_initPerturb(self.scene.model, self.scene.data, viewer.user_scn, viewer.perturb)
            else:
                viewer.perturb.refpos[:] = site_position(self.scene.model, self.scene.data, self.scene.target_site_name)
                viewer.perturb.localpos[:] = 0.0
                viewer.perturb.scale = 0.25
        self._target_drag_perturb_initialized = True

    def _clear_target_perturb(self, viewer) -> None:
        viewer.perturb.select = 0
        viewer.perturb.active = 0
        viewer.perturb.active2 = 0
        self._target_drag_perturb_initialized = False

    def _apply_viewer_target_drag(self, viewer, scripted_target: np.ndarray) -> np.ndarray:
        if not self.config.interactive_target or not self._mouse_drag_enabled:
            if self._target_drag_perturb_initialized or self._target_drag_clear_requested:
                with viewer.lock():
                    self._clear_target_perturb(viewer)
                self._target_drag_clear_requested = False
            return scripted_target
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, self.scene.target_body_name)
        if body_id < 0:
            return scripted_target
        mocap_id = int(self.scene.model.body_mocapid[body_id])
        if mocap_id < 0:
            return scripted_target
        if not self._target_drag_perturb_initialized:
            self._initialize_target_perturb(viewer)
        with viewer.lock():
            mujoco.mjv_applyPerturbPose(self.scene.model, self.scene.data, viewer.perturb, 1)
        dragged = np.array(self.scene.data.mocap_pos[mocap_id], dtype=float)
        if np.linalg.norm(dragged - scripted_target) > 1e-6:
            self._manual_target_offset = dragged - self.motion.position(float(self.scene.data.time))
            scripted_target = dragged
        return scripted_target

    def _handle_key(self, keycode: int) -> None:
        try:
            import glfw
        except Exception:
            return
        if keycode in {glfw.KEY_L, ord("L"), ord("l")}:
            self._mouse_drag_enabled = not self._mouse_drag_enabled
            if self._mouse_drag_enabled:
                self._target_drag_perturb_initialized = False
            else:
                self._target_drag_clear_requested = True
            return
        speed = float(self.config.key_speed_mps)
        mapping = {
            glfw.KEY_LEFT: (0.0, speed, 0.0),
            glfw.KEY_RIGHT: (0.0, -speed, 0.0),
            glfw.KEY_UP: (speed, 0.0, 0.0),
            glfw.KEY_DOWN: (-speed, 0.0, 0.0),
            glfw.KEY_X: (0.0, 0.0, speed),
            glfw.KEY_Z: (0.0, 0.0, -speed),
            ord("X"): (0.0, 0.0, speed),
            ord("x"): (0.0, 0.0, speed),
            ord("Z"): (0.0, 0.0, -speed),
            ord("z"): (0.0, 0.0, -speed),
        }
        if keycode in mapping:
            self._manual_target_velocity[:] = mapping[keycode]
            self._manual_velocity_until = float(self.scene.data.time) + 0.35
        elif keycode in {glfw.KEY_SPACE, getattr(glfw, "KEY_BACKSPACE", -1)}:
            self._manual_target_offset[:] = 0.0
            self._manual_target_velocity[:] = 0.0
            self._manual_velocity_until = 0.0

    def _draw_camera_overlay(
        self,
        observation: CameraObservation | None,
        detection,
        pending: bool,
    ) -> np.ndarray | None:
        if observation is None:
            return None
        image = observation.frame_bgr.copy()
        if detection is not None and detection.mask is not None:
            mask = detection.mask > 0
            color = np.zeros_like(image)
            color[:, :, 1] = 180
            image[mask] = cv2.addWeighted(image[mask], 0.55, color[mask], 0.45, 0)
        if detection is not None and detection.bbox_xyxy is not None:
            x1, y1, x2, y2 = detection.bbox_xyxy.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if detection is not None and detection.centroid_px is not None:
            center = tuple(detection.centroid_px.astype(int))
            cv2.circle(image, center, 4, (0, 0, 255), -1)
        label = f"{self.detector_name} pending" if pending else f"{self.detector_name}"
        if detection is not None:
            label = f"{detection.backend} score={detection.score:.2f}"
        cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, self.config.target, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if self._mouse_drag_enabled:
            cv2.putText(image, "drag:L on", (10, image.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        return image

    def _update_viewer_overlay(self, viewer) -> None:
        if not self.config.camera_overlay or self._latest_overlay_bgr is None:
            return
        viewport = viewer.viewport
        if viewport is None or viewport.width <= 0 or viewport.height <= 0:
            return
        width = min(640, max(300, int(viewport.width * self.config.overlay_width_fraction)))
        height = int(width * self.config.camera.height / self.config.camera.width)
        height = min(height, max(180, int(viewport.height * 0.46)))
        x = max(0, int(viewport.width - width - 12))
        y = 12
        rect_key = (x, y, width, height)
        if self._latest_overlay_rgb is not None and self._overlay_rect_key == rect_key:
            return
        if self._latest_overlay_rgb is None or self._overlay_rect_key != rect_key:
            overlay = cv2.resize(self._latest_overlay_bgr, (width, height), interpolation=cv2.INTER_AREA)
            self._latest_overlay_rgb = overlay[:, :, ::-1].copy()
            self._overlay_rect_key = rect_key
        viewer.set_images((mujoco.MjrRect(x, y, width, height), self._latest_overlay_rgb))


def run_demo(config: DemoConfig) -> RunSummary:
    return VisualServoSimulation(config).run()
