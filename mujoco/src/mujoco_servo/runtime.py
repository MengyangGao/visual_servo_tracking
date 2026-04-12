from __future__ import annotations

import json
import os
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

from .camera import discover_cameras, open_camera
from .config import AppSettings, build_settings, canonical_camera_pose, moving_target_world_position, target_world_position
from .config import project_root
from .control import ServoGains, compute_servo_command
from .geometry import rotation_matrix_to_quaternion_wxyz
from .perception import GroundedSam2Config, OracleBackend, PerceptionSession, build_backend
from .robot import build_robot_spec
from .rendering import MujocoSceneRenderer, MujocoViewerSession, ViewLayout, three_panel_view
from .scene import body_pose_world, build_scene_bundle, set_mocap_body_pose
from .types import CameraFrame, CameraIntrinsics, CameraPose, Detection, ServoTelemetry


@dataclass(slots=True)
class _LoopState:
    filtered_detection: Detection | None = None
    missing_frames: int = 0
    filtered_qpos: np.ndarray | None = None


def _blend_array(previous: np.ndarray | None, current: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if current is None:
        return None if previous is None else previous.copy()
    if previous is None:
        return current.copy()
    prev = np.asarray(previous, dtype=float)
    curr = np.asarray(current, dtype=float)
    if prev.shape != curr.shape:
        return curr.copy()
    return ((1.0 - alpha) * prev + alpha * curr).astype(float)


def _smooth_detection(previous: Detection, current: Detection, alpha: float = 0.35) -> Detection:
    corners = _blend_array(previous.corners_px, current.corners_px, alpha)
    mask = current.mask if current.mask is not None else previous.mask
    target_world = _blend_array(previous.target_position_world, current.target_position_world, alpha)
    return Detection(
        success=True,
        prompt=current.prompt,
        label=current.label,
        score=float((1.0 - alpha) * previous.score + alpha * current.score),
        bbox_xyxy=((1.0 - alpha) * previous.bbox_xyxy + alpha * current.bbox_xyxy).astype(float),
        centroid_px=((1.0 - alpha) * previous.centroid_px + alpha * current.centroid_px).astype(float),
        corners_px=corners,
        mask=mask,
        mask_area_px=int((1.0 - alpha) * previous.mask_area_px + alpha * current.mask_area_px),
        estimated_distance_m=_blend_array(
            np.array([previous.estimated_distance_m], dtype=float) if previous.estimated_distance_m is not None else None,
            np.array([current.estimated_distance_m], dtype=float) if current.estimated_distance_m is not None else None,
            alpha,
        )[0]
        if previous.estimated_distance_m is not None and current.estimated_distance_m is not None
        else current.estimated_distance_m or previous.estimated_distance_m,
        backend=current.backend,
        track_id=current.track_id or previous.track_id,
        target_position_world=target_world,
    )


def _limit_command(previous: np.ndarray | None, current: np.ndarray, max_delta: float) -> np.ndarray:
    command = np.asarray(current, dtype=float).copy()
    if previous is None:
        return command
    prev = np.asarray(previous, dtype=float).reshape(-1)
    if prev.shape != command.shape:
        return command
    delta = np.clip(command - prev, -max_delta, max_delta)
    return prev + delta


def _tracking_lookat(ee_pos: np.ndarray, target_pos: np.ndarray) -> tuple[float, float, float]:
    ee = np.asarray(ee_pos, dtype=float).reshape(3)
    target = np.asarray(target_pos, dtype=float).reshape(3)
    mid = 0.5 * (ee + target)
    mid[2] = float(mid[2] + 0.12)
    return float(mid[0]), float(mid[1]), float(mid[2])


def _world_panel_lookat(prompt: str) -> tuple[float, float, float]:
    target = np.asarray(target_world_position(prompt), dtype=float)
    return float(target[0] - 0.05), float(target[1] - 0.25), float(target[2] + 0.22)


def _camera_panel_lookat(prompt: str, target_pos: np.ndarray) -> tuple[float, float, float]:
    target = np.asarray(target_pos, dtype=float).reshape(3)
    nominal = np.asarray(target_world_position(prompt), dtype=float)
    mid = 0.65 * target + 0.35 * nominal
    mid[2] = float(mid[2] + 0.18)
    return float(mid[0]), float(mid[1]), float(mid[2])


def _sim_target_position(prompt: str, step: int, dt: float) -> np.ndarray:
    return moving_target_world_position(prompt, step * dt)


def _overlay_status(frame_bgr: np.ndarray, detection: Detection, telemetry: ServoTelemetry, title: str) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    if detection.success:
        x1, y1, x2, y2 = detection.bbox_xyxy.astype(int)
        cv2.rectangle(img, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), (0, 255, 0), 2)
        cv2.circle(img, tuple(np.asarray(detection.centroid_px, dtype=int)), 4, (0, 0, 255), -1)
    cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"backend={telemetry.backend} score={telemetry.detection_score:.2f} feat={telemetry.feature_error_px:.1f}px dist={telemetry.target_distance_m:.3f}m stand={telemetry.standoff_error_m:.3f}m",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def _make_writer(path: Path, frame_size: tuple[int, int], fps: float = 30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


def _resolve_output_dir(settings: AppSettings) -> Path:
    if settings.output_dir.is_absolute():
        return settings.output_dir
    return project_root() / settings.output_dir


def _camera_pose_for_real_mode() -> CameraPose:
    return canonical_camera_pose()


def _camera_intrinsics_for_frame(frame: np.ndarray) -> CameraIntrinsics:
    h, w = frame.shape[:2]
    return CameraIntrinsics(fx=0.92 * w, fy=0.92 * w, cx=w / 2.0, cy=h / 2.0, width=w, height=h)


def _world_view_lookat(prompt: str) -> tuple[float, float, float]:
    target = np.asarray(target_world_position(prompt), dtype=float)
    return float(target[0]), float(target[1]), float(target[2] + 0.18)


def _can_use_mujoco_renderer() -> bool:
    forced = os.getenv("MUJOCO_SERVO_FORCE_RENDERER", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if sys.platform == "darwin":
        return Path(sys.executable).name == "mjpython"
    return True


def _vision_config(settings: AppSettings) -> GroundedSam2Config | None:
    normalized = settings.backend.strip().lower()
    if normalized in {"grounded-sam2", "grounded_sam2", "open-vocab", "open_vocab", "auto"}:
        return GroundedSam2Config.from_preset(settings.vision_preset)
    return None


def _hold_control(model: mujoco.MjModel, data: mujoco.MjData, prompt: str, backend: str, step: int, ee_body_name: str, detection_score: float) -> tuple[np.ndarray, ServoTelemetry]:
    ee_pos, ee_rot = body_pose_world(model, data, ee_body_name)
    qpos = np.array(data.qpos.copy(), dtype=float)
    telemetry = ServoTelemetry(
        step=step,
        prompt=prompt,
        backend=backend,
        qpos=qpos.copy(),
        qvel=np.zeros(model.nv, dtype=float),
        ee_position_m=ee_pos.copy(),
        ee_orientation_wxyz=rotation_matrix_to_quaternion_wxyz(ee_rot),
        target_position_m=ee_pos.copy(),
        position_error_m=0.0,
        orientation_error_rad=0.0,
        detection_score=float(detection_score),
    )
    return qpos, telemetry


def run_simulation(settings: AppSettings, stop_event: Optional[threading.Event] = None) -> dict:
    output_dir = _resolve_output_dir(settings)
    robot_spec = build_robot_spec(prefer_reference=settings.use_reference_robot, scene_path=settings.robot_scene_path)
    bundle = build_scene_bundle(robot_spec, settings.prompt, settings.camera_width, settings.camera_height)
    model, data = bundle.model, bundle.data
    target_proto = bundle.target_proto
    ee_start_pos, _ = body_pose_world(model, data, bundle.ee_body_name)
    dt = 1.0 / settings.control_rate_hz
    gains = ServoGains()
    state = _LoopState()
    trace: list[dict] = []
    need_renderer = (settings.show_view or settings.record or settings.backend.strip().lower() not in {"oracle", "simulation", "sim"})
    can_render = need_renderer and _can_use_mujoco_renderer()
    world_renderer = None
    follow_renderer = None
    sensor_renderer = None
    viewer_session = None
    writer = None
    if can_render:
        try:
            world_renderer = MujocoSceneRenderer(
                model,
                width=settings.robot_view_width,
                height=settings.robot_view_height,
                lookat=_world_view_lookat(settings.prompt),
                distance=2.9,
                azimuth=126.0,
                elevation=-20.0,
            )
            follow_renderer = MujocoSceneRenderer(
                model,
                width=settings.robot_view_width,
                height=settings.robot_view_height,
                lookat=_tracking_lookat(ee_start_pos, target_world_position(settings.prompt)),
                distance=1.65,
                azimuth=150.0,
                elevation=-16.0,
            )
            sensor_renderer = MujocoSceneRenderer(
                model,
                width=settings.camera_width,
                height=settings.camera_height,
                lookat=_camera_panel_lookat(settings.prompt, target_world_position(settings.prompt)),
                distance=1.45,
                azimuth=138.0,
                elevation=-18.0,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Robot-view renderer unavailable, falling back to oracle-style simulation: {exc}", RuntimeWarning)
            world_renderer = None
            follow_renderer = None
            sensor_renderer = None
    if settings.show_view and threading.current_thread() is threading.main_thread() and _can_use_mujoco_renderer():
        try:
            viewer_session = MujocoViewerSession(
                model,
                data,
                lookat=_world_view_lookat(settings.prompt),
                distance=2.9,
                azimuth=126.0,
                elevation=-20.0,
            )
        except Exception:
            viewer_session = None

    current_target_world = _sim_target_position(settings.prompt, 0, dt)
    backend_name = settings.backend.strip().lower()
    if sensor_renderer is None and backend_name not in {"oracle", "simulation", "sim"}:
        backend = OracleBackend(lambda _prompt: current_target_world.copy())
    else:
        backend = build_backend(settings.backend, settings.prompt, lambda _prompt: current_target_world.copy(), config=_vision_config(settings))
    session = PerceptionSession(backend, settings.prompt)
    hold_frames = 6
    limit = settings.max_steps
    step = 0
    while step < limit:
        if stop_event is not None and stop_event.is_set():
            break
        current_target_world = _sim_target_position(settings.prompt, step, dt)
        set_mocap_body_pose(model, data, "target", current_target_world)
        mujoco.mj_forward(model, data)
        ee_pos, ee_rot = body_pose_world(model, data, bundle.ee_body_name)
        if sensor_renderer is not None:
            sensor_renderer.set_distance(float(np.clip(1.20 + 1.5 * np.linalg.norm(current_target_world - ee_pos), 1.25, 2.3)))
            sensor_frame = sensor_renderer.render_with_lookat(data, _camera_panel_lookat(settings.prompt, current_target_world))
        else:
            sensor_frame = np.zeros((settings.camera_height, settings.camera_width, 3), dtype=np.uint8)
        raw_detection = session.update(sensor_frame, bundle.camera_intrinsics, bundle.camera_pose)
        if raw_detection.success:
            state.missing_frames = 0
            state.filtered_detection = raw_detection if state.filtered_detection is None else _smooth_detection(state.filtered_detection, raw_detection)
        else:
            state.missing_frames += 1
            if state.missing_frames > hold_frames:
                state.filtered_detection = None
        control_detection = state.filtered_detection if state.filtered_detection is not None else raw_detection if raw_detection.success else None
        if control_detection is None:
            qpos_cmd, telemetry = _hold_control(
                model,
                data,
                settings.prompt,
                raw_detection.backend,
                step,
                bundle.ee_body_name,
                raw_detection.score,
            )
            state.filtered_qpos = qpos_cmd.copy()
        else:
            qpos_cmd, telemetry = compute_servo_command(
                model=model,
                data=data,
                detection=control_detection,
                prototype=target_proto,
                camera_intrinsics=bundle.camera_intrinsics,
                camera_pose=bundle.camera_pose,
                ee_body_name=bundle.ee_body_name,
                gains=gains,
                dt=dt,
            )
            state.filtered_qpos = _limit_command(state.filtered_qpos, qpos_cmd, gains.max_joint_delta)
            qpos_cmd = state.filtered_qpos.copy()
        telemetry.step = step
        for i, name in enumerate(bundle.actuator_names[: min(7, model.nu)]):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = qpos_cmd[i]
        mujoco.mj_step(model, data)
        ee_pos, ee_rot = body_pose_world(model, data, bundle.ee_body_name)
        set_mocap_body_pose(model, data, "vision_camera", ee_pos, ee_rot)
        set_mocap_body_pose(model, data, "vision_target", current_target_world)
        set_mocap_body_pose(model, data, "vision_ee", ee_pos, ee_rot)
        mujoco.mj_forward(model, data)
        world_view = world_renderer.render_with_lookat(data, _world_view_lookat(settings.prompt)) if world_renderer is not None else None
        if follow_renderer is not None:
            follow_renderer.set_distance(float(np.clip(1.00 + 1.35 * np.linalg.norm(current_target_world - ee_pos), 1.10, 2.05)))
            follow_view = follow_renderer.render_with_lookat(data, _tracking_lookat(ee_pos, current_target_world))
        else:
            follow_view = None
        status = "tracking" if raw_detection.success else ("hold" if state.filtered_detection is not None else "searching")
        annotated_sensor = _overlay_status(
            sensor_frame,
            control_detection or raw_detection,
            telemetry,
            title=f"sim | auto | {settings.prompt} | {status}",
        )
        footer_lines = [
            f"backend={backend.name} prompt={settings.prompt} mode=sim control=continuous",
            f"score={telemetry.detection_score:.2f} feat={telemetry.feature_error_px:.1f}px dist={telemetry.target_distance_m:.3f}m standoff={telemetry.standoff_error_m:.3f}m",
            "legend: world=full scene | follow=robot-centered | camera=sensor view",
            "markers: blue=camera | red=target estimate | cyan=ee | orange=object",
        ]
        dashboard_frame = three_panel_view(
            world_view,
            follow_view,
            annotated_sensor,
            layout=ViewLayout(world_title="MuJoCo world", robot_title="Robot follow", camera_title="Sensor / detection"),
            footer_lines=footer_lines,
        )
        if settings.record and writer is None:
            writer = _make_writer(output_dir / "sim_dashboard.mp4", (dashboard_frame.shape[1], dashboard_frame.shape[0]), settings.control_rate_hz)
        if writer is not None:
            writer.write(dashboard_frame)
        if viewer_session is not None:
            viewer_session.sync()
            if not viewer_session.is_running():
                break
        trace.append(
            {
                "step": step,
                "backend": raw_detection.backend,
                "position_error_m": telemetry.position_error_m,
                "standoff_error_m": telemetry.standoff_error_m,
                "target_distance_m": telemetry.target_distance_m,
                "orientation_error_rad": telemetry.orientation_error_rad,
                "feature_error_px": telemetry.feature_error_px,
                "raw_success": raw_detection.success,
                "used_hold": bool(not raw_detection.success and state.filtered_detection is None),
            }
        )
        step += 1
    if world_renderer is not None:
        world_renderer.close()
    if follow_renderer is not None:
        follow_renderer.close()
    if sensor_renderer is not None:
        sensor_renderer.close()
    if writer is not None:
        writer.release()
    if viewer_session is not None:
        viewer_session.close()
    summary = {
        "mode": "sim",
        "prompt": settings.prompt,
        "backend": backend.name,
        "steps": settings.max_steps,
        "final_position_error_m": trace[-1]["position_error_m"] if trace else None,
        "final_standoff_error_m": trace[-1]["standoff_error_m"] if trace else None,
        "final_target_distance_m": trace[-1]["target_distance_m"] if trace else None,
        "final_orientation_error_rad": trace[-1]["orientation_error_rad"] if trace else None,
        "final_feature_error_px": trace[-1]["feature_error_px"] if trace else None,
        "robot": robot_spec.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sim_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "sim_trace.json").write_text(json.dumps(trace, indent=2))
    return summary


def run_camera(
    settings: AppSettings,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    output_dir = _resolve_output_dir(settings)
    robot_spec = build_robot_spec(prefer_reference=settings.use_reference_robot, scene_path=settings.robot_scene_path)
    bundle = build_scene_bundle(robot_spec, settings.prompt, settings.camera_width, settings.camera_height)
    model, data = bundle.model, bundle.data
    target_proto = bundle.target_proto
    ee_start_pos, _ = body_pose_world(model, data, bundle.ee_body_name)
    camera = open_camera(settings.camera_index, width=settings.camera_width, height=settings.camera_height)
    backend = build_backend(settings.backend, settings.prompt, target_world_position, config=_vision_config(settings))
    session = PerceptionSession(backend, settings.prompt)
    gains = ServoGains()
    dt = 1.0 / settings.control_rate_hz
    frame_count = 0
    trace: list[dict] = []
    writer = None
    world_renderer = None
    follow_renderer = None
    viewer_session = None
    if (settings.show_view or settings.record) and _can_use_mujoco_renderer():
        try:
            world_renderer = MujocoSceneRenderer(
                model,
                width=settings.robot_view_width,
                height=settings.robot_view_height,
                lookat=_world_view_lookat(settings.prompt),
                distance=2.9,
                azimuth=126.0,
                elevation=-20.0,
            )
            follow_renderer = MujocoSceneRenderer(
                model,
                width=settings.robot_view_width,
                height=settings.robot_view_height,
                lookat=_tracking_lookat(ee_start_pos, target_world_position(settings.prompt)),
                distance=1.65,
                azimuth=150.0,
                elevation=-16.0,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Robot-view renderer unavailable in camera mode: {exc}", RuntimeWarning)
            world_renderer = None
            follow_renderer = None
    if settings.show_view and threading.current_thread() is threading.main_thread() and _can_use_mujoco_renderer():
        try:
            viewer_session = MujocoViewerSession(
                model,
                data,
                lookat=_world_view_lookat(settings.prompt),
                distance=2.9,
                azimuth=126.0,
                elevation=-20.0,
            )
        except Exception:
            viewer_session = None

    state = _LoopState()
    hold_frames = 6
    try:
        limit = settings.max_steps if settings.run_mode == "auto" or (not settings.show_view and stop_event is None) else None
        while limit is None or frame_count < limit:
            if stop_event is not None and stop_event.is_set():
                break
            frame = camera.read()
            intrinsics = _camera_intrinsics_for_frame(frame.image_bgr)
            camera_pose = _camera_pose_for_real_mode()
            raw_detection = session.update(frame.image_bgr, intrinsics, camera_pose)
            if raw_detection.success:
                state.missing_frames = 0
                state.filtered_detection = raw_detection if state.filtered_detection is None else _smooth_detection(state.filtered_detection, raw_detection)
            else:
                state.missing_frames += 1
                if state.missing_frames > hold_frames:
                    state.filtered_detection = None
            control_detection = state.filtered_detection if state.filtered_detection is not None else raw_detection if raw_detection.success else None
            if control_detection is None:
                qpos_cmd, telemetry = _hold_control(
                    model,
                    data,
                    settings.prompt,
                    raw_detection.backend,
                    frame_count,
                    bundle.ee_body_name,
                    raw_detection.score,
                )
                state.filtered_qpos = qpos_cmd.copy()
            else:
                qpos_cmd, telemetry = compute_servo_command(
                    model=model,
                    data=data,
                    detection=control_detection,
                    prototype=target_proto,
                    camera_intrinsics=intrinsics,
                    camera_pose=camera_pose,
                    ee_body_name=bundle.ee_body_name,
                    gains=gains,
                    dt=dt,
                )
                state.filtered_qpos = _limit_command(state.filtered_qpos, qpos_cmd, gains.max_joint_delta)
                qpos_cmd = state.filtered_qpos.copy()
            telemetry.step = frame_count
            for i, name in enumerate(bundle.actuator_names[: min(7, model.nu)]):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    data.ctrl[aid] = qpos_cmd[i]
            mujoco.mj_step(model, data)
            ee_pos, ee_rot = body_pose_world(model, data, bundle.ee_body_name)
            set_mocap_body_pose(model, data, "vision_camera", bundle.camera_pose.translation_m, bundle.camera_pose.rotation_world_from_cam)
            set_mocap_body_pose(model, data, "vision_target", telemetry.target_position_m)
            set_mocap_body_pose(model, data, "vision_ee", ee_pos, ee_rot)
            mujoco.mj_forward(model, data)
            world_view = world_renderer.render_with_lookat(data, _world_view_lookat(settings.prompt)) if world_renderer is not None else None
            if follow_renderer is not None:
                follow_renderer.set_distance(float(np.clip(1.00 + 1.35 * np.linalg.norm(telemetry.target_position_m - ee_pos), 1.10, 2.05)))
                follow_view = follow_renderer.render_with_lookat(data, _tracking_lookat(ee_pos, telemetry.target_position_m))
            else:
                follow_view = None
            status = "tracking" if raw_detection.success else ("hold" if state.filtered_detection is not None else "searching")
            annotated = _overlay_status(
                frame.image_bgr,
                control_detection or raw_detection,
                telemetry,
                title=f"{settings.mode} | {settings.run_mode} | {settings.prompt} | {status}",
            )
            footer_lines = [
                f"backend={backend.name} prompt={settings.prompt} mode=camera control=continuous",
                f"camera={frame.device_index}:{frame.backend_name} score={telemetry.detection_score:.2f} feat={telemetry.feature_error_px:.1f}px dist={telemetry.target_distance_m:.3f}m stand={telemetry.standoff_error_m:.3f}m",
                "legend: world=full scene | follow=robot-centered | camera=real sensor",
                "markers: blue=camera | red=target estimate | cyan=ee | orange=object",
            ]
            display_frame = three_panel_view(
                world_view,
                follow_view,
                annotated,
                layout=ViewLayout(world_title="MuJoCo world", robot_title="Robot follow", camera_title="Camera / detection"),
                footer_lines=footer_lines,
            )
            if settings.record and writer is None:
                writer = _make_writer(
                    output_dir / "camera_dashboard.mp4",
                    (display_frame.shape[1], display_frame.shape[0]),
                    settings.control_rate_hz,
                )
            if writer is not None:
                writer.write(display_frame)
            if viewer_session is not None:
                viewer_session.sync()
                if not viewer_session.is_running():
                    break
            trace.append(
                {
                    "step": frame_count,
                    "backend": raw_detection.backend,
                    "score": raw_detection.score,
                    "position_error_m": telemetry.position_error_m,
                    "standoff_error_m": telemetry.standoff_error_m,
                    "target_distance_m": telemetry.target_distance_m,
                    "orientation_error_rad": telemetry.orientation_error_rad,
                    "feature_error_px": telemetry.feature_error_px,
                    "bbox": control_detection.bbox_xyxy.tolist() if control_detection is not None else raw_detection.bbox_xyxy.tolist(),
                    "raw_success": raw_detection.success,
                    "used_hold": bool(not raw_detection.success and state.filtered_detection is not None),
                }
            )
            frame_count += 1
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if world_renderer is not None:
            world_renderer.close()
        if follow_renderer is not None:
            follow_renderer.close()
    if viewer_session is not None:
        viewer_session.close()
    summary = {
        "mode": "camera",
        "prompt": settings.prompt,
        "backend": backend.name,
        "frames": frame_count,
        "final_position_error_m": trace[-1]["position_error_m"] if trace else None,
        "final_standoff_error_m": trace[-1]["standoff_error_m"] if trace else None,
        "final_target_distance_m": trace[-1]["target_distance_m"] if trace else None,
        "final_orientation_error_rad": trace[-1]["orientation_error_rad"] if trace else None,
        "final_feature_error_px": trace[-1]["feature_error_px"] if trace else None,
        "camera_index": settings.camera_index,
        "robot": robot_spec.name,
        "vision_preset": settings.vision_preset,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "camera_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "camera_trace.json").write_text(json.dumps(trace, indent=2))
    return summary


def run_gui() -> None:
    from .gui import launch_gui

    launch_gui()
