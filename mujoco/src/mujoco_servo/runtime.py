from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

from .camera import discover_cameras, open_camera
from .config import AppSettings, build_settings, canonical_camera_pose, target_world_position
from .config import project_root
from .control import ServoGains, compute_servo_command
from .geometry import rotation_matrix_to_quaternion_wxyz
from .perception import GroundedSam2Config, OracleBackend, PerceptionSession, build_backend
from .robot import build_robot_spec
from .preview import CameraPreviewWindow
from .rendering import MujocoSceneRenderer, MujocoViewerSession, ViewLayout, side_by_side_view
from .scene import body_pose_world, build_scene_bundle, set_mocap_body_pose
from .types import CameraFrame, CameraIntrinsics, CameraPose, Detection, ServoTelemetry


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
        f"backend={telemetry.backend} score={telemetry.detection_score:.2f} feat={telemetry.feature_error_px:.1f}px pos={telemetry.position_error_m:.3f}m",
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


def _should_open_camera_preview() -> bool:
    forced = os.getenv("MUJOCO_SERVO_ENABLE_TK_PREVIEW", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False
    return sys.platform != "darwin"


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
    session = PerceptionSession(OracleBackend(target_world_position), settings.prompt)
    gains = ServoGains()
    dt = 1.0 / settings.control_rate_hz
    renderer = None
    writer = None
    viewer = None
    if settings.show_view or settings.record:
        lookat = _world_view_lookat(settings.prompt)
        renderer = MujocoSceneRenderer(
            model,
            width=settings.robot_view_width,
            height=settings.robot_view_height,
            lookat=(float(ee_start_pos[0]), float(ee_start_pos[1]), float(ee_start_pos[2] + 0.12)),
            distance=1.2,
            azimuth=160.0,
            elevation=-18.0,
            follow_body_name=bundle.ee_body_name,
        )
    if settings.show_view:
        lookat = _world_view_lookat(settings.prompt)
        viewer = MujocoViewerSession(
            model,
            data,
            lookat=lookat,
            distance=2.4,
            azimuth=128.0,
            elevation=-22.0,
        )
    trace: list[dict] = []
    limit = settings.max_steps
    step = 0
    last_good_detection: Detection | None = None
    while step < limit:
        if stop_event is not None and stop_event.is_set():
            break
        frame = np.zeros((settings.camera_height, settings.camera_width, 3), dtype=np.uint8)
        raw_detection = session.update(frame, bundle.camera_intrinsics, bundle.camera_pose)
        control_detection = raw_detection if raw_detection.success else last_good_detection
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
            if raw_detection.success:
                last_good_detection = raw_detection
        telemetry.step = step
        for i, name in enumerate(bundle.actuator_names[: min(7, model.nu)]):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = qpos_cmd[i]
        mujoco.mj_step(model, data)
        ee_pos, ee_rot = body_pose_world(model, data, bundle.ee_body_name)
        set_mocap_body_pose(model, data, "vision_camera", bundle.camera_pose.translation_m, bundle.camera_pose.rotation_world_from_cam)
        set_mocap_body_pose(model, data, "vision_target", telemetry.target_position_m)
        set_mocap_body_pose(model, data, "vision_ee", ee_pos, ee_rot)
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break
        robot_view = renderer.render(data) if renderer is not None else None
        if settings.record and writer is None and robot_view is not None:
            writer = _make_writer(output_dir / "sim_tracking.mp4", (robot_view.shape[1], robot_view.shape[0]), settings.control_rate_hz)
        if writer is not None and robot_view is not None:
            writer.write(robot_view)
        trace.append(
            {
                "step": step,
                "backend": raw_detection.backend,
                "position_error_m": telemetry.position_error_m,
                "orientation_error_rad": telemetry.orientation_error_rad,
                "feature_error_px": telemetry.feature_error_px,
                "raw_success": raw_detection.success,
                "used_hold": bool(not raw_detection.success and last_good_detection is None),
            }
        )
        step += 1
    if renderer is not None:
        renderer.close()
    if viewer is not None:
        viewer.close()
    if writer is not None:
        writer.release()
    summary = {
        "mode": "sim",
        "prompt": settings.prompt,
        "backend": "oracle",
        "steps": settings.max_steps,
        "final_position_error_m": trace[-1]["position_error_m"] if trace else None,
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
    renderer = None
    viewer = None
    camera_preview: CameraPreviewWindow | None = None
    if settings.show_view or settings.record:
        lookat = _world_view_lookat(settings.prompt)
        renderer = MujocoSceneRenderer(
            model,
            width=settings.robot_view_width,
            height=settings.robot_view_height,
            lookat=(float(ee_start_pos[0]), float(ee_start_pos[1]), float(ee_start_pos[2] + 0.12)),
            distance=1.2,
            azimuth=160.0,
            elevation=-18.0,
            follow_body_name=bundle.ee_body_name,
        )
    if settings.show_view:
        lookat = _world_view_lookat(settings.prompt)
        viewer = MujocoViewerSession(
            model,
            data,
            lookat=lookat,
            distance=2.4,
            azimuth=128.0,
            elevation=-22.0,
        )
        if threading.current_thread() is threading.main_thread() and _should_open_camera_preview():
            try:
                camera_preview = CameraPreviewWindow(title="mujoco-servo camera")
            except Exception:
                camera_preview = None
    last_good_detection: Detection | None = None
    try:
        limit = settings.max_steps if settings.run_mode == "auto" or (not settings.show_view and stop_event is None) else None
        while limit is None or frame_count < limit:
            if stop_event is not None and stop_event.is_set():
                break
            frame = camera.read()
            intrinsics = _camera_intrinsics_for_frame(frame.image_bgr)
            camera_pose = _camera_pose_for_real_mode()
            raw_detection = session.update(frame.image_bgr, intrinsics, camera_pose)
            control_detection = raw_detection if raw_detection.success else last_good_detection
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
                if raw_detection.success:
                    last_good_detection = raw_detection
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
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
            robot_view = renderer.render(data) if renderer is not None else None
            status = "tracking" if raw_detection.success else ("hold" if last_good_detection is not None else "searching")
            annotated = _overlay_status(
                frame.image_bgr,
                raw_detection if raw_detection.success else (last_good_detection or raw_detection),
                telemetry,
                title=f"{settings.mode} | {settings.run_mode} | {settings.prompt} | {status}",
            )
            display_layout = ViewLayout(robot_title="MuJoCo follow", camera_title="Real camera")
            display_frame = side_by_side_view(robot_view, annotated, layout=display_layout) if robot_view is not None else annotated
            if settings.record and writer is None:
                writer = _make_writer(
                    output_dir / "camera_tracking.mp4",
                    (display_frame.shape[1], display_frame.shape[0]),
                    settings.control_rate_hz,
                )
            if writer is not None:
                writer.write(display_frame)
            if camera_preview is not None:
                camera_preview.update(display_frame)
                if not camera_preview.is_open():
                    break
            trace.append(
                {
                    "step": frame_count,
                    "backend": raw_detection.backend,
                    "score": raw_detection.score,
                    "position_error_m": telemetry.position_error_m,
                    "orientation_error_rad": telemetry.orientation_error_rad,
                    "feature_error_px": telemetry.feature_error_px,
                    "bbox": (raw_detection if raw_detection.success else (last_good_detection or raw_detection)).bbox_xyxy.tolist(),
                    "raw_success": raw_detection.success,
                    "used_hold": bool(not raw_detection.success and last_good_detection is not None),
                }
            )
            frame_count += 1
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()
    if viewer is not None:
        viewer.close()
        if camera_preview is not None:
            camera_preview.close()
    summary = {
        "mode": "camera",
        "prompt": settings.prompt,
        "backend": backend.name,
        "frames": frame_count,
        "final_position_error_m": trace[-1]["position_error_m"] if trace else None,
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
