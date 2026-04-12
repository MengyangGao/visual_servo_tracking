from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

from .camera import CameraStream, discover_cameras, open_camera
from .config import AppSettings, build_settings, lookup_target_prototype, target_world_position
from .config import project_root
from .control import ServoGains, compute_servo_command
from .geometry import normalize
from .perception import OracleBackend, PromptGuidedVisionBackend, build_backend
from .robot import build_robot_spec
from .scene import body_pose_world, build_scene_bundle
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
        f"backend={telemetry.backend} score={telemetry.detection_score:.2f} pos_err={telemetry.position_error_m:.3f}m",
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
    return CameraPose.identity()


def _camera_intrinsics_for_frame(frame: np.ndarray) -> CameraIntrinsics:
    h, w = frame.shape[:2]
    return CameraIntrinsics(fx=0.92 * w, fy=0.92 * w, cx=w / 2.0, cy=h / 2.0, width=w, height=h)


def run_simulation(settings: AppSettings, stop_event: Optional[threading.Event] = None) -> dict:
    output_dir = _resolve_output_dir(settings)
    robot_spec = build_robot_spec(prefer_reference=settings.use_reference_robot, scene_path=settings.robot_scene_path)
    bundle = build_scene_bundle(robot_spec, settings.prompt, settings.camera_width, settings.camera_height)
    model, data = bundle.model, bundle.data
    target_proto = bundle.target_proto
    backend = OracleBackend(target_world_position)
    gains = ServoGains()
    dt = 1.0 / settings.control_rate_hz
    trace: list[dict] = []
    limit = settings.max_steps
    step = 0
    while step < limit:
        if stop_event is not None and stop_event.is_set():
            break
        frame = np.zeros((settings.camera_height, settings.camera_width, 3), dtype=np.uint8)
        detection = backend.detect(frame, settings.prompt, bundle.camera_intrinsics, bundle.camera_pose)
        qpos_cmd, telemetry = compute_servo_command(
            model=model,
            data=data,
            detection=detection,
            prototype=target_proto,
            camera_intrinsics=bundle.camera_intrinsics,
            camera_pose=bundle.camera_pose,
            ee_body_name=bundle.ee_body_name,
            gains=gains,
            dt=dt,
        )
        telemetry.step = step
        for i, name in enumerate(bundle.actuator_names[: min(7, model.nu)]):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = qpos_cmd[i]
        mujoco.mj_step(model, data)
        trace.append(
            {
                "step": step,
                "backend": detection.backend,
                "position_error_m": telemetry.position_error_m,
                "orientation_error_rad": telemetry.orientation_error_rad,
            }
        )
        step += 1
    summary = {
        "mode": "sim",
        "prompt": settings.prompt,
        "backend": "oracle",
        "steps": settings.max_steps,
        "final_position_error_m": trace[-1]["position_error_m"] if trace else None,
        "final_orientation_error_rad": trace[-1]["orientation_error_rad"] if trace else None,
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
    camera = open_camera(settings.camera_index, width=settings.camera_width, height=settings.camera_height)
    backend = build_backend(settings.backend, settings.prompt, target_world_position)
    gains = ServoGains()
    dt = 1.0 / settings.control_rate_hz
    frame_count = 0
    trace: list[dict] = []
    writer = None
    if settings.record:
        writer = _make_writer(output_dir / "camera_tracking.mp4", (settings.camera_width, settings.camera_height), settings.control_rate_hz)
    try:
        limit = settings.max_steps if settings.run_mode == "auto" or (not settings.show_view and stop_event is None) else None
        while limit is None or frame_count < limit:
            if stop_event is not None and stop_event.is_set():
                break
            frame = camera.read()
            intrinsics = _camera_intrinsics_for_frame(frame.image_bgr)
            camera_pose = _camera_pose_for_real_mode()
            detection = backend.detect(frame.image_bgr, settings.prompt, intrinsics, camera_pose)
            qpos_cmd, telemetry = compute_servo_command(
                model=model,
                data=data,
                detection=detection,
                prototype=target_proto,
                camera_intrinsics=intrinsics,
                camera_pose=camera_pose,
                ee_body_name=bundle.ee_body_name,
                gains=gains,
                dt=dt,
            )
            telemetry.step = frame_count
            for i, name in enumerate(bundle.actuator_names[: min(7, model.nu)]):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    data.ctrl[aid] = qpos_cmd[i]
            mujoco.mj_step(model, data)
            annotated = _overlay_status(
                frame.image_bgr,
                detection,
                telemetry,
                title=f"{settings.mode} | {settings.run_mode} | {settings.prompt}",
            )
            if writer is not None:
                writer.write(annotated)
            if settings.show_view:
                cv2.imshow("mujoco-servo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            trace.append(
                {
                    "step": frame_count,
                    "backend": detection.backend,
                    "score": detection.score,
                    "position_error_m": telemetry.position_error_m,
                    "orientation_error_rad": telemetry.orientation_error_rad,
                    "bbox": detection.bbox_xyxy.tolist(),
                }
            )
            frame_count += 1
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if settings.show_view:
            cv2.destroyAllWindows()
    summary = {
        "mode": "camera",
        "prompt": settings.prompt,
        "backend": backend.name,
        "frames": frame_count,
        "final_position_error_m": trace[-1]["position_error_m"] if trace else None,
        "final_orientation_error_rad": trace[-1]["orientation_error_rad"] if trace else None,
        "camera_index": settings.camera_index,
        "robot": robot_spec.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "camera_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "camera_trace.json").write_text(json.dumps(trace, indent=2))
    return summary


def run_gui() -> None:
    from .gui import launch_gui

    launch_gui()
