from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class DashboardTelemetry:
    robot: str
    target: str
    servo_mode: str
    actuator_mode: str
    tracking_state: str
    manipulation_state: str
    sim_time_s: float
    position_error_m: float
    image_error_px: float
    contact_force_n: float = 0.0


class DashboardRenderer:
    """Compose a readable 16:9 presentation frame from a camera overlay."""

    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = int(width)
        self.height = int(height)
        if self.width < 640 or self.height < 360:
            raise ValueError("dashboard must be at least 640x360")

    def render(
        self, camera_bgr: np.ndarray, telemetry: DashboardTelemetry
    ) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), (25, 29, 38), dtype=np.uint8)
        panel_width = int(self.width * 0.30)
        view_width = self.width - panel_width
        frame = np.asarray(camera_bgr, dtype=np.uint8)
        scale = min(view_width / frame.shape[1], self.height / frame.shape[0])
        resized = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        y = (self.height - resized.shape[0]) // 2
        x = (view_width - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.rectangle(
            canvas, (view_width, 0), (self.width - 1, self.height - 1), (36, 43, 56), -1
        )
        cv2.putText(
            canvas,
            "MUJOCO VISUAL SERVO",
            (view_width + 22, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            (118, 220, 255),
            1,
            cv2.LINE_AA,
        )
        rows = (
            ("ROBOT", telemetry.robot),
            ("TARGET", telemetry.target),
            ("SERVO", telemetry.servo_mode.upper()),
            ("ACTUATOR", telemetry.actuator_mode.upper()),
            ("TRACKING", telemetry.tracking_state),
            ("TASK", telemetry.manipulation_state),
            ("TIME", f"{telemetry.sim_time_s:6.2f} s"),
            ("POSITION", f"{1000.0 * telemetry.position_error_m:6.1f} mm"),
            ("IMAGE", f"{telemetry.image_error_px:6.1f} px"),
            ("GRIP FORCE", f"{telemetry.contact_force_n:6.2f} N"),
        )
        baseline = 88
        for index, (label, value) in enumerate(rows):
            yy = baseline + index * 41
            cv2.putText(
                canvas,
                label,
                (view_width + 22, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (150, 164, 185),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                value,
                (view_width + 22, yy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (240, 244, 250),
                1,
                cv2.LINE_AA,
            )
        error_ratio = float(np.clip(telemetry.position_error_m / 0.15, 0.0, 1.0))
        left, right, yy = view_width + 22, self.width - 22, self.height - 42
        cv2.rectangle(canvas, (left, yy), (right, yy + 12), (65, 73, 88), -1)
        color = (66, 214, 116) if error_ratio < 0.2 else (58, 171, 245)
        cv2.rectangle(
            canvas,
            (left, yy),
            (left + int((right - left) * (1.0 - error_ratio)), yy + 12),
            color,
            -1,
        )
        return canvas
