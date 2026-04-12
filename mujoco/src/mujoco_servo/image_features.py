from __future__ import annotations

import numpy as np
import cv2


def order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("expected 4 corner points")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    signed_area = 0.0
    for i in range(4):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % 4]
        signed_area += x1 * y2 - x2 * y1
    if signed_area < 0.0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def bbox_corners_xyxy(bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=float).reshape(4)
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return order_corners_clockwise(
        np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float32,
        )
    )


def corners_from_mask(mask: np.ndarray | None, min_area: float = 20.0) -> np.ndarray | None:
    if mask is None:
        return None
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError("mask must be a 2D binary image")
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_corners_clockwise(box)
