from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import AppSettings, CameraIntrinsics, CameraPose, TargetPrototype
from .geometry import look_at_rotation


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return repo_root() / "mujoco"


def reference_root() -> Path:
    return repo_root() / "reference"


def panda_scene_path() -> Path:
    return reference_root() / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"


TARGET_LIBRARY: dict[str, TargetPrototype] = {
    "cup": TargetPrototype("cup", "cylinder", (0.08, 0.08, 0.10), (0.95, 0.35, 0.20, 1.0), aliases=("mug", "cup")),
    "mug": TargetPrototype("cup", "cylinder", (0.08, 0.08, 0.10), (0.95, 0.35, 0.20, 1.0), aliases=("cup", "mug")),
    "phone": TargetPrototype("phone", "box", (0.075, 0.010, 0.150), (0.15, 0.15, 0.15, 1.0), aliases=("phone", "mobile")),
    "mobile": TargetPrototype("phone", "box", (0.075, 0.010, 0.150), (0.15, 0.15, 0.15, 1.0), aliases=("phone", "mobile")),
    "mouse": TargetPrototype("mouse", "box", (0.060, 0.040, 0.100), (0.85, 0.80, 0.75, 1.0), aliases=("mouse",)),
    "apple": TargetPrototype("apple", "sphere", (0.075, 0.075, 0.075), (0.90, 0.18, 0.18, 1.0), aliases=("apple",)),
    "book": TargetPrototype("book", "box", (0.160, 0.025, 0.230), (0.18, 0.25, 0.75, 1.0), aliases=("book",)),
    "bottle": TargetPrototype("bottle", "cylinder", (0.045, 0.045, 0.240), (0.10, 0.50, 0.90, 1.0), aliases=("bottle",)),
    "box": TargetPrototype("box", "box", (0.110, 0.080, 0.110), (0.65, 0.45, 0.20, 1.0), aliases=("box",)),
}


def canonical_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().strip().split())


def lookup_target_prototype(prompt: str) -> TargetPrototype:
    text = canonical_prompt(prompt)
    for key, prototype in TARGET_LIBRARY.items():
        if key in text or any(alias in text for alias in prototype.aliases):
            return prototype
    return TargetPrototype("object", "box", (0.10, 0.10, 0.10), (0.85, 0.25, 0.25, 1.0), aliases=("object",))


def target_world_position(prompt: str) -> np.ndarray:
    key = lookup_target_prototype(prompt).name
    table = {
        "cup": np.array([0.58, 0.10, 0.25], dtype=float),
        "phone": np.array([0.62, -0.08, 0.24], dtype=float),
        "mouse": np.array([0.55, -0.18, 0.24], dtype=float),
        "apple": np.array([0.50, 0.18, 0.26], dtype=float),
        "book": np.array([0.60, 0.00, 0.26], dtype=float),
        "bottle": np.array([0.56, -0.22, 0.28], dtype=float),
        "box": np.array([0.59, 0.03, 0.25], dtype=float),
        "object": np.array([0.58, 0.05, 0.25], dtype=float),
    }
    return table.get(key, table["object"]).copy()


def default_camera_intrinsics(width: int, height: int) -> CameraIntrinsics:
    fx = 0.92 * width
    fy = 0.92 * width
    return CameraIntrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def canonical_camera_pose() -> CameraPose:
    translation = np.array([0.18, -0.95, 0.82], dtype=float)
    look_at = np.array([0.56, 0.00, 0.25], dtype=float)
    rotation = look_at_rotation(look_at - translation, np.array([0.0, 0.0, 1.0], dtype=float))
    return CameraPose(translation_m=translation, rotation_world_from_cam=rotation)


def build_settings(
    prompt: str = "cup",
    backend: str = "auto",
    mode: str = "sim",
    run_mode: str = "auto",
    vision_preset: str = "default",
    max_steps: int = 240,
    camera_index: int | None = None,
    show_view: bool = True,
    record: bool = False,
) -> AppSettings:
    return AppSettings(
        prompt=prompt,
        backend=backend,
        mode=mode,
        run_mode=run_mode,
        vision_preset=vision_preset,
        max_steps=max_steps,
        camera_index=camera_index,
        show_view=show_view,
        record=record,
    )
