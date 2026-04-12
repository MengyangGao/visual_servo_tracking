from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import panda_scene_path, project_root


@dataclass(slots=True)
class RobotSpec:
    name: str
    scene_xml_path: Path | None
    ee_body_name: str = "hand"
    joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    actuator_names: tuple[str, ...] = tuple(f"actuator{i}" for i in range(1, 8))

    @property
    def available(self) -> bool:
        return self.scene_xml_path is not None and self.scene_xml_path.exists()


def build_robot_spec(prefer_reference: bool = True, scene_path: Path | None = None) -> RobotSpec:
    if scene_path is not None:
        return RobotSpec(name="custom", scene_xml_path=scene_path)
    panda_path = panda_scene_path() if prefer_reference else None
    if panda_path is not None and panda_path.exists():
        return RobotSpec(name="franka_panda", scene_xml_path=panda_path)
    return RobotSpec(name="panda_lite", scene_xml_path=None)

