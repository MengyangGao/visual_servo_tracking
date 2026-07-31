from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(slots=True, frozen=True)
class GraspEvidence:
    contact_bodies: tuple[str, ...]
    normal_force_n: float
    opposing_contacts: bool
    stable_frames: int
    relative_slip_m: float
    grasped: bool


class ContactGraspEvaluator:
    """Verify a frictional grasp without constraints, welds, or pose teleportation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        target_body_name: str,
        gripper_body_names: tuple[str, ...],
        attachment_body_name: str,
        min_normal_force_n: float = 0.15,
        max_relative_slip_m: float = 0.006,
        confirmation_frames: int = 8,
    ) -> None:
        self.model = model
        self.target_body_id = self._body_id(target_body_name)
        self.attachment_body_id = self._body_id(attachment_body_name)
        self.gripper_body_ids = tuple(
            self._body_id(name) for name in gripper_body_names
        )
        if len(set(self.gripper_body_ids)) < 2:
            raise ValueError(
                "physical grasp verification requires at least two gripper bodies"
            )
        self.min_normal_force_n = float(min_normal_force_n)
        self.max_relative_slip_m = float(max_relative_slip_m)
        self.confirmation_frames = int(confirmation_frames)
        if not np.isfinite(self.min_normal_force_n) or self.min_normal_force_n <= 0.0:
            raise ValueError("min_normal_force_n must be positive and finite")
        if (
            not np.isfinite(self.max_relative_slip_m)
            or self.max_relative_slip_m <= 0.0
        ):
            raise ValueError("max_relative_slip_m must be positive and finite")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least one")
        self._stable_frames = 0
        self._previous_relative_position: np.ndarray | None = None

    def reset(self) -> None:
        self._stable_frames = 0
        self._previous_relative_position = None

    def evaluate(self, data: mujoco.MjData) -> GraspEvidence:
        contacted: dict[int, list[np.ndarray]] = {}
        total_force = 0.0
        force = np.zeros(6, dtype=float)
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 == self.target_body_id and body2 in self.gripper_body_ids:
                finger_body, sign = body2, 1.0
            elif body2 == self.target_body_id and body1 in self.gripper_body_ids:
                finger_body, sign = body1, -1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, data, index, force)
            normal_force = max(0.0, float(force[0]))
            total_force += normal_force
            normal = sign * np.asarray(contact.frame[:3], dtype=float)
            contacted.setdefault(finger_body, []).append(normal)

        normals = [
            np.mean(values, axis=0)
            / max(np.linalg.norm(np.mean(values, axis=0)), 1e-12)
            for values in contacted.values()
        ]
        opposing = any(
            float(np.dot(normals[i], normals[j])) < -0.25
            for i in range(len(normals))
            for j in range(i + 1, len(normals))
        )
        relative_position = np.asarray(
            data.xpos[self.target_body_id], dtype=float
        ) - np.asarray(data.xpos[self.attachment_body_id], dtype=float)
        slip = (
            0.0
            if self._previous_relative_position is None
            else float(
                np.linalg.norm(relative_position - self._previous_relative_position)
            )
        )
        self._previous_relative_position = relative_position.copy()
        stable = (
            len(contacted) >= 2
            and opposing
            and total_force >= self.min_normal_force_n
            and slip <= self.max_relative_slip_m
        )
        self._stable_frames = self._stable_frames + 1 if stable else 0
        names = tuple(
            sorted(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body#{body_id}"
                for body_id in contacted
            )
        )
        return GraspEvidence(
            contact_bodies=names,
            normal_force_n=total_force,
            opposing_contacts=opposing,
            stable_frames=self._stable_frames,
            relative_slip_m=slip,
            grasped=self._stable_frames >= self.confirmation_frames,
        )

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"body '{name}' not found")
        return int(body_id)
