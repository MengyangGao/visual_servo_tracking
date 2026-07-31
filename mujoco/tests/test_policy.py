from __future__ import annotations

from dataclasses import replace

import numpy as np

from mujoco_servo.policy import (
    GraspPlanner,
    GraspPlanningContext,
    GripperCommand,
    PolicyCommand,
    PolicyObservation,
    PolicyPhase,
    ReactivePickPlacePolicy,
    ReactivePolicyConfig,
    SafetyLimits,
    SafetySupervisor,
)
from mujoco_servo.scene import WorldGraspPoint


def _point(
    name: str, position: tuple[float, float, float], width: float = 0.04
) -> WorldGraspPoint:
    return WorldGraspPoint(
        name,
        np.asarray(position, dtype=float),
        np.array([0.0, 0.0, -1.0]),
        width,
    )


def test_grasp_planner_rejects_width_reach_and_clearance() -> None:
    planner = GraspPlanner()
    context = GraspPlanningContext(
        ee_position=np.array([0.4, 0.0, 0.5]),
        camera_position=np.array([1.0, -1.0, 0.8]),
        support_z=0.2,
        approach_distance_m=0.08,
        max_reach_m=0.8,
        max_gripper_width_m=0.08,
    )
    candidates = planner.plan(
        (
            _point("valid", (0.5, 0.0, 0.25)),
            _point("too-wide", (0.5, 0.0, 0.25), 0.10),
            _point("too-far", (2.0, 0.0, 0.5)),
            _point("below-table", (0.5, 0.0, 0.19)),
        ),
        context,
    )
    assert [candidate.name for candidate in candidates] == ["valid"]
    assert 0.0 < candidates[0].score <= 1.0


def test_reactive_policy_completes_contact_verified_pick_place() -> None:
    policy = ReactivePickPlacePolicy(
        ReactivePolicyConfig(
            stage_tolerance_m=0.02,
            lift_distance_m=0.10,
            transfer_clearance_m=0.10,
            retreat_distance_m=0.08,
            verification_frames=3,
        ),
        np.array([0.5, -0.15, 0.25]),
    )
    grasp = GraspPlanner().select(
        (_point("top", (0.5, 0.1, 0.25)),),
        GraspPlanningContext(
            ee_position=np.array([0.4, 0.1, 0.5]),
            camera_position=np.array([1.0, -1.0, 0.8]),
            support_z=0.2,
            approach_distance_m=0.08,
        ),
    )
    target = np.array([0.5, 0.1, 0.25])
    policy.set_grasp(grasp, target)

    time_s = 0.0
    ee = np.array([0.4, 0.1, 0.5])
    command = None
    for _ in range(30):
        grasped = policy.phase in {
            PolicyPhase.CLOSE,
            PolicyPhase.VERIFY,
            PolicyPhase.LIFT,
            PolicyPhase.TRANSFER,
            PolicyPhase.PLACE,
        }
        placed = policy.phase in {
            PolicyPhase.RELEASE,
            PolicyPhase.RETREAT,
            PolicyPhase.SUCCEEDED,
        }
        command = policy.step(
            PolicyObservation(
                time_s=time_s,
                ee_position=ee,
                target_position=target,
                tracking_valid=True,
                grasped=grasped,
                contact_stable_frames=4 if grasped else 0,
                place_error_m=0.0 if placed else 0.25,
            )
        )
        ee = command.goal_position.copy()
        time_s += 0.1
        if policy.succeeded:
            break

    assert command is not None
    assert policy.phase is PolicyPhase.SUCCEEDED
    assert command.gripper is GripperCommand.OPEN
    assert policy.attempts == 0


def test_reactive_policy_recovers_and_exhausts_attempt_budget() -> None:
    policy = ReactivePickPlacePolicy(
        ReactivePolicyConfig(
            stage_tolerance_m=0.02,
            close_timeout_s=0.1,
            max_attempts=1,
        ),
        np.array([0.5, -0.1, 0.25]),
    )
    point = _point("top", (0.5, 0.0, 0.25))
    grasp = GraspPlanner().select(
        (point,),
        GraspPlanningContext(
            ee_position=np.array([0.5, 0.0, 0.4]),
            camera_position=np.array([1.0, -1.0, 0.8]),
            support_z=0.2,
            approach_distance_m=0.08,
        ),
    )
    target = np.array([0.5, 0.0, 0.25])
    policy.set_grasp(grasp, target)
    ee = grasp.pregrasp_position.copy()
    for time_s in (0.0, 0.01, 0.02, 0.2):
        command = policy.step(
            PolicyObservation(time_s, ee, target, True, grasped=False)
        )
        ee = command.goal_position.copy()
    command = policy.step(PolicyObservation(0.3, ee, target, True, grasped=False))
    assert policy.phase is PolicyPhase.FAILED
    assert command.hold
    assert policy.attempts == 1


def test_safety_supervisor_clamps_workspace_and_fails_closed() -> None:
    supervisor = SafetySupervisor(
        SafetyLimits(
            workspace_min=(0.0, -1.0, 0.1),
            workspace_max=(1.0, 1.0, 1.0),
            max_goal_distance_m=0.2,
            max_normal_force_n=10.0,
        )
    )
    observation = PolicyObservation(
        0.0,
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, 0.0, 0.2]),
        True,
        False,
    )
    command = supervisor.supervise(
        PolicyCommand(
            PolicyPhase.TRANSFER, np.array([2.0, 0.0, 2.0]), GripperCommand.CLOSE
        ),
        observation,
    )
    assert np.linalg.norm(command.goal_position - observation.ee_position) <= 0.2 + 1e-9

    overforce = supervisor.supervise(command, replace(observation, normal_force_n=11.0))
    assert overforce.phase is PolicyPhase.RECOVER
    assert overforce.gripper is GripperCommand.OPEN
    assert overforce.hold
