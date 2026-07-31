from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum

import cv2
import mujoco
import numpy as np

from .config import CameraConfig, DemoConfig, resolve_config
from .core.types import FeatureObservation, ServoMode
from .clock import ControlTick, PhaseAccumulatorClock
from .control import (
    ResolvedRateController,
    ServoState,
    desired_ee_orientation,
    desired_ee_position,
)
from .depth import DepthBackend, build_depth_backend
from .math_utils import tool_z_facing_rotation, vector_alignment_error
from .manipulation import ContactGraspEvaluator, GraspEvidence
from .perception import (
    CameraIntrinsics,
    CameraObservation,
    Detection,
    PerceptionBackend,
    build_perception,
)
from .policy import (
    CartesianPathValidator,
    GraspPlanner,
    GraspPlanningContext,
    GripperCommand,
    PolicyObservation,
    PolicyPhase,
    ReactivePickPlacePolicy,
    ReactivePolicyConfig,
    SafetyLimits,
    SafetySupervisor,
    WorkSurface,
)
from .scene import (
    WorldGraspPoint,
    activate_grasp as activate_scene_grasp,
    body_position,
    build_scene,
    deactivate_grasp as deactivate_scene_grasp,
    frame_position,
    grasp_point_world,
    set_target_position,
    site_position,
)
from .servo import VisualServoObjective
from .targets import TargetMotion, base_position
from .viz import DashboardRenderer, DashboardTelemetry, VideoRecorder
from .vision import align_rotation_to_reference, estimate_pose_6d


class TrackingState(str, Enum):
    TRACKING = "TRACKING"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


class ManipulationState(str, Enum):
    IDLE = "IDLE"
    PREGRASP = "PREGRASP"
    APPROACHING = "APPROACHING"
    CLOSING = "CLOSING"
    LIFTING = "LIFTING"
    TRANSFERRING = "TRANSFERRING"
    PLACING = "PLACING"
    RELEASING = "RELEASING"
    RETREATING = "RETREATING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(slots=True)
class RunSummary:
    steps: int
    robot: str
    task: str
    target: str
    trajectory: str
    detector: str
    final_error_m: float
    final_target_distance_m: float
    mean_error_m: float
    min_error_m: float
    max_error_m: float
    perception_updates: int
    rejected_detections: int
    hold_steps: int
    oracle_truth_steps: int
    truth_fallback_steps: int
    final_perception_age_s: float | None
    mean_camera_render_ms: float
    depth_backend: str
    depth_metric: bool
    final_target_position: tuple[float, float, float]
    final_end_effector_position: tuple[float, float, float]
    final_detected_position: tuple[float, float, float] | None
    final_detection_anchor: str | None
    final_orientation_error_rad: float
    actuator_mode: str = "position"
    rms_error_m: float = 0.0
    p95_error_m: float = 0.0
    settling_time_s: float | None = None
    tracking_state: str = TrackingState.LOST.value
    lost_events: int = 0
    reacquire_events: int = 0
    lost_duration_s: float = 0.0
    saturated_steps: int = 0
    saturation_ratio: float = 0.0
    requested_control_hz: float = 0.0
    effective_control_hz: float = 0.0
    simulated_duration_s: float = 0.0
    wall_duration_s: float = 0.0
    mean_perception_latency_ms: float = 0.0
    p95_perception_latency_ms: float = 0.0
    dropped_camera_frames: int = 0
    depth_confidence: float = 0.0
    depth_valid_fraction: float = 0.0
    depth_inference_ms: float = 0.0
    manipulation_state: str = ManipulationState.IDLE.value
    contact_steps: int = 0
    grasped: bool = False
    target_lift_m: float = 0.0
    steady_state_rms_error_m: float = 0.0
    steady_state_p95_error_m: float = 0.0
    perception_device: str = "none"
    depth_device: str = "none"
    servo_mode: str = "pbvs"
    final_image_error_px: float = 0.0
    grasp_normal_force_n: float = 0.0
    grasp_relative_slip_m: float = 0.0
    policy_name: str = "none"
    policy_phase: str | None = None
    policy_attempts: int = 0
    selected_grasp: str | None = None
    place_position: tuple[float, float, float] | None = None
    place_error_m: float | None = None
    task_succeeded: bool = False
    failure_reason: str | None = None
    grasp_pose_source: str = "descriptor"
    grasp_pose_quality: float = 0.0
    rejected_grasps: tuple[str, ...] = ()
    termination_reason: str = "step_budget"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SimulationState:
    """Public snapshot of positions read from the current MuJoCo state."""

    time_s: float
    target_position: np.ndarray
    end_effector_position: np.ndarray
    camera_position: np.ndarray
    joint_positions: np.ndarray
    detected_position: np.ndarray | None
    detection_backend: str | None
    detection_anchor: str | None
    detection_age_s: float | None
    tracking_state: str = TrackingState.LOST.value
    detection_covariance: np.ndarray | None = None
    detection_capture_time_s: float | None = None
    detection_available_time_s: float | None = None
    detection_quality: float | None = None
    detection_valid_fraction: float | None = None
    manipulation_state: str = ManipulationState.IDLE.value
    in_contact: bool = False
    grasped: bool = False
    grasp_point_position: np.ndarray | None = None
    target_lift_m: float = 0.0
    grasp_normal_force_n: float = 0.0
    grasp_relative_slip_m: float = 0.0
    policy_attempts: int = 0
    policy_phase: str | None = None
    place_error_m: float | None = None
    grasp_pose_source: str = "descriptor"
    grasp_pose_quality: float = 0.0

    def as_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "target_position": self.target_position.tolist(),
            "end_effector_position": self.end_effector_position.tolist(),
            "camera_position": self.camera_position.tolist(),
            "joint_positions": self.joint_positions.tolist(),
            "detected_position": None
            if self.detected_position is None
            else self.detected_position.tolist(),
            "detection_backend": self.detection_backend,
            "detection_anchor": self.detection_anchor,
            "detection_age_s": self.detection_age_s,
            "tracking_state": self.tracking_state,
            "detection_covariance": (
                None
                if self.detection_covariance is None
                else self.detection_covariance.tolist()
            ),
            "detection_capture_time_s": self.detection_capture_time_s,
            "detection_available_time_s": self.detection_available_time_s,
            "detection_quality": self.detection_quality,
            "detection_valid_fraction": self.detection_valid_fraction,
            "manipulation_state": self.manipulation_state,
            "in_contact": self.in_contact,
            "grasped": self.grasped,
            "grasp_point_position": (
                None
                if self.grasp_point_position is None
                else self.grasp_point_position.tolist()
            ),
            "target_lift_m": self.target_lift_m,
            "grasp_normal_force_n": self.grasp_normal_force_n,
            "grasp_relative_slip_m": self.grasp_relative_slip_m,
            "policy_attempts": self.policy_attempts,
            "policy_phase": self.policy_phase,
            "place_error_m": self.place_error_m,
            "grasp_pose_source": self.grasp_pose_source,
            "grasp_pose_quality": self.grasp_pose_quality,
        }


@dataclass(slots=True)
class _StepOutcome:
    servo_state: ServoState
    error_m: float
    hold: bool
    oracle_truth: bool
    perception_updates: int
    rejected_detections: int


class VisualServoSimulation:
    def __init__(self, config: DemoConfig) -> None:
        resolved = resolve_config(config)
        self.config = config
        self.extra_robots = resolved.extra_robots
        self.robot = resolved.robot
        self.extra_targets = resolved.extra_targets
        # Touch/grasp are physical tasks by definition.  Built-in targets stay
        # lightweight visual mocap bodies for tracking demos, and are promoted
        # to free bodies automatically when a manipulation task is selected.
        self.target = (
            replace(resolved.target, dynamics="physical")
            if config.controller.task.strip().lower()
            in {"touch", "grasp", "pick-place"}
            else resolved.target
        )
        self._target_base_position = (
            base_position(self.target)
            if self.target.base_position is not None
            or self.robot.default_target_position is None
            else np.array(self.robot.default_target_position, dtype=float).reshape(3)
        )
        if self.target.dynamics == "physical":
            # Spawn manipulation objects already supported by the work surface.
            # Dropping them from a tracking-demo pose injects needless impact
            # energy and makes grasp evaluation measure a transient accident.
            support_z = 0.215 if config.environment.add_table else 0.0
            self._target_base_position[2] = (
                support_z + 0.5 * self.target.size[2] + 0.002
            )
        self.camera = self._workspace_camera(
            config.camera, self._target_base_position, self.robot.base_position
        )
        self.scene = build_scene(
            self.target,
            self.camera,
            self.robot,
            target_position=self._target_base_position,
            actuator_mode=config.controller.actuator_mode,
            environment=config.environment,
        )
        self.motion = TargetMotion(
            self.target,
            config.trajectory,
            config.seed,
            base_override=self._target_base_position,
        )
        self.detector_name = config.detector.strip().lower()
        self.perception: PerceptionBackend | None = None
        if not self._should_lazy_load_perception():
            self.perception = build_perception(config.detector)
        self.controller = ResolvedRateController(
            self.scene.model,
            self.scene.ee_frame_name,
            self.scene.ee_frame_type,
            self.scene.ee_frame_offset,
            self.scene.robot,
            config.controller,
        )
        self.controller.reset(self.scene.data)
        self.visual_objective = VisualServoObjective(
            mode=config.controller.servo_mode,
            gain=config.controller.position_gain,
            max_speed_mps=config.controller.max_ee_speed,
            hybrid_switch_m=max(0.05, 1.5 * config.controller.standoff_m),
        )
        self._clock = PhaseAccumulatorClock(
            float(self.scene.model.opt.timestep),
            float(config.controller.control_hz),
        )
        self.depth_backend: DepthBackend = build_depth_backend(config.depth)
        self._renderer = None
        self._dashboard = DashboardRenderer()
        self._recorder = (
            None
            if config.record_path is None
            else VideoRecorder(
                config.record_path, fps=config.camera_fps, frame_size=(960, 540)
            )
        )
        self._last_servo_state: ServoState | None = None
        self._last_camera_observation: CameraObservation | None = None
        self._last_accepted_camera_observation: CameraObservation | None = None
        self._detection_observations: dict[int, CameraObservation] = {}
        self._manual_target_offset = np.zeros(3, dtype=float)
        self._manual_target_velocity = np.zeros(3, dtype=float)
        self._manual_velocity_until = 0.0
        self._last_target_update_time: float | None = None
        self._viewer_camera_initialized = False
        self._latest_overlay_bgr: np.ndarray | None = None
        self._latest_overlay_rgb: np.ndarray | None = None
        self._overlay_rect_key: tuple[int, int, int, int] | None = None
        self._last_camera_wall_time = -1.0e9
        self._last_camera_sim_time = -1.0e9
        self._next_camera_sim_time = float(self.scene.data.time)
        self._perception_executor: ThreadPoolExecutor | None = None
        self._perception_future: Future[tuple[CameraObservation, Detection]] | None = (
            None
        )
        self._last_detection: Detection | None = None
        self._last_detection_wall_time: float | None = None
        self._last_detection_sim_time: float | None = None
        self._last_detection_available_wall_time: float | None = None
        self._last_detection_available_sim_time: float | None = None
        self._last_accepted_detection_position: np.ndarray | None = None
        self._last_accepted_detection: Detection | None = None
        self._last_accepted_detection_wall_time: float | None = None
        self._last_accepted_detection_sim_time: float | None = None
        self._last_accepted_detection_available_sim_time: float | None = None
        self._perception_disabled = False
        self._camera_render_times_ms: list[float] = []
        self._last_depth_metric = False
        self._last_depth_confidence = 0.0
        self._last_depth_valid_fraction = 0.0
        self._last_depth_inference_s = 0.0
        self._perception_latencies_s: list[float] = []
        self._pending_detection_deliveries: list[
            tuple[float, Detection, float, float, float]
        ] = []
        self._rng = np.random.default_rng(config.seed)
        self._dropped_camera_frames = 0
        self._step_index = 0
        self._tracking_state = (
            TrackingState.TRACKING
            if self.detector_name == "oracle"
            else TrackingState.LOST
        )
        self._ever_tracked = self.detector_name == "oracle"
        self._last_observed_target: np.ndarray | None = None
        self._last_observed_time_s: float | None = None
        self._reacquire_candidate: np.ndarray | None = None
        self._reacquire_count = 0
        self._reacquire_last_time_s: float | None = None
        self._lost_since_sim_time: float | None = (
            float(self.scene.data.time)
            if self._tracking_state is TrackingState.LOST
            else None
        )
        self._lost_events = 0
        self._reacquire_events = 0
        self._completed_lost_duration_s = 0.0
        task = config.controller.task.strip().lower()
        manipulation_task = task in {"touch", "grasp", "pick-place"}
        self._manipulation_state = (
            ManipulationState.PREGRASP if manipulation_task else ManipulationState.IDLE
        )
        self._grasped = False
        self._grasp_evidence: GraspEvidence | None = None
        self._peak_grasp_normal_force_n = 0.0
        self._max_grasp_relative_slip_m = 0.0
        self._grasp_lost_frames = 0
        self._closing_frames = 0
        self._grasp_evaluator: ContactGraspEvaluator | None = None
        if (
            task in {"grasp", "pick-place"}
            and self.robot.grasp_attachment_body is not None
            and len(self.robot.gripper_contact_bodies) >= 2
        ):
            self._grasp_evaluator = ContactGraspEvaluator(
                self.scene.model,
                target_body_name=self.scene.target_body_name,
                gripper_body_names=self.robot.gripper_contact_bodies,
                attachment_body_name=self.robot.grasp_attachment_body,
                min_normal_force_n=config.controller.grasp_min_normal_force_n,
                max_relative_slip_m=config.controller.grasp_max_relative_slip_m,
                confirmation_frames=config.controller.grasp_confirmation_frames,
            )
        self._contact_steps = 0
        self._grasp_initial_target_z: float | None = None
        self._max_target_lift_m = 0.0
        self._lift_goal_position: np.ndarray | None = None
        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(
            rotation_flat, np.asarray(self.target.quat, dtype=float).reshape(4)
        )
        self._last_target_rotation = rotation_flat.reshape(3, 3).copy()
        self._grasp_pose_source = "descriptor"
        self._grasp_pose_quality = 0.0
        self._work_surface = (
            WorkSurface(
                self.scene.work_surface_center_xy,
                self.scene.work_surface_half_size_xy,
                self.scene.support_z,
            )
            if self.scene.work_surface_center_xy is not None
            and self.scene.work_surface_half_size_xy is not None
            else None
        )
        bounds = self.robot.detection_bounds or (
            (-1.25, -1.25, 0.0),
            (1.25, 1.25, 1.90),
        )
        workspace_min = np.asarray(bounds[0], dtype=float).copy()
        workspace_max = np.asarray(bounds[1], dtype=float).copy()
        workspace_min[2] = max(workspace_min[2], self.scene.support_z + 0.005)
        self._path_validator = CartesianPathValidator(
            workspace_min,
            workspace_max,
            support_z=self.scene.support_z,
        )
        self._place_position = self._resolve_place_position()
        self._grasp_planner = GraspPlanner()
        self._pick_place_policy: ReactivePickPlacePolicy | None = None
        self._safety_supervisor: SafetySupervisor | None = None
        self._gripper_commanded_closed = False
        if task == "pick-place":
            self._pick_place_policy = ReactivePickPlacePolicy(
                ReactivePolicyConfig(
                    stage_tolerance_m=config.controller.grasp_stage_tolerance_m,
                    lift_distance_m=config.controller.grasp_lift_m,
                    max_attempts=config.controller.policy_max_attempts,
                    close_timeout_s=config.controller.policy_close_timeout_s,
                    motion_timeout_s=config.controller.policy_motion_timeout_s,
                    place_tolerance_m=config.controller.policy_place_tolerance_m,
                ),
                self._place_position,
            )
            self._safety_supervisor = SafetySupervisor(
                SafetyLimits(
                    workspace_min=tuple(float(value) for value in workspace_min),
                    workspace_max=tuple(float(value) for value in workspace_max),
                    max_normal_force_n=config.controller.policy_max_normal_force_n,
                ),
                path_is_valid=self._cartesian_path_is_valid,
            )
        self._initial_data = self._snapshot_data()
        if self._tracking_state is TrackingState.LOST:
            self.controller.begin_hold(self.scene.data)

    @staticmethod
    def _workspace_camera(
        camera: CameraConfig,
        target_position: np.ndarray,
        robot_base_position: np.ndarray | tuple[float, float, float],
    ) -> CameraConfig:
        """Frame the selected robot workspace unless a custom pose was supplied."""
        defaults = CameraConfig()
        if (
            getattr(camera, "mount_body", None) is not None
            and camera.position == defaults.position
            and camera.lookat == defaults.lookat
        ):
            return replace(
                camera,
                position=(0.0, 0.0, 0.065),
                lookat=(0.0, 0.0, 0.35),
            )
        if camera.position != defaults.position or camera.lookat != defaults.lookat:
            return camera
        target = np.asarray(target_position, dtype=float).reshape(3)
        # Look across (rather than along) the base-to-target approach line so
        # the arm does not hide the target as it reaches the standoff pose.
        robot_base = np.asarray(robot_base_position, dtype=float).reshape(3)
        radial = target[:2] - robot_base[:2]
        radial_norm = float(np.linalg.norm(radial))
        radial = radial / radial_norm if radial_norm > 1e-9 else np.array([1.0, 0.0])
        side = np.array([radial[1], -radial[0]], dtype=float)
        if side[1] > 0.0:
            side = -side
        position = target + np.array([1.2 * side[0], 1.2 * side[1], 0.7], dtype=float)
        return replace(camera, position=tuple(position), lookat=tuple(target))

    def get_state(self) -> SimulationState:
        """Read a copy of the current simulator and perception state."""
        model = self.scene.model
        data = self.scene.data
        mujoco.mj_forward(model, data)
        target = site_position(model, data, self.scene.target_site_name)
        ee = frame_position(
            model,
            data,
            self.scene.ee_frame_type,
            self.scene.ee_frame_name,
            self.scene.ee_frame_offset,
        )
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.scene.camera_name
        )
        if camera_id < 0:
            raise KeyError(f"camera '{self.scene.camera_name}' missing")
        joint_positions = []
        for joint_name in self.robot.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise KeyError(f"joint '{joint_name}' missing")
            joint_positions.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
        detection = self._last_accepted_detection
        detection_age = self._perception_age_s()
        if (
            detection_age is not None
            and detection_age > self.config.detection_timeout_s
        ):
            detection = None
        detected = None
        if (
            detection is not None
            and detection.success
            and detection.target_position is not None
        ):
            try:
                candidate = np.asarray(detection.target_position, dtype=float).reshape(
                    3
                )
                if np.isfinite(candidate).all():
                    detected = candidate.copy()
            except (TypeError, ValueError):
                detected = None
        covariance = self._detection_covariance(detection)
        world_grasp = self._safe_grasp_point()
        return SimulationState(
            time_s=float(data.time),
            target_position=target,
            end_effector_position=ee,
            camera_position=np.array(data.cam_xpos[camera_id], dtype=float),
            joint_positions=np.asarray(joint_positions, dtype=float),
            detected_position=detected,
            detection_backend=None if detection is None else detection.backend,
            detection_anchor=None if detection is None else detection.anchor_type,
            detection_age_s=detection_age,
            tracking_state=self._tracking_state.value,
            detection_covariance=covariance,
            detection_capture_time_s=self._last_accepted_detection_sim_time,
            detection_available_time_s=self._last_accepted_detection_available_sim_time,
            detection_quality=self._optional_detection_scalar(detection, "quality"),
            detection_valid_fraction=self._optional_detection_scalar(
                detection, "valid_fraction"
            ),
            manipulation_state=self._manipulation_state.value,
            in_contact=self._target_robot_contact(),
            grasped=self._grasped,
            grasp_point_position=None
            if world_grasp is None
            else world_grasp.position.copy(),
            target_lift_m=max(self._target_lift(target), self._max_target_lift_m),
            grasp_normal_force_n=(
                0.0
                if self._grasp_evidence is None
                else self._grasp_evidence.normal_force_n
            ),
            grasp_relative_slip_m=(
                0.0
                if self._grasp_evidence is None
                else self._grasp_evidence.relative_slip_m
            ),
            policy_attempts=(
                0
                if self._pick_place_policy is None
                else self._pick_place_policy.attempts
            ),
            policy_phase=(
                None
                if self._pick_place_policy is None
                else self._pick_place_policy.phase.value
            ),
            place_error_m=self._place_error(target),
            grasp_pose_source=self._grasp_pose_source,
            grasp_pose_quality=self._grasp_pose_quality,
        )

    @property
    def tracking_state(self) -> TrackingState:
        return self._tracking_state

    @property
    def manipulation_state(self) -> ManipulationState:
        return self._manipulation_state

    def observe(self) -> SimulationState:
        """Gym-style observation alias for :meth:`get_state`."""
        return self.get_state()

    def reset(self) -> SimulationState:
        """Restore the initial MuJoCo and runtime state for a repeatable episode."""
        self.close()
        mujoco.mj_resetData(self.scene.model, self.scene.data)
        data = self.scene.data
        for name, value in self._initial_data.items():
            destination = getattr(data, name)
            if np.isscalar(destination):
                setattr(data, name, float(value))
            else:
                destination[...] = value
        mujoco.mj_forward(self.scene.model, data)
        self.controller.reset(data)
        self._clock.reset()
        self._step_index = 0
        self._manual_target_offset.fill(0.0)
        self._manual_target_velocity.fill(0.0)
        self._manual_velocity_until = 0.0
        self._last_target_update_time = None
        self._viewer_camera_initialized = False
        self._latest_overlay_bgr = None
        self._latest_overlay_rgb = None
        self._overlay_rect_key = None
        self._last_camera_wall_time = -1.0e9
        self._last_camera_sim_time = -1.0e9
        self._next_camera_sim_time = float(data.time)
        self._last_detection = None
        self._last_camera_observation = None
        self._last_accepted_camera_observation = None
        self._detection_observations.clear()
        self._last_detection_wall_time = None
        self._last_detection_sim_time = None
        self._last_detection_available_wall_time = None
        self._last_detection_available_sim_time = None
        self._last_accepted_detection_position = None
        self._last_accepted_detection = None
        self._last_accepted_detection_wall_time = None
        self._last_accepted_detection_sim_time = None
        self._last_accepted_detection_available_sim_time = None
        self._perception_disabled = False
        self._camera_render_times_ms.clear()
        self._perception_latencies_s.clear()
        self._last_depth_metric = False
        self._last_depth_confidence = 0.0
        self._last_depth_valid_fraction = 0.0
        self._last_depth_inference_s = 0.0
        self._pending_detection_deliveries.clear()
        self._dropped_camera_frames = 0
        self._rng = np.random.default_rng(self.config.seed)
        self._tracking_state = (
            TrackingState.TRACKING
            if self.detector_name == "oracle"
            else TrackingState.LOST
        )
        self._ever_tracked = self.detector_name == "oracle"
        self._last_observed_target = None
        self._last_observed_time_s = None
        self._reacquire_candidate = None
        self._reacquire_count = 0
        self._reacquire_last_time_s = None
        self._lost_since_sim_time = (
            float(data.time) if self._tracking_state is TrackingState.LOST else None
        )
        self._lost_events = 0
        self._reacquire_events = 0
        self._completed_lost_duration_s = 0.0
        self._manipulation_state = (
            ManipulationState.PREGRASP
            if self.config.controller.task.strip().lower()
            in {"touch", "grasp", "pick-place"}
            else ManipulationState.IDLE
        )
        self._grasped = False
        self._grasp_evidence = None
        self._peak_grasp_normal_force_n = 0.0
        self._max_grasp_relative_slip_m = 0.0
        self._grasp_lost_frames = 0
        self._closing_frames = 0
        if self._grasp_evaluator is not None:
            self._grasp_evaluator.reset()
        self._contact_steps = 0
        self._grasp_initial_target_z = None
        self._max_target_lift_m = 0.0
        self._lift_goal_position = None
        self._gripper_commanded_closed = False
        if self._pick_place_policy is not None:
            self._pick_place_policy.reset(float(data.time))
        rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(
            rotation_flat, np.asarray(self.target.quat, dtype=float).reshape(4)
        )
        self._last_target_rotation = rotation_flat.reshape(3, 3).copy()
        self._grasp_pose_source = "descriptor"
        self._grasp_pose_quality = 0.0
        deactivate_scene_grasp(self.scene)
        reset_backend = getattr(self.perception, "reset", None)
        if callable(reset_backend):
            reset_backend()
        if self._tracking_state is TrackingState.LOST:
            self.controller.begin_hold(data)
        return self.get_state()

    def step(self) -> SimulationState:
        """Advance one controller interval without opening a viewer."""
        tick = self._clock.next_tick()
        self._execute_control_step(None, tick, self._step_index)
        self._step_index += 1
        return self.get_state()

    def close(self) -> None:
        """Release renderer and worker resources; safe to call repeatedly."""
        if self._perception_executor is not None:
            self._perception_executor.shutdown(wait=True, cancel_futures=True)
            self._perception_executor = None
        self._perception_future = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def __enter__(self) -> "VisualServoSimulation":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _snapshot_data(self) -> dict[str, np.ndarray | float]:
        data = self.scene.data
        names = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata")
        snapshot: dict[str, np.ndarray | float] = {
            name: np.array(getattr(data, name), copy=True) for name in names
        }
        snapshot["time"] = float(data.time)
        return snapshot

    def get_body_position(self, name: str) -> np.ndarray:
        """Return a copy of a named MuJoCo body's world position."""
        mujoco.mj_forward(self.scene.model, self.scene.data)
        return body_position(self.scene.model, self.scene.data, name)

    def get_site_position(self, name: str) -> np.ndarray:
        """Return a copy of a named MuJoCo site's world position."""
        mujoco.mj_forward(self.scene.model, self.scene.data)
        return site_position(self.scene.model, self.scene.data, name)

    def get_grasp_point(self, name: str | None = None) -> WorldGraspPoint:
        """Resolve a target-local executable grasp point in world coordinates."""
        mujoco.mj_forward(self.scene.model, self.scene.data)
        selected = self.config.controller.grasp_point if name is None else name
        return grasp_point_world(self.scene, selected)

    def activate_grasp(
        self, name: str | None = None, *, max_distance_m: float | None = None
    ) -> WorldGraspPoint:
        """Close a nearby gripper; contact evidence, not a weld, confirms grasping."""
        selected = self.config.controller.grasp_point if name is None else name
        distance = (
            self.config.controller.grasp_attach_distance_m
            if max_distance_m is None
            else float(max_distance_m)
        )
        point = activate_scene_grasp(self.scene, selected, max_distance_m=distance)
        self.controller.set_gripper_closed(True)
        self._gripper_commanded_closed = True
        self._grasped = False
        target = site_position(
            self.scene.model, self.scene.data, self.scene.target_site_name
        )
        self._grasp_initial_target_z = float(target[2])
        return point

    def release_grasp(self) -> None:
        """Release the target and restore the configured open-gripper control."""
        target = site_position(
            self.scene.model, self.scene.data, self.scene.target_site_name
        )
        self._max_target_lift_m = max(
            self._max_target_lift_m, self._target_lift(target)
        )
        deactivate_scene_grasp(self.scene)
        self.controller.set_gripper_closed(False)
        self._gripper_commanded_closed = False
        self._grasped = False
        self._grasp_evidence = None
        self._grasp_lost_frames = 0
        if self._grasp_evaluator is not None:
            self._grasp_evaluator.reset()
        self._lift_goal_position = None
        self._grasp_initial_target_z = None

    def run(self) -> RunSummary:
        viewer = None
        if self.config.viewer and not self.config.headless:
            viewer = self._try_open_viewer()
        try:
            self._prepare_perception_for_run(viewer)
            if self._uses_async_perception(viewer):
                self._perception_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mujoco-servo-perception"
                )
            summary = self._run_loop(viewer)
        finally:
            if viewer is not None:
                viewer.close()
                self._viewer_camera_initialized = False
            self.close()
        return summary

    def _try_open_viewer(self):
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                self.scene.model, self.scene.data, key_callback=self._handle_key
            )
            self._initialize_viewer(viewer)
            return viewer
        except Exception as exc:
            if sys.platform == "darwin":
                print(
                    f"viewer unavailable ({exc}); retry with `mjpython scripts/demo.py` for the native macOS viewer"
                )
            else:
                print(f"viewer unavailable ({exc}); continuing headless")
            return None

    def _render_camera_observation(
        self, resolve_learned_depth: bool = True
    ) -> CameraObservation | None:
        if self.detector_name == "oracle":
            return None
        render_start = time.perf_counter()
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(
                    self.scene.model,
                    width=self.camera.width,
                    height=self.camera.height,
                )
            except Exception as exc:
                raise RuntimeError(
                    "MuJoCo camera rendering is unavailable. Use a desktop graphics session on macOS "
                    "or configure EGL/OSMesa for headless Linux; oracle mode does not require rendering."
                ) from exc
        self._renderer.update_scene(self.scene.data, camera=self.scene.camera_name)
        rgb = self._renderer.render()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.scene.data, camera=self.scene.camera_name)
        rendered_depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        frame_bgr = rgb[:, :, ::-1].copy()
        rgb_noise_std = float(getattr(self.camera, "rgb_noise_std", 0.0))
        if rgb_noise_std > 0.0:
            noisy = frame_bgr.astype(np.float32) + self._rng.normal(
                0.0, rgb_noise_std, frame_bgr.shape
            )
            frame_bgr = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
        depth_result = None
        if resolve_learned_depth or self.depth_backend.name in {"mujoco", "none"}:
            depth = self.depth_backend.estimate(frame_bgr, rendered_depth)
            depth_result = depth
            depth_m = depth.depth_m
            depth_backend = depth.backend
            depth_metric = depth.metric
        else:
            depth_m = np.asarray(rendered_depth, dtype=np.float32).copy()
            depth_backend = "mujoco-hint"
            depth_metric = True
        dropout_probability = float(getattr(self.camera, "dropout_probability", 0.0))
        if depth_metric:
            depth_m = np.asarray(depth_m, dtype=np.float32).copy()
            depth_noise_std = float(getattr(self.camera, "depth_noise_std", 0.0))
            valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
            if depth_noise_std > 0.0 and np.any(valid_depth):
                depth_m[valid_depth] += self._rng.normal(
                    0.0, depth_noise_std, int(valid_depth.sum())
                )
                depth_m[depth_m <= 0.0] = np.nan
            if dropout_probability > 0.0:
                dropout = self._rng.random(depth_m.shape) < dropout_probability
                depth_m[dropout] = np.nan
                frame_bgr[dropout] = 0
        self._last_depth_metric = depth_metric
        measured_valid_fraction = float(
            np.mean(np.isfinite(depth_m) & (np.asarray(depth_m) > 0.0))
        )
        self._last_depth_confidence = float(
            getattr(depth_result, "confidence", 1.0 if depth_metric else 0.0)
        )
        backend_valid_fraction = float(
            getattr(depth_result, "valid_fraction", measured_valid_fraction)
        )
        self._last_depth_valid_fraction = min(
            backend_valid_fraction, measured_valid_fraction
        )
        self._last_depth_inference_s = float(
            getattr(depth_result, "inference_time_s", 0.0)
        )
        cam_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_CAMERA, self.scene.camera_name
        )
        if cam_id < 0:
            raise RuntimeError(
                f"camera '{self.scene.camera_name}' is missing from the MuJoCo model"
            )
        fovy = float(self.scene.model.cam_fovy[cam_id])
        fy = 0.5 * self.camera.height / np.tan(np.deg2rad(fovy) * 0.5)
        intrinsics = CameraIntrinsics(
            fx=fy,
            fy=fy,
            cx=0.5 * (self.camera.width - 1),
            cy=0.5 * (self.camera.height - 1),
            width=self.camera.width,
            height=self.camera.height,
        )
        observation = CameraObservation(
            frame_bgr=frame_bgr,
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_position=np.array(self.scene.data.cam_xpos[cam_id], dtype=float),
            camera_xmat=np.array(self.scene.data.cam_xmat[cam_id], dtype=float).reshape(
                3, 3
            ),
            wall_time_s=render_start,
            sim_time_s=float(self.scene.data.time),
            depth_backend=depth_backend,
            depth_metric=depth_metric,
        )
        self._camera_render_times_ms.append(
            (time.perf_counter() - render_start) * 1000.0
        )
        return observation

    def _run_loop(self, viewer) -> RunSummary:
        errors: list[float] = []
        error_times: list[float] = []
        hold_steps = 0
        oracle_truth_steps = 0
        truth_fallback_steps = 0
        perception_updates = 0
        rejected_detections = 0
        saturated_steps = 0
        last_state: ServoState | None = None
        data = self.scene.data
        sim_start = float(data.time)
        wall_start = time.perf_counter()
        lost_events_start = self._lost_events
        reacquire_events_start = self._reacquire_events
        lost_duration_start = self._total_lost_duration(sim_start)
        dropped_start = self._dropped_camera_frames
        latency_start = len(self._perception_latencies_s)
        contact_steps_start = self._contact_steps
        termination_reason = "step_budget"
        terminal_steps = 0
        for _ in range(self.config.steps):
            if viewer is not None and not viewer.is_running():
                termination_reason = "viewer_closed"
                break
            tick = self._clock.next_tick()
            outcome = self._execute_control_step(viewer, tick, self._step_index)
            self._step_index += 1
            last_state = outcome.servo_state
            errors.append(outcome.error_m)
            error_times.append(float(last_state.time_s) - sim_start)
            hold_steps += int(outcome.hold)
            oracle_truth_steps += int(outcome.oracle_truth)
            perception_updates += outcome.perception_updates
            rejected_detections += outcome.rejected_detections
            saturated_steps += int(last_state.saturated_joints > 0)

            if viewer is not None:
                self._keep_viewer_camera_free(viewer)
                self._update_viewer_overlay(viewer)
                viewer.sync()
            if self.config.realtime:
                self._sleep_to_simulation_time(sim_start, wall_start)
            policy = self._pick_place_policy
            if (
                self.config.stop_on_terminal
                and policy is not None
                and (policy.succeeded or policy.failed)
            ):
                terminal_steps += 1
                if terminal_steps >= self.config.terminal_settle_steps:
                    termination_reason = (
                        "policy_succeeded" if policy.succeeded else "policy_failed"
                    )
                    break
            else:
                terminal_steps = 0

        completed_steps = len(errors)
        metric_errors = errors
        wall_duration = max(0.0, time.perf_counter() - wall_start)
        simulated_duration = max(0.0, float(data.time) - sim_start)
        final_snapshot = self.get_state()
        final_target_distance = float(
            np.linalg.norm(
                final_snapshot.target_position - final_snapshot.end_effector_position
            )
        )
        if last_state is not None and self.config.controller.task.strip().lower() in {
            "touch",
            "grasp",
            "pick-place",
        }:
            final_desired = last_state.desired_position
        else:
            final_desired = desired_ee_position(
                self.config.controller.task,
                final_snapshot.target_position,
                final_snapshot.end_effector_position,
                self.config.controller,
                self.robot.base_position,
            )
        final_error = float(
            np.linalg.norm(final_desired - final_snapshot.end_effector_position)
        )
        if last_state is None:
            metric_errors = [final_error]
        metric_array = np.asarray(metric_errors, dtype=float)
        steady_start = len(metric_array) // 2
        steady_array = metric_array[steady_start:]
        settling_time = self._settling_time(errors, error_times)
        run_latencies = self._perception_latencies_s[latency_start:]
        detected_position = final_snapshot.detected_position
        final_orientation_error = self._orientation_error(
            final_snapshot.target_position,
            final_desired,
        )
        return RunSummary(
            steps=completed_steps,
            robot=self.robot.name,
            task=self.config.controller.task,
            target=self.target.name,
            trajectory=self.config.trajectory,
            detector=self.detector_name,
            final_error_m=float(final_error),
            final_target_distance_m=final_target_distance,
            mean_error_m=float(np.mean(metric_errors)),
            min_error_m=float(np.min(metric_errors)),
            max_error_m=float(np.max(metric_errors)),
            perception_updates=perception_updates,
            rejected_detections=rejected_detections,
            hold_steps=hold_steps,
            oracle_truth_steps=oracle_truth_steps,
            truth_fallback_steps=truth_fallback_steps,
            final_perception_age_s=self._perception_age_s(),
            mean_camera_render_ms=float(np.mean(self._camera_render_times_ms))
            if self._camera_render_times_ms
            else 0.0,
            depth_backend=self.depth_backend.name,
            depth_metric=self._last_depth_metric,
            final_target_position=tuple(
                float(value) for value in final_snapshot.target_position
            ),
            final_end_effector_position=tuple(
                float(value) for value in final_snapshot.end_effector_position
            ),
            final_detected_position=None
            if detected_position is None
            else tuple(float(value) for value in detected_position),
            final_detection_anchor=final_snapshot.detection_anchor,
            final_orientation_error_rad=final_orientation_error,
            actuator_mode=self.controller.actuator_mode,
            rms_error_m=float(np.sqrt(np.mean(metric_array * metric_array))),
            p95_error_m=float(np.percentile(metric_array, 95.0)),
            settling_time_s=settling_time,
            tracking_state=self._tracking_state.value,
            lost_events=self._lost_events - lost_events_start,
            reacquire_events=self._reacquire_events - reacquire_events_start,
            lost_duration_s=max(
                0.0, self._total_lost_duration(float(data.time)) - lost_duration_start
            ),
            saturated_steps=saturated_steps,
            saturation_ratio=(saturated_steps / completed_steps)
            if completed_steps
            else 0.0,
            requested_control_hz=float(self.config.controller.control_hz),
            effective_control_hz=(completed_steps / simulated_duration)
            if simulated_duration > 0.0
            else 0.0,
            simulated_duration_s=simulated_duration,
            wall_duration_s=wall_duration,
            mean_perception_latency_ms=(float(np.mean(run_latencies)) * 1000.0)
            if run_latencies
            else 0.0,
            p95_perception_latency_ms=(
                float(np.percentile(run_latencies, 95.0)) * 1000.0
            )
            if run_latencies
            else 0.0,
            dropped_camera_frames=self._dropped_camera_frames - dropped_start,
            depth_confidence=self._last_depth_confidence,
            depth_valid_fraction=self._last_depth_valid_fraction,
            depth_inference_ms=self._last_depth_inference_s * 1000.0,
            manipulation_state=self._manipulation_state.value,
            contact_steps=self._contact_steps - contact_steps_start,
            grasped=self._grasped,
            target_lift_m=max(final_snapshot.target_lift_m, self._max_target_lift_m),
            steady_state_rms_error_m=float(
                np.sqrt(np.mean(steady_array * steady_array))
            ),
            steady_state_p95_error_m=float(np.percentile(steady_array, 95.0)),
            perception_device=self._perception_device_name(),
            depth_device=self._depth_device_name(),
            servo_mode=(
                self.config.controller.servo_mode
                if last_state is None
                else last_state.servo_mode
            ),
            final_image_error_px=(
                0.0 if last_state is None else last_state.image_error_px
            ),
            grasp_normal_force_n=(self._peak_grasp_normal_force_n),
            grasp_relative_slip_m=(self._max_grasp_relative_slip_m),
            policy_name=(
                "reactive-pick-place" if self._pick_place_policy is not None else "none"
            ),
            policy_phase=(
                None
                if self._pick_place_policy is None
                else self._pick_place_policy.phase.value
            ),
            policy_attempts=(
                0
                if self._pick_place_policy is None
                else self._pick_place_policy.attempts
            ),
            selected_grasp=(
                None
                if self._pick_place_policy is None
                or self._pick_place_policy.selected_grasp is None
                else self._pick_place_policy.selected_grasp.name
            ),
            place_position=(
                None
                if self._pick_place_policy is None
                else tuple(float(value) for value in self._place_position)
            ),
            place_error_m=self._place_error(final_snapshot.target_position),
            task_succeeded=(
                (
                    self._manipulation_state is ManipulationState.COMPLETE
                    and (
                        self.config.controller.task.strip().lower() != "pick-place"
                        or final_snapshot.place_error_m is not None
                        and final_snapshot.place_error_m
                        <= self.config.controller.policy_place_tolerance_m
                    )
                )
                if self.config.controller.task.strip().lower()
                in {"touch", "grasp", "pick-place"}
                else final_error <= self.config.settling_threshold_m
            ),
            failure_reason=(
                None
                if self._pick_place_policy is None
                else self._pick_place_policy.failure_reason
                or (
                    "placement tolerance was not met"
                    if self._pick_place_policy.succeeded
                    and final_snapshot.place_error_m is not None
                    and final_snapshot.place_error_m
                    > self.config.controller.policy_place_tolerance_m
                    else None
                )
            ),
            grasp_pose_source=self._grasp_pose_source,
            grasp_pose_quality=self._grasp_pose_quality,
            rejected_grasps=tuple(
                f"{item.name}: {item.reason}"
                for item in self._grasp_planner.last_rejections
            ),
            termination_reason=termination_reason,
        )

    def _perception_device_name(self) -> str:
        if self.detector_name == "oracle":
            return "simulator"
        if self.perception is None:
            return "none"
        device = getattr(self.perception, "_device", None)
        return str(device) if device is not None else "cpu"

    def _depth_device_name(self) -> str:
        if self.depth_backend.name == "mujoco":
            return "simulator"
        if self.depth_backend.name == "none":
            return "none"
        return str(getattr(self.depth_backend, "_device_name", "cpu"))

    def _execute_control_step(
        self, viewer, tick: ControlTick, step_index: int
    ) -> _StepOutcome:
        model = self.scene.model
        data = self.scene.data
        time_s = float(data.time)
        physical_target = self.target.dynamics == "physical"
        if physical_target:
            # A physical free body is owned by MuJoCo after reset.  Rewriting
            # qpos here would erase gravity, collision response and grasping.
            requested_target_pos = site_position(
                model, data, self.scene.target_site_name
            )
        else:
            requested_target_pos = self._target_position(time_s)
            set_target_position(model, data, requested_target_pos)
        mujoco.mj_forward(model, data)
        target_pos = site_position(model, data, self.scene.target_site_name)

        detection = self._update_perception(viewer, target_pos)
        perception_updates = 0
        rejected_detections = 0
        if self.detector_name != "oracle" and detection is not None:
            perception_updates, rejected_detections = self._consume_visual_detection(
                detection, time_s
            )
        if self.detector_name != "oracle":
            self._expire_tracking_state(time_s)

        oracle_truth = self.detector_name == "oracle"
        hold_command = (
            not oracle_truth and self._tracking_state is not TrackingState.TRACKING
        )
        occlusion_bridge = (
            self._manipulation_state
            in {
                ManipulationState.CLOSING,
                ManipulationState.LIFTING,
                ManipulationState.TRANSFERRING,
                ManipulationState.PLACING,
                ManipulationState.RELEASING,
                ManipulationState.RETREATING,
                ManipulationState.VERIFYING,
                ManipulationState.RECOVERING,
            }
            and self._last_observed_target is not None
        )
        if oracle_truth:
            command_target = target_pos
        elif (
            not hold_command or occlusion_bridge
        ) and self._last_observed_target is not None:
            command_target = self._last_observed_target
            hold_command = False
        else:
            command_target = None
            hold_command = True
        manipulation_goal: np.ndarray | None = None
        if not hold_command and command_target is not None:
            command_target, manipulation_goal = self._manipulation_targets(
                command_target, target_pos
            )
        if hold_command:
            servo_state = self.controller.hold(
                data, time_s, step_index, tick.duration_s
            )
        else:
            assert command_target is not None
            desired_for_servo = desired_ee_position(
                self.config.controller.task,
                command_target,
                frame_position(
                    model,
                    data,
                    self.scene.ee_frame_type,
                    self.scene.ee_frame_name,
                    self.scene.ee_frame_offset,
                ),
                self.config.controller,
                self.robot.base_position,
            )
            feature = self._latest_target_feature()
            objective = (
                self.visual_objective.compute(
                    ee_position_world=frame_position(
                        model,
                        data,
                        self.scene.ee_frame_type,
                        self.scene.ee_frame_name,
                        self.scene.ee_frame_offset,
                    ),
                    desired_position_world=desired_for_servo,
                    camera=self._last_camera_observation,
                    target_feature=feature,
                    camera_role=self.camera.role,
                    desired_feature_depth_m=self._desired_feature_depth(),
                )
                if (
                    self.visual_objective.mode is ServoMode.PBVS
                    or (
                        self._last_camera_observation is not None
                        and feature is not None
                    )
                )
                else VisualServoObjective(
                    mode=ServoMode.PBVS,
                    gain=self.config.controller.position_gain,
                    max_speed_mps=self.config.controller.max_ee_speed,
                ).compute(
                    ee_position_world=frame_position(
                        model,
                        data,
                        self.scene.ee_frame_type,
                        self.scene.ee_frame_name,
                        self.scene.ee_frame_offset,
                    ),
                    desired_position_world=desired_for_servo,
                )
            )
            servo_state = self.controller.step(
                data,
                command_target,
                time_s,
                step_index,
                tick.duration_s,
                cartesian_velocity_world=objective.linear_velocity_world,
                desired_rotation_world=self._control_orientation(
                    command_target,
                    frame_position(
                        model,
                        data,
                        self.scene.ee_frame_type,
                        self.scene.ee_frame_name,
                        self.scene.ee_frame_offset,
                    ),
                ),
            )
            servo_state.servo_mode = objective.mode.value
            servo_state.image_error_px = objective.image_error_px
        self._last_servo_state = servo_state
        eval_desired = (
            manipulation_goal
            if manipulation_goal is not None
            else desired_ee_position(
                self.config.controller.task,
                target_pos,
                servo_state.ee_position,
                self.config.controller,
                self.robot.base_position,
            )
        )
        error_m = float(np.linalg.norm(eval_desired - servo_state.ee_position))

        for _ in range(tick.substeps):
            if not physical_target:
                if self.config.manual_control and viewer is not None:
                    set_target_position(model, data, requested_target_pos)
                else:
                    set_target_position(
                        model, data, self._target_position(float(data.time))
                    )
            mujoco.mj_step(model, data)
        return _StepOutcome(
            servo_state=servo_state,
            error_m=error_m,
            hold=hold_command,
            oracle_truth=oracle_truth,
            perception_updates=perception_updates,
            rejected_detections=rejected_detections,
        )

    def _latest_target_feature(self) -> FeatureObservation | None:
        detection = self._last_accepted_detection
        observation = self._last_camera_observation
        if detection is None or observation is None or detection.centroid_px is None:
            return None
        pixel = np.asarray(detection.centroid_px, dtype=float).reshape(2)
        x = int(np.clip(round(float(pixel[0])), 0, observation.intrinsics.width - 1))
        y = int(np.clip(round(float(pixel[1])), 0, observation.intrinsics.height - 1))
        x0, x1 = max(0, x - 2), min(observation.intrinsics.width, x + 3)
        y0, y1 = max(0, y - 2), min(observation.intrinsics.height, y + 3)
        window = np.asarray(observation.depth_m[y0:y1, x0:x1], dtype=float)
        valid = window[np.isfinite(window) & (window > 0.0)]
        if valid.size == 0:
            return None
        return FeatureObservation(
            pixel=pixel,
            depth_m=float(np.median(valid)),
            confidence=float(np.clip(detection.score, 0.0, 1.0)),
        )

    def _desired_feature_depth(self) -> float:
        task = self.config.controller.task.strip().lower()
        if task in {"standoff", "front-standoff"}:
            return max(0.03, float(self.config.controller.standoff_m))
        return max(0.03, float(self.target.radius))

    def _control_orientation(
        self, command_target: np.ndarray, ee_position: np.ndarray
    ) -> np.ndarray | None:
        if self.config.controller.task.strip().lower() in {
            "touch",
            "grasp",
            "pick-place",
        }:
            if (
                self.config.controller.task.strip().lower() == "pick-place"
                and self._pick_place_policy is not None
                and self._pick_place_policy.selected_grasp is not None
            ):
                return tool_z_facing_rotation(
                    self._pick_place_policy.selected_grasp.approach
                )
            point = self._safe_grasp_point()
            if point is not None:
                return tool_z_facing_rotation(point.approach)
        if self.camera.role == "eye-in-hand":
            return tool_z_facing_rotation(
                np.asarray(command_target, dtype=float).reshape(3)
                - np.asarray(ee_position, dtype=float).reshape(3)
            )
        return None

    def _manipulation_targets(
        self,
        observed_target: np.ndarray,
        truth_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return controller/evaluation goals and advance touch/grasp stages."""
        task = self.config.controller.task.strip().lower()
        observed = np.asarray(observed_target, dtype=float).reshape(3)
        truth = np.asarray(truth_target, dtype=float).reshape(3)
        if task == "pick-place":
            return self._pick_place_targets(observed, truth)
        if task not in {"touch", "grasp"}:
            return observed, None

        point = self.get_grasp_point()
        # Perception currently estimates translation only.  Preserve the
        # target-local grasp offset using MuJoCo orientation, but translate it
        # from the observed object centre rather than using oracle position.
        observed_point = observed + (point.position - truth)
        pregrasp_observed = (
            observed_point - point.approach * self.config.controller.grasp_approach_m
        )
        pregrasp_truth = (
            point.position - point.approach * self.config.controller.grasp_approach_m
        )
        ee = frame_position(
            self.scene.model,
            self.scene.data,
            self.scene.ee_frame_type,
            self.scene.ee_frame_name,
            self.scene.ee_frame_offset,
        )
        in_contact = self._target_robot_contact()
        if in_contact:
            self._contact_steps += 1

        tolerance = self.config.controller.grasp_stage_tolerance_m
        if self._manipulation_state is ManipulationState.PREGRASP:
            if float(np.linalg.norm(ee - pregrasp_truth)) <= tolerance:
                self._manipulation_state = ManipulationState.APPROACHING

        if self._manipulation_state is ManipulationState.APPROACHING:
            distance = float(np.linalg.norm(ee - point.position))
            if task == "touch" and in_contact:
                self._manipulation_state = ManipulationState.COMPLETE
                self._lift_goal_position = ee.copy()
            elif (
                task == "grasp"
                and distance <= self.config.controller.grasp_attach_distance_m
            ):
                if self._grasp_evaluator is None:
                    self._manipulation_state = ManipulationState.FAILED
                    return ee.copy(), ee.copy()
                try:
                    self.activate_grasp()
                except (KeyError, RuntimeError, ValueError):
                    self._manipulation_state = ManipulationState.FAILED
                else:
                    self._manipulation_state = ManipulationState.CLOSING

        if self._manipulation_state is ManipulationState.CLOSING:
            assert self._grasp_evaluator is not None
            self._closing_frames += 1
            self._grasp_evidence = self._grasp_evaluator.evaluate(self.scene.data)
            self._record_grasp_evidence()
            if self._grasp_evidence.grasped:
                self._grasped = True
                self._manipulation_state = ManipulationState.LIFTING
                self._lift_goal_position = ee + np.array(
                    [0.0, 0.0, self.config.controller.grasp_lift_m], dtype=float
                )
            elif self._closing_frames >= 240:
                self._manipulation_state = ManipulationState.FAILED
                return ee.copy(), ee.copy()

        if self._manipulation_state is ManipulationState.LIFTING:
            assert self._lift_goal_position is not None
            assert self._grasp_evaluator is not None
            self._grasp_evidence = self._grasp_evaluator.evaluate(self.scene.data)
            self._record_grasp_evidence()
            contact_stable = self._grasp_evidence.stable_frames > 0
            self._grasp_lost_frames = (
                0 if contact_stable else self._grasp_lost_frames + 1
            )
            if self._grasp_lost_frames >= self.config.controller.grasp_lost_frames:
                self._grasped = False
                self._manipulation_state = ManipulationState.FAILED
                return ee.copy(), ee.copy()
            current_target = site_position(
                self.scene.model, self.scene.data, self.scene.target_site_name
            )
            if (
                self._target_lift(current_target)
                >= 0.8 * self.config.controller.grasp_lift_m
            ):
                self._manipulation_state = ManipulationState.COMPLETE

        if self._manipulation_state in {
            ManipulationState.CLOSING,
            ManipulationState.LIFTING,
            ManipulationState.COMPLETE,
        }:
            if self._lift_goal_position is not None:
                return self._lift_goal_position.copy(), self._lift_goal_position.copy()
        if self._manipulation_state is ManipulationState.APPROACHING:
            return observed_point, point.position.copy()
        if self._manipulation_state is ManipulationState.CLOSING:
            return observed_point, point.position.copy()
        if self._manipulation_state is ManipulationState.FAILED:
            return ee.copy(), ee.copy()
        return pregrasp_observed, pregrasp_truth

    def _pick_place_targets(
        self, observed_target: np.ndarray, truth_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        policy = self._pick_place_policy
        supervisor = self._safety_supervisor
        if policy is None or supervisor is None or self._grasp_evaluator is None:
            self._manipulation_state = ManipulationState.FAILED
            ee = self._current_ee_position()
            return ee, ee

        ee = self._current_ee_position()
        if self._grasp_initial_target_z is not None:
            self._max_target_lift_m = max(
                self._max_target_lift_m, self._target_lift(truth_target)
            )
        if self._target_robot_contact():
            self._contact_steps += 1
        if self._gripper_commanded_closed:
            self._grasp_evidence = self._grasp_evaluator.evaluate(self.scene.data)
            self._record_grasp_evidence()
            if self._grasp_evidence.grasped:
                self._grasped = True
                self._grasp_lost_frames = 0
            elif self._grasped:
                if self._grasp_evidence.stable_frames > 0:
                    self._grasp_lost_frames = 0
                else:
                    self._grasp_lost_frames += 1
                if self._grasp_lost_frames >= self.config.controller.grasp_lost_frames:
                    self._grasped = False

        if policy.selected_grasp is None:
            try:
                camera_id = mujoco.mj_name2id(
                    self.scene.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    self.scene.camera_name,
                )
                points = self._observed_grasp_points(observed_target)
                candidate = self._grasp_planner.select(
                    points,
                    GraspPlanningContext(
                        ee_position=ee,
                        camera_position=np.asarray(
                            self.scene.data.cam_xpos[camera_id], dtype=float
                        ),
                        support_z=self.scene.support_z,
                        approach_distance_m=self.config.controller.grasp_approach_m,
                        max_reach_m=1.25,
                        max_gripper_width_m=self.robot.max_gripper_width_m,
                        excluded_names=policy.failed_grasps,
                        reachable=lambda point: self.controller.is_position_reachable(
                            self.scene.data,
                            point,
                            tolerance_m=self.config.controller.grasp_stage_tolerance_m,
                        ),
                        path_is_valid=self._cartesian_path_is_valid,
                    ),
                )
                policy.set_grasp(candidate, observed_target)
            except (KeyError, RuntimeError, ValueError) as exc:
                policy.abort(float(self.scene.data.time), str(exc))

        evidence = self._grasp_evidence
        observation = PolicyObservation(
            time_s=float(self.scene.data.time),
            ee_position=ee,
            target_position=np.asarray(observed_target, dtype=float).reshape(3),
            tracking_valid=self._tracking_state is TrackingState.TRACKING,
            grasped=self._grasped,
            contact_stable_frames=0 if evidence is None else evidence.stable_frames,
            normal_force_n=0.0 if evidence is None else evidence.normal_force_n,
            place_error_m=(
                self._place_error(observed_target)
                if self._tracking_state is TrackingState.TRACKING
                else None
            ),
        )
        command = supervisor.supervise(policy.step(observation), observation)
        if (
            command.phase is PolicyPhase.RECOVER
            and policy.phase is not PolicyPhase.RECOVER
        ):
            command = policy.request_recovery(
                ee, float(self.scene.data.time), command.reason or "safety recovery"
            )

        if (
            command.gripper is GripperCommand.CLOSE
            and not self._gripper_commanded_closed
        ):
            try:
                selected = policy.selected_grasp
                self.activate_grasp(None if selected is None else selected.name)
            except (KeyError, RuntimeError, ValueError) as exc:
                policy.abort(float(self.scene.data.time), str(exc))
                command = policy.step(observation)
        elif command.gripper is GripperCommand.OPEN and self._gripper_commanded_closed:
            self.release_grasp()

        self._manipulation_state = self._policy_manipulation_state(policy.phase)
        goal = np.asarray(command.goal_position, dtype=float).reshape(3)
        return goal.copy(), goal.copy()

    def _current_ee_position(self) -> np.ndarray:
        return frame_position(
            self.scene.model,
            self.scene.data,
            self.scene.ee_frame_type,
            self.scene.ee_frame_name,
            self.scene.ee_frame_offset,
        )

    def _record_grasp_evidence(self) -> None:
        evidence = self._grasp_evidence
        if evidence is None:
            return
        self._peak_grasp_normal_force_n = max(
            self._peak_grasp_normal_force_n, evidence.normal_force_n
        )
        self._max_grasp_relative_slip_m = max(
            self._max_grasp_relative_slip_m, evidence.relative_slip_m
        )

    def _observed_grasp_points(
        self, observed_target: np.ndarray
    ) -> tuple[WorldGraspPoint, ...]:
        """Lift descriptor-local grasp geometry into the observed 6D pose."""
        centre = np.asarray(observed_target, dtype=float).reshape(3)
        rotation = self._estimate_target_rotation()
        points = []
        for point in self.target.grasp_points:
            points.append(
                WorldGraspPoint(
                    point.name,
                    centre + rotation @ np.asarray(point.position, dtype=float),
                    rotation @ np.asarray(point.approach, dtype=float),
                    point.width_m,
                )
            )
        return tuple(points)

    def _estimate_target_rotation(self) -> np.ndarray:
        if self.detector_name == "oracle":
            body_id = mujoco.mj_name2id(
                self.scene.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.scene.target_body_name,
            )
            if body_id >= 0:
                rotation = np.asarray(
                    self.scene.data.xmat[body_id], dtype=float
                ).reshape(3, 3)
                self._last_target_rotation = rotation.copy()
                self._grasp_pose_source = "oracle-6d"
                self._grasp_pose_quality = 1.0
                return rotation.copy()
        detection = self._last_accepted_detection
        observation = self._last_accepted_camera_observation
        if (
            detection is not None
            and detection.mask is not None
            and observation is not None
            and observation.depth_metric
        ):
            pose = estimate_pose_6d(observation, detection.mask)
            if pose is not None and pose.quality >= 0.20:
                rotation = align_rotation_to_reference(
                    pose.rotation_world, self._last_target_rotation
                )
                self._last_target_rotation = rotation.copy()
                self._grasp_pose_source = "visual-depth-6d"
                self._grasp_pose_quality = float(pose.quality)
                return rotation
        self._grasp_pose_source = "descriptor-fallback"
        self._grasp_pose_quality = 0.0
        return self._last_target_rotation.copy()

    def _cartesian_path_is_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        return self._path_validator.check(start, end).valid

    @staticmethod
    def _policy_manipulation_state(phase: PolicyPhase) -> ManipulationState:
        return {
            PolicyPhase.ACQUIRE: ManipulationState.PREGRASP,
            PolicyPhase.PREGRASP: ManipulationState.PREGRASP,
            PolicyPhase.APPROACH: ManipulationState.APPROACHING,
            PolicyPhase.CLOSE: ManipulationState.CLOSING,
            PolicyPhase.VERIFY: ManipulationState.CLOSING,
            PolicyPhase.LIFT: ManipulationState.LIFTING,
            PolicyPhase.TRANSFER: ManipulationState.TRANSFERRING,
            PolicyPhase.PLACE: ManipulationState.PLACING,
            PolicyPhase.RELEASE: ManipulationState.RELEASING,
            PolicyPhase.RETREAT: ManipulationState.RETREATING,
            PolicyPhase.VERIFY_PLACE: ManipulationState.VERIFYING,
            PolicyPhase.RECOVER: ManipulationState.RECOVERING,
            PolicyPhase.SUCCEEDED: ManipulationState.COMPLETE,
            PolicyPhase.FAILED: ManipulationState.FAILED,
        }[phase]

    def _resolve_place_position(self) -> np.ndarray:
        configured = self.config.controller.place_position
        half_footprint = 0.5 * max(self.target.size[0], self.target.size[1])
        inset = half_footprint + 0.015
        if configured is not None:
            place = np.asarray(configured, dtype=float).reshape(3).copy()
            if self._work_surface is not None:
                if not self._work_surface.contains(place, inset_m=inset):
                    raise ValueError(
                        "place_position does not fit inside the injected work surface"
                    )
                minimum_z = self.scene.support_z + 0.5 * self.target.size[2]
                if place[2] < minimum_z - 0.005:
                    raise ValueError("place_position would penetrate the work surface")
            if not self.controller.is_position_reachable(
                self.scene.data,
                place,
                tolerance_m=max(self.config.controller.grasp_stage_tolerance_m, 0.035),
            ):
                raise ValueError("place_position is not reachable by numerical IK")
            return place
        place = self._target_base_position.copy()
        if self._work_surface is not None:
            center = np.asarray(self._work_surface.center_xy, dtype=float)
            available = np.asarray(self._work_surface.half_size_xy, dtype=float) - inset
            axis = int(np.argmax(available))
            direction = -1.0 if place[axis] >= center[axis] else 1.0
            place[axis] += direction * min(0.10, max(0.04, 0.75 * available[axis]))
            place = self._work_surface.clamp_xy(place, inset_m=inset)
            place[2] = self.scene.support_z + 0.5 * self.target.size[2] + 0.002
        return place

    def _place_error(self, target_position: np.ndarray) -> float | None:
        if self._pick_place_policy is None:
            return None
        return float(
            np.linalg.norm(
                np.asarray(target_position, dtype=float).reshape(3)
                - self._place_position
            )
        )

    def _safe_grasp_point(self) -> WorldGraspPoint | None:
        try:
            return self.get_grasp_point()
        except (KeyError, RuntimeError, ValueError):
            return None

    def _target_robot_contact(self) -> bool:
        """Report contact between a target geom and any non-world robot body."""
        model = self.scene.model
        data = self.scene.data
        target_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.scene.target_body_name
        )
        if target_body < 0:
            return False
        for contact in data.contact[: data.ncon]:
            body1 = int(model.geom_bodyid[int(contact.geom1)])
            body2 = int(model.geom_bodyid[int(contact.geom2)])
            if body1 == target_body and body2 not in {0, target_body}:
                return True
            if body2 == target_body and body1 not in {0, target_body}:
                return True
        return False

    def _target_lift(self, target_position: np.ndarray) -> float:
        if self._grasp_initial_target_z is None:
            return 0.0
        return max(
            0.0,
            float(np.asarray(target_position, dtype=float)[2])
            - self._grasp_initial_target_z,
        )

    def _sleep_to_simulation_time(self, sim_start: float, wall_start: float) -> None:
        while True:
            simulated_elapsed = max(0.0, float(self.scene.data.time) - sim_start)
            wall_elapsed = max(0.0, time.perf_counter() - wall_start)
            remaining = simulated_elapsed - wall_elapsed
            if remaining <= 0.0:
                return
            time.sleep(min(remaining, 0.01))

    def _settling_time(self, errors: list[float], times: list[float]) -> float | None:
        if not errors:
            return None
        threshold = float(getattr(self.config, "settling_threshold_m", 0.01))
        within = np.asarray(errors, dtype=float) <= threshold
        suffix_all = np.logical_and.accumulate(within[::-1])[::-1]
        indices = np.flatnonzero(suffix_all)
        return None if indices.size == 0 else float(times[int(indices[0])])

    def _consume_visual_detection(
        self, detection: Detection, time_s: float
    ) -> tuple[int, int]:
        if not detection.success or detection.target_position is None:
            return 0, 0
        capture_sim_time = self._detection_capture_sim_time(detection, time_s)
        available_sim_time = self._detection_available_sim_time(detection, time_s)
        capture_wall_time = self._detection_capture_wall_time(detection)
        available_wall_time = self._detection_available_wall_time(detection)
        self._last_detection_sim_time = capture_sim_time
        self._last_detection_available_sim_time = available_sim_time
        self._last_detection_wall_time = capture_wall_time
        self._last_detection_available_wall_time = available_wall_time
        if available_wall_time is not None and capture_wall_time is not None:
            latency = max(0.0, available_wall_time - capture_wall_time)
            self._perception_latencies_s.append(latency)
        if time_s - capture_sim_time > float(self.config.detection_timeout_s):
            return 0, 1
        allow_jump = self._tracking_state is not TrackingState.TRACKING
        if not self._accept_detection(detection, allow_jump=allow_jump):
            if self.config.debug_perception:
                print(
                    f"reject detection pos={detection.target_position} "
                    f"bbox={detection.bbox_xyxy} score={detection.score:.3f}"
                )
            return 0, 1
        position = np.asarray(detection.target_position, dtype=float).reshape(3).copy()
        if self._tracking_state is TrackingState.TRACKING:
            self._commit_detection(
                detection, position, capture_sim_time, capture_wall_time
            )
            return 1, 0
        if not self._ever_tracked:
            self._commit_detection(
                detection, position, capture_sim_time, capture_wall_time
            )
            self._enter_tracking(position, reacquired=False, time_s=time_s)
            return 1, 0

        tolerance = float(getattr(self.config, "reacquire_position_tolerance_m", 0.12))
        if self._tracking_state is TrackingState.LOST:
            self._tracking_state = TrackingState.REACQUIRING
            self._reacquire_candidate = position
            self._reacquire_count = 1
        elif (
            self._reacquire_candidate is None
            or np.linalg.norm(position - self._reacquire_candidate) > tolerance
        ):
            self._reacquire_candidate = position
            self._reacquire_count = 1
        elif (
            self._reacquire_last_time_s is None
            or capture_sim_time > self._reacquire_last_time_s + 1e-12
        ):
            count = self._reacquire_count + 1
            self._reacquire_candidate = (
                self._reacquire_candidate * self._reacquire_count + position
            ) / count
            self._reacquire_count = count
        self._reacquire_last_time_s = capture_sim_time
        confirm_frames = max(
            1, int(getattr(self.config, "reacquire_confirm_frames", 3))
        )
        if self._reacquire_count >= confirm_frames:
            confirmed = self._reacquire_candidate.copy()
            confirmed_detection = replace(detection, target_position=confirmed)
            self._commit_detection(
                confirmed_detection,
                confirmed,
                capture_sim_time,
                capture_wall_time,
            )
            self._enter_tracking(confirmed, reacquired=True, time_s=time_s)
        return 1, 0

    def _commit_detection(
        self,
        detection: Detection,
        position: np.ndarray,
        capture_sim_time: float,
        capture_wall_time: float | None,
    ) -> None:
        self._last_observed_target = position.copy()
        self._last_observed_time_s = capture_sim_time
        self._last_accepted_detection_position = position.copy()
        self._last_accepted_detection = detection
        self._last_accepted_camera_observation = self._last_camera_observation
        self._last_accepted_detection_sim_time = capture_sim_time
        self._last_accepted_detection_available_sim_time = (
            self._last_detection_available_sim_time
        )
        self._last_accepted_detection_wall_time = (
            capture_wall_time or time.perf_counter()
        )

    def _enter_tracking(
        self, position: np.ndarray, *, reacquired: bool, time_s: float
    ) -> None:
        if reacquired:
            self._reacquire_events += 1
        if self._lost_since_sim_time is not None:
            self._completed_lost_duration_s += max(
                0.0, time_s - self._lost_since_sim_time
            )
        self._lost_since_sim_time = None
        self._tracking_state = TrackingState.TRACKING
        self._ever_tracked = True
        self._reacquire_candidate = None
        self._reacquire_count = 0
        self._reacquire_last_time_s = None
        self.controller.end_hold(position)

    def _expire_tracking_state(self, time_s: float) -> None:
        timeout = float(self.config.detection_timeout_s)
        if self._tracking_state is TrackingState.TRACKING:
            if (
                self._last_observed_time_s is None
                or time_s - self._last_observed_time_s > timeout
            ):
                self._enter_lost(time_s)
        elif self._tracking_state is TrackingState.REACQUIRING:
            if (
                self._reacquire_last_time_s is None
                or time_s - self._reacquire_last_time_s > timeout
            ):
                self._tracking_state = TrackingState.LOST
                self._reacquire_candidate = None
                self._reacquire_count = 0
                self._reacquire_last_time_s = None

    def _enter_lost(self, time_s: float) -> None:
        if self._tracking_state is TrackingState.TRACKING:
            self._lost_events += 1
        if self._lost_since_sim_time is None:
            self._lost_since_sim_time = time_s
        self._tracking_state = TrackingState.LOST
        if self._manipulation_state not in {
            ManipulationState.CLOSING,
            ManipulationState.LIFTING,
            ManipulationState.TRANSFERRING,
            ManipulationState.PLACING,
            ManipulationState.RELEASING,
            ManipulationState.RETREATING,
            ManipulationState.VERIFYING,
            ManipulationState.RECOVERING,
        }:
            self._last_observed_target = None
            self._last_observed_time_s = None
        # Old-target jump gating must not make a moved target impossible to
        # reacquire after an occlusion.
        self._last_accepted_detection_position = None
        self._reacquire_candidate = None
        self._reacquire_count = 0
        self._reacquire_last_time_s = None
        self.controller.begin_hold(self.scene.data)

    def _total_lost_duration(self, time_s: float) -> float:
        active = 0.0
        if self._lost_since_sim_time is not None:
            active = max(0.0, time_s - self._lost_since_sim_time)
        return self._completed_lost_duration_s + active

    def _orientation_error(
        self, target_position: np.ndarray, desired_position: np.ndarray
    ) -> float:
        desired_rotation = desired_ee_orientation(
            self.config.controller.task, target_position, desired_position
        )
        if desired_rotation is None:
            return 0.0
        current_rotation = self.controller.frame_rotation(self.scene.data)
        current_axis = current_rotation @ np.asarray(self.robot.tool_axis, dtype=float)
        return float(
            np.linalg.norm(vector_alignment_error(current_axis, desired_rotation[:, 2]))
        )

    def _uses_async_perception(self, viewer) -> bool:
        if viewer is None or self.detector_name == "oracle":
            return False
        # On macOS, keep MuJoCo viewer + perception on the main thread to avoid AppKit thread crashes.
        return sys.platform != "darwin"

    def _should_lazy_load_perception(self) -> bool:
        return self.detector_name == "semantic"

    def _prepare_perception_for_run(self, viewer) -> None:
        if self.detector_name != "semantic":
            return
        if viewer is not None:
            mode = (
                "asynchronously"
                if self._uses_async_perception(viewer)
                else "on the main thread"
            )
            print(f"semantic models will load on first use; inference will run {mode}")

    def _ensure_perception(self) -> PerceptionBackend:
        if self.perception is None:
            self.perception = build_perception(self.config.detector)
        return self.perception

    def _update_perception(
        self, viewer, truth_position: np.ndarray
    ) -> Detection | None:
        perception = (
            self._ensure_perception()
            if not self._uses_async_perception(viewer)
            else None
        )
        if self.detector_name == "oracle":
            detection = perception.detect(
                None, truth_position, self.target, self._perception_prompt()
            )
            now_wall = time.perf_counter()
            self._last_detection = detection
            self._last_accepted_detection = detection
            self._last_accepted_detection_position = truth_position.copy()
            self._last_detection_sim_time = float(self.scene.data.time)
            self._last_detection_available_sim_time = self._last_detection_sim_time
            self._last_accepted_detection_sim_time = self._last_detection_sim_time
            self._last_accepted_detection_available_sim_time = (
                self._last_detection_sim_time
            )
            self._last_detection_wall_time = now_wall
            self._last_detection_available_wall_time = now_wall
            self._last_accepted_detection_wall_time = now_wall
            return detection
        if self._uses_async_perception(viewer):
            return self._update_async_perception(truth_position)
        due_detection = self._pop_due_detection()
        if not self._should_sample_camera_now():
            return due_detection
        if self._drop_perception_frame():
            return due_detection
        observation = self._render_camera_observation()
        observation = self._resolve_observation_depth(observation)
        detection = perception.detect(
            observation, truth_position, self.target, self._perception_prompt()
        )
        self._debug_detection(detection, observation, truth_position)
        self._latest_overlay_bgr = self._draw_camera_overlay(
            observation, detection, False
        )
        self._record_dashboard(self._latest_overlay_bgr)
        self._latest_overlay_rgb = None
        delivered = self._queue_detection(
            detection, observation, defer=due_detection is not None
        )
        return due_detection if due_detection is not None else delivered

    def _update_async_perception(self, truth_position: np.ndarray) -> Detection | None:
        completed_detection = self._pop_due_detection()
        if self._perception_future is not None and self._perception_future.done():
            try:
                observation, detection = self._perception_future.result()
                delivered = self._queue_detection(
                    detection,
                    observation,
                    defer=completed_detection is not None,
                )
                if completed_detection is None:
                    completed_detection = delivered
                self._latest_overlay_bgr = self._draw_camera_overlay(
                    observation, detection, False
                )
                self._record_dashboard(self._latest_overlay_bgr)
                self._latest_overlay_rgb = None
            except Exception as exc:
                self._perception_disabled = True
                self._last_detection = Detection(False, self.detector_name, None)
                completed_detection = self._last_detection
                print(f"perception worker failed: {exc}", file=sys.stderr)
            finally:
                self._perception_future = None
        if self._perception_disabled:
            return completed_detection
        should_sample = self._should_sample_camera_now()
        if should_sample:
            if self._perception_future is not None or self._drop_perception_frame():
                self._dropped_camera_frames += int(self._perception_future is not None)
            else:
                observation = self._render_camera_observation(
                    resolve_learned_depth=False
                )
                self._latest_overlay_bgr = self._draw_camera_overlay(
                    observation, None, True
                )
                self._record_dashboard(self._latest_overlay_bgr)
                self._latest_overlay_rgb = None
                if self._perception_executor is not None:
                    prompt = self._perception_prompt()
                    target = self.target
                    truth = truth_position.copy()
                    self._perception_future = self._perception_executor.submit(
                        self._detect_in_worker,
                        observation,
                        truth,
                        target,
                        prompt,
                    )
        return completed_detection

    def _queue_detection(
        self,
        detection: Detection,
        observation: CameraObservation | None,
        *,
        defer: bool,
    ) -> Detection | None:
        if observation is not None:
            self._detection_observations[id(detection)] = observation
        capture_sim = (
            float(self.scene.data.time)
            if observation is None
            else float(observation.sim_time_s)
        )
        capture_wall = time.perf_counter()
        if observation is not None and observation.wall_time_s > 0.0:
            capture_wall = float(observation.wall_time_s)
        latency = float(getattr(self.config, "perception_latency_s", 0.0))
        jitter = float(getattr(self.config, "perception_jitter_s", 0.0))
        if jitter > 0.0:
            latency += float(self._rng.normal(0.0, jitter))
        latency = max(0.0, latency)
        inference_time = (
            self._first_finite_attribute(detection, ("inference_time_s",)) or 0.0
        )
        available_sim = capture_sim + latency
        available_wall = max(
            time.perf_counter(), capture_wall + latency, capture_wall + inference_time
        )
        if defer or float(self.scene.data.time) + 1e-12 < available_sim:
            self._pending_detection_deliveries.append(
                (available_sim, detection, capture_sim, capture_wall, available_wall)
            )
            self._pending_detection_deliveries.sort(key=lambda item: item[0])
            return None
        self._mark_detection_available(
            detection,
            capture_sim,
            capture_wall,
            max(available_sim, float(self.scene.data.time)),
            available_wall,
        )
        return detection

    def _pop_due_detection(self) -> Detection | None:
        if not self._pending_detection_deliveries:
            return None
        now_sim = float(self.scene.data.time)
        if self._pending_detection_deliveries[0][0] > now_sim + 1e-12:
            return None
        available_sim, detection, capture_sim, capture_wall, available_wall = (
            self._pending_detection_deliveries.pop(0)
        )
        self._mark_detection_available(
            detection,
            capture_sim,
            capture_wall,
            max(available_sim, now_sim),
            max(available_wall, time.perf_counter()),
        )
        return detection

    def _mark_detection_available(
        self,
        detection: Detection,
        capture_sim: float,
        capture_wall: float,
        available_sim: float,
        available_wall: float,
    ) -> None:
        observation = self._detection_observations.pop(id(detection), None)
        if observation is not None:
            self._last_camera_observation = observation
        self._last_detection = detection
        self._last_detection_sim_time = capture_sim
        self._last_detection_wall_time = capture_wall
        self._last_detection_available_sim_time = available_sim
        self._last_detection_available_wall_time = available_wall

    def _drop_perception_frame(self) -> bool:
        probability = float(getattr(self.config, "perception_drop_probability", 0.0))
        dropped = probability > 0.0 and float(self._rng.random()) < probability
        if dropped:
            self._dropped_camera_frames += 1
        return dropped

    def _perception_prompt(self) -> str:
        return self.config.perception_prompt or self.target.name

    def _should_sample_camera_now(
        self, now: float | None = None, *, simulation_time: bool = False
    ) -> bool:
        del simulation_time
        camera_period = 1.0 / float(self.config.camera_fps)
        sample_time = float(self.scene.data.time) if now is None else float(now)
        if sample_time + 1e-12 < self._next_camera_sim_time:
            return False
        missed = max(
            0, int(np.floor((sample_time - self._next_camera_sim_time) / camera_period))
        )
        if missed:
            self._dropped_camera_frames += missed
        self._next_camera_sim_time += (missed + 1) * camera_period
        self._last_camera_sim_time = sample_time
        self._last_camera_wall_time = time.perf_counter()
        return True

    def _perception_age_s(self) -> float | None:
        if self._last_accepted_detection_sim_time is not None:
            return max(
                0.0,
                float(self.scene.data.time) - self._last_accepted_detection_sim_time,
            )
        if self._last_accepted_detection_wall_time is not None:
            return max(
                0.0, time.perf_counter() - self._last_accepted_detection_wall_time
            )
        return None

    @staticmethod
    def _first_finite_attribute(value, names: tuple[str, ...]) -> float | None:
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is None:
                continue
            try:
                number = float(candidate)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                return number
        return None

    def _detection_capture_sim_time(
        self, detection: Detection, fallback: float
    ) -> float:
        timestamp = self._first_finite_attribute(
            detection,
            ("capture_sim_time_s", "captured_sim_time_s", "sim_time_s"),
        )
        if timestamp is None and self._last_detection is detection:
            timestamp = self._last_detection_sim_time
        if timestamp is None and self._last_detection is detection:
            timestamp = self._first_finite_attribute(detection, ("measurement_time_s",))
        return float(fallback if timestamp is None else timestamp)

    def _detection_available_sim_time(
        self, detection: Detection, fallback: float
    ) -> float:
        timestamp = self._first_finite_attribute(
            detection,
            ("available_sim_time_s", "availability_sim_time_s"),
        )
        if timestamp is None and self._last_detection is detection:
            timestamp = self._last_detection_available_sim_time
        return float(fallback if timestamp is None else timestamp)

    def _detection_capture_wall_time(self, detection: Detection) -> float | None:
        timestamp = self._first_finite_attribute(
            detection,
            ("capture_wall_time_s", "captured_wall_time_s", "wall_time_s"),
        )
        if timestamp is None and self._last_detection is detection:
            timestamp = self._last_detection_wall_time
        if timestamp is None and self._last_detection is detection:
            timestamp = self._first_finite_attribute(detection, ("capture_time_s",))
            if timestamp is not None and timestamp <= 0.0:
                timestamp = None
        return time.perf_counter() if timestamp is None else timestamp

    def _detection_available_wall_time(self, detection: Detection) -> float:
        timestamp = self._first_finite_attribute(
            detection,
            ("available_wall_time_s", "availability_wall_time_s"),
        )
        if timestamp is None and self._last_detection is detection:
            timestamp = self._last_detection_available_wall_time
        return time.perf_counter() if timestamp is None else timestamp

    @staticmethod
    def _detection_covariance(detection: Detection | None) -> np.ndarray | None:
        if detection is None:
            return None
        covariance = getattr(detection, "covariance", None)
        if covariance is None:
            covariance = getattr(detection, "position_covariance", None)
        if covariance is None:
            covariance = getattr(detection, "covariance_xyz", None)
        if covariance is None:
            return None
        try:
            value = np.asarray(covariance, dtype=float).reshape(3, 3)
        except (TypeError, ValueError):
            return None
        return value.copy() if np.isfinite(value).all() else None

    @classmethod
    def _optional_detection_scalar(
        cls, detection: Detection | None, name: str
    ) -> float | None:
        if detection is None:
            return None
        value = cls._first_finite_attribute(detection, (name,))
        return value if value is not None and value > 0.0 else None

    def _accept_detection(
        self, detection: Detection, *, allow_jump: bool = False
    ) -> bool:
        if detection.target_position is None:
            return False
        try:
            position = np.asarray(detection.target_position, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(position).all():
            return False
        covariance = self._detection_covariance(detection)
        if getattr(detection, "covariance", None) is not None:
            if covariance is None:
                return False
            eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
            if (
                float(np.min(eigenvalues)) < -1e-9
                or float(np.max(np.diag(covariance))) > 0.04
            ):
                return False
        valid_fraction = self._first_finite_attribute(detection, ("valid_fraction",))
        if valid_fraction is not None and 0.0 < valid_fraction < 0.01:
            return False
        quality = self._first_finite_attribute(detection, ("quality",))
        if quality is not None and 0.0 < quality < 0.05:
            return False
        if self.robot.detection_bounds is None:
            # Custom robot descriptors may omit workspace bounds.  Avoid
            # silently imposing Panda coordinates while still rejecting
            # clearly nonsensical world estimates.
            lower = np.full(3, -5.0, dtype=float)
            upper = np.full(3, 5.0, dtype=float)
        else:
            lower = np.array(self.robot.detection_bounds[0], dtype=float).reshape(3)
            upper = np.array(self.robot.detection_bounds[1], dtype=float).reshape(3)
        if np.any(position < lower) or np.any(position > upper):
            return False
        if not allow_jump and self._last_accepted_detection_position is not None:
            jump = float(
                np.linalg.norm(position - self._last_accepted_detection_position)
            )
            if jump > 0.28:
                return False
        return True

    def _detect_in_worker(
        self,
        observation: CameraObservation,
        truth_position: np.ndarray,
        target,
        prompt: str,
    ) -> tuple[CameraObservation, Detection]:
        perception = self._ensure_perception()
        observation = self._resolve_observation_depth(observation)
        detection = perception.detect(observation, truth_position, target, prompt)
        self._debug_detection(detection, observation, truth_position)
        return observation, detection

    def _debug_detection(
        self,
        detection: Detection,
        observation: CameraObservation | None,
        truth_position: np.ndarray,
    ) -> None:
        if not self.config.debug_perception:
            return
        err = None
        try:
            if detection.target_position is not None:
                err = float(
                    np.linalg.norm(
                        np.asarray(detection.target_position, dtype=float).reshape(3)
                        - truth_position
                    )
                )
        except (TypeError, ValueError):
            err = None
        depth_backend = observation.depth_backend if observation is not None else "none"
        mask_area = None if detection.mask is None else int((detection.mask > 0).sum())
        print(
            "perception "
            f"backend={detection.backend} success={detection.success} score={detection.score:.3f} "
            f"anchor={detection.anchor_type} pos={detection.target_position} truth_err_m={err} "
            f"bbox={detection.bbox_xyxy} mask_area={mask_area} depth={depth_backend}"
        )

    def _resolve_observation_depth(
        self, observation: CameraObservation | None
    ) -> CameraObservation | None:
        if observation is None or observation.depth_backend != "mujoco-hint":
            return observation
        depth = self.depth_backend.estimate(observation.frame_bgr, observation.depth_m)
        self._last_depth_metric = depth.metric
        self._last_depth_confidence = float(
            getattr(depth, "confidence", 1.0 if depth.metric else 0.0)
        )
        self._last_depth_valid_fraction = float(
            getattr(depth, "valid_fraction", np.mean(np.isfinite(depth.depth_m)))
        )
        self._last_depth_inference_s = float(getattr(depth, "inference_time_s", 0.0))
        return replace(
            observation,
            depth_m=depth.depth_m,
            depth_backend=depth.backend,
            depth_metric=depth.metric,
        )

    def _target_position(self, time_s: float) -> np.ndarray:
        if self.config.manual_control:
            self._integrate_manual_target_velocity(time_s)
            position = self.motion.position(time_s) + self._manual_target_offset
            if self.robot.detection_bounds is not None:
                lower, upper = self.robot.detection_bounds
                position = np.clip(
                    position,
                    np.asarray(lower, dtype=float),
                    np.asarray(upper, dtype=float),
                )
            return position
        return self.motion.position(time_s)

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
        target = site_position(
            self.scene.model, self.scene.data, self.scene.target_site_name
        )
        ee = frame_position(
            self.scene.model,
            self.scene.data,
            self.scene.ee_frame_type,
            self.scene.ee_frame_name,
            self.scene.ee_frame_offset,
        )
        midpoint = 0.55 * target + 0.45 * ee
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = midpoint
            viewer.cam.distance = 1.35
            viewer.cam.azimuth = 132.0
            viewer.cam.elevation = -24.0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = False
        self._viewer_camera_initialized = True

    def _keep_viewer_camera_free(self, viewer) -> None:
        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            return
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.fixedcamid = -1

    def _handle_key(self, keycode: int) -> None:
        try:
            import glfw
        except Exception:
            return
        if not self.config.manual_control:
            return
        speed = float(self.config.key_speed_mps)
        mapping = {
            glfw.KEY_LEFT: (0.0, speed, 0.0),
            glfw.KEY_RIGHT: (0.0, -speed, 0.0),
            glfw.KEY_UP: (speed, 0.0, 0.0),
            glfw.KEY_DOWN: (-speed, 0.0, 0.0),
            glfw.KEY_PERIOD: (0.0, 0.0, speed),
            glfw.KEY_COMMA: (0.0, 0.0, -speed),
            ord("."): (0.0, 0.0, speed),
            ord(","): (0.0, 0.0, -speed),
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
            if np.any(mask):
                color = np.zeros_like(image)
                color[:, :, 1] = 180
                image[mask] = cv2.addWeighted(image[mask], 0.55, color[mask], 0.45, 0)
        if detection is not None and detection.bbox_xyxy is not None:
            bbox = self._overlay_bbox(detection.bbox_xyxy, image.shape)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if detection is not None and detection.centroid_px is not None:
            center = self._overlay_point(detection.centroid_px, image.shape)
            if center is not None:
                cv2.drawMarker(
                    image, center, (255, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA
                )
        label = f"{self.detector_name} pending" if pending else f"{self.detector_name}"
        if detection is not None:
            label = f"{detection.backend} score={detection.score:.2f}"
            if detection.anchor_type != "unknown":
                label = f"{label} {detection.anchor_type}"
        if observation is not None:
            depth_label = f"depth={observation.depth_backend}"
            if not observation.depth_metric:
                depth_label = f"{depth_label} relative"
            cv2.putText(
                image,
                depth_label,
                (10, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            self.config.target,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def _record_dashboard(self, camera_overlay: np.ndarray | None) -> None:
        if self._recorder is None or camera_overlay is None:
            return
        state = self._last_servo_state
        evidence = self._grasp_evidence
        dashboard = self._dashboard.render(
            camera_overlay,
            DashboardTelemetry(
                robot=self.robot.name,
                target=self.target.name,
                servo_mode=(
                    self.config.controller.servo_mode
                    if state is None
                    else state.servo_mode
                ),
                actuator_mode=self.controller.actuator_mode,
                tracking_state=self._tracking_state.value,
                manipulation_state=(
                    self.config.controller.task.strip().upper()
                    if self._manipulation_state is ManipulationState.IDLE
                    else self._manipulation_state.value
                ),
                sim_time_s=float(self.scene.data.time),
                position_error_m=0.0 if state is None else state.position_error_m,
                image_error_px=0.0 if state is None else state.image_error_px,
                contact_force_n=0.0 if evidence is None else evidence.normal_force_n,
                policy_phase=(
                    None
                    if self._pick_place_policy is None
                    else self._pick_place_policy.phase.value
                ),
            ),
        )
        self._recorder.write(dashboard)

    @staticmethod
    def _overlay_bbox(
        bbox_xyxy: np.ndarray, image_shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int] | None:
        try:
            bbox = np.asarray(bbox_xyxy, dtype=float).reshape(4)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(bbox).all():
            return None
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        left = max(0, min(w - 1, int(np.floor(x1))))
        top = max(0, min(h - 1, int(np.floor(y1))))
        right = max(0, min(w - 1, int(np.ceil(x2))))
        bottom = max(0, min(h - 1, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    @staticmethod
    def _overlay_point(
        point_xy: np.ndarray, image_shape: tuple[int, int, int]
    ) -> tuple[int, int] | None:
        try:
            point = np.asarray(point_xy, dtype=float).reshape(2)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(point).all():
            return None
        h, w = image_shape[:2]
        x = max(0, min(w - 1, int(round(point[0]))))
        y = max(0, min(h - 1, int(round(point[1]))))
        return x, y

    def _update_viewer_overlay(self, viewer) -> None:
        if not self.config.camera_overlay or self._latest_overlay_bgr is None:
            return
        if not hasattr(viewer, "viewport") or not callable(
            getattr(viewer, "set_images", None)
        ):
            return
        viewport = viewer.viewport
        if viewport is None or viewport.width <= 0 or viewport.height <= 0:
            return
        width = min(
            640, max(300, int(viewport.width * self.config.overlay_width_fraction))
        )
        height = int(width * self.camera.height / self.camera.width)
        height = min(height, max(180, int(viewport.height * 0.46)))
        x = max(0, int(viewport.width - width - 12))
        y = max(0, int(viewport.height - height - 12))
        rect_key = (x, y, width, height)
        if self._latest_overlay_rgb is not None and self._overlay_rect_key == rect_key:
            return
        if self._latest_overlay_rgb is None or self._overlay_rect_key != rect_key:
            overlay = cv2.resize(
                self._latest_overlay_bgr, (width, height), interpolation=cv2.INTER_AREA
            )
            self._latest_overlay_rgb = overlay[:, :, ::-1].copy()
            self._overlay_rect_key = rect_key
        viewer.set_images(
            (mujoco.MjrRect(x, y, width, height), self._latest_overlay_rgb)
        )


def run_demo(config: DemoConfig) -> RunSummary:
    return VisualServoSimulation(config).run()
