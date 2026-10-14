from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def horizontal_distance_to(self, other: "Vector3") -> float:
        return math.hypot(self.x - other.x, self.z - other.z)

    def distance_to(self, other: "Vector3") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass(frozen=True)
class RiskSnapshot:
    object_id: int
    source_id: int
    observation_step: int
    x: float
    y: float
    z: float
    speed: float
    heading_deg: float
    velocity_x: float | None
    velocity_z: float | None
    received_step: int | None
    velocity_previous_observation_step: int | None = None
    velocity_basis: str | None = None

    @property
    def position(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    def age_at(self, current_step: int) -> int:
        if current_step < self.observation_step:
            raise ValueError("current_step cannot precede observation_step.")
        return current_step - self.observation_step


@dataclass(frozen=True)
class LinkForecast:
    receiver_id: int
    current_step: int
    predicted_steps: tuple[int, ...]
    reachable_steps: tuple[int, ...]
    first_unreachable_step: int | None
    available: bool


@dataclass(frozen=True)
class UplinkAction:
    sender_id: int
    object_id: int


@dataclass(frozen=True)
class DownlinkAction:
    receiver_id: int
    object_id: int


@dataclass(frozen=True)
class RelayAction:
    sender_id: int
    receiver_id: int
    object_id: int


@dataclass(frozen=True)
class RiskSchedulingAction:
    uplinks: tuple[UplinkAction, ...] = ()
    downlinks: tuple[DownlinkAction, ...] = ()
    relays: tuple[RelayAction, ...] = ()

    @property
    def transmitting_vehicle_ids(self) -> tuple[int, ...]:
        return tuple(action.sender_id for action in self.uplinks) + tuple(
            action.sender_id for action in self.relays
        )

    @property
    def receiving_vehicle_ids(self) -> tuple[int, ...]:
        return tuple(action.receiver_id for action in self.downlinks) + tuple(
            action.receiver_id for action in self.relays
        )


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: int
    position: Vector3
    speed: float
    angle_x: float
    angle_y: float
    prefab_name: str


@dataclass(frozen=True)
class RiskTaskView:
    task_id: str
    vehicle_id: int
    conflict_point_id: int
    conflict_point: Vector3
    crossing_step: int
    decision_step: int


@dataclass(frozen=True)
class RiskVehicleState:
    vehicle_id: int
    snapshot: VehicleSnapshot
    visible_object_ids: tuple[int, ...]
    known_object_ids: tuple[int, ...]
    data_plane_object_ids: tuple[int, ...]
    caoi_steps: float | None
    # Own causal motion is control information, never a forwardable object.
    self_snapshot: RiskSnapshot | None = None


@dataclass(frozen=True)
class RiskObservation:
    step_index: int
    simulation_frame: int
    game_time: str
    record_time_utc: str
    elapsed_seconds: float
    delta_seconds: float
    active_vehicles: tuple[RiskVehicleState, ...]
    tasks: tuple[RiskTaskView, ...]
    uplink_candidates: tuple[UplinkAction, ...]
    downlink_candidates: tuple[DownlinkAction, ...]
    relay_candidates: tuple[RelayAction, ...]
    rsu_known_object_ids: tuple[int, ...]
    # The following fields make the information available to policies explicit.
    # They are deliberately snapshots of the public observation, rather than
    # references to a communication environment's private caches.
    rsu_snapshots: tuple[RiskSnapshot, ...] = ()
    receiver_known_observation_steps: tuple[tuple[int, int, int], ...] = ()
    last_sent_snapshots: tuple[tuple[int, RiskSnapshot], ...] = ()
    step_times: tuple[float, ...] = ()
    rsu_position: Vector3 = Vector3(0.0, 0.0, 0.0)
    link_forecasts: tuple[tuple[int, int, LinkForecast], ...] = ()
    rsu_visible_object_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BuildingSnapshot:
    building_id: int
    position: Vector3
    angle_y: float
    width_cells: int
    length_cells: int
    size_x: float
    size_y: float
    size_z: float
    center_offset_x: float
    center_offset_y: float
    center_offset_z: float
    min_y: float
    max_y: float
    prefab_name: str


@dataclass(frozen=True)
class FrameRecord:
    step_index: int
    simulation_frame: int
    game_time: str
    record_time_utc: str
    session_seconds: float | None
    elapsed_seconds: float
    delta_seconds: float
    vehicles: dict[int, VehicleSnapshot]


@dataclass(frozen=True)
class VehicleReplay:
    frames: tuple[FrameRecord, ...]

    @property
    def vehicle_ids(self) -> tuple[int, ...]:
        seen: set[int] = set()
        for frame in self.frames:
            seen.update(frame.vehicles.keys())
        return tuple(sorted(seen))

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class LinkBudget:
    distance_m: float
    los: bool
    path_loss_db: float
    snr_db: float
    rate_mbps: float
    reachable: bool
    vehicle_blocked: bool = False
    vehicle_blocker_count: int = 0
    vehicle_blockage_loss_db: float = 0.0
    allocated_bandwidth_hz: float = 0.0
    raw_rate_mbps: float = 0.0


@dataclass(frozen=True)
class VehicleKnowledge:
    vehicle_id: int
    packet_timestamps: tuple[float, ...]
    packet_updated_flags: tuple[bool, ...]
    caoi_seconds: float

    @property
    def freshest_timestamp(self) -> float | None:
        if not self.packet_timestamps:
            return None
        return max(self.packet_timestamps)

    @property
    def aoi_seconds(self) -> float:
        return self.caoi_seconds


@dataclass(frozen=True)
class TaskTemplate:
    task_id: str
    required_packet_ids: tuple[int, ...]


@dataclass(frozen=True)
class VehicleTaskState:
    active_task_id: str | None = None
    required_packet_ids: tuple[int, ...] = ()
    completed_task_ids: tuple[str, ...] = ()
    cooldown_until_step: int = 0
    can_request_new_task: bool = False

    @property
    def completed_task_count(self) -> int:
        return len(self.completed_task_ids)


@dataclass(frozen=True)
class VehicleState:
    snapshot: VehicleSnapshot
    knowledge: VehicleKnowledge
    is_autonomous: bool = False
    priority_weight: float = 1.0
    task_state: VehicleTaskState = VehicleTaskState()


@dataclass(frozen=True)
class LinkAction:
    link_type: Literal["v2i", "v2v"]
    transmitter_id: str
    receiver_id: int
    transmitter_position: Vector3
    receiver_position: Vector3
    resource_block_count: int = 1
    packet_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SchedulingAction:
    v2i_links: tuple[LinkAction, ...] = ()
    v2v_links: tuple[LinkAction, ...] = ()

    @property
    def links(self) -> tuple[LinkAction, ...]:
        return self.v2i_links + self.v2v_links


@dataclass(frozen=True)
class ControlBroadcastAction:
    transmitter_id: str
    receiver_vehicle_ids: tuple[int, ...]
    reliable: bool = True


@dataclass(frozen=True)
class CandidateLink:
    action: LinkAction
    link_budget: LinkBudget
    transmitter_packet_timestamps: tuple[float, ...] | None
    transmitter_packet_updated_flags: tuple[bool, ...] | None
    receiver_packet_timestamps: tuple[float, ...]
    receiver_packet_updated_flags: tuple[bool, ...]
    fresher_packet_count: int
    expected_packet_count: float
    delivered_packet_count: float
    post_receiver_packet_timestamps: tuple[float, ...]
    post_receiver_packet_updated_flags: tuple[bool, ...]
    post_caoi_seconds: float
    freshness_gain_seconds: float
    receiver_is_autonomous: bool = False
    receiver_priority_weight: float = 1.0
    priority_gain_seconds: float = 0.0
    relay_peer_count: int = 0
    relay_peer_total_gain_seconds: float = 0.0
    relay_peer_total_priority_gain_seconds: float = 0.0
    transferable_packet_ids: tuple[int, ...] = ()
    selected_packet_ids: tuple[int, ...] = ()
    resource_block_count: int = 1
    allocated_bandwidth_hz: float = 0.0
    complete_packet_capacity: int = 0
    task_progress_gain: int = 0
    task_reward_gain: float = 0.0
    task_required_packet_count: int = 0
    task_missing_packet_count: int = 0

    @property
    def transmitter_freshest_timestamp(self) -> float | None:
        if not self.transmitter_packet_timestamps:
            return None
        return max(self.transmitter_packet_timestamps)

    @property
    def receiver_freshest_timestamp(self) -> float | None:
        if not self.receiver_packet_timestamps:
            return None
        return max(self.receiver_packet_timestamps)


@dataclass(frozen=True)
class StepObservation:
    step_index: int
    simulation_frame: int
    game_time: str
    record_time_utc: str
    elapsed_seconds: float
    delta_seconds: float
    active_vehicles: tuple[VehicleState, ...]
    candidate_v2i_links: tuple[CandidateLink, ...]
    candidate_v2v_links: tuple[CandidateLink, ...]
    rsu_resource_block_count: int = 0
    v2v_resource_block_count: int = 0

    @property
    def candidate_links(self) -> tuple[CandidateLink, ...]:
        return self.candidate_v2i_links + self.candidate_v2v_links

    @property
    def v2i_candidate_count(self) -> int:
        return len(self.candidate_v2i_links)

    @property
    def v2v_candidate_count(self) -> int:
        return len(self.candidate_v2v_links)

    @property
    def autonomous_vehicle_count(self) -> int:
        return sum(vehicle.is_autonomous for vehicle in self.active_vehicles)

    @property
    def human_vehicle_count(self) -> int:
        return len(self.active_vehicles) - self.autonomous_vehicle_count


@dataclass(frozen=True)
class StepResult:
    observation: StepObservation
    control_broadcast: ControlBroadcastAction
    chosen_v2i_links: tuple[CandidateLink, ...]
    chosen_v2v_links: tuple[CandidateLink, ...]
    successful_v2i_links: tuple[CandidateLink, ...]
    successful_v2v_links: tuple[CandidateLink, ...]
    failed_v2i_links: tuple[CandidateLink, ...]
    failed_v2v_links: tuple[CandidateLink, ...]
    invalid_action: bool
    mean_caoi_seconds: float
    mean_autonomous_caoi_seconds: float
    mean_human_caoi_seconds: float
    mean_priority_weighted_caoi_seconds: float
    max_caoi_seconds: float
    age_violation_count: int
    next_observation: StepObservation | None
    task_reward: float = 0.0
    completed_tasks: tuple["TaskCompletion", ...] = ()
    delivered_packet_count: int = 0
    used_rsu_resource_blocks: int = 0
    used_v2v_resource_blocks: int = 0

    @property
    def chosen_links(self) -> tuple[CandidateLink, ...]:
        return self.chosen_v2i_links + self.chosen_v2v_links

    @property
    def successful_links(self) -> tuple[CandidateLink, ...]:
        return self.successful_v2i_links + self.successful_v2v_links

    @property
    def failed_links(self) -> tuple[CandidateLink, ...]:
        return self.failed_v2i_links + self.failed_v2v_links

    @property
    def chosen_candidate(self) -> CandidateLink | None:
        return self.chosen_links[0] if self.chosen_links else None

    @property
    def success(self) -> bool:
        return bool(self.successful_links)

    @property
    def empty_action(self) -> bool:
        return not self.invalid_action and not self.chosen_links

    @property
    def mean_aoi_seconds(self) -> float:
        return self.mean_caoi_seconds

    @property
    def mean_autonomous_aoi_seconds(self) -> float:
        return self.mean_autonomous_caoi_seconds

    @property
    def mean_human_aoi_seconds(self) -> float:
        return self.mean_human_caoi_seconds

    @property
    def mean_priority_weighted_aoi_seconds(self) -> float:
        return self.mean_priority_weighted_caoi_seconds

    @property
    def max_aoi_seconds(self) -> float:
        return self.max_caoi_seconds


@dataclass(frozen=True)
class TaskCompletion:
    vehicle_id: int
    task_id: str
    required_packet_ids: tuple[int, ...]
    mean_packet_aoi_seconds: float
    reward: float
