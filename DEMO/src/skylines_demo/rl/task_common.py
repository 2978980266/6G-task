from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import ResourceConfig
from ..types import (
    CandidateLink,
    LinkAction,
    SchedulingAction,
    StepObservation,
    Vector3,
    VehicleReplay,
)

TASK_CANDIDATE_SCALAR_FEATURE_COUNT = 20
TASK_GLOBAL_FEATURE_COUNT = 10


@dataclass(frozen=True)
class TaskObservationScaling:
    max_distance_m: float
    max_rate_mbps: float


@dataclass(frozen=True)
class TaskEncodingContext:
    top_k: int
    max_vehicle_count: int
    max_v2i_candidate_count: int
    max_v2v_candidate_count: int
    terminal_steps: int
    packet_count: int
    age_tolerance_seconds: float
    task_reward_base: float
    max_completed_tasks_per_vehicle: int
    max_resource_blocks_per_link: int
    autonomy_enabled: bool
    scaling: TaskObservationScaling

    @property
    def candidate_feature_count(self) -> int:
        return TASK_CANDIDATE_SCALAR_FEATURE_COUNT + self.packet_count

    @property
    def observation_feature_count(self) -> int:
        return (
            self.top_k * self.candidate_feature_count
            + TASK_GLOBAL_FEATURE_COUNT
        )


def rank_task_candidates(
    observation: StepObservation,
) -> list[CandidateLink]:
    deliverable = [
        candidate
        for candidate in observation.candidate_links
        if candidate.delivered_packet_count >= 1.0
        and candidate.selected_packet_ids
    ]
    return sorted(deliverable, key=_task_candidate_sort_key)


def build_task_scheduling_action(
    observation: StepObservation,
    action_indices: Sequence[int],
    top_k: int,
    resource_config: ResourceConfig,
    allow_multi_v2i_from_rsu: bool,
    max_active_links: int | None = None,
) -> SchedulingAction:
    if max_active_links is not None and max_active_links <= 0:
        return SchedulingAction()

    ranked_candidates = rank_task_candidates(observation)[:top_k]
    selected: list[CandidateLink] = []
    selected_indices: set[int] = set()
    used_vehicle_nodes: set[int] = set()
    used_rsu_blocks = 0
    used_v2v_blocks = 0
    used_rsu_rate = 0.0
    used_v2v_rate = 0.0
    v2i_used = False

    for raw_index in action_indices:
        action_index = int(raw_index)
        if action_index < 0 or action_index >= top_k:
            continue
        if action_index in selected_indices:
            continue
        if action_index >= len(ranked_candidates):
            continue
        if max_active_links is not None and len(selected) >= max_active_links:
            break

        candidate = ranked_candidates[action_index]
        if (
            candidate.action.link_type == "v2i"
            and v2i_used
            and not allow_multi_v2i_from_rsu
        ):
            continue

        nodes = _vehicle_nodes_for_action(candidate.action)
        if any(node in used_vehicle_nodes for node in nodes):
            continue

        if candidate.action.link_type == "v2i":
            next_blocks = used_rsu_blocks + candidate.resource_block_count
            next_rate = used_rsu_rate + candidate.link_budget.rate_mbps
            if next_blocks > resource_config.rsu_resource_block_count:
                continue
            if (
                next_rate
                > resource_config.rsu_total_rate_limit_mbps + 1e-9
            ):
                continue
            used_rsu_blocks = next_blocks
            used_rsu_rate = next_rate
            v2i_used = True
        else:
            next_blocks = used_v2v_blocks + candidate.resource_block_count
            next_rate = used_v2v_rate + candidate.link_budget.rate_mbps
            if next_blocks > resource_config.v2v_resource_block_count:
                continue
            if (
                next_rate
                > resource_config.v2v_total_rate_limit_mbps + 1e-9
            ):
                continue
            used_v2v_blocks = next_blocks
            used_v2v_rate = next_rate

        selected.append(candidate)
        selected_indices.add(action_index)
        used_vehicle_nodes.update(nodes)

    return SchedulingAction(
        v2i_links=tuple(
            candidate.action
            for candidate in selected
            if candidate.action.link_type == "v2i"
        ),
        v2v_links=tuple(
            candidate.action
            for candidate in selected
            if candidate.action.link_type == "v2v"
        ),
    )


def build_task_action_mask(
    ranked_candidates: Sequence[CandidateLink],
    top_k: int,
) -> np.ndarray:
    mask = np.zeros(top_k + 1, dtype=bool)
    mask[: min(len(ranked_candidates), top_k)] = True
    mask[top_k] = True
    return mask


def encode_task_observation(
    observation: StepObservation,
    context: TaskEncodingContext,
) -> np.ndarray:
    encoded = np.zeros(
        context.observation_feature_count,
        dtype=np.float32,
    )
    ranked_candidates = rank_task_candidates(observation)[: context.top_k]
    candidate_feature_count = context.candidate_feature_count

    for slot, candidate in enumerate(ranked_candidates):
        base = slot * candidate_feature_count
        receiver_current_caoi = (
            candidate.post_caoi_seconds + candidate.freshness_gain_seconds
        )
        required_count = max(candidate.task_required_packet_count, 1)
        delivered_count = max(int(candidate.delivered_packet_count), 1)
        link_block_count = (
            observation.rsu_resource_block_count
            if candidate.action.link_type == "v2i"
            else observation.v2v_resource_block_count
        )

        scalar_features = (
            1.0,
            1.0 if candidate.action.link_type == "v2i" else 0.0,
            1.0 if candidate.receiver_is_autonomous else 0.0,
            1.0 if candidate.link_budget.los else 0.0,
            1.0 if candidate.link_budget.vehicle_blocked else 0.0,
            receiver_current_caoi / context.age_tolerance_seconds,
            candidate.freshness_gain_seconds
            / context.age_tolerance_seconds,
            candidate.priority_gain_seconds
            / context.age_tolerance_seconds,
            candidate.task_reward_gain / context.task_reward_base,
            candidate.task_progress_gain / required_count,
            candidate.task_missing_packet_count / context.packet_count,
            candidate.task_required_packet_count / context.packet_count,
            candidate.delivered_packet_count / context.packet_count,
            candidate.complete_packet_capacity / context.packet_count,
            candidate.relay_peer_count / context.max_vehicle_count,
            candidate.relay_peer_total_priority_gain_seconds
            / context.age_tolerance_seconds,
            candidate.link_budget.rate_mbps
            / context.scaling.max_rate_mbps,
            candidate.link_budget.distance_m
            / context.scaling.max_distance_m,
            candidate.resource_block_count / max(link_block_count, 1),
            candidate.task_progress_gain / delivered_count,
        )
        for offset, value in enumerate(scalar_features):
            encoded[base + offset] = _clip01(value)

        packet_mask_offset = base + TASK_CANDIDATE_SCALAR_FEATURE_COUNT
        for packet_id in candidate.selected_packet_ids:
            packet_index = packet_id - 1
            if 0 <= packet_index < context.packet_count:
                encoded[packet_mask_offset + packet_index] = np.float32(1.0)

    active_vehicle_count = len(observation.active_vehicles)
    active_task_count = 0
    total_missing_packets = 0
    total_required_packets = 0
    total_completed_tasks = 0
    cooling_vehicle_count = 0
    weighted_caoi_sum = 0.0
    priority_weight_sum = 0.0

    for vehicle in observation.active_vehicles:
        task_state = vehicle.task_state
        if task_state.active_task_id is not None:
            active_task_count += 1
        missing_count = sum(
            not vehicle.knowledge.packet_updated_flags[packet_id - 1]
            for packet_id in task_state.required_packet_ids
        )
        total_missing_packets += missing_count
        total_required_packets += len(task_state.required_packet_ids)
        total_completed_tasks += task_state.completed_task_count
        if (
            task_state.active_task_id is None
            and not task_state.can_request_new_task
            and task_state.completed_task_count
            < context.max_completed_tasks_per_vehicle
        ):
            cooling_vehicle_count += 1
        weighted_caoi_sum += (
            vehicle.knowledge.caoi_seconds * vehicle.priority_weight
        )
        priority_weight_sum += vehicle.priority_weight

    mean_weighted_caoi = (
        weighted_caoi_sum / priority_weight_sum
        if priority_weight_sum > 0.0
        else 0.0
    )
    max_completed_tasks = (
        active_vehicle_count * context.max_completed_tasks_per_vehicle
    )
    global_features = (
        observation.step_index / max(context.terminal_steps - 1, 1),
        active_vehicle_count / context.max_vehicle_count,
        observation.v2i_candidate_count
        / context.max_v2i_candidate_count,
        observation.v2v_candidate_count
        / context.max_v2v_candidate_count,
        mean_weighted_caoi / context.age_tolerance_seconds,
        active_task_count / max(active_vehicle_count, 1),
        total_missing_packets / max(total_required_packets, 1),
        total_completed_tasks / max(max_completed_tasks, 1),
        cooling_vehicle_count / max(active_vehicle_count, 1),
        1.0 if context.autonomy_enabled else 0.0,
    )
    global_offset = context.top_k * candidate_feature_count
    for offset, value in enumerate(global_features):
        encoded[global_offset + offset] = _clip01(value)
    return encoded


def build_task_encoding_context(
    *,
    top_k: int,
    replay: VehicleReplay,
    rsu_position: Vector3,
    terminal_steps: int,
    packet_count: int,
    age_tolerance_seconds: float,
    task_reward_base: float,
    max_completed_tasks_per_vehicle: int,
    resource_config: ResourceConfig,
    autonomy_enabled: bool,
) -> TaskEncodingContext:
    max_vehicle_count = max(len(replay.vehicle_ids), 1)
    max_link_variants = max(resource_config.max_resource_blocks_per_link, 1)
    return TaskEncodingContext(
        top_k=top_k,
        max_vehicle_count=max_vehicle_count,
        max_v2i_candidate_count=max(
            max_vehicle_count * max_link_variants,
            1,
        ),
        max_v2v_candidate_count=max(
            max_vehicle_count
            * (max_vehicle_count - 1)
            * max_link_variants,
            1,
        ),
        terminal_steps=max(terminal_steps, 1),
        packet_count=max(packet_count, 1),
        age_tolerance_seconds=max(age_tolerance_seconds, 1e-9),
        task_reward_base=max(task_reward_base, 1e-9),
        max_completed_tasks_per_vehicle=max(
            max_completed_tasks_per_vehicle,
            1,
        ),
        max_resource_blocks_per_link=max_link_variants,
        autonomy_enabled=autonomy_enabled,
        scaling=scan_task_observation_scaling(
            replay=replay,
            rsu_position=rsu_position,
            resource_config=resource_config,
        ),
    )


def scan_task_observation_scaling(
    replay: VehicleReplay,
    rsu_position: Vector3,
    resource_config: ResourceConfig,
) -> TaskObservationScaling:
    max_distance_m = 1.0
    for frame in replay.frames:
        snapshots = tuple(frame.vehicles.values())
        for receiver in snapshots:
            max_distance_m = max(
                max_distance_m,
                rsu_position.distance_to(receiver.position),
            )
        for transmitter in snapshots:
            for receiver in snapshots:
                if transmitter.vehicle_id == receiver.vehicle_id:
                    continue
                max_distance_m = max(
                    max_distance_m,
                    transmitter.position.distance_to(receiver.position),
                )
    return TaskObservationScaling(
        max_distance_m=max_distance_m,
        max_rate_mbps=max(
            resource_config.vehicle_link_rate_limit_mbps,
            1e-9,
        ),
    )


def _task_candidate_sort_key(
    candidate: CandidateLink,
) -> tuple[object, ...]:
    return (
        -candidate.task_reward_gain,
        -candidate.task_progress_gain,
        -candidate.relay_peer_total_priority_gain_seconds,
        -candidate.priority_gain_seconds,
        -candidate.delivered_packet_count,
        candidate.resource_block_count,
        -candidate.link_budget.rate_mbps,
        0 if candidate.action.link_type == "v2i" else 1,
        candidate.action.transmitter_id,
        candidate.action.receiver_id,
        candidate.selected_packet_ids,
    )


def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
    nodes = [action.receiver_id]
    if (
        action.link_type == "v2v"
        and action.transmitter_id.startswith("vehicle:")
    ):
        nodes.append(int(action.transmitter_id.split(":", 1)[1]))
    return tuple(nodes)


def _clip01(value: float) -> np.float32:
    return np.float32(min(max(value, 0.0), 1.0))
