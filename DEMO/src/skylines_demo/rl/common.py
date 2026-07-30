from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import CandidateLink, LinkAction, SchedulingAction, StepObservation, Vector3, VehicleReplay
from ..wireless.interfaces import LinkEvaluator

CANDIDATE_FEATURE_COUNT = 14
GLOBAL_FEATURE_COUNT = 5


@dataclass(frozen=True)
class ObservationScaling:
    max_distance_m: float
    max_rate_mbps: float


@dataclass(frozen=True)
class EncodingContext:
    top_k: int
    max_vehicle_count: int
    max_v2v_candidate_count: int
    terminal_steps: int
    packet_count: int
    age_tolerance_seconds: float
    autonomy_enabled: bool
    scaling: ObservationScaling


def rank_deliverable_candidates(observation: StepObservation) -> list[CandidateLink]:
    deliverable = [
        candidate for candidate in observation.candidate_links if candidate.delivered_packet_count > 0.0
    ]
    return sorted(deliverable, key=_candidate_sort_key)


def build_seeded_scheduling_action(
    observation: StepObservation,
    action_index: int,
    top_k: int,
    allow_multi_v2i_from_rsu: bool,
    max_active_links: int | None = None,
) -> SchedulingAction:
    if max_active_links is not None and max_active_links <= 0:
        return SchedulingAction()

    ranked_candidates = rank_deliverable_candidates(observation)
    seed_candidate = decode_seed_candidate(
        ranked_candidates=ranked_candidates,
        action_index=action_index,
        top_k=top_k,
    )
    if seed_candidate is None:
        return SchedulingAction()

    selected = [seed_candidate]
    used_vehicle_nodes = set(_vehicle_nodes_for_action(seed_candidate.action))
    v2i_used = seed_candidate.action.link_type == "v2i"
    remaining_candidates = [candidate for candidate in ranked_candidates if candidate != seed_candidate]

    for candidate in remaining_candidates:
        if max_active_links is not None and len(selected) >= max_active_links:
            break
        if candidate.action.link_type == "v2i" and v2i_used and not allow_multi_v2i_from_rsu:
            continue
        nodes = _vehicle_nodes_for_action(candidate.action)
        if any(node in used_vehicle_nodes for node in nodes):
            continue
        selected.append(candidate)
        used_vehicle_nodes.update(nodes)
        if candidate.action.link_type == "v2i":
            v2i_used = True

    return build_scheduling_action(selected)


def decode_seed_candidate(
    ranked_candidates: list[CandidateLink],
    action_index: int,
    top_k: int,
) -> CandidateLink | None:
    if action_index < 0 or action_index >= top_k + 1:
        return None
    if action_index == top_k:
        return None
    if action_index >= len(ranked_candidates):
        return None
    return ranked_candidates[action_index]


def build_action_mask(ranked_candidates: list[CandidateLink], top_k: int) -> np.ndarray:
    mask = np.zeros(top_k + 1, dtype=bool)
    candidate_count = min(len(ranked_candidates), top_k)
    if candidate_count > 0:
        mask[:candidate_count] = True
    mask[top_k] = True
    return mask


def encode_observation(
    observation: StepObservation,
    context: EncodingContext,
) -> np.ndarray:
    encoded = np.zeros(
        context.top_k * CANDIDATE_FEATURE_COUNT + GLOBAL_FEATURE_COUNT,
        dtype=np.float32,
    )
    ranked_candidates = rank_deliverable_candidates(observation)[: context.top_k]
    for slot, candidate in enumerate(ranked_candidates):
        base = slot * CANDIDATE_FEATURE_COUNT
        receiver_current_caoi = candidate.post_caoi_seconds + candidate.freshness_gain_seconds
        encoded[base + 0] = np.float32(1.0)
        encoded[base + 1] = np.float32(1.0 if candidate.action.link_type == "v2i" else 0.0)
        encoded[base + 2] = np.float32(1.0 if candidate.receiver_is_autonomous else 0.0)
        encoded[base + 3] = np.float32(1.0 if candidate.link_budget.los else 0.0)
        encoded[base + 4] = np.float32(1.0 if candidate.link_budget.vehicle_blocked else 0.0)
        encoded[base + 5] = _clip01(receiver_current_caoi / context.age_tolerance_seconds)
        encoded[base + 6] = _clip01(candidate.freshness_gain_seconds / context.age_tolerance_seconds)
        encoded[base + 7] = _clip01(candidate.priority_gain_seconds / context.age_tolerance_seconds)
        encoded[base + 8] = _clip01(candidate.delivered_packet_count / context.packet_count)
        encoded[base + 9] = _clip01(candidate.fresher_packet_count / context.packet_count)
        encoded[base + 10] = _clip01(candidate.relay_peer_count / max(context.max_vehicle_count, 1))
        encoded[base + 11] = _clip01(
            candidate.relay_peer_total_priority_gain_seconds / context.age_tolerance_seconds
        )
        encoded[base + 12] = _clip01(candidate.link_budget.rate_mbps / context.scaling.max_rate_mbps)
        encoded[base + 13] = _clip01(candidate.link_budget.distance_m / context.scaling.max_distance_m)

    global_offset = context.top_k * CANDIDATE_FEATURE_COUNT
    encoded[global_offset + 0] = _clip01(
        observation.step_index / max(context.terminal_steps - 1, 1)
    )
    encoded[global_offset + 1] = _clip01(
        len(observation.active_vehicles) / max(context.max_vehicle_count, 1)
    )
    encoded[global_offset + 2] = _clip01(
        observation.v2i_candidate_count / max(context.max_vehicle_count, 1)
    )
    encoded[global_offset + 3] = _clip01(
        observation.v2v_candidate_count / max(context.max_v2v_candidate_count, 1)
    )
    encoded[global_offset + 4] = np.float32(1.0 if context.autonomy_enabled else 0.0)
    return encoded


def scan_observation_scaling(
    replay: VehicleReplay,
    rsu_position: Vector3,
    link_evaluator: LinkEvaluator,
) -> ObservationScaling:
    max_distance_m = 1.0
    max_rate_mbps = 1.0

    for frame in replay.frames:
        snapshots = tuple(frame.vehicles.values())
        for receiver in snapshots:
            blockers = tuple(
                snapshot.position
                for snapshot in snapshots
                if snapshot.vehicle_id != receiver.vehicle_id
            )
            budget = link_evaluator.evaluate(
                transmitter_position=rsu_position,
                receiver_position=receiver.position,
                blockers=blockers,
            )
            max_distance_m = max(max_distance_m, budget.distance_m)
            max_rate_mbps = max(max_rate_mbps, budget.rate_mbps)

        for transmitter in snapshots:
            for receiver in snapshots:
                if transmitter.vehicle_id == receiver.vehicle_id:
                    continue
                blockers = tuple(
                    snapshot.position
                    for snapshot in snapshots
                    if snapshot.vehicle_id not in (transmitter.vehicle_id, receiver.vehicle_id)
                )
                budget = link_evaluator.evaluate(
                    transmitter_position=transmitter.position,
                    receiver_position=receiver.position,
                    blockers=blockers,
                )
                max_distance_m = max(max_distance_m, budget.distance_m)
                max_rate_mbps = max(max_rate_mbps, budget.rate_mbps)

    return ObservationScaling(
        max_distance_m=max_distance_m,
        max_rate_mbps=max_rate_mbps,
    )


def compute_mean_priority_weighted_caoi(observation: StepObservation) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for vehicle in observation.active_vehicles:
        weight = vehicle.priority_weight
        weighted_sum += vehicle.knowledge.caoi_seconds * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def build_scheduling_action(selected: list[CandidateLink]) -> SchedulingAction:
    v2i_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2i")
    v2v_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2v")
    return SchedulingAction(v2i_links=v2i_links, v2v_links=v2v_links)


def _candidate_sort_key(candidate: CandidateLink) -> tuple[float, float, float, float, int, str, int]:
    return (
        -candidate.priority_gain_seconds,
        -candidate.freshness_gain_seconds,
        -candidate.delivered_packet_count,
        -candidate.link_budget.rate_mbps,
        0 if candidate.action.link_type == "v2i" else 1,
        candidate.action.transmitter_id,
        candidate.action.receiver_id,
    )


def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
    nodes = [action.receiver_id]
    if action.link_type == "v2v" and action.transmitter_id.startswith("vehicle:"):
        nodes.append(int(action.transmitter_id.split(":", 1)[1]))
    return tuple(nodes)


def _clip01(value: float) -> np.float32:
    return np.float32(min(max(value, 0.0), 1.0))
