from __future__ import annotations

from dataclasses import replace

from ..config import RiskCommConfig
from ..risk.sufficiency import receiver_version
from ..types import RelayAction, RiskObservation, RiskSchedulingAction
from .risk_receiver_policy import select_receiver_value_action


def select_relay_action(
    observation: RiskObservation,
    config: RiskCommConfig,
) -> tuple[RelayAction, ...]:
    """Choose relay candidates using only public receiver/task value."""

    if not config.enable_v2v_relay:
        return ()

    def score(candidate: RelayAction) -> tuple[int, int, int, int, int] | None:
        tasks = [task for task in observation.tasks
                 if task.vehicle_id == candidate.receiver_id
                 and task.decision_step > observation.step_index]
        source_version = receiver_version(observation, candidate.sender_id, candidate.object_id)
        target_version = receiver_version(observation, candidate.receiver_id, candidate.object_id)
        if not tasks or source_version <= target_version or candidate.object_id == candidate.receiver_id:
            return None
        # Only ID/version summaries are public.  Sender motion state is unknown;
        # RSU's snapshot must never stand in for the actual relay version.
        remaining = min(task.decision_step for task in tasks) - observation.step_index
        return (remaining, -(source_version - target_version), candidate.receiver_id,
                candidate.sender_id, candidate.object_id)

    scores = {candidate: value for candidate in observation.relay_candidates
              if (value := score(candidate)) is not None}
    selected: list[RelayAction] = []
    occupied: set[int] = set()
    for candidate in sorted(scores, key=lambda c: scores[c]):
        if len(selected) >= config.relay_slots_per_step:
            break
        if candidate.sender_id in occupied or candidate.receiver_id in occupied:
            continue
        selected.append(candidate)
        occupied.update((candidate.sender_id, candidate.receiver_id))
    return tuple(selected)


def select_receiver_value_with_relay_action(
    observation: RiskObservation,
    config: RiskCommConfig,
) -> RiskSchedulingAction:
    """Extend receiver-value scheduling with the explicit relay path."""

    relays = list(select_relay_action(observation, config))
    occupied = {
        endpoint
        for relay in relays
        for endpoint in (relay.sender_id, relay.receiver_id)
    }
    remaining_observation = replace(
        observation,
        uplink_candidates=tuple(
            candidate
            for candidate in observation.uplink_candidates
            if candidate.sender_id not in occupied
        ),
        downlink_candidates=tuple(
            candidate
            for candidate in observation.downlink_candidates
            if candidate.receiver_id not in occupied
        ),
    )
    base = select_receiver_value_action(remaining_observation, config)
    return RiskSchedulingAction(
        uplinks=base.uplinks,
        downlinks=base.downlinks,
        relays=tuple(relays),
    )


__all__ = [
    "select_receiver_value_with_relay_action",
    "select_relay_action",
]
