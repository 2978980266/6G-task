from __future__ import annotations

from collections.abc import Iterable

from ..config import RiskCommConfig
from ..risk.sufficiency import predicted_object_distance_to_task
from ..types import (
    DownlinkAction,
    RiskObservation,
    RiskSchedulingAction,
    RiskTaskView,
    UplinkAction,
)


def _candidate_downlinks(observation: RiskObservation) -> tuple[DownlinkAction, ...]:
    return tuple(dict.fromkeys(observation.downlink_candidates))


def _candidate_uplinks(observation: RiskObservation) -> tuple[UplinkAction, ...]:
    return tuple(dict.fromkeys(observation.uplink_candidates))


def _build_feasible_action(
    observation: RiskObservation,
    config: RiskCommConfig,
    *,
    downlinks: Iterable[DownlinkAction],
    uplinks: Iterable[UplinkAction],
    downlink_limit: int | None = None,
    uplink_limit: int | None = None,
) -> RiskSchedulingAction:
    """Build a legal action from ranked public candidates.

    The environment remains the final authority on validity.  This helper only
    handles the policy-side, deterministic one-vehicle-per-step constraint and
    never manufactures a candidate that was absent from the observation.
    """

    candidate_downlinks = set(observation.downlink_candidates)
    candidate_uplinks = set(observation.uplink_candidates)
    maximum_downlinks = (
        config.downlink_slots_per_step
        if downlink_limit is None
        else downlink_limit
    )
    maximum_uplinks = (
        config.uplink_slots_per_step if uplink_limit is None else uplink_limit
    )

    selected_downlinks: list[DownlinkAction] = []
    selected_uplinks: list[UplinkAction] = []
    occupied: set[int] = set()

    for downlink_candidate in downlinks:
        if len(selected_downlinks) >= maximum_downlinks:
            break
        if downlink_candidate not in candidate_downlinks:
            continue
        if downlink_candidate.receiver_id in occupied:
            continue
        selected_downlinks.append(downlink_candidate)
        occupied.add(downlink_candidate.receiver_id)

    for uplink_candidate in uplinks:
        if len(selected_uplinks) >= maximum_uplinks:
            break
        if uplink_candidate not in candidate_uplinks:
            continue
        if uplink_candidate.sender_id in occupied:
            continue
        selected_uplinks.append(uplink_candidate)
        occupied.add(uplink_candidate.sender_id)

    return RiskSchedulingAction(
        uplinks=tuple(selected_uplinks),
        downlinks=tuple(selected_downlinks),
    )


def _rsu_snapshot_map(observation: RiskObservation):
    return {snapshot.object_id: snapshot for snapshot in observation.rsu_snapshots}


def select_unlimited_action(
    observation: RiskObservation,
    config: RiskCommConfig,
) -> RiskSchedulingAction:
    """Select every currently feasible candidate for slot calibration."""

    return _build_feasible_action(
        observation,
        config,
        downlinks=_candidate_downlinks(observation),
        uplinks=_candidate_uplinks(observation),
        downlink_limit=len(observation.downlink_candidates),
        uplink_limit=len(observation.uplink_candidates),
    )


def select_age_action(
    observation: RiskObservation,
    config: RiskCommConfig,
) -> RiskSchedulingAction:
    """Prioritize the RSU snapshots with the largest public age."""

    snapshots = _rsu_snapshot_map(observation)

    def age_key(candidate: DownlinkAction) -> tuple[int, int, int]:
        snapshot = snapshots.get(candidate.object_id)
        age = (
            observation.step_index - snapshot.observation_step
            if snapshot is not None
            else -1
        )
        return (-age, candidate.receiver_id, candidate.object_id)

    downlinks = sorted(_candidate_downlinks(observation), key=age_key)
    uplinks = sorted(
        _candidate_uplinks(observation),
        key=lambda item: (item.sender_id, item.object_id),
    )
    return _build_feasible_action(
        observation,
        config,
        downlinks=downlinks,
        uplinks=uplinks,
    )


def select_event_trigger_action(
    observation: RiskObservation,
    config: RiskCommConfig,
) -> RiskSchedulingAction:
    """Send only RSU updates that moved beyond the configured trigger."""

    snapshots = _rsu_snapshot_map(observation)
    last_sent = {
        (receiver_id, snapshot.object_id): snapshot
        for receiver_id, snapshot in observation.last_sent_snapshots
    }

    def should_send(candidate: DownlinkAction) -> bool:
        current = snapshots.get(candidate.object_id)
        previous = last_sent.get((candidate.receiver_id, candidate.object_id))
        if current is None or previous is None:
            return True
        displacement = current.position.horizontal_distance_to(previous.position)
        return displacement >= config.event_trigger_displacement_m

    downlinks = [
        candidate
        for candidate in _candidate_downlinks(observation)
        if should_send(candidate)
    ]
    uplinks = sorted(
        _candidate_uplinks(observation),
        key=lambda item: (item.sender_id, item.object_id),
    )
    return _build_feasible_action(
        observation,
        config,
        downlinks=downlinks,
        uplinks=uplinks,
    )


def select_deadline_action(
    observation: RiskObservation,
    config: RiskCommConfig,
) -> RiskSchedulingAction:
    """Prioritize receivers whose observed task decisions are most urgent."""

    vehicle_positions = {
        state.vehicle_id: state.snapshot.position
        for state in observation.active_vehicles
    }
    tasks_by_vehicle: dict[int, list[RiskTaskView]] = {}
    for task in observation.tasks:
        tasks_by_vehicle.setdefault(task.vehicle_id, []).append(task)

    def deadline_key(candidate: DownlinkAction) -> tuple[float, float, int, int]:
        tasks = tasks_by_vehicle.get(candidate.receiver_id, [])
        if not tasks:
            return (float("inf"), float("inf"), candidate.receiver_id, candidate.object_id)
        task = min(tasks, key=lambda item: (item.decision_step, item.task_id))
        distance = vehicle_positions[candidate.receiver_id].horizontal_distance_to(
            task.conflict_point
        )
        return (
            float(task.decision_step),
            distance,
            candidate.receiver_id,
            candidate.object_id,
        )

    downlinks = sorted(_candidate_downlinks(observation), key=deadline_key)
    uplinks = sorted(
        _candidate_uplinks(observation),
        key=lambda item: (item.sender_id, item.object_id),
    )
    return _build_feasible_action(
        observation,
        config,
        downlinks=downlinks,
        uplinks=uplinks,
    )


def select_relation_action(
    observation: RiskObservation, config: RiskCommConfig, *, related: bool,
) -> RiskSchedulingAction:
    """Fixed geometric comparison policy, using public snapshots and task maps."""
    def key(candidate: DownlinkAction) -> tuple[float, int, int]:
        distance = min((predicted_object_distance_to_task(observation, candidate, task, config)
                        for task in observation.tasks if task.vehicle_id == candidate.receiver_id),
                       default=float("inf"))
        return (distance if related else -distance, candidate.receiver_id, candidate.object_id)

    return _build_feasible_action(
        observation, config,
        downlinks=sorted(observation.downlink_candidates, key=key),
        uplinks=sorted(observation.uplink_candidates, key=lambda c: (c.sender_id, c.object_id)),
    )


__all__ = [
    "select_age_action",
    "select_deadline_action",
    "select_event_trigger_action",
    "select_unlimited_action",
]
