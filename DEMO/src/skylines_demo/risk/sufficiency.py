from __future__ import annotations

from dataclasses import dataclass

from ..config import RiskCommConfig
from ..types import DownlinkAction, RiskObservation, RiskTaskView
from .model import own_schedule_eta, predict_pair


@dataclass(frozen=True)
class ReceiverDeliveryValue:
    receiver_id: int
    object_id: int
    value: float
    urgency: float
    freshness_gain: float
    distance_m: float
    eligible: bool = False
    task_id: str | None = None
    decision_step: int | None = None


def receiver_version(observation: RiskObservation, receiver_id: int, object_id: int) -> int:
    return max((version for receiver, obj, version in observation.receiver_known_observation_steps
                if receiver == receiver_id and obj == object_id), default=-1)


def estimate_receiver_delivery_value(
    observation: RiskObservation,
    candidate: DownlinkAction,
    config: RiskCommConfig | None = None,
) -> ReceiverDeliveryValue:
    """Filter legal updates for future tasks before scoring with the fixed model."""
    config = config or RiskCommConfig()
    snapshot = next((s for s in observation.rsu_snapshots if s.object_id == candidate.object_id), None)
    gain = float(max(snapshot.observation_step - receiver_version(
        observation, candidate.receiver_id, candidate.object_id), 0)) if snapshot else 0.0
    eligible_tasks = []
    if snapshot is not None and gain > 0 and candidate.object_id != candidate.receiver_id:
        for task in observation.tasks:
            if task.vehicle_id != candidate.receiver_id or task.decision_step <= observation.step_index:
                continue
            distance = predicted_object_distance_to_task(observation, candidate, task, config)
            if distance <= config.conflict_cluster_radius_m + 1e-9:
                urgency = 1.0 / (1.0 + task.decision_step - observation.step_index)
                eligible_tasks.append((urgency + 1.0 / (1.0 + distance) + gain, urgency, distance, task))
    if not eligible_tasks:
        return ReceiverDeliveryValue(candidate.receiver_id, candidate.object_id, 0.0, 0.0,
                                     gain, float("inf"))
    value, urgency, distance, task = min(eligible_tasks, key=lambda item: (-item[0], item[3].task_id))
    return ReceiverDeliveryValue(candidate.receiver_id, candidate.object_id, value, urgency, gain,
                                 distance, True, task.task_id, task.decision_step)


def predicted_object_distance_to_task(
    observation: RiskObservation,
    candidate: DownlinkAction,
    task: RiskTaskView,
    config: RiskCommConfig | None = None,
) -> float:
    config = config or RiskCommConfig()
    snapshot = next((s for s in observation.rsu_snapshots if s.object_id == candidate.object_id), None)
    if snapshot is None:
        return float("inf")
    receiver = next((v for v in observation.active_vehicles
                     if v.vehicle_id == candidate.receiver_id), None)
    if receiver is None:
        return float("inf")
    prediction = predict_pair(
        receiver.self_snapshot, snapshot, task.conflict_point, observation.elapsed_seconds,
        observation.step_times[:observation.step_index + 1], config,
        own_schedule_eta(config, task.crossing_step, observation.step_index, observation.delta_seconds),
    )
    return (prediction.distance_m if prediction.risk and prediction.distance_m is not None
            else float("inf"))


def prediction_triggered_task(
    observation: RiskObservation, candidate: DownlinkAction, config: RiskCommConfig,
) -> ReceiverDeliveryValue | None:
    """A forecast may change priority only for an otherwise eligible delivery."""
    value = estimate_receiver_delivery_value(observation, candidate, config)
    if not config.link_prediction or not value.eligible:
        return None
    forecast = next((f for receiver, obj, f in observation.link_forecasts
                     if receiver == candidate.receiver_id and obj == candidate.object_id), None)
    if (forecast is None or not forecast.available or forecast.first_unreachable_step is None
            or value.decision_step is None or forecast.first_unreachable_step >= value.decision_step):
        return None
    return value


def rank_downlink_candidates(
    observation: RiskObservation, config: RiskCommConfig | None = None,
) -> tuple[DownlinkAction, ...]:
    values = {c: estimate_receiver_delivery_value(observation, c, config)
              for c in dict.fromkeys(observation.downlink_candidates)}
    return tuple(sorted((c for c, v in values.items() if v.eligible),
                        key=lambda c: (-values[c].value, c.receiver_id, c.object_id)))
