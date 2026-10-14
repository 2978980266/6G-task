from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

from ..config import RiskCommConfig
from ..perception.snapshots import KnowledgeCache
from ..types import RiskSnapshot, Vector3
from .conflicts import RiskTask
from .model import predict_pair

RiskDecisionOutcome = Literal["insufficient", "timeout", "risk", "no_risk"]


@dataclass(frozen=True)
class RiskObjectDecision:
    object_id: int
    observation_step: int
    distance_m: float | None
    enters_risk_radius: bool | None
    reason: str
    source_id: int | None = None
    received_step: int | None = None
    input_kind: str | None = None
    age_seconds: float | None = None
    ego_distance_m: float | None = None
    object_arrival_seconds: float | None = None
    ego_arrival_seconds: float | None = None
    heading_difference_deg: float | None = None
    temporal_gap_seconds: float | None = None
    predicted_risk: bool | None = None
    stationary: bool = False
    snapshot: RiskSnapshot | None = None
    enters_event_area: bool | None = None


@dataclass(frozen=True)
class RiskDecision:
    outcome: RiskDecisionOutcome
    known_object_ids: tuple[int, ...]
    predicted_object_ids: tuple[int, ...]
    object_results: tuple[RiskObjectDecision, ...] = ()
    reason: str | None = None
    ego_snapshot: RiskSnapshot | None = None


def snapshot_trajectory_distance(
    snapshot: RiskSnapshot,
    conflict_point: Vector3,
    current_time_seconds: float,
    step_times: Sequence[float],
    extrapolation_seconds: float,
) -> float | None:
    """Horizontal distance from the age-adjusted future segment to a conflict point."""
    if _snapshot_trajectory_invalid_reason(
        snapshot, conflict_point, current_time_seconds, step_times, extrapolation_seconds
    ) is not None:
        return None
    velocity_x = snapshot.velocity_x
    velocity_z = snapshot.velocity_z
    if velocity_x is None or velocity_z is None:
        return None

    age_seconds = current_time_seconds - step_times[snapshot.observation_step]
    current_x = snapshot.x + velocity_x * age_seconds
    current_z = snapshot.z + velocity_z * age_seconds
    delta_x = velocity_x * extrapolation_seconds
    delta_z = velocity_z * extrapolation_seconds
    offset_x = conflict_point.x - current_x
    offset_z = conflict_point.z - current_z
    if not all(math.isfinite(value) for value in (offset_x, offset_z, delta_x, delta_z)):
        return None
    length = math.hypot(delta_x, delta_z)
    if not math.isfinite(length):
        return None
    if length == 0.0:
        distance = math.hypot(offset_x, offset_z)
    else:
        direction_x = delta_x / length
        direction_z = delta_z / length
        projection = offset_x * direction_x + offset_z * direction_z
        along_segment = min(max(projection, 0.0), length)
        distance = math.hypot(
            offset_x - along_segment * direction_x,
            offset_z - along_segment * direction_z,
        )
    return distance if math.isfinite(distance) else None


def _snapshot_trajectory_invalid_reason(
    snapshot: RiskSnapshot,
    conflict_point: Vector3,
    current_time_seconds: float,
    step_times: Sequence[float],
    extrapolation_seconds: float,
) -> str | None:
    if snapshot.velocity_x is None or snapshot.velocity_z is None:
        return "missing_velocity"
    if not all(math.isfinite(value) for value in (snapshot.velocity_x, snapshot.velocity_z)):
        return "invalid_velocity"
    if not 0 <= snapshot.observation_step < len(step_times):
        return "invalid_observation_step"
    if any(not math.isfinite(value) for value in step_times) or any(
        second <= first for first, second in zip(step_times, step_times[1:])
    ):
        return "invalid_time_axis"
    if not math.isfinite(current_time_seconds):
        return "invalid_current_time"
    if current_time_seconds < step_times[snapshot.observation_step]:
        return "future_observation"
    if not math.isfinite(extrapolation_seconds) or extrapolation_seconds < 0.0:
        return "invalid_extrapolation_horizon"
    if not all(math.isfinite(value) for value in (
        snapshot.x, snapshot.z, conflict_point.x, conflict_point.z
    )):
        return "invalid_position"
    return None


def execute_risk(
    task: RiskTask,
    knowledge: KnowledgeCache,
    current_step: int,
    current_time_seconds: float,
    step_times: Sequence[float],
    config: RiskCommConfig,
    ego_snapshot: RiskSnapshot | None = None,
    scheduled_ego_arrival_seconds: float | None = None,
) -> RiskDecision:
    # The pair model consumes owner/map plus current knowledge. A may explicitly
    # receive offline-derived own ETA; B ignores it. No opponent IDs/labels or
    # future sample times are used by either prediction mode.
    step_times = step_times[:current_step + 1]
    known_object_ids = tuple(
        object_id
        for object_id in knowledge.known_object_ids
        if object_id != task.vehicle_id
    )
    if not known_object_ids:
        return RiskDecision(
            outcome="timeout",
            known_object_ids=(),
            predicted_object_ids=(),
            reason="no_known_objects",
            ego_snapshot=ego_snapshot,
        )

    predicted_object_ids: list[int] = []
    object_results: list[RiskObjectDecision] = []
    for object_id in known_object_ids:
        snapshot = knowledge.snapshot_for(object_id)
        prediction = predict_pair(ego_snapshot, snapshot, task.conflict_point,
                                  current_time_seconds, step_times, config,
                                  scheduled_ego_arrival_seconds)
        distance = prediction.distance_m
        enters_risk_radius = (
            distance <= config.risk_radius_m + 1e-9 if distance is not None else None
        )
        object_results.append(RiskObjectDecision(
            object_id=object_id,
            observation_step=snapshot.observation_step,
            distance_m=distance,
            enters_risk_radius=enters_risk_radius,
            reason=prediction.reason,
            source_id=snapshot.source_id,
            received_step=snapshot.received_step,
            input_kind="local" if snapshot.received_step is None else "received",
            age_seconds=(current_time_seconds - step_times[snapshot.observation_step]
                         if 0 <= snapshot.observation_step < len(step_times) else None),
            ego_distance_m=prediction.ego_distance_m,
            object_arrival_seconds=prediction.object_arrival_seconds,
            ego_arrival_seconds=prediction.ego_arrival_seconds,
            heading_difference_deg=prediction.heading_difference_deg,
            temporal_gap_seconds=prediction.temporal_gap_seconds,
            predicted_risk=prediction.risk,
            stationary=prediction.stationary,
            snapshot=snapshot,
            enters_event_area=(distance <= config.conflict_cluster_radius_m + 1e-9
                               if distance is not None else None),
        ))
        if prediction.risk:
            predicted_object_ids.append(object_id)

    predicted = tuple(sorted(predicted_object_ids))
    return RiskDecision(
        outcome=("risk" if predicted else "insufficient"
                 if any(item.predicted_risk is None for item in object_results) else "no_risk"),
        known_object_ids=known_object_ids,
        predicted_object_ids=predicted,
        object_results=tuple(object_results),
        reason=("predicted_crossing" if predicted else "unresolved_motion"
                if any(item.predicted_risk is None for item in object_results) else "no_predicted_crossing"),
        ego_snapshot=ego_snapshot,
    )
