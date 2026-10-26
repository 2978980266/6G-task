from __future__ import annotations

import math
from typing import Sequence

from ..config import RiskCommConfig
from ..types import LinkForecast, RiskSnapshot, Vector3
from .communication import SnapshotCommunication
from .timing import estimate_step_seconds


def project_snapshot_position(
    snapshot: RiskSnapshot | None, time_seconds: float, step_times: Sequence[float],
) -> Vector3 | None:
    """Project an original legal snapshot; missing motion remains unknown.

    Uses the measured displacement velocity, as the former contact predictor
    did. In particular a stale received position is not treated as current and
    missing velocity is not silently replaced with zero or private self motion.
    """
    if snapshot is None or snapshot.velocity_x is None or snapshot.velocity_z is None:
        return None
    if (not 0 <= snapshot.observation_step < len(step_times)
            or not step_times or any(not math.isfinite(t) for t in step_times)
            or any(b <= a for a, b in zip(step_times, step_times[1:]))
            or not all(math.isfinite(v) for v in
                       (time_seconds, snapshot.x, snapshot.y, snapshot.z,
                        snapshot.velocity_x, snapshot.velocity_z))):
        return None
    age = time_seconds - step_times[snapshot.observation_step]
    if age < 0:
        return None
    x, z = snapshot.x + snapshot.velocity_x * age, snapshot.z + snapshot.velocity_z * age
    return Vector3(x, snapshot.y, z) if math.isfinite(x) and math.isfinite(z) else None


def predict_downlink_window(
    snapshot: RiskSnapshot,
    receiver_id: int,
    current_step: int,
    rsu_snapshots: tuple[RiskSnapshot, ...],
    step_times: tuple[float, ...],
    config: RiskCommConfig,
    *,
    communication_model: SnapshotCommunication,
    current_time_seconds: float,
    fallback_step_seconds: float,
    rsu_position: Vector3 = Vector3(0.0, 0.0, 0.0),
) -> LinkForecast:
    """Predict contact using only RSU-held motion and a historical time prefix.

    Receiver and blocker snapshots come from current RSU sensing or actual
    retained receptions. Unobserved blockers are unknown, so this is a contact
    estimate, not a guarantee about the true physical channel. The environment
    still checks the actual channel independently when executing an action.
    """

    del snapshot
    predicted_steps = tuple(
        current_step + config.decision_lead_steps + offset
        for offset in range(config.prediction_horizon_steps)
    )
    def unknown(reason: str) -> LinkForecast:
        return LinkForecast(receiver_id, current_step, predicted_steps, (), None, False, reason)

    if not predicted_steps:
        return unknown("empty_prediction_horizon")
    times = step_times[:current_step + 1]
    receiver = next((s for s in rsu_snapshots if s.object_id == receiver_id), None)
    if receiver is None:
        return unknown("missing_receiver_state")
    if project_snapshot_position(receiver, current_time_seconds, times) is None:
        return unknown("receiver_motion_unknown")
    step_seconds = estimate_step_seconds(times, fallback_step_seconds)

    reachable_steps: list[int] = []
    for predicted_step in predicted_steps:
        offset_seconds = (predicted_step - current_step) * step_seconds
        time = current_time_seconds + offset_seconds
        predicted_position = project_snapshot_position(receiver, time, times)
        if predicted_position is None:
            return unknown("receiver_motion_unknown")
        blockers: list[Vector3] = []
        for state in sorted(rsu_snapshots, key=lambda s: s.object_id):
            if state.object_id == receiver_id:
                continue
            position = project_snapshot_position(state, time, times)
            if position is None:
                return unknown("known_blocker_motion_unknown")
            blockers.append(position)
        if communication_model.reachable(rsu_position, predicted_position, 0, tuple(blockers)):
            reachable_steps.append(predicted_step)

    reachable = tuple(reachable_steps)
    first_unreachable = next(
        (step for step in predicted_steps if step not in set(reachable)),
        None,
    )
    return LinkForecast(
        receiver_id=receiver_id,
        current_step=current_step,
        predicted_steps=predicted_steps,
        reachable_steps=reachable,
        first_unreachable_step=first_unreachable,
        available=True,
    )


__all__ = ["predict_downlink_window"]
