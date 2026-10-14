from __future__ import annotations

from ..config import RiskCommConfig
from ..types import LinkForecast, RiskSnapshot, Vector3, VehicleReplay
from .communication import SnapshotCommunication
from .timing import estimate_step_seconds


def predict_downlink_window(
    snapshot: RiskSnapshot,
    receiver_id: int,
    current_step: int,
    replay_history: VehicleReplay,
    config: RiskCommConfig,
    *,
    communication_model: SnapshotCommunication,
    fallback_step_seconds: float,
    rsu_position: Vector3 = Vector3(0.0, 0.0, 0.0),
) -> LinkForecast:
    """Predict a short RSU-to-receiver contact window causally.

    Only frames at or before ``current_step`` are read.  The future positions
    in a replay are deliberately ignored; the receiver is extrapolated from
    its last two available observations.  The object snapshot is accepted as
    an explicit input so callers can associate a forecast with a downlink
    candidate, although the window itself depends on the receiver geometry.
    """

    del snapshot
    predicted_steps = tuple(
        current_step + config.decision_lead_steps + offset
        for offset in range(config.prediction_horizon_steps)
    )
    if not predicted_steps:
        return LinkForecast(
            receiver_id=receiver_id,
            current_step=current_step,
            predicted_steps=(),
            reachable_steps=(),
            first_unreachable_step=None,
            available=False,
        )

    historical_frames = tuple(
        frame for frame in replay_history.frames[:current_step + 1]
    )
    current_frame = next(
        (frame for frame in historical_frames
         if frame.step_index == current_step and receiver_id in frame.vehicles),
        None,
    )
    if current_frame is None:
        return LinkForecast(
            receiver_id=receiver_id,
            current_step=current_step,
            predicted_steps=predicted_steps,
            reachable_steps=(),
            first_unreachable_step=predicted_steps[0],
            available=False,
        )

    previous = next(
        (
            frame
            for frame in reversed(historical_frames)
            if frame.step_index < current_frame.step_index and receiver_id in frame.vehicles
        ),
        None,
    )
    if previous is None:
        return LinkForecast(
            receiver_id=receiver_id,
            current_step=current_step,
            predicted_steps=predicted_steps,
            reachable_steps=(),
            first_unreachable_step=predicted_steps[0],
            available=False,
        )

    elapsed = current_frame.elapsed_seconds - previous.elapsed_seconds
    if elapsed <= 0.0:
        return LinkForecast(
            receiver_id=receiver_id,
            current_step=current_step,
            predicted_steps=predicted_steps,
            reachable_steps=(),
            first_unreachable_step=predicted_steps[0],
            available=False,
        )

    current = current_frame.vehicles[receiver_id]
    previous_snapshot = previous.vehicles[receiver_id]
    velocity_x = (current.position.x - previous_snapshot.position.x) / elapsed
    velocity_z = (current.position.z - previous_snapshot.position.z) / elapsed
    step_seconds = estimate_step_seconds(
        tuple(frame.elapsed_seconds for frame in historical_frames), fallback_step_seconds,
    )

    reachable_steps: list[int] = []
    for predicted_step in predicted_steps:
        offset_seconds = (predicted_step - current_step) * step_seconds
        predicted_position = Vector3(
            x=current.position.x + velocity_x * offset_seconds,
            y=current.position.y,
            z=current.position.z + velocity_z * offset_seconds,
        )
        blockers = []
        for object_id, state in sorted(current_frame.vehicles.items()):
            if object_id == receiver_id:
                continue
            previous_frame = next((frame for frame in reversed(historical_frames)
                                   if frame.step_index < current_step and object_id in frame.vehicles), None)
            vx = vz = 0.0
            if previous_frame is not None:
                duration = current_frame.elapsed_seconds - previous_frame.elapsed_seconds
                if duration > 0:
                    old = previous_frame.vehicles[object_id].position
                    vx = (state.position.x - old.x) / duration
                    vz = (state.position.z - old.z) / duration
            blockers.append(Vector3(state.position.x + vx * offset_seconds,
                                    state.position.y, state.position.z + vz * offset_seconds))
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
