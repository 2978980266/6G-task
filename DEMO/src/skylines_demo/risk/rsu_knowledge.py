"""RSU 估计只读自身合法快照，不能继承接收车的私有自身运动。"""
from __future__ import annotations

from ..config import RiskCommConfig
from ..types import RSU_INFORMATION_VERSION, RSUObservation, RiskTaskView, Vector3
from .model import PairPrediction, own_schedule_eta, predict_pair


def rsu_information_contract() -> dict[str, object]:
    """Versioned result/metadata description, separate from vehicle model rights."""
    return dict(
        version=RSU_INFORMATION_VERSION,
        motion="current_rsu_sensing_or_actual_retained_receptions_with_original_age_and_source",
        receiver_prediction="rsu_held_receiver_snapshot_only_unknown_is_not_safe",
        link_prediction="rsu_held_receiver_and_blockers_historical_time_prefix_only",
        private_self_motion="vehicle_executor_and_explicit_offline_world_diagnostics_only",
        ideal_control="vehicle_ids_public_tasks_object_ids_versions_and_link_feedback",
        control_plane_transport_and_full_accounting_implemented=False,
    )


def public_task_eta(config: RiskCommConfig, task: RiskTaskView,
                    current_step: int, interval_seconds: float) -> float | None:
    # B must not even access crossing_step, or reconstruct it from the deadline.
    if config.ego_eta_mode == "causal-self-motion":
        return None
    if task.crossing_step is None:
        return None
    return own_schedule_eta(config, task.crossing_step, current_step, interval_seconds)


def predict_receiver_pair(
    observation: RSUObservation, receiver_id: int, object_id: int, point: Vector3,
    config: RiskCommConfig, scheduled_ego_arrival_seconds: float | None = None,
) -> PairPrediction:
    """Estimate a receiver's decision; unknown is distinct from a safe prediction.

    The frozen model already compensates both original snapshot ages using the
    historical time prefix. No synthetic current snapshot or velocity is made.
    The real vehicle executor continues to use its own independent self motion.
    """
    obj = observation.snapshot_for(object_id)
    if obj is None:
        return PairPrediction(None, "missing_object_state")
    return predict_pair(
        observation.snapshot_for(receiver_id), obj, point, observation.elapsed_seconds,
        observation.step_times[:observation.step_index + 1], config,
        scheduled_ego_arrival_seconds=scheduled_ego_arrival_seconds,
    )
