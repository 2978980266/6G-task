from __future__ import annotations

from dataclasses import replace

from ..config import RiskCommConfig
from ..types import DownlinkAction, RiskObservation, RiskSchedulingAction
from .risk_baselines import _build_feasible_action
from ..risk.sufficiency import estimate_receiver_delivery_value, prediction_triggered_task


def select_receiver_value_action(
    observation: RiskObservation,
    config: RiskCommConfig,
    *,
    enable_link_prediction: bool | None = None,
) -> RiskSchedulingAction:
    """Rank downlinks by public receiver-side information value.

    The score uses only current task deadlines, current receiver geometry,
    known observation versions, and the RSU snapshots already exposed by the
    observation.  It intentionally does not inspect conflict truth or future
    replay frames.
    """

    prediction_enabled = (
        config.link_prediction
        if enable_link_prediction is None
        else enable_link_prediction
    )
    prediction_config = replace(config, link_prediction=prediction_enabled)
    def value_key(candidate: DownlinkAction) -> tuple[float, float, float, int, int]:
        value = estimate_receiver_delivery_value(observation, candidate, config)
        forecast_priority = float(
            prediction_enabled and prediction_triggered_task(observation, candidate, prediction_config) is not None
        )
        return (
            -(value.value + forecast_priority),
            -value.freshness_gain,
            -value.urgency,
            candidate.receiver_id,
            candidate.object_id,
        )

    downlinks = sorted(
        (candidate for candidate in observation.downlink_candidates
         if estimate_receiver_delivery_value(observation, candidate, config).eligible),
        key=value_key,
    )
    uplinks = sorted(
        observation.uplink_candidates,
        key=lambda item: (item.sender_id, item.object_id),
    )
    return _build_feasible_action(
        observation,
        config,
        downlinks=downlinks,
        uplinks=uplinks,
    )


__all__ = ["select_receiver_value_action"]
