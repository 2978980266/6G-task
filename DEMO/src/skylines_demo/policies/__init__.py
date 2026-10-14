from typing import TYPE_CHECKING, Any

from .aoi_gain_policy import MaxAoiGainSetPolicy, MaxCaoiGainSetPolicy
from .random_policy import RandomEligiblePolicy, RandomEligibleSetPolicy
from .two_stage_policy import TwoStageRelayGreedyPolicy
from .task_policy import (
    MaxTaskRewardGainPolicy,
    RandomTaskAwarePolicy,
    TwoStageTaskRelayGreedyPolicy,
)
from .risk_baselines import (
    select_age_action,
    select_deadline_action,
    select_event_trigger_action,
    select_unlimited_action,
)
from .risk_receiver_policy import select_receiver_value_action
from .risk_relay_policy import (
    select_receiver_value_with_relay_action,
    select_relay_action,
)

if TYPE_CHECKING:
    from ..rl.policy import PpoInferencePolicy

__all__ = [
    "MaxAoiGainSetPolicy",
    "MaxCaoiGainSetPolicy",
    "PpoInferencePolicy",
    "RandomEligiblePolicy",
    "RandomEligibleSetPolicy",
    "RandomTaskAwarePolicy",
    "MaxTaskRewardGainPolicy",
    "TwoStageRelayGreedyPolicy",
    "TwoStageTaskRelayGreedyPolicy",
    "select_age_action",
    "select_deadline_action",
    "select_event_trigger_action",
    "select_receiver_value_action",
    "select_receiver_value_with_relay_action",
    "select_relay_action",
    "select_unlimited_action",
]


def __getattr__(name: str) -> Any:
    if name == "PpoInferencePolicy":
        from ..rl.policy import PpoInferencePolicy

        return PpoInferencePolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
