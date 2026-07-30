from typing import TYPE_CHECKING, Any

from .aoi_gain_policy import MaxAoiGainSetPolicy, MaxCaoiGainSetPolicy
from .random_policy import RandomEligiblePolicy, RandomEligibleSetPolicy
from .two_stage_policy import TwoStageRelayGreedyPolicy
from .task_policy import (
    MaxTaskRewardGainPolicy,
    RandomTaskAwarePolicy,
    TwoStageTaskRelayGreedyPolicy,
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
]


def __getattr__(name: str) -> Any:
    if name == "PpoInferencePolicy":
        from ..rl.policy import PpoInferencePolicy

        return PpoInferencePolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
