from __future__ import annotations

from .task_common import (
    TASK_CANDIDATE_SCALAR_FEATURE_COUNT,
    TASK_GLOBAL_FEATURE_COUNT,
    build_task_scheduling_action,
    rank_task_candidates,
)
from .policy import PpoInferencePolicy

_OPTIONAL_RL_IMPORT_ERROR: Exception | None = None
try:
    from .gym_env import PpoSchedulingGymEnv
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on optional RL deps
    _OPTIONAL_RL_IMPORT_ERROR = exc
    class PpoSchedulingGymEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "缺少 PPO 训练依赖。请先安装 gymnasium、stable-baselines3 和 torch。"
            ) from _OPTIONAL_RL_IMPORT_ERROR


__all__ = [
    "PpoInferencePolicy",
    "PpoSchedulingGymEnv",
    "TASK_CANDIDATE_SCALAR_FEATURE_COUNT",
    "TASK_GLOBAL_FEATURE_COUNT",
    "build_task_scheduling_action",
    "rank_task_candidates",
]
