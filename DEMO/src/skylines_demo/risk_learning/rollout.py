"""安全输入/路径与训练标量目标分列；只接受完整自然终止回合。"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

from .domain import SafeStep
from .encoding import EncodedStep
from .policy import ActionPath


@dataclass(frozen=True)
class StepRecord:
    encoded: EncodedStep
    step: SafeStep
    path: ActionPath


@dataclass(frozen=True)
class EpisodeRecord:
    records: tuple[StepRecord, ...]
    rewards: tuple[float, ...]
    terminated: bool

    def validate(self) -> None:
        if (not self.terminated or len(self.records) != 23 or len(self.rewards) != 23
                or tuple(r.step.step_index for r in self.records) != tuple(range(23))):
            raise ValueError("只优化完整23step自然终止episode；截断不bootstrap或伪结算。")
        compute_returns(self.rewards)


def compute_returns(rewards: tuple[float, ...]) -> tuple[float, ...]:
    if not all(isfinite(r) for r in rewards):
        raise ValueError("reward含非有限值。")
    return tuple(fsum(rewards[t:]) for t in range(len(rewards)))
