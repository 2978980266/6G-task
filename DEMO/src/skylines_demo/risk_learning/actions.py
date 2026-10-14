"""组合状态只在当前候选快照内变化，不推进环境或缓存。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..types import DownlinkAction, RelayAction, RiskSchedulingAction, UplinkAction
from .domain import SafeStep

COMBINATION_VERSION = "risk-stop-dynamic-single-vehicle-v1"


@dataclass(frozen=True)
class SelectionState:
    selected_indices: tuple[int, ...]
    occupied_vehicle_ids: frozenset[int]
    remaining: tuple[int, int, int]
    stopped: bool = False


def initial_state(step: SafeStep) -> SelectionState:
    return SelectionState((), frozenset(), step.slots)


def legal_mask(step: SafeStep, state: SelectionState) -> Any:
    import numpy as np
    mask = np.zeros(len(step.candidates) + 1, dtype=bool)
    if state.stopped:
        return mask
    mask[0] = True
    for i, c in enumerate(step.candidates, 1):
        kind, sender, receiver, _ = c.key
        slot = ("U", "D", "R").index(kind)
        endpoints = {node for node in (sender, receiver) if node != 0}
        mask[i] = (i not in state.selected_indices and state.remaining[slot] > 0
                   and not endpoints & state.occupied_vehicle_ids)
    return mask


def append_choice(step: SafeStep, state: SelectionState, index: int) -> SelectionState:
    if type(index) is not int or not 0 <= index <= len(step.candidates) or not legal_mask(step, state)[index]:
        raise ValueError("非法、重复、冲突、padding或STOP后选择。")
    if index == 0:
        return replace(state, stopped=True)
    kind, sender, receiver, _ = step.candidates[index - 1].key
    remaining = list(state.remaining)
    remaining[("U", "D", "R").index(kind)] -= 1
    return SelectionState((*state.selected_indices, index),
                          state.occupied_vehicle_ids | {n for n in (sender, receiver) if n != 0},
                          tuple(remaining))  # type: ignore[arg-type]


def to_environment_action(step: SafeStep, state: SelectionState) -> RiskSchedulingAction:
    # 再验证全部前缀，拒绝手工构造的不合法SelectionState。
    checked = initial_state(step)
    for index in state.selected_indices:
        checked = append_choice(step, checked, index)
    if (checked.occupied_vehicle_ids, checked.remaining) != (state.occupied_vehicle_ids, state.remaining):
        raise ValueError("组合状态与选择前缀不一致。")
    actions = tuple(step.candidates[i - 1].action for i in state.selected_indices)
    return RiskSchedulingAction(tuple(c for c in actions if isinstance(c, UplinkAction)),
                                tuple(c for c in actions if isinstance(c, DownlinkAction)),
                                tuple(c for c in actions if isinstance(c, RelayAction)))
