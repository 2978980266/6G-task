from __future__ import annotations

import random

from ..config import ResourceConfig
from ..types import CandidateLink, LinkAction, SchedulingAction, StepObservation


class RandomTaskAwarePolicy:
    def __init__(
        self,
        resource_config: ResourceConfig,
        seed: int = 7,
        selection_probability: float = 0.65,
        max_links: int | None = None,
        allow_multi_v2i_from_rsu: bool = True,
    ) -> None:
        self._resource_config = resource_config
        self._seed = seed
        self._selection_probability = selection_probability
        self._max_links = max_links
        self._allow_multi_v2i_from_rsu = allow_multi_v2i_from_rsu
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        candidates = list(observation.candidate_links)
        self._rng.shuffle(candidates)
        candidates.sort(
            key=lambda candidate: (
                candidate.task_progress_gain > 0,
                candidate.task_reward_gain > 0.0,
            ),
            reverse=True,
        )
        selected = _select_resource_feasible(
            candidates,
            self._resource_config,
            should_select=lambda: self._rng.random() <= self._selection_probability,
            max_links=self._max_links,
            allow_multi_v2i_from_rsu=self._allow_multi_v2i_from_rsu,
        )
        return _build_scheduling_action(selected)


class MaxTaskRewardGainPolicy:
    def __init__(
        self,
        resource_config: ResourceConfig,
        max_links: int | None = None,
        allow_multi_v2i_from_rsu: bool = True,
    ) -> None:
        self._resource_config = resource_config
        self._max_links = max_links
        self._allow_multi_v2i_from_rsu = allow_multi_v2i_from_rsu

    def reset(self) -> None:
        pass

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        candidates = sorted(
            observation.candidate_links,
            key=_task_candidate_sort_key,
            reverse=True,
        )
        selected = _select_resource_feasible(
            candidates,
            self._resource_config,
            max_links=self._max_links,
            allow_multi_v2i_from_rsu=self._allow_multi_v2i_from_rsu,
        )
        return _build_scheduling_action(selected)


class TwoStageTaskRelayGreedyPolicy:
    def __init__(
        self,
        resource_config: ResourceConfig,
        relay_bonus_weight: float = 0.5,
        max_links: int | None = None,
        allow_multi_v2i_from_rsu: bool = True,
    ) -> None:
        self._resource_config = resource_config
        self._relay_bonus_weight = relay_bonus_weight
        self._max_links = max_links
        self._allow_multi_v2i_from_rsu = allow_multi_v2i_from_rsu

    def reset(self) -> None:
        pass

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        v2i_candidates = sorted(
            observation.candidate_v2i_links,
            key=self._seed_sort_key,
            reverse=True,
        )
        v2v_candidates = sorted(
            observation.candidate_v2v_links,
            key=_task_candidate_sort_key,
            reverse=True,
        )
        selected = _select_resource_feasible(
            v2i_candidates + v2v_candidates,
            self._resource_config,
            max_links=self._max_links,
            allow_multi_v2i_from_rsu=self._allow_multi_v2i_from_rsu,
        )
        return _build_scheduling_action(selected)

    def _seed_sort_key(self, candidate: CandidateLink) -> tuple[float, ...]:
        return (
            candidate.task_reward_gain,
            float(candidate.task_progress_gain),
            self._relay_bonus_weight
            * candidate.relay_peer_total_priority_gain_seconds,
            candidate.priority_gain_seconds,
            float(candidate.delivered_packet_count),
            -float(candidate.resource_block_count),
        )


def _select_resource_feasible(
    candidates: list[CandidateLink],
    resource_config: ResourceConfig,
    should_select=None,
    max_links: int | None = None,
    allow_multi_v2i_from_rsu: bool = True,
) -> list[CandidateLink]:
    if max_links is not None and max_links <= 0:
        return []

    selected: list[CandidateLink] = []
    used_vehicle_nodes: set[int] = set()
    used_rsu_blocks = 0
    used_v2v_blocks = 0
    used_rsu_rate = 0.0
    used_v2v_rate = 0.0
    v2i_used = False

    for candidate in candidates:
        if max_links is not None and len(selected) >= max_links:
            break
        if (
            candidate.action.link_type == "v2i"
            and v2i_used
            and not allow_multi_v2i_from_rsu
        ):
            continue
        nodes = _vehicle_nodes_for_action(candidate.action)
        if any(node in used_vehicle_nodes for node in nodes):
            continue
        if should_select is not None and not should_select():
            continue

        if candidate.action.link_type == "v2i":
            next_blocks = used_rsu_blocks + candidate.resource_block_count
            next_rate = used_rsu_rate + candidate.link_budget.rate_mbps
            if next_blocks > resource_config.rsu_resource_block_count:
                continue
            if next_rate > resource_config.rsu_total_rate_limit_mbps + 1e-9:
                continue
            used_rsu_blocks = next_blocks
            used_rsu_rate = next_rate
        else:
            next_blocks = used_v2v_blocks + candidate.resource_block_count
            next_rate = used_v2v_rate + candidate.link_budget.rate_mbps
            if next_blocks > resource_config.v2v_resource_block_count:
                continue
            if next_rate > resource_config.v2v_total_rate_limit_mbps + 1e-9:
                continue
            used_v2v_blocks = next_blocks
            used_v2v_rate = next_rate

        selected.append(candidate)
        used_vehicle_nodes.update(nodes)
        if candidate.action.link_type == "v2i":
            v2i_used = True

    return selected


def _task_candidate_sort_key(candidate: CandidateLink) -> tuple[float, ...]:
    return (
        candidate.task_reward_gain,
        float(candidate.task_progress_gain),
        candidate.relay_peer_total_priority_gain_seconds,
        candidate.priority_gain_seconds,
        float(candidate.delivered_packet_count),
        candidate.link_budget.rate_mbps,
        -float(candidate.resource_block_count),
    )


def _build_scheduling_action(
    selected: list[CandidateLink],
) -> SchedulingAction:
    return SchedulingAction(
        v2i_links=tuple(
            candidate.action
            for candidate in selected
            if candidate.action.link_type == "v2i"
        ),
        v2v_links=tuple(
            candidate.action
            for candidate in selected
            if candidate.action.link_type == "v2v"
        ),
    )


def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
    nodes = [action.receiver_id]
    if (
        action.link_type == "v2v"
        and action.transmitter_id.startswith("vehicle:")
    ):
        nodes.append(int(action.transmitter_id.split(":", 1)[1]))
    return tuple(nodes)
