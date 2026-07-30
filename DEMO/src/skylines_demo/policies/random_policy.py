from __future__ import annotations

import random
from typing import Sequence

from ..types import CandidateLink, LinkAction, SchedulingAction, StepObservation


class RandomEligiblePolicy:
    def __init__(self, seed: int = 7) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_action(self, observation: StepObservation) -> LinkAction | None:
        reachable = [
            candidate.action for candidate in observation.candidate_links if candidate.delivered_packet_count > 0.0
        ]
        if not reachable:
            return None
        return self._rng.choice(reachable)


class RandomEligibleSetPolicy:
    def __init__(
        self,
        seed: int = 7,
        max_links: int | None = None,
        selection_probability: float = 0.5,
        allow_multi_v2i_from_rsu: bool = False,
    ) -> None:
        self._seed = seed
        self._max_links = max_links
        self._selection_probability = selection_probability
        self._allow_multi_v2i_from_rsu = allow_multi_v2i_from_rsu
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        candidates = [candidate for candidate in observation.candidate_links if candidate.delivered_packet_count > 0.0]
        candidates = _weighted_shuffle(candidates, self._rng)
        selected = _select_random_non_conflicting(
            candidates,
            rng=self._rng,
            selection_probability=self._selection_probability,
            max_links=self._max_links,
            allow_multi_v2i_from_rsu=self._allow_multi_v2i_from_rsu,
        )
        return _build_scheduling_action(selected)


def _select_random_non_conflicting(
    candidates: list[CandidateLink],
    rng: random.Random,
    selection_probability: float,
    max_links: int | None = None,
    allow_multi_v2i_from_rsu: bool = False,
) -> list[CandidateLink]:
    selected: list[CandidateLink] = []
    used_vehicle_nodes: set[int] = set()
    skipped_available: list[CandidateLink] = []
    v2i_used = False

    for candidate in candidates:
        if candidate.action.link_type == "v2i" and v2i_used and not allow_multi_v2i_from_rsu:
            continue
        nodes = _vehicle_nodes_for_action(candidate.action)
        if any(node in used_vehicle_nodes for node in nodes):
            continue
        if rng.random() > selection_probability:
            skipped_available.append(candidate)
            continue
        selected.append(candidate)
        used_vehicle_nodes.update(nodes)
        if candidate.action.link_type == "v2i":
            v2i_used = True
        if max_links is not None and len(selected) >= max_links:
            break

    if not selected and skipped_available:
        selected.append(skipped_available[0])

    return selected


def _weighted_shuffle(candidates: Sequence[CandidateLink], rng: random.Random) -> list[CandidateLink]:
    pool = list(candidates)
    shuffled: list[CandidateLink] = []
    while pool:
        weights = [max(candidate.priority_gain_seconds, 1e-6) for candidate in pool]
        index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        shuffled.append(pool.pop(index))
    return shuffled


def _build_scheduling_action(selected: list[CandidateLink]) -> SchedulingAction:
    v2i_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2i")
    v2v_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2v")
    return SchedulingAction(v2i_links=v2i_links, v2v_links=v2v_links)


def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
    nodes = [action.receiver_id]
    if action.link_type == "v2v" and action.transmitter_id.startswith("vehicle:"):
        nodes.append(int(action.transmitter_id.split(":", 1)[1]))
    return tuple(nodes)
