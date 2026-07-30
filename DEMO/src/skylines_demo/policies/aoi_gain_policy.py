from __future__ import annotations

from ..types import CandidateLink, LinkAction, SchedulingAction, StepObservation


class MaxCaoiGainSetPolicy:
    def __init__(self, allow_multi_v2i_from_rsu: bool = False) -> None:
        self._allow_multi_v2i_from_rsu = allow_multi_v2i_from_rsu

    def reset(self) -> None:
        pass

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        candidates = [candidate for candidate in observation.candidate_links if candidate.delivered_packet_count > 0.0]
        candidates.sort(
            key=lambda candidate: (
                candidate.priority_gain_seconds,
                candidate.freshness_gain_seconds,
                candidate.link_budget.rate_mbps,
            ),
            reverse=True,
        )
        selected = _select_non_conflicting(
            candidates,
            allow_multi_v2i_from_rsu=self._allow_multi_v2i_from_rsu,
        )
        return _build_scheduling_action(selected)


class MaxAoiGainSetPolicy(MaxCaoiGainSetPolicy):
    """Backward-compatible alias for older scripts and command names."""


def _select_non_conflicting(
    candidates: list[CandidateLink],
    allow_multi_v2i_from_rsu: bool = False,
) -> list[CandidateLink]:
    selected: list[CandidateLink] = []
    used_vehicle_nodes: set[int] = set()
    v2i_used = False
    for candidate in candidates:
        if candidate.action.link_type == "v2i" and v2i_used and not allow_multi_v2i_from_rsu:
            continue
        nodes = _vehicle_nodes_for_action(candidate.action)
        if any(node in used_vehicle_nodes for node in nodes):
            continue
        selected.append(candidate)
        used_vehicle_nodes.update(nodes)
        if candidate.action.link_type == "v2i":
            v2i_used = True
    return selected



def _build_scheduling_action(selected: list[CandidateLink]) -> SchedulingAction:
    v2i_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2i")
    v2v_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2v")
    return SchedulingAction(v2i_links=v2i_links, v2v_links=v2v_links)



def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
    nodes = [action.receiver_id]
    if action.link_type == "v2v" and action.transmitter_id.startswith("vehicle:"):
        nodes.append(int(action.transmitter_id.split(":", 1)[1]))
    return tuple(nodes)
