from __future__ import annotations

from ..types import CandidateLink, LinkAction, SchedulingAction, StepObservation


class TwoStageRelayGreedyPolicy:
    def __init__(
        self,
        allow_multi_v2i_from_rsu: bool = False,
        relay_bonus_weight: float = 0.35,
    ) -> None:
        self._allow_multi_v2i_from_rsu = allow_multi_v2i_from_rsu
        self._relay_bonus_weight = relay_bonus_weight

    def reset(self) -> None:
        pass

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        v2i_candidates = [
            candidate for candidate in observation.candidate_v2i_links if candidate.delivered_packet_count > 0.0
        ]
        v2v_candidates = [
            candidate for candidate in observation.candidate_v2v_links if candidate.delivered_packet_count > 0.0
        ]

        selected: list[CandidateLink] = []
        used_vehicle_nodes: set[int] = set()
        v2i_used = False

        if v2i_candidates:
            v2i_candidates.sort(key=self._v2i_priority_key, reverse=True)
            best_seed = v2i_candidates[0]
            selected.append(best_seed)
            used_vehicle_nodes.update(_vehicle_nodes_for_action(best_seed.action))
            v2i_used = True

        v2v_candidates.sort(
            key=lambda candidate: (
                candidate.priority_gain_seconds,
                candidate.freshness_gain_seconds,
                candidate.link_budget.rate_mbps,
            ),
            reverse=True,
        )
        v2i_used = _append_non_conflicting(
            selected,
            v2v_candidates,
            used_vehicle_nodes,
            v2i_used=v2i_used,
            allow_multi_v2i_from_rsu=False,
        )

        if self._allow_multi_v2i_from_rsu:
            remaining_v2i = [candidate for candidate in v2i_candidates if candidate not in selected]
            remaining_v2i.sort(key=self._v2i_priority_key, reverse=True)
            _append_non_conflicting(
                selected,
                remaining_v2i,
                used_vehicle_nodes,
                v2i_used=v2i_used,
                allow_multi_v2i_from_rsu=True,
            )

        return _build_scheduling_action(selected)

    def _v2i_priority_key(
        self,
        candidate: CandidateLink,
    ) -> tuple[float, int, float, float, float]:
        score = (
            candidate.priority_gain_seconds
            + self._relay_bonus_weight * candidate.relay_peer_total_priority_gain_seconds
        )
        return (
            score,
            candidate.relay_peer_count,
            candidate.priority_gain_seconds,
            candidate.freshness_gain_seconds,
            candidate.link_budget.rate_mbps,
        )



def _append_non_conflicting(
    selected: list[CandidateLink],
    candidates: list[CandidateLink],
    used_vehicle_nodes: set[int],
    v2i_used: bool,
    allow_multi_v2i_from_rsu: bool,
) -> bool:
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
    return v2i_used



def _build_scheduling_action(selected: list[CandidateLink]) -> SchedulingAction:
    v2i_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2i")
    v2v_links = tuple(candidate.action for candidate in selected if candidate.action.link_type == "v2v")
    return SchedulingAction(v2i_links=v2i_links, v2v_links=v2v_links)



def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
    nodes = [action.receiver_id]
    if action.link_type == "v2v" and action.transmitter_id.startswith("vehicle:"):
        nodes.append(int(action.transmitter_id.split(":", 1)[1]))
    return tuple(nodes)
