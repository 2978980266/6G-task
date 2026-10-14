"""Explicit cost/selection experiments using only public IDs, versions and deadlines.

These ordering keys are online heuristics, never rewards or known true gains.
Equal RSU-local versions are skipped; this can forgo persistent data-plane storage.
The existing snapshot/cache rules remain the execution authority.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from ..config import RiskCommConfig
from ..risk.sufficiency import receiver_version
from ..types import RiskObservation, RiskSchedulingAction, UplinkAction
from .risk_baselines import _build_feasible_action
from .risk_receiver_policy import select_receiver_value_action
from .risk_relay_policy import select_receiver_value_with_relay_action

UplinkMode = Literal["version", "demand"]
Selector = Callable[[RiskObservation, RiskCommConfig], RiskSchedulingAction]


@dataclass(frozen=True)
class PublicUplinkRank:
    candidate: UplinkAction
    source_version: int
    rsu_version: int
    beneficiary_ids: tuple[int, ...]
    nearest_deadline: int | None
    eligible: bool
    reason: str
    # Lower key is preferred. No motion state or truth is read to construct it.
    key: tuple[int, int, int, int, int]


def select_deadline_stopped_action(
    observation: RiskObservation, config: RiskCommConfig, *,
    selector: Selector = select_receiver_value_action,
) -> RiskSchedulingAction:
    """Recheck public deadlines each step; a_t cannot change the decision at t."""
    if not any(task.decision_step >= observation.step_index + 1 for task in observation.tasks):
        return RiskSchedulingAction()
    return selector(observation, config)


def rank_public_uplinks(
    observation: RiskObservation, *, mode: UplinkMode,
) -> tuple[PublicUplinkRank, ...]:
    """Rank current local reports without inspecting their unreported object state.

    An uplink at t is at RSU at t+1; a subsequent downlink is usable at t+2.
    Demand counts distinct receivers, not tasks or offline related-object IDs.
    No future contact guarantee is assumed. An unknown RSU version is -1.
    """
    if mode not in {"version", "demand"}:
        raise ValueError("Unknown public uplink mode.")
    rsu_versions = {snapshot.object_id: snapshot.observation_step
                    for snapshot in observation.rsu_snapshots}
    visible = {state.vehicle_id: set(state.visible_object_ids)
               for state in observation.active_vehicles}
    ranks = []
    for candidate in dict.fromkeys(observation.uplink_candidates):
        source = receiver_version(observation, candidate.sender_id, candidate.object_id)
        rsu = rsu_versions.get(candidate.object_id, -1)
        beneficiaries = {
            task.vehicle_id for task in observation.tasks
            if task.decision_step >= observation.step_index + 2
            and task.vehicle_id != candidate.object_id
            and receiver_version(observation, task.vehicle_id, candidate.object_id) < source
        }
        deadline = min((task.decision_step for task in observation.tasks
                        if task.vehicle_id in beneficiaries
                        and task.decision_step >= observation.step_index + 2), default=None)
        reason = "eligible"
        if (candidate.object_id == candidate.sender_id
                or candidate.object_id not in visible.get(candidate.sender_id, set())
                or source != observation.step_index):
            reason = "no_confirmed_current_local_version"
        elif source <= rsu:
            reason = "rsu_version_not_older"
        elif mode == "demand" and not beneficiaries:
            reason = "no_public_two_hop_receiver_demand"
        gain = source - rsu
        key = ((-len(beneficiaries), deadline if deadline is not None else 2**63, -gain,
                candidate.sender_id, candidate.object_id) if mode == "demand"
               else (0, 0, -gain, candidate.sender_id, candidate.object_id))
        ranks.append(PublicUplinkRank(candidate, source, rsu, tuple(sorted(beneficiaries)),
                                      deadline, reason == "eligible", reason, key))
    return tuple(sorted(ranks, key=lambda rank: rank.key))


def select_public_uplink_action(
    observation: RiskObservation, config: RiskCommConfig, *, mode: UplinkMode,
    with_relay: bool = False,
) -> RiskSchedulingAction:
    """Keep receiver-value downlinks/relay order; replace only uplink selection."""
    if not any(task.decision_step > observation.step_index for task in observation.tasks):
        return RiskSchedulingAction()
    base_selector = (select_receiver_value_with_relay_action if with_relay
                     else select_receiver_value_action)
    base = base_selector(observation, config)
    occupied = {node for relay in base.relays for node in (relay.sender_id, relay.receiver_id)}
    remaining = replace(observation, uplink_candidates=tuple(
        c for c in observation.uplink_candidates if c.sender_id not in occupied))
    action = _build_feasible_action(
        remaining, config, downlinks=base.downlinks,
        uplinks=(rank.candidate for rank in rank_public_uplinks(observation, mode=mode)
                 if rank.eligible),
    )
    return replace(action, relays=base.relays)
