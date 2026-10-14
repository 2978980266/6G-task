from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from ..config import ResourceConfig, RiskCommConfig
from ..perception.communication import SnapshotCommunication
from ..perception.sensors import SensorModel
from ..perception.snapshots import build_snapshot
from ..types import RiskSnapshot, Vector3, VehicleReplay
from ..wireless.interfaces import LineOfSightEvaluator, LinkEvaluator
from .conflicts import ConflictCatalog


@dataclass(frozen=True)
class _Holding:
    snapshot: RiskSnapshot
    path: tuple[dict[str, int], ...] = ()


def _object_snapshots(
    replay: VehicleReplay, seen: tuple[dict[int, set[int]], ...], obj: int,
) -> tuple[dict[int, RiskSnapshot], ...]:
    """Use the same actual observation history as the environment for velocity."""
    history: dict[int, tuple[Any, float, int]] = {}
    result = []
    for step, frame in enumerate(replay.frames):
        local = {}
        for node, objects in seen[step].items():
            if obj not in objects:
                continue
            previous, previous_time, previous_step = history.get(node, (None, None, None))
            local[node] = build_snapshot(obj, node, step, frame.vehicles[obj], previous,
                                         frame.elapsed_seconds, previous_time,
                                         previous_observation_step=previous_step,
                                         velocity_basis="local_history")
            history[node] = frame.vehicles[obj], frame.elapsed_seconds, step
        result.append(local)
    return tuple(result)


def _delivery_paths(
    replay: VehicleReplay, local: tuple[dict[int, RiskSnapshot], ...],
    receiver: int, obj: int, deadline: int, contact: SnapshotCommunication,
    rsu_position: Vector3, config: RiskCommConfig, *, allow_relay: bool,
) -> list[_Holding]:
    """Search alternative single-object paths; never pool them into one cache.

    A state retains one original version and its prior received nodes.  Each
    proposed edge is made available only on the following step.  Distinct
    histories remain alternatives, so every returned path is independently
    executable with at most one data link in any step.
    """
    received: dict[tuple[int, int, int, frozenset[int]], _Holding] = {}
    delivered: list[_Holding] = []
    for step in range(deadline):
        frame = replay.frames[step]
        available = [(node, _Holding(snapshot)) for node, snapshot in sorted(local[step].items())]
        available.extend((key[0], holding) for key, holding in received.items()
                         if key[0] == 0 or key[0] in frame.vehicles)
        updates: dict[tuple[int, int, int, frozenset[int]], _Holding] = {}
        for sender, holding in available:
            snapshot = holding.snapshot
            current_local = local[step].get(sender)
            if current_local is not None and current_local.observation_step > snapshot.observation_step:
                continue
            prior_receivers = frozenset(edge['receiver_id'] for edge in holding.path)
            targets = []
            # The environment accepts only a current local version for uplink.
            if sender != 0 and not holding.path and config.uplink_slots_per_step > 0:
                targets.append(0)
            if sender == 0 and config.downlink_slots_per_step > 0:
                targets.extend(sorted(frame.vehicles))
            if (sender != 0 and allow_relay and config.enable_v2v_relay
                    and config.relay_slots_per_step > 0):
                targets.extend(sorted(frame.vehicles))
            for target in targets:
                if target in (sender, obj) or target in prior_receivers:
                    continue
                target_local = local[step].get(target)
                # Uplink can preserve an equal locally visible RSU version in
                # its received cache; local visibility expires next step.
                if (target != 0 and target_local is not None
                        and target_local.observation_step >= snapshot.observation_step):
                    continue
                if not contact.reachable_nodes(frame, rsu_position, sender, target):
                    continue
                edge = {'sender_id': sender, 'receiver_id': target, 'object_id': obj,
                        'send_step': step, 'usable_step': step + 1,
                        'source_id': snapshot.source_id, 'observation_step': snapshot.observation_step}
                updated = _Holding(replace(snapshot, received_step=step + 1), (*holding.path, edge))
                key = (target, snapshot.source_id, snapshot.observation_step, prior_receivers | {target})
                # The first arrival with the same received-node history can
                # wait for every later contact without another transmission.
                if key not in received:
                    updates.setdefault(key, updated)
                if target == receiver:
                    delivered.append(updated)
        received.update(updates)
    return delivered


def _visibility(replay: VehicleReplay, sensor: SensorModel) -> tuple[dict[int, set[int]], ...]:
    return tuple({observer: set(sensor.visible_object_ids(observer, frame))
                  for observer in (0, *sorted(frame.vehicles))} for frame in replay.frames)


def build_perception_opportunities(
    replay: VehicleReplay, catalog: ConflictCatalog, sensor: SensorModel,
) -> dict[str, Any]:
    """Diagnostic visibility only; separate decision-frame and schedulable history."""
    visibility = _visibility(replay, sensor)
    before: list[dict[str, Any]] = []
    decision: list[dict[str, Any]] = []
    for task in catalog.tasks:
        for step in range(task.decision_step + 1):
            frame = replay.frames[step]
            seen = visibility[step]
            if task.vehicle_id not in frame.vehicles:
                continue
            for obj in sorted(frame.vehicles):
                if obj == task.vehicle_id or obj in seen[task.vehicle_id]:
                    continue
                helpers = tuple(observer for observer in sorted(frame.vehicles)
                                if observer != task.vehicle_id and obj in seen[observer])
                rsu_visible = obj in seen[0]
                if not rsu_visible and not helpers:
                    continue
                record = {"task_id": task.task_id, "receiver_id": task.vehicle_id,
                          "object_id": obj, "step": step, "decision_step": task.decision_step,
                          "rsu_visible": rsu_visible, "helper_ids": helpers}
                (decision if step == task.decision_step else before).append(record)

    def counts(records, rsu_visible):
        selected = [r for r in records if r["rsu_visible"] == rsu_visible]
        return {"task_count": len({r["task_id"] for r in selected}),
                "task_object_count": len({(r["task_id"], r["receiver_id"], r["object_id"]) for r in selected}),
                "step_opportunity_count": len(selected)}

    return {"before_decision_rsu_visibility": counts(before, True),
            "before_decision_helper_visibility": counts(before, False),
            "decision_frame_rsu_visibility": counts(decision, True),
            "decision_frame_helper_visibility": counts(decision, False),
            "visibility_opportunity_records": before,
            "decision_frame_visibility_records": decision}


def build_delivery_opportunities(
    replay: VehicleReplay, catalog: ConflictCatalog, rsu_position: Vector3,
    los_model: LineOfSightEvaluator, channel_model: LinkEvaluator,
    resource_config: ResourceConfig, config: RiskCommConfig,
) -> dict[str, Any]:
    """Exact isolated-path diagnostics, not jointly achievable throughput.

    Each witness is one object along a legal store-and-forward path: one edge
    per step, nonzero slots, reachable endpoints and next-step availability.
    Paths for different tasks are alternatives, never added as simultaneous
    capacity.  Only this diagnostic reads future scene geometry and truth.
    """
    sensor = SensorModel(rsu_position, los_model, config)
    seen = _visibility(replay, sensor)
    contact = SnapshotCommunication(channel_model, resource_config)
    records: list[dict[str, Any]] = []
    late: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    object_history: dict[int, tuple[dict[int, RiskSnapshot], ...]] = {}
    for task in catalog.tasks:
        for obj in sorted(set(task.related_vehicle_ids)):
            receiver = task.vehicle_id
            # A local object at the decision does not need a communication rescue.
            local_at_decision = obj in seen[task.decision_step].get(receiver, set())
            if obj not in object_history:
                object_history[obj] = _object_snapshots(replay, seen, obj)
            local = object_history[obj]
            direct_paths = _delivery_paths(replay, local, receiver, obj, task.decision_step,
                contact, rsu_position, config, allow_relay=False)
            first_direct = min((h.path[-1]['usable_step'] for h in direct_paths), default=None)
            direct_late = not local_at_decision and first_direct is None
            pair = {"task_id": task.task_id, "receiver_id": receiver, "object_id": obj,
                    "decision_step": task.decision_step, "receiver_local_at_decision": local_at_decision,
                    "earliest_direct_usable_step": first_direct, "direct_late": direct_late}
            pair_results.append(pair)
            if direct_late:
                late.append(pair)
                relay_paths = _delivery_paths(replay, local, receiver, obj, task.decision_step,
                    contact, rsu_position, config, allow_relay=True)
                # Multiple helpers in a step represent one receiver/object opportunity.
                unique_steps: dict[int, dict[str, Any]] = {}
                for holding in sorted(relay_paths, key=lambda h: (-h.snapshot.observation_step, len(h.path))):
                    edge = holding.path[-1]
                    if edge['sender_id'] == 0 or obj in seen[edge['send_step']][0]:
                        continue
                    record = {"task_id": task.task_id, "receiver_id": receiver,
                              "object_id": obj, "sender_id": edge['sender_id'],
                              "send_step": edge['send_step'], "usable_step": edge['usable_step'],
                              "decision_step": task.decision_step, "source_id": holding.snapshot.source_id,
                              "observation_step": holding.snapshot.observation_step,
                              "snapshot": asdict(holding.snapshot), "path": holding.path}
                    unique_steps.setdefault(edge["send_step"], record)
                records.extend(unique_steps.values())
    return {
        "opportunity_scope": "isolated_legal_paths_not_joint_capacity",
        "direct_delivery_late_task_count": len({r["task_id"] for r in late}),
        "direct_delivery_late_task_object_count": len(late),
        "relay_can_deliver_early_task_count": len({r["task_id"] for r in records}),
        "relay_benefit_task_object_count": len({(r["task_id"], r["receiver_id"], r["object_id"]) for r in records}),
        "relay_benefit_step_opportunity_count": len(records),
        "rsu_invisible_other_vehicle_visible_task_count": len({r["task_id"] for r in records}),
        "relay_benefit_records": records,
        "delivery_opportunity_pairs": pair_results,
    }
