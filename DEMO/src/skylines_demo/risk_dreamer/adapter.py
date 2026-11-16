"""Quadratic public relations, linear nodes, and lossless S0 message assembly."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, isfinite, radians, sin
from typing import Iterator

import numpy as np

from ..config import RiskCommConfig
from ..risk_learning.domain import CandidateKey, build_safe_step
from ..risk_learning.encoding import encode_step
from ..types import (RSU_INFORMATION_VERSION, DownlinkAction, RelayAction,
                     RiskObservation, RiskSchedulingAction, UplinkAction)

ENCODING_VERSION = "risk-pair-67-node-16-v1"
PAIR_WIDTH, NODE_WIDTH = 67, 16
KNOWLEDGE_AGE, KNOWLEDGE_KNOWN, KNOWLEDGE_LOCAL = 64, 65, 66
NODE_ACTIVE, NODE_TASKS, NODE_DEADLINE, NODE_FUTURE = 12, 13, 14, 15


@dataclass(frozen=True)
class Catalogue:
    """Row/vehicle correspondence, with no stored message-triple vocabulary.

    candidates[i,j]: U(i,j)[0:32], D(i,j)[32:64], owner/object knowledge[64:67].
    legal[i,j]: U(i,j), D(i,j), R contact(i,j). Relay messages additionally
    require a newer source version, a future receiver task and distinct IDs.
    """
    vehicle_ids: tuple[int, ...]
    relay: bool

    def __post_init__(self) -> None:
        if (type(self.relay) is not bool or not self.vehicle_ids
                or any(type(i) is not int or i <= 0 for i in self.vehicle_ids)
                or len(set(self.vehicle_ids)) != len(self.vehicle_ids)):
            raise ValueError("Require nonempty unique positive vehicle IDs and a boolean relay flag.")
        object.__setattr__(self, "vehicle_ids", tuple(sorted(self.vehicle_ids)))

    @classmethod
    def from_initial_observation(cls, obs: RiskObservation, relay: bool) -> Catalogue:
        if obs.step_index != 0:
            raise ValueError("Build row correspondence from the initial observation only.")
        return cls(tuple(v.vehicle_id for v in obs.active_vehicles), relay)

    @property
    def node_count(self) -> int:
        return len(self.vehicle_ids)

    @property
    def action_size(self) -> int:
        # Slots contain kind (0 pad, 1 U, 2 D, 3 R), sender, receiver, object.
        # Row indices are one-based; 0 denotes RSU or padding, never an ID input.
        return (4 + int(self.relay)) * 4

    def _validate_ids(self, obs: RiskObservation) -> None:
        seen = {v.vehicle_id for v in obs.active_vehicles}
        for vehicle in obs.active_vehicles:
            seen.update(vehicle.visible_object_ids)
            seen.update(vehicle.known_object_ids)
            seen.update(vehicle.data_plane_object_ids)
        seen.update(task.vehicle_id for task in obs.tasks)
        seen.update(obs.rsu_known_object_ids)
        seen.update(obs.rsu_visible_object_ids)
        for snapshot in obs.rsu_snapshots:
            seen.update((snapshot.object_id, snapshot.source_id))
        for receiver, obj, _ in obs.receiver_known_observation_steps:
            seen.update((receiver, obj))
        for receiver, snapshot in obs.last_sent_snapshots:
            seen.update((receiver, snapshot.object_id, snapshot.source_id))
        for action in obs.uplink_candidates:
            seen.update((action.sender_id, action.object_id))
        for action in obs.downlink_candidates:
            seen.update((action.receiver_id, action.object_id))
        for action in obs.relay_candidates:
            seen.update((action.sender_id, action.receiver_id, action.object_id))
        if seen - {0} - set(self.vehicle_ids):
            raise ValueError("Observation contains IDs outside the initial catalogue.")

    def encode(self, obs: RiskObservation, config: RiskCommConfig) -> dict[str, np.ndarray]:
        if obs.information_contract != RSU_INFORMATION_VERSION:
            raise ValueError("Incompatible RSU information contract.")
        if config.enable_v2v_relay != self.relay:
            raise ValueError("Relay configuration differs from row metadata.")
        self._validate_ids(obs)
        safe = build_safe_step(replace(obs, relay_candidates=()), config)
        encoded = encode_step(safe)
        n = self.node_count
        indices = {vehicle_id: i for i, vehicle_id in enumerate(self.vehicle_ids)}
        candidates = np.zeros((n, n, PAIR_WIDTH), dtype=np.float32)
        legal = np.zeros((n, n, 3), dtype=bool)
        for row, (kind, sender, receiver, obj) in enumerate(encoded.keys):
            i, j = indices[sender if kind == "U" else receiver], indices[obj]
            channel = int(kind == "D")
            candidates[i, j, 32 * channel:32 * (channel + 1)] = encoded.candidate_features[row]
            legal[i, j, channel] = True
        for receiver, obj, version in obs.receiver_known_observation_steps:
            candidates[indices[receiver], indices[obj], KNOWLEDGE_AGE] = (obs.step_index - version) / 23.
            candidates[indices[receiver], indices[obj], KNOWLEDGE_KNOWN] = 1.
        for vehicle in obs.active_vehicles:
            for obj in vehicle.visible_object_ids:
                candidates[indices[vehicle.vehicle_id], indices[obj], KNOWLEDGE_LOCAL] = 1.
        if self.relay:
            for action in obs.relay_candidates:
                legal[indices[action.sender_id], indices[action.receiver_id], 2] = True
        nodes = np.zeros((n, NODE_WIDTH), dtype=np.float32)
        snapshots = {s.object_id: s for s in obs.rsu_snapshots}
        visible = set(obs.rsu_visible_object_ids)
        active = {v.vehicle_id for v in obs.active_vehicles}
        for i, vehicle_id in enumerate(self.vehicle_ids):
            snapshot = snapshots.get(vehicle_id)
            if snapshot is not None:
                if snapshot.observation_step > obs.step_index:
                    raise ValueError("RSU snapshot cannot originate in a future step.")
                vx, vz = snapshot.velocity_x, snapshot.velocity_z
                velocity_known = vx is not None and vz is not None and isfinite(vx) and isfinite(vz)
                heading = radians(snapshot.heading_deg)
                nodes[i, :12] = (
                    (snapshot.x - obs.rsu_position.x) / 120.,
                    (snapshot.y - obs.rsu_position.y) / 120.,
                    (snapshot.z - obs.rsu_position.z) / 120., sin(heading), cos(heading),
                    vx / 30. if velocity_known else 0., vz / 30. if velocity_known else 0.,
                    float(velocity_known), (obs.step_index - snapshot.observation_step) / 23.,
                    float(snapshot.source_id == 0), float(vehicle_id in visible), 1.,
                )
            future = [task for task in obs.tasks
                      if task.vehicle_id == vehicle_id and task.decision_step > obs.step_index]
            nodes[i, NODE_ACTIVE:] = (
                float(vehicle_id in active), len(future) / 46.,
                (min(task.decision_step for task in future) - obs.step_index) / 23. if future else 0.,
                float(bool(future)),
            )
        if not np.isfinite(nodes).all() or not np.isfinite(candidates).all():
            raise ValueError("Nonfinite public pair/node features.")
        return dict(candidates=candidates, legal=legal, nodes=nodes,
                    **{"global": encoded.global_features})

    def legal_messages(self, observation: dict[str, np.ndarray]) -> Iterator[CandidateKey]:
        """Audit iterator; never cached or used as a neural output vocabulary."""
        legal, pairs, nodes = observation["legal"], observation["candidates"], observation["nodes"]
        ids = self.vehicle_ids
        for channel, kind in enumerate(("U", "D")):
            for i, j in zip(*np.where(legal[..., channel])):
                yield (kind, ids[i] if kind == "U" else 0, 0 if kind == "U" else ids[i], ids[j])
        if self.relay:
            for s, r in zip(*np.where(legal[..., 2])):
                if nodes[r, NODE_FUTURE] <= .5:
                    continue
                for o in range(self.node_count):
                    if (len({s, r, o}) == 3 and pairs[s, o, KNOWLEDGE_KNOWN] > .5
                            and (pairs[r, o, KNOWLEDGE_KNOWN] <= .5
                                 or pairs[s, o, KNOWLEDGE_AGE] < pairs[r, o, KNOWLEDGE_AGE])):
                        yield ("R", ids[s], ids[r], ids[o])

    def to_action(self, selected: tuple[CandidateKey, ...]) -> RiskSchedulingAction:
        counts = {"U": 0, "D": 0, "R": 0}
        occupied: set[int] = set()
        ids = set(self.vehicle_ids)
        uplinks, downlinks, relays = [], [], []
        for kind, sender, receiver, obj in selected:
            if (kind not in counts or obj not in ids
                    or (kind == "U" and (sender not in ids or receiver != 0 or sender == obj))
                    or (kind == "D" and (sender != 0 or receiver not in ids or receiver == obj))
                    or (kind == "R" and (not self.relay or sender not in ids or receiver not in ids
                                         or len({sender, receiver, obj}) != 3))):
                raise ValueError("Invalid message key.")
            endpoints = {node for node in (sender, receiver) if node}
            if occupied & endpoints:
                raise ValueError("A vehicle cannot participate in multiple links.")
            occupied.update(endpoints)
            counts[kind] += 1
            if counts[kind] > {"U": 2, "D": 2, "R": int(self.relay)}[kind]:
                raise ValueError("Selection exceeds the 2U+2D(+1R) slots.")
            if kind == "U":
                uplinks.append(UplinkAction(sender, obj))
            elif kind == "D":
                downlinks.append(DownlinkAction(receiver, obj))
            else:
                relays.append(RelayAction(sender, receiver, obj))
        return RiskSchedulingAction(tuple(uplinks), tuple(downlinks), tuple(relays))

    def validate_selection(self, observation: dict[str, np.ndarray], selected: tuple[CandidateKey, ...]) -> None:
        self.to_action(selected)
        indices = {vehicle_id: i for i, vehicle_id in enumerate(self.vehicle_ids)}
        legal, pairs, nodes = observation["legal"], observation["candidates"], observation["nodes"]
        for kind, sender, receiver, obj in selected:
            o = indices[obj]
            if kind in ("U", "D"):
                valid = legal[indices[sender if kind == "U" else receiver], o, int(kind == "D")]
            else:
                s, r = indices[sender], indices[receiver]
                valid = (legal[s, r, 2] and nodes[r, NODE_FUTURE] > .5
                         and pairs[s, o, KNOWLEDGE_KNOWN] > .5
                         and (pairs[r, o, KNOWLEDGE_KNOWN] <= .5
                              or pairs[s, o, KNOWLEDGE_AGE] < pairs[r, o, KNOWLEDGE_AGE]))
            if not valid:
                raise ValueError("Selection contains messages outside the current S0 domain.")
