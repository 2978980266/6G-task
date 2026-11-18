"""Episode-local causal identities over an immutable fixed slot catalogue."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import RiskCommConfig
from ..risk_learning.domain import CandidateKey
from ..types import RiskObservation, RiskSchedulingAction, RiskSnapshot
from .adapter import Catalogue


class DynamicCatalogue:
    """Bind public vehicle IDs on first observation, retaining slots until reset.

    Pass ``catalogue`` to DreamerAgent, ``encode`` observations here, then map
    each batch item's ActorOutput.selections back with ``to_action``. Capacity
    is an episode-wide distinct-ID budget, including retained public state.
    """

    def __init__(self, capacity: int, relay: bool) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.catalogue = Catalogue(tuple(range(1, capacity + 1)), relay)
        self._actual_to_slot: dict[int, int] = {}
        self._slot_to_actual: dict[int, int] = {}

    @property
    def node_count(self) -> int:
        return self.catalogue.node_count

    @property
    def action_size(self) -> int:
        return self.catalogue.action_size

    @property
    def relay(self) -> bool:
        return self.catalogue.relay

    def reset(self) -> None:
        """Clear episode identities while preserving the agent's catalogue."""
        self._actual_to_slot.clear()
        self._slot_to_actual.clear()

    @staticmethod
    def _observed_ids(obs: RiskObservation) -> set[int]:
        seen = set(obs.rsu_known_object_ids) | set(obs.rsu_visible_object_ids)
        for vehicle in obs.active_vehicles:
            seen.add(vehicle.vehicle_id)
            seen.update(vehicle.visible_object_ids)
            seen.update(vehicle.known_object_ids)
            seen.update(vehicle.data_plane_object_ids)
        seen.update(task.vehicle_id for task in obs.tasks)
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
        for receiver, obj, forecast in obs.link_forecasts:
            seen.update((receiver, obj, forecast.receiver_id))
        if any(type(vehicle_id) is not int or vehicle_id < 0 for vehicle_id in seen):
            raise ValueError("Observation IDs must be nonnegative integers (0 is RSU).")
        return seen - {0}

    def encode(self, obs: RiskObservation, config: RiskCommConfig) -> dict[str, np.ndarray]:
        """Encode only the supplied public observation; never consult replay."""
        new_ids = sorted(self._observed_ids(obs) - self._actual_to_slot.keys())
        if len(self._actual_to_slot) + len(new_ids) > self.node_count:
            raise ValueError("Capacity exhausted by distinct observed vehicle IDs in this episode.")
        # Stage all bindings so exhaustion or an encoder rejection is atomic.
        mapping = dict(self._actual_to_slot)
        for vehicle_id in new_ids:
            mapping[vehicle_id] = len(mapping) + 1

        def slot(vehicle_id: int) -> int:
            return 0 if vehicle_id == 0 else mapping[vehicle_id]

        def snapshot(value: RiskSnapshot) -> RiskSnapshot:
            return replace(value, object_id=slot(value.object_id), source_id=slot(value.source_id))

        remapped = replace(
            obs,
            active_vehicles=tuple(replace(
                vehicle, vehicle_id=slot(vehicle.vehicle_id),
                visible_object_ids=tuple(map(slot, vehicle.visible_object_ids)),
                known_object_ids=tuple(map(slot, vehicle.known_object_ids)),
                data_plane_object_ids=tuple(map(slot, vehicle.data_plane_object_ids)),
            ) for vehicle in obs.active_vehicles),
            tasks=tuple(replace(task, vehicle_id=slot(task.vehicle_id)) for task in obs.tasks),
            rsu_known_object_ids=tuple(map(slot, obs.rsu_known_object_ids)),
            rsu_visible_object_ids=tuple(map(slot, obs.rsu_visible_object_ids)),
            rsu_snapshots=tuple(map(snapshot, obs.rsu_snapshots)),
            receiver_known_observation_steps=tuple(
                (slot(receiver), slot(obj), version)
                for receiver, obj, version in obs.receiver_known_observation_steps),
            last_sent_snapshots=tuple(
                (slot(receiver), snapshot(value)) for receiver, value in obs.last_sent_snapshots),
            uplink_candidates=tuple(replace(
                action, sender_id=slot(action.sender_id), object_id=slot(action.object_id),
            ) for action in obs.uplink_candidates),
            downlink_candidates=tuple(replace(
                action, receiver_id=slot(action.receiver_id), object_id=slot(action.object_id),
            ) for action in obs.downlink_candidates),
            relay_candidates=tuple(replace(
                action, sender_id=slot(action.sender_id), receiver_id=slot(action.receiver_id),
                object_id=slot(action.object_id),
            ) for action in obs.relay_candidates),
            link_forecasts=tuple(
                (slot(receiver), slot(obj), replace(forecast, receiver_id=slot(forecast.receiver_id)))
                for receiver, obj, forecast in obs.link_forecasts),
        )
        encoded = self.catalogue.encode(remapped, config)
        self._actual_to_slot = mapping
        self._slot_to_actual = {value: key for key, value in mapping.items()}
        return encoded

    def to_action(self, selections: tuple[CandidateKey, ...]) -> RiskSchedulingAction:
        """Map one ActorOutput batch item's slot keys into actual vehicle IDs."""
        if any(type(node) is not int for _, sender, receiver, obj in selections
               for node in (sender, receiver, obj)):
            raise ValueError("Message slot IDs must be integers.")
        action = self.catalogue.to_action(selections)

        def actual(slot_id: int) -> int:
            try:
                return self._slot_to_actual[slot_id]
            except KeyError as error:
                raise ValueError(f"Unbound vehicle slot: {slot_id}.") from error

        return replace(
            action,
            uplinks=tuple(replace(item, sender_id=actual(item.sender_id),
                                  object_id=actual(item.object_id)) for item in action.uplinks),
            downlinks=tuple(replace(item, receiver_id=actual(item.receiver_id),
                                    object_id=actual(item.object_id)) for item in action.downlinks),
            relays=tuple(replace(item, sender_id=actual(item.sender_id),
                                 receiver_id=actual(item.receiver_id), object_id=actual(item.object_id))
                         for item in action.relays),
        )

    def validate_selection(self, encoded: dict[str, np.ndarray],
                           selections: tuple[CandidateKey, ...]) -> None:
        self.catalogue.validate_selection(encoded, selections)
