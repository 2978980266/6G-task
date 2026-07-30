from __future__ import annotations

import random
from typing import Protocol

from ..types import Observation


class SchedulingPolicy(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def select_action(self, observation: Observation) -> int | None:
        ...


class RandomReachablePolicy:
    name = "random_reachable"

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return None

    def select_action(self, observation: Observation) -> int | None:
        candidates = [
            vehicle.vehicle_id
            for vehicle in observation.active_vehicles
            if vehicle.link_state.reachable
        ]
        if not candidates:
            return None
        return self._rng.choice(candidates)


class RoundRobinReachablePolicy:
    name = "round_robin_reachable"

    def __init__(self) -> None:
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def select_action(self, observation: Observation) -> int | None:
        candidates = [
            vehicle.vehicle_id
            for vehicle in observation.active_vehicles
            if vehicle.link_state.reachable
        ]
        if not candidates:
            return None
        index = self._cursor % len(candidates)
        self._cursor += 1
        return candidates[index]


class MaxAoiReachablePolicy:
    name = "max_aoi_reachable"

    def reset(self) -> None:
        return None

    def select_action(self, observation: Observation) -> int | None:
        candidates = [
            vehicle
            for vehicle in observation.active_vehicles
            if vehicle.link_state.reachable
        ]
        if not candidates:
            return None
        chosen = max(candidates, key=lambda vehicle: (vehicle.aoi_seconds, vehicle.link_state.distance_m))
        return chosen.vehicle_id


def default_policies(seed: int = 7) -> list[SchedulingPolicy]:
    return [
        RandomReachablePolicy(seed=seed),
        RoundRobinReachablePolicy(),
        MaxAoiReachablePolicy(),
    ]
