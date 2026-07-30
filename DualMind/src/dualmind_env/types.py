from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def horizontal_distance_to(self, other: "Vector3") -> float:
        return math.hypot(self.x - other.x, self.z - other.z)


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: int
    position: Vector3
    speed: float
    angle_x: float
    angle_y: float
    prefab_name: str


@dataclass(frozen=True)
class BuildingSnapshot:
    building_id: int
    position: Vector3
    angle_y: float
    width_cells: int
    length_cells: int
    size_x: float
    size_y: float
    size_z: float
    center_offset_x: float
    center_offset_y: float
    center_offset_z: float
    min_y: float
    max_y: float
    prefab_name: str


@dataclass(frozen=True)
class LinkState:
    vehicle_id: int
    distance_m: float
    los: bool
    path_loss_db: float
    snr_db: float
    rate_mbps: float
    reachable: bool


@dataclass(frozen=True)
class ActiveVehicleObservation:
    vehicle_id: int
    position: Vector3
    speed: float
    aoi_seconds: float
    link_state: LinkState
    prefab_name: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["position"] = asdict(self.position)
        payload["link_state"] = asdict(self.link_state)
        return payload


@dataclass(frozen=True)
class Observation:
    step_index: int
    simulation_frame: int
    game_time: str
    record_time_utc: str
    active_vehicles: tuple[ActiveVehicleObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "simulation_frame": self.simulation_frame,
            "game_time": self.game_time,
            "record_time_utc": self.record_time_utc,
            "active_vehicles": [vehicle.to_dict() for vehicle in self.active_vehicles],
        }
