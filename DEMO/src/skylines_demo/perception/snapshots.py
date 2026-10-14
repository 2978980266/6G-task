from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import mean

from ..types import RiskSnapshot, VehicleSnapshot


def build_snapshot(
    object_id: int,
    source_id: int,
    observation_step: int,
    current: VehicleSnapshot,
    previous: VehicleSnapshot | None,
    current_time_seconds: float,
    previous_time_seconds: float | None,
    received_step: int | None = None,
    previous_observation_step: int | None = None,
    velocity_basis: str | None = None,
) -> RiskSnapshot:
    if current.vehicle_id != object_id:
        raise ValueError("current vehicle id must match object_id.")
    velocity_x: float | None = None
    velocity_z: float | None = None
    if previous is not None:
        if previous.vehicle_id != object_id:
            raise ValueError("previous vehicle id must match object_id.")
        if previous_time_seconds is None:
            raise ValueError("previous_time_seconds is required with previous snapshot.")
        delta_seconds = current_time_seconds - previous_time_seconds
        if delta_seconds <= 0.0:
            raise ValueError("observation times must increase.")
        velocity_x = (
            current.position.x - previous.position.x
        ) / delta_seconds
        velocity_z = (
            current.position.z - previous.position.z
        ) / delta_seconds

    return RiskSnapshot(
        object_id=object_id,
        source_id=source_id,
        observation_step=observation_step,
        x=current.position.x,
        y=current.position.y,
        z=current.position.z,
        speed=current.speed,
        heading_deg=current.angle_y,
        velocity_x=velocity_x,
        velocity_z=velocity_z,
        received_step=received_step,
        velocity_previous_observation_step=previous_observation_step,
        velocity_basis=velocity_basis if previous is not None else None,
    )


@dataclass
class KnowledgeCache:
    owner_id: int
    _local_snapshots: dict[int, RiskSnapshot] = field(default_factory=dict, init=False)
    _received_snapshots: dict[int, RiskSnapshot] = field(default_factory=dict, init=False)

    def begin_step(self) -> None:
        """Expire local visibility; received versions retain their provenance."""
        self._local_snapshots.clear()

    def observe_local(self, snapshot: RiskSnapshot) -> None:
        if snapshot.source_id != self.owner_id:
            raise ValueError("local snapshot source_id must equal the cache owner.")
        current = self._local_snapshots.get(snapshot.object_id)
        if current is None or snapshot.observation_step > current.observation_step:
            self._local_snapshots[snapshot.object_id] = snapshot

    def receive(self, snapshot: RiskSnapshot, received_step: int) -> None:
        if received_step < snapshot.observation_step:
            raise ValueError("received_step cannot precede observation_step.")
        received_snapshot = replace(snapshot, received_step=received_step)
        current = self._received_snapshots.get(snapshot.object_id)
        if current is None or snapshot.observation_step > current.observation_step:
            self._received_snapshots[snapshot.object_id] = received_snapshot

    def snapshot_for(self, object_id: int) -> RiskSnapshot:
        local = self._local_snapshots.get(object_id)
        received = self._received_snapshots.get(object_id)
        if local is not None and (received is None or local.observation_step > received.observation_step):
            return local
        if received is not None:
            return received
        raise KeyError(f"Object {object_id} is not known by vehicle {self.owner_id}.")

    def received_snapshot_for(self, object_id: int) -> RiskSnapshot:
        return self._received_snapshots[object_id]

    @property
    def known_object_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._local_snapshots.keys() | self._received_snapshots.keys()))

    @property
    def data_plane_object_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._received_snapshots))

    def caoi_seconds(self, current_step: int) -> float | None:
        snapshots = [
            self._received_snapshots[object_id]
            for object_id in sorted(self._received_snapshots)
        ]
        if not snapshots:
            return None
        return mean(snapshot.age_at(current_step) for snapshot in snapshots)
