from __future__ import annotations

from dataclasses import dataclass

from ..config import ResourceConfig
from ..types import FrameRecord, Vector3
from ..wireless.interfaces import LinkEvaluator


@dataclass(frozen=True)
class SnapshotCommunication:
    """Snapshot contact rule: channel reachability, without packet/RB budgets."""

    channel: LinkEvaluator
    resource: ResourceConfig

    def reachable(
        self, start: Vector3, end: Vector3, sender_id: int,
        blockers: tuple[Vector3, ...] = (),
    ) -> bool:
        power = self.resource.rsu_tx_power_dbm if sender_id == 0 else self.resource.vehicle_tx_power_dbm
        return self.channel.evaluate(
            start, end, blockers=blockers, tx_power_dbm=power,
        ).reachable

    def reachable_nodes(self, frame: FrameRecord, rsu_position: Vector3,
                        sender_id: int, receiver_id: int) -> bool:
        if any(node != 0 and node not in frame.vehicles for node in (sender_id, receiver_id)):
            return False
        start = rsu_position if sender_id == 0 else frame.vehicles[sender_id].position
        end = rsu_position if receiver_id == 0 else frame.vehicles[receiver_id].position
        blockers = tuple(state.position for node, state in sorted(frame.vehicles.items())
                         if node not in (sender_id, receiver_id))
        return self.reachable(start, end, sender_id, blockers)
