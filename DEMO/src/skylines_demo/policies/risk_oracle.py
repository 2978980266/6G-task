from __future__ import annotations

from ..config import RiskCommConfig
from ..perception.communication import SnapshotCommunication
from ..risk.conflicts import ConflictCatalog
from ..risk.sufficiency import receiver_version
from ..types import RiskObservation, RiskSchedulingAction, VehicleReplay
from .risk_baselines import _build_feasible_action


def select_diagnostic_oracle_action(
    observation: RiskObservation, config: RiskCommConfig, catalog: ConflictCatalog,
    *, replay: VehicleReplay | None = None,
    communication: SnapshotCommunication | None = None,
) -> RiskSchedulingAction:
    """Truth-enabled diagnostic only; each gap belongs to a receiver and task."""
    current = observation.step_index
    active_task_ids = {task.task_id for task in observation.tasks}
    tasks = [task for task in catalog.tasks if task.task_id in active_task_ids
             and task.true_risk and task.decision_step > current]
    snapshots = {snapshot.object_id: snapshot for snapshot in observation.rsu_snapshots}

    def downlink_deadline(candidate):
        snapshot = snapshots.get(candidate.object_id)
        # Public candidates already guarantee a source; synthetic callers must
        # also provide its version so confirmed updates can be checked.
        version = snapshot.observation_step if snapshot else -1
        if version <= receiver_version(observation, candidate.receiver_id, candidate.object_id):
            return None
        return min((task.decision_step for task in tasks
                    if task.vehicle_id == candidate.receiver_id
                    and candidate.object_id in task.related_vehicle_ids
                    and current + 1 <= task.decision_step), default=None)

    def uplink_deadline(candidate):
        if config.downlink_slots_per_step <= 0:
            return None
        rsu_snapshot = snapshots.get(candidate.object_id)
        # Current local sensing expires at the next step. Only a data-plane
        # reception proves that this version is already retained by the RSU.
        rsu_version = (rsu_snapshot.observation_step
                       if rsu_snapshot is not None and rsu_snapshot.received_step is not None else -1)
        if rsu_version >= current:
            return None
        deadlines = []
        for task in tasks:
            if candidate.object_id not in task.related_vehicle_ids or current + 2 > task.decision_step:
                continue
            if receiver_version(observation, task.vehicle_id, candidate.object_id) >= current:
                continue
            if replay is not None and communication is not None:
                if not any(communication.reachable_nodes(frame, observation.rsu_position, 0, task.vehicle_id)
                           for frame in replay.frames[current + 1:task.decision_step]):
                    continue
            deadlines.append(task.decision_step)
        return min(deadlines, default=None)

    down = {candidate: downlink_deadline(candidate) for candidate in observation.downlink_candidates}
    up = {candidate: uplink_deadline(candidate) for candidate in observation.uplink_candidates}
    return _build_feasible_action(
        observation, config,
        downlinks=sorted((c for c, d in down.items() if d is not None),
                         key=lambda c: (down[c], c.receiver_id, c.object_id)),
        uplinks=sorted((c for c, d in up.items() if d is not None),
                       key=lambda c: (up[c], c.sender_id, c.object_id)),
    )
