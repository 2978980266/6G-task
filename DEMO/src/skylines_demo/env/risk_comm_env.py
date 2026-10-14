from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from ..bootstrap import ScenarioRuntime
from ..config import RiskCommConfig
from ..perception.sensors import SensorModel
from ..perception.snapshots import KnowledgeCache, build_snapshot
from ..perception.link_forecast import predict_downlink_window
from ..perception.communication import SnapshotCommunication
from ..perception.timing import estimate_step_seconds
from ..risk.conflicts import ConflictCatalog, RiskTask, build_conflict_catalog
from ..risk.evaluator import RiskEventResult, evaluate_decision
from ..risk.executor import RiskDecision, execute_risk
from ..risk.model import own_schedule_eta
from ..risk.scoring import RiskScoreLedger, RiskStepFeedback
from ..types import (
    DownlinkAction,
    RelayAction,
    RiskObservation,
    RiskSchedulingAction,
    RiskSnapshot,
    RiskTaskView,
    LinkForecast,
    RiskVehicleState,
    UplinkAction,
    VehicleSnapshot,
)


@dataclass(frozen=True)
class RiskDeliveryRecord:
    link_type: str
    sender_id: int
    receiver_id: int
    send_step: int
    usable_step: int
    snapshot: RiskSnapshot
    before_received_snapshot: RiskSnapshot | None
    after_received_snapshot: RiskSnapshot


@dataclass(frozen=True)
class RiskStepResult:
    observation: RiskObservation
    action: RiskSchedulingAction
    invalid_action: bool
    delivered_uplinks: tuple[RiskSnapshot, ...]
    delivered_downlinks: tuple[RiskSnapshot, ...]
    delivered_relays: tuple[RiskSnapshot, ...]
    risk_decisions: tuple[tuple[str, RiskDecision], ...]
    event_results: tuple[RiskEventResult, ...]
    mean_caoi_steps: float | None
    zero_receive_vehicle_count: int
    next_observation: RiskObservation | None
    feedback: RiskStepFeedback
    objective: dict[str, object]
    delivery_records: tuple[RiskDeliveryRecord, ...] = ()

    @property
    def reward(self) -> float:
        """Offline feedback; the current action cannot change this step's decision."""
        return self.feedback.reward


class RiskCommEnv:
    """Step-ordered perception-risk communication environment."""

    def __init__(
        self,
        runtime: ScenarioRuntime,
        risk_config: RiskCommConfig,
    ) -> None:
        self._runtime = runtime
        self._risk_config = risk_config
        self._replay = runtime.replay
        self._step_times = tuple(frame.elapsed_seconds for frame in self._replay.frames)
        self._communication = SnapshotCommunication(runtime.channel_model, runtime.config.resource)
        self._sensor = SensorModel(
            rsu_position=runtime.config.rsu.position,
            los_evaluator=runtime.los_model,
            config=risk_config,
        )
        self._catalog: ConflictCatalog = build_conflict_catalog(
            self._replay,
            risk_config,
        )
        self._tasks_by_step: dict[int, tuple[RiskTask, ...]] = {}
        for task in self._catalog.tasks:
            self._tasks_by_step.setdefault(task.decision_step, tuple())
            self._tasks_by_step[task.decision_step] = (
                *self._tasks_by_step[task.decision_step],
                task,
            )
        self._step_index = 0
        self._rsu_knowledge = KnowledgeCache(owner_id=0)
        self._vehicle_knowledge: dict[int, KnowledgeCache] = {}
        self._local_history: dict[tuple[int, int], tuple[VehicleSnapshot, float]] = {}
        self._self_history: dict[int, tuple[VehicleSnapshot, float]] = {}
        self._self_snapshots: dict[int, RiskSnapshot] = {}
        self._current_local_snapshots: dict[tuple[int, int], RiskSnapshot] = {}
        self._last_sent_snapshots: dict[tuple[int, int], RiskSnapshot] = {}
        self._current_observation: RiskObservation | None = None
        self._current_decisions: tuple[tuple[str, RiskDecision], ...] = ()
        self._score_ledger = RiskScoreLedger(self._catalog.tasks, len(self._replay))
        self._send_history: list[RiskDeliveryRecord] = []

    @property
    def catalog(self) -> ConflictCatalog:
        return self._catalog

    @property
    def current_observation(self) -> RiskObservation | None:
        return self._current_observation

    @property
    def objective_summary(self) -> dict[str, object]:
        """Evaluation-only totals, deliberately outside RiskObservation/caches."""
        summary = self._score_ledger.summary()
        steps_run = cast(int, summary["steps_run"])
        last_deadline = max((task.decision_step for task in self._catalog.tasks), default=-1)
        summary["after_all_task_deadlines_send_count"] = sum(
            record.usable_step > last_deadline for record in self._send_history
        )
        summary["repeated_or_older_received_version_send_count"] = sum(
            record.before_received_snapshot is not None
            and record.snapshot.observation_step <= record.before_received_snapshot.observation_step
            for record in self._send_history
        )
        for link_type, slots in (
            ("uplink", self._risk_config.uplink_slots_per_step),
            ("downlink", self._risk_config.downlink_slots_per_step),
            ("relay", self._risk_config.relay_slots_per_step if self._risk_config.enable_v2v_relay else 0),
        ):
            budget = steps_run * slots
            summary[f"{link_type}_slot_utilization"] = (
                cast(int, summary[f"{link_type}_count"]) / budget if budget else None
            )
        return summary

    def reset(self) -> RiskObservation:
        self._step_index = 0
        self._rsu_knowledge = KnowledgeCache(owner_id=0)
        self._vehicle_knowledge = {}
        self._local_history = {}
        self._self_history = {}
        self._self_snapshots = {}
        self._current_local_snapshots = {}
        self._last_sent_snapshots = {}
        self._current_observation = None
        self._current_decisions = ()
        self._score_ledger = RiskScoreLedger(self._catalog.tasks, len(self._replay))
        self._send_history = []
        return self._prepare_observation(self._step_index)

    def step(self, action: RiskSchedulingAction) -> RiskStepResult:
        if self._current_observation is None:
            raise RuntimeError("Call reset() before step().")

        observation = self._current_observation
        risk_decisions = self._current_decisions
        event_results = tuple(
            evaluate_decision(
                task=self._task_for_id(task_id),
                decision=decision,
            )
            for task_id, decision in risk_decisions
        )
        invalid_action = not self._is_valid_action(action, observation)
        delivered_uplinks: tuple[RiskSnapshot, ...] = ()
        delivered_downlinks: tuple[RiskSnapshot, ...] = ()
        delivered_relays: tuple[RiskSnapshot, ...] = ()
        delivery_records: tuple[RiskDeliveryRecord, ...] = ()
        if not invalid_action:
            delivered_uplinks, delivered_downlinks, delivered_relays, delivery_records = self._deliver(action)
        feedback = self._score_ledger.settle_step(
            observation.step_index, event_results,
            uplink_count=len(delivered_uplinks), downlink_count=len(delivered_downlinks),
            relay_count=len(delivered_relays),
        )
        self._send_history.extend(delivery_records)

        delivery_step = self._step_index + 1
        delivery_frame = self._replay.frames[min(delivery_step, len(self._replay) - 1)]
        for vehicle_id in delivery_frame.vehicles:
            self._vehicle_knowledge.setdefault(vehicle_id, KnowledgeCache(owner_id=vehicle_id))
        caoi_values = [
            self._vehicle_knowledge[vehicle_id].caoi_seconds(delivery_step)
            for vehicle_id in sorted(delivery_frame.vehicles)
        ]
        known_caoi_values = [value for value in caoi_values if value is not None]
        mean_caoi_steps = (
            sum(known_caoi_values) / len(known_caoi_values)
            if known_caoi_values
            else None
        )
        zero_receive_vehicle_count = sum(value is None for value in caoi_values)

        next_observation: RiskObservation | None = None
        if self._step_index + 1 < len(self._replay):
            self._step_index += 1
            next_observation = self._prepare_observation(self._step_index)
        else:
            self._current_observation = None
            self._current_decisions = ()

        return RiskStepResult(
            observation=observation,
            action=action,
            invalid_action=invalid_action,
            delivered_uplinks=delivered_uplinks,
            delivered_downlinks=delivered_downlinks,
            delivered_relays=delivered_relays,
            risk_decisions=risk_decisions,
            event_results=event_results,
            mean_caoi_steps=mean_caoi_steps,
            zero_receive_vehicle_count=zero_receive_vehicle_count,
            next_observation=next_observation,
            feedback=feedback,
            objective=self.objective_summary,
            delivery_records=delivery_records,
        )

    def _prepare_observation(self, step_index: int) -> RiskObservation:
        frame = self._replay.frames[step_index]
        active_vehicle_ids = tuple(sorted(frame.vehicles))
        for vehicle_id in active_vehicle_ids:
            self._vehicle_knowledge.setdefault(
                vehicle_id,
                KnowledgeCache(owner_id=vehicle_id),
            )

        self._current_local_snapshots = {}
        self._self_snapshots = {}
        for vehicle_id in active_vehicle_ids:
            previous_own = self._self_history.get(vehicle_id)
            self._self_snapshots[vehicle_id] = build_snapshot(
                vehicle_id, vehicle_id, step_index, frame.vehicles[vehicle_id],
                previous_own[0] if previous_own else None, frame.elapsed_seconds,
                previous_own[1] if previous_own else None,
                previous_observation_step=(self._step_times.index(previous_own[1]) if previous_own else None),
                velocity_basis="self_history",
            )
            self._self_history[vehicle_id] = (frame.vehicles[vehicle_id], frame.elapsed_seconds)
        self._rsu_knowledge.begin_step()
        for cache in self._vehicle_knowledge.values():
            cache.begin_step()
        observer_ids = (0,) + active_vehicle_ids
        rsu_visible_object_ids = self._sensor.visible_object_ids(0, frame)
        for observer_id in observer_ids:
            visible_object_ids = self._sensor.visible_object_ids(observer_id, frame)
            cache = self._rsu_knowledge if observer_id == 0 else self._vehicle_knowledge[observer_id]
            for object_id in visible_object_ids:
                target = frame.vehicles[object_id]
                previous_entry = self._local_history.get((observer_id, object_id))
                previous_snapshot = previous_entry[0] if previous_entry is not None else None
                previous_time = previous_entry[1] if previous_entry is not None else None
                velocity_basis = "local_history"
                # A retained actual reception can supply a past position when
                # this observer has never seen the object locally before.
                if object_id in cache.data_plane_object_ids:
                    received = cache.received_snapshot_for(object_id)
                    received_time = self._step_times[received.observation_step]
                    if (received_time < frame.elapsed_seconds
                            and (previous_time is None or received_time > previous_time)):
                        previous_snapshot = VehicleSnapshot(
                            object_id, received.position, received.speed, 0.0,
                            received.heading_deg, "received_snapshot",
                        )
                        previous_time = received_time
                        velocity_basis = "received_history"
                snapshot = build_snapshot(
                    object_id=object_id,
                    source_id=observer_id,
                    observation_step=step_index,
                    current=target,
                    previous=previous_snapshot,
                    current_time_seconds=frame.elapsed_seconds,
                    previous_time_seconds=previous_time,
                    previous_observation_step=(self._step_times.index(previous_time)
                                               if previous_time is not None else None),
                    velocity_basis=velocity_basis,
                )
                cache.observe_local(snapshot)
                self._current_local_snapshots[(observer_id, object_id)] = snapshot
                self._local_history[(observer_id, object_id)] = (
                    target,
                    frame.elapsed_seconds,
                )

        decision_tasks = self._tasks_by_step.get(step_index, ())
        risk_decisions = tuple(
            (
                task.task_id,
                execute_risk(
                    task=task,
                    knowledge=self._vehicle_knowledge[task.vehicle_id],
                    current_step=step_index,
                    current_time_seconds=frame.elapsed_seconds,
                    step_times=self._step_times[:step_index + 1],
                    config=self._risk_config,
                    ego_snapshot=self._self_snapshots[task.vehicle_id],
                    scheduled_ego_arrival_seconds=own_schedule_eta(
                        self._risk_config, task.crossing_step, step_index,
                        estimate_step_seconds(self._step_times[:step_index + 1],
                                              self._runtime.config.environment.fallback_step_seconds),
                    ),
                ),
            )
            for task in decision_tasks
        )
        self._current_decisions = risk_decisions

        pending_tasks = tuple(
            task
            for task in self._catalog.tasks
            if task.decision_step >= step_index
            and task.vehicle_id in active_vehicle_ids
        )

        uplink_candidates = tuple(
            UplinkAction(sender_id=observer_id, object_id=object_id)
            for observer_id in active_vehicle_ids
            for object_id in self._sensor.visible_object_ids(observer_id, frame)
            if self._link_reachable(observer_id, 0)
        )
        downlink_candidates = tuple(
            DownlinkAction(receiver_id=receiver_id, object_id=object_id)
            for receiver_id in active_vehicle_ids
            for object_id in self._rsu_knowledge.known_object_ids
            if object_id != receiver_id
            and self._link_reachable(0, receiver_id)
            and self._can_update(self._vehicle_knowledge[receiver_id], object_id)
        )
        relay_candidates = self._build_relay_candidates(active_vehicle_ids)
        link_forecasts: tuple[tuple[int, int, LinkForecast], ...] = ()
        if self._risk_config.link_prediction:
            link_forecasts = tuple(
                (
                    candidate.receiver_id,
                    candidate.object_id,
                    predict_downlink_window(
                        snapshot=self._rsu_knowledge.snapshot_for(candidate.object_id),
                        receiver_id=candidate.receiver_id,
                        current_step=step_index,
                        replay_history=self._replay,
                        rsu_position=self._runtime.config.rsu.position,
                        communication_model=self._communication,
                        fallback_step_seconds=self._runtime.config.environment.fallback_step_seconds,
                        config=self._risk_config,
                    ),
                )
                for candidate in downlink_candidates
            )
        task_views = tuple(
            RiskTaskView(
                task_id=task.task_id,
                vehicle_id=task.vehicle_id,
                conflict_point_id=task.conflict_point_id,
                conflict_point=task.conflict_point,
                crossing_step=task.crossing_step,
                decision_step=task.decision_step,
            )
            for task in pending_tasks
        )
        vehicle_states = tuple(
            RiskVehicleState(
                vehicle_id=vehicle_id,
                snapshot=frame.vehicles[vehicle_id],
                visible_object_ids=self._sensor.visible_object_ids(vehicle_id, frame),
                known_object_ids=self._vehicle_knowledge[vehicle_id].known_object_ids,
                data_plane_object_ids=self._vehicle_knowledge[vehicle_id].data_plane_object_ids,
                caoi_steps=self._vehicle_knowledge[vehicle_id].caoi_seconds(step_index),
                self_snapshot=self._self_snapshots[vehicle_id],
            )
            for vehicle_id in active_vehicle_ids
        )
        observation = RiskObservation(
            step_index=frame.step_index,
            simulation_frame=frame.simulation_frame,
            game_time=frame.game_time,
            record_time_utc=frame.record_time_utc,
            elapsed_seconds=frame.elapsed_seconds,
            delta_seconds=estimate_step_seconds(
                self._step_times[:step_index + 1],
                self._runtime.config.environment.fallback_step_seconds,
            ),
            active_vehicles=vehicle_states,
            tasks=task_views,
            uplink_candidates=uplink_candidates,
            downlink_candidates=downlink_candidates,
            relay_candidates=relay_candidates,
            rsu_known_object_ids=self._rsu_knowledge.known_object_ids,
            rsu_snapshots=tuple(
                self._rsu_knowledge.snapshot_for(object_id)
                for object_id in self._rsu_knowledge.known_object_ids
            ),
            receiver_known_observation_steps=tuple(
                (receiver_id, object_id, snapshot.observation_step)
                for receiver_id in active_vehicle_ids
                for object_id in self._vehicle_knowledge[receiver_id].known_object_ids
                for snapshot in (
                    self._vehicle_knowledge[receiver_id].snapshot_for(object_id),
                )
            ),
            last_sent_snapshots=tuple(
                (key[0], self._last_sent_snapshots[key])
                for key in sorted(self._last_sent_snapshots)
            ),
            step_times=self._step_times[:step_index + 1],
            rsu_position=self._runtime.config.rsu.position,
            link_forecasts=link_forecasts,
            rsu_visible_object_ids=rsu_visible_object_ids,
        )
        self._current_observation = observation
        return observation

    def _build_relay_candidates(
        self,
        active_vehicle_ids: tuple[int, ...],
    ) -> tuple[RelayAction, ...]:
        if not self._risk_config.enable_v2v_relay:
            return ()
        candidates: list[RelayAction] = []
        for sender_id in active_vehicle_ids:
            sender_cache = self._vehicle_knowledge[sender_id]
            for object_id in sender_cache.known_object_ids:
                for receiver_id in active_vehicle_ids:
                    if receiver_id == sender_id or receiver_id == object_id:
                        continue
                    if not self._link_reachable(sender_id, receiver_id):
                        continue
                    if self._can_update_between(
                        sender_cache,
                        self._vehicle_knowledge[receiver_id],
                        object_id,
                    ):
                        candidates.append(
                            RelayAction(
                                sender_id=sender_id,
                                receiver_id=receiver_id,
                                object_id=object_id,
                            )
                        )
        return tuple(candidates)

    def _is_valid_action(
        self,
        action: RiskSchedulingAction,
        observation: RiskObservation,
    ) -> bool:
        if any(item.sender_id == item.object_id for item in action.uplinks):
            return False
        if any(item.receiver_id == item.object_id for item in action.downlinks + action.relays):
            return False
        if any(not self._link_reachable(item.sender_id, 0) for item in action.uplinks):
            return False
        if any(not self._link_reachable(0, item.receiver_id) for item in action.downlinks):
            return False
        if any(not self._link_reachable(item.sender_id, item.receiver_id) for item in action.relays):
            return False
        if len(action.uplinks) > self._risk_config.uplink_slots_per_step:
            return False
        if len(action.downlinks) > self._risk_config.downlink_slots_per_step:
            return False
        if len(action.relays) > self._risk_config.relay_slots_per_step:
            return False
        if not set(action.uplinks).issubset(set(observation.uplink_candidates)):
            return False
        if not set(action.downlinks).issubset(set(observation.downlink_candidates)):
            return False
        if not set(action.relays).issubset(set(observation.relay_candidates)):
            return False
        if len({item.sender_id for item in action.uplinks}) != len(action.uplinks):
            return False
        if len({item.receiver_id for item in action.downlinks}) != len(action.downlinks):
            return False
        nodes = list(action.transmitting_vehicle_ids) + list(action.receiving_vehicle_ids)
        return len(set(nodes)) == len(nodes)

    def _link_reachable(self, sender_id: int, receiver_id: int) -> bool:
        frame = self._replay.frames[self._step_index]
        return self._communication.reachable_nodes(
            frame, self._runtime.config.rsu.position, sender_id, receiver_id,
        )

    def _deliver(
        self,
        action: RiskSchedulingAction,
    ) -> tuple[tuple[RiskSnapshot, ...], tuple[RiskSnapshot, ...], tuple[RiskSnapshot, ...],
               tuple[RiskDeliveryRecord, ...]]:
        delivery_step = self._step_index + 1
        downlink_snapshots = tuple(
            self._rsu_knowledge.snapshot_for(item.object_id)
            for item in action.downlinks
        )
        relay_snapshots = tuple(
            self._vehicle_knowledge[item.sender_id].snapshot_for(item.object_id)
            for item in action.relays
        )
        uplink_snapshots = tuple(
            self._current_local_snapshots[(item.sender_id, item.object_id)]
            for item in action.uplinks
        )

        records: list[RiskDeliveryRecord] = []

        def receive(link_type: str, sender: int, receiver: int,
                    cache: KnowledgeCache, snapshot: RiskSnapshot) -> None:
            previous = (cache.received_snapshot_for(snapshot.object_id)
                        if snapshot.object_id in cache.data_plane_object_ids else None)
            cache.receive(snapshot, received_step=delivery_step)
            records.append(RiskDeliveryRecord(
                link_type, sender, receiver, self._step_index, delivery_step,
                replace(snapshot, received_step=delivery_step), previous,
                cache.received_snapshot_for(snapshot.object_id),
            ))

        for item, snapshot in zip(action.uplinks, uplink_snapshots):
            receive("uplink", item.sender_id, 0, self._rsu_knowledge, snapshot)
        for downlink_item, snapshot in zip(action.downlinks, downlink_snapshots):
            receive("downlink", 0, downlink_item.receiver_id,
                    self._vehicle_knowledge[downlink_item.receiver_id], snapshot)
            self._last_sent_snapshots[(downlink_item.receiver_id, downlink_item.object_id)] = snapshot
        for relay_item, snapshot in zip(action.relays, relay_snapshots):
            receive("relay", relay_item.sender_id, relay_item.receiver_id,
                    self._vehicle_knowledge[relay_item.receiver_id], snapshot)
        return (
            tuple(replace(snapshot, received_step=delivery_step) for snapshot in uplink_snapshots),
            tuple(replace(snapshot, received_step=delivery_step) for snapshot in downlink_snapshots),
            tuple(replace(snapshot, received_step=delivery_step) for snapshot in relay_snapshots),
            tuple(records),
        )

    def _can_update(self, cache: KnowledgeCache, object_id: int) -> bool:
        return self._can_update_between(self._rsu_knowledge, cache, object_id)

    @staticmethod
    def _can_update_between(
        source_cache: KnowledgeCache,
        receiver_cache: KnowledgeCache,
        object_id: int,
    ) -> bool:
        if object_id not in receiver_cache.known_object_ids:
            return True
        current = receiver_cache.snapshot_for(object_id)
        candidate = source_cache.snapshot_for(object_id)
        return candidate.observation_step > current.observation_step

    def _task_for_id(self, task_id: str) -> RiskTask:
        for task in self._catalog.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown risk task: {task_id}")
