from __future__ import annotations

import random
from statistics import mean
from typing import Literal

from ..config import (
    AutonomyConfig,
    EnvironmentConfig,
    ResourceConfig,
    TaskConfig,
)
from ..tasks import TaskManager, evaluate_task_reward
from ..types import (
    CandidateLink,
    ControlBroadcastAction,
    LinkAction,
    LinkBudget,
    SchedulingAction,
    StepObservation,
    StepResult,
    Vector3,
    VehicleKnowledge,
    VehicleReplay,
    VehicleState,
)
from ..wireless.interfaces import LinkEvaluator


class TaskRelaySchedulingEnv:
    def __init__(
        self,
        replay: VehicleReplay,
        rsu_position: Vector3,
        link_evaluator: LinkEvaluator,
        environment_config: EnvironmentConfig,
        resource_config: ResourceConfig,
        task_config: TaskConfig,
        autonomy_config: AutonomyConfig,
        seed: int | None = None,
    ) -> None:
        self._replay = replay
        self._rsu_position = rsu_position
        self._link_evaluator = link_evaluator
        self._environment_config = environment_config
        self._resource_config = resource_config
        self._task_config = task_config
        self._autonomy_config = autonomy_config
        self._vehicle_ids = replay.vehicle_ids
        self._seed = seed
        configured_max_steps = environment_config.max_steps
        self._terminal_step = min(
            (
                configured_max_steps
                if configured_max_steps is not None
                else len(replay)
            ),
            len(replay),
        )
        self._task_manager = TaskManager(task_config, self._vehicle_ids, seed)
        self._vehicle_packet_timestamps: dict[int, tuple[float, ...]] = {}
        self._vehicle_packet_updated_flags: dict[int, tuple[bool, ...]] = {}
        self._initial_caoi_by_vehicle_id: dict[int, float] = {}
        self._autonomous_vehicle_ids: set[int] = set()
        self._initial_rng = random.Random(seed)
        self._autonomy_rng = random.Random(
            None if seed is None else seed + 1_000_003
        )
        self._current_step = 0
        self._ready = False
        self._validate_configuration()

    @property
    def task_templates(self):
        return self._task_manager.templates

    def reset(self) -> StepObservation:
        if self._terminal_step <= 0:
            raise ValueError("Environment has no available steps.")
        self._initial_rng = random.Random(self._seed)
        self._autonomy_rng = random.Random(
            None if self._seed is None else self._seed + 1_000_003
        )
        self._initial_caoi_by_vehicle_id = {
            vehicle_id: self._sample_initial_caoi()
            for vehicle_id in self._vehicle_ids
        }
        self._vehicle_packet_timestamps = {
            vehicle_id: self._build_initial_packet_state(vehicle_id)
            for vehicle_id in self._vehicle_ids
        }
        self._vehicle_packet_updated_flags = {
            vehicle_id: tuple(
                False for _ in range(self._environment_config.packet_count)
            )
            for vehicle_id in self._vehicle_ids
        }
        self._autonomous_vehicle_ids = self._sample_autonomous_vehicle_ids()
        self._task_manager.reset()
        self._current_step = 0
        self._ready = True
        self._begin_task_step(0)
        return self.preview_step(0)

    def preview_step(self, step_index: int) -> StepObservation:
        if step_index < 0 or step_index >= self._terminal_step:
            raise IndexError(f"Step index out of range: {step_index}")

        return self._preview_replay_step(step_index)

    def preview_terminal_step(self) -> StepObservation:
        """Preview the first replay frame after a configured truncation."""
        if not self._ready:
            raise RuntimeError("Call reset() before previewing a terminal step.")
        if self._terminal_step >= len(self._replay):
            raise RuntimeError("The episode ends at the replay boundary.")
        if self._current_step != self._terminal_step:
            raise RuntimeError(
                "Terminal preview is only available immediately after truncation."
            )

        self._begin_task_step(self._current_step)
        return self._preview_replay_step(self._current_step)

    def _preview_replay_step(self, step_index: int) -> StepObservation:
        if step_index < 0 or step_index >= len(self._replay):
            raise IndexError(f"Replay step index out of range: {step_index}")

        frame = self._replay.frames[step_index]
        current_time = frame.elapsed_seconds
        active_vehicles: list[VehicleState] = []
        for vehicle_id, snapshot in frame.vehicles.items():
            packet_timestamps = self._vehicle_packet_timestamps[vehicle_id]
            packet_flags = self._vehicle_packet_updated_flags[vehicle_id]
            active_vehicles.append(
                VehicleState(
                    snapshot=snapshot,
                    knowledge=VehicleKnowledge(
                        vehicle_id=vehicle_id,
                        packet_timestamps=packet_timestamps,
                        packet_updated_flags=packet_flags,
                        caoi_seconds=self._calculate_caoi(
                            packet_timestamps,
                            current_time,
                        ),
                    ),
                    is_autonomous=vehicle_id in self._autonomous_vehicle_ids,
                    priority_weight=self._priority_weight_for_vehicle(vehicle_id),
                    task_state=self._task_manager.state_for(vehicle_id, step_index),
                )
            )
        active_vehicles.sort(key=lambda item: item.snapshot.vehicle_id)
        active_tuple = tuple(active_vehicles)
        v2i_candidates, v2v_candidates = self._enumerate_candidate_links(
            active_vehicles=active_tuple,
            current_time=current_time,
            delta_seconds=frame.delta_seconds,
        )
        return StepObservation(
            step_index=frame.step_index,
            simulation_frame=frame.simulation_frame,
            game_time=frame.game_time,
            record_time_utc=frame.record_time_utc,
            elapsed_seconds=current_time,
            delta_seconds=frame.delta_seconds,
            active_vehicles=active_tuple,
            candidate_v2i_links=tuple(v2i_candidates),
            candidate_v2v_links=tuple(v2v_candidates),
            rsu_resource_block_count=self._resource_config.rsu_resource_block_count,
            v2v_resource_block_count=self._resource_config.v2v_resource_block_count,
        )

    def step(self, action: SchedulingAction | LinkAction | None) -> StepResult:
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        if self._current_step >= self._terminal_step:
            raise RuntimeError("Environment episode has already terminated.")

        observation = self.preview_step(self._current_step)
        scheduling_action = self._normalize_action(action)
        chosen_links, invalid_action = self._resolve_action(
            scheduling_action,
            observation,
        )
        chosen_v2i, chosen_v2v = self._split_candidate_links(chosen_links)

        if invalid_action:
            successful_links: tuple[CandidateLink, ...] = ()
            failed_links = chosen_links
        else:
            successful_links = tuple(
                candidate
                for candidate in chosen_links
                if candidate.delivered_packet_count >= 1.0
            )
            failed_links = tuple(
                candidate
                for candidate in chosen_links
                if candidate.delivered_packet_count < 1.0
            )
            self._apply_successful_links(successful_links)

        successful_v2i, successful_v2v = self._split_candidate_links(successful_links)
        failed_v2i, failed_v2v = self._split_candidate_links(failed_links)
        completion_time = observation.elapsed_seconds + observation.delta_seconds
        active_vehicle_ids = tuple(
            vehicle.snapshot.vehicle_id
            for vehicle in observation.active_vehicles
        )
        completed_tasks = self._task_manager.complete_ready_tasks(
            step_index=self._current_step,
            active_vehicle_ids=active_vehicle_ids,
            packet_timestamps_by_vehicle=self._vehicle_packet_timestamps,
            packet_flags_by_vehicle=self._vehicle_packet_updated_flags,
            completion_time=completion_time,
        )

        caoi_entries = self._post_action_caoi_entries(
            observation,
            completion_time,
        )
        caoi_values = [entry[2] for entry in caoi_entries]
        autonomous_caoi = [entry[2] for entry in caoi_entries if entry[0]]
        human_caoi = [entry[2] for entry in caoi_entries if not entry[0]]
        weight_sum = sum(entry[1] for entry in caoi_entries)
        weighted_sum = sum(entry[1] * entry[2] for entry in caoi_entries)

        control_broadcast = ControlBroadcastAction(
            transmitter_id="rsu-control",
            receiver_vehicle_ids=active_vehicle_ids,
            reliable=True,
        )
        used_rsu_blocks = (
            0
            if invalid_action
            else sum(
                candidate.resource_block_count
                for candidate in chosen_v2i
            )
        )
        used_v2v_blocks = (
            0
            if invalid_action
            else sum(
                candidate.resource_block_count
                for candidate in chosen_v2v
            )
        )
        delivered_packet_count = sum(
            int(candidate.delivered_packet_count)
            for candidate in successful_links
        )

        self._current_step += 1
        if self._current_step < self._terminal_step:
            self._begin_task_step(self._current_step)
            next_observation = self.preview_step(self._current_step)
        else:
            next_observation = None

        return StepResult(
            observation=observation,
            control_broadcast=control_broadcast,
            chosen_v2i_links=chosen_v2i,
            chosen_v2v_links=chosen_v2v,
            successful_v2i_links=successful_v2i,
            successful_v2v_links=successful_v2v,
            failed_v2i_links=failed_v2i,
            failed_v2v_links=failed_v2v,
            invalid_action=invalid_action,
            mean_caoi_seconds=mean(caoi_values) if caoi_values else 0.0,
            mean_autonomous_caoi_seconds=(
                mean(autonomous_caoi) if autonomous_caoi else 0.0
            ),
            mean_human_caoi_seconds=mean(human_caoi) if human_caoi else 0.0,
            mean_priority_weighted_caoi_seconds=(
                weighted_sum / weight_sum if weight_sum > 0.0 else 0.0
            ),
            max_caoi_seconds=max(caoi_values) if caoi_values else 0.0,
            age_violation_count=self._count_age_violations(caoi_values),
            next_observation=next_observation,
            task_reward=sum(completion.reward for completion in completed_tasks),
            completed_tasks=completed_tasks,
            delivered_packet_count=delivered_packet_count,
            used_rsu_resource_blocks=used_rsu_blocks,
            used_v2v_resource_blocks=used_v2v_blocks,
        )

    def _enumerate_candidate_links(
        self,
        active_vehicles: tuple[VehicleState, ...],
        current_time: float,
        delta_seconds: float,
    ) -> tuple[list[CandidateLink], list[CandidateLink]]:
        v2i_candidates: list[CandidateLink] = []
        v2v_candidates: list[CandidateLink] = []
        rsu_timestamps = tuple(
            current_time for _ in range(self._environment_config.packet_count)
        )
        rsu_flags = tuple(
            True for _ in range(self._environment_config.packet_count)
        )
        demand_counts = self._packet_demand_counts(active_vehicles)

        for receiver in active_vehicles:
            transferable = self._sorted_transferable_packet_ids(
                sender_timestamps=rsu_timestamps,
                sender_flags=rsu_flags,
                receiver=receiver,
                demand_counts=demand_counts,
            )
            v2i_candidates.extend(
                self._build_link_variants(
                    link_type="v2i",
                    transmitter_id="rsu",
                    transmitter_position=self._rsu_position,
                    receiver=receiver,
                    sender_timestamps=rsu_timestamps,
                    sender_flags=rsu_flags,
                    transferable_packet_ids=transferable,
                    active_vehicles=active_vehicles,
                    current_time=current_time,
                    delta_seconds=delta_seconds,
                )
            )

        for sender in active_vehicles:
            for receiver in active_vehicles:
                if sender.snapshot.vehicle_id == receiver.snapshot.vehicle_id:
                    continue
                transferable = self._sorted_transferable_packet_ids(
                    sender_timestamps=sender.knowledge.packet_timestamps,
                    sender_flags=sender.knowledge.packet_updated_flags,
                    receiver=receiver,
                    demand_counts=demand_counts,
                )
                if not transferable:
                    continue
                v2v_candidates.extend(
                    self._build_link_variants(
                        link_type="v2v",
                        transmitter_id=f"vehicle:{sender.snapshot.vehicle_id}",
                        transmitter_position=sender.snapshot.position,
                        receiver=receiver,
                        sender_timestamps=sender.knowledge.packet_timestamps,
                        sender_flags=sender.knowledge.packet_updated_flags,
                        transferable_packet_ids=transferable,
                        active_vehicles=active_vehicles,
                        current_time=current_time,
                        delta_seconds=delta_seconds,
                    )
                )
        return v2i_candidates, v2v_candidates

    def _build_link_variants(
        self,
        link_type: Literal["v2i", "v2v"],
        transmitter_id: str,
        transmitter_position: Vector3,
        receiver: VehicleState,
        sender_timestamps: tuple[float, ...],
        sender_flags: tuple[bool, ...],
        transferable_packet_ids: tuple[int, ...],
        active_vehicles: tuple[VehicleState, ...],
        current_time: float,
        delta_seconds: float,
    ) -> list[CandidateLink]:
        if not transferable_packet_ids:
            return []
        resource_block_hz = (
            self._resource_config.rsu_resource_block_hz
            if link_type == "v2i"
            else self._resource_config.v2v_resource_block_hz
        )
        total_blocks = (
            self._resource_config.rsu_resource_block_count
            if link_type == "v2i"
            else self._resource_config.v2v_resource_block_count
        )
        tx_power_dbm = (
            self._resource_config.rsu_tx_power_dbm
            if link_type == "v2i"
            else self._resource_config.vehicle_tx_power_dbm
        )
        variants: list[CandidateLink] = []
        seen_selected_sets: set[tuple[int, ...]] = set()
        max_blocks = min(
            self._resource_config.max_resource_blocks_per_link,
            total_blocks,
        )
        for resource_blocks in range(1, max_blocks + 1):
            allocated_bandwidth_hz = resource_blocks * resource_block_hz
            budget = self._evaluate_link_budget(
                transmitter_position=transmitter_position,
                receiver_position=receiver.snapshot.position,
                active_vehicles=active_vehicles,
                excluded_vehicle_ids=self._excluded_vehicle_ids(
                    link_type,
                    transmitter_id,
                    receiver.snapshot.vehicle_id,
                ),
                bandwidth_hz=allocated_bandwidth_hz,
                tx_power_dbm=tx_power_dbm,
            )
            if not budget.reachable:
                continue
            expected_packet_count = self._expected_packet_count(
                budget.rate_mbps,
                delta_seconds,
            )
            complete_capacity = int(expected_packet_count)
            if complete_capacity <= 0:
                continue
            selected_packet_ids = transferable_packet_ids[:complete_capacity]
            if not selected_packet_ids or selected_packet_ids in seen_selected_sets:
                continue
            seen_selected_sets.add(selected_packet_ids)
            variants.append(
                self._build_candidate(
                    link_type=link_type,
                    transmitter_id=transmitter_id,
                    transmitter_position=transmitter_position,
                    receiver=receiver,
                    sender_timestamps=sender_timestamps,
                    sender_flags=sender_flags,
                    transferable_packet_ids=transferable_packet_ids,
                    selected_packet_ids=selected_packet_ids,
                    resource_blocks=resource_blocks,
                    budget=budget,
                    expected_packet_count=expected_packet_count,
                    active_vehicles=active_vehicles,
                    current_time=current_time,
                    delta_seconds=delta_seconds,
                )
            )
        return variants

    def _build_candidate(
        self,
        link_type: Literal["v2i", "v2v"],
        transmitter_id: str,
        transmitter_position: Vector3,
        receiver: VehicleState,
        sender_timestamps: tuple[float, ...],
        sender_flags: tuple[bool, ...],
        transferable_packet_ids: tuple[int, ...],
        selected_packet_ids: tuple[int, ...],
        resource_blocks: int,
        budget: LinkBudget,
        expected_packet_count: float,
        active_vehicles: tuple[VehicleState, ...],
        current_time: float,
        delta_seconds: float,
    ) -> CandidateLink:
        post_timestamps, post_flags = self._apply_complete_packet_transfer(
            receiver.knowledge.packet_timestamps,
            receiver.knowledge.packet_updated_flags,
            sender_timestamps,
            sender_flags,
            selected_packet_ids,
        )
        completion_time = current_time + delta_seconds
        empty_post_caoi = self._calculate_caoi(
            receiver.knowledge.packet_timestamps,
            completion_time,
        )
        post_caoi = self._calculate_caoi(post_timestamps, completion_time)
        freshness_gain = max(empty_post_caoi - post_caoi, 0.0)
        required_ids = set(receiver.task_state.required_packet_ids)
        missing_before = {
            packet_id
            for packet_id in required_ids
            if not receiver.knowledge.packet_updated_flags[packet_id - 1]
        }
        task_progress_gain = len(missing_before.intersection(selected_packet_ids))
        task_reward_gain = self._estimate_task_completion_reward_gain(
            receiver,
            post_timestamps,
            post_flags,
            completion_time,
        )
        (
            relay_peer_count,
            relay_peer_gain,
            relay_peer_priority_gain,
        ) = self._estimate_packet_demand_spread(
            relay_sender=receiver,
            packet_ids=selected_packet_ids,
            post_timestamps=post_timestamps,
            post_flags=post_flags,
            active_vehicles=active_vehicles,
            completion_time=completion_time,
            delta_seconds=delta_seconds,
        )
        action = LinkAction(
            link_type=link_type,
            transmitter_id=transmitter_id,
            receiver_id=receiver.snapshot.vehicle_id,
            transmitter_position=transmitter_position,
            receiver_position=receiver.snapshot.position,
            resource_block_count=resource_blocks,
            packet_ids=selected_packet_ids,
        )
        return CandidateLink(
            action=action,
            link_budget=budget,
            transmitter_packet_timestamps=sender_timestamps,
            transmitter_packet_updated_flags=sender_flags,
            receiver_packet_timestamps=receiver.knowledge.packet_timestamps,
            receiver_packet_updated_flags=receiver.knowledge.packet_updated_flags,
            fresher_packet_count=len(transferable_packet_ids),
            expected_packet_count=expected_packet_count,
            delivered_packet_count=float(len(selected_packet_ids)),
            post_receiver_packet_timestamps=post_timestamps,
            post_receiver_packet_updated_flags=post_flags,
            post_caoi_seconds=post_caoi,
            freshness_gain_seconds=freshness_gain,
            receiver_is_autonomous=receiver.is_autonomous,
            receiver_priority_weight=receiver.priority_weight,
            priority_gain_seconds=freshness_gain * receiver.priority_weight,
            relay_peer_count=relay_peer_count,
            relay_peer_total_gain_seconds=relay_peer_gain,
            relay_peer_total_priority_gain_seconds=relay_peer_priority_gain,
            transferable_packet_ids=transferable_packet_ids,
            selected_packet_ids=selected_packet_ids,
            resource_block_count=resource_blocks,
            allocated_bandwidth_hz=budget.allocated_bandwidth_hz,
            complete_packet_capacity=int(expected_packet_count),
            task_progress_gain=task_progress_gain,
            task_reward_gain=task_reward_gain,
            task_required_packet_count=len(required_ids),
            task_missing_packet_count=len(missing_before),
        )

    def _resolve_action(
        self,
        action: SchedulingAction,
        observation: StepObservation,
    ) -> tuple[tuple[CandidateLink, ...], bool]:
        candidate_map = {
            candidate.action: candidate
            for candidate in observation.candidate_links
        }
        chosen: list[CandidateLink] = []
        invalid = any(
            link.link_type != "v2i"
            for link in action.v2i_links
        ) or any(
            link.link_type != "v2v"
            for link in action.v2v_links
        )
        for link in action.links:
            candidate = candidate_map.get(link)
            if candidate is None:
                invalid = True
                continue
            chosen.append(candidate)
        if len(chosen) != len(action.links):
            invalid = True

        max_active_links = self._environment_config.max_active_links
        if max_active_links is not None and len(action.links) > max_active_links:
            invalid = True
        if (
            not self._environment_config.allow_multi_v2i_from_rsu
            and len(action.v2i_links) > 1
        ):
            invalid = True

        used_vehicle_nodes: set[int] = set()
        for link in action.links:
            for vehicle_id in self._vehicle_nodes_for_action(link):
                if vehicle_id in used_vehicle_nodes:
                    invalid = True
                used_vehicle_nodes.add(vehicle_id)

        chosen_v2i, chosen_v2v = self._split_candidate_links(tuple(chosen))
        if sum(item.resource_block_count for item in chosen_v2i) > (
            self._resource_config.rsu_resource_block_count
        ):
            invalid = True
        if sum(item.resource_block_count for item in chosen_v2v) > (
            self._resource_config.v2v_resource_block_count
        ):
            invalid = True
        if sum(item.link_budget.rate_mbps for item in chosen_v2i) > (
            self._resource_config.rsu_total_rate_limit_mbps + 1e-9
        ):
            invalid = True
        if sum(item.link_budget.rate_mbps for item in chosen_v2v) > (
            self._resource_config.v2v_total_rate_limit_mbps + 1e-9
        ):
            invalid = True
        return tuple(chosen), invalid

    def _apply_successful_links(
        self,
        successful_links: tuple[CandidateLink, ...],
    ) -> None:
        for candidate in successful_links:
            receiver_id = candidate.action.receiver_id
            self._vehicle_packet_timestamps[receiver_id] = (
                candidate.post_receiver_packet_timestamps
            )
            self._vehicle_packet_updated_flags[receiver_id] = (
                candidate.post_receiver_packet_updated_flags
            )

    def _evaluate_link_budget(
        self,
        transmitter_position: Vector3,
        receiver_position: Vector3,
        active_vehicles: tuple[VehicleState, ...],
        excluded_vehicle_ids: set[int],
        bandwidth_hz: float,
        tx_power_dbm: float,
    ):
        blockers = tuple(
            vehicle.snapshot.position
            for vehicle in active_vehicles
            if vehicle.snapshot.vehicle_id not in excluded_vehicle_ids
        )
        return self._link_evaluator.evaluate(
            transmitter_position,
            receiver_position,
            blockers=blockers,
            bandwidth_hz=bandwidth_hz,
            tx_power_dbm=tx_power_dbm,
            rate_limit_mbps=self._resource_config.vehicle_link_rate_limit_mbps,
        )

    def _sorted_transferable_packet_ids(
        self,
        sender_timestamps: tuple[float, ...],
        sender_flags: tuple[bool, ...],
        receiver: VehicleState,
        demand_counts: dict[int, int],
    ) -> tuple[int, ...]:
        required_ids = set(receiver.task_state.required_packet_ids)
        receiver_flags = receiver.knowledge.packet_updated_flags
        receiver_timestamps = receiver.knowledge.packet_timestamps
        packet_ids = [
            index + 1
            for index, (
                sender_timestamp,
                sender_flag,
                receiver_timestamp,
                receiver_flag,
            ) in enumerate(
                zip(
                    sender_timestamps,
                    sender_flags,
                    receiver_timestamps,
                    receiver_flags,
                    strict=True,
                )
            )
            if sender_flag
            and (
                not receiver_flag
                or sender_timestamp > receiver_timestamp
            )
        ]
        packet_ids.sort(
            key=lambda packet_id: (
                packet_id in required_ids and not receiver_flags[packet_id - 1],
                packet_id in required_ids,
                demand_counts.get(packet_id, 0),
                sender_timestamps[packet_id - 1]
                - receiver_timestamps[packet_id - 1],
                -receiver_timestamps[packet_id - 1],
            ),
            reverse=True,
        )
        return tuple(packet_ids)

    @staticmethod
    def _apply_complete_packet_transfer(
        receiver_timestamps: tuple[float, ...],
        receiver_flags: tuple[bool, ...],
        sender_timestamps: tuple[float, ...],
        sender_flags: tuple[bool, ...],
        selected_packet_ids: tuple[int, ...],
    ) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        updated_timestamps = list(receiver_timestamps)
        updated_flags = list(receiver_flags)
        for packet_id in selected_packet_ids:
            index = packet_id - 1
            updated_timestamps[index] = sender_timestamps[index]
            updated_flags[index] = sender_flags[index]
        return tuple(updated_timestamps), tuple(updated_flags)

    def _estimate_task_completion_reward_gain(
        self,
        receiver: VehicleState,
        post_timestamps: tuple[float, ...],
        post_flags: tuple[bool, ...],
        completion_time: float,
    ) -> float:
        required_ids = receiver.task_state.required_packet_ids
        if not required_ids:
            return 0.0
        post_evaluation = evaluate_task_reward(
            config=self._task_config,
            required_packet_ids=required_ids,
            packet_timestamps=post_timestamps,
            packet_flags=post_flags,
            completion_time=completion_time,
        )
        if post_evaluation is None:
            return 0.0
        baseline_evaluation = evaluate_task_reward(
            config=self._task_config,
            required_packet_ids=required_ids,
            packet_timestamps=receiver.knowledge.packet_timestamps,
            packet_flags=receiver.knowledge.packet_updated_flags,
            completion_time=completion_time,
        )
        baseline_reward = (
            baseline_evaluation[1]
            if baseline_evaluation is not None
            else 0.0
        )
        return max(post_evaluation[1] - baseline_reward, 0.0)

    def _estimate_packet_demand_spread(
        self,
        relay_sender: VehicleState,
        packet_ids: tuple[int, ...],
        post_timestamps: tuple[float, ...],
        post_flags: tuple[bool, ...],
        active_vehicles: tuple[VehicleState, ...],
        completion_time: float,
        delta_seconds: float,
    ) -> tuple[int, float, float]:
        peer_count = 0
        total_gain = 0.0
        total_priority_gain = 0.0
        max_blocks = min(
            self._resource_config.max_resource_blocks_per_link,
            self._resource_config.v2v_resource_block_count,
        )
        bandwidth_hz = max_blocks * self._resource_config.v2v_resource_block_hz
        for peer in active_vehicles:
            peer_id = peer.snapshot.vehicle_id
            if peer_id == relay_sender.snapshot.vehicle_id:
                continue
            required = set(peer.task_state.required_packet_ids)
            transferable = tuple(
                packet_id
                for packet_id in packet_ids
                if packet_id in required
                and post_flags[packet_id - 1]
                and (
                    not peer.knowledge.packet_updated_flags[packet_id - 1]
                    or post_timestamps[packet_id - 1]
                    > peer.knowledge.packet_timestamps[packet_id - 1]
                )
            )
            if not transferable:
                continue

            budget = self._evaluate_link_budget(
                transmitter_position=relay_sender.snapshot.position,
                receiver_position=peer.snapshot.position,
                active_vehicles=active_vehicles,
                excluded_vehicle_ids={
                    relay_sender.snapshot.vehicle_id,
                    peer_id,
                },
                bandwidth_hz=bandwidth_hz,
                tx_power_dbm=self._resource_config.vehicle_tx_power_dbm,
            )
            if not budget.reachable:
                continue
            complete_capacity = int(
                self._expected_packet_count(
                    budget.rate_mbps,
                    delta_seconds,
                )
            )
            delivered_packet_ids = transferable[:complete_capacity]
            if not delivered_packet_ids:
                continue
            peer_count += 1

            peer_post_timestamps, _ = self._apply_complete_packet_transfer(
                peer.knowledge.packet_timestamps,
                peer.knowledge.packet_updated_flags,
                post_timestamps,
                post_flags,
                delivered_packet_ids,
            )
            baseline_caoi = self._calculate_caoi(
                peer.knowledge.packet_timestamps,
                completion_time,
            )
            post_caoi = self._calculate_caoi(
                peer_post_timestamps,
                completion_time,
            )
            gain = max(baseline_caoi - post_caoi, 0.0)
            total_gain += gain
            total_priority_gain += gain * peer.priority_weight
        return peer_count, total_gain, total_priority_gain

    @staticmethod
    def _packet_demand_counts(
        active_vehicles: tuple[VehicleState, ...],
    ) -> dict[int, int]:
        counts: dict[int, int] = {}
        for vehicle in active_vehicles:
            for packet_id in vehicle.task_state.required_packet_ids:
                if not vehicle.knowledge.packet_updated_flags[packet_id - 1]:
                    counts[packet_id] = counts.get(packet_id, 0) + 1
        return counts

    def _expected_packet_count(
        self,
        rate_mbps: float,
        delta_seconds: float,
    ) -> float:
        delivered_bits = (
            max(rate_mbps, 0.0)
            * 1_000_000.0
            * max(delta_seconds, 0.0)
        )
        return delivered_bits / self._environment_config.packet_size_bits

    def _post_action_caoi_entries(
        self,
        observation: StepObservation,
        completion_time: float,
    ) -> list[tuple[bool, float, float]]:
        entries: list[tuple[bool, float, float]] = []
        for vehicle in observation.active_vehicles:
            vehicle_id = vehicle.snapshot.vehicle_id
            entries.append(
                (
                    vehicle.is_autonomous,
                    vehicle.priority_weight,
                    self._calculate_caoi(
                        self._vehicle_packet_timestamps[vehicle_id],
                        completion_time,
                    ),
                )
            )
        return entries

    def _build_initial_packet_state(self, vehicle_id: int) -> tuple[float, ...]:
        initial_caoi = self._initial_caoi_by_vehicle_id[vehicle_id]
        initial_time = self._replay.frames[0].elapsed_seconds
        return tuple(
            initial_time - initial_caoi
            for _ in range(self._environment_config.packet_count)
        )

    def _sample_initial_caoi(self) -> float:
        lower = self._environment_config.initial_caoi_min_seconds
        upper = self._environment_config.initial_caoi_max_seconds
        if lower is None and upper is None:
            return self._environment_config.initial_caoi_seconds
        if lower is None or upper is None:
            raise ValueError(
                "initial_caoi_min_seconds and initial_caoi_max_seconds must be set together."
            )
        if upper < lower:
            raise ValueError(
                "initial_caoi_max_seconds must be >= initial_caoi_min_seconds."
            )
        return self._initial_rng.uniform(lower, upper)

    def _sample_autonomous_vehicle_ids(self) -> set[int]:
        if not self._autonomy_config.enabled:
            return set()
        penetration_rate = min(
            max(self._autonomy_config.penetration_rate, 0.0),
            1.0,
        )
        count = round(len(self._vehicle_ids) * penetration_rate)
        count = min(max(count, 0), len(self._vehicle_ids))
        return set(self._autonomy_rng.sample(list(self._vehicle_ids), count))

    def _priority_weight_for_vehicle(self, vehicle_id: int) -> float:
        if vehicle_id in self._autonomous_vehicle_ids:
            return self._autonomy_config.priority_weight
        return 1.0

    def _begin_task_step(self, step_index: int) -> None:
        frame = self._replay.frames[step_index]
        self._task_manager.begin_step(
            step_index,
            tuple(sorted(frame.vehicles)),
        )

    def _validate_configuration(self) -> None:
        if self._environment_config.packet_count <= 0:
            raise ValueError("packet_count must be positive.")
        if self._environment_config.packet_size_bits <= 0.0:
            raise ValueError("packet_size_bits must be positive.")
        if self._environment_config.fallback_step_seconds <= 0.0:
            raise ValueError("fallback_step_seconds must be positive.")
        if (
            self._environment_config.max_steps is not None
            and self._environment_config.max_steps <= 0
        ):
            raise ValueError("max_steps must be positive when configured.")
        if (
            self._environment_config.max_active_links is not None
            and self._environment_config.max_active_links < 0
        ):
            raise ValueError("max_active_links must be non-negative.")
        if self._environment_config.initial_caoi_seconds < 0.0:
            raise ValueError("initial_caoi_seconds must be non-negative.")
        initial_caoi_min = self._environment_config.initial_caoi_min_seconds
        initial_caoi_max = self._environment_config.initial_caoi_max_seconds
        if (initial_caoi_min is None) != (initial_caoi_max is None):
            raise ValueError(
                "initial_caoi_min_seconds and initial_caoi_max_seconds must be set together."
            )
        if initial_caoi_min is not None and initial_caoi_min < 0.0:
            raise ValueError("initial_caoi_min_seconds must be non-negative.")
        if initial_caoi_max is not None and initial_caoi_max < 0.0:
            raise ValueError("initial_caoi_max_seconds must be non-negative.")
        if (
            initial_caoi_min is not None
            and initial_caoi_max is not None
            and initial_caoi_max < initial_caoi_min
        ):
            raise ValueError(
                "initial_caoi_max_seconds must be >= initial_caoi_min_seconds."
            )
        if (
            self._environment_config.age_tolerance_seconds is not None
            and self._environment_config.age_tolerance_seconds < 0.0
        ):
            raise ValueError("age_tolerance_seconds must be non-negative.")
        if self._task_config.freshness_decay_seconds <= 0.0:
            raise ValueError("freshness_decay_seconds must be positive.")
        if self._task_config.max_completed_tasks_per_vehicle < 0:
            raise ValueError("max_completed_tasks_per_vehicle must be non-negative.")
        if self._task_config.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative.")
        if self._task_config.reward_base < 0.0:
            raise ValueError("reward_base must be non-negative.")
        if self._resource_config.rsu_total_bandwidth_hz <= 0.0:
            raise ValueError("rsu_total_bandwidth_hz must be positive.")
        if self._resource_config.rsu_resource_block_hz <= 0.0:
            raise ValueError("rsu_resource_block_hz must be positive.")
        if self._resource_config.v2v_total_bandwidth_hz <= 0.0:
            raise ValueError("v2v_total_bandwidth_hz must be positive.")
        if self._resource_config.v2v_resource_block_hz <= 0.0:
            raise ValueError("v2v_resource_block_hz must be positive.")
        if self._resource_config.max_resource_blocks_per_link <= 0:
            raise ValueError("max_resource_blocks_per_link must be positive.")
        if self._resource_config.vehicle_link_rate_limit_mbps <= 0.0:
            raise ValueError("vehicle_link_rate_limit_mbps must be positive.")
        if self._resource_config.rsu_total_rate_limit_mbps < 0.0:
            raise ValueError("rsu_total_rate_limit_mbps must be non-negative.")
        if self._resource_config.v2v_total_rate_limit_mbps < 0.0:
            raise ValueError("v2v_total_rate_limit_mbps must be non-negative.")
        _ = (
            self._resource_config.rsu_resource_block_count,
            self._resource_config.v2v_resource_block_count,
        )
        if not 0.0 <= self._autonomy_config.penetration_rate <= 1.0:
            raise ValueError("autonomy penetration_rate must be within 0..1.")
        if self._autonomy_config.priority_weight <= 0.0:
            raise ValueError("autonomy priority_weight must be positive.")
        packet_count = self._environment_config.packet_count
        task_ids = [task_id for task_id, _ in self._task_config.templates]
        if (
            self._task_config.enabled
            and self._task_config.max_completed_tasks_per_vehicle > 0
            and not task_ids
        ):
            raise ValueError("At least one task template is required.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task template IDs must be unique.")
        for task_id, packet_ids in self._task_config.templates:
            if not packet_ids:
                raise ValueError(f"Task {task_id!r} must require at least one packet.")
            if len(packet_ids) != len(set(packet_ids)):
                raise ValueError(
                    f"Task {task_id!r} contains duplicate packet IDs."
                )
            invalid_ids = [
                packet_id
                for packet_id in packet_ids
                if packet_id < 1 or packet_id > packet_count
            ]
            if invalid_ids:
                raise ValueError(
                    f"Task {task_id!r} contains packet IDs outside 1..{packet_count}: {invalid_ids}"
                )

    def _count_age_violations(self, caoi_values: list[float]) -> int:
        tolerance = self._environment_config.age_tolerance_seconds
        if tolerance is None:
            return 0
        return sum(value > tolerance for value in caoi_values)

    @staticmethod
    def _calculate_caoi(
        packet_timestamps: tuple[float, ...],
        current_time: float,
    ) -> float:
        if not packet_timestamps:
            return 0.0
        return mean(
            max(current_time - timestamp, 0.0)
            for timestamp in packet_timestamps
        )

    @staticmethod
    def _normalize_action(
        action: SchedulingAction | LinkAction | None,
    ) -> SchedulingAction:
        if action is None:
            return SchedulingAction()
        if isinstance(action, SchedulingAction):
            return action
        if isinstance(action, LinkAction):
            if action.link_type == "v2i":
                return SchedulingAction(v2i_links=(action,))
            return SchedulingAction(v2v_links=(action,))
        raise TypeError(f"Unsupported action type: {type(action)!r}")

    @staticmethod
    def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
        nodes = [action.receiver_id]
        if (
            action.link_type == "v2v"
            and action.transmitter_id.startswith("vehicle:")
        ):
            nodes.append(int(action.transmitter_id.split(":", 1)[1]))
        return tuple(nodes)

    @staticmethod
    def _excluded_vehicle_ids(
        link_type: Literal["v2i", "v2v"],
        transmitter_id: str,
        receiver_id: int,
    ) -> set[int]:
        excluded = {receiver_id}
        if link_type == "v2v":
            excluded.add(int(transmitter_id.split(":", 1)[1]))
        return excluded

    @staticmethod
    def _split_candidate_links(
        candidates: tuple[CandidateLink, ...],
    ) -> tuple[tuple[CandidateLink, ...], tuple[CandidateLink, ...]]:
        return (
            tuple(
                candidate
                for candidate in candidates
                if candidate.action.link_type == "v2i"
            ),
            tuple(
                candidate
                for candidate in candidates
                if candidate.action.link_type == "v2v"
            ),
        )
