from __future__ import annotations

import random
from statistics import mean

from ..config import AutonomyConfig, EnvironmentConfig
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


class RelaySchedulingEnv:
    def __init__(
        self,
        replay: VehicleReplay,
        rsu_position: Vector3,
        link_evaluator: LinkEvaluator,
        environment_config: EnvironmentConfig,
        autonomy_config: AutonomyConfig,
        seed: int | None = None,
    ) -> None:
        self._replay = replay
        self._rsu_position = rsu_position
        self._link_evaluator = link_evaluator
        self._environment_config = environment_config
        self._autonomy_config = autonomy_config
        self._vehicle_ids = replay.vehicle_ids
        self._seed = seed
        self._terminal_step = min(environment_config.max_steps or len(replay), len(replay))
        self._vehicle_packet_timestamps: dict[int, tuple[float, ...]] = {}
        self._vehicle_packet_updated_flags: dict[int, tuple[bool, ...]] = {}
        self._initial_caoi_by_vehicle_id: dict[int, float] = {}
        self._autonomous_vehicle_ids: set[int] = set()
        self._rng = random.Random(seed)
        self._current_step = 0
        self._ready = False

    def reset(self) -> StepObservation:
        if self._terminal_step <= 0:
            raise ValueError("Environment has no available steps.")
        self._rng = random.Random(self._seed)
        self._initial_caoi_by_vehicle_id = {
            vehicle_id: self._sample_initial_caoi() for vehicle_id in self._vehicle_ids
        }
        self._vehicle_packet_timestamps = {
            vehicle_id: self._build_initial_packet_state(vehicle_id) for vehicle_id in self._vehicle_ids
        }
        self._vehicle_packet_updated_flags = {
            vehicle_id: tuple(False for _ in range(self._environment_config.packet_count))
            for vehicle_id in self._vehicle_ids
        }
        self._autonomous_vehicle_ids = self._sample_autonomous_vehicle_ids()
        self._current_step = 0
        self._ready = True
        return self.preview_step(self._current_step)

    def preview_step(self, step_index: int) -> StepObservation:
        if step_index < 0 or step_index >= self._terminal_step:
            raise IndexError(f"Step index out of range: {step_index}")

        frame = self._replay.frames[step_index]
        current_time = frame.elapsed_seconds
        active_vehicles: list[VehicleState] = []

        for vehicle_id, snapshot in frame.vehicles.items():
            packet_timestamps = self._packet_state_for_vehicle(vehicle_id)
            packet_updated_flags = self._packet_update_flags_for_vehicle(vehicle_id)
            active_vehicles.append(
                VehicleState(
                    snapshot=snapshot,
                    knowledge=VehicleKnowledge(
                        vehicle_id=vehicle_id,
                        packet_timestamps=packet_timestamps,
                        packet_updated_flags=packet_updated_flags,
                        caoi_seconds=self._calculate_caoi(packet_timestamps, current_time),
                    ),
                    is_autonomous=vehicle_id in self._autonomous_vehicle_ids,
                    priority_weight=self._priority_weight_for_vehicle(vehicle_id),
                )
            )

        active_vehicles.sort(key=lambda item: item.snapshot.vehicle_id)
        candidate_v2i_links, candidate_v2v_links = self._enumerate_candidate_links(
            active_vehicles=tuple(active_vehicles),
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
            active_vehicles=tuple(active_vehicles),
            candidate_v2i_links=tuple(candidate_v2i_links),
            candidate_v2v_links=tuple(candidate_v2v_links),
        )

    def step(self, action: SchedulingAction | LinkAction | None) -> StepResult:
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        if self._current_step >= self._terminal_step:
            raise RuntimeError("Environment episode has already terminated.")

        observation = self.preview_step(self._current_step)
        scheduling_action = self._normalize_action(action)
        chosen_links, invalid_action = self._resolve_action(scheduling_action, observation)
        chosen_v2i_links, chosen_v2v_links = self._split_candidate_links(chosen_links)

        if invalid_action:
            successful_links: tuple[CandidateLink, ...] = ()
            failed_links = chosen_links
            packet_updates: dict[int, tuple[tuple[float, ...], tuple[bool, ...]]] = {}
        else:
            successful_links = tuple(candidate for candidate in chosen_links if candidate.delivered_packet_count > 0.0)
            failed_links = tuple(candidate for candidate in chosen_links if candidate.delivered_packet_count <= 0.0)
            packet_updates = self._apply_successful_links(successful_links)

        successful_v2i_links, successful_v2v_links = self._split_candidate_links(successful_links)
        failed_v2i_links, failed_v2v_links = self._split_candidate_links(failed_links)

        caoi_entries = self._post_action_caoi_entries(observation, packet_updates)
        caoi_values = [entry[2] for entry in caoi_entries]
        autonomous_caoi_values = [entry[2] for entry in caoi_entries if entry[0]]
        human_caoi_values = [entry[2] for entry in caoi_entries if not entry[0]]
        weighted_caoi_values = [entry[2] * entry[1] for entry in caoi_entries]
        weight_sum = sum(entry[1] for entry in caoi_entries)
        mean_caoi_seconds = mean(caoi_values) if caoi_values else 0.0
        mean_autonomous_caoi_seconds = mean(autonomous_caoi_values) if autonomous_caoi_values else 0.0
        mean_human_caoi_seconds = mean(human_caoi_values) if human_caoi_values else 0.0
        mean_priority_weighted_caoi_seconds = (
            sum(weighted_caoi_values) / weight_sum if weight_sum > 0.0 else 0.0
        )
        max_caoi_seconds = max(caoi_values) if caoi_values else 0.0
        age_violation_count = self._count_age_violations(caoi_values)

        control_broadcast = ControlBroadcastAction(
            transmitter_id="rsu-control",
            receiver_vehicle_ids=tuple(
                vehicle.snapshot.vehicle_id for vehicle in observation.active_vehicles
            ),
            reliable=True,
        )

        self._current_step += 1
        next_observation = (
            None if self._current_step >= self._terminal_step else self.preview_step(self._current_step)
        )
        return StepResult(
            observation=observation,
            control_broadcast=control_broadcast,
            chosen_v2i_links=chosen_v2i_links,
            chosen_v2v_links=chosen_v2v_links,
            successful_v2i_links=successful_v2i_links,
            successful_v2v_links=successful_v2v_links,
            failed_v2i_links=failed_v2i_links,
            failed_v2v_links=failed_v2v_links,
            invalid_action=invalid_action,
            mean_caoi_seconds=mean_caoi_seconds,
            mean_autonomous_caoi_seconds=mean_autonomous_caoi_seconds,
            mean_human_caoi_seconds=mean_human_caoi_seconds,
            mean_priority_weighted_caoi_seconds=mean_priority_weighted_caoi_seconds,
            max_caoi_seconds=max_caoi_seconds,
            age_violation_count=age_violation_count,
            next_observation=next_observation,
        )

    def _normalize_action(self, action: SchedulingAction | LinkAction | None) -> SchedulingAction:
        if action is None:
            return SchedulingAction()
        if isinstance(action, SchedulingAction):
            return action
        if isinstance(action, LinkAction):
            if action.link_type == "v2i":
                return SchedulingAction(v2i_links=(action,))
            return SchedulingAction(v2v_links=(action,))
        raise TypeError(f"Unsupported action type: {type(action)!r}")

    def _resolve_action(
        self,
        action: SchedulingAction,
        observation: StepObservation,
    ) -> tuple[tuple[CandidateLink, ...], bool]:
        candidate_map = {candidate.action: candidate for candidate in observation.candidate_links}
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

        if not self._environment_config.allow_multi_v2i_from_rsu and len(action.v2i_links) > 1:
            invalid = True

        used_vehicle_nodes: set[int] = set()
        for link in action.links:
            for vehicle_id in self._vehicle_nodes_for_action(link):
                if vehicle_id in used_vehicle_nodes:
                    invalid = True
                used_vehicle_nodes.add(vehicle_id)

        return tuple(chosen), invalid

    def _apply_successful_links(
        self,
        successful_links: tuple[CandidateLink, ...],
    ) -> dict[int, tuple[tuple[float, ...], tuple[bool, ...]]]:
        packet_updates: dict[int, tuple[tuple[float, ...], tuple[bool, ...]]] = {}
        for candidate in successful_links:
            packet_updates[candidate.action.receiver_id] = (
                candidate.post_receiver_packet_timestamps,
                candidate.post_receiver_packet_updated_flags,
            )
        for receiver_id, (packet_timestamps, packet_updated_flags) in packet_updates.items():
            self._vehicle_packet_timestamps[receiver_id] = packet_timestamps
            self._vehicle_packet_updated_flags[receiver_id] = packet_updated_flags
        return packet_updates

    def _packet_state_for_vehicle(self, vehicle_id: int) -> tuple[float, ...]:
        cached = self._vehicle_packet_timestamps.get(vehicle_id)
        if cached is not None:
            return cached
        built = self._build_initial_packet_state(vehicle_id)
        self._vehicle_packet_timestamps[vehicle_id] = built
        return built

    def _packet_update_flags_for_vehicle(self, vehicle_id: int) -> tuple[bool, ...]:
        cached = self._vehicle_packet_updated_flags.get(vehicle_id)
        if cached is not None:
            return cached
        built = tuple(False for _ in range(self._environment_config.packet_count))
        self._vehicle_packet_updated_flags[vehicle_id] = built
        return built

    def _build_initial_packet_state(self, vehicle_id: int) -> tuple[float, ...]:
        initial_caoi = self._initial_caoi_for_vehicle(vehicle_id)
        initial_timestamp = self._replay.frames[0].elapsed_seconds - initial_caoi
        return tuple(initial_timestamp for _ in range(self._environment_config.packet_count))

    def _initial_caoi_for_vehicle(self, vehicle_id: int) -> float:
        cached = self._initial_caoi_by_vehicle_id.get(vehicle_id)
        if cached is not None:
            return cached
        sampled = self._sample_initial_caoi()
        self._initial_caoi_by_vehicle_id[vehicle_id] = sampled
        return sampled

    def _sample_initial_caoi(self) -> float:
        lower = self._environment_config.initial_caoi_min_seconds
        upper = self._environment_config.initial_caoi_max_seconds
        if lower is None and upper is None:
            return self._environment_config.initial_caoi_seconds
        if lower is None or upper is None:
            raise ValueError("initial_caoi_min_seconds and initial_caoi_max_seconds must be set together.")
        if upper < lower:
            raise ValueError("initial_caoi_max_seconds must be >= initial_caoi_min_seconds.")
        return self._rng.uniform(lower, upper)

    def _sample_autonomous_vehicle_ids(self) -> set[int]:
        if not self._autonomy_config.enabled:
            return set()
        penetration_rate = min(max(self._autonomy_config.penetration_rate, 0.0), 1.0)
        vehicle_count = len(self._vehicle_ids)
        autonomous_count = round(vehicle_count * penetration_rate)
        if autonomous_count <= 0:
            return set()
        autonomous_count = min(autonomous_count, vehicle_count)
        return set(self._rng.sample(list(self._vehicle_ids), autonomous_count))

    def _priority_weight_for_vehicle(self, vehicle_id: int) -> float:
        if vehicle_id in self._autonomous_vehicle_ids:
            return self._autonomy_config.priority_weight
        return 1.0

    def _post_action_caoi_entries(
        self,
        observation: StepObservation,
        packet_updates: dict[int, tuple[tuple[float, ...], tuple[bool, ...]]],
    ) -> list[tuple[bool, float, float]]:
        caoi_entries: list[tuple[bool, float, float]] = []
        current_time = observation.elapsed_seconds + observation.delta_seconds
        for vehicle in observation.active_vehicles:
            packet_timestamps = packet_updates.get(vehicle.snapshot.vehicle_id, (vehicle.knowledge.packet_timestamps, vehicle.knowledge.packet_updated_flags))[0]
            caoi_entries.append(
                (
                    vehicle.is_autonomous,
                    vehicle.priority_weight,
                    self._calculate_caoi(packet_timestamps, current_time),
                )
            )
        return caoi_entries

    def _count_age_violations(self, caoi_values: list[float]) -> int:
        tolerance = self._environment_config.age_tolerance_seconds
        if tolerance is None:
            return 0
        return sum(caoi > tolerance for caoi in caoi_values)

    def _enumerate_candidate_links(
        self,
        active_vehicles: tuple[VehicleState, ...],
        current_time: float,
        delta_seconds: float,
    ) -> tuple[list[CandidateLink], list[CandidateLink]]:
        v2i_candidates: list[CandidateLink] = []
        v2v_candidates: list[CandidateLink] = []
        rsu_packet_timestamps = tuple(
            current_time for _ in range(self._environment_config.packet_count)
        )
        rsu_packet_updated_flags = tuple(True for _ in range(self._environment_config.packet_count))

        for vehicle in active_vehicles:
            budget = self._evaluate_link_budget(
                self._rsu_position,
                vehicle.snapshot.position,
                active_vehicles,
                excluded_vehicle_ids={vehicle.snapshot.vehicle_id},
            )
            if not budget.reachable:
                continue
            expected_packet_count = self._expected_packet_count(budget.rate_mbps, delta_seconds)
            fresher_indices = self._sorted_improvement_indices(
                rsu_packet_timestamps,
                vehicle.knowledge.packet_timestamps,
                rsu_packet_updated_flags,
                vehicle.knowledge.packet_updated_flags,
            )
            post_packets, post_flags, delivered_packet_count = self._apply_packet_transfer(
                receiver_packet_timestamps=vehicle.knowledge.packet_timestamps,
                receiver_packet_updated_flags=vehicle.knowledge.packet_updated_flags,
                sender_packet_timestamps=rsu_packet_timestamps,
                sender_packet_updated_flags=rsu_packet_updated_flags,
                transferable_indices=fresher_indices,
                expected_packet_count=expected_packet_count,
            )
            current_caoi = vehicle.knowledge.caoi_seconds
            post_caoi = self._calculate_caoi(post_packets, current_time)
            gain = max(current_caoi - post_caoi, 0.0)
            priority_gain = gain * vehicle.priority_weight
            relay_peer_count, relay_peer_total_gain, relay_peer_total_priority_gain = (
                self._estimate_seed_relay_potential(
                    seed_vehicle_id=vehicle.snapshot.vehicle_id,
                    seed_position=vehicle.snapshot.position,
                    seed_packet_timestamps=post_packets,
                    seed_packet_updated_flags=post_flags,
                    active_vehicles=active_vehicles,
                    current_time=current_time,
                    delta_seconds=delta_seconds,
                )
            )
            v2i_candidates.append(
                CandidateLink(
                    action=LinkAction(
                        link_type="v2i",
                        transmitter_id="rsu",
                        receiver_id=vehicle.snapshot.vehicle_id,
                        transmitter_position=self._rsu_position,
                        receiver_position=vehicle.snapshot.position,
                    ),
                    link_budget=budget,
                    transmitter_packet_timestamps=rsu_packet_timestamps,
                    transmitter_packet_updated_flags=rsu_packet_updated_flags,
                    receiver_packet_timestamps=vehicle.knowledge.packet_timestamps,
                    receiver_packet_updated_flags=vehicle.knowledge.packet_updated_flags,
                    fresher_packet_count=len(fresher_indices),
                    expected_packet_count=expected_packet_count,
                    delivered_packet_count=delivered_packet_count,
                    post_receiver_packet_timestamps=post_packets,
                    post_receiver_packet_updated_flags=post_flags,
                    post_caoi_seconds=post_caoi,
                    freshness_gain_seconds=gain,
                    receiver_is_autonomous=vehicle.is_autonomous,
                    receiver_priority_weight=vehicle.priority_weight,
                    priority_gain_seconds=priority_gain,
                    relay_peer_count=relay_peer_count,
                    relay_peer_total_gain_seconds=relay_peer_total_gain,
                    relay_peer_total_priority_gain_seconds=relay_peer_total_priority_gain,
                )
            )

        for sender in active_vehicles:
            for receiver in active_vehicles:
                if sender.snapshot.vehicle_id == receiver.snapshot.vehicle_id:
                    continue
                fresher_indices = self._sorted_improvement_indices(
                    sender.knowledge.packet_timestamps,
                    receiver.knowledge.packet_timestamps,
                    sender.knowledge.packet_updated_flags,
                    receiver.knowledge.packet_updated_flags,
                )
                if not fresher_indices:
                    continue
                budget = self._evaluate_link_budget(
                    sender.snapshot.position,
                    receiver.snapshot.position,
                    active_vehicles,
                    excluded_vehicle_ids={sender.snapshot.vehicle_id, receiver.snapshot.vehicle_id},
                )
                if not budget.reachable:
                    continue
                expected_packet_count = self._expected_packet_count(budget.rate_mbps, delta_seconds)
                post_packets, post_flags, delivered_packet_count = self._apply_packet_transfer(
                    receiver_packet_timestamps=receiver.knowledge.packet_timestamps,
                    receiver_packet_updated_flags=receiver.knowledge.packet_updated_flags,
                    sender_packet_timestamps=sender.knowledge.packet_timestamps,
                    sender_packet_updated_flags=sender.knowledge.packet_updated_flags,
                    transferable_indices=fresher_indices,
                    expected_packet_count=expected_packet_count,
                )
                current_caoi = receiver.knowledge.caoi_seconds
                post_caoi = self._calculate_caoi(post_packets, current_time)
                gain = max(current_caoi - post_caoi, 0.0)
                priority_gain = gain * receiver.priority_weight
                v2v_candidates.append(
                    CandidateLink(
                        action=LinkAction(
                            link_type="v2v",
                            transmitter_id=f"vehicle:{sender.snapshot.vehicle_id}",
                            receiver_id=receiver.snapshot.vehicle_id,
                            transmitter_position=sender.snapshot.position,
                            receiver_position=receiver.snapshot.position,
                        ),
                        link_budget=budget,
                        transmitter_packet_timestamps=sender.knowledge.packet_timestamps,
                        transmitter_packet_updated_flags=sender.knowledge.packet_updated_flags,
                        receiver_packet_timestamps=receiver.knowledge.packet_timestamps,
                        receiver_packet_updated_flags=receiver.knowledge.packet_updated_flags,
                        fresher_packet_count=len(fresher_indices),
                        expected_packet_count=expected_packet_count,
                        delivered_packet_count=delivered_packet_count,
                        post_receiver_packet_timestamps=post_packets,
                        post_receiver_packet_updated_flags=post_flags,
                        post_caoi_seconds=post_caoi,
                        freshness_gain_seconds=gain,
                        receiver_is_autonomous=receiver.is_autonomous,
                        receiver_priority_weight=receiver.priority_weight,
                        priority_gain_seconds=priority_gain,
                    )
                )

        return v2i_candidates, v2v_candidates

    def _estimate_seed_relay_potential(
        self,
        seed_vehicle_id: int,
        seed_position: Vector3,
        seed_packet_timestamps: tuple[float, ...],
        seed_packet_updated_flags: tuple[bool, ...],
        active_vehicles: tuple[VehicleState, ...],
        current_time: float,
        delta_seconds: float,
    ) -> tuple[int, float, float]:
        relay_peer_count = 0
        relay_peer_total_gain = 0.0
        relay_peer_total_priority_gain = 0.0
        for receiver in active_vehicles:
            receiver_vehicle_id = receiver.snapshot.vehicle_id
            if receiver_vehicle_id == seed_vehicle_id:
                continue
            fresher_indices = self._sorted_improvement_indices(
                seed_packet_timestamps,
                receiver.knowledge.packet_timestamps,
                seed_packet_updated_flags,
                receiver.knowledge.packet_updated_flags,
            )
            if not fresher_indices:
                continue
            budget = self._evaluate_link_budget(
                seed_position,
                receiver.snapshot.position,
                active_vehicles,
                excluded_vehicle_ids={seed_vehicle_id, receiver_vehicle_id},
            )
            if not budget.reachable:
                continue
            expected_packet_count = self._expected_packet_count(budget.rate_mbps, delta_seconds)
            post_packets, _, _ = self._apply_packet_transfer(
                receiver_packet_timestamps=receiver.knowledge.packet_timestamps,
                receiver_packet_updated_flags=receiver.knowledge.packet_updated_flags,
                sender_packet_timestamps=seed_packet_timestamps,
                sender_packet_updated_flags=seed_packet_updated_flags,
                transferable_indices=fresher_indices,
                expected_packet_count=expected_packet_count,
            )
            current_caoi = receiver.knowledge.caoi_seconds
            post_caoi = self._calculate_caoi(post_packets, current_time)
            gain = max(current_caoi - post_caoi, 0.0)
            if gain <= 0.0:
                continue
            relay_peer_count += 1
            relay_peer_total_gain += gain
            relay_peer_total_priority_gain += gain * receiver.priority_weight
        return relay_peer_count, relay_peer_total_gain, relay_peer_total_priority_gain

    def _evaluate_link_budget(
        self,
        transmitter_position: Vector3,
        receiver_position: Vector3,
        active_vehicles: tuple[VehicleState, ...],
        excluded_vehicle_ids: set[int],
    ) -> LinkBudget:
        blockers = tuple(
            vehicle.snapshot.position
            for vehicle in active_vehicles
            if vehicle.snapshot.vehicle_id not in excluded_vehicle_ids
        )
        return self._link_evaluator.evaluate(
            transmitter_position,
            receiver_position,
            blockers=blockers,
        )

    def _expected_packet_count(self, rate_mbps: float, delta_seconds: float) -> float:
        delivered_bits = max(rate_mbps, 0.0) * 1_000_000.0 * max(delta_seconds, 0.0)
        packet_size_bits = self._environment_config.packet_size_bits
        if packet_size_bits <= 0.0:
            raise ValueError("packet_size_bits must be positive.")
        return delivered_bits / packet_size_bits

    @staticmethod
    def _calculate_caoi(packet_timestamps: tuple[float, ...], current_time: float) -> float:
        if not packet_timestamps:
            return 0.0
        return mean(max(current_time - timestamp, 0.0) for timestamp in packet_timestamps)

    @staticmethod
    def _sorted_improvement_indices(
        sender_packet_timestamps: tuple[float, ...],
        receiver_packet_timestamps: tuple[float, ...],
        sender_packet_updated_flags: tuple[bool, ...],
        receiver_packet_updated_flags: tuple[bool, ...],
    ) -> list[int]:
        candidate_indices = [
            index
            for index, (
                sender_timestamp,
                receiver_timestamp,
                sender_updated,
                receiver_updated,
            ) in enumerate(
                zip(
                    sender_packet_timestamps,
                    receiver_packet_timestamps,
                    sender_packet_updated_flags,
                    receiver_packet_updated_flags,
                    strict=True,
                )
            )
            if sender_updated
            and (
                not receiver_updated
                or sender_timestamp > receiver_timestamp
            )
        ]
        candidate_indices.sort(
            key=lambda index: (
                sender_packet_timestamps[index] - receiver_packet_timestamps[index],
                -receiver_packet_timestamps[index],
            ),
            reverse=True,
        )
        return candidate_indices

    @staticmethod
    def _apply_packet_transfer(
        receiver_packet_timestamps: tuple[float, ...],
        receiver_packet_updated_flags: tuple[bool, ...],
        sender_packet_timestamps: tuple[float, ...],
        sender_packet_updated_flags: tuple[bool, ...],
        transferable_indices: list[int],
        expected_packet_count: float,
    ) -> tuple[tuple[float, ...], tuple[bool, ...], float]:
        if not transferable_indices or expected_packet_count <= 0.0:
            return receiver_packet_timestamps, receiver_packet_updated_flags, 0.0

        delivered_packet_count = min(expected_packet_count, float(len(transferable_indices)))
        updated_packets = list(receiver_packet_timestamps)
        updated_flags = list(receiver_packet_updated_flags)
        full_packets = int(delivered_packet_count)

        for index in transferable_indices[:full_packets]:
            updated_packets[index] = sender_packet_timestamps[index]
            updated_flags[index] = sender_packet_updated_flags[index]

        remainder = delivered_packet_count - full_packets
        if remainder > 1e-9 and full_packets < len(transferable_indices):
            partial_index = transferable_indices[full_packets]
            receiver_timestamp = receiver_packet_timestamps[partial_index]
            sender_timestamp = sender_packet_timestamps[partial_index]
            updated_packets[partial_index] = (
                (1.0 - remainder) * receiver_timestamp + remainder * sender_timestamp
            )
            updated_flags[partial_index] = sender_packet_updated_flags[partial_index]

        return tuple(updated_packets), tuple(updated_flags), delivered_packet_count

    @staticmethod
    def _vehicle_nodes_for_action(action: LinkAction) -> tuple[int, ...]:
        nodes = [action.receiver_id]
        if action.link_type == "v2v" and action.transmitter_id.startswith("vehicle:"):
            nodes.append(int(action.transmitter_id.split(":", 1)[1]))
        return tuple(nodes)

    @staticmethod
    def _split_candidate_links(
        candidate_links: tuple[CandidateLink, ...],
    ) -> tuple[tuple[CandidateLink, ...], tuple[CandidateLink, ...]]:
        v2i_links = tuple(candidate for candidate in candidate_links if candidate.action.link_type == "v2i")
        v2v_links = tuple(candidate for candidate in candidate_links if candidate.action.link_type == "v2v")
        return v2i_links, v2v_links
