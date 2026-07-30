from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from statistics import mean

from ..config import TaskConfig
from ..types import TaskCompletion, TaskTemplate, VehicleTaskState


@dataclass
class _MutableVehicleTaskState:
    task_order: tuple[str, ...]
    next_task_index: int = 0
    active_task_id: str | None = None
    completed_task_ids: tuple[str, ...] = ()
    cooldown_until_step: int = 0


class TaskManager:
    def __init__(
        self,
        config: TaskConfig,
        vehicle_ids: tuple[int, ...],
        seed: int | None,
    ) -> None:
        self._config = config
        self._templates = {
            task_id: TaskTemplate(task_id=task_id, required_packet_ids=packet_ids)
            for task_id, packet_ids in config.templates
        }
        self._vehicle_ids = vehicle_ids
        self._seed = seed
        self._unseeded_rng = random.Random() if seed is None else None
        self._states: dict[int, _MutableVehicleTaskState] = {}

    @property
    def templates(self) -> tuple[TaskTemplate, ...]:
        return tuple(self._templates.values())

    def reset(self) -> None:
        if self._seed is None:
            assert self._unseeded_rng is not None
            episode_seed = self._unseeded_rng.getrandbits(64)
        else:
            episode_seed = self._seed
        self._states = {
            vehicle_id: _MutableVehicleTaskState(
                task_order=self._build_task_order(vehicle_id, episode_seed)
            )
            for vehicle_id in self._vehicle_ids
        }

    def begin_step(self, step_index: int, active_vehicle_ids: tuple[int, ...]) -> None:
        if not self._config.enabled:
            return
        for vehicle_id in active_vehicle_ids:
            state = self._states[vehicle_id]
            if state.active_task_id is not None:
                continue
            if len(state.completed_task_ids) >= self._config.max_completed_tasks_per_vehicle:
                continue
            if step_index < state.cooldown_until_step:
                continue
            if state.next_task_index >= len(state.task_order):
                continue
            state.active_task_id = state.task_order[state.next_task_index]
            state.next_task_index += 1

    def state_for(self, vehicle_id: int, step_index: int) -> VehicleTaskState:
        state = self._states[vehicle_id]
        required_packet_ids = (
            self._templates[state.active_task_id].required_packet_ids
            if state.active_task_id is not None
            else ()
        )
        can_request = (
            self._config.enabled
            and state.active_task_id is None
            and len(state.completed_task_ids) < self._config.max_completed_tasks_per_vehicle
            and step_index >= state.cooldown_until_step
            and state.next_task_index < len(state.task_order)
        )
        return VehicleTaskState(
            active_task_id=state.active_task_id,
            required_packet_ids=required_packet_ids,
            completed_task_ids=state.completed_task_ids,
            cooldown_until_step=state.cooldown_until_step,
            can_request_new_task=can_request,
        )

    def complete_ready_tasks(
        self,
        step_index: int,
        active_vehicle_ids: tuple[int, ...],
        packet_timestamps_by_vehicle: dict[int, tuple[float, ...]],
        packet_flags_by_vehicle: dict[int, tuple[bool, ...]],
        completion_time: float,
    ) -> tuple[TaskCompletion, ...]:
        completions: list[TaskCompletion] = []
        if not self._config.enabled:
            return ()

        for vehicle_id in active_vehicle_ids:
            state = self._states[vehicle_id]
            if state.active_task_id is None:
                continue
            template = self._templates[state.active_task_id]
            reward_evaluation = evaluate_task_reward(
                config=self._config,
                required_packet_ids=template.required_packet_ids,
                packet_timestamps=packet_timestamps_by_vehicle[vehicle_id],
                packet_flags=packet_flags_by_vehicle[vehicle_id],
                completion_time=completion_time,
            )
            if reward_evaluation is None:
                continue

            mean_packet_aoi_seconds, reward = reward_evaluation
            completions.append(
                TaskCompletion(
                    vehicle_id=vehicle_id,
                    task_id=template.task_id,
                    required_packet_ids=template.required_packet_ids,
                    mean_packet_aoi_seconds=mean_packet_aoi_seconds,
                    reward=reward,
                )
            )
            state.completed_task_ids = state.completed_task_ids + (template.task_id,)
            state.active_task_id = None
            state.cooldown_until_step = step_index + self._config.cooldown_steps + 1

        return tuple(completions)

    def _build_task_order(
        self,
        vehicle_id: int,
        episode_seed: int,
    ) -> tuple[str, ...]:
        task_ids = list(self._templates)
        seed_material = f"{episode_seed}:{vehicle_id}:task-order".encode("utf-8")
        derived_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            byteorder="big",
            signed=False,
        )
        random.Random(derived_seed).shuffle(task_ids)
        return tuple(task_ids)


def evaluate_task_reward(
    config: TaskConfig,
    required_packet_ids: tuple[int, ...],
    packet_timestamps: tuple[float, ...],
    packet_flags: tuple[bool, ...],
    completion_time: float,
) -> tuple[float, float] | None:
    if config.freshness_decay_seconds <= 0.0:
        raise ValueError("freshness_decay_seconds must be positive.")
    if not required_packet_ids:
        return None
    required_indices = tuple(packet_id - 1 for packet_id in required_packet_ids)
    if not all(
        0 <= index < len(packet_flags)
        and index < len(packet_timestamps)
        and packet_flags[index]
        for index in required_indices
    ):
        return None

    ages = tuple(
        max(completion_time - packet_timestamps[index], 0.0)
        for index in required_indices
    )
    freshness_values = tuple(
        math.exp(-age / config.freshness_decay_seconds)
        for age in ages
    )
    return mean(ages), config.reward_base * mean(freshness_values)
