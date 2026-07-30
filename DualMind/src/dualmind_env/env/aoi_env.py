from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from ..config import ChannelModelConfig, EnvironmentConfig
from ..data.trajectory_loader import TrajectoryReplay
from ..types import ActiveVehicleObservation, BuildingSnapshot, Observation, Vector3
from ..wireless.channel import ChannelEvaluator


@dataclass(frozen=True)
class StepResult:
    observation: Observation | None
    reward: float
    done: bool
    info: dict[str, Any]


class AoiSchedulingEnv:
    def __init__(
        self,
        replay: TrajectoryReplay,
        buildings: list[BuildingSnapshot],
        rsu_position: Vector3,
        channel_config: ChannelModelConfig,
        environment_config: EnvironmentConfig,
    ) -> None:
        self._replay = replay
        self._buildings = buildings
        self._rsu_position = rsu_position
        self._channel_config = channel_config
        self._environment_config = environment_config
        self._channel = ChannelEvaluator(channel_config, rsu_position, buildings)
        self._terminal_step = environment_config.max_steps or len(replay)
        self._terminal_step = min(self._terminal_step, len(replay))
        self._ages: dict[int, float] = {}
        self._current_step = 0
        self._started = False

    def reset(self) -> Observation:
        self._ages.clear()
        self._current_step = 0
        self._started = True
        if self._terminal_step <= 0:
            raise ValueError("Environment has no available steps.")
        return self._build_observation(self._current_step)

    def step(self, action_vehicle_id: int | None) -> StepResult:
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self._current_step >= self._terminal_step:
            raise RuntimeError("Episode is already finished.")

        observation = self._build_observation(self._current_step)
        active_ids = [vehicle.vehicle_id for vehicle in observation.active_vehicles]

        for vehicle_id in active_ids:
            current_age = self._ages.get(vehicle_id, self._environment_config.initial_aoi_seconds)
            self._ages[vehicle_id] = current_age + self._environment_config.slot_seconds

        chosen = next(
            (vehicle for vehicle in observation.active_vehicles if vehicle.vehicle_id == action_vehicle_id),
            None,
        )
        selected_aoi_before = chosen.aoi_seconds if chosen is not None else 0.0
        success = bool(chosen and chosen.link_state.reachable)
        if success and action_vehicle_id is not None:
            self._ages[action_vehicle_id] = 0.0

        mean_aoi = mean(self._ages[vehicle_id] for vehicle_id in active_ids) if active_ids else 0.0
        reachable_rates = [vehicle.link_state.rate_mbps for vehicle in observation.active_vehicles if vehicle.link_state.reachable]
        mean_reachable_rate = mean(reachable_rates) if reachable_rates else 0.0
        reachable_vehicles = [vehicle for vehicle in observation.active_vehicles if vehicle.link_state.reachable]
        best_reachable = max(
            reachable_vehicles,
            key=lambda vehicle: (vehicle.aoi_seconds, vehicle.link_state.distance_m),
            default=None,
        )
        reward = -mean_aoi
        if success:
            reward += self._environment_config.success_bonus
            reward += self._environment_config.aoi_priority_bonus_scale * selected_aoi_before
        elif action_vehicle_id is not None and active_ids:
            reward -= self._environment_config.invalid_action_penalty

        info = {
            "selected_vehicle_id": action_vehicle_id,
            "success": success,
            "action_valid": chosen is not None,
            "selected_vehicle_aoi_before": selected_aoi_before,
            "active_vehicle_count": len(active_ids),
            "reachable_vehicle_count": sum(vehicle.link_state.reachable for vehicle in observation.active_vehicles),
            "best_reachable_vehicle_id": best_reachable.vehicle_id if best_reachable is not None else None,
            "best_reachable_aoi": best_reachable.aoi_seconds if best_reachable is not None else 0.0,
            "mean_aoi": mean_aoi,
            "mean_reachable_rate_mbps": mean_reachable_rate,
        }

        self._current_step += 1
        done = self._current_step >= self._terminal_step
        next_observation = None if done else self._build_observation(self._current_step)
        return StepResult(observation=next_observation, reward=reward, done=done, info=info)

    def _build_observation(self, step_index: int) -> Observation:
        frame = self._replay.frames[step_index]
        active: list[ActiveVehicleObservation] = []

        for vehicle in frame.vehicles.values():
            link_state = self._channel.evaluate(vehicle)
            aoi_seconds = self._ages.get(vehicle.vehicle_id, self._environment_config.initial_aoi_seconds)
            active.append(
                ActiveVehicleObservation(
                    vehicle_id=vehicle.vehicle_id,
                    position=vehicle.position,
                    speed=vehicle.speed,
                    aoi_seconds=aoi_seconds,
                    link_state=link_state,
                    prefab_name=vehicle.prefab_name,
                )
            )

        active.sort(key=lambda item: item.vehicle_id)
        return Observation(
            step_index=step_index,
            simulation_frame=frame.simulation_frame,
            game_time=frame.game_time,
            record_time_utc=frame.record_time_utc,
            active_vehicles=tuple(active),
        )
