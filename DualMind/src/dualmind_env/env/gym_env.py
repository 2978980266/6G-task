from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import ChannelModelConfig, EnvironmentConfig
from ..data.trajectory_loader import TrajectoryReplay
from ..types import BuildingSnapshot, Observation, Vector3
from ..wireless.channel import ChannelEvaluator
from .aoi_env import AoiSchedulingEnv


@dataclass(frozen=True)
class ObservationScaling:
    max_abs_relative_x: float
    max_abs_relative_z: float
    max_distance_m: float
    max_speed: float
    max_aoi_seconds: float
    max_abs_snr_db: float
    max_rate_mbps: float


class AoiSchedulingGymEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    _VEHICLE_FEATURES = 10
    _GLOBAL_FEATURES = 2

    def __init__(
        self,
        replay: TrajectoryReplay,
        buildings: list[BuildingSnapshot],
        rsu_position: Vector3,
        channel_config: ChannelModelConfig,
        environment_config: EnvironmentConfig,
    ) -> None:
        super().__init__()
        self._replay = replay
        self._raw_env = AoiSchedulingEnv(
            replay=replay,
            buildings=buildings,
            rsu_position=rsu_position,
            channel_config=channel_config,
            environment_config=environment_config,
        )
        self._rsu_position = rsu_position
        self._vehicle_ids = replay.vehicle_ids
        self._max_candidate_count = len(self._vehicle_ids)
        available_steps = min(environment_config.max_steps or len(replay), len(replay))
        self._terminal_steps = max(1, available_steps)
        self._scaling = self._build_scaling(
            replay=replay,
            buildings=buildings,
            rsu_position=rsu_position,
            channel_config=channel_config,
            environment_config=environment_config,
        )
        self._latest_observation: Observation | None = None
        self._latest_ranked_candidates = []

        feature_count = self._max_candidate_count * self._VEHICLE_FEATURES + self._GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(feature_count,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._max_candidate_count)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options

        observation = self._raw_env.reset()
        self._latest_observation = observation
        self._latest_ranked_candidates = self._rank_candidates(observation)
        encoded = self._encode_observation(observation)
        info = self._build_reset_info(observation)
        return encoded, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        current_ranked_candidates = list(self._latest_ranked_candidates)
        current_action_mask = self._build_action_mask(current_ranked_candidates)
        action_vehicle_id = self._decode_action(action, current_ranked_candidates)
        result = self._raw_env.step(action_vehicle_id)
        self._latest_observation = result.observation
        self._latest_ranked_candidates = (
            self._rank_candidates(result.observation)
            if result.observation is not None
            else []
        )

        terminated = result.done
        truncated = False
        next_observation = (
            self._encode_observation(result.observation)
            if result.observation is not None
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = dict(result.info)
        info["selected_action_index"] = int(action)
        info["selected_vehicle_id"] = action_vehicle_id
        info["candidate_vehicle_ids"] = [vehicle.vehicle_id for vehicle in current_ranked_candidates]
        info["action_mask"] = current_action_mask.astype(np.int8)
        return next_observation, result.reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._build_action_mask(self._latest_ranked_candidates)

    def get_action_mapping(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for index, vehicle in enumerate(self._latest_ranked_candidates):
            mapping[index] = vehicle.vehicle_id
        return mapping

    def get_latest_raw_observation(self) -> Observation | None:
        return self._latest_observation

    def _decode_action(self, action: int, ranked_candidates) -> int | None:
        if action < 0 or action >= len(ranked_candidates):
            return None
        return ranked_candidates[action].vehicle_id

    def _build_reset_info(self, observation: Observation) -> dict[str, object]:
        return {
            "step_index": observation.step_index,
            "simulation_frame": observation.simulation_frame,
            "active_vehicle_count": len(observation.active_vehicles),
            "candidate_vehicle_ids": [vehicle.vehicle_id for vehicle in self._latest_ranked_candidates],
            "action_mask": self.action_masks().astype(np.int8),
        }

    def _encode_observation(self, observation: Observation) -> np.ndarray:
        encoded = np.zeros(self.observation_space.shape, dtype=np.float32)

        ranked_candidates = self._rank_candidates(observation)
        self._latest_ranked_candidates = ranked_candidates

        for slot, vehicle in enumerate(ranked_candidates):
            base = slot * self._VEHICLE_FEATURES
            relative_x = vehicle.position.x - self._rsu_position.x
            relative_z = vehicle.position.z - self._rsu_position.z

            encoded[base + 0] = 1.0
            encoded[base + 1] = 1.0 if vehicle.link_state.reachable else 0.0
            encoded[base + 2] = self._normalize_positive(
                vehicle.aoi_seconds,
                self._scaling.max_aoi_seconds,
            )
            encoded[base + 3] = self._normalize_signed(
                relative_x,
                self._scaling.max_abs_relative_x,
            )
            encoded[base + 4] = self._normalize_signed(
                relative_z,
                self._scaling.max_abs_relative_z,
            )
            encoded[base + 5] = self._normalize_positive(
                vehicle.link_state.distance_m,
                self._scaling.max_distance_m,
            )
            encoded[base + 6] = self._normalize_positive(
                vehicle.speed,
                self._scaling.max_speed,
            )
            encoded[base + 7] = 1.0 if vehicle.link_state.los else 0.0
            encoded[base + 8] = self._normalize_signed(
                vehicle.link_state.snr_db,
                self._scaling.max_abs_snr_db,
            )
            encoded[base + 9] = self._normalize_positive(
                vehicle.link_state.rate_mbps,
                self._scaling.max_rate_mbps,
            )

        global_offset = self._max_candidate_count * self._VEHICLE_FEATURES
        encoded[global_offset] = self._normalize_positive(
            observation.step_index,
            max(self._terminal_steps - 1, 1),
        )
        encoded[global_offset + 1] = self._normalize_positive(
            len(observation.active_vehicles),
            max(self._max_candidate_count, 1),
        )
        return encoded

    @staticmethod
    def _rank_candidates(observation: Observation | None):
        if observation is None:
            return []
        return sorted(
            observation.active_vehicles,
            key=lambda vehicle: (
                not vehicle.link_state.reachable,
                -vehicle.aoi_seconds,
                vehicle.link_state.distance_m,
                vehicle.vehicle_id,
            ),
        )

    def _build_action_mask(self, ranked_candidates) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        for slot, vehicle in enumerate(ranked_candidates):
            mask[slot] = vehicle.link_state.reachable
        return mask

    @staticmethod
    def _normalize_positive(value: float, scale: float) -> np.float32:
        if scale <= 0:
            return np.float32(0.0)
        return np.float32(min(value / scale, 1.0))

    @staticmethod
    def _normalize_signed(value: float, scale: float) -> np.float32:
        if scale <= 0:
            return np.float32(0.0)
        normalized = max(min(value / scale, 1.0), -1.0)
        return np.float32(normalized)

    @staticmethod
    def _build_scaling(
        replay: TrajectoryReplay,
        buildings: list[BuildingSnapshot],
        rsu_position: Vector3,
        channel_config: ChannelModelConfig,
        environment_config: EnvironmentConfig,
    ) -> ObservationScaling:
        evaluator = ChannelEvaluator(channel_config, rsu_position, buildings)

        max_abs_relative_x = 1.0
        max_abs_relative_z = 1.0
        max_distance_m = 1.0
        max_speed = 1.0
        max_abs_snr_db = 1.0
        max_rate_mbps = 1.0

        for frame in replay.frames:
            for vehicle in frame.vehicles.values():
                link_state = evaluator.evaluate(vehicle)
                relative_x = abs(vehicle.position.x - rsu_position.x)
                relative_z = abs(vehicle.position.z - rsu_position.z)
                max_abs_relative_x = max(max_abs_relative_x, relative_x)
                max_abs_relative_z = max(max_abs_relative_z, relative_z)
                max_distance_m = max(max_distance_m, link_state.distance_m)
                max_speed = max(max_speed, vehicle.speed)
                max_abs_snr_db = max(max_abs_snr_db, abs(link_state.snr_db))
                max_rate_mbps = max(max_rate_mbps, link_state.rate_mbps)

        available_steps = min(environment_config.max_steps or len(replay), len(replay))
        max_aoi_seconds = max(
            environment_config.slot_seconds,
            environment_config.slot_seconds * max(available_steps, 1),
        )

        return ObservationScaling(
            max_abs_relative_x=max_abs_relative_x,
            max_abs_relative_z=max_abs_relative_z,
            max_distance_m=max_distance_m,
            max_speed=max_speed,
            max_aoi_seconds=max_aoi_seconds,
            max_abs_snr_db=max_abs_snr_db,
            max_rate_mbps=max_rate_mbps,
        )
