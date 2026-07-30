from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..bootstrap import ScenarioRuntime
from ..env import TaskRelaySchedulingEnv
from ..types import CandidateLink, SchedulingAction, StepObservation
from .task_common import (
    TaskEncodingContext,
    build_task_action_mask,
    build_task_encoding_context,
    build_task_scheduling_action,
    encode_task_observation,
    rank_task_candidates,
)


@dataclass(frozen=True)
class PpoStepInfo:
    selected_action_indices: tuple[int, ...]
    used_empty_action: bool
    reward: float


class PpoSchedulingGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        runtime: ScenarioRuntime,
        top_k: int = 8,
        action_slots: int = 8,
        seed: int | None = None,
        task_reward_weight: float = 1.0,
        task_progress_weight: float = 0.1,
        caoi_gain_weight: float = 0.05,
        empty_action_penalty: float = 0.05,
        invalid_action_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if action_slots <= 0:
            raise ValueError("action_slots must be positive.")
        self._runtime = runtime
        self._top_k = top_k
        self._action_slots = action_slots
        self._default_seed = seed
        self._episode_index = 0
        self._task_reward_weight = task_reward_weight
        self._task_progress_weight = task_progress_weight
        self._caoi_gain_weight = caoi_gain_weight
        self._empty_action_penalty = empty_action_penalty
        self._invalid_action_penalty = invalid_action_penalty
        self._terminal_steps = min(runtime.config.environment.max_steps or len(runtime.replay), len(runtime.replay))
        self._encoding_context: TaskEncodingContext = build_task_encoding_context(
            top_k=top_k,
            replay=runtime.replay,
            rsu_position=runtime.config.rsu.position,
            terminal_steps=max(self._terminal_steps, 1),
            packet_count=max(runtime.config.environment.packet_count, 1),
            age_tolerance_seconds=runtime.config.environment.age_tolerance_seconds or 8.0,
            task_reward_base=runtime.config.tasks.reward_base,
            max_completed_tasks_per_vehicle=(
                runtime.config.tasks.max_completed_tasks_per_vehicle
            ),
            resource_config=runtime.config.resource,
            autonomy_enabled=runtime.config.autonomy.enabled,
        )
        self._observation_shape = (
            self._encoding_context.observation_feature_count,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._observation_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(action_slots, top_k + 1, dtype=np.int64)
        )
        self._raw_env = self._build_raw_env(seed)
        self._latest_observation: StepObservation | None = None
        self._latest_ranked_candidates: list[CandidateLink] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options

        if seed is not None:
            self._default_seed = seed
            self._episode_index = 0
        effective_seed = (
            None
            if self._default_seed is None
            else self._default_seed + self._episode_index
        )
        self._episode_index += 1
        self._raw_env = self._build_raw_env(effective_seed)
        observation = self._raw_env.reset()
        self._latest_observation = observation
        self._latest_ranked_candidates = rank_task_candidates(observation)
        return self._encode_latest_observation(), self._build_reset_info(
            observation,
            effective_seed,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._latest_observation is None:
            raise RuntimeError("Call reset() before step().")

        current_observation = self._latest_observation
        current_ranked_candidates = list(self._latest_ranked_candidates)
        action_indices = tuple(
            int(index)
            for index in np.asarray(action, dtype=np.int64).reshape(-1)
        )
        if len(action_indices) != self._action_slots:
            raise ValueError(
                f"Expected {self._action_slots} action slots, got {len(action_indices)}."
            )
        action_mask = build_task_action_mask(
            current_ranked_candidates,
            self._top_k,
        )
        scheduling_action = build_task_scheduling_action(
            observation=current_observation,
            action_indices=action_indices,
            top_k=self._top_k,
            resource_config=self._runtime.config.resource,
            allow_multi_v2i_from_rsu=self._runtime.config.environment.allow_multi_v2i_from_rsu,
            max_active_links=self._runtime.config.environment.max_active_links,
        )
        used_empty_action = not scheduling_action.links
        result = self._raw_env.step(scheduling_action)
        task_reward_gain = sum(
            candidate.task_reward_gain
            for candidate in result.successful_links
        )
        task_progress_gain = sum(
            candidate.task_progress_gain
            for candidate in result.successful_links
        )
        total_priority_weight = sum(
            vehicle.priority_weight
            for vehicle in current_observation.active_vehicles
        )
        caoi_gain = (
            sum(
                candidate.priority_gain_seconds
                for candidate in result.successful_links
            )
            / max(total_priority_weight, 1e-9)
        )
        reward = (
            self._task_reward_weight * task_reward_gain
            + self._task_progress_weight * task_progress_gain
            + self._caoi_gain_weight * caoi_gain
        )
        if current_ranked_candidates and used_empty_action:
            reward -= self._empty_action_penalty
        if result.invalid_action:
            reward -= self._invalid_action_penalty

        episode_reached_data_end = self._terminal_steps >= len(self._runtime.replay)
        terminal_observation = None
        if result.next_observation is None and not episode_reached_data_end:
            terminal_observation = self._raw_env.preview_terminal_step()

        self._latest_observation = result.next_observation or terminal_observation
        self._latest_ranked_candidates = (
            rank_task_candidates(self._latest_observation)
            if self._latest_observation is not None
            else []
        )
        next_observation = (
            self._encode_latest_observation()
            if self._latest_observation is not None
            else np.zeros(self._observation_shape, dtype=np.float32)
        )
        terminated = result.next_observation is None and episode_reached_data_end
        truncated = result.next_observation is None and not episode_reached_data_end
        info = self._build_step_info(
            action_indices=action_indices,
            action_mask=action_mask,
            scheduling_action=scheduling_action,
            reward=reward,
            task_reward_gain=task_reward_gain,
            task_progress_gain=task_progress_gain,
            caoi_gain=caoi_gain,
            result=result,
        )
        return next_observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return build_task_action_mask(
            self._latest_ranked_candidates,
            self._top_k,
        )

    def get_latest_raw_observation(self) -> StepObservation | None:
        return self._latest_observation

    def get_action_mapping(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for slot, candidate in enumerate(self._latest_ranked_candidates[: self._top_k]):
            mapping[slot] = (
                f"{candidate.action.link_type}:{candidate.action.transmitter_id}->{candidate.action.receiver_id}"
            )
        mapping[self._top_k] = "empty"
        return mapping

    def _encode_latest_observation(self) -> np.ndarray:
        if self._latest_observation is None:
            return np.zeros(self._observation_shape, dtype=np.float32)
        return encode_task_observation(
            self._latest_observation,
            self._encoding_context,
        )

    def _build_reset_info(
        self,
        observation: StepObservation,
        episode_seed: int | None,
    ) -> dict[str, object]:
        return {
            "step_index": observation.step_index,
            "simulation_frame": observation.simulation_frame,
            "episode_seed": episode_seed,
            "active_vehicle_count": len(observation.active_vehicles),
            "deliverable_candidate_count": len(self._latest_ranked_candidates),
            "action_mask": self.action_masks().astype(np.int8),
            "action_slots": self._action_slots,
            "task_aware": True,
        }

    def _build_step_info(
        self,
        action_indices: tuple[int, ...],
        action_mask: np.ndarray,
        scheduling_action: SchedulingAction,
        reward: float,
        task_reward_gain: float,
        task_progress_gain: int,
        caoi_gain: float,
        result,
    ) -> dict[str, object]:
        return {
            "selected_action_indices": action_indices,
            "used_empty_action": not scheduling_action.links,
            "chosen_link_count": len(result.chosen_links),
            "successful_link_count": len(result.successful_links),
            "invalid_action": result.invalid_action,
            "task_reward": result.task_reward,
            "task_reward_gain": task_reward_gain,
            "task_progress_gain": task_progress_gain,
            "completed_task_count": len(result.completed_tasks),
            "caoi_gain": caoi_gain,
            "mean_priority_weighted_caoi": result.mean_priority_weighted_caoi_seconds,
            "mean_caoi": result.mean_caoi_seconds,
            "age_violation_count": result.age_violation_count,
            "action_mask": action_mask.astype(np.int8),
            "reward": reward,
        }

    def _build_raw_env(self, seed: int | None) -> TaskRelaySchedulingEnv:
        return TaskRelaySchedulingEnv(
            replay=self._runtime.replay,
            rsu_position=self._runtime.config.rsu.position,
            link_evaluator=self._runtime.channel_model,
            environment_config=self._runtime.config.environment,
            resource_config=self._runtime.config.resource,
            task_config=self._runtime.config.tasks,
            autonomy_config=self._runtime.config.autonomy,
            seed=seed,
        )
