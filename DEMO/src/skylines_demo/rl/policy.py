from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..bootstrap import load_runtime
from ..types import SchedulingAction, StepObservation
from .task_common import (
    TaskEncodingContext,
    build_task_encoding_context,
    build_task_scheduling_action,
    encode_task_observation,
)
from .metadata import (
    load_ppo_metadata,
    metadata_path_for_model,
    validate_ppo_metadata,
)


class PpoInferencePolicy:
    def __init__(
        self,
        config_path: str | Path,
        model_path: str | Path,
        top_k: int | None = None,
        action_slots: int | None = None,
        allow_legacy_model: bool = False,
    ) -> None:
        resolved_model_path = _resolve_model_path(model_path)
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            from stable_baselines3 import PPO
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModuleNotFoundError(
                "缺少 PPO 推理依赖。请先安装 gymnasium、stable-baselines3 和 torch。"
            ) from exc

        self._runtime = load_runtime(config_path)
        self._model = PPO.load(str(resolved_model_path))
        raw_nvec = getattr(self._model.action_space, "nvec", None)
        if raw_nvec is None:
            raise ValueError(
                "Loaded PPO model must use the task-aware MultiDiscrete action space."
            )
        nvec = np.asarray(raw_nvec, dtype=np.int64).reshape(-1)
        if nvec.size <= 0 or np.any(nvec != nvec[0]):
            raise ValueError(
                "Loaded PPO model has an invalid task-aware action space."
            )
        derived_top_k = int(nvec[0]) - 1
        derived_action_slots = int(nvec.size)
        if derived_top_k <= 0:
            raise ValueError("Loaded PPO model has an invalid action space.")
        if top_k is not None and top_k != derived_top_k:
            raise ValueError(
                f"Provided top_k={top_k} does not match the PPO model action space top_k={derived_top_k}."
            )
        if (
            action_slots is not None
            and action_slots != derived_action_slots
        ):
            raise ValueError(
                "Provided action_slots="
                f"{action_slots} does not match the PPO model "
                f"action_slots={derived_action_slots}."
            )

        self._top_k = derived_top_k
        self._action_slots = derived_action_slots
        self._encoding_context: TaskEncodingContext = (
            build_task_encoding_context(
                top_k=self._top_k,
                replay=self._runtime.replay,
                rsu_position=self._runtime.config.rsu.position,
                terminal_steps=max(
                    min(
                        self._runtime.config.environment.max_steps
                        or len(self._runtime.replay),
                        len(self._runtime.replay),
                    ),
                    1,
                ),
                packet_count=max(
                    self._runtime.config.environment.packet_count,
                    1,
                ),
                age_tolerance_seconds=(
                    self._runtime.config.environment.age_tolerance_seconds
                    or 8.0
                ),
                task_reward_base=self._runtime.config.tasks.reward_base,
                max_completed_tasks_per_vehicle=(
                    self._runtime.config.tasks.max_completed_tasks_per_vehicle
                ),
                resource_config=self._runtime.config.resource,
                autonomy_enabled=self._runtime.config.autonomy.enabled,
            )
        )
        expected_feature_count = (
            self._encoding_context.observation_feature_count
        )
        raw_observation_shape = self._model.observation_space.shape
        if raw_observation_shape is None:
            raise ValueError("Loaded PPO model observation space must define a shape.")
        observation_shape = tuple(
            int(dimension)
            for dimension in raw_observation_shape
        )
        if np.prod(observation_shape) != expected_feature_count:
            raise ValueError(
                "Loaded PPO model observation size does not match the expected "
                "task-aware DEMO PPO encoding."
            )
        model_metadata_path = metadata_path_for_model(resolved_model_path)
        self._metadata = load_ppo_metadata(model_metadata_path)
        validate_ppo_metadata(
            self._metadata,
            metadata_path=model_metadata_path,
            runtime=self._runtime,
            top_k=self._top_k,
            action_slots=self._action_slots,
            observation_feature_count=expected_feature_count,
            allow_legacy_model=allow_legacy_model,
        )

    def reset(self) -> None:
        pass

    def select_action(self, observation: StepObservation) -> SchedulingAction:
        encoded_observation = encode_task_observation(
            observation,
            self._encoding_context,
        )
        action, _ = self._model.predict(encoded_observation, deterministic=True)
        action_indices = tuple(
            int(index)
            for index in np.asarray(action, dtype=np.int64).reshape(-1)
        )
        if len(action_indices) != self._action_slots:
            raise ValueError(
                "Loaded PPO model returned an unexpected number of action slots."
            )
        return build_task_scheduling_action(
            observation=observation,
            action_indices=action_indices,
            top_k=self._top_k,
            resource_config=self._runtime.config.resource,
            allow_multi_v2i_from_rsu=self._runtime.config.environment.allow_multi_v2i_from_rsu,
            max_active_links=self._runtime.config.environment.max_active_links,
        )


def _resolve_model_path(model_path: str | Path) -> Path:
    candidate = Path(model_path)
    if candidate.exists():
        return candidate
    zipped_candidate = candidate if candidate.suffix == ".zip" else candidate.with_suffix(".zip")
    if zipped_candidate.exists():
        return zipped_candidate
    raise FileNotFoundError(f"PPO model not found: {candidate}")
