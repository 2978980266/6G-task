from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import build_environment, load_runtime
from skylines_demo.rl import PpoSchedulingGymEnv
from skylines_demo.rl.metadata import (
    build_ppo_metadata,
    validate_ppo_metadata,
)
from skylines_demo.rl.policy import PpoInferencePolicy
from skylines_demo.rl.task_common import (
    build_task_encoding_context,
    build_task_scheduling_action,
    encode_task_observation,
    rank_task_candidates,
)


def main() -> None:
    test_task_candidate_ranking_is_stable()
    test_seeded_task_reset_is_reproducible()
    test_task_candidates_carry_packets_and_resource_blocks()
    test_task_observation_shape_is_stable()
    test_multislot_action_remains_legal()
    test_gym_episode_seed_advances_reproducibly()
    test_gym_max_steps_is_reported_as_truncation()
    test_ppo_metadata_rejects_config_mismatch()
    test_gym_wrapper_uses_attributed_task_reward_and_multidiscrete_actions()
    test_preexisting_task_completion_does_not_reward_ppo_action()
    test_legacy_model_requires_explicit_compatibility_mode()
    test_missing_model_is_rejected()
    print("task-aware ppo wrapper checks passed")


def test_task_candidate_ranking_is_stable() -> None:
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    observation = env.reset()
    ranked_a = rank_task_candidates(observation)
    ranked_b = rank_task_candidates(observation)
    assert [candidate.action for candidate in ranked_a] == [
        candidate.action for candidate in ranked_b
    ]


def test_seeded_task_reset_is_reproducible() -> None:
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    first = env.reset()
    second = env.reset()
    first_state = tuple(
        (
            vehicle.snapshot.vehicle_id,
            vehicle.knowledge.packet_timestamps,
            vehicle.knowledge.packet_updated_flags,
            vehicle.task_state.active_task_id,
            vehicle.task_state.required_packet_ids,
            vehicle.is_autonomous,
        )
        for vehicle in first.active_vehicles
    )
    second_state = tuple(
        (
            vehicle.snapshot.vehicle_id,
            vehicle.knowledge.packet_timestamps,
            vehicle.knowledge.packet_updated_flags,
            vehicle.task_state.active_task_id,
            vehicle.task_state.required_packet_ids,
            vehicle.is_autonomous,
        )
        for vehicle in second.active_vehicles
    )
    assert first_state == second_state


def test_task_candidates_carry_packets_and_resource_blocks() -> None:
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    observation = env.reset()
    ranked = rank_task_candidates(observation)
    assert ranked
    assert all(candidate.link_budget.reachable for candidate in ranked)
    assert all(candidate.resource_block_count >= 1 for candidate in ranked)
    assert all(candidate.selected_packet_ids for candidate in ranked)
    assert all(
        candidate.action.packet_ids == candidate.selected_packet_ids
        for candidate in ranked
    )
    assert all(
        candidate.action.resource_block_count
        == candidate.resource_block_count
        for candidate in ranked
    )


def test_task_observation_shape_is_stable() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    observation = env.reset()
    context = build_task_encoding_context(
        top_k=12,
        replay=runtime.replay,
        rsu_position=runtime.config.rsu.position,
        terminal_steps=len(runtime.replay),
        packet_count=runtime.config.environment.packet_count,
        age_tolerance_seconds=(
            runtime.config.environment.age_tolerance_seconds or 8.0
        ),
        task_reward_base=runtime.config.tasks.reward_base,
        max_completed_tasks_per_vehicle=(
            runtime.config.tasks.max_completed_tasks_per_vehicle
        ),
        resource_config=runtime.config.resource,
        autonomy_enabled=runtime.config.autonomy.enabled,
    )
    encoded = encode_task_observation(observation, context)
    assert encoded.shape == (context.observation_feature_count,)
    assert encoded.dtype == np.float32
    assert np.all(encoded >= 0.0)
    assert np.all(encoded <= 1.0)


def test_multislot_action_remains_legal() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    env = build_environment(DEFAULT_CONFIG_PATH, seed=11)
    observation = env.reset()
    action = build_task_scheduling_action(
        observation=observation,
        action_indices=tuple(range(8)),
        top_k=16,
        resource_config=runtime.config.resource,
        allow_multi_v2i_from_rsu=(
            runtime.config.environment.allow_multi_v2i_from_rsu
        ),
        max_active_links=runtime.config.environment.max_active_links,
    )
    result = env.step(action)
    assert not result.invalid_action
    assert result.used_rsu_resource_blocks <= (
        runtime.config.resource.rsu_resource_block_count
    )
    assert result.used_v2v_resource_blocks <= (
        runtime.config.resource.v2v_resource_block_count
    )
    assert all(candidate.selected_packet_ids for candidate in result.chosen_links)


def test_gym_episode_seed_advances_reproducibly() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    gym_env = PpoSchedulingGymEnv(
        runtime=runtime,
        top_k=16,
        action_slots=8,
        seed=7,
    )
    _, first_info = gym_env.reset()
    _, second_info = gym_env.reset()
    _, explicit_reset_info = gym_env.reset(seed=7)

    assert first_info["episode_seed"] == 7
    assert second_info["episode_seed"] == 8
    assert explicit_reset_info["episode_seed"] == 7


def test_gym_max_steps_is_reported_as_truncation() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    short_config = replace(
        runtime.config,
        environment=replace(runtime.config.environment, max_steps=1),
    )
    short_runtime = replace(runtime, config=short_config)
    gym_env = PpoSchedulingGymEnv(
        runtime=short_runtime,
        top_k=16,
        action_slots=8,
        seed=7,
    )
    gym_env.reset()

    next_observation, _, terminated, truncated, _ = gym_env.step(
        np.full(8, 16, dtype=np.int64)
    )

    assert not terminated
    assert truncated
    assert np.any(next_observation != 0.0)


def test_ppo_metadata_rejects_config_mismatch() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    metadata = build_ppo_metadata(
        runtime=runtime,
        top_k=16,
        action_slots=8,
        observation_feature_count=730,
        training_seed=7,
        task_reward_weight=1.0,
        task_progress_weight=0.1,
        caoi_gain_weight=0.05,
        empty_action_penalty=0.05,
        invalid_action_penalty=0.5,
    )
    metadata["scenario_name"] = "different-scenario"

    try:
        validate_ppo_metadata(
            metadata,
            metadata_path="in-memory.metadata.json",
            runtime=runtime,
            top_k=16,
            action_slots=8,
            observation_feature_count=730,
            allow_legacy_model=False,
        )
    except ValueError as exc:
        assert "configuration" in str(exc).lower()
        return
    raise AssertionError("Mismatched PPO metadata should be rejected.")


def test_gym_wrapper_uses_attributed_task_reward_and_multidiscrete_actions() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    gym_env = PpoSchedulingGymEnv(
        runtime=runtime,
        top_k=16,
        action_slots=8,
        seed=7,
    )
    encoded, reset_info = gym_env.reset()
    assert encoded.shape == gym_env.observation_space.shape
    assert tuple(gym_env.action_space.nvec) == (17,) * 8
    assert reset_info["task_aware"] is True

    _, reward, _, _, step_info = gym_env.step(
        np.arange(8, dtype=np.int64)
    )
    assert isinstance(reward, float)
    assert "task_reward" in step_info
    assert "task_progress_gain" in step_info
    assert "completed_task_count" in step_info
    assert not step_info["invalid_action"]


def test_preexisting_task_completion_does_not_reward_ppo_action() -> None:
    runtime = load_runtime(DEFAULT_CONFIG_PATH)
    gym_env = PpoSchedulingGymEnv(
        runtime=runtime,
        top_k=16,
        action_slots=8,
        seed=7,
    )
    _, reset_info = gym_env.reset()
    assert reset_info["deliverable_candidate_count"] > 0

    raw_env = gym_env._raw_env
    observation = gym_env.get_latest_raw_observation()
    assert observation is not None
    vehicle = next(
        vehicle
        for vehicle in observation.active_vehicles
        if vehicle.task_state.required_packet_ids
    )
    vehicle_id = vehicle.snapshot.vehicle_id
    raw_env._vehicle_packet_updated_flags[vehicle_id] = tuple(
        True for _ in vehicle.knowledge.packet_updated_flags
    )

    _, reward, _, _, step_info = gym_env.step(
        np.full(8, 16, dtype=np.int64)
    )
    assert step_info["task_reward"] > 0.0
    assert step_info["task_reward_gain"] == 0.0
    assert reward == -gym_env._empty_action_penalty


def test_legacy_model_requires_explicit_compatibility_mode() -> None:
    legacy_path = ROOT / "outputs" / "ppo_reward_attribution_smoke" / "ppo_scheduler"
    if not legacy_path.with_suffix(".zip").exists():
        return

    try:
        PpoInferencePolicy(
            config_path=DEFAULT_CONFIG_PATH,
            model_path=legacy_path,
        )
    except ValueError as exc:
        assert "metadata" in str(exc).lower()
        return
    raise AssertionError(
        "A PPO model without metadata should require explicit compatibility mode."
    )


def test_missing_model_is_rejected() -> None:
    missing_path = ROOT / "outputs" / "ppo" / "missing_model"
    try:
        PpoInferencePolicy(
            config_path=DEFAULT_CONFIG_PATH,
            model_path=missing_path,
        )
    except FileNotFoundError:
        return
    raise AssertionError("Missing PPO model path should raise FileNotFoundError.")


if __name__ == "__main__":
    main()
