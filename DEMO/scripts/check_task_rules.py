from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import build_environment
from skylines_demo.config import (
    ChannelConfig,
    TaskConfig,
    dump_config_payload,
    load_config,
    load_config_payload,
)
from skylines_demo.data import load_vehicle_replay
from skylines_demo.geometry import BuildingLineOfSight
from skylines_demo.policies import MaxTaskRewardGainPolicy
from skylines_demo.tasks import TaskManager
from skylines_demo.types import (
    BuildingSnapshot,
    SchedulingAction,
    Vector3,
    VehicleKnowledge,
    VehicleState,
    VehicleTaskState,
)
from skylines_demo.wireless import LogDistanceChannelModel


def main() -> None:
    test_default_task_templates_match_design()
    test_default_seed_comes_from_scenario_config()
    test_initial_tasks_are_seed_reproducible()
    test_default_initial_caoi_starts_at_zero()
    test_initial_caoi_is_anchored_to_first_frame()
    test_replay_step_deltas_are_forward_aligned()
    test_unseeded_task_orders_change_between_resets()
    test_disabled_tasks_do_not_advertise_requests()
    test_autonomy_assignment_and_priority_are_seeded()
    test_no_partial_packets_are_exposed()
    test_unreachable_data_links_are_excluded()
    test_allocated_bandwidth_matches_resource_blocks()
    test_multi_v2i_obeys_resource_limits()
    test_full_episode_obeys_resource_and_control_rules()
    test_task_policy_obeys_configured_link_limits()
    test_duplicate_vehicle_is_invalid()
    test_misclassified_link_action_is_invalid()
    test_v2v_requires_real_or_fresher_packets()
    test_relay_spread_counts_equal_timestamp_missing_packets()
    test_task_completion_limit_and_no_repeats()
    test_task_cooldown()
    test_task_reward_formula_and_incremental_gain()
    test_three_dimensional_distance()
    test_height_aware_building_los()
    print("task-aware environment rule checks passed")


def test_default_seed_comes_from_scenario_config() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    default_observation = build_environment(DEFAULT_CONFIG_PATH).reset()
    explicit_observation = build_environment(
        DEFAULT_CONFIG_PATH,
        seed=config.seed,
    ).reset()
    assert _active_tasks(default_observation) == _active_tasks(explicit_observation)
    assert _vehicle_roles(default_observation) == _vehicle_roles(
        explicit_observation
    )


def test_default_task_templates_match_design() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    templates = config.tasks.templates
    assert 6 <= len(templates) <= 8
    assert all(7 <= len(packet_ids) <= 8 for _, packet_ids in templates)
    assert all(len(packet_ids) == len(set(packet_ids)) for _, packet_ids in templates)
    assert {
        packet_id
        for _, packet_ids in templates
        for packet_id in packet_ids
    } == set(range(1, config.environment.packet_count + 1))
    demand_counts: dict[int, int] = defaultdict(int)
    for _, packet_ids in templates:
        for packet_id in packet_ids:
            demand_counts[packet_id] += 1
    assert max(demand_counts.values()) > 1


def test_initial_tasks_are_seed_reproducible() -> None:
    obs_a = build_environment(DEFAULT_CONFIG_PATH, seed=17).reset()
    obs_b = build_environment(DEFAULT_CONFIG_PATH, seed=17).reset()
    obs_c = build_environment(DEFAULT_CONFIG_PATH, seed=18).reset()
    tasks_a = _active_tasks(obs_a)
    tasks_b = _active_tasks(obs_b)
    tasks_c = _active_tasks(obs_c)
    assert tasks_a == tasks_b
    assert tasks_a != tasks_c


def test_default_initial_caoi_starts_at_zero() -> None:
    observation = build_environment(DEFAULT_CONFIG_PATH, seed=7).reset()
    assert observation.active_vehicles
    assert all(
        abs(vehicle.knowledge.caoi_seconds) < 1e-9
        for vehicle in observation.active_vehicles
    )
    assert all(
        not any(vehicle.knowledge.packet_updated_flags)
        for vehicle in observation.active_vehicles
    )
    assert observation.candidate_v2i_links


def test_initial_caoi_is_anchored_to_first_frame() -> None:
    config_path = _build_temp_config(
        environment_overrides={
            "initial_caoi_seconds": 1.25,
            "initial_caoi_min_seconds": None,
            "initial_caoi_max_seconds": None,
        }
    )
    observation = build_environment(config_path, seed=7).reset()
    assert observation.active_vehicles
    assert all(
        abs(vehicle.knowledge.caoi_seconds - 1.25) < 1e-9
        for vehicle in observation.active_vehicles
    )


def test_replay_step_deltas_are_forward_aligned() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    replay = load_vehicle_replay(
        config.trajectory_csv,
        fallback_step_seconds=config.environment.fallback_step_seconds,
    )
    for current_frame, next_frame in zip(
        replay.frames[:-1],
        replay.frames[1:],
        strict=True,
    ):
        assert abs(
            current_frame.elapsed_seconds
            + current_frame.delta_seconds
            - next_frame.elapsed_seconds
        ) < 1e-9
    assert (
        abs(
            replay.frames[-1].delta_seconds
            - config.environment.fallback_step_seconds
        )
        < 1e-9
    )


def test_unseeded_task_orders_change_between_resets() -> None:
    task_config = load_config(DEFAULT_CONFIG_PATH).tasks
    vehicle_ids = tuple(range(1, 129))
    manager = TaskManager(task_config, vehicle_ids, seed=None)
    manager.reset()
    manager.begin_step(0, vehicle_ids)
    first_orders = {
        vehicle_id: manager.state_for(vehicle_id, 0).active_task_id
        for vehicle_id in vehicle_ids
    }
    manager.reset()
    manager.begin_step(0, vehicle_ids)
    second_orders = {
        vehicle_id: manager.state_for(vehicle_id, 0).active_task_id
        for vehicle_id in vehicle_ids
    }
    assert first_orders != second_orders


def test_disabled_tasks_do_not_advertise_requests() -> None:
    manager = TaskManager(
        TaskConfig(enabled=False),
        (1,),
        seed=7,
    )
    manager.reset()
    manager.begin_step(0, (1,))
    state = manager.state_for(1, 0)
    assert state.active_task_id is None
    assert not state.can_request_new_task


def test_autonomy_assignment_and_priority_are_seeded() -> None:
    config_path = _build_temp_config(
        autonomy_overrides={
            "enabled": True,
            "penetration_rate": 0.3,
            "priority_weight": 2.0,
        }
    )
    observation_a = build_environment(config_path, seed=17).reset()
    observation_b = build_environment(config_path, seed=17).reset()
    observation_c = build_environment(config_path, seed=18).reset()
    roles_a = _vehicle_roles(observation_a)
    roles_b = _vehicle_roles(observation_b)
    roles_c = _vehicle_roles(observation_c)
    assert roles_a == roles_b
    assert roles_a != roles_c

    autonomous_candidates = [
        candidate
        for candidate in observation_a.candidate_v2i_links
        if candidate.receiver_is_autonomous
    ]
    human_candidates = [
        candidate
        for candidate in observation_a.candidate_v2i_links
        if not candidate.receiver_is_autonomous
    ]
    assert autonomous_candidates
    assert human_candidates
    autonomous_candidate = autonomous_candidates[0]
    human_candidate = human_candidates[0]
    assert autonomous_candidate.receiver_priority_weight > 1.0
    assert abs(
        autonomous_candidate.priority_gain_seconds
        - autonomous_candidate.freshness_gain_seconds
        * autonomous_candidate.receiver_priority_weight
    ) < 1e-9
    assert human_candidate.receiver_priority_weight == 1.0
    assert abs(
        human_candidate.priority_gain_seconds
        - human_candidate.freshness_gain_seconds
    ) < 1e-9


def test_no_partial_packets_are_exposed() -> None:
    config_path = _build_temp_config(
        environment_overrides={"packet_size_bits": 1_000_000_000_000.0}
    )
    env = build_environment(config_path, seed=7)
    observation = env.reset()
    assert not observation.candidate_links
    result = env.step(None)
    assert result.delivered_packet_count == 0
    assert all(
        not any(flags)
        for flags in env._vehicle_packet_updated_flags.values()
    )


def test_unreachable_data_links_are_excluded() -> None:
    config_path = _build_temp_config(
        environment_overrides={"packet_size_bits": 1.0},
        channel_overrides={"min_rate_mbps": 1000.0},
    )
    observation = build_environment(config_path, seed=7).reset()
    assert not observation.candidate_links


def test_allocated_bandwidth_matches_resource_blocks() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    observation = build_environment(DEFAULT_CONFIG_PATH, seed=7).reset()
    assert observation.candidate_v2i_links
    for candidate in observation.candidate_links:
        block_hz = (
            config.resource.rsu_resource_block_hz
            if candidate.action.link_type == "v2i"
            else config.resource.v2v_resource_block_hz
        )
        expected_bandwidth_hz = candidate.resource_block_count * block_hz
        assert abs(
            candidate.allocated_bandwidth_hz - expected_bandwidth_hz
        ) < 1e-9
        assert abs(
            candidate.link_budget.allocated_bandwidth_hz
            - expected_bandwidth_hz
        ) < 1e-9
        assert candidate.link_budget.rate_mbps <= (
            candidate.link_budget.raw_rate_mbps + 1e-9
        )
        assert candidate.link_budget.rate_mbps <= (
            config.resource.vehicle_link_rate_limit_mbps + 1e-9
        )


def test_multi_v2i_obeys_resource_limits() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    policy = MaxTaskRewardGainPolicy(config.resource)
    observation = env.reset()
    action = policy.select_action(observation)
    assert len(action.v2i_links) > 1
    result = env.step(action)
    assert not result.invalid_action
    assert result.used_rsu_resource_blocks <= config.resource.rsu_resource_block_count
    assert (
        sum(candidate.link_budget.rate_mbps for candidate in result.chosen_v2i_links)
        <= config.resource.rsu_total_rate_limit_mbps + 1e-9
    )
    assert all(
        candidate.delivered_packet_count.is_integer()
        for candidate in result.successful_links
    )


def test_full_episode_obeys_resource_and_control_rules() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    policy = MaxTaskRewardGainPolicy(config.resource)
    observation = env.reset()
    max_v2i_delivered_packets = 0
    step_count = 0

    while observation is not None:
        assert all(
            candidate.link_budget.reachable
            for candidate in observation.candidate_links
        )
        result = env.step(policy.select_action(observation))
        step_count += 1
        assert not result.invalid_action
        assert result.control_broadcast.reliable
        assert set(result.control_broadcast.receiver_vehicle_ids) == {
            vehicle.snapshot.vehicle_id
            for vehicle in result.observation.active_vehicles
        }
        assert result.used_rsu_resource_blocks <= (
            config.resource.rsu_resource_block_count
        )
        assert result.used_v2v_resource_blocks <= (
            config.resource.v2v_resource_block_count
        )
        assert sum(
            candidate.link_budget.rate_mbps
            for candidate in result.chosen_v2i_links
        ) <= config.resource.rsu_total_rate_limit_mbps + 1e-9
        assert sum(
            candidate.link_budget.rate_mbps
            for candidate in result.chosen_v2v_links
        ) <= config.resource.v2v_total_rate_limit_mbps + 1e-9
        assert all(
            candidate.link_budget.reachable
            for candidate in result.chosen_links
        )
        for candidate in result.chosen_links:
            block_hz = (
                config.resource.rsu_resource_block_hz
                if candidate.action.link_type == "v2i"
                else config.resource.v2v_resource_block_hz
            )
            assert abs(
                candidate.allocated_bandwidth_hz
                - candidate.resource_block_count * block_hz
            ) < 1e-9

        vehicle_nodes: list[int] = []
        for candidate in result.chosen_links:
            vehicle_nodes.append(candidate.action.receiver_id)
            if candidate.action.link_type == "v2v":
                vehicle_nodes.append(
                    int(candidate.action.transmitter_id.split(":", 1)[1])
                )
        assert len(vehicle_nodes) == len(set(vehicle_nodes))

        max_v2i_delivered_packets = max(
            max_v2i_delivered_packets,
            sum(
                int(candidate.delivered_packet_count)
                for candidate in result.successful_v2i_links
            ),
        )
        observation = result.next_observation

    assert step_count > 0
    assert max_v2i_delivered_packets >= 10

    empty_env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    empty_observation = empty_env.reset()
    empty_result = empty_env.step(None)
    assert empty_result.control_broadcast.reliable
    assert set(empty_result.control_broadcast.receiver_vehicle_ids) == {
        vehicle.snapshot.vehicle_id
        for vehicle in empty_observation.active_vehicles
    }
    assert empty_result.used_rsu_resource_blocks == 0
    assert empty_result.used_v2v_resource_blocks == 0
    assert empty_result.delivered_packet_count == 0


def test_task_policy_obeys_configured_link_limits() -> None:
    config_path = _build_temp_config(
        environment_overrides={
            "max_active_links": 2,
            "allow_multi_v2i_from_rsu": False,
        }
    )
    config = load_config(config_path)
    env = build_environment(config_path, seed=7)
    observation = env.reset()
    policy = MaxTaskRewardGainPolicy(
        config.resource,
        max_links=config.environment.max_active_links,
        allow_multi_v2i_from_rsu=config.environment.allow_multi_v2i_from_rsu,
    )
    action = policy.select_action(observation)
    assert len(action.links) <= 2
    assert len(action.v2i_links) <= 1
    result = env.step(action)
    assert not result.invalid_action


def test_duplicate_vehicle_is_invalid() -> None:
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    observation = env.reset()
    candidate = observation.candidate_v2i_links[0]
    action = SchedulingAction(
        v2i_links=(candidate.action, candidate.action),
    )
    result = env.step(action)
    assert result.invalid_action
    assert not result.successful_links
    assert result.used_rsu_resource_blocks == 0
    assert result.used_v2v_resource_blocks == 0


def test_misclassified_link_action_is_invalid() -> None:
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    observation = env.reset()
    v2i_action = observation.candidate_v2i_links[0].action
    result = env.step(SchedulingAction(v2v_links=(v2i_action,)))
    assert result.invalid_action
    assert not result.successful_links
    assert result.used_rsu_resource_blocks == 0
    assert result.used_v2v_resource_blocks == 0


def test_v2v_requires_real_or_fresher_packets() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    policy = MaxTaskRewardGainPolicy(config.resource)
    observation = env.reset()
    result = env.step(policy.select_action(observation))
    next_observation = result.next_observation
    assert next_observation is not None
    assert next_observation.candidate_v2v_links
    for candidate in next_observation.candidate_v2v_links:
        sender_flags = candidate.transmitter_packet_updated_flags
        sender_timestamps = candidate.transmitter_packet_timestamps
        assert sender_flags is not None
        assert sender_timestamps is not None
        for packet_id in candidate.selected_packet_ids:
            index = packet_id - 1
            assert sender_flags[index]
            assert (
                not candidate.receiver_packet_updated_flags[index]
                or
                sender_timestamps[index]
                > candidate.receiver_packet_timestamps[index]
            )


def test_relay_spread_counts_equal_timestamp_missing_packets() -> None:
    observation = build_environment(DEFAULT_CONFIG_PATH, seed=7).reset()
    assert observation.candidate_v2i_links
    assert any(
        candidate.relay_peer_count > 0
        for candidate in observation.candidate_v2i_links
    )
    assert all(
        abs(candidate.relay_peer_total_gain_seconds) < 1e-9
        for candidate in observation.candidate_v2i_links
    )


def test_task_completion_limit_and_no_repeats() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    policy = MaxTaskRewardGainPolicy(config.resource)
    observation = env.reset()
    completed: dict[int, list[str]] = defaultdict(list)
    total_reward = 0.0
    while observation is not None:
        result = env.step(policy.select_action(observation))
        total_reward += result.task_reward
        for completion in result.completed_tasks:
            completed[completion.vehicle_id].append(completion.task_id)
        observation = result.next_observation
    assert total_reward > 0.0
    assert completed
    for task_ids in completed.values():
        assert len(task_ids) <= config.tasks.max_completed_tasks_per_vehicle
        assert len(task_ids) == len(set(task_ids))


def test_task_cooldown() -> None:
    task_config = TaskConfig(
        enabled=True,
        max_completed_tasks_per_vehicle=3,
        cooldown_steps=1,
        reward_base=10.0,
        freshness_decay_seconds=4.0,
        templates=(
            ("A", (1,)),
            ("B", (2,)),
            ("C", (3,)),
        ),
    )
    manager = TaskManager(task_config, (1,), seed=7)
    manager.reset()
    manager.begin_step(0, (1,))
    first_task = manager.state_for(1, 0).active_task_id
    assert first_task is not None
    manager.complete_ready_tasks(
        step_index=0,
        active_vehicle_ids=(1,),
        packet_timestamps_by_vehicle={1: (0.0, 0.0, 0.0)},
        packet_flags_by_vehicle={1: (True, True, True)},
        completion_time=0.3,
    )
    manager.begin_step(1, (1,))
    assert manager.state_for(1, 1).active_task_id is None
    manager.begin_step(2, (1,))
    second_task = manager.state_for(1, 2).active_task_id
    assert second_task is not None
    assert second_task != first_task


def test_task_reward_formula_and_incremental_gain() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    completion_time = 10.0
    packet_timestamps = tuple(
        9.0 if index == 0 else 7.0 if index == 1 else 0.0
        for index in range(config.environment.packet_count)
    )
    packet_flags = tuple(
        index in (0, 1)
        for index in range(config.environment.packet_count)
    )
    manager = TaskManager(
        TaskConfig(
            enabled=True,
            max_completed_tasks_per_vehicle=1,
            cooldown_steps=1,
            reward_base=10.0,
            freshness_decay_seconds=4.0,
            templates=(("A", (1, 2)),),
        ),
        (1,),
        seed=7,
    )
    manager.reset()
    manager.begin_step(0, (1,))
    completions = manager.complete_ready_tasks(
        step_index=0,
        active_vehicle_ids=(1,),
        packet_timestamps_by_vehicle={1: packet_timestamps},
        packet_flags_by_vehicle={1: packet_flags},
        completion_time=completion_time,
    )
    assert len(completions) == 1
    expected_reward = 10.0 * (
        math.exp(-1.0 / 4.0) + math.exp(-3.0 / 4.0)
    ) / 2.0
    assert abs(completions[0].mean_packet_aoi_seconds - 2.0) < 1e-9
    assert abs(completions[0].reward - expected_reward) < 1e-9

    env = build_environment(DEFAULT_CONFIG_PATH, seed=7)
    observation = env.reset()
    base_timestamps = tuple(
        0.0 for _ in range(config.environment.packet_count)
    )
    all_flags = tuple(
        True for _ in range(config.environment.packet_count)
    )
    post_timestamps = list(base_timestamps)
    post_timestamps[0] = 8.0
    receiver = VehicleState(
        snapshot=observation.active_vehicles[0].snapshot,
        knowledge=VehicleKnowledge(
            vehicle_id=observation.active_vehicles[0].snapshot.vehicle_id,
            packet_timestamps=base_timestamps,
            packet_updated_flags=all_flags,
            caoi_seconds=completion_time,
        ),
        task_state=VehicleTaskState(
            active_task_id="A",
            required_packet_ids=(1, 2),
        ),
    )
    estimated_gain = env._estimate_task_completion_reward_gain(
        receiver=receiver,
        post_timestamps=tuple(post_timestamps),
        post_flags=all_flags,
        completion_time=completion_time,
    )
    baseline_reward = config.tasks.reward_base * math.exp(
        -completion_time / config.tasks.freshness_decay_seconds
    )
    post_reward = config.tasks.reward_base * (
        math.exp(-2.0 / config.tasks.freshness_decay_seconds)
        + math.exp(-completion_time / config.tasks.freshness_decay_seconds)
    ) / 2.0
    assert abs(estimated_gain - (post_reward - baseline_reward)) < 1e-9


def test_three_dimensional_distance() -> None:
    channel = LogDistanceChannelModel(ChannelConfig(), _AlwaysLos())
    budget = channel.evaluate(
        Vector3(0.0, 0.0, 0.0),
        Vector3(0.0, 10.0, 0.0),
        bandwidth_hz=5_000_000.0,
    )
    assert abs(budget.distance_m - 10.0) < 1e-9


def test_height_aware_building_los() -> None:
    building = BuildingSnapshot(
        building_id=1,
        position=Vector3(5.0, 0.0, 0.0),
        angle_y=0.0,
        width_cells=1,
        length_cells=1,
        size_x=4.0,
        size_y=5.0,
        size_z=4.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
        center_offset_z=0.0,
        min_y=0.0,
        max_y=5.0,
        prefab_name="test",
    )
    los = BuildingLineOfSight([building], height_clearance_m=0.0)
    assert not los.has_line_of_sight(
        Vector3(0.0, 2.0, 0.0),
        Vector3(10.0, 2.0, 0.0),
    )
    assert los.has_line_of_sight(
        Vector3(0.0, 10.0, 0.0),
        Vector3(10.0, 10.0, 0.0),
    )


def _active_tasks(observation) -> dict[int, str | None]:
    return {
        vehicle.snapshot.vehicle_id: vehicle.task_state.active_task_id
        for vehicle in observation.active_vehicles
    }


def _vehicle_roles(observation) -> dict[int, bool]:
    return {
        vehicle.snapshot.vehicle_id: vehicle.is_autonomous
        for vehicle in observation.active_vehicles
    }


def _build_temp_config(
    environment_overrides: dict | None = None,
    channel_overrides: dict | None = None,
    autonomy_overrides: dict | None = None,
) -> Path:
    payload = load_config_payload(DEFAULT_CONFIG_PATH)
    if environment_overrides:
        environment = dict(payload.get("environment", {}))
        environment.update(environment_overrides)
        payload["environment"] = environment
    if channel_overrides:
        channel = dict(payload.get("channel", {}))
        channel.update(channel_overrides)
        payload["channel"] = channel
    if autonomy_overrides:
        autonomy = dict(payload.get("autonomy", {}))
        autonomy.update(autonomy_overrides)
        payload["autonomy"] = autonomy
    path = Path(tempfile.gettempdir()) / "skylines_task_rule_check.yaml"
    path.write_text(
        dump_config_payload(payload),
        encoding="utf-8",
    )
    return path


class _AlwaysLos:
    def has_line_of_sight(self, start: Vector3, end: Vector3) -> bool:
        del start, end
        return True


if __name__ == "__main__":
    main()
