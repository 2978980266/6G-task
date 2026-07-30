from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import build_environment
from skylines_demo.config import load_config
from skylines_demo.policies import (
    MaxTaskRewardGainPolicy,
    RandomTaskAwarePolicy,
    TwoStageTaskRelayGreedyPolicy,
)
from skylines_demo.types import CandidateLink, StepObservation, VehicleState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace every step of a task-aware V2I/V2V episode, including "
            "task state, selected links, packet delivery, and post-action CAoI."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the scenario config YAML.",
    )
    parser.add_argument(
        "--policy",
        choices=(
            "random-task-aware",
            "max-task-reward-gain",
            "two-stage-task-relay-greedy",
            "random-set",
            "max-caoi-gain",
            "max-aoi-gain",
            "two-stage-relay-greedy",
            "ppo",
        ),
        default="max-task-reward-gain",
        help="Policy to replay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the scenario config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum number of steps to trace. Defaults to the entire episode.",
    )
    parser.add_argument(
        "--show-all-vehicles",
        action="store_true",
        help="Print every active vehicle's task and packet state before each step.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional output path for one structured JSON record per step.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained PPO model when --policy ppo is used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional PPO top-k override. Must match the trained model.",
    )
    parser.add_argument(
        "--action-slots",
        type=int,
        default=None,
        help="Optional PPO action-slot override. Must match the trained model.",
    )
    parser.add_argument(
        "--allow-legacy-model",
        action="store_true",
        help=(
            "Allow loading a PPO model without a metadata sidecar. "
            "Use only for explicitly identified legacy models."
        ),
    )
    return parser.parse_args()


def build_policy(
    policy_name: str,
    seed: int,
    config: Any,
    config_path: str,
    model_path: str | None,
    top_k: int | None,
    action_slots: int | None,
    allow_legacy_model: bool,
) -> Any:
    if policy_name == "ppo":
        if not model_path:
            raise ValueError("PPO policy requires --model-path.")
        from skylines_demo.rl.policy import PpoInferencePolicy

        return PpoInferencePolicy(
            config_path=config_path,
            model_path=model_path,
            top_k=top_k,
            action_slots=action_slots,
            allow_legacy_model=allow_legacy_model,
        )
    if policy_name in ("max-task-reward-gain", "max-caoi-gain", "max-aoi-gain"):
        return MaxTaskRewardGainPolicy(
            config.resource,
            max_links=config.environment.max_active_links,
            allow_multi_v2i_from_rsu=config.environment.allow_multi_v2i_from_rsu,
        )
    if policy_name in ("two-stage-task-relay-greedy", "two-stage-relay-greedy"):
        return TwoStageTaskRelayGreedyPolicy(
            config.resource,
            max_links=config.environment.max_active_links,
            allow_multi_v2i_from_rsu=config.environment.allow_multi_v2i_from_rsu,
        )
    if policy_name in ("random-task-aware", "random-set"):
        return RandomTaskAwarePolicy(
            resource_config=config.resource,
            seed=seed,
            max_links=config.environment.max_active_links,
            allow_multi_v2i_from_rsu=config.environment.allow_multi_v2i_from_rsu,
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def packet_ids_held_by(vehicle: VehicleState) -> tuple[int, ...]:
    return tuple(
        index + 1
        for index, updated in enumerate(vehicle.knowledge.packet_updated_flags)
        if updated
    )


def missing_required_packet_ids(vehicle: VehicleState) -> tuple[int, ...]:
    return tuple(
        packet_id
        for packet_id in vehicle.task_state.required_packet_ids
        if not vehicle.knowledge.packet_updated_flags[packet_id - 1]
    )


def candidate_record(candidate: CandidateLink) -> dict[str, Any]:
    action = candidate.action
    budget = candidate.link_budget
    return {
        "type": action.link_type,
        "transmitter": action.transmitter_id,
        "receiver": action.receiver_id,
        "packets": list(action.packet_ids),
        "resource_blocks": candidate.resource_block_count,
        "allocated_bandwidth_mhz": round(
            candidate.allocated_bandwidth_hz / 1_000_000.0,
            3,
        ),
        "distance_m": round(budget.distance_m, 3),
        "los": budget.los,
        "vehicle_blocked": budget.vehicle_blocked,
        "vehicle_blocker_count": budget.vehicle_blocker_count,
        "path_loss_db": round(budget.path_loss_db, 3),
        "snr_db": round(budget.snr_db, 3),
        "raw_rate_mbps": round(budget.raw_rate_mbps, 3),
        "effective_rate_mbps": round(budget.rate_mbps, 3),
        "complete_packet_capacity": candidate.complete_packet_capacity,
        "delivered_packet_count": int(candidate.delivered_packet_count),
        "freshness_gain_seconds": round(candidate.freshness_gain_seconds, 3),
        "task_progress_gain": candidate.task_progress_gain,
        "task_reward_gain": round(candidate.task_reward_gain, 3),
        "relay_peer_count": candidate.relay_peer_count,
        "relay_peer_total_gain_seconds": round(
            candidate.relay_peer_total_gain_seconds,
            3,
        ),
    }


def vehicle_record(vehicle: VehicleState) -> dict[str, Any]:
    task = vehicle.task_state
    return {
        "vehicle_id": vehicle.snapshot.vehicle_id,
        "role": "AV" if vehicle.is_autonomous else "HUM",
        "position": {
            "x": round(vehicle.snapshot.position.x, 3),
            "y": round(vehicle.snapshot.position.y, 3),
            "z": round(vehicle.snapshot.position.z, 3),
        },
        "speed": round(vehicle.snapshot.speed, 3),
        "caoi_seconds": round(vehicle.knowledge.caoi_seconds, 3),
        "held_packets": list(packet_ids_held_by(vehicle)),
        "active_task_id": task.active_task_id,
        "required_packets": list(task.required_packet_ids),
        "missing_required_packets": list(missing_required_packet_ids(vehicle)),
        "completed_tasks": list(task.completed_task_ids),
        "cooldown_until_step": task.cooldown_until_step,
        "can_request_new_task": task.can_request_new_task,
    }


def print_vehicle_state(vehicle: VehicleState) -> None:
    task = vehicle.task_state
    print(
        f"  车辆 {vehicle.snapshot.vehicle_id} "
        f"[{'AV' if vehicle.is_autonomous else 'HUM'}] "
        f"CAoI={vehicle.knowledge.caoi_seconds:.3f}s "
        f"持包={list(packet_ids_held_by(vehicle))} "
        f"任务={task.active_task_id or '-'} "
        f"需求={list(task.required_packet_ids)} "
        f"缺失={list(missing_required_packet_ids(vehicle))} "
        f"已完成={list(task.completed_task_ids)} "
        f"冷却至={task.cooldown_until_step}"
    )


def print_link(prefix: str, candidate: CandidateLink) -> None:
    record = candidate_record(candidate)
    blockage = (
        f"车辆遮挡={record['vehicle_blocker_count']}"
        if record["vehicle_blocked"]
        else "车辆遮挡=否"
    )
    print(
        f"  {prefix} {record['type'].upper()} "
        f"{record['transmitter']} -> 车辆 {record['receiver']} "
        f"包={record['packets']} RB={record['resource_blocks']} "
        f"带宽={record['allocated_bandwidth_mhz']:.3f}MHz "
        f"距离={record['distance_m']:.3f}m LoS={record['los']} {blockage} "
        f"路径损耗={record['path_loss_db']:.3f}dB "
        f"SNR={record['snr_db']:.3f}dB "
        f"原始速率={record['raw_rate_mbps']:.3f}Mbps "
        f"有效速率={record['effective_rate_mbps']:.3f}Mbps "
        f"容量={record['complete_packet_capacity']}包 "
        f"CAoI收益={record['freshness_gain_seconds']:.3f}s "
        f"任务进度+={record['task_progress_gain']} "
        f"任务奖励增量={record['task_reward_gain']:.3f} "
        f"中继潜力={record['relay_peer_count']}"
    )


def selected_candidates(
    observation: StepObservation,
    action_links: tuple[Any, ...],
) -> tuple[CandidateLink, ...]:
    action_set = set(action_links)
    return tuple(
        candidate
        for candidate in observation.candidate_links
        if candidate.action in action_set
    )


def trace_step(
    observation: StepObservation,
    result: Any,
    action_links: tuple[Any, ...],
    show_all_vehicles: bool,
) -> dict[str, Any]:
    selected = selected_candidates(observation, action_links)
    relevant_vehicle_ids = {
        candidate.action.receiver_id for candidate in selected
    }
    relevant_vehicle_ids.update(
        int(candidate.action.transmitter_id.split(":", 1)[1])
        for candidate in selected
        if candidate.action.link_type == "v2v"
    )

    print()
    print(
        f"{'=' * 24} STEP {observation.step_index} {'=' * 24}\n"
        f"仿真帧={observation.simulation_frame} "
        f"时间={observation.elapsed_seconds:.3f}s "
        f"步长={observation.delta_seconds:.3f}s "
        f"在场车辆={len(observation.active_vehicles)} "
        f"候选链路={len(observation.candidate_links)} "
        f"(V2I={observation.v2i_candidate_count}, "
        f"V2V={observation.v2v_candidate_count})"
    )
    print("动作前任务与持包状态：")
    vehicles_to_print = (
        observation.active_vehicles
        if show_all_vehicles
        else tuple(
            vehicle
            for vehicle in observation.active_vehicles
            if vehicle.snapshot.vehicle_id in relevant_vehicle_ids
        )
    )
    if vehicles_to_print:
        for vehicle in vehicles_to_print:
            print_vehicle_state(vehicle)
    else:
        print("  本 step 没有被策略选中的数据链路。")

    print("策略选择：")
    if selected:
        for candidate in selected:
            print_link("选择", candidate)
    else:
        print("  空数据动作。")

    print("执行结果：")
    if result.invalid_action:
        print("  动作非法：环境没有更新任何 packet，也没有占用数据资源。")
    elif result.successful_links:
        for candidate in result.successful_links:
            print_link("成功", candidate)
    else:
        print("  没有成功的数据链路。")
    for candidate in result.failed_links:
        print_link("失败", candidate)

    print(
        f"  控制广播=可靠，接收车辆数={len(result.control_broadcast.receiver_vehicle_ids)} "
        f"RSU资源块={result.used_rsu_resource_blocks}/"
        f"{observation.rsu_resource_block_count} "
        f"V2V资源块={result.used_v2v_resource_blocks}/"
        f"{observation.v2v_resource_block_count} "
        f"完整交付={result.delivered_packet_count}包"
    )
    print(
        f"  动作后 CAoI：均值={result.mean_caoi_seconds:.3f}s "
        f"最大值={result.max_caoi_seconds:.3f}s "
        f"超阈值车辆={result.age_violation_count}"
    )
    if result.completed_tasks:
        print("任务完成：")
        for completion in result.completed_tasks:
            print(
                f"  车辆 {completion.vehicle_id} 完成 {completion.task_id} "
                f"需求包={list(completion.required_packet_ids)} "
                f"完成时平均AoI={completion.mean_packet_aoi_seconds:.3f}s "
                f"奖励={completion.reward:.3f}"
            )
    else:
        print(f"任务完成：无，本 step 任务奖励={result.task_reward:.3f}")

    next_vehicles = (
        []
        if result.next_observation is None
        else [
            vehicle_record(vehicle)
            for vehicle in result.next_observation.active_vehicles
            if show_all_vehicles
            or vehicle.snapshot.vehicle_id in relevant_vehicle_ids
            or vehicle.snapshot.vehicle_id
            in {completion.vehicle_id for completion in result.completed_tasks}
        ]
    )
    if next_vehicles:
        print("下一观测中的相关车辆状态：")
        for vehicle in next_vehicles:
            print(
                f"  车辆 {vehicle['vehicle_id']} CAoI={vehicle['caoi_seconds']:.3f}s "
                f"持包={vehicle['held_packets']} "
                f"任务={vehicle['active_task_id'] or '-'} "
                f"缺失={vehicle['missing_required_packets']} "
                f"已完成={vehicle['completed_tasks']}"
            )

    return {
        "step_index": observation.step_index,
        "simulation_frame": observation.simulation_frame,
        "elapsed_seconds": observation.elapsed_seconds,
        "delta_seconds": observation.delta_seconds,
        "candidate_counts": {
            "total": len(observation.candidate_links),
            "v2i": observation.v2i_candidate_count,
            "v2v": observation.v2v_candidate_count,
        },
        "selected_links": [candidate_record(candidate) for candidate in selected],
        "successful_links": [
            candidate_record(candidate) for candidate in result.successful_links
        ],
        "failed_links": [
            candidate_record(candidate) for candidate in result.failed_links
        ],
        "invalid_action": result.invalid_action,
        "control_broadcast": {
            "reliable": result.control_broadcast.reliable,
            "receiver_vehicle_ids": list(
                result.control_broadcast.receiver_vehicle_ids
            ),
        },
        "resource_usage": {
            "rsu_blocks_used": result.used_rsu_resource_blocks,
            "rsu_blocks_total": observation.rsu_resource_block_count,
            "v2v_blocks_used": result.used_v2v_resource_blocks,
            "v2v_blocks_total": observation.v2v_resource_block_count,
        },
        "post_action": {
            "delivered_packet_count": result.delivered_packet_count,
            "mean_caoi_seconds": result.mean_caoi_seconds,
            "max_caoi_seconds": result.max_caoi_seconds,
            "age_violation_count": result.age_violation_count,
            "task_reward": result.task_reward,
            "completed_tasks": [
                {
                    "vehicle_id": completion.vehicle_id,
                    "task_id": completion.task_id,
                    "required_packet_ids": list(completion.required_packet_ids),
                    "mean_packet_aoi_seconds": completion.mean_packet_aoi_seconds,
                    "reward": completion.reward,
                }
                for completion in result.completed_tasks
            ],
        },
        "pre_action_vehicle_state": [
            vehicle_record(vehicle) for vehicle in observation.active_vehicles
        ],
        "next_observation_vehicle_state": next_vehicles,
    }


def main() -> None:
    args = parse_args()
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive when provided.")

    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    env = build_environment(args.config, seed=seed)
    policy = build_policy(
        policy_name=args.policy,
        seed=seed,
        config=config,
        config_path=args.config,
        model_path=args.model_path,
        top_k=args.top_k,
        action_slots=args.action_slots,
        allow_legacy_model=args.allow_legacy_model,
    )
    policy.reset()
    observation = env.reset()
    trace_path = Path(args.jsonl) if args.jsonl else None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"开始追踪：policy={args.policy} seed={seed} "
        f"最大步数={args.steps if args.steps is not None else '整个episode'}"
    )
    trace_file = (
        trace_path.open("w", encoding="utf-8")
        if trace_path is not None
        else None
    )
    try:
        shown_steps = 0
        while observation is not None:
            if args.steps is not None and shown_steps >= args.steps:
                break
            action = policy.select_action(observation)
            result = env.step(action)
            record = trace_step(
                observation=observation,
                result=result,
                action_links=action.links,
                show_all_vehicles=args.show_all_vehicles,
            )
            if trace_file is not None:
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            observation = result.next_observation
            shown_steps += 1
    finally:
        if trace_file is not None:
            trace_file.close()

    if trace_path is not None:
        print(f"\n结构化追踪已写入：{trace_path}")


if __name__ == "__main__":
    main()
