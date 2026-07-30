from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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

# Unified experiment entry.
# Change these values directly, then run this file.
# random-task-aware
# max-task-reward-gain
# two-stage-task-relay-greedy
# Old names remain aliases for compatibility.
CONFIG_PATH = str(DEFAULT_CONFIG_PATH)
POLICY = "max-task-reward-gain"
SUMMARY_JSON: str | None = None
MODEL_PATH: str | None = None
TOP_K: int | None = None
ACTION_SLOTS: int | None = None


def build_policy(
    policy_name: str,
    seed: int,
    config,
    config_path: str,
    model_path: str | None = None,
    top_k: int | None = None,
    action_slots: int | None = None,
    allow_legacy_model: bool = False,
):
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


def run_experiment(
    config_path: str,
    policy_name: str = "random-set",
    seed: int | None = None,
    summary_json: str | None = None,
    model_path: str | None = None,
    top_k: int | None = None,
    action_slots: int | None = None,
    allow_legacy_model: bool = False,
) -> dict[str, float | int | bool | str]:
    config = load_config(config_path)
    effective_seed = config.seed if seed is None else seed
    env = build_environment(config_path, seed=effective_seed)
    allow_multi_v2i = config.environment.allow_multi_v2i_from_rsu
    policy = build_policy(
        policy_name=policy_name,
        seed=effective_seed,
        config=config,
        config_path=config_path,
        model_path=model_path,
        top_k=top_k,
        action_slots=action_slots,
        allow_legacy_model=allow_legacy_model,
    )

    observation = env.reset()
    policy.reset()

    steps = 0
    total_caoi = 0.0
    total_autonomous_caoi = 0.0
    total_human_caoi = 0.0
    total_priority_weighted_caoi = 0.0
    total_max_caoi = 0.0
    episode_max_caoi = 0.0
    empty_actions = 0
    invalid_action_count = 0
    control_broadcast_count = 0
    total_candidates = 0
    total_v2i_candidates = 0
    total_v2v_candidates = 0
    total_scheduled_links = 0
    total_scheduled_v2i_links = 0
    total_scheduled_v2v_links = 0
    attempted_count = 0
    v2i_attempted_count = 0
    v2v_attempted_count = 0
    success_count = 0
    v2i_success_count = 0
    v2v_success_count = 0
    failed_count = 0
    age_violation_count = 0
    autonomous_vehicle_ids: set[int] = set()
    human_vehicle_ids: set[int] = set()
    total_task_reward = 0.0
    completed_task_count = 0
    total_completed_task_aoi = 0.0
    completed_task_ids_by_vehicle: dict[int, list[str]] = {}
    total_delivered_packets = 0
    total_v2i_delivered_packets = 0
    total_v2v_delivered_packets = 0
    total_used_rsu_resource_blocks = 0
    total_used_v2v_resource_blocks = 0

    while observation is not None:
        for vehicle in observation.active_vehicles:
            if vehicle.is_autonomous:
                autonomous_vehicle_ids.add(vehicle.snapshot.vehicle_id)
            else:
                human_vehicle_ids.add(vehicle.snapshot.vehicle_id)
        action = policy.select_action(observation)
        result = env.step(action)
        steps += 1
        total_caoi += result.mean_caoi_seconds
        total_autonomous_caoi += result.mean_autonomous_caoi_seconds
        total_human_caoi += result.mean_human_caoi_seconds
        total_priority_weighted_caoi += result.mean_priority_weighted_caoi_seconds
        total_max_caoi += result.max_caoi_seconds
        episode_max_caoi = max(episode_max_caoi, result.max_caoi_seconds)
        total_candidates += len(result.observation.candidate_links)
        total_v2i_candidates += result.observation.v2i_candidate_count
        total_v2v_candidates += result.observation.v2v_candidate_count
        total_scheduled_links += len(result.chosen_links)
        total_scheduled_v2i_links += len(result.chosen_v2i_links)
        total_scheduled_v2v_links += len(result.chosen_v2v_links)
        attempted_count += len(result.chosen_links)
        success_count += len(result.successful_links)
        failed_count += len(result.failed_links)
        age_violation_count += result.age_violation_count
        total_task_reward += result.task_reward
        completed_task_count += len(result.completed_tasks)
        total_delivered_packets += result.delivered_packet_count
        total_v2i_delivered_packets += sum(
            int(candidate.delivered_packet_count)
            for candidate in result.successful_v2i_links
        )
        total_v2v_delivered_packets += sum(
            int(candidate.delivered_packet_count)
            for candidate in result.successful_v2v_links
        )
        total_used_rsu_resource_blocks += result.used_rsu_resource_blocks
        total_used_v2v_resource_blocks += result.used_v2v_resource_blocks
        for completion in result.completed_tasks:
            total_completed_task_aoi += completion.mean_packet_aoi_seconds
            completed_task_ids_by_vehicle.setdefault(
                completion.vehicle_id,
                [],
            ).append(completion.task_id)
        if result.control_broadcast.reliable:
            control_broadcast_count += 1

        if result.empty_action:
            empty_actions += 1
        if result.invalid_action:
            invalid_action_count += 1

        for candidate in result.chosen_links:
            if candidate.action.link_type == "v2i":
                v2i_attempted_count += 1
            else:
                v2v_attempted_count += 1
        for candidate in result.successful_links:
            if candidate.action.link_type == "v2i":
                v2i_success_count += 1
            else:
                v2v_success_count += 1

        observation = result.next_observation

    summary = {
        "policy": policy_name,
        "seed": effective_seed,
        "model_path": model_path or "",
        "allow_legacy_model": allow_legacy_model,
        "environment_mode": "task-aware",
        "task_aware": True,
        "allow_multi_v2i_from_rsu": allow_multi_v2i,
        "steps": steps,
        "control_broadcast_count": control_broadcast_count,
        "mean_step_caoi": total_caoi / max(steps, 1),
        "mean_autonomous_caoi": total_autonomous_caoi / max(steps, 1),
        "mean_human_caoi": total_human_caoi / max(steps, 1),
        "mean_priority_weighted_caoi": total_priority_weighted_caoi / max(steps, 1),
        "autonomous_vehicle_count": len(autonomous_vehicle_ids),
        "human_vehicle_count": len(human_vehicle_ids),
        "mean_step_max_caoi": total_max_caoi / max(steps, 1),
        "episode_max_caoi": episode_max_caoi,
        "attempted_count": attempted_count,
        "v2i_attempted_count": v2i_attempted_count,
        "v2v_attempted_count": v2v_attempted_count,
        "success_count": success_count,
        "v2i_success_count": v2i_success_count,
        "v2v_success_count": v2v_success_count,
        "failed_count": failed_count,
        "invalid_action_count": invalid_action_count,
        "empty_actions": empty_actions,
        "age_violation_count": age_violation_count,
        "mean_scheduled_links_per_step": total_scheduled_links / max(steps, 1),
        "mean_v2i_scheduled_per_step": total_scheduled_v2i_links / max(steps, 1),
        "mean_v2v_scheduled_per_step": total_scheduled_v2v_links / max(steps, 1),
        "mean_candidate_count": total_candidates / max(steps, 1),
        "mean_v2i_candidate_count": total_v2i_candidates / max(steps, 1),
        "mean_v2v_candidate_count": total_v2v_candidates / max(steps, 1),
        "total_task_reward": total_task_reward,
        "completed_task_count": completed_task_count,
        "mean_completed_task_packet_aoi": (
            total_completed_task_aoi / max(completed_task_count, 1)
        ),
        "vehicles_with_completed_tasks": len(completed_task_ids_by_vehicle),
        "total_delivered_packets": total_delivered_packets,
        "total_v2i_delivered_packets": total_v2i_delivered_packets,
        "total_v2v_delivered_packets": total_v2v_delivered_packets,
        "mean_delivered_packets_per_step": total_delivered_packets / max(steps, 1),
        "mean_v2i_delivered_packets_per_step": (
            total_v2i_delivered_packets / max(steps, 1)
        ),
        "mean_v2v_delivered_packets_per_step": (
            total_v2v_delivered_packets / max(steps, 1)
        ),
        "mean_used_rsu_resource_blocks": (
            total_used_rsu_resource_blocks / max(steps, 1)
        ),
        "mean_used_v2v_resource_blocks": (
            total_used_v2v_resource_blocks / max(steps, 1)
        ),
    }

    if summary_json:
        output_path = Path(summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a task-aware V2I/V2V experiment in the DEMO scene."
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to the scenario config YAML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the scenario config.",
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
        default=POLICY,
        help="Experiment policy to run.",
    )
    parser.add_argument("--summary-json", default=SUMMARY_JSON, help="Optional path to write a JSON summary.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to a trained PPO model when --policy ppo is used.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Optional PPO top-k override. Must match the trained model if provided.")
    parser.add_argument(
        "--action-slots",
        type=int,
        default=ACTION_SLOTS,
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
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Print the full flat metric set instead of the concise grouped summary.",
    )
    return parser.parse_args()


def build_compact_summary(
    summary: dict[str, float | int | bool | str],
) -> dict[str, object]:
    return {
        "policy": summary["policy"],
        "seed": summary["seed"],
        "environment_mode": summary["environment_mode"],
        "steps": summary["steps"],
        "caoi": {
            "mean_step": summary["mean_step_caoi"],
            "episode_max": summary["episode_max_caoi"],
            "age_violations": summary["age_violation_count"],
        },
        "tasks": {
            "completed": summary["completed_task_count"],
            "vehicles_completed": summary["vehicles_with_completed_tasks"],
            "total_reward": summary["total_task_reward"],
            "mean_completed_packet_aoi": summary[
                "mean_completed_task_packet_aoi"
            ],
        },
        "delivery_per_step": {
            "total": summary["mean_delivered_packets_per_step"],
            "v2i": summary["mean_v2i_delivered_packets_per_step"],
            "v2v": summary["mean_v2v_delivered_packets_per_step"],
        },
        "execution": {
            "attempted_links": summary["attempted_count"],
            "successful_links": summary["success_count"],
            "failed_links": summary["failed_count"],
            "invalid_actions": summary["invalid_action_count"],
            "empty_actions": summary["empty_actions"],
        },
    }


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        config_path=args.config,
        policy_name=args.policy,
        seed=args.seed,
        summary_json=args.summary_json,
        model_path=args.model_path,
        top_k=args.top_k,
        action_slots=args.action_slots,
        allow_legacy_model=args.allow_legacy_model,
    )
    output = summary if args.full_output else build_compact_summary(summary)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
