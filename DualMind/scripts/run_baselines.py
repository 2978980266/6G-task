from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dualmind_env.baselines.policies import default_policies
from dualmind_env.config import load_config
from dualmind_env.data.building_loader import load_buildings
from dualmind_env.data.trajectory_loader import load_vehicle_replay
from dualmind_env.env.aoi_env import AoiSchedulingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-1 AoI scheduling baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON scenario config.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline_summary.json",
        help="Where to save the baseline summary JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used by stochastic baselines.",
    )
    return parser.parse_args()


def run_policy(policy, env: AoiSchedulingEnv) -> dict[str, float | int | str]:
    observation = env.reset()
    policy.reset()

    total_reward = 0.0
    total_aoi = 0.0
    total_reachable_rate = 0.0
    total_successes = 0
    total_decisions = 0
    steps = 0

    while True:
        action = policy.select_action(observation)
        result = env.step(action)

        total_reward += result.reward
        total_aoi += float(result.info["mean_aoi"])
        total_reachable_rate += float(result.info["mean_reachable_rate_mbps"])
        total_successes += int(bool(result.info["success"]))
        total_decisions += 1
        steps += 1

        if result.done:
            break
        assert result.observation is not None
        observation = result.observation

    return {
        "policy": policy.name,
        "steps": steps,
        "total_reward": total_reward,
        "mean_step_aoi": total_aoi / max(steps, 1),
        "success_rate": total_successes / max(total_decisions, 1),
        "mean_reachable_rate_mbps": total_reachable_rate / max(steps, 1),
    }


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    headers = [
        ("policy", 24),
        ("steps", 8),
        ("total_reward", 16),
        ("mean_step_aoi", 16),
        ("success_rate", 14),
        ("mean_rate_mbps", 18),
    ]
    print(" ".join(name.ljust(width) for name, width in headers))
    print("-" * (sum(width for _, width in headers) + len(headers) - 1))
    for row in rows:
        print(
            f"{str(row['policy']).ljust(24)} "
            f"{str(row['steps']).ljust(8)} "
            f"{row['total_reward']:<16.4f} "
            f"{row['mean_step_aoi']:<16.4f} "
            f"{row['success_rate']:<14.4f} "
            f"{row['mean_reachable_rate_mbps']:<18.4f}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    replay = load_vehicle_replay(config.trajectory_csv)
    buildings = load_buildings(config.building_csv)

    results: list[dict[str, float | int | str]] = []
    for policy in default_policies(seed=args.seed):
        env = AoiSchedulingEnv(
            replay=replay,
            buildings=buildings,
            rsu_position=config.rsu.position,
            channel_config=config.channel,
            environment_config=config.environment,
        )
        results.append(run_policy(policy, env))

    print(f"scenario: {config.scenario_name}")
    print(f"trajectory frames: {len(replay.frames)}")
    print(f"unique vehicles: {len(replay.vehicle_ids)}")
    print(f"buildings: {len(buildings)}")
    print_table(results)

    output_path = args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved summary to: {output_path}")


if __name__ == "__main__":
    main()
