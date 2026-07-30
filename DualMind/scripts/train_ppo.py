from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from dualmind_env.env.gym_env import AoiSchedulingGymEnv
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "缺少 PPO 运行依赖。请先安装项目依赖后再运行 train_ppo.py，至少需要 gymnasium 和 stable-baselines3。"
    ) from exc

from dualmind_env.config import load_config
from dualmind_env.data.building_loader import load_buildings
from dualmind_env.data.trajectory_loader import load_vehicle_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO scheduler baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON scenario config.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for training and evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ppo",
        help="Directory used for the trained model and evaluation summary.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of deterministic evaluation episodes after training.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Verbosity level passed to stable-baselines3 PPO.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help="Optional tensorboard log directory. If omitted, no tensorboard files are written.",
    )
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=None,
        help="Optional JSON file for per-step deterministic evaluation traces.",
    )
    return parser.parse_args()


def build_env_factory(
    config_path: Path,
) -> tuple[Callable[[], AoiSchedulingGymEnv], str, int, int]:
    config = load_config(config_path)
    replay = load_vehicle_replay(config.trajectory_csv)
    buildings = load_buildings(config.building_csv)

    def make_env() -> AoiSchedulingGymEnv:
        return AoiSchedulingGymEnv(
            replay=replay,
            buildings=buildings,
            rsu_position=config.rsu.position,
            channel_config=config.channel,
            environment_config=config.environment,
        )

    return make_env, config.scenario_name, len(replay.frames), len(replay.vehicle_ids)


def evaluate_model(
    model: PPO,
    env: AoiSchedulingGymEnv,
    episodes: int,
) -> tuple[dict[str, float | int | str], list[dict[str, object]]]:
    total_reward = 0.0
    total_aoi = 0.0
    total_reachable_rate = 0.0
    total_successes = 0
    total_decisions = 0
    total_steps = 0
    traces: list[dict[str, object]] = []

    for episode_index in range(max(episodes, 1)):
        observation, _ = env.reset()

        while True:
            raw_observation = env.get_latest_raw_observation()
            action, _ = model.predict(observation, deterministic=True)
            next_observation, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            total_aoi += float(info["mean_aoi"])
            total_reachable_rate += float(info["mean_reachable_rate_mbps"])
            total_successes += int(bool(info["success"]))
            total_decisions += 1
            total_steps += 1
            traces.append(
                {
                    "episode_index": episode_index,
                    "step_index": int(raw_observation.step_index) if raw_observation is not None else None,
                    "selected_action_index": int(action),
                    "candidate_vehicle_ids": [int(vehicle_id) for vehicle_id in info["candidate_vehicle_ids"]]
                    if "candidate_vehicle_ids" in info
                    else [],
                    "selected_vehicle_id": info["selected_vehicle_id"],
                    "selected_vehicle_aoi_before": float(info["selected_vehicle_aoi_before"]),
                    "best_reachable_vehicle_id": info["best_reachable_vehicle_id"],
                    "best_reachable_aoi": float(info["best_reachable_aoi"]),
                    "action_valid": bool(info["action_valid"]),
                    "success": bool(info["success"]),
                    "active_vehicle_count": int(info["active_vehicle_count"]),
                    "reachable_vehicle_count": int(info["reachable_vehicle_count"]),
                    "mean_aoi_after_step": float(info["mean_aoi"]),
                }
            )

            if terminated or truncated:
                break
            observation = next_observation

    summary = {
        "policy": "ppo",
        "episodes": max(episodes, 1),
        "steps": total_steps,
        "total_reward": total_reward / max(episodes, 1),
        "mean_step_aoi": total_aoi / max(total_steps, 1),
        "success_rate": total_successes / max(total_decisions, 1),
        "mean_reachable_rate_mbps": total_reachable_rate / max(total_steps, 1),
    }
    return summary, traces


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_env, scenario_name, frame_count, vehicle_count = build_env_factory(args.config)
    vec_env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        verbose=args.verbose,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log else None,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model_path = args.output_dir / "ppo_scheduler"
    model.save(model_path)

    eval_env = make_env()
    eval_env.reset(seed=args.seed)
    evaluation, traces = evaluate_model(model, eval_env, episodes=args.eval_episodes)
    summary = {
        "scenario": scenario_name,
        "training_timesteps": args.timesteps,
        "trajectory_frames": frame_count,
        "unique_vehicles": vehicle_count,
        "evaluation": evaluation,
    }

    summary_path = args.output_dir / "ppo_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trace_path = args.trace_json or (args.output_dir / "ppo_eval_trace.json")
    trace_path.write_text(
        json.dumps(traces, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"scenario: {scenario_name}")
    print(f"trajectory frames: {frame_count}")
    print(f"unique vehicles: {vehicle_count}")
    print(f"saved model to: {model_path}.zip")
    print(json.dumps(summary["evaluation"], indent=2, ensure_ascii=False))
    print(f"saved summary to: {summary_path}")
    print(f"saved evaluation trace to: {trace_path}")


if __name__ == "__main__":
    main()
