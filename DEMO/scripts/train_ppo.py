from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
EVALUATION_SEED_OFFSET = 1_000_000
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Windows + Anaconda + torch may load duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "缺少 PPO 训练依赖。请先安装 gymnasium、stable-baselines3 和 torch。"
    ) from exc

from skylines_demo.bootstrap import ScenarioRuntime, load_runtime
from skylines_demo.config import load_config
from skylines_demo.rl import PpoSchedulingGymEnv
from skylines_demo.rl.metadata import (
    build_ppo_metadata,
    metadata_path_for_model,
    write_ppo_metadata,
)
from run_experiment import run_experiment


class TrainingProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        report_interval: int,
    ) -> None:
        super().__init__(verbose=0)
        self._total_timesteps = max(total_timesteps, 1)
        self._report_interval = max(report_interval, 0)
        self._started_at = 0.0
        self._next_report = self._report_interval
        self._last_reported_step = -1

    def _on_training_start(self) -> None:
        self._started_at = time.monotonic()
        report_description = (
            f"每 {self._report_interval} timesteps 输出一次进度"
            if self._report_interval > 0
            else "周期进度输出已关闭"
        )
        print(
            "[PPO] 开始训练："
            f"目标 {self._total_timesteps} timesteps，"
            f"{report_description}。",
            flush=True,
        )

    def _on_step(self) -> bool:
        current_step = int(self.num_timesteps)
        should_report = (
            self._report_interval > 0
            and current_step >= self._next_report
        )
        if should_report or current_step >= self._total_timesteps:
            self._print_progress(current_step)
            if self._report_interval > 0:
                while self._next_report <= current_step:
                    self._next_report += self._report_interval
        return True

    def _on_training_end(self) -> None:
        current_step = int(self.num_timesteps)
        if current_step != self._last_reported_step:
            self._print_progress(current_step)
        elapsed_seconds = max(time.monotonic() - self._started_at, 0.0)
        print(
            f"[PPO] 训练完成，实际执行 {current_step} timesteps，"
            f"总用时 {_format_duration(elapsed_seconds)}。",
            flush=True,
        )

    def _print_progress(self, current_step: int) -> None:
        elapsed_seconds = max(time.monotonic() - self._started_at, 1e-9)
        speed = current_step / elapsed_seconds
        remaining_steps = max(self._total_timesteps - current_step, 0)
        eta_seconds = remaining_steps / speed if speed > 0.0 else 0.0
        progress = min(current_step / self._total_timesteps, 1.0)
        print(
            f"[PPO] {current_step}/{self._total_timesteps} "
            f"({progress:.1%}) | "
            f"已用 {_format_duration(elapsed_seconds)} | "
            f"{speed:.1f} steps/s | "
            f"预计剩余 {_format_duration(eta_seconds)}",
            flush=True,
        )
        self._last_reported_step = current_step


class PpoCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        output_dir: Path,
        runtime: ScenarioRuntime,
        top_k: int,
        action_slots: int,
        observation_feature_count: int,
        training_seed: int,
        task_reward_weight: float,
        task_progress_weight: float,
        caoi_gain_weight: float,
        empty_action_penalty: float,
        invalid_action_penalty: float,
        requested_training_timesteps: int,
    ) -> None:
        super().__init__(verbose=0)
        self._save_freq = max(save_freq, 0)
        self._next_save = self._save_freq
        self._output_dir = output_dir
        self._runtime = runtime
        self._top_k = top_k
        self._action_slots = action_slots
        self._observation_feature_count = observation_feature_count
        self._training_seed = training_seed
        self._task_reward_weight = task_reward_weight
        self._task_progress_weight = task_progress_weight
        self._caoi_gain_weight = caoi_gain_weight
        self._empty_action_penalty = empty_action_penalty
        self._invalid_action_penalty = invalid_action_penalty
        self._requested_training_timesteps = requested_training_timesteps

    def _on_step(self) -> bool:
        if self._save_freq <= 0:
            return True
        current_step = int(self.num_timesteps)
        if current_step < self._next_save:
            return True

        checkpoint_path = self._output_dir / f"ppo_scheduler_checkpoint_{current_step}"
        self.model.save(str(checkpoint_path))
        metadata = build_ppo_metadata(
            runtime=self._runtime,
            top_k=self._top_k,
            action_slots=self._action_slots,
            observation_feature_count=self._observation_feature_count,
            training_seed=self._training_seed,
            task_reward_weight=self._task_reward_weight,
            task_progress_weight=self._task_progress_weight,
            caoi_gain_weight=self._caoi_gain_weight,
            empty_action_penalty=self._empty_action_penalty,
            invalid_action_penalty=self._invalid_action_penalty,
            requested_training_timesteps=self._requested_training_timesteps,
            actual_training_timesteps=current_step,
        )
        write_ppo_metadata(metadata_path_for_model(checkpoint_path), metadata)
        print(
            f"[PPO] 已保存 checkpoint：{checkpoint_path}.zip，"
            f"实际 timesteps={current_step}。",
            flush=True,
        )
        while self._next_save <= current_step:
            self._next_save += self._save_freq
        return True


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds_part:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {seconds_part:02d}s"
    return f"{seconds_part:d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO baseline for the DEMO relay scheduling task.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the scenario config YAML.")
    parser.add_argument("--timesteps", type=int, default=300_000, help="Total PPO training timesteps.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the scenario config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "ppo",
        help="Directory used for the trained model and summaries.",
    )
    parser.add_argument("--eval-episodes", type=int, default=3, help="Number of deterministic evaluation episodes.")
    parser.add_argument("--top-k", type=int, default=16, help="Number of ranked task candidates exposed to PPO.")
    parser.add_argument(
        "--action-slots",
        type=int,
        default=8,
        help="Number of candidate-selection slots in each PPO action.",
    )
    parser.add_argument(
        "--task-reward-weight",
        type=float,
        default=1.0,
        help="Weight applied to the environment task completion reward.",
    )
    parser.add_argument(
        "--task-progress-weight",
        type=float,
        default=0.1,
        help="Shaping reward per newly delivered required packet.",
    )
    parser.add_argument(
        "--caoi-gain-weight",
        type=float,
        default=0.05,
        help="Shaping weight for priority-weighted CAoI improvement.",
    )
    parser.add_argument(
        "--empty-action-penalty",
        type=float,
        default=0.05,
        help="Penalty applied when PPO chooses the empty action while deliverable candidates exist.",
    )
    parser.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=0.5,
        help="Penalty applied if a generated scheduling action is invalid.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=512,
        help="Rollout length used by Stable-Baselines3 PPO.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5_000,
        help=(
            "Print training progress every N timesteps. "
            "Use 0 to disable periodic reports."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50_000,
        help=(
            "Save a PPO checkpoint every N timesteps. "
            "Use 0 to disable periodic checkpoints."
        ),
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
        help="Optional tensorboard log directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime(args.config)
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative.")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.action_slots <= 0:
        raise ValueError("--action-slots must be positive.")

    def make_env() -> PpoSchedulingGymEnv:
        return PpoSchedulingGymEnv(
            runtime=runtime,
            top_k=args.top_k,
            action_slots=args.action_slots,
            seed=seed,
            task_reward_weight=args.task_reward_weight,
            task_progress_weight=args.task_progress_weight,
            caoi_gain_weight=args.caoi_gain_weight,
            empty_action_penalty=args.empty_action_penalty,
            invalid_action_penalty=args.invalid_action_penalty,
        )

    vec_env = DummyVecEnv([make_env])
    observation_feature_count = int(vec_env.observation_space.shape[0])
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=args.verbose,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log else None,
    )
    progress_callback = TrainingProgressCallback(
        total_timesteps=args.timesteps,
        report_interval=args.progress_interval,
    )
    callbacks: list[BaseCallback] = [progress_callback]
    if args.checkpoint_interval > 0:
        callbacks.append(
            PpoCheckpointCallback(
                save_freq=args.checkpoint_interval,
                output_dir=args.output_dir,
                runtime=runtime,
                top_k=args.top_k,
                action_slots=args.action_slots,
                observation_feature_count=observation_feature_count,
                training_seed=seed,
                task_reward_weight=args.task_reward_weight,
                task_progress_weight=args.task_progress_weight,
                caoi_gain_weight=args.caoi_gain_weight,
                empty_action_penalty=args.empty_action_penalty,
                invalid_action_penalty=args.invalid_action_penalty,
                requested_training_timesteps=args.timesteps,
            )
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    actual_training_timesteps = int(model.num_timesteps)
    model_path = args.output_dir / "ppo_scheduler"
    print(f"[PPO] 正在保存模型：{model_path}.zip", flush=True)
    model.save(str(model_path))
    metadata = build_ppo_metadata(
        runtime=runtime,
        top_k=args.top_k,
        action_slots=args.action_slots,
        observation_feature_count=observation_feature_count,
        training_seed=seed,
        task_reward_weight=args.task_reward_weight,
        task_progress_weight=args.task_progress_weight,
        caoi_gain_weight=args.caoi_gain_weight,
        empty_action_penalty=args.empty_action_penalty,
        invalid_action_penalty=args.invalid_action_penalty,
        requested_training_timesteps=args.timesteps,
        actual_training_timesteps=actual_training_timesteps,
    )
    metadata_path = metadata_path_for_model(model_path)
    write_ppo_metadata(metadata_path, metadata)
    print(f"[PPO] 已保存模型元数据：{metadata_path}", flush=True)

    episode_summaries: list[dict[str, float | int | bool | str]] = []
    for episode_index in range(max(args.eval_episodes, 1)):
        evaluation_seed = EVALUATION_SEED_OFFSET + seed + episode_index
        print(
            f"[PPO] 开始评估回合 {episode_index + 1}/"
            f"{max(args.eval_episodes, 1)}，seed={evaluation_seed}。",
            flush=True,
        )
        summary = run_experiment(
            config_path=args.config,
            policy_name="ppo",
            seed=evaluation_seed,
            model_path=str(model_path),
            top_k=args.top_k,
            action_slots=args.action_slots,
        )
        episode_summaries.append(summary)

    evaluation = summarize_episode_results(episode_summaries)
    summary_payload = {
        "scenario": config.scenario_name,
        "seed": seed,
        "evaluation_seed_offset": EVALUATION_SEED_OFFSET,
        "requested_training_timesteps": args.timesteps,
        "training_timesteps": actual_training_timesteps,
        "top_k": args.top_k,
        "action_slots": args.action_slots,
        "task_reward_weight": args.task_reward_weight,
        "task_progress_weight": args.task_progress_weight,
        "caoi_gain_weight": args.caoi_gain_weight,
        "empty_action_penalty": args.empty_action_penalty,
        "invalid_action_penalty": args.invalid_action_penalty,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "progress_interval": args.progress_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "trajectory_frames": len(runtime.replay.frames),
        "unique_vehicles": len(runtime.replay.vehicle_ids),
        "evaluation": evaluation,
        "episode_summaries": episode_summaries,
    }

    summary_path = args.output_dir / "ppo_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"saved model to: {model_path}.zip")
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))
    print(f"saved summary to: {summary_path}")


def summarize_episode_results(
    episode_summaries: list[dict[str, float | int | bool | str]],
) -> dict[str, float | int | str]:
    if not episode_summaries:
        return {"policy": "ppo", "episodes": 0}

    numeric_keys = [
        "steps",
        "control_broadcast_count",
        "mean_step_caoi",
        "mean_autonomous_caoi",
        "mean_human_caoi",
        "mean_priority_weighted_caoi",
        "autonomous_vehicle_count",
        "human_vehicle_count",
        "mean_step_max_caoi",
        "episode_max_caoi",
        "attempted_count",
        "v2i_attempted_count",
        "v2v_attempted_count",
        "success_count",
        "v2i_success_count",
        "v2v_success_count",
        "failed_count",
        "invalid_action_count",
        "empty_actions",
        "age_violation_count",
        "mean_scheduled_links_per_step",
        "mean_v2i_scheduled_per_step",
        "mean_v2v_scheduled_per_step",
        "mean_candidate_count",
        "mean_v2i_candidate_count",
        "mean_v2v_candidate_count",
        "total_task_reward",
        "completed_task_count",
        "mean_completed_task_packet_aoi",
        "vehicles_with_completed_tasks",
        "total_delivered_packets",
        "total_v2i_delivered_packets",
        "total_v2v_delivered_packets",
        "mean_delivered_packets_per_step",
        "mean_v2i_delivered_packets_per_step",
        "mean_v2v_delivered_packets_per_step",
        "mean_used_rsu_resource_blocks",
        "mean_used_v2v_resource_blocks",
    ]
    aggregated: dict[str, float | int | str] = {
        "policy": "ppo",
        "episodes": len(episode_summaries),
    }
    for key in numeric_keys:
        values = [float(summary[key]) for summary in episode_summaries]
        aggregated[key] = sum(values) / max(len(values), 1)
    return aggregated


if __name__ == "__main__":
    main()
