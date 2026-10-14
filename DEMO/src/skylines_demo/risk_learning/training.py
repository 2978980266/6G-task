"""有限完整回合PPO；本模块的导入不会启动训练。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
from typing import Any, Sequence

from .domain import DOMAIN_ID
from .policy import RiskCandidatePolicy
from .rollout import EpisodeRecord, compute_returns


@dataclass(frozen=True)
class TrainingSpec:
    mode: str
    relay: bool
    train_seed: int
    episodes: int
    phase: str
    domain: str = DOMAIN_ID

    def __post_init__(self) -> None:
        if (self.mode not in {"B", "A"} or type(self.relay) is not bool
                or type(self.train_seed) is not int or self.train_seed < 0
                or self.domain != DOMAIN_ID
                or (self.phase, self.episodes) not in {("smoke", 2), ("fixed", 200)}):
            raise ValueError("明确指定B/A、S0、seed及完整smoke=2或fixed=200预算。")

    @property
    def rollout_episodes(self) -> int:
        return 2 if self.phase == "smoke" else 20

    @property
    def minibatch_size(self) -> int:
        return 23 if self.phase == "smoke" else 92

    @property
    def epochs(self) -> int:
        return 1 if self.phase == "smoke" else 5

    @property
    def world_steps(self) -> int:
        return self.episodes * 23

    @property
    def updates(self) -> int:
        return self.episodes // self.rollout_episodes

    def optimizer_spec(self) -> dict[str, Any]:
        return dict(algorithm="Adam", learning_rate=3e-4, clip=.2, value_coefficient=.5,
                    max_gradient_norm=.5, entropy_coefficient=0., gamma=1., beta=0.,
                    advantage_normalization="whole_rollout_population_std", epochs=self.epochs,
                    minibatch_size=self.minibatch_size, rollout_episodes=self.rollout_episodes,
                    device="cpu")


def initialize_seed(seed: int) -> tuple[Any, dict[str, int]]:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.Generator(device="cpu").manual_seed(seed), dict(python=seed, numpy=seed,
                                                               torch=seed, sampling_generator=seed)


def capture_rng_state(sampling: Any, minibatch: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        raise TypeError("需要已初始化的NumPy MT19937 legacy随机状态。")
    # 只存基本类型与Torch张量，保持严格weights_only重载，不放宽反序列化。
    serializable_numpy = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    return dict(python=random.getstate(), numpy=serializable_numpy, torch=torch.get_rng_state(),
                sampling=sampling.get_state(), minibatch=minibatch.get_state())


def clipped_policy_loss(new: Any, old: Any, advantages: Any) -> Any:
    import torch
    ratio = torch.exp(new - old)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("PPO ratio非有限。")
    return -torch.minimum(ratio * advantages, torch.clamp(ratio, .8, 1.2) * advantages).mean()


@dataclass(frozen=True)
class UpdateDiagnostics:
    world_steps: int
    optimizer_steps: int
    initial_ratio_max_error: float
    policy_losses: tuple[float, ...]
    value_losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]


class PPOTrainer:
    def __init__(self, policy: RiskCandidatePolicy, spec: TrainingSpec) -> None:
        import torch
        self.policy = policy
        self.spec = spec
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        self.generator = torch.Generator().manual_seed(spec.train_seed)

    def update(self, episodes: Sequence[EpisodeRecord]) -> UpdateDiagnostics:
        import torch
        if len(episodes) != self.spec.rollout_episodes:
            raise ValueError("必须收齐规定完整rollout再更新。")
        for episode in episodes:
            episode.validate()
        records = [record for episode in episodes for record in episode.records]
        returns = torch.tensor([g for e in episodes for g in compute_returns(e.rewards)])
        old = torch.tensor([r.path.joint_log_prob for r in records])
        values = torch.tensor([r.path.value for r in records])
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        if not all(torch.isfinite(x).all() for x in (returns, old, values, advantages)):
            raise FloatingPointError("完整训练目标/旧路径非有限，拒绝更新。")
        with torch.no_grad():
            initial = torch.stack([self.policy.evaluate_path(r.encoded, r.step, r.path)[0] for r in records])
        error = float((torch.exp(initial - old) - 1.).abs().max())
        if error > 1e-5:
            raise ValueError("采样与重算路径概率不一致；更新前ratio必须为1。")
        snapshot = deepcopy(self.policy.state_dict()), deepcopy(self.optimizer.state_dict()), self.generator.get_state()
        policy_losses: list[float] = []
        value_losses: list[float] = []
        norms: list[float] = []
        try:
            for _ in range(self.spec.epochs):
                order = torch.randperm(len(records), generator=self.generator)
                for start in range(0, len(records), self.spec.minibatch_size):
                    ids = order[start:start + self.spec.minibatch_size]
                    calculated = [self.policy.evaluate_path(records[i].encoded, records[i].step,
                                                          records[i].path) for i in ids.tolist()]
                    logp = torch.stack([p for p, _ in calculated])
                    predicted = torch.stack([v for _, v in calculated])
                    ploss = clipped_policy_loss(logp, old[ids], advantages[ids])
                    vloss = ((predicted - returns[ids]) ** 2).mean()
                    loss = ploss + .5 * vloss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("PPO loss非有限。")
                    self.optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5, error_if_nonfinite=True)
                    self.optimizer.step()
                    if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
                        raise FloatingPointError("PPO更新后参数非有限。")
                    policy_losses.append(float(ploss.detach()))
                    value_losses.append(float(vloss.detach()))
                    norms.append(float(norm))
        except Exception:
            self.policy.load_state_dict(snapshot[0])
            self.optimizer.load_state_dict(snapshot[1])
            self.generator.set_state(snapshot[2])
            raise
        return UpdateDiagnostics(len(records), len(norms), error, tuple(policy_losses), tuple(value_losses), tuple(norms))


def collect_episode(env: Any, config: Any, policy: RiskCandidatePolicy, generator: Any,
                    writer: Any, decoding: str = "sample", global_cap: int | None = None) -> tuple[EpisodeRecord, dict[str, Any]]:
    import json
    from .domain import build_safe_step
    from .encoding import encode_step
    from .evidence import action_from_path, apply_global_cap, encoded_payload, finish_episode, step_payload
    from .rollout import StepRecord
    obs = env.reset()
    rows: list[dict[str, Any]] = []
    records, rewards = [], []
    remaining = global_cap
    # 逐step独占流写；异常/中断保留已经写出的partial，不能优化它。
    with (writer.directory / "steps.jsonl").open("x", encoding="utf-8") as handle:
        while obs is not None:
            step = build_safe_step(obs, config)
            encoded = encode_step(step)
            path = policy.greedy(encoded, step) if decoding == "greedy" else policy.sample(encoded, step, generator)
            proposal = action_from_path(step, path)
            action = apply_global_cap(proposal, remaining) if remaining is not None else proposal
            result = env.step(action)
            row = step_payload(result)
            row.update(safe_input=asdict(step), encoded_input=encoded_payload(encoded), path=asdict(path),
                       proposal_before_cap=asdict(proposal), remaining_cap=remaining)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
            if result.invalid_action:
                raise AssertionError("学习路径产生非法动作，停止。")
            records.append(StepRecord(encoded, step, path))
            rewards.append(result.reward)
            if remaining is not None:
                remaining -= result.feedback.send_count
            obs = result.next_observation
    episode = EpisodeRecord(tuple(records), tuple(rewards), True)
    episode.validate()
    run = finish_episode(rows, config, env.objective_summary)
    writer.json("run.json", run)
    return episode, run


def run_training(spec: TrainingSpec, output_dir: Path, config_path: Path, risk_path: Path) -> dict[str, Any]:
    from ..bootstrap import load_runtime
    from ..config import load_risk_config
    from ..env.risk_comm_env import RiskCommEnv
    from .evidence import ExclusiveWriter
    from .metadata import freeze_manifest, save_checkpoint
    if output_dir.exists():
        raise FileExistsError("输出目录已存在，拒绝加载输入/训练。")
    manifest = freeze_manifest(config_path, risk_path)
    config = replace(load_risk_config(risk_path), enable_v2v_relay=spec.relay, link_prediction=False,
                     ego_eta_mode="causal-self-motion" if spec.mode == "B" else "public-crossing-step")
    runtime = load_runtime(config_path)
    writer = ExclusiveWriter(output_dir)
    writer.json("training_manifest.json", dict(training_spec=asdict(spec), optimizer=spec.optimizer_spec(),
                                               freeze=manifest, risk_config=asdict(config),
                                               requested_world_steps=spec.world_steps, requested_updates=spec.updates,
                                               scene_seed=runtime.config.seed, train_seed=spec.train_seed))
    updates, completed = 0, 0
    try:
        generator, seeds = initialize_seed(spec.train_seed)
        policy = RiskCandidatePolicy()
        trainer = PPOTrainer(policy, spec)
        writer.json("seeds.json", seeds)

        def save(count: int) -> None:
            save_checkpoint(output_dir / f"episode_{count:04d}.pt", policy, spec, manifest,
                            completed_episodes=count, updates=updates, optimizer_state=trainer.optimizer.state_dict(),
                            rng_state=capture_rng_state(generator, trainer.generator))

        save(0)
        diagnostics, episode_summaries = [], []
        rollout: list[EpisodeRecord] = []
        for index in range(1, spec.episodes + 1):
            episode_writer = ExclusiveWriter(output_dir / f"episode_{index:04d}")
            episode, run = collect_episode(RiskCommEnv(runtime, config), config, policy, generator, episode_writer)
            rollout.append(episode)
            completed = index
            episode_summaries.append(dict(episode=index, world_steps=index * 23, objective=run["objective"],
                                          legacy=run["legacy"], invalid_action_count=0))
            if index % spec.rollout_episodes == 0:
                diag = trainer.update(rollout)
                updates += 1
                diagnostics.append(asdict(diag))
                writer.json(f"update_{updates:04d}.json", asdict(diag))
                rollout = []
            if index in ({2} if spec.phase == "smoke" else {100, 200}):
                save(index)
            print(f"{spec.mode}/relay_{spec.relay}/seed{spec.train_seed}: episode {index}/{spec.episodes}, world-step {index*23}, updates {updates}", flush=True)
        summary = dict(completed_episodes=completed, actual_world_steps=completed * 23, updates=updates,
                       training_spec=asdict(spec), episodes=episode_summaries, update_diagnostics=diagnostics,
                       limitation="同固定场景内部流程验证；烟雾不证明性能收益，固定预算需另行授权。")
        assert completed * 23 == spec.world_steps and updates == spec.updates
        writer.json("training_summary.json", summary)
        return summary
    except BaseException as exc:
        writer.json("partial.json", dict(completed_episodes=completed, actual_world_steps=completed * 23,
                                         updates=updates, error=repr(exc), automatic_resume=False))
        raise
