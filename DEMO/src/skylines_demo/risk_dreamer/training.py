"""Bounded full-episode Dreamer training and independent, strict checkpoints.

Only Catalogue.encode outputs enter the learner. Real-world diagnostics are
written to evidence logs after execution and are never replay observations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, is_dataclass, replace
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import torch

from ..bootstrap import load_runtime
from ..config import load_risk_config
from ..env.risk_comm_env import RiskCommEnv
from ..risk.model import model_version, own_information_permission
from ..risk.rsu_knowledge import rsu_information_contract
from ..risk.scoring import OBJECTIVE_VERSION, OBJECTIVE_WEIGHTS
from ..risk_learning.domain import DOMAIN_ID
from ..risk_learning.evidence import step_payload
from .adapter import Catalogue, ENCODING_VERSION, PAIR_WIDTH, NODE_WIDTH
from .config import DreamerConfig
from .learning import ALGORITHM_VERSION, DreamerAgent


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_VERSION = "rsu-dreamer-pair-node-portable-v1"
EPISODE_STEPS = 23
OBSERVATION_KEYS = {"candidates", "legal", "global", "nodes"}


def json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite JSON evidence.")
        return value
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        handle.write("\n")


def initialize_seed(seed: int, threads: int = 1) -> np.random.Generator:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    if type(threads) is not int or threads < 1:
        raise ValueError("CPU threads must be a positive integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class TrainingSpec:
    mode: str
    relay: bool
    train_seed: int
    episodes: int
    size: str = "default"
    updates_per_episode: int = 1
    prefill_episodes: int = 1
    domain: str = DOMAIN_ID
    threads: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"B", "A"} or type(self.relay) is not bool:
            raise ValueError("Require mode B/A and an explicit relay boolean.")
        if type(self.train_seed) is not int or not 0 <= self.train_seed < 2**32:
            raise ValueError("train_seed must be an integer in [0, 2**32).")
        if self.size not in {"smoke", "default"} or self.domain != DOMAIN_ID:
            raise ValueError("Require smoke/default architecture and the S0 domain.")
        for name in ("episodes", "updates_per_episode", "prefill_episodes", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.prefill_episodes >= self.episodes:
            raise ValueError("prefill_episodes must be less than episodes, so training occurs.")

    @property
    def world_steps(self) -> int:
        return self.episodes * EPISODE_STEPS

    @property
    def updates(self) -> int:
        return (self.episodes - self.prefill_episodes) * self.updates_per_episode

    def model_config(self) -> DreamerConfig:
        return DreamerConfig.smoke() if self.size == "smoke" else DreamerConfig()


@dataclass
class Episode:
    observations: list[dict[str, np.ndarray]]
    actions: np.ndarray
    rewards: np.ndarray
    continues: np.ndarray
    # Collection keeps audit rows for disk traces. Replay
    # deliberately omits them; only the numerical arrays are training inputs.
    records: list[dict[str, Any]] | None

    def validate(self) -> None:
        if (len(self.observations) != EPISODE_STEPS + 1
                or self.actions.ndim != 2 or self.actions.shape[0] != EPISODE_STEPS
                or self.actions.shape[1] < 1
                or self.rewards.shape != (EPISODE_STEPS,)
                or self.continues.shape != (EPISODE_STEPS,)
                or (self.records is not None and len(self.records) != EPISODE_STEPS)):
            raise ValueError("Replay accepts complete 23-step episodes only.")
        if not np.array_equal(self.continues, np.r_[np.ones(EPISODE_STEPS - 1), 0.]):
            raise ValueError("Continuation must end once, at the actual final transition.")
        if not all(np.isfinite(a).all() for a in (self.actions, self.rewards, self.continues)):
            raise ValueError("Nonfinite replay transition.")
        first = self.observations[0]
        if set(first) != OBSERVATION_KEYS:
            raise ValueError("Replay observations must contain only encoder outputs.")
        nodes = first["nodes"].shape[0]
        shapes = dict(candidates=(nodes, nodes, PAIR_WIDTH), legal=(nodes, nodes, 3),
                      **{"global": (6,)}, nodes=(nodes, NODE_WIDTH))
        for observation in self.observations:
            if set(observation) != OBSERVATION_KEYS:
                raise ValueError("Unexpected replay observation fields.")
            for key, array in observation.items():
                dtype = np.dtype(bool if key == "legal" else np.float32)
                if array.shape != shapes[key] or array.dtype != dtype or not np.isfinite(array).all():
                    raise ValueError(f"Invalid replay observation {key}.")
        if any(np.any(a) for a in self.observations[-1].values()):
            raise ValueError("The unobserved terminal placeholder must be all zero.")


class Replay:
    """Bounded complete episodes; every sample filters from the real reset."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be a positive episode count.")
        self.episodes: deque[Episode] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: Episode) -> None:
        episode.validate()
        # Do not keep the audit's Python float copies of encoded observations
        # or offline diagnostics in RAM. The original episode remains intact
        # for the collector's summaries; sampling uses these arrays unchanged.
        self.episodes.append(Episode(episode.observations, episode.actions,
                                     episode.rewards, episode.continues, None))

    def sample(self, batch_size: int, rng: np.random.Generator,
               device: str | torch.device = "cpu") -> dict[str, Any]:
        if not self.episodes or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Sampling requires replay data and a positive batch size.")
        selected = [self.episodes[int(i)] for i in rng.integers(len(self.episodes), size=batch_size)]
        for episode in selected:
            episode.validate()
        observations = {
            key: torch.as_tensor(np.stack([[obs[key] for obs in episode.observations]
                                          for episode in selected]), device=device)
            for key in sorted(OBSERVATION_KEYS)
        }
        observed = torch.ones((batch_size, EPISODE_STEPS + 1), dtype=torch.bool, device=device)
        observed[:, -1] = False
        return dict(observations=observations,
                    actions=torch.as_tensor(np.stack([e.actions for e in selected]), device=device),
                    rewards=torch.as_tensor(np.stack([e.rewards for e in selected]), device=device),
                    continues=torch.as_tensor(np.stack([e.continues for e in selected]), device=device),
                    observed=observed)


EpisodeReplay = Replay


@torch.no_grad()
def collect_episode(env: RiskCommEnv, risk_config: Any, agent: DreamerAgent,
                    catalogue: Catalogue, *, greedy: bool = False,
                    trace_path: Path | None = None) -> tuple[Episode, dict[str, Any]]:
    """Return complete replay plus the offline objective; reset latent each call."""
    obs = env.reset()
    if Catalogue.from_initial_observation(obs, catalogue.relay).vehicle_ids != catalogue.vehicle_ids:
        raise ValueError("Initial public participant IDs differ from the catalogue.")
    agent.eval()
    prior = None
    observations, actions, rewards, continues, records = [], [], [], [], []
    trace = Path(trace_path).open("x", encoding="utf-8") if trace_path is not None else None
    try:
        while obs is not None:
            if len(actions) >= EPISODE_STEPS:
                raise ValueError("Environment exceeded the complete 23-step episode contract.")
            encoded = {key: np.array(value, copy=True) for key, value in catalogue.encode(obs, risk_config).items()}
            tensors = {key: torch.as_tensor(value, device=next(agent.parameters()).device).unsqueeze(0)
                       for key, value in encoded.items()}
            state, output = agent.observe(tensors, prior=prior, greedy=greedy)
            selections = output.selections[0]
            catalogue.validate_selection(encoded, selections)
            action = catalogue.to_action(selections)
            result = env.step(action)
            if result.invalid_action:
                raise AssertionError("Dreamer selected an invalid real S0 action.")
            done = result.next_observation is None
            observations.append(encoded)
            actions.append(output.actions[0].detach().cpu().numpy().copy())
            rewards.append(float(result.reward))
            continues.append(float(not done))
            row = json_safe(dict(step_payload(result), encoded_input=encoded,
                                 selection_messages=selections, model_action=actions[-1],
                                 log_prob=float(output.log_prob[0]), entropy=float(output.entropy[0]),
                                 done=done))
            records.append(row)
            if trace is not None:
                trace.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                trace.flush()
            # Carry the action-conditioned prior into the NEXT real posterior.
            prior = agent.world.transition(state, output.actions, sample=not greedy)
            obs = result.next_observation
    finally:
        if trace is not None:
            trace.close()
    if not observations:
        raise ValueError("Environment produced an empty episode.")
    observations.append({key: np.zeros_like(value) for key, value in observations[-1].items()})
    episode = Episode(observations, np.asarray(actions, dtype=np.float32),
                      np.asarray(rewards, dtype=np.float32), np.asarray(continues, dtype=np.float32), records)
    episode.validate()
    objective = json_safe(env.objective_summary)
    if (not objective["evaluation_complete"] or objective["steps_run"] != EPISODE_STEPS
            or objective["pending_task_count"] != 0
            or not math.isclose(math.fsum(rewards), -objective["objective_sum"], rel_tol=1e-7, abs_tol=1e-6)):
        raise AssertionError("Incomplete episode or reward sum does not equal -J.")
    return episode, objective


def effective_risk_config(spec: TrainingSpec, risk_path: Path) -> Any:
    return replace(load_risk_config(risk_path), enable_v2v_relay=spec.relay, link_prediction=False,
                   ego_eta_mode="causal-self-motion" if spec.mode == "B" else "public-crossing-step")


def dependencies() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "torch", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def build_manifest(config_path: Path, risk_path: Path, spec: TrainingSpec,
                   runtime: Any | None = None) -> dict[str, Any]:
    """Record semantic configuration without hashing source, data or models."""
    runtime = load_runtime(config_path) if runtime is None else runtime
    scenario = asdict(runtime.config)
    scenario.pop("trajectory_csv")
    scenario.pop("building_csv")
    risk = effective_risk_config(spec, risk_path)
    return json_safe(dict(scenario=scenario, risk=asdict(risk),
                          domain=DOMAIN_ID, encoding=ENCODING_VERSION,
                          slots=[2, 2, int(spec.relay)], episode_steps=EPISODE_STEPS,
                          permission=own_information_permission(risk),
                          rsu_information_contract=rsu_information_contract(),
                          executor_model=model_version(risk),
                          objective_version=OBJECTIVE_VERSION,
                          objective_weights=asdict(OBJECTIVE_WEIGHTS)))


def catalogue_metadata(catalogue: Catalogue) -> dict[str, Any]:
    return dict(vehicle_ids=list(catalogue.vehicle_ids), relay=catalogue.relay,
                pairs=catalogue.node_count ** 2,
                nodes=catalogue.node_count,
                action_size=catalogue.action_size)


def _metadata(spec: TrainingSpec, catalogue: Catalogue, manifest: dict[str, Any],
              counters: dict[str, int], objective: dict[str, Any]) -> dict[str, Any]:
    return json_safe(dict(schema=CHECKPOINT_VERSION, algorithm=ALGORITHM_VERSION,
                          training_spec=asdict(spec), architecture=asdict(spec.model_config()),
                          catalogue=catalogue_metadata(catalogue), manifest=manifest,
                          dependencies=dependencies(), counters=counters, objective=objective,
                          optimizer=dict(name="Adam", separate=["model", "actor", "critic"]),
                          automatic_resume=False))


def save_checkpoint(model_path: Path, agent: DreamerAgent, spec: TrainingSpec,
                    catalogue: Catalogue, manifest: dict[str, Any], *,
                    completed_episodes: int, world_steps: int, updates: int,
                    objective: dict[str, Any], rng: np.random.Generator) -> Path:
    model_path = Path(model_path)
    sidecar = model_path.with_suffix(".metadata.json")
    if model_path.exists() or sidecar.exists():
        raise FileExistsError("Checkpoint or sidecar already exists.")
    counters = dict(completed_episodes=completed_episodes, world_steps=world_steps, updates=updates)
    metadata = _metadata(spec, catalogue, manifest, counters, objective)
    numpy_state = np.random.get_state()
    rng_state = dict(torch=torch.get_rng_state(), python=random.getstate(),
                     numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                     replay=json_safe(rng.bit_generator.state))
    payload = dict(metadata=metadata, model_state=agent.state_dict(),
                   optimizer_states=agent.optimizer_state(), rng_state=rng_state)
    with model_path.open("xb") as handle:
        torch.save(payload, handle)
    write_json(sidecar, metadata)
    return sidecar


def load_checkpoint(model_path: Path, config_path: Path, risk_path: Path,
                    metadata_path: Path | None = None) -> tuple[DreamerAgent, Catalogue, TrainingSpec, dict[str, Any]]:
    """Validate configuration, architecture and checkpoint tensors."""
    model_path = Path(model_path)
    sidecar = Path(metadata_path) if metadata_path is not None else model_path.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Malformed Dreamer checkpoint metadata.")
    if metadata.get("schema") != CHECKPOINT_VERSION:
        raise ValueError("Incompatible Dreamer checkpoint format; this branch uses portable pair-node checkpoints.")
    try:
        spec = TrainingSpec(**metadata["training_spec"])
        ids = metadata["catalogue"]["vehicle_ids"]
        if not isinstance(ids, list) or ids != sorted(set(ids)) or not ids:
            raise ValueError("Catalogue IDs must be a nonempty sorted unique list.")
        catalogue = Catalogue(tuple(ids), spec.relay)
        expected_counters = dict(completed_episodes=spec.episodes, world_steps=spec.world_steps,
                                 updates=spec.updates)
        if (metadata["counters"] != expected_counters
                or any(type(v) is not int for v in metadata["counters"].values())):
            raise ValueError("Checkpoint counters do not match the explicit training budget.")
        objective = metadata["objective"]
        if (objective["steps_run"] != EPISODE_STEPS or not objective["evaluation_complete"]
                or objective["pending_task_count"] != 0
                or objective["objective_version"] != OBJECTIVE_VERSION):
            raise ValueError("Checkpoint objective is incomplete or incompatible.")
        expected = _metadata(spec, catalogue, build_manifest(config_path, risk_path, spec),
                             expected_counters, objective)
        saved = metadata
        if any(saved.get(key) != value for key, value in expected.items()
               if key != "dependencies"):
            raise ValueError("Dreamer configuration or architecture mismatch.")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed Dreamer checkpoint metadata.") from exc
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "model_state", "optimizer_states", "rng_state"}:
        raise ValueError("Unexpected Dreamer checkpoint payload.")
    if payload["metadata"] != saved:
        raise ValueError("Checkpoint payload and sidecar metadata differ.")
    if set(payload["optimizer_states"]) != {"model", "actor", "critic"}:
        raise ValueError("Missing separate Dreamer optimizer states.")
    for name, optimizer in payload["optimizer_states"].items():
        if not optimizer["state"]:
            raise ValueError("Checkpoint has no optimizer update evidence.")
        for state in optimizer["state"].values():
            step = float(state["step"])
            # A conditional pointer head may receive no gradient in a batch.
            # Adam counts updates per parameter, not calls to agent.update().
            valid = math.isfinite(step) and step.is_integer() and 0 < step <= spec.updates
            if not valid or (name != "actor" and step != spec.updates):
                raise ValueError("Optimizer update count differs from checkpoint counters.")
    if (set(payload["rng_state"]) != {"torch", "python", "numpy", "replay"}
            or not isinstance(payload["rng_state"]["torch"], torch.Tensor)):
        raise ValueError("Missing checkpoint RNG state.")
    # Participant identity is checked from the actual reset, never a future frame.
    runtime = load_runtime(config_path)
    initial = RiskCommEnv(runtime, effective_risk_config(spec, risk_path)).reset()
    if Catalogue.from_initial_observation(initial, spec.relay).vehicle_ids != catalogue.vehicle_ids:
        raise ValueError("Checkpoint IDs differ from the initial public observation.")
    agent = DreamerAgent(spec.model_config(), catalogue)
    target = agent.state_dict()
    state = payload["model_state"]
    if set(target) != set(state):
        raise ValueError("Checkpoint model tensor keys differ from the architecture.")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != target[key].shape
                or value.dtype != target[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid model tensor: {key}.")
    # Identity/structural buffers must agree with the independently built catalogue.
    structural = {"world.bins", "actor.slot_limits", "critic.bins", "slow_critic.bins"}
    for key in structural:
        if not torch.equal(state[key], target[key]):
            raise ValueError(f"Checkpoint structural buffer differs: {key}.")
    agent.load_state_dict(state, strict=True)
    agent.eval()
    return agent, catalogue, spec, metadata


def run_training(spec: TrainingSpec, output_dir: Path, config_path: Path,
                 risk_path: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; automatic resume is disabled.")
    rng = initialize_seed(spec.train_seed, spec.threads)
    runtime = load_runtime(config_path)
    risk = effective_risk_config(spec, risk_path)
    env = RiskCommEnv(runtime, risk)
    catalogue = Catalogue.from_initial_observation(env.reset(), spec.relay)
    config = spec.model_config()
    agent = DreamerAgent(config, catalogue)
    manifest = build_manifest(config_path, risk_path, spec, runtime)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "training_manifest.json", dict(training_spec=asdict(spec),
               architecture=asdict(config), catalogue=catalogue_metadata(catalogue), manifest=manifest,
               dependencies=dependencies(), requested_world_steps=spec.world_steps,
               requested_updates=spec.updates, automatic_resume=False))
    replay = Replay(config.replay_capacity)
    initial_actor = {key: value.detach().clone() for key, value in agent.actor.state_dict().items()}
    updates, steps, completed = 0, 0, 0
    summaries = []
    try:
        with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_log:
            for index in range(1, spec.episodes + 1):
                episode_dir = output_dir / f"episode_{index:04d}"
                episode_dir.mkdir(exist_ok=False)
                episode, objective = collect_episode(env, risk, agent, catalogue,
                                                      trace_path=episode_dir / "steps.jsonl")
                replay.add(episode)
                completed = index
                steps += len(episode.rewards)
                summary = dict(episode=index, world_steps=steps, objective=objective,
                               reward_sum=math.fsum(row["reward"] for row in episode.records),
                               invalid_action_count=0, terminated=True, truncated=False)
                summaries.append(summary)
                write_json(episode_dir / "summary.json", summary)
                if index > spec.prefill_episodes:
                    for _ in range(spec.updates_per_episode):
                        metrics = json_safe(agent.update(replay.sample(config.batch_size, rng)))
                        if not metrics or any(type(v) not in (int, float) or not math.isfinite(v)
                                              for v in metrics.values()):
                            raise FloatingPointError("Nonfinite or empty Dreamer update metrics.")
                        updates += 1
                        row = dict(update=updates, episode=index, world_steps=steps,
                                   replay_episodes=len(replay), **metrics)
                        metrics_log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        metrics_log.flush()
                print(json.dumps(dict(event="episode", episode=index, episodes=spec.episodes,
                                      world_steps=steps, updates=updates,
                                      objective=objective["objective_sum"])), flush=True)
        if (completed, steps, updates) != (spec.episodes, spec.world_steps, spec.updates):
            raise AssertionError("Actual training counters differ from the requested budget.")
        if build_manifest(config_path, risk_path, spec, runtime) != manifest:
            raise ValueError("Configuration changed during training; checkpoint refused.")
        actor_changed = any(not torch.equal(initial_actor[key], value)
                            for key, value in agent.actor.state_dict().items())
        model_path = output_dir / "model.pt"
        metadata_path = save_checkpoint(model_path, agent, spec, catalogue, manifest,
                                        completed_episodes=completed, world_steps=steps, updates=updates,
                                        objective=objective, rng=rng)
        result = dict(algorithm=ALGORITHM_VERSION, completed_episodes=completed,
                      actual_world_steps=steps, updates=updates, training_spec=asdict(spec),
                      episodes=summaries, actor_changed=actor_changed, model_path=str(model_path.resolve()),
                      metadata_path=str(metadata_path.resolve()),
                      limitation="Bounded same-scene verification; no performance or generalization claim.")
        write_json(output_dir / "training_summary.json", result)
        return json_safe(result)
    except BaseException as exc:
        write_json(output_dir / "partial.json", dict(completed_episodes=completed,
                   actual_world_steps=steps, updates=updates, error=repr(exc), automatic_resume=False))
        raise


def run_evaluation(model_path: Path, output_dir: Path, config_path: Path,
                   risk_path: Path, *, metadata_path: Path | None = None,
                   repeat: int = 2, mode: str | None = None,
                   relay: bool | None = None, threads: int = 1) -> dict[str, Any]:
    if repeat != 2:
        raise ValueError("Greedy deterministic verification requires exactly two full repeats.")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Evaluation output already exists.")
    initialize_seed(0, threads)
    agent, catalogue, spec, metadata = load_checkpoint(model_path, config_path, risk_path, metadata_path)
    if (mode is not None and mode != spec.mode) or (relay is not None and relay != spec.relay):
        raise ValueError("Evaluation mode/relay differs from the saved training contract.")
    risk = effective_risk_config(spec, risk_path)
    runtime = load_runtime(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "evaluation_manifest.json", dict(checkpoint_metadata=metadata, decoding="greedy", repeats=repeat,
               deterministic_latent="categorical_mode", threads=threads))
    results = []
    for index in range(repeat):
        # Deliberately different seeds also verify that greedy latent inference
        # consumes no sampling decisions and each call resets recurrent memory.
        initialize_seed((spec.train_seed + index) % 2**32, threads)
        directory = output_dir / f"repeat_{index + 1:02d}"
        directory.mkdir(exist_ok=False)
        episode, objective = collect_episode(RiskCommEnv(runtime, risk), risk, agent,
                                              catalogue, greedy=True, trace_path=directory / "steps.jsonl")
        row = dict(objective=objective, reward_sum=math.fsum(r["reward"] for r in episode.records),
                   invalid_action_count=0,
                   world_steps=len(episode.rewards), terminated=True, truncated=False)
        results.append(row)
        write_json(directory / "summary.json", row)
    if (results[0] != results[1]
            or (output_dir / "repeat_01/steps.jsonl").read_bytes()
            != (output_dir / "repeat_02/steps.jsonl").read_bytes()):
        raise AssertionError("Repeated full greedy evaluations differ.")
    if build_manifest(config_path, risk_path, spec, runtime) != metadata["manifest"]:
        raise ValueError("Configuration changed during evaluation.")
    summary = dict(algorithm=ALGORITHM_VERSION, mode=spec.mode, relay=spec.relay,
                   decoding="greedy", repeats=results, greedy_identical=True,
                   independent_samples=False,
                   limitation="Repeated same-scene verification; no generalization claim.")
    write_json(output_dir / "evaluation_summary.json", summary)
    return json_safe(summary)
