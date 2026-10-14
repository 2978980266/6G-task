"""内容指纹和严格独立sidecar，没有legacy或任意忽略字段入口。"""
from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import version
from importlib import import_module
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

from ..bootstrap import load_runtime
from ..config import load_risk_config
from ..env.risk_comm_env import RiskCommEnv
from ..risk.model import model_version, own_information_permission
from ..risk.scoring import OBJECTIVE_VERSION, OBJECTIVE_WEIGHTS
from .actions import COMBINATION_VERSION
from .encoding import ENCODING_VERSION, FEATURE_NAMES, GLOBAL_NAMES, SCALES
from .evidence import fingerprint
from .policy import NETWORK_SPEC, POLICY_VERSION, RiskCandidatePolicy
from .training import TrainingSpec

TASK_FINGERPRINT = "1b091a60bb9e16cfed79526455cd71b5cf0c01a1b19cde5dc4c72e7aec05eabf"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest(config_path: Path, risk_path: Path) -> dict[str, Any]:
    if (file_hash(config_path) != "9932058fa2ee8aecbc604f5ee73019afc04bc5c60c96d906235d36b4c21b653c"
            or file_hash(risk_path) != "18e3b06c07ef98a8abb33b7d181569f5a5586f0a2763dd8e81d38483c954fe61"):
        raise ValueError("配置内容偏离已批准冻结快照；禁止调传感器/模型/阈值。")
    runtime = load_runtime(config_path)
    cfg = load_risk_config(risk_path)
    catalog = RiskCommEnv(runtime, cfg).catalog
    if (len(runtime.replay), len(catalog.points), len(catalog.passages), len(catalog.tasks),
            sum(t.true_risk for t in catalog.tasks)) != (23, 15, 60, 46, 8):
        raise ValueError("冻结场景/目录计数漂移。")
    if fingerprint(asdict(catalog)) != TASK_FINGERPRINT or runtime.config.seed != 7:
        raise ValueError("冻结任务或场景seed漂移。")
    root = Path(__file__).resolve().parents[3]
    paths = sorted(set((root / "src").rglob("*.py")) | set((root / "scripts").glob("*.py")))
    sources = {p.relative_to(root.parent).as_posix(): file_hash(p) for p in paths}
    data = {p.relative_to(root.parent).as_posix(): file_hash(p)
            for directory in (runtime.config.trajectory_csv, runtime.config.building_csv)
            for p in sorted(directory.glob("*.csv"))}
    additions = {"DEMO/scripts/" + name + ".py" for name in
                 ("check_risk_learning", "train_risk_scheduler", "evaluate_risk_scheduler", "run_risk_learning_suite")}
    additions.add("DEMO/src/skylines_demo/policies/risk_training_controls.py")
    frozen_sources = {p: h for p, h in sources.items()
                      if "/risk_learning/" not in p and p not in additions}
    if (fingerprint(frozen_sources) != "22cfca1e19e86fff1a1fcf243fd933c55e859f3db47de3b544178d9e5062c94c"
            or fingerprint(data) != "ca0f35c80f944dc77bbac09e86737383468575f82fc52e44247948eb8b047af1"):
        raise ValueError("既有冻结源码或默认CSV内容漂移，停止学习。")
    return dict(task_fingerprint=TASK_FINGERPRINT, source_hashes=sources,
                source_fingerprint=fingerprint(sources), data_hashes=data, data_fingerprint=fingerprint(data),
                config_sha256=file_hash(config_path), risk_sha256=file_hash(risk_path),
                frozen_source_hashes=frozen_sources,
                scene_seed=runtime.config.seed, scenario_config=asdict(runtime.config), risk_config=asdict(cfg))


def dependencies() -> dict[str, str]:
    return dict(python=platform.python_version(), numpy=version("numpy"), torch=version("torch"))


def contract(spec: TrainingSpec, manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = replace(load_risk_config(Path(__file__).resolve().parents[3] / "data/config/risk.yaml"),
                  ego_eta_mode="causal-self-motion" if spec.mode == "B" else "public-crossing-step",
                  enable_v2v_relay=spec.relay, link_prediction=False)
    # 所有字段完整比较；变更权限/域/权重/训练参数均无静默兼容。
    return dict(algorithm=POLICY_VERSION, network=NETWORK_SPEC, encoding=ENCODING_VERSION,
                feature_names=FEATURE_NAMES, scales=SCALES, global_names=GLOBAL_NAMES,
                domain=spec.domain, combination=COMBINATION_VERSION,
                stop={"gate": "independent_sigmoid_then_conditional_softmax", "initial": .2, "forced_probability": 1.},
                mode=spec.mode, permission=own_information_permission(cfg), model_version=model_version(cfg),
                predict=False, relay=spec.relay, slots=[2, 2, int(spec.relay)],
                objective_version=OBJECTIVE_VERSION, weights=asdict(OBJECTIVE_WEIGHTS), gamma=1., beta=0.,
                episode_steps=23, optimizer=spec.optimizer_spec(), dependencies=dependencies(),
                freeze=manifest, training_spec=asdict(spec), requested_world_steps=spec.world_steps,
                limitation="单固定场景内部评测；训练seed不是独立场景，不证明泛化或性能收益。")


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _validate_counts(spec: TrainingSpec, episodes: int, updates: int) -> None:
    if (type(episodes) is not int or type(updates) is not int or not 0 <= episodes <= spec.episodes
            or episodes not in ({0, 2} if spec.phase == "smoke" else {0, 100, 200})
            or updates != episodes // spec.rollout_episodes):
        raise ValueError("checkpoint计数/预算不合法。")


def save_checkpoint(model_path: Path, policy: RiskCandidatePolicy, spec: TrainingSpec,
                    manifest: dict[str, Any], *, completed_episodes: int, updates: int,
                    optimizer_state: Any, rng_state: Any = None) -> None:
    import_module("numpy")
    import torch
    sidecar = model_path.with_suffix(".metadata.json")
    for target in (model_path, sidecar):
        if target.exists():
            raise FileExistsError(f"checkpoint已存在：{target}")
    _validate_counts(spec, completed_episodes, updates)
    payload = dict(policy=policy.state_dict(), optimizer=optimizer_state, rng=rng_state,
                   completed_episodes=completed_episodes, updates=updates, world_steps=completed_episodes * 23)
    with model_path.open("xb") as handle:
        torch.save(payload, handle)
    write_model_sidecar(model_path, spec, manifest, completed_episodes=completed_episodes, updates=updates)


def write_model_sidecar(model_path: Path, spec: TrainingSpec, manifest: dict[str, Any], *,
                        completed_episodes: int, updates: int) -> Path:
    _validate_counts(spec, completed_episodes, updates)
    payload = dict(contract(spec, manifest), model_sha256=file_hash(model_path),
                   completed_episodes=completed_episodes, updates=updates, actual_world_steps=completed_episodes * 23)
    path = model_path.with_suffix(".metadata.json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    return path


def read_spec(model_path: Path, metadata_path: Path | None = None) -> TrainingSpec:
    expected = model_path.with_suffix(".metadata.json")
    if metadata_path is not None and metadata_path.resolve() != expected.resolve():
        raise ValueError("必须使用同名sidecar。")
    try:
        data = json.loads(expected.read_text(encoding="utf-8"))
        return TrainingSpec(**data["training_spec"])
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError("缺失或非法风险metadata；没有legacy加载。") from exc


def load_risk_policy(model_path: Path, expected_spec: TrainingSpec,
                     expected_manifest: dict[str, Any]) -> RiskCandidatePolicy:
    try:
        meta = json.loads(model_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        episodes, updates = meta["completed_episodes"], meta["updates"]
        _validate_counts(expected_spec, episodes, updates)
        expected = dict(contract(expected_spec, expected_manifest), completed_episodes=episodes, updates=updates,
                        actual_world_steps=episodes * 23, model_sha256=file_hash(model_path))
        if _canonical(meta) != _canonical(expected):
            raise ValueError("metadata严格合同/模型内容不匹配。")
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError("风险metadata验证失败。") from exc
    import_module("numpy")
    import torch
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if (state["completed_episodes"], state["updates"], state["world_steps"]) != (episodes, updates, episodes * 23):
        raise ValueError("模型内部预算计数不匹配。")
    policy = RiskCandidatePolicy()
    policy.load_state_dict(state["policy"])
    if not all(torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError("加载参数含非有限值。")
    return policy
