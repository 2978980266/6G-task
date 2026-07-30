from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from ..bootstrap import ScenarioRuntime


PPO_METADATA_VERSION = 1
PPO_ENCODING_VERSION = "task-aware-v1"


def build_ppo_metadata(
    *,
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
    requested_training_timesteps: int | None = None,
    actual_training_timesteps: int | None = None,
) -> dict[str, object]:
    return {
        "metadata_version": PPO_METADATA_VERSION,
        "encoding_version": PPO_ENCODING_VERSION,
        "model_type": "stable-baselines3-ppo",
        "scenario_name": runtime.config.scenario_name,
        "config_fingerprint": fingerprint(runtime.config),
        "scene_fingerprint": _scene_fingerprint(runtime),
        "top_k": int(top_k),
        "action_slots": int(action_slots),
        "action_nvec": [int(top_k + 1)] * int(action_slots),
        "observation_feature_count": int(observation_feature_count),
        "packet_count": int(runtime.config.environment.packet_count),
        "training_seed": int(training_seed),
        "episode_seed_strategy": "base_seed_plus_episode_index",
        "reward_weights": {
            "task_reward_weight": float(task_reward_weight),
            "task_progress_weight": float(task_progress_weight),
            "caoi_gain_weight": float(caoi_gain_weight),
            "empty_action_penalty": float(empty_action_penalty),
            "invalid_action_penalty": float(invalid_action_penalty),
        },
        "requested_training_timesteps": (
            int(requested_training_timesteps)
            if requested_training_timesteps is not None
            else None
        ),
        "actual_training_timesteps": (
            int(actual_training_timesteps)
            if actual_training_timesteps is not None
            else None
        ),
    }


def fingerprint(value: Any) -> str:
    canonical = _canonicalize(value)
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def metadata_path_for_model(model_path: str | Path) -> Path:
    candidate = Path(model_path)
    if candidate.suffix.lower() != ".zip":
        candidate = candidate.with_suffix(".zip")
    return candidate.with_suffix(".metadata.json")


def write_ppo_metadata(path: str | Path, metadata: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_ppo_metadata(path: str | Path) -> dict[str, object] | None:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PPO metadata must be a JSON object: {metadata_path}")
    return payload


def validate_ppo_metadata(
    metadata: dict[str, object] | None,
    *,
    metadata_path: str | Path,
    runtime: ScenarioRuntime,
    top_k: int,
    action_slots: int,
    observation_feature_count: int,
    allow_legacy_model: bool,
) -> None:
    if metadata is None:
        if allow_legacy_model:
            return
        raise ValueError(
            "PPO model metadata is missing. Re-train the model or pass "
            f"--allow-legacy-model for explicit compatibility mode: {metadata_path}"
        )

    expected = {
        "metadata_version": PPO_METADATA_VERSION,
        "encoding_version": PPO_ENCODING_VERSION,
        "model_type": "stable-baselines3-ppo",
        "scenario_name": runtime.config.scenario_name,
        "config_fingerprint": fingerprint(runtime.config),
        "scene_fingerprint": _scene_fingerprint(runtime),
        "top_k": int(top_k),
        "action_slots": int(action_slots),
        "action_nvec": [int(top_k + 1)] * int(action_slots),
        "observation_feature_count": int(observation_feature_count),
        "packet_count": int(runtime.config.environment.packet_count),
    }
    mismatches = [
        f"{key}: model={metadata.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "PPO model metadata does not match the current configuration or "
            f"encoder ({metadata_path}): " + "; ".join(mismatches)
        )


def _scene_fingerprint(runtime: ScenarioRuntime) -> str:
    replay_payload = []
    for frame in runtime.replay.frames:
        vehicles = {}
        for vehicle_id, snapshot in sorted(frame.vehicles.items()):
            vehicles[str(vehicle_id)] = {
                "position": snapshot.position,
                "speed": snapshot.speed,
                "angle_x": snapshot.angle_x,
                "angle_y": snapshot.angle_y,
                "prefab_name": snapshot.prefab_name,
            }
        replay_payload.append(
            {
                "step_index": frame.step_index,
                "simulation_frame": frame.simulation_frame,
                "elapsed_seconds": frame.elapsed_seconds,
                "delta_seconds": frame.delta_seconds,
                "vehicles": vehicles,
            }
        )
    return fingerprint(
        {
            "replay": replay_payload,
            "buildings": runtime.buildings,
        }
    )


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return _canonicalize(asdict(cast(Any, value)))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonicalize(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
