from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .types import Vector3


@dataclass(frozen=True)
class RSUConfig:
    position: Vector3

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "RSUConfig":
        return RSUConfig(
            position=Vector3(
                x=float(payload["x"]),
                y=float(payload["y"]),
                z=float(payload["z"]),
            )
        )


@dataclass(frozen=True)
class ChannelModelConfig:
    bandwidth_hz: float = 100_000_000.0
    tx_power_dbm: float = 30.0
    noise_power_dbm: float = -84.0
    path_loss_offset_db: float = 61.4
    path_loss_exponent: float = 2.1
    nlos_extra_loss_db: float = 25.0
    min_distance_m: float = 1.0
    min_rate_mbps: float = 10.0
    building_cell_size_m: float = 8.0

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ChannelModelConfig":
        return ChannelModelConfig(
            bandwidth_hz=float(payload.get("bandwidth_hz", 100_000_000.0)),
            tx_power_dbm=float(payload.get("tx_power_dbm", 30.0)),
            noise_power_dbm=float(payload.get("noise_power_dbm", -84.0)),
            path_loss_offset_db=float(payload.get("path_loss_offset_db", 61.4)),
            path_loss_exponent=float(payload.get("path_loss_exponent", 2.1)),
            nlos_extra_loss_db=float(payload.get("nlos_extra_loss_db", 25.0)),
            min_distance_m=float(payload.get("min_distance_m", 1.0)),
            min_rate_mbps=float(payload.get("min_rate_mbps", 10.0)),
            building_cell_size_m=float(payload.get("building_cell_size_m", 8.0)),
        )


@dataclass(frozen=True)
class EnvironmentConfig:
    slot_seconds: float = 1.0
    initial_aoi_seconds: float = 0.0
    max_steps: int | None = None
    success_bonus: float = 1.0
    aoi_priority_bonus_scale: float = 0.25
    invalid_action_penalty: float = 1.0

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "EnvironmentConfig":
        max_steps = payload.get("max_steps")
        return EnvironmentConfig(
            slot_seconds=float(payload.get("slot_seconds", 1.0)),
            initial_aoi_seconds=float(payload.get("initial_aoi_seconds", 0.0)),
            max_steps=int(max_steps) if max_steps is not None else None,
            success_bonus=float(payload.get("success_bonus", 1.0)),
            aoi_priority_bonus_scale=float(payload.get("aoi_priority_bonus_scale", 0.25)),
            invalid_action_penalty=float(payload.get("invalid_action_penalty", 1.0)),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_name: str
    trajectory_csv: Path
    building_csv: Path
    rsu: RSUConfig
    channel: ChannelModelConfig
    environment: EnvironmentConfig

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ScenarioConfig":
        return ScenarioConfig(
            scenario_name=str(payload.get("scenario_name", "unnamed_scenario")),
            trajectory_csv=Path(payload["trajectory_csv"]),
            building_csv=Path(payload["building_csv"]),
            rsu=RSUConfig.from_dict(payload["rsu"]),
            channel=ChannelModelConfig.from_dict(payload.get("channel", {})),
            environment=EnvironmentConfig.from_dict(payload.get("environment", {})),
        )


def load_config(path: str | Path) -> ScenarioConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return ScenarioConfig.from_dict(payload)

