from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ScenarioConfig, load_config
from .data import load_buildings, load_vehicle_replay
from .env import RelaySchedulingEnv, TaskRelaySchedulingEnv
from .geometry import BuildingLineOfSight
from .types import BuildingSnapshot, VehicleReplay
from .wireless import LogDistanceChannelModel


@dataclass(frozen=True)
class ScenarioRuntime:
    config: ScenarioConfig
    replay: VehicleReplay
    buildings: list[BuildingSnapshot]
    los_model: BuildingLineOfSight
    channel_model: LogDistanceChannelModel


def load_runtime(config_path: str | Path) -> ScenarioRuntime:
    config = load_config(config_path)
    replay = load_vehicle_replay(
        config.trajectory_csv,
        fallback_step_seconds=config.environment.fallback_step_seconds,
    )
    buildings = load_buildings(config.building_csv)
    los_model = BuildingLineOfSight(
        buildings,
        fallback_cell_size_m=config.channel.building_cell_size_m,
        height_clearance_m=config.channel.building_height_clearance_m,
    )
    channel_model = LogDistanceChannelModel(config.channel, los_model)
    return ScenarioRuntime(
        config=config,
        replay=replay,
        buildings=buildings,
        los_model=los_model,
        channel_model=channel_model,
    )


def build_environment(
    config_path: str | Path,
    seed: int | None = None,
) -> TaskRelaySchedulingEnv:
    runtime = load_runtime(config_path)
    effective_seed = runtime.config.seed if seed is None else seed
    return TaskRelaySchedulingEnv(
        replay=runtime.replay,
        rsu_position=runtime.config.rsu.position,
        link_evaluator=runtime.channel_model,
        environment_config=runtime.config.environment,
        resource_config=runtime.config.resource,
        task_config=runtime.config.tasks,
        autonomy_config=runtime.config.autonomy,
        seed=effective_seed,
    )


def build_legacy_environment(
    config_path: str | Path,
    seed: int | None = None,
) -> RelaySchedulingEnv:
    runtime = load_runtime(config_path)
    effective_seed = runtime.config.seed if seed is None else seed
    return RelaySchedulingEnv(
        replay=runtime.replay,
        rsu_position=runtime.config.rsu.position,
        link_evaluator=runtime.channel_model,
        environment_config=runtime.config.environment,
        autonomy_config=runtime.config.autonomy,
        seed=effective_seed,
    )
