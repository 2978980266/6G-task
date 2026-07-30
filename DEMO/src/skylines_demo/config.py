from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - exercised only when PyYAML is unavailable.
    yaml = None

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
class ChannelConfig:
    bandwidth_hz: float = 100_000_000.0
    tx_power_dbm: float = 30.0
    noise_power_dbm: float = -84.0
    noise_density_dbm_per_hz: float | None = None
    receiver_noise_figure_db: float = 10.0
    path_loss_offset_db: float = 61.4
    path_loss_exponent: float = 2.1
    nlos_extra_loss_db: float = 25.0
    min_distance_m: float = 1.0
    min_rate_mbps: float = 10.0
    implementation_margin_db: float = 0.0
    rate_efficiency_factor: float = 1.0
    max_spectral_efficiency_bps_hz: float | None = None
    building_cell_size_m: float = 8.0
    building_height_clearance_m: float = 1.0
    enable_vehicle_blockage: bool = False
    vehicle_blockage_radius_m: float = 3.5
    vehicle_blockage_loss_db: float = 18.0
    vehicle_blockage_height_m: float = 2.5

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ChannelConfig":
        noise_density = payload.get("noise_density_dbm_per_hz")
        max_spectral_efficiency = payload.get("max_spectral_efficiency_bps_hz")
        return ChannelConfig(
            bandwidth_hz=float(payload.get("bandwidth_hz", 100_000_000.0)),
            tx_power_dbm=float(payload.get("tx_power_dbm", 30.0)),
            noise_power_dbm=float(payload.get("noise_power_dbm", -84.0)),
            noise_density_dbm_per_hz=(
                float(noise_density) if noise_density is not None else None
            ),
            receiver_noise_figure_db=float(payload.get("receiver_noise_figure_db", 10.0)),
            path_loss_offset_db=float(payload.get("path_loss_offset_db", 61.4)),
            path_loss_exponent=float(payload.get("path_loss_exponent", 2.1)),
            nlos_extra_loss_db=float(payload.get("nlos_extra_loss_db", 25.0)),
            min_distance_m=float(payload.get("min_distance_m", 1.0)),
            min_rate_mbps=float(payload.get("min_rate_mbps", 10.0)),
            implementation_margin_db=float(payload.get("implementation_margin_db", 0.0)),
            rate_efficiency_factor=float(payload.get("rate_efficiency_factor", 1.0)),
            max_spectral_efficiency_bps_hz=(
                float(max_spectral_efficiency)
                if max_spectral_efficiency is not None
                else None
            ),
            building_cell_size_m=float(payload.get("building_cell_size_m", 8.0)),
            building_height_clearance_m=float(
                payload.get("building_height_clearance_m", 1.0)
            ),
            enable_vehicle_blockage=bool(payload.get("enable_vehicle_blockage", False)),
            vehicle_blockage_radius_m=float(payload.get("vehicle_blockage_radius_m", 3.5)),
            vehicle_blockage_loss_db=float(payload.get("vehicle_blockage_loss_db", 18.0)),
            vehicle_blockage_height_m=float(payload.get("vehicle_blockage_height_m", 2.5)),
        )


@dataclass(frozen=True)
class EnvironmentConfig:
    initial_caoi_seconds: float = 0.0
    initial_caoi_min_seconds: float | None = None
    initial_caoi_max_seconds: float | None = None
    packet_count: int = 25
    packet_size_bits: float = 2_400_000.0
    fallback_step_seconds: float = 0.3
    max_steps: int | None = None
    max_active_links: int | None = None
    allow_multi_v2i_from_rsu: bool = True
    age_tolerance_seconds: float | None = None

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "EnvironmentConfig":
        max_steps = payload.get("max_steps")
        max_active_links = payload.get("max_active_links")
        age_tolerance_seconds = payload.get("age_tolerance_seconds")
        initial_caoi_seconds = payload.get(
            "initial_caoi_seconds",
            payload.get("initial_aoi_seconds", 0.0),
        )
        initial_caoi_min_seconds = payload.get(
            "initial_caoi_min_seconds",
            payload.get("initial_aoi_min_seconds"),
        )
        initial_caoi_max_seconds = payload.get(
            "initial_caoi_max_seconds",
            payload.get("initial_aoi_max_seconds"),
        )
        return EnvironmentConfig(
            initial_caoi_seconds=float(initial_caoi_seconds),
            initial_caoi_min_seconds=(
                float(initial_caoi_min_seconds) if initial_caoi_min_seconds is not None else None
            ),
            initial_caoi_max_seconds=(
                float(initial_caoi_max_seconds) if initial_caoi_max_seconds is not None else None
            ),
            packet_count=int(payload.get("packet_count", 25)),
            packet_size_bits=float(payload.get("packet_size_bits", 2_400_000.0)),
            fallback_step_seconds=float(payload.get("fallback_step_seconds", 0.3)),
            max_steps=int(max_steps) if max_steps is not None else None,
            max_active_links=int(max_active_links) if max_active_links is not None else None,
            allow_multi_v2i_from_rsu=bool(payload.get("allow_multi_v2i_from_rsu", True)),
            age_tolerance_seconds=(
                float(age_tolerance_seconds) if age_tolerance_seconds is not None else None
            ),
        )


@dataclass(frozen=True)
class ResourceConfig:
    rsu_total_bandwidth_hz: float = 40_000_000.0
    rsu_resource_block_hz: float = 5_000_000.0
    rsu_total_rate_limit_mbps: float = 100.0
    v2v_total_bandwidth_hz: float = 40_000_000.0
    v2v_resource_block_hz: float = 5_000_000.0
    v2v_total_rate_limit_mbps: float = 60.0
    vehicle_link_rate_limit_mbps: float = 30.0
    max_resource_blocks_per_link: int = 2
    rsu_tx_power_dbm: float = 26.0
    vehicle_tx_power_dbm: float = 23.0

    @property
    def rsu_resource_block_count(self) -> int:
        return _resource_block_count(
            self.rsu_total_bandwidth_hz,
            self.rsu_resource_block_hz,
        )

    @property
    def v2v_resource_block_count(self) -> int:
        return _resource_block_count(
            self.v2v_total_bandwidth_hz,
            self.v2v_resource_block_hz,
        )

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ResourceConfig":
        return ResourceConfig(
            rsu_total_bandwidth_hz=float(
                payload.get("rsu_total_bandwidth_hz", 40_000_000.0)
            ),
            rsu_resource_block_hz=float(
                payload.get("rsu_resource_block_hz", 5_000_000.0)
            ),
            rsu_total_rate_limit_mbps=float(
                payload.get("rsu_total_rate_limit_mbps", 100.0)
            ),
            v2v_total_bandwidth_hz=float(
                payload.get("v2v_total_bandwidth_hz", 40_000_000.0)
            ),
            v2v_resource_block_hz=float(
                payload.get("v2v_resource_block_hz", 5_000_000.0)
            ),
            v2v_total_rate_limit_mbps=float(
                payload.get("v2v_total_rate_limit_mbps", 60.0)
            ),
            vehicle_link_rate_limit_mbps=float(
                payload.get("vehicle_link_rate_limit_mbps", 30.0)
            ),
            max_resource_blocks_per_link=int(
                payload.get("max_resource_blocks_per_link", 2)
            ),
            rsu_tx_power_dbm=float(payload.get("rsu_tx_power_dbm", 26.0)),
            vehicle_tx_power_dbm=float(payload.get("vehicle_tx_power_dbm", 23.0)),
        )


_DEFAULT_TASK_TEMPLATES = (
    ("T1", (1, 2, 3, 5, 6, 8, 12)),
    ("T2", (1, 3, 9, 10, 15, 16, 20)),
    ("T3", (2, 4, 6, 11, 14, 20, 25)),
    ("T4", (2, 5, 7, 8, 12, 17, 21, 25)),
    ("T5", (1, 3, 6, 9, 13, 18, 22, 24)),
    ("T6", (1, 4, 8, 10, 11, 15, 19, 23)),
    ("T7", (2, 3, 5, 7, 14, 16, 20, 25)),
    ("T8", (4, 6, 8, 12, 13, 17, 21, 24)),
)


@dataclass(frozen=True)
class TaskConfig:
    enabled: bool = True
    max_completed_tasks_per_vehicle: int = 3
    cooldown_steps: int = 1
    reward_base: float = 10.0
    freshness_decay_seconds: float = 4.0
    templates: tuple[tuple[str, tuple[int, ...]], ...] = _DEFAULT_TASK_TEMPLATES

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "TaskConfig":
        raw_templates = payload.get("templates")
        templates = (
            _parse_task_templates(raw_templates)
            if isinstance(raw_templates, dict)
            else _DEFAULT_TASK_TEMPLATES
        )
        return TaskConfig(
            enabled=bool(payload.get("enabled", True)),
            max_completed_tasks_per_vehicle=int(
                payload.get("max_completed_tasks_per_vehicle", 3)
            ),
            cooldown_steps=int(payload.get("cooldown_steps", 1)),
            reward_base=float(payload.get("reward_base", 10.0)),
            freshness_decay_seconds=float(
                payload.get("freshness_decay_seconds", 4.0)
            ),
            templates=templates,
        )


@dataclass(frozen=True)
class AutonomyConfig:
    enabled: bool = False
    penetration_rate: float = 0.0
    priority_weight: float = 2.0

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AutonomyConfig":
        return AutonomyConfig(
            enabled=bool(payload.get("enabled", False)),
            penetration_rate=float(payload.get("penetration_rate", 0.0)),
            priority_weight=float(payload.get("priority_weight", 2.0)),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_name: str
    seed: int
    trajectory_csv: Path
    building_csv: Path
    rsu: RSUConfig
    channel: ChannelConfig
    environment: EnvironmentConfig
    resource: ResourceConfig
    tasks: TaskConfig
    autonomy: AutonomyConfig

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ScenarioConfig":
        return ScenarioConfig(
            scenario_name=str(payload.get("scenario_name", "unnamed_scenario")),
            seed=int(payload.get("seed", 7)),
            trajectory_csv=Path(payload["trajectory_csv"]),
            building_csv=Path(payload["building_csv"]),
            rsu=RSUConfig.from_dict(payload["rsu"]),
            channel=ChannelConfig.from_dict(payload.get("channel", {})),
            environment=EnvironmentConfig.from_dict(payload.get("environment", {})),
            resource=ResourceConfig.from_dict(payload.get("resource", {})),
            tasks=TaskConfig.from_dict(payload.get("tasks", {})),
            autonomy=AutonomyConfig.from_dict(payload.get("autonomy", {})),
        )


def load_config(path: str | Path) -> ScenarioConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        text = handle.read()
    payload = _load_yaml_payload(text)
    return ScenarioConfig.from_dict(payload)


def dump_config_payload(payload: dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    return _dump_simple_yaml(payload)


def load_config_payload(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return _load_yaml_payload(config_path.read_text(encoding="utf-8-sig"))


def _load_yaml_payload(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)
    return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        key, separator, raw_value = line_without_comment.strip().partition(":")
        if not separator:
            raise ValueError(f"Invalid config line: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        raw_value = raw_value.strip()
        if not raw_value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _dump_simple_yaml(payload: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_dump_simple_yaml(value, indent + 2).rstrip())
        else:
            lines.append(f"{prefix}{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _resource_block_count(total_bandwidth_hz: float, resource_block_hz: float) -> int:
    if total_bandwidth_hz <= 0.0 or resource_block_hz <= 0.0:
        raise ValueError("Resource bandwidth values must be positive.")
    block_count = int(total_bandwidth_hz // resource_block_hz)
    if block_count <= 0:
        raise ValueError("Total bandwidth must contain at least one resource block.")
    return block_count


def _parse_task_templates(
    payload: dict[str, Any],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    templates: list[tuple[str, tuple[int, ...]]] = []
    for task_id, raw_packet_ids in payload.items():
        if isinstance(raw_packet_ids, str):
            packet_ids = tuple(
                int(part.strip())
                for part in raw_packet_ids.split(",")
                if part.strip()
            )
        elif isinstance(raw_packet_ids, (list, tuple)):
            packet_ids = tuple(int(packet_id) for packet_id in raw_packet_ids)
        else:
            raise ValueError(f"Invalid packet list for task {task_id!r}.")
        unique_packet_ids = tuple(dict.fromkeys(packet_ids))
        if not unique_packet_ids:
            raise ValueError(f"Task {task_id!r} must require at least one packet.")
        templates.append((str(task_id), unique_packet_ids))
    if not templates:
        raise ValueError("At least one task template is required.")
    return tuple(templates)
