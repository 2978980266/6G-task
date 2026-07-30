from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import load_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the DEMO scene data.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the scenario config YAML.",
    )
    return parser.parse_args()


def _format_range(values: list[float]) -> str:
    return f"{min(values):.3f} ~ {max(values):.3f} (mean {sum(values) / len(values):.3f})"


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args.config)

    vehicle_files = sorted(runtime.config.trajectory_csv.glob("*.csv")) if runtime.config.trajectory_csv.is_dir() else [runtime.config.trajectory_csv]
    building_files = sorted(runtime.config.building_csv.glob("*.csv")) if runtime.config.building_csv.is_dir() else [runtime.config.building_csv]

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for frame in runtime.replay.frames:
        for vehicle in frame.vehicles.values():
            xs.append(vehicle.position.x)
            ys.append(vehicle.position.y)
            zs.append(vehicle.position.z)

    bxs = [building.position.x for building in runtime.buildings]
    bys = [building.position.y for building in runtime.buildings]
    bzs = [building.position.z for building in runtime.buildings]

    print(f"scenario_name: {runtime.config.scenario_name}")
    print(f"seed: {runtime.config.seed}")
    print(f"vehicle_files: {len(vehicle_files)}")
    print(f"building_files: {len(building_files)}")
    print(f"step_count: {len(runtime.replay.frames)}")
    print(f"unique_vehicle_ids: {len(runtime.replay.vehicle_ids)}")
    print(f"building_count: {len(runtime.buildings)}")
    print(f"packet_count: {runtime.config.environment.packet_count}")
    print(f"packet_size_bits: {runtime.config.environment.packet_size_bits:.0f}")
    initial_caoi_min = runtime.config.environment.initial_caoi_min_seconds
    initial_caoi_max = runtime.config.environment.initial_caoi_max_seconds
    if initial_caoi_min is None and initial_caoi_max is None:
        print(
            "initial_caoi_seconds: "
            f"{runtime.config.environment.initial_caoi_seconds:.3f} (fixed)"
        )
    else:
        assert initial_caoi_min is not None and initial_caoi_max is not None
        print(
            "initial_caoi_seconds: "
            f"{initial_caoi_min:.3f} ~ {initial_caoi_max:.3f} (random)"
        )
    print(f"task_template_count: {len(runtime.config.tasks.templates)}")
    print(f"autonomy_enabled: {runtime.config.autonomy.enabled}")
    print(
        "autonomy_penetration_rate: "
        f"{runtime.config.autonomy.penetration_rate:.3f}"
    )
    print(
        "rsu_resources: "
        f"{runtime.config.resource.rsu_resource_block_count} blocks, "
        f"{runtime.config.resource.rsu_total_rate_limit_mbps:.1f} Mbps total"
    )
    print(
        "v2v_resources: "
        f"{runtime.config.resource.v2v_resource_block_count} blocks, "
        f"{runtime.config.resource.v2v_total_rate_limit_mbps:.1f} Mbps total"
    )
    print(
        "vehicle_link_rate_limit_mbps: "
        f"{runtime.config.resource.vehicle_link_rate_limit_mbps:.1f}"
    )
    print(
        "rsu_position: "
        f"({runtime.config.rsu.position.x:.3f}, {runtime.config.rsu.position.y:.3f}, {runtime.config.rsu.position.z:.3f})"
    )
    if xs:
        print(f"vehicle_x: {_format_range(xs)}")
        print(f"vehicle_y: {_format_range(ys)}")
        print(f"vehicle_z: {_format_range(zs)}")
    if bxs:
        print(f"building_x: {_format_range(bxs)}")
        print(f"building_y: {_format_range(bys)}")
        print(f"building_z: {_format_range(bzs)}")

    step_deltas = [frame.delta_seconds for frame in runtime.replay.frames]
    if step_deltas:
        print(f"step_delta_seconds: {_format_range(step_deltas)}")


if __name__ == "__main__":
    main()
