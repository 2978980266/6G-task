from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
DEFAULT_RISK_CONFIG_PATH = ROOT / "data" / "config" / "risk.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.perception.sensors import SensorModel
from skylines_demo.risk.conflicts import build_conflict_catalog
from skylines_demo.risk.opportunities import build_perception_opportunities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect conflict tasks and perception visibility in the risk scene."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the existing scenario config YAML.",
    )
    parser.add_argument(
        "--risk-config",
        default=str(DEFAULT_RISK_CONFIG_PATH),
        help="Path to the risk communication config YAML.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON summary instead of human-readable lines.",
    )
    return parser.parse_args()


def build_scene_summary(
    config_path: str | Path,
    risk_config_path: str | Path,
) -> dict[str, Any]:
    runtime = load_runtime(config_path)
    risk_config = load_risk_config(risk_config_path)
    catalog = build_conflict_catalog(runtime.replay, risk_config)
    sensor = SensorModel(
        rsu_position=runtime.config.rsu.position,
        los_evaluator=runtime.los_model,
        config=risk_config,
    )

    opportunities = build_perception_opportunities(runtime.replay, catalog, sensor)
    return {
        "scenario_name": runtime.config.scenario_name,
        "seed": runtime.config.seed,
        "step_count": len(runtime.replay),
        "vehicle_count": len(runtime.replay.vehicle_ids),
        "building_count": len(runtime.buildings),
        "conflict_point_count": len(catalog.points),
        "risk_task_count": sum(task.true_risk for task in catalog.tasks),
        "non_risk_task_count": sum(not task.true_risk for task in catalog.tasks),
        "task_count": len(catalog.tasks),
        "task_vehicle_invisible_rsu_visible_object_count": (
            opportunities["decision_frame_rsu_visibility"]["task_object_count"]
        ),
        "task_vehicle_and_rsu_invisible_other_vehicle_visible_object_count": (
            opportunities["decision_frame_helper_visibility"]["task_object_count"]
        ),
        "legacy_visibility_fields_scope": "decision_frame_task_object_pairs",
        **opportunities,
        "risk_config": asdict(risk_config),
    }


def main() -> None:
    args = parse_args()
    summary = build_scene_summary(args.config, args.risk_config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
