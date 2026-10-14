from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
DEFAULT_RISK_CONFIG_PATH = ROOT / "data" / "config" / "risk.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.risk.opportunities import build_delivery_opportunities
from skylines_demo.risk.conflicts import build_conflict_catalog
from skylines_demo.types import RiskSchedulingAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check risk communication relay rules.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG_PATH))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _relay_config(config_path: str | Path, risk_config_path: str | Path):
    runtime = load_runtime(config_path)
    config = load_risk_config(risk_config_path)
    return runtime, replace(
        config,
        enable_v2v_relay=True,
        vehicle_sensor_range_m=200.0,
        vehicle_sensor_fov_deg=360.0,
        rsu_sensor_range_m=200.0,
    )


def _check_unknown_sender_object(config_path: str | Path, risk_config_path: str | Path) -> None:
    runtime, config = _relay_config(config_path, risk_config_path)
    env = RiskCommEnv(runtime=runtime, risk_config=config)
    observation = env.reset()
    assert observation.relay_candidates
    relay = observation.relay_candidates[0]
    invalid = env.step(
        RiskSchedulingAction(
            relays=(replace(relay, object_id=999999),),
        )
    )
    assert invalid.invalid_action
    assert invalid.delivered_relays == ()


def _check_same_step_receive_forward(
    config_path: str | Path,
    risk_config_path: str | Path,
) -> None:
    runtime, config = _relay_config(config_path, risk_config_path)
    env = RiskCommEnv(runtime=runtime, risk_config=config)
    observation = env.reset()
    relay = next(item for item in observation.relay_candidates
                 if any(candidate.receiver_id == item.sender_id
                        for candidate in observation.downlink_candidates))
    downlink = next(
        candidate
        for candidate in observation.downlink_candidates
        if candidate.receiver_id == relay.sender_id
    )
    invalid = env.step(
        RiskSchedulingAction(downlinks=(downlink,), relays=(relay,))
    )
    assert invalid.invalid_action
    assert invalid.delivered_downlinks == ()
    assert invalid.delivered_relays == ()


def _check_relay_snapshot_version(
    config_path: str | Path,
    risk_config_path: str | Path,
) -> None:
    runtime, config = _relay_config(config_path, risk_config_path)
    env = RiskCommEnv(runtime=runtime, risk_config=config)
    observation = env.reset()
    relay = observation.relay_candidates[0]
    result = env.step(RiskSchedulingAction(relays=(relay,)))
    assert not result.invalid_action
    assert result.delivered_relays[0].observation_step == observation.step_index
    receiver_cache = env._vehicle_knowledge[relay.receiver_id]
    received = receiver_cache.snapshot_for(relay.object_id)
    assert received.observation_step == observation.step_index
    assert received.received_step == observation.step_index + 1


def _build_scene_summary(runtime, config) -> dict[str, Any]:
    catalog = build_conflict_catalog(runtime.replay, config)
    return build_delivery_opportunities(
        runtime.replay, catalog, runtime.config.rsu.position, runtime.los_model,
        runtime.channel_model, runtime.config.resource, config,
    )


def run_checks(config_path: str | Path, risk_config_path: str | Path) -> dict[str, Any]:
    checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("unknown_sender_object_rejected", lambda: _check_unknown_sender_object(config_path, risk_config_path)),
        ("same_step_receive_forward_rejected", lambda: _check_same_step_receive_forward(config_path, risk_config_path)),
        ("relay_snapshot_version_preserved", lambda: _check_relay_snapshot_version(config_path, risk_config_path)),
    )
    results: list[dict[str, object]] = []
    for name, check in checks:
        try:
            check()
        except Exception as error:  # pragma: no cover - exercised by CLI failures.
            results.append({"name": name, "passed": False, "detail": str(error)})
        else:
            results.append({"name": name, "passed": True, "detail": "ok"})
    runtime, config = _relay_config(config_path, risk_config_path)
    scene_config = replace(
        load_risk_config(risk_config_path),
        enable_v2v_relay=True,
    )
    return {
        "scenario_name": runtime.config.scenario_name,
        "seed": runtime.config.seed,
        "rule_test_risk_config": asdict(config),
        "scene_risk_config": asdict(scene_config),
        "passed": all(item["passed"] for item in results),
        "checks": results,
        "scene": _build_scene_summary(runtime, scene_config),
    }


def main() -> int:
    args = parse_args()
    summary = run_checks(args.config, args.risk_config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        for item in summary["checks"]:
            print(f"{item['name']}: {'PASS' if item['passed'] else 'FAIL'} - {item['detail']}")
        for name, value in summary["scene"].items():
            print(f"{name}={value}")
        print(f"passed={summary['passed']}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
