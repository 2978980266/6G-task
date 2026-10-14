from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
DEFAULT_RISK_CONFIG_PATH = ROOT / "data" / "config" / "risk.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import RiskCommConfig, load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.perception.snapshots import KnowledgeCache, build_snapshot
from skylines_demo.risk.conflicts import RiskTask
from skylines_demo.risk.evaluator import evaluate_decision
from skylines_demo.risk.executor import RiskDecision, execute_risk
from skylines_demo.types import RiskSnapshot, RiskSchedulingAction, Vector3, VehicleSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check risk communication invariants.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG_PATH))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _vehicle(vehicle_id: int, x: float, z: float) -> VehicleSnapshot:
    return VehicleSnapshot(
        vehicle_id=vehicle_id,
        position=Vector3(x, 0.0, z),
        speed=6.2,
        angle_x=0.0,
        angle_y=90.0,
        prefab_name="check",
    )


def _snapshot(observation_step: int, source_id: int = 1) -> RiskSnapshot:
    return build_snapshot(
        object_id=2,
        source_id=source_id,
        observation_step=observation_step,
        current=_vehicle(2, float(observation_step + 1), 0.0),
        previous=_vehicle(2, float(observation_step), 0.0),
        current_time_seconds=float(observation_step + 1),
        previous_time_seconds=float(observation_step),
    )


def _check_snapshot_versions() -> None:
    cache = KnowledgeCache(owner_id=1)
    cache.observe_local(_snapshot(3, source_id=1))
    cache.receive(_snapshot(2, source_id=1), received_step=4)
    assert cache.snapshot_for(2).observation_step == 3
    assert cache.caoi_seconds(5) == 3.0


def _check_risk_extremes() -> None:
    task = RiskTask(
        task_id="check-task",
        vehicle_id=1,
        conflict_point_id=0,
        conflict_point=Vector3(0.0, 0.0, 0.0),
        crossing_time=2.0,
        crossing_step=2,
        decision_step=1,
        true_risk=True,
        related_vehicle_ids=(2,),
    )
    timeout = RiskDecision(
        outcome="timeout",
        known_object_ids=(),
        predicted_object_ids=(),
    )
    result = evaluate_decision(task, timeout)
    assert result.communication_error is True
    assert result.model_error is False

    cache = KnowledgeCache(owner_id=1)
    cache.observe_local(_snapshot(0, source_id=1))
    decision = execute_risk(
        task=task,
        knowledge=cache,
        current_step=1,
        current_time_seconds=1.0,
        step_times=(0.0, 1.0),
        config=RiskCommConfig(),
    )
    assert decision.known_object_ids == (2,)
    assert decision.outcome == "insufficient"  # Known objects do not supply ego motion.


def _check_crossing_model_and_abstention() -> None:
    task = RiskTask("model-check", 1, 0, Vector3(0, 0, 0), 9999, 4, 0, True, (2,))
    own = replace(_snapshot(0), object_id=1, x=0, z=-10, heading_deg=0,
                  velocity_x=0, velocity_z=10)
    obj = replace(_snapshot(0), x=-10, velocity_x=10)

    def decision(snapshot: RiskSnapshot) -> RiskDecision:
        cache = KnowledgeCache(1)
        cache.observe_local(snapshot)
        return execute_risk(task, cache, 0, 0, (0,), RiskCommConfig(),
                            ego_snapshot=own, scheduled_ego_arrival_seconds=1)

    assert decision(obj).outcome == "risk"
    assert decision(replace(obj, heading_deg=0)).outcome == "no_risk"
    assert decision(replace(obj, heading_deg=180)).outcome == "no_risk"
    assert decision(replace(obj, x=-100)).outcome == "no_risk"
    unknown = decision(replace(obj, velocity_x=None, velocity_z=None))
    assert unknown.outcome == "insufficient"
    event = evaluate_decision(task, unknown)
    assert event.missed and event.abstained and event.insufficient and not event.correct


def _check_environment_order(
    config_path: str,
    risk_config_path: str,
) -> None:
    runtime = load_runtime(config_path)
    risk_config = replace(
        load_risk_config(risk_config_path),
        rsu_sensor_range_m=1.0,
        vehicle_sensor_fov_deg=360.0,
    )
    env = RiskCommEnv(runtime=runtime, risk_config=risk_config)
    observation = env.reset()
    assert observation.uplink_candidates
    uplink = observation.uplink_candidates[0]

    delivered = env.step(RiskSchedulingAction(uplinks=(uplink,)))
    assert not delivered.invalid_action
    assert len(delivered.delivered_uplinks) == 1
    assert delivered.next_observation is not None
    assert any(
        candidate.object_id == uplink.object_id
        for candidate in delivered.next_observation.downlink_candidates
    )

    env = RiskCommEnv(runtime=runtime, risk_config=risk_config)
    observation = env.reset()
    uplink = observation.uplink_candidates[0]
    invalid = env.step(RiskSchedulingAction(uplinks=(uplink, uplink)))
    assert invalid.invalid_action
    assert invalid.delivered_uplinks == ()


def _run_checks(config_path: str, risk_config_path: str) -> list[dict[str, object]]:
    checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("snapshot_versions", _check_snapshot_versions),
        ("risk_error_classification", _check_risk_extremes),
        ("crossing_heading_arrival_gap_and_unknown_motion", _check_crossing_model_and_abstention),
        (
            "environment_step_order_and_action_atomicity",
            lambda: _check_environment_order(config_path, risk_config_path),
        ),
    )
    results: list[dict[str, object]] = []
    for name, check in checks:
        try:
            check()
        except Exception as error:  # pragma: no cover - exercised by failed CLI checks.
            results.append({"name": name, "passed": False, "detail": str(error)})
        else:
            results.append({"name": name, "passed": True, "detail": "ok"})
    return results


def main() -> int:
    args = parse_args()
    checks = _run_checks(args.config, args.risk_config)
    summary = {"passed": all(item["passed"] for item in checks), "checks": checks}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        for item in checks:
            print(f"{item['name']}: {'PASS' if item['passed'] else 'FAIL'} - {item['detail']}")
        print(f"passed={summary['passed']}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
