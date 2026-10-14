"""Fixed synthetic mechanism checks, separate from default-scene performance."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.config import RiskCommConfig
from skylines_demo.perception.snapshots import build_snapshot
from skylines_demo.risk.model import MODEL_VERSION, predict_pair
from skylines_demo.types import RiskSnapshot, Vector3, VehicleSnapshot


NOW = 1.0
PAST_TIMES = (0.0, 0.5, 1.0)
POINT = Vector3(0.0, 0.0, 0.0)
FROZEN_THRESHOLDS = {
    "conflict_cluster_radius_m": 8.0,
    "crossing_heading_min_deg": 45.0,
    "crossing_heading_max_deg": 135.0,
    "conflict_time_gap_seconds": 1.5,
    "extrapolation_seconds": 1.5,
}


@dataclass(frozen=True)
class Sample:
    time_seconds: float
    x: float
    z: float
    heading_deg: float


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ego: tuple[Sample, ...]
    obj: tuple[Sample, ...]
    fixed_expected_event: bool
    scheduled_eta: float
    expected_a: bool | None
    expected_b: bool | None
    reason_a: str
    reason_b: str
    limitation: str | None = None


def _straight(x: float, z: float, vx: float, vz: float, heading: float) -> tuple[Sample, ...]:
    return tuple(Sample(t, x + vx * (t - NOW), z + vz * (t - NOW), heading)
                 for t in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0))


def fixed_scenarios() -> tuple[Scenario, ...]:
    """Event labels are declared before invoking any prediction implementation."""
    ego = _straight(0, -10, 0, 10, 0)
    positive = "compatible_crossing_passages"
    cases = [
        Scenario("simultaneous-crossing", ego, _straight(-10, 0, 10, 0, 90),
                 True, 1.0, True, True, positive, positive),
        Scenario("time-separated-crossing", ego, _straight(-40, 0, 10, 0, 90),
                 False, 1.0, False, False, "arrival_time_mismatch", "arrival_time_mismatch"),
        Scenario("same-heading", ego, _straight(0, -10, 0, 10, 0),
                 False, 1.0, False, False, "non_crossing_heading", "non_crossing_heading"),
        Scenario("opposite-heading", ego, _straight(0, 10, 0, -10, 180),
                 False, 1.0, False, False, "non_crossing_heading", "non_crossing_heading"),
        Scenario("outside-event-area", ego, _straight(-10, 9, 10, 0, 90),
                 False, 1.0, False, False, "outside_event_area", "outside_event_area"),
        Scenario("stationary-occupancy-seven-metres", ego,
                 (Sample(0, 17, 0, 270), Sample(0.5, 7, 0, 270),
                  Sample(1, 7, 0, 270), Sample(2, 7, 0, 270), Sample(3, 7, 0, 270)),
                 True, 1.0, True, True, positive, positive),
    ]
    cases.append(Scenario(
        "object-future-turn", _straight(0, -14, 0, 10, 0),
        (Sample(0, -15, 9, 90), Sample(0.5, -10, 9, 90), Sample(1, -5, 9, 90),
         Sample(1.9, -5, 0, 180), Sample(2.4, 0, 0, 90), Sample(3.4, 10, 0, 90)),
        True, 1.4, False, False, "outside_event_area", "outside_event_area",
        "对象未来以恒定10 m/s转弯；当前东向切线距点9 m。真实经过同时发生，A/B均漏判，"
        "显示未观测转向的模型局限，不能据此声称安全。",
    ))
    cases.append(Scenario(
        "ego-future-turn",
        (Sample(0, -10, -15, 0), Sample(0.5, -10, -10, 0), Sample(1, -10, -5, 0),
         Sample(1.5, -10, 0, 0), Sample(2.5, 0, 0, 90), Sample(3.5, 10, 0, 90)),
        _straight(0, -15, 0, 10, 0),
        True, 1.5, False, None, "non_crossing_heading", "ego_passage_not_predictable",
        "自身未来以恒定10 m/s转弯后与对象交叉；当前航向相同且自身切线距点10 m。"
        "A仍因当前航向漏判，B明确弃权；公开自身ETA不能修复未来航向未知。",
    ))
    return tuple(cases)


def _closest_point_reference(samples: tuple[Sample, ...]) -> dict[str, float]:
    """Analytic closest point of fixture polylines, without catalog/model calls.

    Fixtures contain one continuous visit each. For the stationary fixture the
    incoming moving segment attains the minimum first, supplying its heading.
    This small reference is not a general replacement for visit extraction.
    """
    candidates: list[tuple[float, float, float]] = []
    for start, end in zip(samples, samples[1:]):
        dx, dz = end.x - start.x, end.z - start.z
        length_sq = dx * dx + dz * dz
        if length_sq == 0.0:
            # Stationary ties use the earlier moving arrival in these fixtures.
            continue
        fraction = min(1.0, max(0.0, -(start.x * dx + start.z * dz) / length_sq))
        distance_sq = (start.x + fraction * dx) ** 2 + (start.z + fraction * dz) ** 2
        time = start.time_seconds + fraction * (end.time_seconds - start.time_seconds)
        heading = math.degrees(math.atan2(dx, dz)) % 360.0
        candidates.append((distance_sq, time, heading))
    distance_sq, crossing_time, heading = min(candidates)
    return {"distance_m": math.sqrt(distance_sq), "crossing_time_seconds": crossing_time,
            "heading_deg": heading}


def independent_event_reference(case: Scenario) -> dict[str, object]:
    own = _closest_point_reference(case.ego)
    other = _closest_point_reference(case.obj)
    angle = abs((other["heading_deg"] - own["heading_deg"] + 180.0) % 360.0 - 180.0)
    gap = abs(other["crossing_time_seconds"] - own["crossing_time_seconds"])
    # Frozen event criteria, independent of prediction implementation/config changes.
    risk = own["distance_m"] <= 8.0 and other["distance_m"] <= 8.0 and 45 <= angle <= 135 and gap <= 1.5
    return {"risk": risk, "fixed_expected_risk": case.fixed_expected_event,
            "ego": own, "object": other, "heading_difference_deg": angle,
            "temporal_gap_seconds": gap,
            "reference_agrees_with_fixed_label": risk is case.fixed_expected_event,
            "method": "解析分段直线最近点与预先固定的事件标签；静止最近点取最早到达，"
                      "不调用冲突目录或预测模型，不使用默认场景任务。"}


def _snapshot(samples: tuple[Sample, ...], object_id: int) -> RiskSnapshot:
    previous, current = (sample for sample in samples if sample.time_seconds in (0.5, NOW))

    def vehicle(sample: Sample) -> VehicleSnapshot:
        return VehicleSnapshot(object_id, Vector3(sample.x, 0.0, sample.z),
                               999.0, 0.0, sample.heading_deg, "independent-synthetic")

    return build_snapshot(object_id, 1, 2, vehicle(current), vehicle(previous), NOW, 0.5,
                          previous_observation_step=1, velocity_basis="local_history")


def build_report() -> dict[str, object]:
    config = RiskCommConfig()
    thresholds_unchanged = all(getattr(config, name) == value
                               for name, value in FROZEN_THRESHOLDS.items())
    configs = {"A": replace(config, ego_eta_mode="public-crossing-step"),
               "B": replace(config, ego_eta_mode="causal-self-motion")}
    results = []
    for case in fixed_scenarios():
        expected = independent_event_reference(case)
        ego, obj = _snapshot(case.ego, 1), _snapshot(case.obj, 2)
        a = predict_pair(ego, obj, POINT, NOW, PAST_TIMES, configs["A"],
                         scheduled_ego_arrival_seconds=case.scheduled_eta)
        b = predict_pair(ego, obj, POINT, NOW, PAST_TIMES, configs["B"])
        mechanism_passed = (a.risk is case.expected_a and b.risk is case.expected_b
                            and a.reason == case.reason_a and b.reason == case.reason_b)
        results.append({
            "scenario_id": case.scenario_id,
            "expected_event": expected,
            "trajectory_evidence": {"ego": [asdict(s) for s in case.ego],
                                    "object": [asdict(s) for s in case.obj]},
            "prediction_inputs": {"now_seconds": NOW, "past_step_times": PAST_TIMES,
                                  "ego_snapshot": asdict(ego), "object_snapshot": asdict(obj),
                                  "scheduled_ego_arrival_seconds_A": case.scheduled_eta,
                                  "scheduled_ego_arrival_seconds_B": None,
                                  "future_waypoint_arrays_passed_to_predict_pair": False,
                                  "public_future_ego_passage_information_used": {"A": True, "B": False}},
            "predictions": {"A": asdict(a), "B": asdict(b)},
            "fixed_mechanism_expectations": {"A": case.expected_a, "B": case.expected_b},
            "mechanism_passed": mechanism_passed,
            "event_matched": {"A": a.risk is case.fixed_expected_event,
                              "B": b.risk is case.fixed_expected_event},
            "classification": "known_model_limitation" if case.limitation else "analytic_match",
            "limitation": case.limitation,
        })
    passed = thresholds_unchanged and all(
        s["mechanism_passed"] and s["expected_event"]["reference_agrees_with_fixed_label"]
        for s in results
    )
    return {
        "diagnostic": "independent-synthetic-risk-mechanisms",
        "model_version": MODEL_VERSION,
        "config": asdict(config),
        "mode_configs": {key: asdict(value) for key, value in configs.items()},
        "modes": {"A": "提供合成任务的公开自身经过ETA；该输入含未来自身经过信息权限。",
                  "B": "不提供公开经过ETA，只使用自身当前/历史观测估速与当前航向。"},
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "thresholds_unchanged": thresholds_unchanged,
        "thresholds_fitted": False,
        "default_scene_or_default_error_tasks_used": False,
        "passed": passed,
        "scenario_count": len(results),
        "known_limitation_count": sum(s["classification"] == "known_model_limitation" for s in results),
        "scenarios": results,
        "scope": "passed仅表示固定机制结果与局限均得到复现；不表示所有事件预测正确，"
                 "不构成真实场景策略收益、系统上界或模型通用有效性证明。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check independent fixed synthetic risk mechanisms.")
    parser.add_argument("--output", type=Path, help="Write a new JSON file; existing files are preserved.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    report = build_report()
    payload = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
