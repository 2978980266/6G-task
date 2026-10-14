"""Reproducible, offline diagnostics; privileged information never enters policies."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.perception.snapshots import KnowledgeCache, build_snapshot
from skylines_demo.risk.evaluator import evaluate_decision
from skylines_demo.risk.executor import execute_risk
from skylines_demo.risk.model import model_version, own_information_permission, own_schedule_eta
from skylines_demo.types import RiskSchedulingAction
import run_risk_compare as compare
from trace_risk_episode import _step_payload


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def compare_task_effect(before, after, deliveries, task):
    """Compare complete empty-action replay with actual replay, including velocities.

    A decision-step-only cache removal would retain velocities learned from earlier
    communication, so it is deliberately not used as the causal counterfactual.
    """
    predicted = {o["object_id"] for o in after["object_results"] if o["predicted_risk"]}
    predicted_before = {o["object_id"] for o in before["object_results"] if o["predicted_risk"]}
    truth_ids = set(task.related_vehicle_ids)
    diagnostics = {o["object_id"]: o.get("diagnostic") for o in after["object_results"]}
    receipts = []
    for delivery in deliveries:
        if (delivery["receiver_id"] != task.vehicle_id
                or delivery["usable_step"] > task.decision_step):
            continue
        diag = diagnostics.get(delivery["object_id"])
        adopted = bool(diag and diag["input_kind"] == "received"
                       and diag["source_id"] == delivery["source_id"]
                       and diag["observation_step"] == delivery["observation_step"]
                       and diag["received_step"] == delivery["usable_step"])
        velocity_basis = bool(diag and diag.get("snapshot")
                              and diag["snapshot"].get("velocity_basis") == "received_history"
                              and diag["snapshot"].get("velocity_previous_observation_step")
                              == delivery["observation_step"])
        receipts.append({**delivery, "true_conflict_object": delivery["object_id"] in truth_ids,
                         "adopted_at_decision": adopted, "used_for_local_velocity": velocity_basis})
    recovered = task.true_risk and before["missed"] and not after["missed"]
    return {
        "task_id": task.task_id, "receiver_id": task.vehicle_id,
        "decision_step": task.decision_step, "true_risk": task.true_risk,
        "related_vehicle_ids": list(task.related_vehicle_ids),
        "no_communication_outcome": before["outcome"], "outcome": after["outcome"],
        "outcome_changed": before["outcome"] != after["outcome"],
        "task_label_recovery": bool(recovered),
        "wrong_object_only_recovery": bool(recovered and not predicted & truth_ids),
        "related_object_recovery": bool(predicted & truth_ids and not predicted_before & truth_ids),
        "negative_abstention_removed": bool(not task.true_risk and before["abstained"] and after["correct"]),
        "negative_abstention_added": bool(not task.true_risk and not before["abstained"] and after["abstained"]),
        "false_positive_added": bool(not before["false_positive"] and after["false_positive"]),
        "false_positive_removed": bool(before["false_positive"] and not after["false_positive"]),
        "received_before_decision": receipts,
        "effective_receipt_count": sum(r["adopted_at_decision"] or r["used_for_local_velocity"] for r in receipts),
        "known_true_objects": [o["object_id"] for o in after["object_results"] if o["true_conflict"] and o["known"]],
        "predicted_true_objects": sorted(predicted & truth_ids),
        "predicted_non_true_objects": sorted(predicted - truth_ids),
    }


def capture_run(runtime, config, name, max_steps=None):
    effective = compare._effective_policy_config(config, runtime, name)
    env = RiskCommEnv(runtime, effective)
    observation = env.reset()
    events, deliveries, steps = [], [], []
    while observation is not None and (max_steps is None or observation.step_index < max_steps):
        if name == "diagnostic-oracle":
            action = compare.select_diagnostic_oracle_action(
                observation, effective, env.catalog, replay=runtime.replay,
                communication=env._communication,
            )
        else:
            action = compare._POLICIES[name](observation, effective, compare.ConflictCatalog((), (), ()))
        result = env.step(action)
        events.extend(asdict(event) for event in result.event_results)
        deliveries.extend(dict(
            link_type=r.link_type, sender_id=r.sender_id, receiver_id=r.receiver_id,
            object_id=r.snapshot.object_id, source_id=r.snapshot.source_id,
            observation_step=r.snapshot.observation_step, send_step=r.send_step, usable_step=r.usable_step,
            snapshot=asdict(r.snapshot),
        ) for r in result.delivery_records if r.receiver_id != 0)
        steps.append(_step_payload(result))
        observation = result.next_observation
    metrics = compare.run_policy(runtime, config, name, max_steps=max_steps)
    assert events == metrics["event_results"], "Diagnostic replay diverged from public experiment entry"
    return metrics, events, deliveries, steps


def ideal_current_information(runtime, config, max_steps=None):
    """Extra permission: all current objects and their entire past are observable.

    Same predictor and own task clock; no future states/times, truth labels or
    true-opponent IDs. Instant synthetic sensing bypasses real sensing, links,
    slots and hop delay. This is not an online communication policy or an upper bound.
    """
    env = RiskCommEnv(runtime, config)
    observation = env.reset()
    history = {}
    events = []
    while observation is not None and (max_steps is None or observation.step_index < max_steps):
        frame = runtime.replay.frames[observation.step_index]
        snapshots = {}
        for object_id, current in sorted(frame.vehicles.items()):
            previous = history.get(object_id)
            snapshots[object_id] = build_snapshot(
                object_id, 0, observation.step_index, current,
                previous[0] if previous else None, frame.elapsed_seconds,
                previous[1] if previous else None,
                previous_observation_step=previous[2] if previous else None,
                velocity_basis="ideal_global_past",
            )
            history[object_id] = current, frame.elapsed_seconds, observation.step_index
        for task in env.catalog.tasks:
            if task.decision_step != observation.step_index:
                continue
            cache = KnowledgeCache(0)
            for object_id, snapshot in snapshots.items():
                if object_id != task.vehicle_id:
                    cache.observe_local(snapshot)
            own = next(v.self_snapshot for v in observation.active_vehicles if v.vehicle_id == task.vehicle_id)
            decision = execute_risk(
                task, cache, observation.step_index, frame.elapsed_seconds, observation.step_times, config,
                ego_snapshot=own, scheduled_ego_arrival_seconds=own_schedule_eta(
                    config, task.crossing_step, observation.step_index, observation.delta_seconds),
            )
            events.append(asdict(evaluate_decision(task, decision)))
        observation = env.step(RiskSchedulingAction()).next_observation
    return {
        "permission": "all_current_objects_and_global_past_instant_synthetic_sensing",
        "uses_other_vehicle_future_or_labels_in_prediction": False,
        "own_eta_uses_offline_passage_step": config.ego_eta_mode == "public-crossing-step",
        "own_information_permission": own_information_permission(config),
        "validated_upper_bound": False,
        "task_count": len(events), "event_results": events,
        **{f"{key}_count": sum(e[key] for e in events) for key in (
            "correct", "missed", "false_positive", "timeout", "insufficient", "abstained")},
        "positive_with_related_prediction_count": sum(e["true_risk"] and any(
            o["true_conflict"] and o["predicted_risk"] for o in e["object_results"]) for e in events),
    }


def check_speed_units(runtime):
    ratios, dots = [], []
    previous = {}
    for frame in runtime.replay.frames:
        for object_id, current in sorted(frame.vehicles.items()):
            old = previous.get(object_id)
            if old is not None and frame.elapsed_seconds > old[1]:
                dx = current.position.x - old[0].position.x
                dz = current.position.z - old[0].position.z
                distance = math.hypot(dx, dz)
                if distance > 0:
                    heading = math.radians(current.angle_y)
                    dots.append((dx * math.sin(heading) + dz * math.cos(heading)) / distance)
                    if current.speed > 0:
                        ratios.append(distance / (frame.elapsed_seconds - old[1]) / current.speed)
            previous[object_id] = current, frame.elapsed_seconds
    return {"ratio_count": len(ratios), "displacement_rate_divided_by_csv_speed": {
        "min": min(ratios) if ratios else None, "median": median(ratios) if ratios else None,
        "max": max(ratios) if ratios else None},
        "mean_heading_displacement_dot": sum(dots) / len(dots) if dots else None,
        "model_uses_csv_speed_as_metres_per_second": False}


def build_audit(config_path, risk_config_path, policy_names, max_steps=None, ego_eta_mode=None):
    runtime = load_runtime(config_path)
    config = load_risk_config(risk_config_path)
    if ego_eta_mode is not None:
        config = replace(config, ego_eta_mode=ego_eta_mode)
    catalog = RiskCommEnv(runtime, config).catalog
    metrics, base_events, _, base_steps = capture_run(runtime, config, "no-communication", max_steps)
    baseline = {e["task_id"]: e for e in base_events}
    tasks = {t.task_id: t for t in catalog.tasks}
    runs = {"no-communication": {"policy": "no-communication", "metrics": metrics, "task_effects": []}}
    traces = {"no-communication": base_steps}
    cases = [(name, config, name) for name in policy_names if name != "no-communication"]
    cases += [(f"ablation-p{int(p)}-r{int(r)}", replace(config, link_prediction=p, enable_v2v_relay=r),
               "receiver-value-relay") for p in (False, True) for r in (False, True)]
    for key, case_config, policy in cases:
        result, events, deliveries, steps = capture_run(runtime, case_config, policy, max_steps)
        effects = [compare_task_effect(baseline[e["task_id"]], e, deliveries, tasks[e["task_id"]]) for e in events]
        effect_counts = {name: sum(e[name] for e in effects) for name in (
            "outcome_changed", "task_label_recovery", "wrong_object_only_recovery", "related_object_recovery",
            "negative_abstention_removed", "negative_abstention_added", "false_positive_added", "false_positive_removed")}
        runs[key] = {"policy": policy, "metrics": result, "task_effects": effects,
                     "communication_effect_counts": effect_counts,
                     "extra_permission": "truth_opponents_and_future_link_geometry" if policy == "diagnostic-oracle"
                     else "larger_slots" if policy in {"unlimited", "unlimited-delivery"} else None,
                     "validated_upper_bound": False}
        traces[key] = steps
    summary = {
        "model_version": model_version(config),
        "own_information_permission": own_information_permission(config),
        "own_eta_uses_offline_passage_step": config.ego_eta_mode == "public-crossing-step",
        "task_generation_permission": "offline_full_trajectory_catalog_and_public_task_deadlines_in_both_modes",
        "model_event_area_radius_m": config.conflict_cluster_radius_m,
        "legacy_core_diagnostic_radius_m": config.risk_radius_m,
        "ego_timing_rule": ("public_own_crossing_step_times_median_past_interval_with_configured_fallback"
                            if config.ego_eta_mode == "public-crossing-step"
                            else "current_self_heading_and_legal_displacement_speed_no_future_own_schedule"),
        "no_communication_counterfactual": "complete_empty_action_replay_including_velocity_estimation",
        "risk_config": asdict(config), "scenario_config": asdict(runtime.config),
        "catalog": {"point_count": len(catalog.points), "passage_count": len(catalog.passages),
                    "task_count": len(catalog.tasks), "positive_count": sum(t.true_risk for t in catalog.tasks),
                    "fingerprint": fingerprint(asdict(catalog)), "tasks": [asdict(t) for t in catalog.tasks]},
        "step_limit": max_steps, "runs": runs,
        "ideal_current_information": ideal_current_information(runtime, config, max_steps),
        "speed_unit_check": check_speed_units(runtime),
    }
    summary["fingerprint"] = fingerprint(summary)
    return summary, traces


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="复核独立风险模型与逐任务通信因果证据")
    parser.add_argument("--config", default=str(ROOT / "data/config/train.yaml"))
    parser.add_argument("--risk-config", default=str(ROOT / "data/config/risk.yaml"))
    parser.add_argument("--ego-eta-mode", choices=("public-crossing-step", "causal-self-motion"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policies", default="age,event-trigger,deadline,receiver-value,related,unrelated,unlimited,diagnostic-oracle")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    names = tuple(name.strip() for name in args.policies.split(",") if name.strip())
    unknown = set(names) - set(compare._POLICIES)
    if unknown or (args.steps is not None and args.steps <= 0):
        parser.error(f"无效策略或 step：{sorted(unknown)}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary, traces = build_audit(args.config, args.risk_config, names, args.steps, args.ego_eta_mode)
    write_json(output / "audit.json", summary)
    for name, rows in traces.items():
        with (output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    if args.repeat:
        repeated, repeated_traces = build_audit(args.config, args.risk_config, names, args.steps, args.ego_eta_mode)
        write_json(output / "audit_repeat.json", repeated)
        identical = fingerprint(summary) == fingerprint(repeated) and fingerprint(traces) == fingerprint(repeated_traces)
        write_json(output / "reproducibility.json", {"identical": identical,
                   "summary_fingerprint": fingerprint(summary), "trace_fingerprint": fingerprint(traces)})
        if not identical:
            raise AssertionError("重复运行不一致")
    files = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").glob("*.py")),
             *sorted((ROOT / "tests").glob("*.py")), Path(args.config), Path(args.risk_config)]
    for key in ("trajectory_csv", "building_csv"):
        input_path = Path(summary["scenario_config"][key])
        files.extend(sorted(input_path.rglob("*.csv")) if input_path.is_dir() else [input_path])
    write_json(output / "manifest.json", {"command": [sys.executable, *sys.argv],
               "files": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    print(f"输出：{output}\n任务集合：{summary['catalog']['fingerprint']}")
    for name, run in summary["runs"].items():
        data = run["metrics"]
        print(f"{name}: 正确={data['correct_count']} 漏报={data['missed_count']} "
              f"误报={data['false_positive_count']} 超时={data['timeout_count']} "
              f"无法预测={data['insufficient_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
