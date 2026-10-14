"""Fixed-model communication study; no training and no privileged policy inputs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.policies.risk_public_uplink import (
    rank_public_uplinks, select_deadline_stopped_action, select_public_uplink_action,
)
from skylines_demo.policies.risk_relay_policy import select_receiver_value_with_relay_action
from skylines_demo.risk.model import model_version, own_information_permission
from skylines_demo.types import RiskSchedulingAction
import run_risk_compare as compare
from trace_risk_episode import _step_payload

CAPS = (20, 30, 40, 47)
METHODS = ("no-communication", "age", "event-trigger", "deadline", "receiver-value",
           "related", "unrelated", "receiver-value-relay", "receiver-value-stop",
           "receiver-value-stop-relay", "version-uplink", "version-uplink-relay",
           "demand-uplink", "demand-uplink-relay")
TASK_FINGERPRINT = "1b091a60bb9e16cfed79526455cd71b5cf0c01a1b19cde5dc4c72e7aec05eabf"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def select_method(name):
    if name == "receiver-value-stop":
        return select_deadline_stopped_action
    if name == "receiver-value-stop-relay":
        return partial(select_deadline_stopped_action, selector=select_receiver_value_with_relay_action)
    if name.startswith(("version-uplink", "demand-uplink")):
        return partial(select_public_uplink_action, mode=name.split("-")[0],
                       with_relay=name.endswith("-relay"))
    return lambda obs, config: compare._POLICIES[name](obs, config, compare.ConflictCatalog((), (), ()))


def cap_action(action, remaining):
    """Predeclared global cap: downstream D, then R, then U; keep legal subset."""
    if remaining < 0:
        raise ValueError("Remaining budget must be non-negative.")
    downlinks = action.downlinks[:remaining]
    remaining -= len(downlinks)
    relays = action.relays[:remaining]
    remaining -= len(relays)
    uplinks = action.uplinks[:remaining]
    return RiskSchedulingAction(uplinks, downlinks, relays)


def capture_run(runtime, config, name, cap=None):
    env = RiskCommEnv(runtime, config)
    obs = env.reset()
    select = select_method(name)
    steps = []
    remaining = cap
    while obs is not None:
        action = select(obs, config)
        proposal = asdict(action)
        if remaining is not None:
            action = cap_action(action, remaining)
        result = env.step(action)
        if result.invalid_action:
            raise AssertionError("Formal study requires every action to be legal.")
        row = _step_payload(result)
        row["selection_evidence"] = {
            "proposal_before_budget_cap": proposal,
            "remaining_global_budget_before_action": remaining,
            "uplink_ranks": [asdict(r) for r in rank_public_uplinks(obs, mode=name.split("-")[0])]
            if name.startswith(("version-uplink", "demand-uplink")) else [],
        }
        steps.append(row)
        if remaining is not None:
            remaining -= result.feedback.send_count
        obs = result.next_observation
    objective = env.objective_summary
    events = [e for s in steps for e in s["event_results"]]
    scores = [e for s in steps for e in s["task_scores"]]
    deliveries = [e for s in steps for e in s["delivery_records"]]
    assert objective["evaluation_complete"] and len(events) == 46 and len(steps) == 23
    assert fingerprint(asdict(env.catalog)) == TASK_FINGERPRINT
    assert isclose(sum(s["reward"] for s in steps), -objective["objective_sum"])
    assert len(deliveries) == objective["send_count"]
    assert cap is None or objective["send_count"] <= cap
    for row in steps:
        action = row["action"]
        assert len(action["uplinks"]) <= 2 and len(action["downlinks"]) <= 2
        assert len(action["relays"]) <= (1 if config.enable_v2v_relay else 0)
        for delivery in row["delivery_records"]:
            assert delivery["usable_step"] == delivery["send_step"] + 1
            assert delivery["snapshot"]["observation_step"] <= delivery["send_step"]
    legacy = {field + "_count": sum(e[field] for e in events) for field in
              ("correct", "missed", "false_positive", "timeout", "insufficient", "abstained",
               "communication_error", "model_error", "mixed_error")}
    legacy["communication_involved_count"] = legacy["communication_error_count"] + legacy["mixed_error_count"]
    legacy["model_involved_count"] = legacy["model_error_count"] + legacy["mixed_error_count"]
    caoi = [s["caoi"]["mean_caoi_steps"] for s in steps if s["caoi"]["mean_caoi_steps"] is not None]
    legacy["mean_caoi_steps"] = sum(caoi) / len(caoi) if caoi else None
    legacy["mean_zero_receive_vehicle_count"] = sum(s["caoi"]["zero_receive_vehicle_count"] for s in steps) / len(steps)
    return dict(policy=name, global_send_cap=cap, cap_order="downlink_relay_uplink",
                risk_config=asdict(config), risk_model_version=model_version(config),
                own_information_permission=own_information_permission(config),
                catalog_fingerprint=TASK_FINGERPRINT, episode_terminated=True,
                episode_truncated=False, invalid_action_count=0, objective=objective,
                legacy=legacy, steps=steps, event_results=events, task_scores=scores)


def receipt_chain(record, deliveries):
    """Reconstruct actual provenance; every preceding hop must be usable by send."""
    snapshot = record["snapshot"]
    if record["sender_id"] == snapshot["source_id"]:
        return [record]
    previous = [r for r in deliveries if r["receiver_id"] == record["sender_id"]
                and r["usable_step"] <= record["send_step"]
                and r["snapshot"]["source_id"] == snapshot["source_id"]
                and r["snapshot"]["object_id"] == snapshot["object_id"]
                and r["snapshot"]["observation_step"] == snapshot["observation_step"]]
    if not previous:
        raise AssertionError("Forwarded snapshot has no prior usable actual delivery.")
    earlier = max(previous, key=lambda r: r["usable_step"])
    for key in ("x", "y", "z", "heading_deg", "velocity_x", "velocity_z", "velocity_basis",
                "velocity_previous_observation_step"):
        assert earlier["snapshot"][key] == snapshot[key], "Forwarding altered original state."
    return receipt_chain(earlier, deliveries) + [record]


def decision_receipts(run, event):
    deliveries = [r for step in run["steps"] for r in step["delivery_records"]]
    diagnostics = {o["object_id"]: o["diagnostic"] for o in event["object_results"] if o["diagnostic"]}
    result = []
    for record in deliveries:
        if record["receiver_id"] != event["receiver_id"] or record["usable_step"] > event["decision_step"]:
            continue
        obj = record["snapshot"]["object_id"]
        diagnostic = diagnostics.get(obj)
        adopted = bool(diagnostic and diagnostic["input_kind"] == "received"
                       and diagnostic["source_id"] == record["snapshot"]["source_id"]
                       and diagnostic["observation_step"] == record["snapshot"]["observation_step"]
                       and diagnostic["received_step"] == record["usable_step"])
        velocity_candidate = bool(diagnostic and diagnostic["snapshot"]["velocity_basis"] == "received_history"
                                  and diagnostic["snapshot"]["velocity_previous_observation_step"]
                                  == record["snapshot"]["observation_step"])
        chain = receipt_chain(record, deliveries)
        result.append(dict(receipt=record, adopted_at_decision=adopted,
                           possible_local_velocity_support=velocity_candidate, hop_chain=chain,
                           path_send_count=len(chain), path_cost=0.01 * len(chain)))
    return result


def compare_runs(before, after):
    events_before = {e["task_id"]: e for e in before["event_results"]}
    scores_before = {s["task_id"]: s for s in before["task_scores"]}
    scores_after = {s["task_id"]: s for s in after["task_scores"]}
    rows = []
    for event in after["event_results"]:
        old = events_before[event["task_id"]]
        b, a = scores_before[event["task_id"]], scores_after[event["task_id"]]
        old_alarm = set(b["true_alarm_object_ids"])
        new_alarm = set(a["true_alarm_object_ids"])
        label_recovered = bool(old["missed"] and not event["missed"])
        row = dict(task_id=event["task_id"], receiver_id=event["receiver_id"],
                   decision_step=event["decision_step"], true_risk=event["true_risk"],
                   before_score=b, after_score=a, outcome_changed=old["outcome"] != event["outcome"],
                   task_label_recovery=label_recovered,
                   wrong_object_only_label_recovery=bool(label_recovered and a["wrong_object_only_alarm"]),
                   true_object_recovered=bool(a["positive_success"] and not b["positive_success"]),
                   true_object_success_lost=bool(b["positive_success"] and not a["positive_success"]),
                   new_true_alarm_object_ids=sorted(new_alarm - old_alarm),
                   lost_true_alarm_object_ids=sorted(old_alarm - new_alarm),
                   abstention_removed=bool(old["abstained"] and not event["abstained"]),
                   abstention_added=bool(not old["abstained"] and event["abstained"]),
                   negative_abstention_to_no_risk=bool(not event["true_risk"] and old["abstained"]
                                                     and event["outcome"] == "no_risk"),
                   false_positive_added=bool(not old["false_positive"] and event["false_positive"]),
                   false_positive_removed=bool(old["false_positive"] and not event["false_positive"]),
                   extra_alarm_objects_added=sorted(set(a["extra_alarm_object_ids"]) - set(b["extra_alarm_object_ids"])),
                   extra_alarm_objects_removed=sorted(set(b["extra_alarm_object_ids"]) - set(a["extra_alarm_object_ids"])),
                   loss_delta=a["task_loss"] - b["task_loss"],
                   before_object_results=old["object_results"], after_object_results=event["object_results"],
                   before_receipts=decision_receipts(before, old), after_receipts=decision_receipts(after, event))
        rows.append(row)
    fields = ("task_label_recovery", "wrong_object_only_label_recovery", "true_object_recovered",
              "true_object_success_lost", "abstention_removed", "abstention_added",
              "negative_abstention_to_no_risk", "false_positive_added", "false_positive_removed")
    b, a = before["objective"], after["objective"]
    return dict(before=before["policy"], after=after["policy"],
                before_cap=before["global_send_cap"], after_cap=after["global_send_cap"],
                all_event_results_identical=before["event_results"] == after["event_results"],
                all_task_scores_identical=before["task_scores"] == after["task_scores"],
                actual_equal_send_count=b["send_count"] == a["send_count"],
                send_delta=a["send_count"] - b["send_count"],
                task_loss_delta=a["task_loss_sum"] - b["task_loss_sum"],
                objective_delta=a["objective_sum"] - b["objective_sum"],
                counts={f: sum(r[f] for r in rows) for f in fields}, tasks=rows,
                cost_attribution="Shared sends charged once per step; per-task path costs are diagnostic and non-additive.")


def save_run(folder, run):
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "run.json", run)
    with (folder / "steps.jsonl").open("x", encoding="utf-8") as handle:
        for row in run["steps"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="固定 B 通信实验与显式 A 对照，不启动训练。")
    parser.add_argument("--config", type=Path, default=ROOT / "data/config/train.yaml")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "data/config/risk.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("输出目录已存在，拒绝覆盖历史。")
    runtime, base = load_runtime(args.config), load_risk_config(args.risk_config)
    if (base.uplink_slots_per_step, base.downlink_slots_per_step, base.relay_slots_per_step) != (2, 2, 1):
        parser.error("本实验只接受固定名额2+2、中继1。")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = sorted(set((ROOT / "src").rglob("*.py")) | set((ROOT / "scripts").glob("*.py")))
    hashes = {str(p.relative_to(ROOT.parent)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    write_json(args.output_dir / "manifest.json", dict(command=sys.argv, source_hashes=hashes,
               source_fingerprint=fingerprint(hashes), caps=CAPS, methods=METHODS,
               scenario_config={**asdict(runtime.config), "trajectory_csv": str(runtime.config.trajectory_csv),
                                "building_csv": str(runtime.config.building_csv)},
               risk_config=asdict(base), task_fingerprint=TASK_FINGERPRINT,
               study_scope="Single fixed replay, deterministic repeat checks, no training, no validated upper bound."))
    write_json(args.output_dir / "catalog.json", asdict(RiskCommEnv(runtime, base).catalog))
    runs, summary, pairs = {}, {}, {}
    for mode, eta in (("B", "causal-self-motion"), ("A", "public-crossing-step")):
        for name in METHODS:
            cfg = replace(base, ego_eta_mode=eta, link_prediction=False, enable_v2v_relay=name.endswith("-relay"))
            run = capture_run(runtime, cfg, name)
            assert fingerprint(run) == fingerprint(capture_run(runtime, cfg, name))
            key = f"{mode}/{name}"
            runs[key] = run
            save_run(args.output_dir / key, run)
            summary[key] = dict(objective=run["objective"], legacy=run["legacy"],
                                fingerprint=fingerprint(run), repeat_identical=True,
                                risk_model_version=run["risk_model_version"],
                                global_send_cap=None, invalid_action_count=0)
            print(key, run["objective"]["task_loss_sum"], run["objective"]["send_count"],
                  run["objective"]["objective_sum"], flush=True)
        assert runs[f"{mode}/receiver-value"]["event_results"] == runs[f"{mode}/receiver-value-stop"]["event_results"]
        for name in METHODS:
            for baseline in ("no-communication", "receiver-value", "receiver-value-stop"):
                if name != baseline:
                    pairs[f"{mode}/{baseline}__{name}"] = compare_runs(runs[f"{mode}/{baseline}"], runs[f"{mode}/{name}"])
        for name in ("receiver-value", "receiver-value-stop", "version-uplink", "demand-uplink"):
            pairs[f"{mode}/{name}__relay"] = compare_runs(runs[f"{mode}/{name}"], runs[f"{mode}/{name}-relay"])
        # Same cap is a constraint control; only equal ACTUAL M is equal cost.
        for cap in CAPS:
            for name in ("receiver-value-stop", "version-uplink", "demand-uplink"):
                for relay in (False, True):
                    method = name + ("-relay" if relay else "")
                    cfg = replace(base, ego_eta_mode=eta, link_prediction=False, enable_v2v_relay=relay)
                    run = capture_run(runtime, cfg, method, cap)
                    assert fingerprint(run) == fingerprint(capture_run(runtime, cfg, method, cap))
                    key = f"{mode}/cap{cap}/{method}"
                    runs[key] = run
                    save_run(args.output_dir / key, run)
                    summary[key] = dict(objective=run["objective"], legacy=run["legacy"],
                                        fingerprint=fingerprint(run), repeat_identical=True,
                                        global_send_cap=cap, risk_model_version=run["risk_model_version"],
                                        invalid_action_count=0)
                pairs[f"{mode}/cap{cap}/{name}__relay"] = compare_runs(
                    runs[f"{mode}/cap{cap}/{name}"], runs[f"{mode}/cap{cap}/{name}-relay"])
        # Supplementary cost matching uses only uncapped send counts, never task labels.
        # Do not pad a frugal policy with useless transmissions to reach a larger cap.
        for name in ("receiver-value-stop", "version-uplink", "demand-uplink"):
            cap = min(runs[f"{mode}/{name}"]["objective"]["send_count"],
                      runs[f"{mode}/{name}-relay"]["objective"]["send_count"])
            for relay in (False, True):
                method = name + ("-relay" if relay else "")
                cfg = replace(base, ego_eta_mode=eta, link_prediction=False, enable_v2v_relay=relay)
                run = capture_run(runtime, cfg, method, cap)
                assert fingerprint(run) == fingerprint(capture_run(runtime, cfg, method, cap))
                key = f"{mode}/costmatched/{method}"
                runs[key] = run
                save_run(args.output_dir / key, run)
                summary[key] = dict(objective=run["objective"], legacy=run["legacy"],
                                    fingerprint=fingerprint(run), repeat_identical=True,
                                    global_send_cap=cap, risk_model_version=run["risk_model_version"],
                                    invalid_action_count=0)
            pairs[f"{mode}/costmatched/{name}__relay"] = compare_runs(
                runs[f"{mode}/costmatched/{name}"], runs[f"{mode}/costmatched/{name}-relay"])
    write_json(args.output_dir / "summary.json", summary)
    pair_dir = args.output_dir / "pairs"
    pair_dir.mkdir()
    pair_index = {}
    for index, (key, pair) in enumerate(pairs.items()):
        path = f"pair_{index:03d}.json"
        write_json(pair_dir / path, pair)
        pair_index[key] = dict(path=f"pairs/{path}", **{k: v for k, v in pair.items() if k != "tasks"})
    write_json(args.output_dir / "pair_index.json", pair_index)
    rows = [{"run": key, **value["objective"], **value["legacy"], "global_send_cap": value["global_send_cap"]}
            for key, value in summary.items()]
    with (args.output_dir / "summary.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"完成 {len(runs)} 个完整回合对照（每项重复一次），{len(pairs)} 组逐任务配对：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
