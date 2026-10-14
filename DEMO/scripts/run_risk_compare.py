from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
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

from skylines_demo.bootstrap import ScenarioRuntime, load_runtime
from skylines_demo.config import RiskCommConfig, load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.policies.risk_baselines import (
    select_age_action,
    select_deadline_action,
    select_event_trigger_action,
    select_unlimited_action,
    select_relation_action,
)
from skylines_demo.policies.risk_receiver_policy import select_receiver_value_action
from skylines_demo.policies.risk_relay_policy import (
    select_receiver_value_with_relay_action,
)
from skylines_demo.risk.conflicts import ConflictCatalog
from skylines_demo.risk.model import model_version, own_information_permission
from skylines_demo.risk.sufficiency import (
    prediction_triggered_task,
)
from skylines_demo.policies.risk_oracle import select_diagnostic_oracle_action
from skylines_demo.perception.communication import SnapshotCommunication
from skylines_demo.types import RiskObservation, RiskSchedulingAction

PolicySelector = Callable[
    [RiskObservation, RiskCommConfig, ConflictCatalog],
    RiskSchedulingAction,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rule-based risk communication policies on one common replay."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG_PATH))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--policies",
        default="unlimited,age,event-trigger,deadline,receiver-value,diagnostic-oracle",
        help="Comma-separated policy names.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--link-prediction",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable causal communication-window prediction for this comparison.",
    )
    parser.add_argument(
        "--enable-v2v-relay",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable the explicit vehicle relay path for this comparison.",
    )
    parser.add_argument(
        "--compare-link-prediction",
        action="store_true",
        help="Include otherwise-identical link_prediction=false/true results.",
    )
    parser.add_argument(
        "--compare-relay",
        action="store_true",
        help="Include otherwise-identical enable_v2v_relay=false/true results.",
    )
    return parser.parse_args()


def _standard_selector(
    selector: Callable[[RiskObservation, RiskCommConfig], RiskSchedulingAction],
) -> PolicySelector:
    def wrapped(
        observation: RiskObservation,
        config: RiskCommConfig,
        catalog: ConflictCatalog,
    ) -> RiskSchedulingAction:
        del catalog
        return selector(observation, config)

    return wrapped


def _select_diagnostic_oracle_action(
    observation: RiskObservation,
    config: RiskCommConfig,
    catalog: ConflictCatalog,
) -> RiskSchedulingAction:
    return select_diagnostic_oracle_action(observation, config, catalog)


def _select_related_action(
    observation: RiskObservation,
    config: RiskCommConfig,
    catalog: ConflictCatalog,
) -> RiskSchedulingAction:
    del catalog
    return select_relation_action(observation, config, related=True)


def _select_unrelated_action(
    observation: RiskObservation,
    config: RiskCommConfig,
    catalog: ConflictCatalog,
) -> RiskSchedulingAction:
    del catalog
    return select_relation_action(observation, config, related=False)


_POLICIES: dict[str, PolicySelector] = {
    "no-communication": lambda observation, config, catalog: RiskSchedulingAction(),
    "unlimited": _standard_selector(select_unlimited_action),
    "unlimited-delivery": _standard_selector(select_unlimited_action),
    "age": _standard_selector(select_age_action),
    "event-trigger": _standard_selector(select_event_trigger_action),
    "deadline": _standard_selector(select_deadline_action),
    "receiver-value": _standard_selector(select_receiver_value_action),
    "receiver-value-relay": _standard_selector(select_receiver_value_with_relay_action),
    "diagnostic-oracle": _select_diagnostic_oracle_action,
    "related": _select_related_action,
    "unrelated": _select_unrelated_action,
}


def _effective_policy_config(
    config: RiskCommConfig,
    runtime: ScenarioRuntime,
    policy_name: str,
) -> RiskCommConfig:
    if policy_name not in {"unlimited", "unlimited-delivery"}:
        return config
    vehicle_capacity = max(len(runtime.replay.vehicle_ids), 1)
    return replace(
        config,
        uplink_slots_per_step=vehicle_capacity,
        downlink_slots_per_step=vehicle_capacity,
        relay_slots_per_step=vehicle_capacity,
    )


def _empty_metrics() -> dict[str, object]:
    return {
        "task_count": 0,
        "true_risk_count": 0,
        "correct_count": 0,
        "delivered_uplink_count": 0,
        "delivered_downlink_count": 0,
        "delivered_relay_count": 0,
        "invalid_action_count": 0,
        "zero_receive_vehicle_count": 0,
        "caoi_samples": [],
        "missed_count": 0,
        "false_positive_count": 0,
        "timeout_count": 0,
        "insufficient_count": 0,
        "abstained_count": 0,
        "communication_error_count": 0,
        "model_error_count": 0,
        "mixed_error_count": 0,
        "event_results": [],
        "early_send_records": [],
        "prediction_trigger_records": [],
        "prediction_extra_send_records": [],
        "prediction_extra_send_count": 0,
        "delivery_schedule": [],
        "early_send_count": 0,
        "invalid_early_send_count": 0,
    }


def _cap_action(action: RiskSchedulingAction, remaining: int) -> RiskSchedulingAction:
    if remaining < 0:
        raise ValueError("remaining delivery budget must be non-negative")
    total = len(action.uplinks) + len(action.downlinks) + len(action.relays)
    if total <= remaining:
        return action
    selected_downlinks = list(action.downlinks[:remaining])
    remaining -= len(selected_downlinks)
    selected_uplinks = list(action.uplinks[:remaining])
    remaining -= len(selected_uplinks)
    selected_relays = list(action.relays[:remaining])
    return RiskSchedulingAction(
        uplinks=tuple(selected_uplinks),
        downlinks=tuple(selected_downlinks),
        relays=tuple(selected_relays),
    )


def _classify_prediction_sends(metrics, delivery_schedule) -> None:
    """Compare changed selections against every actual vehicle reception path."""
    for record in metrics["prediction_trigger_records"]:
        baseline_step = min((item["send_step"] for item in delivery_schedule
                             if item["receiver_id"] == record["receiver_id"]
                             and item["object_id"] == record["object_id"]
                             and item["observation_step"] >= record["observation_step"]), default=None)
        record["counterfactual_send_step"] = baseline_step
        record["actual_earlier"] = baseline_step is not None and record["send_step"] < baseline_step
        record["additional_delivery"] = baseline_step is None
        if record["actual_earlier"]:
            metrics["early_send_records"].append(record)
        if record["additional_delivery"]:
            metrics["prediction_extra_send_records"].append(record)
        metrics["early_send_count"] += int(record["actual_earlier"])
        metrics["prediction_extra_send_count"] += int(record["additional_delivery"])


def run_policy(
    runtime: ScenarioRuntime,
    risk_config: RiskCommConfig,
    policy_name: str,
    max_steps: int | None = None,
    delivery_cap: int | None = None,
) -> dict[str, object]:
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative.")
    if delivery_cap is not None and delivery_cap < 0:
        raise ValueError("delivery_cap must be non-negative.")
    try:
        selector = _POLICIES[policy_name]
    except KeyError as error:
        known = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown risk policy {policy_name!r}; choose from {known}.") from error

    effective_config = _effective_policy_config(risk_config, runtime, policy_name)
    env = RiskCommEnv(runtime=runtime, risk_config=effective_config)
    observation = env.reset()
    metrics = _empty_metrics()
    metrics["task_scores"] = []
    metrics["reward_feedback"] = []
    step_limit = len(runtime.replay) if max_steps is None else min(max_steps, len(runtime.replay))
    steps_run = 0
    remaining_delivery_budget = delivery_cap
    while observation is not None and steps_run < step_limit:
        action = (
            select_diagnostic_oracle_action(
                observation, effective_config, env.catalog, replay=runtime.replay,
                communication=SnapshotCommunication(runtime.channel_model, runtime.config.resource),
            ) if policy_name == "diagnostic-oracle"
            else selector(observation, effective_config, ConflictCatalog((), (), ()))
        )
        if remaining_delivery_budget is not None:
            action = _cap_action(action, remaining_delivery_budget)
        triggered = []
        if effective_config.link_prediction and policy_name in {"receiver-value", "receiver-value-relay"}:
            no_prediction_config = replace(effective_config, link_prediction=False)
            no_prediction_action = (select_receiver_value_with_relay_action(observation, no_prediction_config)
                                    if policy_name == "receiver-value-relay"
                                    else select_receiver_value_action(observation, no_prediction_config))
            for candidate in action.downlinks:
                value = prediction_triggered_task(observation, candidate, effective_config)
                if value is not None and candidate not in no_prediction_action.downlinks:
                    snapshot = next(s for s in observation.rsu_snapshots if s.object_id == candidate.object_id)
                    triggered.append({
                        "task_id": value.task_id, "receiver_id": candidate.receiver_id,
                        "object_id": candidate.object_id, "observation_step": snapshot.observation_step,
                        "send_step": observation.step_index, "usable_step": observation.step_index + 1,
                        "decision_step": value.decision_step,
                        "reason": "window_closing_changed_selection",
                    })
        result = env.step(action)
        metrics["task_scores"].extend(asdict(score) for score in result.feedback.task_scores)
        metrics["reward_feedback"].append(asdict(result.feedback))
        if not result.invalid_action:
            metrics["prediction_trigger_records"].extend(triggered)
            metrics["delivery_schedule"].extend(
                {"link_type": delivery.link_type, "sender_id": delivery.sender_id,
                 "receiver_id": delivery.receiver_id, "object_id": delivery.snapshot.object_id,
                 "source_id": delivery.snapshot.source_id,
                 "observation_step": delivery.snapshot.observation_step,
                 "send_step": delivery.send_step, "usable_step": delivery.usable_step}
                for delivery in result.delivery_records
                if delivery.link_type in {"downlink", "relay"}
            )
        metrics["event_results"].extend(asdict(event) for event in result.event_results)
        metrics["task_count"] = int(metrics["task_count"]) + len(result.event_results)
        metrics["true_risk_count"] = int(metrics["true_risk_count"]) + sum(
            item.true_risk for item in result.event_results
        )
        metrics["correct_count"] = int(metrics["correct_count"]) + sum(
            item.correct for item in result.event_results
        )
        metrics["delivered_uplink_count"] = int(metrics["delivered_uplink_count"]) + len(
            result.delivered_uplinks
        )
        metrics["delivered_downlink_count"] = int(metrics["delivered_downlink_count"]) + len(
            result.delivered_downlinks
        )
        metrics["delivered_relay_count"] = int(metrics["delivered_relay_count"]) + len(
            result.delivered_relays
        )
        if remaining_delivery_budget is not None:
            remaining_delivery_budget -= (
                len(result.delivered_uplinks)
                + len(result.delivered_downlinks)
                + len(result.delivered_relays)
            )
        metrics["invalid_action_count"] = int(metrics["invalid_action_count"]) + int(
            result.invalid_action
        )
        metrics["zero_receive_vehicle_count"] = int(
            metrics["zero_receive_vehicle_count"]
        ) + result.zero_receive_vehicle_count
        if result.mean_caoi_steps is not None:
            caoi_samples = metrics["caoi_samples"]
            assert isinstance(caoi_samples, list)
            caoi_samples.append(result.mean_caoi_steps)
        for event in result.event_results:
            for field_name in (
                "missed",
                "false_positive",
                "timeout",
                "insufficient",
                "abstained",
                "communication_error",
                "model_error",
                "mixed_error",
            ):
                if getattr(event, field_name):
                    key = f"{field_name}_count"
                    metrics[key] = int(metrics[key]) + 1
        steps_run += 1
        observation = result.next_observation

    caoi_samples = metrics.pop("caoi_samples")
    assert isinstance(caoi_samples, list)
    metrics["steps_run"] = steps_run
    metrics["objective"] = env.objective_summary
    metrics["episode_terminated"] = observation is None
    metrics["episode_truncated"] = observation is not None
    metrics["communication_involved_count"] = int(metrics["communication_error_count"]) + int(metrics["mixed_error_count"])
    metrics["model_involved_count"] = int(metrics["model_error_count"]) + int(metrics["mixed_error_count"])
    metrics["effective_risk_config"] = asdict(effective_config)
    metrics["risk_model_version"] = model_version(effective_config)
    metrics["own_information_permission"] = own_information_permission(effective_config)
    metrics["model_event_area_radius_m"] = effective_config.conflict_cluster_radius_m
    events = metrics["event_results"]
    metrics["positive_abstention_count"] = sum(e["true_risk"] and e["abstained"] for e in events)
    metrics["negative_abstention_count"] = sum(not e["true_risk"] and e["abstained"] for e in events)
    metrics["true_positive_count"] = sum(e["true_risk"] and e["correct"] for e in events)
    metrics["true_negative_count"] = sum(not e["true_risk"] and e["correct"] for e in events)
    metrics["positive_with_related_prediction_count"] = sum(e["true_risk"] and any(
        o["true_conflict"] and o["predicted_risk"] for o in e["object_results"]) for e in events)
    metrics["positive_alarm_without_related_prediction_count"] = sum(e["true_risk"] and e["correct"]
        and not any(o["true_conflict"] and o["predicted_risk"] for o in e["object_results"]) for e in events)
    metrics["mean_caoi_steps"] = (
        sum(caoi_samples) / len(caoi_samples) if caoi_samples else None
    )
    metrics["mean_zero_receive_vehicle_count"] = (
        float(metrics["zero_receive_vehicle_count"]) / steps_run
        if steps_run
        else None
    )
    if effective_config.link_prediction and policy_name in {"receiver-value", "receiver-value-relay"}:
        baseline = run_policy(runtime, replace(risk_config, link_prediction=False), policy_name,
                              max_steps=max_steps, delivery_cap=delivery_cap)
        _classify_prediction_sends(metrics, baseline["delivery_schedule"])
    canonical = json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    metrics["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return metrics


def build_comparison_summary(
    config_path: str | Path,
    risk_config_path: str | Path,
    policy_names: tuple[str, ...],
    max_steps: int | None = None,
    link_prediction: bool | None = None,
    enable_v2v_relay: bool | None = None,
) -> dict[str, object]:
    runtime = load_runtime(config_path)
    risk_config = load_risk_config(risk_config_path)
    if link_prediction is not None:
        risk_config = replace(risk_config, link_prediction=link_prediction)
    if enable_v2v_relay is not None:
        risk_config = replace(risk_config, enable_v2v_relay=enable_v2v_relay)
    if not policy_names:
        raise ValueError("At least one policy is required.")
    unknown = sorted(set(policy_names) - set(_POLICIES))
    if unknown:
        raise ValueError(f"Unknown risk policies: {unknown}")

    step_limit = len(runtime.replay) if max_steps is None else min(max_steps, len(runtime.replay))
    common_task_count = sum(
        task.decision_step < step_limit
        for task in RiskCommEnv(runtime=runtime, risk_config=risk_config).catalog.tasks
    )
    policies = {
        name: run_policy(runtime, risk_config, name, max_steps=max_steps)
        for name in policy_names
    }
    summary: dict[str, object] = {
        "scenario_name": runtime.config.scenario_name,
        "scenario_config": {**asdict(runtime.config),
                            "trajectory_csv": str(runtime.config.trajectory_csv),
                            "building_csv": str(runtime.config.building_csv)},
        "seed": runtime.config.seed,
        "risk_config": asdict(risk_config),
        "risk_model_version": model_version(risk_config),
        "own_information_permission": own_information_permission(risk_config),
        "decision_rule": "crossing_heading_and_estimated_passage_gap_unknown_motion_abstains",
        "communication_rule": "channel_reachable_without_packet_or_resource_budgets",
        "caoi_version_rule": "last_data_plane_received_version",
        "error_cause_rule": "exclusive_communication_model_mixed",
        "calibration_rule": "diagnostic_only_no_validated_upper_bound",
        "relay_ranking_rule": "sender_version_summaries_motion_unknown",
        "step_count": step_limit,
        "link_prediction": risk_config.link_prediction,
        "enable_v2v_relay": risk_config.enable_v2v_relay,
        "common_task_count": common_task_count,
        "policies": policies,
    }
    if {"unlimited", "diagnostic-oracle"} <= set(policy_names):
        summary["slot_calibration"] = {
            "uplink_slots_per_step": risk_config.uplink_slots_per_step,
            "downlink_slots_per_step": risk_config.downlink_slots_per_step,
            "validated_upper_bound": False,
            "oracle_communication_misses": policies["diagnostic-oracle"]["communication_involved_count"],
            "unlimited_communication_misses": policies["unlimited"]["communication_involved_count"],
            "decision": "retain_configured_slots_without_automatic_calibration",
        }
    if {"related", "unrelated"} <= set(policy_names):
        related_count = int(policies["related"]["delivered_uplink_count"]) + int(
            policies["related"]["delivered_downlink_count"]
        )
        unrelated_count = int(policies["unrelated"]["delivered_uplink_count"]) + int(
            policies["unrelated"]["delivered_downlink_count"]
        )
        aligned_count = min(related_count, unrelated_count)
        aligned_related = run_policy(
            runtime,
            risk_config,
            "related",
            max_steps=max_steps,
            delivery_cap=aligned_count,
        )
        aligned_unrelated = run_policy(
            runtime,
            risk_config,
            "unrelated",
            max_steps=max_steps,
            delivery_cap=aligned_count,
        )
        related_caoi = aligned_related["mean_caoi_steps"]
        unrelated_caoi = aligned_unrelated["mean_caoi_steps"]
        caoi_gap = (
            abs(float(related_caoi) - float(unrelated_caoi))
            if related_caoi is not None and unrelated_caoi is not None
            else None
        )
        summary["evidence_1_aligned"] = {
            "aligned_delivery_count": aligned_count,
            "cause_comparison_rule": "communication_error_plus_mixed_error",
            "related": aligned_related,
            "unrelated": aligned_unrelated,
            "caoi_gap": caoi_gap,
            "communication_error_gap": abs(
                int(aligned_related["communication_involved_count"])
                - int(aligned_unrelated["communication_involved_count"])
            ),
            "leakage_differs": (
                aligned_related["communication_involved_count"]
                != aligned_unrelated["communication_involved_count"]
            ),
        }
    evidence_policy_names = {"age", "event-trigger", "deadline", "receiver-value"}
    if evidence_policy_names <= set(policy_names):
        evidence_2_policies = {
            name: policies[name] for name in sorted(evidence_policy_names)
        }
        receiver_errors = int(policies["receiver-value"]["communication_involved_count"])
        comparator_errors = [
            int(policies[name]["communication_involved_count"])
            for name in ("age", "event-trigger", "deadline")
        ]
        summary["evidence_2"] = {
            **evidence_2_policies,
            "policies": evidence_2_policies,
            "receiver_value_lower_communication_error": receiver_errors
            < min(comparator_errors),
            "model_error_range_rule": "model_error_plus_mixed_error",
            "model_error_range": [
                min(int(metrics["model_involved_count"]) for metrics in evidence_2_policies.values()),
                max(int(metrics["model_involved_count"]) for metrics in evidence_2_policies.values()),
            ],
        }
    return summary


def add_ablation_comparisons(
    summary: dict[str, object],
    config_path: str | Path,
    risk_config_path: str | Path,
    policy_names: tuple[str, ...],
    max_steps: int | None,
    *,
    compare_link_prediction: bool,
    compare_relay: bool,
) -> dict[str, object]:
    if compare_link_prediction:
        summary["prediction_consuming_policies"] = [p for p in policy_names if p in {"receiver-value", "receiver-value-relay"}]
        summary["link_prediction_comparison"] = {
            "false": build_comparison_summary(
                config_path,
                risk_config_path,
                policy_names,
                max_steps=max_steps,
                link_prediction=False,
                enable_v2v_relay=bool(summary["enable_v2v_relay"]),
            )["policies"],
            "true": build_comparison_summary(
                config_path,
                risk_config_path,
                policy_names,
                max_steps=max_steps,
                link_prediction=True,
                enable_v2v_relay=bool(summary["enable_v2v_relay"]),
            )["policies"],
        }
    if compare_relay:
        summary["relay_consuming_policies"] = [p for p in policy_names if p == "receiver-value-relay"]
        summary["relay_comparison"] = {
            "false": build_comparison_summary(
                config_path,
                risk_config_path,
                policy_names,
                max_steps=max_steps,
                link_prediction=bool(summary["link_prediction"]),
                enable_v2v_relay=False,
            )["policies"],
            "true": build_comparison_summary(
                config_path,
                risk_config_path,
                policy_names,
                max_steps=max_steps,
                link_prediction=bool(summary["link_prediction"]),
                enable_v2v_relay=True,
            )["policies"],
        }
    return summary


def main() -> int:
    args = parse_args()
    policy_names = tuple(
        name.strip() for name in args.policies.split(",") if name.strip()
    )
    try:
        summary = build_comparison_summary(
            config_path=args.config,
            risk_config_path=args.risk_config,
            policy_names=policy_names,
            max_steps=args.steps,
            link_prediction=args.link_prediction,
            enable_v2v_relay=args.enable_v2v_relay,
        )
        summary = add_ablation_comparisons(
            summary,
            args.config,
            args.risk_config,
            policy_names,
            args.steps,
            compare_link_prediction=args.compare_link_prediction,
            compare_relay=args.compare_relay,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"scenario={summary['scenario_name']} seed={summary['seed']}")
        for name, metrics in summary["policies"].items():
            assert isinstance(metrics, dict)
            print(
                f"{name}: tasks={metrics['task_count']} "
                f"downlinks={metrics['delivered_downlink_count']} "
                f"caoi={metrics['mean_caoi_steps']} "
                f"missed={metrics['missed_count']} "
                f"communication_error={metrics['communication_error_count']} "
                f"objective={json.dumps(metrics['objective'], ensure_ascii=False, sort_keys=True)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
