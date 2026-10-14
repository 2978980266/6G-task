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
from skylines_demo.config import RiskCommConfig, load_risk_config
from skylines_demo.risk.model import model_version, own_information_permission
from skylines_demo.env.risk_comm_env import RiskCommEnv, RiskStepResult
from skylines_demo.policies.risk_baselines import (
    select_age_action,
    select_deadline_action,
    select_event_trigger_action,
)
from skylines_demo.policies.risk_receiver_policy import select_receiver_value_action
from skylines_demo.policies.risk_relay_policy import (
    select_receiver_value_with_relay_action,
)
from skylines_demo.types import RiskObservation, RiskSchedulingAction

PolicySelector = Callable[[RiskObservation, RiskCommConfig], RiskSchedulingAction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace one risk communication episode.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG_PATH))
    parser.add_argument(
        "--policy",
        choices=("no-communication", "age", "event-trigger", "deadline", "receiver-value", "receiver-value-relay"),
        default="receiver-value",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--link-prediction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-v2v-relay", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=None)
    return parser.parse_args()


def _snapshot_payload(snapshot: Any) -> dict[str, object]:
    return asdict(snapshot)


def _observation_payload(observation: RiskObservation) -> dict[str, object]:
    return {
        "step_index": observation.step_index,
        "simulation_frame": observation.simulation_frame,
        "elapsed_seconds": observation.elapsed_seconds,
        "delta_seconds": observation.delta_seconds,
        "rsu_known_object_ids": observation.rsu_known_object_ids,
        "rsu_visible_object_ids": observation.rsu_visible_object_ids,
        "rsu_snapshots": [
            _snapshot_payload(snapshot) for snapshot in observation.rsu_snapshots
        ],
        "receiver_known_observation_steps": observation.receiver_known_observation_steps,
        "last_sent_snapshots": [
            {"receiver_id": receiver_id, "snapshot": _snapshot_payload(snapshot)}
            for receiver_id, snapshot in observation.last_sent_snapshots
        ],
        "active_vehicles": [
            {
                "vehicle_id": state.vehicle_id,
                "position": asdict(state.snapshot.position),
                "visible_object_ids": state.visible_object_ids,
                "known_object_ids": state.known_object_ids,
                "data_plane_object_ids": state.data_plane_object_ids,
                "caoi_steps": state.caoi_steps,
                "self_snapshot": asdict(state.self_snapshot) if state.self_snapshot else None,
            }
            for state in observation.active_vehicles
        ],
        "visible_relations": [
            {"observer_id": 0, "object_ids": observation.rsu_visible_object_ids}
        ]
        + [
            {"observer_id": state.vehicle_id, "object_ids": state.visible_object_ids}
            for state in observation.active_vehicles
        ],
        "tasks": [asdict(task) for task in observation.tasks],
        "candidates": {
            "uplinks": [asdict(candidate) for candidate in observation.uplink_candidates],
            "downlinks": [asdict(candidate) for candidate in observation.downlink_candidates],
            "relays": [asdict(candidate) for candidate in observation.relay_candidates],
        },
        "link_forecasts": [
            {
                "receiver_id": receiver_id,
                "object_id": object_id,
                "forecast": asdict(forecast),
            }
            for receiver_id, object_id, forecast in observation.link_forecasts
        ],
    }


def _compact_next_observation(observation: RiskObservation | None) -> object:
    if observation is None:
        return None
    return {
        "step_index": observation.step_index,
        "elapsed_seconds": observation.elapsed_seconds,
        "rsu_known_object_ids": observation.rsu_known_object_ids,
        "rsu_visible_object_ids": observation.rsu_visible_object_ids,
        "receiver_known_observation_steps": observation.receiver_known_observation_steps,
        "vehicles": [
            {
                "vehicle_id": state.vehicle_id,
                "known_object_ids": state.known_object_ids,
                "data_plane_object_ids": state.data_plane_object_ids,
                "caoi_steps": state.caoi_steps,
            }
            for state in observation.active_vehicles
        ],
    }


def _action_payload(action: RiskSchedulingAction) -> dict[str, object]:
    return {
        "uplinks": [asdict(item) for item in action.uplinks],
        "downlinks": [asdict(item) for item in action.downlinks],
        "relays": [asdict(item) for item in action.relays],
    }


def _step_payload(result: RiskStepResult) -> dict[str, object]:
    return {
        "observation": _observation_payload(result.observation),
        "control_plane": {
            "rsu_known_object_ids": result.observation.rsu_known_object_ids,
            "tasks": [asdict(task) for task in result.observation.tasks],
        },
        "risk_decisions": [
            {"task_id": task_id, "decision": asdict(decision)}
            for task_id, decision in result.risk_decisions
        ],
        "event_results": [asdict(event) for event in result.event_results],
        "task_scores": [asdict(score) for score in result.feedback.task_scores],
        "reward_feedback": asdict(result.feedback),
        "reward": result.reward,
        "objective": result.objective,
        "delivery_records": [asdict(record) for record in result.delivery_records],
        "action": {
            **_action_payload(result.action),
            "invalid": result.invalid_action,
        },
        "delivery": {
            "uplinks": [_snapshot_payload(item) for item in result.delivered_uplinks],
            "downlinks": [_snapshot_payload(item) for item in result.delivered_downlinks],
            "relays": [_snapshot_payload(item) for item in result.delivered_relays],
        },
        "caoi": {
            "mean_caoi_steps": result.mean_caoi_steps,
            "zero_receive_vehicle_count": result.zero_receive_vehicle_count,
        },
        "resource_use": {
            "uplink_slots": len(result.delivered_uplinks),
            "downlink_slots": len(result.delivered_downlinks),
            "relay_slots": len(result.delivered_relays),
        },
        "next_observation": _compact_next_observation(result.next_observation),
    }


def _selector(policy_name: str) -> PolicySelector:
    selectors: dict[str, PolicySelector] = {
        "no-communication": lambda observation, config: RiskSchedulingAction(),
        "age": select_age_action,
        "event-trigger": select_event_trigger_action,
        "deadline": select_deadline_action,
        "receiver-value": select_receiver_value_action,
        "receiver-value-relay": select_receiver_value_with_relay_action,
    }
    return selectors[policy_name]


def build_trace(
    config_path: str | Path,
    risk_config_path: str | Path,
    policy_name: str,
    max_steps: int | None,
    link_prediction: bool | None = None,
    enable_v2v_relay: bool | None = None,
) -> dict[str, object]:
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative.")
    runtime = load_runtime(config_path)
    risk_config = load_risk_config(risk_config_path)
    if link_prediction is not None:
        risk_config = replace(risk_config, link_prediction=link_prediction)
    if enable_v2v_relay is not None:
        risk_config = replace(risk_config, enable_v2v_relay=enable_v2v_relay)
    env = RiskCommEnv(runtime=runtime, risk_config=risk_config)
    observation = env.reset()
    selector = _selector(policy_name)
    step_limit = len(runtime.replay) if max_steps is None else min(max_steps, len(runtime.replay))
    steps: list[dict[str, object]] = []
    while observation is not None and len(steps) < step_limit:
        action = selector(observation, risk_config)
        result = env.step(action)
        steps.append(_step_payload(result))
        observation = result.next_observation
    return {
        "scenario_name": runtime.config.scenario_name,
        "scenario_config": {**asdict(runtime.config),
                            "trajectory_csv": str(runtime.config.trajectory_csv),
                            "building_csv": str(runtime.config.building_csv)},
        "seed": runtime.config.seed,
        "policy": policy_name,
        "link_prediction": risk_config.link_prediction,
        "enable_v2v_relay": risk_config.enable_v2v_relay,
        "risk_config": asdict(risk_config),
        "risk_model_version": model_version(risk_config),
        "own_information_permission": own_information_permission(risk_config),
        "communication_rule": "channel_reachable_without_packet_or_resource_budgets",
        "caoi_version_rule": "last_data_plane_received_version",
        "error_cause_rule": "exclusive_communication_model_mixed",
        "relay_ranking_rule": "sender_version_summaries_motion_unknown",
        "objective": env.objective_summary,
        "episode_terminated": observation is None,
        "episode_truncated": observation is not None,
        "steps": steps,
    }


def main() -> int:
    args = parse_args()
    if args.jsonl is not None and args.jsonl.exists():
        print(f"Refusing to overwrite existing trace: {args.jsonl}", file=sys.stderr)
        return 2
    if args.steps is not None and args.steps < 0:
        print("--steps must be non-negative.", file=sys.stderr)
        return 2
    trace = build_trace(
        config_path=args.config,
        risk_config_path=args.risk_config,
        policy_name=args.policy,
        max_steps=args.steps,
        link_prediction=args.link_prediction,
        enable_v2v_relay=args.enable_v2v_relay,
    )
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.jsonl.open("x", encoding="utf-8") as handle:
                for step in trace["steps"]:
                    handle.write(json.dumps(step, ensure_ascii=False, sort_keys=True) + "\n")
        except FileExistsError:
            print(f"Refusing to overwrite existing trace: {args.jsonl}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(trace, ensure_ascii=False, sort_keys=True))
    elif args.jsonl is not None:
        print(f"wrote={args.jsonl} steps={len(trace['steps'])}")
    else:
        for step in trace["steps"]:
            observation = step["observation"]
            action = step["action"]
            assert isinstance(observation, dict)
            assert isinstance(action, dict)
            feedback = step["reward_feedback"]
            print(
                f"step={observation['step_index']} "
                f"uplinks={len(action['uplinks'])} "
                f"downlinks={len(action['downlinks'])} "
                f"relays={len(action['relays'])} "
                f"invalid={action['invalid']} "
                f"task_loss={feedback['task_loss_sum']} "
                f"send_cost={feedback['send_cost_sum']} "
                f"objective={feedback['objective_sum']} "
                f"reward={step['reward']}"
            )
        print(f"objective_summary={json.dumps(trace['objective'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
