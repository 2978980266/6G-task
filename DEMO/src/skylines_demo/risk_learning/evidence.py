"""离线证据分支，可保存完整真值诊断，绝不回接网络。"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from math import isclose
from pathlib import Path
from typing import Any, Callable

from ..config import RiskCommConfig
from ..env.risk_comm_env import RiskCommEnv, RiskStepResult
from ..risk.scoring import OBJECTIVE_WEIGHTS
from ..types import RiskObservation, RiskSchedulingAction
from .actions import append_choice, initial_state, legal_mask, to_environment_action
from .domain import build_safe_step
from .encoding import encode_step
from .policy import ActionPath


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


class ExclusiveWriter:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError(f"输出已存在：{self.directory}")
        self.directory.mkdir(parents=True, exist_ok=False)

    def json(self, name: str, value: Any) -> None:
        with (self.directory / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)

    def jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.directory / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def apply_global_cap(proposal: RiskSchedulingAction, remaining: int) -> RiskSchedulingAction:
    if type(remaining) is not int or remaining < 0:
        raise ValueError("cap剩余必须是非负整数。")
    downlinks = proposal.downlinks[:remaining]
    remaining -= len(downlinks)
    relays = proposal.relays[:remaining]
    remaining -= len(relays)
    return RiskSchedulingAction(proposal.uplinks[:remaining], downlinks, relays)


def cost_match_label(before_m: int, after_m: int, cap: int) -> str:
    if before_m < 0 or after_m < 0 or cap < max(before_m, after_m):
        raise ValueError("实际发送超过声明cap。")
    if before_m == after_m == 0:
        return "zero_send_control"
    return "equal_actual_cost" if before_m == after_m else "budget_constraint_control"


def action_from_path(step: Any, path: ActionPath) -> RiskSchedulingAction:
    state = initial_state(step)
    for index in path.indices:
        state = append_choice(step, state, index)
    return to_environment_action(step, state)


def step_payload(result: RiskStepResult) -> dict[str, Any]:
    return dict(observation=asdict(result.observation), action={**asdict(result.action), "invalid": result.invalid_action},
                event_results=[asdict(e) for e in result.event_results],
                risk_decisions=[{"task_id": key, "decision": asdict(d)} for key, d in result.risk_decisions],
                task_scores=[asdict(s) for s in result.feedback.task_scores],
                reward_feedback=asdict(result.feedback), reward=result.reward, objective=result.objective,
                delivery_records=[asdict(r) for r in result.delivery_records],
                caoi=dict(mean_caoi_steps=result.mean_caoi_steps, zero_receive_vehicle_count=result.zero_receive_vehicle_count),
                next_observation=asdict(result.next_observation) if result.next_observation else None)


def receipt_chain(record: dict[str, Any], deliveries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshot = record["snapshot"]
    if record["sender_id"] == snapshot["source_id"]:
        return [record]
    previous = [r for r in deliveries if r["receiver_id"] == record["sender_id"]
                and r["usable_step"] <= record["send_step"]
                and all(r["snapshot"][k] == snapshot[k] for k in ("object_id", "source_id", "observation_step"))]
    if not previous:
        raise AssertionError("中继缺少此前实际可用的原始版本。")
    earlier = max(previous, key=lambda r: r["usable_step"])
    for key in ("x", "y", "z", "speed", "heading_deg", "velocity_x", "velocity_z", "velocity_basis",
                "velocity_previous_observation_step"):
        assert earlier["snapshot"][key] == snapshot[key], "原始状态被改写。"
    return receipt_chain(earlier, deliveries) + [record]


def verify_episode(run: dict[str, Any]) -> None:
    steps, scores, objective = run["steps"], run["task_scores"], run["objective"]
    assert len(steps) == 23 and objective["evaluation_complete"]
    assert len(scores) == len({s["task_id"] for s in scores}) == 46
    assert objective["settled_task_count"] == 46 and objective["pending_task_count"] == 0
    assert isclose(sum(s["reward"] for s in steps), -objective["objective_sum"])
    deliveries = [r for s in steps for r in s["delivery_records"]]
    assert len(deliveries) == objective["send_count"]
    for row in steps:
        assert not row["action"]["invalid"]
        assert len(row["action"]["uplinks"]) <= 2 and len(row["action"]["downlinks"]) <= 2
        assert len(row["action"]["relays"]) <= int(run["risk_config"]["enable_v2v_relay"])
        endpoints = ([a["sender_id"] for a in row["action"]["uplinks"]]
                     + [a["receiver_id"] for a in row["action"]["downlinks"]]
                     + [n for a in row["action"]["relays"] for n in (a["sender_id"], a["receiver_id"])])
        assert len(endpoints) == len(set(endpoints))
    for record in deliveries:
        assert record["usable_step"] == record["send_step"] + 1
        assert record["snapshot"]["observation_step"] <= record["send_step"]
        assert record["receiver_id"] != record["snapshot"]["object_id"]
        receipt_chain(record, deliveries)
    assert objective["initial_task_loss_sum"] == 5.


def finish_episode(steps: list[dict[str, Any]], config: RiskCommConfig,
                   objective: dict[str, Any]) -> dict[str, Any]:
    events = [e for s in steps for e in s["event_results"]]
    scores = [e for s in steps for e in s["task_scores"]]
    legacy = {key + "_count": sum(e[key] for e in events) for key in
              ("correct", "missed", "false_positive", "timeout", "insufficient", "abstained",
               "communication_error", "model_error", "mixed_error")}
    legacy["communication_involved_count"] = legacy["communication_error_count"] + legacy["mixed_error_count"]
    legacy["model_involved_count"] = legacy["model_error_count"] + legacy["mixed_error_count"]
    ages = [s["caoi"]["mean_caoi_steps"] for s in steps if s["caoi"]["mean_caoi_steps"] is not None]
    legacy["mean_caoi_steps"] = sum(ages) / len(ages) if ages else None
    legacy["mean_zero_receive_vehicle_count"] = sum(s["caoi"]["zero_receive_vehicle_count"] for s in steps) / len(steps)
    run = dict(steps=steps, event_results=events, task_scores=scores, objective=objective,
               legacy=legacy, risk_config=asdict(config), episode_terminated=True,
               episode_truncated=False, invalid_action_count=0,
               strict_failure_evidence=[dict(score=s, event=next(e for e in events if e["task_id"] == s["task_id"]))
                                        for s in scores if s["task_id"] in {"risk-10697-11-1", "risk-4966-5-2"}])
    verify_episode(run)
    return run


def capture_episode(env: RiskCommEnv, selector: Callable[[RiskObservation], RiskSchedulingAction],
                    config: RiskCommConfig, writer: ExclusiveWriter | None = None,
                    global_cap: int | None = None) -> dict[str, Any]:
    obs: RiskObservation | None = env.reset()
    rows = []
    remaining = global_cap
    while obs is not None:
        step = build_safe_step(obs, config)
        proposal = selector(obs)
        action = apply_global_cap(proposal, remaining) if remaining is not None else proposal
        result = env.step(action)
        if result.invalid_action:
            raise AssertionError("正式运行非法动作，停止。")
        row = step_payload(result)
        row["safe_input"] = asdict(step)
        row["encoded_input"] = encoded_payload(encode_step(step))
        state = initial_state(step)
        indices, masks, prefixes = [], [], []
        ordered = (*proposal.relays, *proposal.downlinks, *proposal.uplinks)
        for message in (*ordered, None):
            index = next((i for i, c in enumerate(step.candidates, 1) if c.action == message),
                         0 if message is None else -1)
            prefixes.append(state.selected_indices)
            masks.append(legal_mask(step, state).tolist())
            indices.append(index)
            state = append_choice(step, state, index)
        row["selection_evidence"] = dict(indices=indices, masks=masks, prefixes=prefixes,
                                          domain_filtered_count=sum(step.raw_counts) - len(step.candidates),
                                          filter_reasons=dict(D="no_future_task_or_no_fixed_model_risk",
                                                              R="relay_off_or_no_future_task_or_no_newer_version",
                                                              U="unconfirmed_current_local_or_self_object"))
        row["proposal_before_cap"] = asdict(proposal)
        row["remaining_cap"] = remaining
        rows.append(row)
        if remaining is not None:
            remaining -= result.feedback.send_count
        obs = result.next_observation
    run = finish_episode(rows, config, env.objective_summary)
    if writer:
        writer.jsonl("steps.jsonl", rows)
        writer.json("run.json", run)
    return run


def decision_receipts(run: dict[str, Any], event: dict[str, Any]) -> list[dict[str, Any]]:
    deliveries = [r for s in run["steps"] for r in s["delivery_records"]]
    diagnostics = {o["object_id"]: o["diagnostic"] for o in event["object_results"] if o["diagnostic"]}
    rows = []
    for r in deliveries:
        if r["receiver_id"] != event["receiver_id"] or r["usable_step"] > event["decision_step"]:
            continue
        d = diagnostics.get(r["snapshot"]["object_id"])
        adopted = bool(d and d["input_kind"] == "received" and d["source_id"] == r["snapshot"]["source_id"]
                       and d["observation_step"] == r["snapshot"]["observation_step"]
                       and d["received_step"] == r["usable_step"])
        support = bool(d and d["snapshot"]["velocity_basis"] == "received_history"
                       and d["snapshot"]["velocity_previous_observation_step"] == r["snapshot"]["observation_step"])
        chain = receipt_chain(r, deliveries)
        rows.append(dict(receipt=r, adopted_at_decision=adopted, possible_local_velocity_support=support,
                         hop_chain=chain, path_send_count=len(chain), path_cost=OBJECTIVE_WEIGHTS.send * len(chain)))
    return rows


def pair_episodes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old_events = {e["task_id"]: e for e in before["event_results"]}
    old_scores = {s["task_id"]: s for s in before["task_scores"]}
    new_scores = {s["task_id"]: s for s in after["task_scores"]}
    if old_events.keys() != new_scores.keys() or len(new_scores) != 46:
        raise ValueError("配对必须覆盖同一完整46任务。")
    rows = []
    for event in after["event_results"]:
        key = event["task_id"]
        old, b, a = old_events[key], old_scores[key], new_scores[key]
        recovered = old["missed"] and not event["missed"]
        rows.append(dict(task_id=key, before_score=b, after_score=a,
                         before_object_results=old["object_results"], after_object_results=event["object_results"],
                         true_object_recovered=a["positive_success"] and not b["positive_success"],
                         true_object_success_lost=b["positive_success"] and not a["positive_success"],
                         task_label_recovery=recovered,
                         wrong_object_only_label_recovery=recovered and a["wrong_object_only_alarm"],
                         negative_abstention_removed=not event["true_risk"] and old["abstained"] and not event["abstained"],
                         negative_abstention_added=not event["true_risk"] and not old["abstained"] and event["abstained"],
                         false_positive_to_abstention=old["false_positive"] and event["abstained"],
                         extra_alarm_objects_added=sorted(set(a["extra_alarm_object_ids"]) - set(b["extra_alarm_object_ids"])),
                         extra_alarm_objects_removed=sorted(set(b["extra_alarm_object_ids"]) - set(a["extra_alarm_object_ids"])),
                         loss_delta=a["task_loss"] - b["task_loss"],
                         before_receipts=decision_receipts(before, old), after_receipts=decision_receipts(after, event)))
    b, a = before["objective"], after["objective"]
    return dict(tasks=rows, task_loss_delta=a["task_loss_sum"] - b["task_loss_sum"],
                objective_delta=a["objective_sum"] - b["objective_sum"],
                send_delta=a["send_count"] - b["send_count"],
                actual_equal_send_count=a["send_count"] == b["send_count"],
                cost_attribution="共享发送按实际跳计一次；逐任务路径费用不可跨任务相加。")


def capture_cost_matched_pair(env_factory: Callable[[], RiskCommEnv], config: RiskCommConfig,
                             before_selector_factory: Callable[[], Callable[[RiskObservation], RiskSchedulingAction]],
                             after_selector_factory: Callable[[], Callable[[RiskObservation], RiskSchedulingAction]],
                             writer: ExclusiveWriter, *, after_env_factory: Callable[[], RiskCommEnv] | None = None,
                             after_config: RiskCommConfig | None = None) -> dict[str, Any]:
    """只按未封顶实际M取cap，全部四回放留档；选择器工厂保证采样状态重新初始化。"""
    right_config = after_config if after_config is not None else config
    if replace(config, enable_v2v_relay=right_config.enable_v2v_relay) != right_config:
        raise ValueError("费用匹配只允许同模式/域，配置差异仅限显式中继开关。")
    factories = (env_factory, after_env_factory if after_env_factory is not None else env_factory)
    selectors = (before_selector_factory, after_selector_factory)
    configs = (config, right_config)
    # 先检查全组目标；不调用环境或选择器工厂、不读取收益来挑cap。
    names = ("uncapped_before", "uncapped_after", "capped_before", "capped_after")
    if any((writer.directory / name).exists() for name in (*names, "cost_match.json")):
        raise FileExistsError("费用匹配目标已存在。")
    uncapped = [capture_episode(factory(), selector(), cfg, ExclusiveWriter(writer.directory / name))
                for factory, selector, cfg, name in zip(factories, selectors, configs, names[:2])]
    sends = [r["objective"]["send_count"] for r in uncapped]
    cap = min(sends)
    capped = [capture_episode(factory(), selector(), cfg, ExclusiveWriter(writer.directory / name), cap)
              for factory, selector, cfg, name in zip(factories, selectors, configs, names[2:])]
    actual = [r["objective"]["send_count"] for r in capped]
    result = dict(cap=cap, cap_basis="min_uncapped_actual_M_only", uncapped_actual_sends=sends,
                  capped_actual_sends=actual, label=cost_match_label(actual[0], actual[1], cap),
                  uncapped_pair=pair_episodes(*uncapped), capped_pair=pair_episodes(*capped),
                  limitation="D/R/U截取是补充启发式；相同cap不保证等实际费用，不补无用发送。")
    writer.json("cost_match.json", result)
    return result


def encoded_payload(encoded: Any) -> dict[str, Any]:
    return dict(candidate_features=encoded.candidate_features.tolist(), global_features=encoded.global_features.tolist(),
                keys=encoded.keys, padding_mask=encoded.padding_mask.tolist())
