"""固定场景的原环境分支、完整动作枚举及可证明的乐观松弛。"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict
import itertools
import pickle
from typing import Iterator

from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.perception.snapshots import build_snapshot
from skylines_demo.risk.evaluator import evaluate_decision
from skylines_demo.risk.model import predict_pair
from skylines_demo.risk.scoring import score_event
from skylines_demo.types import RiskSchedulingAction, UplinkAction

EMPTY = RiskSchedulingAction()


def clone_env(env: RiskCommEnv) -> RiskCommEnv:
    """只共享静态场景和不可变值；所有可写容器独立。仍调用原 step。"""
    other = copy(env)

    def cache(c):
        result = copy(c)
        result._local_snapshots = c._local_snapshots.copy()
        result._received_snapshots = c._received_snapshots.copy()
        return result

    other._rsu_knowledge = cache(env._rsu_knowledge)
    other._vehicle_knowledge = {k: cache(v) for k, v in env._vehicle_knowledge.items()}
    for name in ("_local_history", "_self_history", "_self_snapshots", "_current_local_snapshots",
                 "_last_sent_snapshots"):
        setattr(other, name, getattr(env, name).copy())
    other._score_ledger = copy(env._score_ledger)
    other._score_ledger._scores = env._score_ledger._scores.copy()
    other._score_ledger._feedback = env._score_ledger._feedback.copy()
    other._send_history = env._send_history.copy()
    return other


def cache_payload(env):
    def item(cache):
        return {"owner_id": cache.owner_id,
                "local": {k: asdict(v) for k, v in sorted(cache._local_snapshots.items())},
                "received": {k: asdict(v) for k, v in sorted(cache._received_snapshots.items())}}
    return {"rsu": item(env._rsu_knowledge),
            "vehicles": {k: item(v) for k, v in sorted(env._vehicle_knowledge.items())},
            "local_history": [(key, asdict(value[0]), value[1])
                              for key, value in sorted(env._local_history.items())],
            "self_history": [(key, asdict(value[0]), value[1])
                             for key, value in sorted(env._self_history.items())]}


def state_key(env: RiskCommEnv) -> bytes:
    # 保守地保留完整反馈和发送历史，不凭版本、较新或来源假设合并。
    def cache(c):
        return c.owner_id, tuple(sorted(c._local_snapshots.items())), tuple(sorted(c._received_snapshots.items()))
    fields = (env._step_index, env._current_observation, env._current_decisions,
              cache(env._rsu_knowledge), tuple((k, cache(v)) for k, v in sorted(env._vehicle_knowledge.items())),
              tuple(sorted(env._local_history.items())), tuple(sorted(env._self_history.items())),
              tuple(sorted(env._self_snapshots.items())), tuple(sorted(env._current_local_snapshots.items())),
              tuple(sorted(env._last_sent_snapshots.items())), tuple(env._score_ledger._scores.items()),
              tuple(env._score_ledger._feedback), tuple(env._send_history))
    return pickle.dumps(fields, protocol=5)


def endpoints(message):
    return frozenset(getattr(message, name) for name in ("sender_id", "receiver_id")
                     if getattr(message, name, 0) != 0)


def legal_actions(obs, cfg) -> Iterator[RiskSchedulingAction]:
    """不截候选，保留各类型内部所有顺序，包括同对象双上报的先到覆盖差异。"""
    pools = (obs.uplink_candidates, obs.downlink_candidates,
             obs.relay_candidates if cfg.enable_v2v_relay else ())
    slots = (cfg.uplink_slots_per_step, cfg.downlink_slots_per_step,
             cfg.relay_slots_per_step if cfg.enable_v2v_relay else 0)

    def group(pool, maximum, occupied):
        yield (), occupied
        if not maximum:
            return
        for c in pool:
            nodes = endpoints(c)
            if nodes & occupied:
                continue
            if (getattr(c, "receiver_id", None) == c.object_id
                    or isinstance(c, UplinkAction) and c.sender_id == c.object_id):
                continue
            if maximum:
                for tail, used in group(pool, maximum - 1, occupied | nodes):
                    yield (c, *tail), used

    for u, occupied_u in group(pools[0], slots[0], frozenset()):
        for d, occupied_d in group(pools[1], slots[1], occupied_u):
            for r, _ in group(pools[2], slots[2], occupied_d):
                yield RiskSchedulingAction(u, d, r)


def repair(action, obs, cfg):
    used = set()
    selected = []
    for proposal, pool, maximum in zip((action.uplinks, action.downlinks, action.relays),
                                      (obs.uplink_candidates, obs.downlink_candidates, obs.relay_candidates),
                                      (cfg.uplink_slots_per_step, cfg.downlink_slots_per_step,
                                       cfg.relay_slots_per_step if cfg.enable_v2v_relay else 0)):
        rows = []
        available = set(pool)
        for c in proposal:
            nodes = endpoints(c)
            if c in available and not used & nodes and len(rows) < maximum:
                rows.append(c)
                used.update(nodes)
        selected.append(tuple(rows))
    return RiskSchedulingAction(*selected)


def rollout(env, plan, *, strict=True, from_current=False, keep_states=True, limits=None):
    e = clone_env(env)
    if not from_current:
        e.reset()
    states, results, actual = [], [], []
    for proposed in plan:
        if limits:
            limits.tick()
        obs = e.current_observation
        if obs is None:
            raise ValueError("动作数超过真实回合。")
        if keep_states:
            states.append(clone_env(e))
        a = proposed if strict else repair(proposed, obs, e._risk_config)
        result = e.step(a)
        if result.invalid_action:
            raise AssertionError(f"非法 oracle 动作，step={obs.step_index}")
        results.append(result)
        actual.append(a)
    if e.current_observation is not None:
        raise ValueError("方案没有完整覆盖真实回合。")
    return {"objective": e.objective_summary, "plan": actual, "states": states,
            "results": results, "final_env": e,
            "task_scores": [asdict(s) for s in e._score_ledger._scores.values()]}


class VisibilityTable:
    def __init__(self, env):
        self.table = {(frame.step_index, owner): env._sensor.visible_object_ids(owner, frame)
                      for frame in env._replay.frames for owner in (0, *frame.vehicles)}

    def visible_object_ids(self, owner, frame):
        return self.table[frame.step_index, owner]


class ContactTable:
    def __init__(self, env):
        self.table = {(frame.step_index, a, b): env._communication.reachable_nodes(
            frame, env._runtime.config.rsu.position, a, b)
            for frame in env._replay.frames for a in (0, *frame.vehicles)
            for b in (0, *frame.vehicles) if a != b}

    def reachable_nodes(self, frame, rsu_position, sender, receiver):
        return self.table.get((frame.step_index, sender, receiver), False)


def accelerate(env):
    """只查表加速固定几何，不替换缓存、估速、执行、风险模型和评分。"""
    if env._risk_config.link_prediction:
        raise ValueError("本实验要求预测关闭。")
    e = clone_env(env)
    e._sensor = VisibilityTable(env)
    e._communication = ContactTable(env)
    e.reset()
    return e


def relaxed_bound(env):
    """路径/容量放宽、每对象独立选版本/估速依据的任务损失下界。

    每个真实快照都在某次本地可见时生成。此前位置只能来自固定的上次
    本地可见帧，或更晚且严格早于生成帧的实际接收；此处允许所有这样的
    帧，不检查其是否真的能收到。过往本地快照也允许免费保留/转发。
    截止时仍本地可见的对象不可删除。选择之间取消资源/路径/缓存耦合，
    所得集合包含全部真实可达的判断输入，因此只用于 LB，不用于投递。
    """
    e = clone_env(env)
    e.reset()
    frames, cfg = e._replay.frames, e._risk_config
    visibility = VisibilityTable(e)
    generated = {}
    history = {}
    for frame in frames:
        t = frame.step_index
        for owner in (0, *frame.vehicles):
            for obj in visibility.visible_object_ids(owner, frame):
                p = history.get((owner, obj))
                previous = ([None] if p is None else [p]) + list(range(0 if p is None else p+1, t))
                variants = []
                for k in previous:
                    if k is not None and obj not in frames[k].vehicles:
                        continue
                    variants.append(build_snapshot(obj, owner, t, frame.vehicles[obj],
                        frames[k].vehicles[obj] if k is not None else None, frame.elapsed_seconds,
                        frames[k].elapsed_seconds if k is not None else None,
                        previous_observation_step=k,
                        velocity_basis="local_history" if k == p else "received_history"))
                generated[owner, obj, t] = tuple(variants)
                history[owner, obj] = t
    rows = []
    while e.current_observation is not None:
        obs = e.current_observation
        for task in e.catalog.tasks:
            if task.decision_step != obs.step_index:
                continue
            t = obs.step_index
            if t == 0:
                decision = dict(e._current_decisions)[task.task_id]
                loss = score_event(evaluate_decision(task, decision)).task_loss
                rows.append({"task_id": task.task_id, "decision_step": t,
                             "loss_lower_bound": loss, "method": "exact_initial_decision"})
                continue
            forced = set(visibility.visible_object_ids(task.vehicle_id, frames[t]))
            options = {}
            for obj in e._replay.vehicle_ids:
                if obj == task.vehicle_id:
                    continue
                variants = (generated[task.vehicle_id, obj, t] if obj in forced else
                            itertools.chain.from_iterable(v for (owner, oid, step), v in generated.items()
                                                          if oid == obj and step < t))
                outcomes = set()
                for snapshot in variants:
                    outcomes.add(predict_pair(e._self_snapshots[task.vehicle_id], snapshot,
                                             task.conflict_point, obs.elapsed_seconds,
                                             obs.step_times, cfg).risk)
                options[obj] = outcomes
            if task.true_risk:
                loss = 0. if any(True in options.get(obj, ()) for obj in task.related_vehicle_ids) else 5.
            else:
                # bit 1 known, bit 2 unresolved, bit 4 alarm. Optional objects can be absent.
                states = {0}
                for obj, outcomes in options.items():
                    flags = {1 | (2 if value is None else 4 if value else 0) for value in outcomes}
                    if obj not in forced:
                        flags.add(0)
                    states = {a | b for a in states for b in flags}
                loss = min(1. if flags & 4 else .25 if flags & 2 or not flags & 1 else 0.
                           for flags in states)
            rows.append({"task_id": task.task_id, "decision_step": t, "loss_lower_bound": loss,
                         "method": "independent_snapshot_history_superset",
                         "forced_local_object_ids": sorted(forced),
                         "possible_predictions": {obj: sorted(outcomes, key=str) for obj, outcomes in options.items()}})
        e.step(EMPTY)
    return rows
