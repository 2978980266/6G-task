"""B 固定回合的有预算搜索；可行解启发式与完整证明搜索明确分开。"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import os
import time

from semantics import (EMPTY, accelerate as accelerate, cache_payload as cache_payload,
                       clone_env, endpoints, legal_actions, relaxed_bound as relaxed_bound,
                       repair as repair, rollout, state_key)
from skylines_demo.risk.evaluator import evaluate_decision
from skylines_demo.risk.scoring import score_event
from skylines_demo.types import RiskSchedulingAction


class BudgetStop(RuntimeError):
    pass


@cache
def _windows_memory_query():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
             "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
             "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    query.restype = wintypes.BOOL
    return ctypes, Counters, current, query


def memory_mib():
    if os.name == "nt":
        ctypes, Counters, current, query = _windows_memory_query()
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = current()
        if not query(handle, ctypes.byref(counters), counters.cb):
            raise OSError("无法读取进程内存，拒绝无保护求解。")
        return max(counters.WorkingSetSize, counters.PagefileUsage) / 1024**2
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if os.sys.platform == "darwin" else 1024)


@dataclass
class Limits:
    seconds: float = 600.
    max_nodes: int = 500000
    memory_limit_mib: float = 1024.
    start: float = field(default_factory=time.perf_counter)
    nodes: int = 0
    peak_memory_mib: float = 0.
    last_progress: float = 0.
    progress: object = None

    def tick(self, increment=1):
        now = time.perf_counter()
        if now - self.start >= self.seconds:
            raise BudgetStop("time_limit")
        if self.nodes + increment > self.max_nodes:
            raise BudgetStop("node_limit")
        self.nodes += increment
        if self.nodes % 64 == 0 or increment == 0:
            self.peak_memory_mib = max(self.peak_memory_mib, memory_mib())
            if self.peak_memory_mib >= self.memory_limit_mib:
                raise BudgetStop("memory_limit")
        if self.progress and now - self.last_progress > 20:
            self.last_progress = now
            self.progress(self)


def cents(value):
    return round(value * 100)


def exact_search(env, initial_plan, lower_by_step, limits, *, on_improvement=None,
                 certified_root_lower_bound=None):
    """完整懒枚举所有合法有序联合动作，无 beam/top_k/新鲜度剪枝。

    LB = 已付损失/发送费 + 当步固定判断损失 + 未来独立松弛损失。
    停止时保守保留根松弛 LB，不将已探索分支的较高界冒充全树界。
    """
    e = clone_env(env)
    e.reset()
    incumbent = rollout(e, initial_plan)
    best = {"upper_bound": incumbent["objective"]["objective_sum"], "plan": incumbent["plan"]}
    lower = [cents(x) for x in lower_by_step]
    root_lb = max(sum(lower) / 100, certified_root_lower_bound or 0.)
    if cents(root_lb) > cents(best["upper_bound"]):
        raise AssertionError("证书下界超过合法UB。")
    visited, pruned, hits = 0, 0, 0
    memo = {}

    def visit(node, prefix, paid):
        nonlocal visited, pruned, hits
        limits.tick(0)
        visited += 1
        obs = node.current_observation
        if obs is None:
            if paid < cents(best["upper_bound"]):
                best.update(upper_bound=paid/100, plan=prefix)
                if on_improvement:
                    on_improvement(best["plan"], best["upper_bound"], "exact")
            return
        t = obs.step_index
        due = sum(cents(score_event(evaluate_decision(node._task_for_id(key), decision)).task_loss)
                  for key, decision in node._current_decisions)
        bound = paid + due + sum(lower[t+1:])
        if bound >= cents(best["upper_bound"]):
            pruned += 1
            return
        key = state_key(node)
        if key in memo and memo[key] <= paid:
            hits += 1
            return
        # 记忆表满后不再保存新键；遗忘不会删除搜索分支或损害证明。
        if len(memo) < 4096:
            memo[key] = paid
        # 全部未来任务已判断后，所有非空动作仅增加费用；空后缀严格支配。
        future = any(task.decision_step > t for task in node.catalog.tasks)
        actions = legal_actions(obs, node._risk_config) if future else (EMPTY,)
        for action in actions:
            limits.tick()
            child = clone_env(node)
            result = child.step(action)
            if result.invalid_action:
                raise AssertionError("完整动作枚举器产生非法动作。")
            visit(child, prefix + [action], paid + cents(result.feedback.objective_sum))

    status = "optimal"
    try:
        if cents(root_lb) == cents(best["upper_bound"]):
            status = "bound_certificate_optimal"
        else:
            visit(e, [], 0)
    except BudgetStop as exc:
        status = str(exc)
    proven = status in {"optimal", "bound_certificate_optimal"}
    lb = best["upper_bound"] if proven else root_lb
    return dict(best, status=status, proven_optimal=proven, lower_bound=lb,
                absolute_gap=best["upper_bound"]-lb,
                relative_gap=(best["upper_bound"]-lb)/best["upper_bound"] if best["upper_bound"] else 0.,
                exact_visited_nodes=visited, pruned_nodes=pruned, memo_hits=hits, memo_entries=len(memo),
                proof_scope="all_environment_legal_ordered_joint_actions")


def edits(action, message, kind, cfg):
    """可行解邻域：插入一条消息，并显式替换端点冲突或同类满额消息。"""
    groups = [list(action.uplinks), list(action.downlinks), list(action.relays)]
    nodes = endpoints(message)
    groups = [[c for c in group if not endpoints(c) & nodes] for group in groups]
    maximum = (cfg.uplink_slots_per_step, cfg.downlink_slots_per_step,
               cfg.relay_slots_per_step if cfg.enable_v2v_relay else 0)[kind]
    if not maximum:
        return
    base = groups[kind]
    possibilities = [base] if len(base) < maximum else [base[:i] + base[i+1:] for i in range(len(base))]
    for items in possibilities:
        for index in range(len(items)+1):
            changed = groups.copy()
            changed[kind] = items[:index] + [message] + items[index:]
            yield RiskSchedulingAction(*(tuple(group) for group in changed))


def improve(env, plan, limits, until, on_improvement=None):
    """仅用于 UB。全回合评价真实推演；接受严格改善，绝不宣称局部穷尽为全局证明。"""
    best = rollout(env, plan)
    trials, accepted, pair_trials = 0, 0, 0

    def test(t, changed, stage):
        nonlocal best, trials, accepted, pair_trials
        limits.tick(0)
        if time.perf_counter() >= until:
            raise BudgetStop("heuristic_phase_end")
        if limits.nodes >= limits.max_nodes * .55:
            raise BudgetStop("heuristic_node_quota")
        trials += 1
        pair_trials += stage == "two_hop"
        suffix = rollout(best["states"][t], changed[t:], strict=False, from_current=True,
                         keep_states=False, limits=limits)
        if cents(suffix["objective"]["objective_sum"]) >= cents(best["objective"]["objective_sum"]):
            return None
        repaired = best["plan"][:t] + suffix["plan"]
        best = rollout(env, repaired)
        accepted += 1
        if on_improvement:
            on_improvement(best["plan"], best["objective"]["objective_sum"], stage)
        return best

    try:
        while True:
            changed_any = False
            # 删除会完整重演未来；不假定删除未改变 outcome 的消息必然无效。
            for t, action in enumerate(best["plan"]):
                for kind, group in enumerate((action.uplinks, action.downlinks, action.relays)):
                    for i in range(len(group)):
                        parts = [action.uplinks, action.downlinks, action.relays]
                        parts[kind] = group[:i] + group[i+1:]
                        changed = best["plan"].copy()
                        changed[t] = RiskSchedulingAction(*parts)
                        if test(t, changed, "delete"):
                            changed_any = True
                            break
                    if changed_any:
                        break
                if changed_any:
                    break
            if changed_any:
                continue
            for t, state in enumerate(best["states"]):
                obs = state.current_observation
                if not any(task.decision_step > t for task in state.catalog.tasks):
                    continue
                for kind, pool in ((1, obs.downlink_candidates), (2, obs.relay_candidates),
                                   (0, obs.uplink_candidates)):
                    def rank(c):
                        receiver = getattr(c, "receiver_id", 0)
                        tasks = [task for task in state.catalog.tasks if task.decision_step > t
                                 and (not receiver or task.vehicle_id == receiver)]
                        related = any(c.object_id in task.related_vehicle_ids for task in tasks)
                        return (not related, min((task.decision_step for task in tasks), default=999), str(c))
                    for candidate in sorted(pool, key=rank):
                        for replacement in edits(best["plan"][t], candidate, kind, state._risk_config):
                            if replacement == best["plan"][t]:
                                continue
                            changed = best["plan"].copy()
                            changed[t] = replacement
                            if test(t, changed, "single_message"):
                                changed_any = True
                                break
                        if changed_any:
                            break
                    if changed_any:
                        break
                if changed_any:
                    break
            if changed_any:
                continue
            # U 本身可能只有费用；联合改 U(t)+D(s) 可越过这一局部障碍。
            for t, state in enumerate(best["states"][:-2]):
                obs = state.current_observation
                for up in obs.uplink_candidates:
                    for replacement in edits(best["plan"][t], up, 0, state._risk_config):
                        changed = best["plan"].copy()
                        changed[t] = replacement
                        limits.tick(0)
                        if time.perf_counter() >= until:
                            raise BudgetStop("heuristic_phase_end")
                        if limits.nodes >= limits.max_nodes * .55:
                            raise BudgetStop("heuristic_node_quota")
                        branch = rollout(state, changed[t:], strict=False, from_current=True, limits=limits)
                        for offset, later in enumerate(branch["states"][1:], 1):
                            s = t + offset
                            for down in later.current_observation.downlink_candidates:
                                if down.object_id != up.object_id:
                                    continue
                                if not any(task.vehicle_id == down.receiver_id and task.decision_step > s
                                           for task in env.catalog.tasks):
                                    continue
                                for second in edits(branch["plan"][offset], down, 1, state._risk_config):
                                    both = best["plan"][:t] + branch["plan"].copy()
                                    both[s] = second
                                    if test(t, both, "two_hop"):
                                        changed_any = True
                                        break
                                if changed_any:
                                    break
                            if changed_any:
                                break
                        if changed_any:
                            break
                    if changed_any:
                        break
                if changed_any:
                    break
            if not changed_any:
                break
    except BudgetStop as exc:
        return best["plan"], dict(status=str(exc), trials=trials, accepted=accepted, two_hop_trials=pair_trials)
    return best["plan"], dict(status="local_neighborhood_exhausted", trials=trials,
                             accepted=accepted, two_hop_trials=pair_trials)
