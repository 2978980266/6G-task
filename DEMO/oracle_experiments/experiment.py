"""两个 B 固定场景入口共用的实验记录；没有通用场景/模型配置入口。"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from semantics import EMPTY, accelerate, cache_payload, endpoints, rollout
from solver import Limits, exact_search, improve, relaxed_bound
from certificates import fixed_receiver_certificate, refine_coupled_receptions
from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.policies.risk_baselines import select_age_action, select_deadline_action
from skylines_demo.policies.risk_oracle import select_diagnostic_oracle_action
from skylines_demo.policies.risk_public_uplink import select_public_uplink_action
from skylines_demo.policies.risk_receiver_policy import select_receiver_value_action
from skylines_demo.policies.risk_relay_policy import select_receiver_value_with_relay_action
from skylines_demo.risk.model import model_version
from skylines_demo.types import DownlinkAction, RelayAction, RiskSchedulingAction, UplinkAction


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_plan(plan):
    return [{"step_index": i, **asdict(action)} for i, action in enumerate(plan)]


def decode_plan(rows):
    if [row["step_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("动作 step 必须从 0 连续排列。")
    return [RiskSchedulingAction(tuple(UplinkAction(**a) for a in row["uplinks"]),
                                 tuple(DownlinkAction(**a) for a in row["downlinks"]),
                                 tuple(RelayAction(**a) for a in row["relays"])) for row in rows]


class Writer:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=False)

    def json(self, name, value):
        with (self.path / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)

    def jsonl(self, name, rows):
        with (self.path / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def manifest(env):
    sources = {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(
        set((ROOT / "src").rglob("*.py")) | set((ROOT / "scripts").glob("*.py")))}
    additions = {"DEMO/scripts/"+name+".py" for name in
                 ("check_risk_learning", "train_risk_scheduler", "evaluate_risk_scheduler", "run_risk_learning_suite")}
    additions.add("DEMO/src/skylines_demo/policies/risk_training_controls.py")
    frozen = {p: h for p, h in sources.items() if "/risk_learning/" not in p and p not in additions}
    data = {p.relative_to(REPO).as_posix(): file_hash(p)
            for directory in (env._runtime.config.trajectory_csv, env._runtime.config.building_csv)
            for p in sorted(directory.glob("*.csv"))}
    solver_sources = {p.relative_to(REPO).as_posix(): file_hash(p)
                      for p in sorted(Path(__file__).parent.glob("*.py"))}
    config_hashes = {name: file_hash(ROOT / "data/config" / name) for name in ("train.yaml", "risk.yaml")}
    if fingerprint(frozen) != "22cfca1e19e86fff1a1fcf243fd933c55e859f3db47de3b544178d9e5062c94c":
        raise ValueError("既有冻结源码漂移；停止，不修改 sidecar。")
    if fingerprint(data) != "ca0f35c80f944dc77bbac09e86737383468575f82fc52e44247948eb8b047af1":
        raise ValueError("默认数据漂移。")
    if config_hashes != {"train.yaml": "9932058fa2ee8aecbc604f5ee73019afc04bc5c60c96d906235d36b4c21b653c",
                         "risk.yaml": "18e3b06c07ef98a8abb33b7d181569f5a5586f0a2763dd8e81d38483c954fe61"}:
        raise ValueError("冻结配置漂移。")
    if fingerprint(asdict(env.catalog)) != "1b091a60bb9e16cfed79526455cd71b5cf0c01a1b19cde5dc4c72e7aec05eabf":
        raise ValueError("冻结任务目录漂移。")
    if (len(env._replay.vehicle_ids), len(env._runtime.buildings), len(env._replay),
        len(env.catalog.tasks), sum(t.true_risk for t in env.catalog.tasks)) != (16, 18, 23, 46, 8):
        raise ValueError("场景计数漂移。")
    return dict(source_hashes=sources, source_fingerprint=fingerprint(sources),
                frozen_source_fingerprint=fingerprint(frozen), data_hashes=data, data_fingerprint=fingerprint(data),
                solver_source_hashes=solver_sources, solver_source_fingerprint=fingerprint(solver_sources),
                config_hashes=config_hashes, task_fingerprint=fingerprint(asdict(env.catalog)),
                scenario_config=asdict(env._runtime.config), effective_risk_config=asdict(env._risk_config),
                model_version=model_version(env._risk_config), python=platform.python_version(),
                solver_dependencies="Python standard library only; no optimization package or training dependency",
                scenario="20260623_151408", mode="B", domain="all_environment_legal_actions_including_empty")


def ordered_action_count(obs, cfg):
    """按占用车辆集合聚合计数，保留所有消息与顺序；只用于规模诊断。"""
    vehicles = {state.vehicle_id: 1 << i for i, state in enumerate(obs.active_vehicles)}

    def pool_counts(pool, maximum):
        counts = {0: 1}

        def add(used, remaining):
            if not remaining:
                return
            for c in pool:
                bits = sum(vehicles[v] for v in endpoints(c))
                if bits & used:
                    continue
                key = used | bits
                counts[key] = counts.get(key, 0) + 1
                add(key, remaining - 1)

        add(0, maximum)
        return counts

    pools = [pool_counts(obs.uplink_candidates, cfg.uplink_slots_per_step),
             pool_counts(obs.downlink_candidates, cfg.downlink_slots_per_step),
             pool_counts(obs.relay_candidates, cfg.relay_slots_per_step if cfg.enable_v2v_relay else 0)]
    combined = {0: 1}
    for pool in pools:
        after = {}
        for a, na in combined.items():
            for b, nb in pool.items():
                if not a & b:
                    key = a | b
                    after[key] = after.get(key, 0) + na*nb
        combined = after
    return sum(combined.values())


def verify_and_save(env, plan, directory):
    """新建原 RiskCommEnv，从 reset 完整回放；逐字段对照查表搜索环境。"""
    writer = Writer(directory)
    fast = rollout(accelerate(env), plan, strict=True)
    original = RiskCommEnv(env._runtime, env._risk_config)
    original.reset()
    rows, scores, deliveries = [], [], []
    for i, action in enumerate(plan):
        before = cache_payload(original)
        if before != cache_payload(fast["states"][i]):
            raise AssertionError("加速缓存/历史与原环境动作前不一致。")
        result = original.step(action)
        if result.invalid_action or asdict(result) != asdict(fast["results"][i]):
            raise AssertionError(f"原环境回放不合法或与搜索状态漂移，step={i}")
        records = [asdict(r) for r in result.delivery_records]
        for record in records:
            s = record["snapshot"]
            if not (record["usable_step"] == i+1 and s["observation_step"] <= i
                    and s["received_step"] == i+1 and record["receiver_id"] != s["object_id"]):
                raise AssertionError("投递时序或对象非法。")
        deliveries.extend(records)
        scores.extend(asdict(s) for s in result.feedback.task_scores)
        rows.append(dict(step_index=i, action=asdict(action), observation=asdict(result.observation),
                         invalid_action=result.invalid_action, delivery_records=records,
                         risk_decisions=[{"task_id": k, "decision": asdict(d)} for k, d in result.risk_decisions],
                         event_results=[asdict(e) for e in result.event_results],
                         task_scores=[asdict(s) for s in result.feedback.task_scores],
                         feedback=asdict(result.feedback), cache_before=before,
                         cache_after=cache_payload(original), objective=result.objective))
    objective = original.objective_summary
    if (original.current_observation is not None or len(plan) != 23 or len(scores) != 46
            or len({s["task_id"] for s in scores}) != 46 or not objective["evaluation_complete"]):
        raise AssertionError("未完整回放全部任务。")
    f = sum(s["positive_failure"] for s in scores)
    fp = sum(s["negative_false_positive"] for s in scores)
    abstain = sum(s["negative_abstention"] for s in scores)
    loss, sends = 5*f + fp + .25*abstain, len(deliveries)
    if (objective["task_loss_sum"] != loss or objective["send_count"] != sends
            or abs(objective["objective_sum"] - loss - .01*sends) > 1e-9
            or abs(sum(r["feedback"]["reward"] for r in rows) + loss + .01*sends) > 1e-9):
        raise AssertionError("独立汇总与唯一评分不一致。")
    if cache_payload(original) != cache_payload(fast["final_env"]) or scores != fast["task_scores"]:
        raise AssertionError("末状态或全任务计分不一致。")
    writer.jsonl("actions.jsonl", encode_plan(plan))
    writer.jsonl("replay.jsonl", rows)
    writer.jsonl("deliveries.jsonl", deliveries)
    writer.json("task_scores.json", scores)
    writer.json("objective.json", objective)
    verification = dict(passed=True, original_reset=True, steps=23, task_scores=46,
                        actual_delivery_count=sends, illegal_actions=0, full_result_fields_equal=True,
                        all_cache_and_velocity_history_equal=True, independently_recomputed_L=loss,
                        independently_recomputed_J=loss+.01*sends, initial_loss=objective["initial_task_loss_sum"])
    writer.json("verification.json", verification)
    return dict(objective=objective, verification=verification, task_scores=scores)


def compare(before, after):
    a, b = before["objective"], after["objective"]
    old = {s["task_id"]: s for s in before["task_scores"]}
    changes = [{"task_id": s["task_id"], "before": old[s["task_id"]], "after": s}
               for s in after["task_scores"] if old[s["task_id"]] != s]
    return dict(delta_loss=b["task_loss_sum"]-a["task_loss_sum"],
                delta_send_count=b["send_count"]-a["send_count"],
                delta_send_cost=.01*(b["send_count"]-a["send_count"]),
                delta_J=b["objective_sum"]-a["objective_sum"], changed_task_scores=changes,
                positive_recovered=[c["task_id"] for c in changes
                                    if c["before"]["positive_failure"] and c["after"]["positive_success"]],
                false_positive_to_abstention=[c["task_id"] for c in changes
                    if c["before"]["negative_false_positive"] and c["after"]["negative_abstention"]])


def main(relay: bool):
    parser = argparse.ArgumentParser(description=f"固定场景 B 离线调度；中继{'开' if relay else '关'}，首次预算至多600秒。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--node-limit", type=int, default=500000)
    parser.add_argument("--memory-mib", type=float, default=1024.)
    if relay:
        parser.add_argument("--seed-plan", help="关中继组的 actions.jsonl，仅作为初始可行解，严格回放。")
    args = parser.parse_args()
    if not 0 < args.seconds <= 600 or args.node_limit < 1 or args.memory_mib < 128:
        parser.error("要求0<seconds<=600、node-limit>=1、memory-mib>=128。")
    writer = Writer(args.output_dir)
    limits = Limits(args.seconds, args.node_limit, args.memory_mib)
    try:
        return run(relay, args, writer, limits)
    except BaseException as exc:
        writer.json("failure.json", dict(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                                        elapsed_seconds=time.perf_counter()-limits.start, nodes=limits.nodes))
        raise


def run(relay, args, writer, limits):
    runtime = load_runtime(ROOT / "data/config/train.yaml")
    cfg = replace(load_risk_config(ROOT / "data/config/risk.yaml"), ego_eta_mode="causal-self-motion",
                  enable_v2v_relay=relay, link_prediction=False)
    env = RiskCommEnv(runtime, cfg)
    env.reset()
    before_manifest = manifest(env)
    parameters = dict(seconds=args.seconds, node_limit=args.node_limit, memory_limit_mib=args.memory_mib,
                      heuristic_fraction=.55, heuristic_transition_quota_fraction=.55, exact_memo_entry_limit=4096,
                      candidates_truncated=False, action_order_preserved=True,
                      heuristic="strict descent: delete/insert/replace one message; two-hop U+D mutations",
                      coupled_refinement="fixed 7019/13138 past-position R0 plus later D refresh; charged to same budget",
                      root_certificate="fixed-B-receiver-minimum-inbound-v1",
                      lower_bound="independent per-object snapshot/history superset; communication free",
                      exact="lazy full ordered joint-action branch and bound; conservative full mutable state memo",
                      started_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    writer.json("manifest.json", dict(before_manifest, solver_parameters=parameters))
    writer.json("implementation_plan.json", dict(
        design="原环境step+独立可变状态副本；几何查表；完整有序动作分支定界；松弛LB；真实回放UB。",
        alternatives={"dynamic_programming": "缓存/估速/来源状态组合大；仅精确相同状态可合并。",
                      "optimization_encoding": "条件速度依据和先到版本覆盖需重新编码，10分钟内不宜用未验证替代语义。"},
        steps=["穷举合成实例", "原环境/查表跨步核对", "冻结清单和基线回放", "有预算可行解搜索与证明搜索",
               "从原环境reset逐字段复核", "46任务/费用/上下界分解", "更新根目录README与AGENTS"],
        constraints="仅B固定会话；不训练、不运行A/S1/新场景；不修改src/scripts/YAML/历史sidecar；不调用子智能体或Git写操作。"))
    for name in ("train.yaml", "risk.yaml"):
        (writer.path / name).write_bytes((ROOT / "data/config" / name).read_bytes())
    snapshot = Writer(writer.path / "solver_source")
    for source in Path(__file__).parent.glob("*.py"):
        (snapshot.path / source.name).write_bytes(source.read_bytes())
    writer.json("catalog.json", asdict(env.catalog))
    fast = accelerate(env)
    lower_rows = relaxed_bound(fast)
    receiver_certificate = fixed_receiver_certificate(env, lower_rows)
    writer.json("receiver_bound_certificate.json", receiver_certificate)
    root_lower_bound = receiver_certificate["lower_bound"]
    lower_by_step = [sum(r["loss_lower_bound"] for r in lower_rows if r["decision_step"] == i) for i in range(23)]
    writer.json("lower_bound.json", dict(lower_bound=sum(lower_by_step), task_relaxations=lower_rows,
                proof="每个实际本地生成快照的先前估速位置均被枚举；允许不可达接收、免费旧本地保留和独立选择；"
                      "强制保留当前本地可见对象；移除路径、时序容量与多任务耦合只扩大判断输入集合。"
                      "各任务最小非负损失之和加0发送费不超过任一合法方案的J。step0按真实初始判断计入。"))
    print(json.dumps({"阶段": "输入核对", "中继": relay, "有效下界": root_lower_bound}, ensure_ascii=False), flush=True)
    selectors = {
        "no-communication": lambda o: EMPTY,
        "age": lambda o: select_age_action(o, cfg),
        "deadline": lambda o: select_deadline_action(o, cfg),
        "receiver-value": lambda o: select_receiver_value_action(o, cfg),
        "demand-uplink": lambda o: select_public_uplink_action(o, cfg, mode="demand"),
        "diagnostic-oracle": lambda o: select_diagnostic_oracle_action(o, cfg, env.catalog,
                                      replay=runtime.replay, communication=fast._communication),
    }
    if relay:
        selectors.update({"receiver-value-relay": lambda o: select_receiver_value_with_relay_action(o, cfg),
                          "demand-uplink-relay": lambda o: select_public_uplink_action(o, cfg, mode="demand", with_relay=True)})
    baselines, plans = {}, {}
    for name, selector in selectors.items():
        fast.reset()
        plan = []
        while fast.current_observation is not None:
            action = selector(fast.current_observation)
            result = fast.step(action)
            if result.invalid_action:
                raise AssertionError("已有规则产生非法动作，停止阶段。")
            plan.append(action)
        plans[name] = plan
        baselines[name] = verify_and_save(env, plan, writer.path / ("baseline_"+name))
        print(json.dumps({"基线": name, "J": baselines[name]["objective"]["objective_sum"]}, ensure_ascii=False), flush=True)
    if relay and args.seed_plan:
        source = Path(args.seed_plan).resolve()
        seed = decode_plan([json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()])
        if len(seed) != 23 or any(a.relays for a in seed):
            raise ValueError("关中继初始方案须完整23步且不含R。")
        plans["off-seed"] = seed
        baselines["off-seed"] = verify_and_save(env, seed, writer.path / "baseline_off-seed")
        writer.json("seed_plan_source.json", dict(path=str(source), sha256=file_hash(source),
                                                  verification=baselines["off-seed"]["verification"]))
    writer.json("baselines.json", baselines)
    no = rollout(fast, [EMPTY]*23)
    writer.json("search_scale.json", [dict(step_index=i,
                candidates=[len(s.current_observation.uplink_candidates), len(s.current_observation.downlink_candidates),
                            len(s.current_observation.relay_candidates)],
                ordered_legal_joint_actions=ordered_action_count(s.current_observation, cfg))
                for i, s in enumerate(no["states"])])
    best_name = min(plans, key=lambda name: baselines[name]["objective"]["objective_sum"])
    best_plan = plans[best_name]
    best_j = baselines[best_name]["objective"]["objective_sum"]
    improvements = []
    incumbent_dir = Writer(writer.path / "incumbents")

    def save(plan, j, stage):
        nonlocal best_plan, best_j
        best_plan, best_j = plan.copy(), j
        row = dict(index=len(improvements), elapsed_seconds=time.perf_counter()-limits.start,
                   objective_sum=j, stage=stage, total_transition_nodes=limits.nodes, plan=encode_plan(plan))
        improvements.append(row)
        incumbent_dir.json(f"{row['index']:04d}.json", row)
        print(json.dumps({"阶段": stage, "最好J": j, "用时": round(row["elapsed_seconds"], 2)}, ensure_ascii=False), flush=True)

    save(best_plan, best_j, "initial_"+best_name)

    def progress(state):
        print(json.dumps({"阶段": "有预算搜索", "用时": round(time.perf_counter()-state.start, 1),
                          "转移节点": state.nodes, "最好J": best_j, "LB": root_lower_bound,
                          "内存MiB": round(state.peak_memory_mib, 1)}, ensure_ascii=False), flush=True)

    limits.progress = progress
    try:
        best_plan, heuristic = improve(fast, best_plan, limits, limits.start+args.seconds*.55, save)
        best_plan, coupled = refine_coupled_receptions(fast, best_plan, limits, root_lower_bound, save)
        result = exact_search(fast, best_plan, lower_by_step, limits, on_improvement=save,
                              certified_root_lower_bound=root_lower_bound)
    except KeyboardInterrupt:
        heuristic = {"status": "interrupted"}
        coupled = {"status": "interrupted"}
        result = dict(status="interrupted", proven_optimal=False, plan=best_plan, upper_bound=best_j,
                      lower_bound=root_lower_bound, absolute_gap=best_j-root_lower_bound,
                      relative_gap=(best_j-root_lower_bound)/best_j)
    solve_elapsed = time.perf_counter()-limits.start
    # 中间持久化的可行方案也逐一复核；这部分不继续搜索、不追加求解预算。
    incumbent_replays = Writer(writer.path / "incumbent_replays")
    replay_index = []
    for row in improvements:
        checked = verify_and_save(env, decode_plan(row["plan"]),
                                  incumbent_replays.path / f"{row['index']:04d}")
        if abs(checked["objective"]["objective_sum"]-row["objective_sum"]) > 1e-9:
            raise AssertionError("中间方案与原环境完整回放不同。")
        replay_index.append(dict(index=row["index"], verification=checked["verification"]))
    incumbent_replays.json("index.json", replay_index)
    best = verify_and_save(env, result["plan"], writer.path / "best")
    if abs(best["objective"]["objective_sum"]-result["upper_bound"]) > 1e-9:
        raise AssertionError("求解UB与原环境回放不同。")
    if manifest(env) != before_manifest:
        raise AssertionError("执行期间源码/输入发生漂移，停止并保留证据。")
    result = dict(result, plan=encode_plan(result["plan"]))
    summary = dict(mode="B", relay=relay, scenario="20260623_151408", solver=result,
                   heuristic=heuristic, coupled_refinement=coupled, receiver_bound_certificate=receiver_certificate,
                   solver_parameters=parameters, solve_elapsed_seconds=solve_elapsed,
                   total_transition_nodes=limits.nodes, peak_memory_mib=limits.peak_memory_mib,
                   objective=best["objective"], original_replay_verification=best["verification"],
                   improvements=improvements, comparisons={name: compare(baseline, best) for name, baseline in baselines.items()},
                   frozen_unchanged=True, limitation="固定场景的离线动作方案；未证明最优时UB仅为合法方案成本，不是效果上界。")
    writer.json("summary.json", summary)
    writer.json("artifact_index.json", {p.relative_to(writer.path).as_posix(): file_hash(p)
                                      for p in sorted(writer.path.rglob("*")) if p.is_file()})
    print(json.dumps({"完成": str(writer.path), "状态": result["status"], "已证明最优": result["proven_optimal"],
                      "LB": result["lower_bound"], "UB": result["upper_bound"], "目标": best["objective"]},
                     ensure_ascii=False), flush=True)
    return 0
