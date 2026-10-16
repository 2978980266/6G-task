"""只针对冻结 B 场景的必要接收次数证明及其耦合消息邻域。"""
from __future__ import annotations

from dataclasses import asdict, replace

from semantics import EMPTY, rollout
from solver import BudgetStop, cents
from skylines_demo.risk.scoring import OBJECTIVE_WEIGHTS
from skylines_demo.types import RiskSchedulingAction

CERTIFICATE_VERSION = "fixed-B-receiver-minimum-inbound-v1"
PROOF = """零接收车辆的缓存、合法历史与判断和无通信完全相同。
每个入站 D/R 只对应一辆接收车，U 费非负，可按接收车给损失加必要通信费的下界。
普通车辆取 min(无接收损失, 逐任务松弛损失+0.01)。
7019 在期限1首次本地看到13138且没有过去位置。消除此任务的负弃权，必须
在step0收到13138@0；其他对象不能补13138的估速，也不能删除这个当前未知对象，
报警则成为更差的误报。所有step0原始快照都缺合法速度。
若全回合仅这一条入站消息，期限18时13138不在本地视野；纯本地刷新不替换
数据面缺速版本，故这个版本必造成弃权或更差误报。期限12另有固定自身不可预测
的0.25损失。因而零/一条入站时7019损失至少0.50，两条及以上至少0.25，
开中继该车最低贡献为min(0.50,0.50+0.01,0.25+0.02)=0.27。
关中继step0不存在13138到7019的下行，上行再下行最早step2可用，期限1的
弃权无法消除，7019总损失至少0.50。以整数分汇总，关中继LB=11.37，开中继LB=11.14。
此证明忽略路径共享和车辆/名额竞争，只放宽约束，适用于完整合法动作集合。
达到该LB的原环境合法方案即为全局最优；未达到时只给有效下界。"""


def fixed_receiver_certificate(env, lower_rows):
    if env._risk_config.ego_eta_mode != "causal-self-motion" or env._risk_config.link_prediction:
        raise ValueError("该证明只适用冻结B且预测关闭。")
    if (OBJECTIVE_WEIGHTS.positive_failure, OBJECTIVE_WEIGHTS.negative_false_positive,
        OBJECTIVE_WEIGHTS.negative_abstention, OBJECTIVE_WEIGHTS.send) != (5., 1., .25, .01):
        raise ValueError("证明权重漂移。")
    no = rollout(env, [EMPTY]*len(env._replay))
    catalog = {t.task_id: t for t in env.catalog.tasks}
    lb = {r["task_id"]: cents(r["loss_lower_bound"]) for r in lower_rows}
    first, last, own = (catalog[name] for name in
                        ("risk-7019-6-3", "risk-7019-3-2", "risk-7019-4-2"))
    if ((first.vehicle_id, first.decision_step, first.true_risk, last.decision_step,
         last.true_risk, own.decision_step, own.true_risk) != (7019, 1, False, 18, False, 12, False)):
        raise ValueError("固定任务证明条件漂移。")
    start, step1, step18 = (no["states"][i] for i in (0, 1, 18))
    if ((7019, 13138) in start._current_local_snapshots
            or step1._current_local_snapshots[7019, 13138].velocity_x is not None
            or 13138 in step18._sensor.visible_object_ids(7019, step18._replay.frames[18])
            or any(d.receiver_id == 7019 and d.object_id == 13138
                   for d in start.current_observation.downlink_candidates) or lb[own.task_id] != 25):
        raise ValueError("首次本地观测、失见或期限条件漂移。")
    forwardable = [s for (_, obj), s in start._current_local_snapshots.items() if obj == 13138]
    if not forwardable or any(s.observation_step != 0 or s.velocity_x is not None
                              or s.velocity_z is not None for s in forwardable):
        raise ValueError("step0快照速度条件漂移。")
    single_checks = []
    if env._risk_config.enable_v2v_relay:
        for relay in start.current_observation.relay_candidates:
            if relay.receiver_id != 7019 or relay.object_id != 13138:
                continue
            plan = [EMPTY]*len(env._replay)
            plan[0] = RiskSchedulingAction(relays=(relay,))
            run = rollout(env, plan)
            scores = {s["task_id"]: s for s in run["task_scores"]}
            if scores[last.task_id]["task_loss"] < .25 or scores[own.task_id]["task_loss"] < .25:
                raise ValueError("单接收保留缺速版本的证明条件不成立。")
            single_checks.append(dict(sender_id=relay.sender_id,
                                      first_outcome=scores[first.task_id]["outcome"],
                                      last_outcome=scores[last.task_id]["outcome"]))
    rows = {}
    for score in no["task_scores"]:
        owner = catalog[score["task_id"]].vehicle_id
        row = rows.setdefault(owner, dict(no_receive_loss_cents=0, relaxed_loss_cents=0))
        row["no_receive_loss_cents"] += cents(score["task_loss"])
        row["relaxed_loss_cents"] += lb[score["task_id"]]
    for owner, row in rows.items():
        if owner == 7019:
            if (row["no_receive_loss_cents"], row["relaxed_loss_cents"]) != (50, 25):
                raise ValueError("7019目录损失条件漂移。")
            row["lower_bound_cents"] = 27 if env._risk_config.enable_v2v_relay else 50
        else:
            row["lower_bound_cents"] = min(row["no_receive_loss_cents"], row["relaxed_loss_cents"]+1)
    lower = sum(r["lower_bound_cents"] for r in rows.values())
    if lower != (1114 if env._risk_config.enable_v2v_relay else 1137):
        raise ValueError("固定场景接收次数证书汇总漂移。")
    return dict(version=CERTIFICATE_VERSION, conditions_verified=True, lower_bound=lower/100,
                lower_bound_cents=lower, receiver_bounds=rows, single_inbound_checks=single_checks,
                forwardable_step0_13138=[asdict(s) for s in forwardable], proof=PROOF)


def refine_coupled_receptions(env, plan, limits, lower_bound, on_improvement=None):
    """固定联合邻域，只提供UB：R0补过去位置，再D刷新同一对象的缺速数据面缓存。"""
    if not env._risk_config.enable_v2v_relay:
        return plan, dict(status="relay_off", trials=0)
    best = rollout(env, plan)
    if cents(best["objective"]["objective_sum"]) == cents(lower_bound):
        return plan, dict(status="bound_already_met", trials=0)
    start = best["states"][0]
    trials = 0
    try:
        for relay in start.current_observation.relay_candidates:
            if relay.receiver_id != 7019 or relay.object_id != 13138:
                continue
            first = plan.copy()
            first[0] = replace(first[0], relays=(*first[0].relays, relay))
            if not start._is_valid_action(first[0], start.current_observation):
                continue
            branch = rollout(env, first, limits=limits)
            for t, state in enumerate(branch["states"][1:], 1):
                for down in state.current_observation.downlink_candidates:
                    if down.receiver_id != 7019 or down.object_id != 13138:
                        continue
                    candidate = first.copy()
                    candidate[t] = replace(candidate[t], downlinks=(*candidate[t].downlinks, down))
                    if not state._is_valid_action(candidate[t], state.current_observation):
                        continue
                    run = rollout(env, candidate, limits=limits)
                    trials += 1
                    if cents(run["objective"]["objective_sum"]) < cents(best["objective"]["objective_sum"]):
                        best = run
                        if on_improvement:
                            on_improvement(run["plan"], run["objective"]["objective_sum"], "coupled_past_and_refresh")
                        return best["plan"], dict(status="improved", trials=trials)
    except BudgetStop as exc:
        return best["plan"], dict(status=str(exc), trials=trials)
    return best["plan"], dict(status="no_improvement", trials=trials)
