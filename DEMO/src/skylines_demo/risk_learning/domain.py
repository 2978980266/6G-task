"""在线白名单投影。输出只含公开派生量，没有目录/反馈/运动表引用。"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..config import RiskCommConfig
from ..risk.model import PairPrediction, own_schedule_eta, predict_pair
from ..types import DownlinkAction, RelayAction, RiskObservation, UplinkAction, Vector3

DOMAIN_ID = "receiver-risk-domain-v1"
CandidateKey = tuple[str, int, int, int]
MessageAction = UplinkAction | DownlinkAction | RelayAction


@dataclass(frozen=True)
class SafeTask:
    receiver_key: int
    conflict_point: Vector3
    decision_step: int


@dataclass(frozen=True)
class SafeCandidate:
    key: CandidateKey
    action: MessageAction
    features: tuple[float, ...]


@dataclass(frozen=True)
class SafeStep:
    step_index: int
    mode: str
    domain_id: str
    candidates: tuple[SafeCandidate, ...]
    global_features: tuple[float, ...]
    slots: tuple[int, int, int]
    raw_counts: tuple[int, int, int]


def _finite_pair(value: float | None) -> tuple[float, float]:
    return (float(value), 1.) if value is not None and isfinite(value) else (0., 0.)


def build_safe_step(obs: RiskObservation, config: RiskCommConfig,
                    domain_id: str = DOMAIN_ID) -> SafeStep:
    if domain_id != DOMAIN_ID:
        raise ValueError("首轮仅授权S0；S1需要额外授权。")
    if config.link_prediction:
        raise ValueError("学习阶段预测必须关闭。")
    if (config.uplink_slots_per_step, config.downlink_slots_per_step,
            config.relay_slots_per_step) != (2, 2, 1):
        raise ValueError("要求固定2U+2D，显式中继1R。")
    t = obs.step_index
    tasks = tuple(SafeTask(task.vehicle_id, task.conflict_point, task.decision_step)
                  for task in obs.tasks)
    # A权限在独立分支读取；B连crossing属性都不访问，更不由deadline反推。
    schedules = (tuple(own_schedule_eta(config, task.crossing_step, t, obs.delta_seconds)
                       for task in obs.tasks) if config.ego_eta_mode == "public-crossing-step"
                 else (None,) * len(tasks))
    versions = {(receiver, obj): version for receiver, obj, version
                in obs.receiver_known_observation_steps}
    rsu = {s.object_id: s for s in obs.rsu_snapshots}
    visible = {v.vehicle_id: v.visible_object_ids for v in obs.active_vehicles}

    def version(receiver: int, obj: int) -> int:
        return versions.get((receiver, obj), -1)

    candidates: list[SafeCandidate] = []
    pools: tuple[tuple[str, tuple[MessageAction, ...]], ...] = (
        ("U", obs.uplink_candidates), ("D", obs.downlink_candidates),
        ("R", obs.relay_candidates if config.enable_v2v_relay else ()))
    for kind, actions in pools:
        for action in dict.fromkeys(actions):
            sender = action.sender_id if isinstance(action, (UplinkAction, RelayAction)) else 0
            receiver = action.receiver_id if isinstance(action, (DownlinkAction, RelayAction)) else 0
            obj = action.object_id
            if (kind == "U" and obj == sender) or (kind != "U" and obj == receiver):
                continue
            rsu_version = rsu[obj].observation_step if obj in rsu else -1
            source = rsu_version if kind == "D" else version(sender, obj)
            target = rsu_version if kind == "U" else version(receiver, obj)
            future = [(task, schedule) for task, schedule in zip(tasks, schedules)
                      if task.receiver_key == receiver and task.decision_step > t]
            future.sort(key=lambda item: (item[0].decision_step, item[0].conflict_point.x,
                                         item[0].conflict_point.y, item[0].conflict_point.z))
            local = obj in visible.get(sender, ()) and source == t
            if kind == "U" and not local:
                continue
            if kind == "R" and (not future or source < 0 or source <= target):
                continue
            predictions: list[PairPrediction] = []
            if kind == "D":
                if not future or source < 0 or source <= target:
                    continue
                ego = next((v.self_snapshot for v in obs.active_vehicles
                            if v.vehicle_id == receiver), None)
                predictions = [predict_pair(ego, rsu[obj], task.conflict_point,
                                           obs.elapsed_seconds, obs.step_times[:t + 1], config,
                                           scheduled_ego_arrival_seconds=schedule)
                               for task, schedule in future]
                # 与现有eligible条件一致；只复用正式模型，不读旧task_id排序。
                if not any(p.risk and p.distance_m is not None
                           and p.distance_m <= config.conflict_cluster_radius_m + 1e-9
                           for p in predictions):
                    continue
            beneficiaries = {task.receiver_key for task in tasks
                             if kind == "U" and task.decision_step >= t + 2
                             and task.receiver_key != obj
                             and version(task.receiver_key, obj) < source}
            deadline = min((task.decision_step for task in tasks
                            if task.receiver_key in beneficiaries and task.decision_step >= t + 2),
                           default=None)
            features = [float(kind == name) for name in ("U", "D", "R")]
            for ver in (source, rsu_version, target):
                features.extend((float(t - ver) if ver >= 0 else 0., float(ver >= 0)))
            features.extend((float(source - rsu_version) if source >= 0 and rsu_version >= 0 else 0.,
                             float(source - target) if source >= 0 and target >= 0 else 0.))
            features.extend((float(len(future)), float(future[0][0].decision_step - t) if future else 0.,
                             float(bool(future))))
            features.extend((float(len(beneficiaries)), float(deadline - t) if deadline is not None else 0.,
                             float(deadline is not None)))
            p = predictions[0] if predictions else None
            features.extend((float(p is not None and p.risk is True),
                             float(p is not None and p.risk is False),
                             float(p is not None and p.risk is None)))
            for value in ((p.distance_m, p.ego_arrival_seconds, p.object_arrival_seconds,
                           p.temporal_gap_seconds) if p is not None else (None,) * 4):
                features.extend(_finite_pair(value))
            velocity_known = False
            if kind == "D":
                vx, vz = rsu[obj].velocity_x, rsu[obj].velocity_z
                velocity_known = vx is not None and vz is not None and isfinite(vx) and isfinite(vz)
            features.extend((float(kind == "D" and rsu[obj].source_id == 0),
                             float(velocity_known),
                             float(obj in obs.rsu_visible_object_ids), float(kind == "U" and local)))
            assert len(features) == 32
            candidates.append(SafeCandidate((kind, sender, receiver, obj), action, tuple(features)))
    order = {"U": 0, "D": 1, "R": 2}
    candidates.sort(key=lambda c: (order[c.key[0]], *c.key[1:]))
    slots = (2, 2, int(config.enable_v2v_relay))
    return SafeStep(t, "B" if config.ego_eta_mode == "causal-self-motion" else "A", domain_id,
                    tuple(candidates), (t / 23., obs.delta_seconds / .3, 1., 1., float(slots[2]),
                                        float(any(task.decision_step >= t + 1 for task in tasks))),
                    slots, (len(obs.uplink_candidates), len(obs.downlink_candidates),
                            len(obs.relay_candidates)))
