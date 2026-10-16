"""独立 oracle 验证；合成实例只用于单元核对，不改变默认场景。"""
from __future__ import annotations

import importlib
import itertools
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skylines_demo.bootstrap import ScenarioRuntime, load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.geometry import BuildingLineOfSight
from skylines_demo.perception.snapshots import KnowledgeCache
from skylines_demo.risk.conflicts import ConflictCatalog, RiskTask
from skylines_demo.risk.evaluator import evaluate_decision
from skylines_demo.risk.executor import execute_risk
from skylines_demo.risk.scoring import RiskScoreLedger, score_event
from skylines_demo.types import (
    DownlinkAction, FrameRecord, RiskSchedulingAction, RiskSnapshot, UplinkAction,
    Vector3, VehicleReplay, VehicleSnapshot,
)
from skylines_demo.wireless import LogDistanceChannelModel

ROOT = Path(__file__).resolve().parents[1]


def tiny_env(relay=False):
    base = load_runtime(ROOT / "data/config/train.yaml")
    cfg = replace(load_risk_config(ROOT / "data/config/risk.yaml"),
                  enable_v2v_relay=relay, vehicle_sensor_range_m=.01)
    frames = tuple(FrameRecord(t, t, "fixture", "fixture", .3*t, .3*t, .3, {
        1: VehicleSnapshot(1, Vector3(0, 0, -20+5*t), 0, 0, 0, "fixture"),
        2: VehicleSnapshot(2, Vector3(-20+5*t, 0, 0), 0, 0, 90, "fixture"),
        3: VehicleSnapshot(3, Vector3(20, 0, 20), 0, 0, 0, "fixture"),
    }) for t in range(3))
    los = BuildingLineOfSight([])
    runtime = ScenarioRuntime(replace(base.config, rsu=replace(base.config.rsu, position=Vector3(0, 30, 0))),
                              VehicleReplay(frames), [], los, LogDistanceChannelModel(base.config.channel, los))
    env = RiskCommEnv(runtime, cfg)
    tasks = tuple(RiskTask(name, 1, 0, Vector3(0, 0, 0), 1.2, 4, d, True, (2,))
                  for name, d in (("initial", 0), ("shared-a", 2), ("shared-b", 2)))
    env._catalog = ConflictCatalog((), (), tasks)
    env._tasks_by_step = {0: (tasks[0],), 2: tasks[1:]}
    env._score_ledger = RiskScoreLedger(tasks, 3)
    return env


class OracleChecks(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module("solver")
        except ImportError as exc:
            self.fail(f"离线求解器尚未实现：{exc}")

    def test_tiny_exact_optimum_and_shared_message(self):
        api = self.api()
        env = tiny_env()
        env.reset()
        result = api.exact_search(env, [RiskSchedulingAction()]*3, [5., 0., 0.],
                                  api.Limits(60, 100000, 512))
        self.assertTrue(result["proven_optimal"])
        self.assertAlmostEqual(result["upper_bound"], 5.01)
        replay = api.rollout(env, result["plan"], strict=True)
        self.assertEqual(replay["objective"]["send_count"], 1)
        self.assertEqual(replay["objective"]["positive_success_count"], 2)
        self.assertEqual(replay["objective"]["initial_task_loss_sum"], 5.)
        certificate = api.exact_search(env, result["plan"], [5., 0., 0.], api.Limits(0, 1, 512),
                                       certified_root_lower_bound=5.01)
        self.assertTrue(certificate["proven_optimal"])
        self.assertEqual(certificate["exact_visited_nodes"], 0)

    def test_independent_exhaustion_of_tiny_episode(self):
        api = self.api()
        env = tiny_env()
        env.reset()
        leaves = []

        def brute(e):
            if e.current_observation is None:
                leaves.append(e.objective_summary["objective_sum"])
                return
            obs = e.current_observation
            # No U/R at reset or later (relay off, no local vehicle sensing).
            for size in range(3):
                for ordered in itertools.permutations(obs.downlink_candidates, size):
                    action = RiskSchedulingAction(downlinks=ordered)
                    if e._is_valid_action(action, obs):
                        child = api.clone_env(e)
                        child.step(action)
                        brute(child)

        brute(env)
        self.assertGreater(len(leaves), 1000)
        self.assertAlmostEqual(min(leaves), 5.01)

    def test_original_environment_and_lookup_acceleration_agree(self):
        api = self.api()
        import random
        from dataclasses import asdict
        runtime = load_runtime(ROOT / "data/config/train.yaml")
        for relay in (False, True):
            cfg = replace(load_risk_config(ROOT / "data/config/risk.yaml"), enable_v2v_relay=relay)
            original = RiskCommEnv(runtime, cfg)
            original.reset()
            fast = api.accelerate(original)
            rng = random.Random(918)
            while original.current_observation is not None:
                obs = original.current_observation
                pools = [list(obs.uplink_candidates), list(obs.downlink_candidates), list(obs.relay_candidates)]
                for pool in pools:
                    rng.shuffle(pool)
                proposal = RiskSchedulingAction(*(tuple(pool) for pool in pools))
                action = api.repair(proposal, obs, cfg)
                a, b = original.step(action), fast.step(action)
                self.assertEqual(asdict(a), asdict(b))
                self.assertEqual(api.cache_payload(original), api.cache_payload(fast))

    def test_budget_stop_preserves_feasible_ub_and_valid_lb(self):
        api = self.api()
        env = tiny_env()
        env.reset()
        result = api.exact_search(env, [RiskSchedulingAction()]*3, [5., 0., 0.], api.Limits(0, 1, 512))
        self.assertFalse(result["proven_optimal"])
        self.assertEqual((result["lower_bound"], result["upper_bound"]), (5., 15.))
        self.assertEqual(api.rollout(env, result["plan"])["objective"]["objective_sum"], 15.)

    def test_memory_monitor_reuses_windows_ctypes(self):
        api = self.api()
        import ctypes
        import os
        if os.name != "nt":
            self.skipTest("Windows专用监测回归。")
        api.memory_mib()
        before = len(ctypes._pointer_type_cache)
        for _ in range(128):
            self.assertGreater(api.memory_mib(), 0.)
        self.assertEqual(len(ctypes._pointer_type_cache), before)

    def test_same_version_uplink_order_changes_retained_provenance(self):
        api = self.api()

        class Sensor:
            def visible_object_ids(self, owner, frame):
                return (2,) if owner == 3 or owner == 1 and frame.step_index == 1 else ()

        env = tiny_env()
        env._sensor = Sensor()
        env.reset()
        env.step(RiskSchedulingAction())
        first = api.clone_env(env)
        second = api.clone_env(env)
        first.step(RiskSchedulingAction(uplinks=(UplinkAction(1, 2), UplinkAction(3, 2))))
        second.step(RiskSchedulingAction(uplinks=(UplinkAction(3, 2), UplinkAction(1, 2))))
        a = first._rsu_knowledge.received_snapshot_for(2)
        b = second._rsu_knowledge.received_snapshot_for(2)
        self.assertEqual((a.source_id, a.velocity_x, a.received_step), (1, None, 2))
        self.assertEqual((b.source_id, b.velocity_previous_observation_step, b.received_step), (3, 0, 2))
        self.assertAlmostEqual(b.velocity_x, 5/.3)
        self.assertNotEqual(api.state_key(first), api.state_key(second))

    def test_actual_reception_supplies_later_first_local_velocity(self):
        api = self.api()

        class Sensor:
            def visible_object_ids(self, owner, frame):
                return (2,) if owner == 0 or owner == 1 and frame.step_index == 2 else ()

        env = tiny_env()
        env._sensor = Sensor()
        env.reset()
        env.step(RiskSchedulingAction(downlinks=(DownlinkAction(1, 2),)))
        env.step(RiskSchedulingAction())
        local = env._vehicle_knowledge[1].snapshot_for(2)
        self.assertEqual((local.source_id, local.observation_step, local.velocity_basis,
                          local.velocity_previous_observation_step), (1, 2, "received_history", 0))
        self.assertAlmostEqual(local.velocity_x, 10/.6)
        original_key = api.state_key(env)
        api.rollout(env, [RiskSchedulingAction()], from_current=True)
        self.assertEqual(api.state_key(env), original_key)

    def test_exclusive_output_and_action_round_trip(self):
        try:
            io = importlib.import_module("experiment")
        except ImportError as exc:
            self.fail(f"独立证据入口尚未实现：{exc}")
        import uuid
        import json
        # 检查产物保留在工作区，遵守本次不删除文件的要求。
        path = ROOT / "outputs" / ("oracle_io_check_" + uuid.uuid4().hex)
        writer = io.Writer(path)
        writer.json("value.json", {"中文": 1})
        with self.assertRaises(FileExistsError):
            io.Writer(path)
        self.assertEqual(json.loads((path / "value.json").read_text(encoding="utf-8")), {"中文": 1})
        actions = [RiskSchedulingAction(uplinks=(UplinkAction(3, 2),)), RiskSchedulingAction()]
        self.assertEqual(io.decode_plan(io.encode_plan(actions)), actions)

    def test_full_actions_match_independent_brute_force(self):
        api = self.api()
        env = tiny_env(True)

        class Sensor:
            def visible_object_ids(self, owner, frame):
                return (2,) if owner == 3 else (3,) if owner == 1 else (1, 2, 3) if owner == 0 else ()

        env._sensor = Sensor()
        obs = env.reset()
        # Independently enumerate every ordered tuple, then ask the original validator.
        pools = (obs.uplink_candidates[:3], obs.downlink_candidates[:4], obs.relay_candidates[:2])
        obs = replace(obs, uplink_candidates=pools[0], downlink_candidates=pools[1], relay_candidates=pools[2])
        expected = set()
        for u in range(3):
            for d in range(3):
                for r in range(2):
                    for x, y, z in itertools.product(itertools.permutations(pools[0], u),
                                                      itertools.permutations(pools[1], d),
                                                      itertools.permutations(pools[2], r)):
                        a = RiskSchedulingAction(x, y, z)
                        if env._is_valid_action(a, obs):
                            expected.add(a)
        self.assertEqual(set(api.legal_actions(obs, env._risk_config)), expected)
        self.assertIn(RiskSchedulingAction(), expected)
        self.assertTrue(any(a.uplinks and a.downlinks for a in expected))
        self.assertTrue(any(a.relays for a in expected))

    def test_relay_transports_original_version_one_step_later(self):
        api = self.api()
        env = tiny_env(True)
        env.reset()
        env.step(RiskSchedulingAction(downlinks=(DownlinkAction(3, 2),)))
        relay = next(c for c in env.current_observation.relay_candidates
                     if c.sender_id == 3 and c.receiver_id == 1 and c.object_id == 2)
        result = env.step(RiskSchedulingAction(relays=(relay,)))
        self.assertFalse(result.invalid_action)
        delivered = result.delivery_records[0]
        self.assertEqual((delivered.send_step, delivered.usable_step, delivered.snapshot.source_id,
                          delivered.snapshot.observation_step, delivered.snapshot.received_step), (1, 2, 0, 0, 2))
        self.assertIsNone(delivered.snapshot.velocity_x)
        self.assertEqual(api.cache_payload(env)["vehicles"][1]["received"][2]["source_id"], 0)

    def test_newer_snapshot_can_increase_negative_loss(self):
        self.api()
        env = tiny_env()
        env.reset()
        task = replace(env.catalog.tasks[1], true_risk=False, related_vehicle_ids=())
        ego = RiskSnapshot(1, 1, 2, 0, 0, -10, 0, 0, 0, 10/.6, None)
        cache = KnowledgeCache(1)
        old = RiskSnapshot(2, 0, 0, -20, 0, 0, 0, 0, 10/.6, 0, None)
        cache.receive(old, 1)
        before = score_event(evaluate_decision(task, execute_risk(task, cache, 2, .6, (0., .3, .6),
                                                                 env._risk_config, ego)))
        cache.receive(replace(old, observation_step=1, x=-15, heading_deg=90), 2)
        after = score_event(evaluate_decision(task, execute_risk(task, cache, 2, .6, (0., .3, .6),
                                                                env._risk_config, ego)))
        self.assertEqual((before.task_loss, after.task_loss), (0., 1.))

    def test_snapshot_origin_velocity_and_clone_isolation(self):
        api = self.api()
        env = tiny_env()
        env.reset()
        a = RiskSnapshot(2, 3, 0, -20, 0, 0, 0, 90, 10, 0, None, None, "local_history")
        env._rsu_knowledge.receive(a, 1)
        child = api.clone_env(env)
        key = api.state_key(child)
        child._rsu_knowledge.receive(replace(a, source_id=1, velocity_x=None), 1)
        self.assertEqual(child._rsu_knowledge.received_snapshot_for(2).source_id, 3)
        child._rsu_knowledge._received_snapshots[2] = replace(a, received_step=1, source_id=1,
                                                           velocity_x=None, velocity_basis=None)
        self.assertNotEqual(key, api.state_key(child))
        self.assertEqual(env._rsu_knowledge.received_snapshot_for(2).velocity_x, 10)

    def test_delayed_delivery_deadline_and_last_step_cost(self):
        api = self.api()
        env = tiny_env()
        no = api.rollout(env, [RiskSchedulingAction()]*3, strict=True)
        last = [RiskSchedulingAction(), RiskSchedulingAction(),
                RiskSchedulingAction(downlinks=(DownlinkAction(1, 2),))]
        late = api.rollout(env, last, strict=True)
        self.assertAlmostEqual(late["objective"]["objective_sum"], 15.01)
        self.assertEqual(no["task_scores"], late["task_scores"])
        timely = [RiskSchedulingAction(), RiskSchedulingAction(downlinks=(DownlinkAction(1, 2),)),
                  RiskSchedulingAction()]
        ok = api.rollout(env, timely, strict=True)
        self.assertAlmostEqual(ok["objective"]["objective_sum"], 5.01)
        self.assertEqual(ok["results"][1].delivery_records[0].usable_step, 2)

    def test_wrong_object_alarm_is_strict_failure(self):
        self.api()
        env = tiny_env()
        env.reset()
        task = env.catalog.tasks[1]
        cache = KnowledgeCache(1)
        cache.receive(RiskSnapshot(3, 0, 1, -15, 0, 0, 0, 90, 16.6666666667, 0, None), 2)
        ego = RiskSnapshot(1, 1, 2, 0, 0, -10, 0, 0, 0, 16.6666666667, None)
        decision = execute_risk(task, cache, 2, .6, (0., .3, .6), env._risk_config, ego)
        score = score_event(evaluate_decision(task, decision))
        self.assertEqual(decision.outcome, "risk")
        self.assertEqual(score.task_loss, 5.)
        self.assertTrue(score.wrong_object_only_alarm)

    def test_relaxation_is_below_tiny_optimum(self):
        api = self.api()
        env = tiny_env(True)
        env.reset()
        lower = api.relaxed_bound(env)
        self.assertLessEqual(sum(x["loss_lower_bound"] for x in lower), 5.01)
        self.assertEqual(lower[0]["loss_lower_bound"], 5.)

    def test_fixed_receiver_lower_bound_certificate(self):
        api = self.api()
        try:
            cert = importlib.import_module("certificates")
        except ImportError as exc:
            self.fail(f"固定场景接收次数证明尚未实现：{exc}")
        runtime = load_runtime(ROOT / "data/config/train.yaml")
        for relay, expected in ((False, 11.37), (True, 11.14)):
            env = RiskCommEnv(runtime, replace(load_risk_config(ROOT / "data/config/risk.yaml"),
                                               enable_v2v_relay=relay))
            env.reset()
            result = cert.fixed_receiver_certificate(env, api.relaxed_bound(env))
            self.assertEqual(result["lower_bound"], expected)
            self.assertTrue(result["conditions_verified"])

    def test_coupled_refinement_repairs_persistent_missing_velocity(self):
        api = self.api()
        try:
            cert = importlib.import_module("certificates")
        except ImportError as exc:
            self.fail(f"固定联合邻域尚未实现：{exc}")
        # 合成三步实例，重命名为该固定邻域的角色；不另行优化完整场景。
        base = tiny_env(True)
        ids = {1: 7019, 2: 13138, 3: 1380}
        frames = tuple(replace(f, vehicles={ids[k]: replace(v, vehicle_id=ids[k])
                                            for k, v in f.vehicles.items()}) for f in base._replay.frames)
        runtime = replace(base._runtime, replay=VehicleReplay(frames))
        env = RiskCommEnv(runtime, base._risk_config)
        tasks = tuple(replace(t, vehicle_id=7019, related_vehicle_ids=(13138,)) for t in base.catalog.tasks)
        env._catalog = ConflictCatalog((), (), tasks)
        env._tasks_by_step = {0: (tasks[0],), 2: tasks[1:]}

        class Sensor:
            def visible_object_ids(self, owner, frame):
                return (13138,) if owner == 1380 else (7019, 13138, 1380) if owner == 0 else ()

        env._sensor = Sensor()
        env.reset()
        plan, stats = cert.refine_coupled_receptions(env, [RiskSchedulingAction()]*3,
                                                    api.Limits(30, 1000, 512), 5.)
        result = api.rollout(env, plan)
        self.assertEqual(stats["status"], "improved")
        self.assertAlmostEqual(result["objective"]["objective_sum"], 5.02)
        self.assertEqual((len(plan[0].relays), len(plan[1].downlinks)), (1, 1))

    def test_invalid_relay_conflict_is_atomic(self):
        api = self.api()
        env = tiny_env(True)
        env.reset()
        env.step(RiskSchedulingAction(downlinks=(DownlinkAction(3, 2),)))
        obs = env.current_observation
        relay = next(c for c in obs.relay_candidates if c.sender_id == 3 and c.receiver_id == 1)
        result = env.step(RiskSchedulingAction(downlinks=(DownlinkAction(3, 1),), relays=(relay,)))
        self.assertTrue(result.invalid_action)
        self.assertEqual(result.feedback.send_count, 0)
        single = api.clone_env(env)
        self.assertEqual(single._vehicle_knowledge[3].received_snapshot_for(2).observation_step, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
