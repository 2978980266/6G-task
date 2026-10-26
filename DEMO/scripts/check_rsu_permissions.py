"""RSU 权限定向回归；仅合成回放与推理，不训练、不搜索、不改默认配置。"""
from __future__ import annotations

from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skylines_demo.bootstrap import ScenarioRuntime, load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.geometry import BuildingLineOfSight
from skylines_demo.perception.communication import SnapshotCommunication
from skylines_demo.perception.link_forecast import predict_downlink_window, project_snapshot_position
from skylines_demo.policies.risk_baselines import select_deadline_action, select_relation_action
from skylines_demo.policies.risk_oracle import select_diagnostic_oracle_action
from skylines_demo.policies.risk_public_uplink import select_public_uplink_action
from skylines_demo.policies.risk_receiver_policy import select_receiver_value_action
from skylines_demo.policies.risk_relay_policy import select_receiver_value_with_relay_action
from skylines_demo.risk.conflicts import ConflictCatalog, RiskTask
from skylines_demo.risk.model import predict_pair
from skylines_demo.risk.sufficiency import predicted_object_distance_to_task
from skylines_demo.risk_learning.domain import build_safe_step
from skylines_demo.types import (
    DownlinkAction, FrameRecord, RelayAction, RiskSchedulingAction, RiskSnapshot,
    UplinkAction, Vector3, VehicleReplay, VehicleSnapshot,
)
from skylines_demo.wireless import LogDistanceChannelModel


class FixedVisibility:
    """固定允许的 ID/版本摘要；私有运动变化不改变这些控制输入。"""

    def visible_object_ids(self, owner, frame):
        return {0: (2,), 1: (), 2: (), 3: (1,)}[owner]


def fixture_env(*, private_x=0., perturb_step=2, private_helper_x=30.,
                prediction=False, mode="B", relay=False):
    base = load_runtime(ROOT / "data/config/train.yaml")
    frames = tuple(FrameRecord(t, t, "fixture", "fixture", .5*t, .5*t, .5, {
        1: VehicleSnapshot(1, Vector3(private_x if t >= perturb_step else 0., 0, -20+5*t),
                           999., 0., 0., "private"),
        2: VehicleSnapshot(2, Vector3(-20+5*t, 0, 0), 999., 0., 90., "object"),
        3: VehicleSnapshot(3, Vector3(private_helper_x, 0, 30), 999., 0., 0., "helper"),
    }) for t in range(5))
    los = BuildingLineOfSight([])
    runtime = ScenarioRuntime(replace(base.config, rsu=replace(base.config.rsu, position=Vector3(0, 30, 0))),
                              VehicleReplay(frames), [], los, LogDistanceChannelModel(base.config.channel, los))
    cfg = replace(load_risk_config(ROOT / "data/config/risk.yaml"), link_prediction=prediction,
                  enable_v2v_relay=relay,
                  ego_eta_mode="causal-self-motion" if mode == "B" else "public-crossing-step")
    env = RiskCommEnv(runtime, cfg)
    task = RiskTask("fixture-positive", 1, 0, Vector3(0, 0, 0), 2., 4, 3, True, (2,))
    env._catalog = ConflictCatalog((), (), (task,))
    env._tasks_by_step = {3: (task,)}
    env._sensor = FixedVisibility()
    # 固定允许的物理链路反馈；此测试不把真实位置导致的链路变化当泄漏。
    env._link_reachable = lambda sender, receiver: True
    return env


def received_receiver_env(**kwargs):
    env = fixture_env(**kwargs)
    env.reset()
    for _ in range(2):
        result = env.step(RiskSchedulingAction(uplinks=(UplinkAction(3, 1),)))
        assert not result.invalid_action
    return env


class RSUPermissionChecks(unittest.TestCase):
    def test_online_vehicle_summaries_contain_no_private_motion(self):
        env = fixture_env()
        obs = env.reset()
        for state in obs.active_vehicles:
            self.assertFalse(hasattr(state, "snapshot"), "在线摘要泄漏真实当前位置")
            self.assertFalse(hasattr(state, "self_snapshot"), "在线摘要泄漏车端自身运动")
        self.assertFalse(hasattr(obs, "replay"))
        self.assertEqual(obs.rsu_known_object_ids, (2,))
        self.assertEqual(tuple(s.object_id for s in obs.rsu_snapshots), (2,))

    def test_private_motion_does_not_change_online_input_with_fixed_allowed_inputs(self):
        for mode in ("B", "A"):
            for relay in (False, True):
                for prediction in (False, True):
                    with self.subTest(mode=mode, relay=relay, prediction=prediction):
                        a = received_receiver_env(mode=mode, relay=relay, prediction=prediction)
                        b = received_receiver_env(mode=mode, relay=relay, prediction=prediction, private_x=777.)
                        self.assertEqual(asdict(a.current_observation), asdict(b.current_observation))

    def test_private_motion_does_not_change_ppo_candidates_features_or_rule_actions(self):
        for mode in ("B", "A"):
            for relay in (False, True):
                a = received_receiver_env(mode=mode, relay=relay)
                b = received_receiver_env(mode=mode, relay=relay, private_x=777.)
                x, y, cfg = a.current_observation, b.current_observation, a._risk_config
                sx, sy = build_safe_step(x, cfg), build_safe_step(y, cfg)
                self.assertTrue(any(c.key[0] == "D" for c in sx.candidates))
                self.assertEqual(sx, sy)
                selectors = (select_deadline_action, select_receiver_value_action,
                             select_receiver_value_with_relay_action,
                             lambda obs, c: select_relation_action(obs, c, related=True),
                             lambda obs, c: select_public_uplink_action(obs, c, mode="demand", with_relay=relay))
                for selector in selectors:
                    self.assertEqual(selector(x, cfg), selector(y, cfg))

    def test_ppo_encoding_probabilities_and_decoded_actions_use_the_same_safe_input(self):
        from skylines_demo.risk_learning.encoding import encode_step
        from skylines_demo.risk_learning.evidence import action_from_path
        from skylines_demo.risk_learning.policy import RiskCandidatePolicy
        from skylines_demo.risk_learning.training import initialize_seed
        import numpy as np

        initialize_seed(19)
        policy = RiskCandidatePolicy()
        for mode in ("B", "A"):
            a = received_receiver_env(mode=mode, relay=True)
            b = received_receiver_env(mode=mode, relay=True, private_x=777.)
            x = build_safe_step(a.current_observation, a._risk_config)
            y = build_safe_step(b.current_observation, b._risk_config)
            ex, ey = encode_step(x), encode_step(y)
            np.testing.assert_array_equal(ex.candidate_features, ey.candidate_features)
            np.testing.assert_array_equal(ex.global_features, ey.global_features)
            px, py = policy.greedy(ex, x), policy.greedy(ey, y)
            self.assertEqual(px, py)
            self.assertEqual(action_from_path(x, px), action_from_path(y, py))

    def test_rsu_local_history_does_not_grant_state_after_visibility_is_lost(self):
        class FadingVisibility(FixedVisibility):
            def visible_object_ids(self, owner, frame):
                if owner == 0 and frame.step_index < 2:
                    return (1, 2)
                return super().visible_object_ids(owner, frame)

        env = fixture_env()
        env._sensor = FadingVisibility()
        env.reset()
        env.step(RiskSchedulingAction())
        obs = env.step(RiskSchedulingAction()).next_observation
        self.assertIn((0, 1), env._local_history)
        self.assertNotIn(1, obs.rsu_known_object_ids)
        self.assertIsNone(obs.snapshot_for(1))

    def test_default_step_zero_sensing_still_sees_all_sixteen_cars(self):
        env = RiskCommEnv(load_runtime(ROOT / "data/config/train.yaml"),
                          load_risk_config(ROOT / "data/config/risk.yaml"))
        obs = env.reset()
        self.assertEqual(len(obs.rsu_visible_object_ids), 16)
        self.assertEqual(len(obs.rsu_snapshots), 16)
        self.assertTrue(all(s.source_id == 0 and s.observation_step == 0 and s.received_step is None
                            for s in obs.rsu_snapshots))

    def test_unknown_receiver_is_not_filled_from_its_self_motion(self):
        env = fixture_env()
        env.reset()
        obs = env.step(RiskSchedulingAction()).next_observation
        candidate = DownlinkAction(1, 2)
        self.assertIn(candidate, obs.downlink_candidates)
        self.assertTrue(math.isinf(predicted_object_distance_to_task(obs, candidate, obs.tasks[0], env._risk_config)))
        self.assertFalse(any(c.key[0] == "D" for c in build_safe_step(obs, env._risk_config).candidates))

    def test_uplink_usable_next_step_preserves_source_and_original_version(self):
        env = fixture_env()
        before = env.reset()
        self.assertNotIn(1, before.rsu_known_object_ids)
        result = env.step(RiskSchedulingAction(uplinks=(UplinkAction(3, 1),)))
        self.assertNotIn(1, result.observation.rsu_known_object_ids)
        received = next(s for s in result.next_observation.rsu_snapshots if s.object_id == 1)
        self.assertEqual((received.source_id, received.observation_step, received.received_step), (3, 0, 1))
        self.assertIsNone(received.velocity_x)
        self.assertIsNone(received.velocity_basis)
        self.assertEqual((result.delivery_records[0].send_step, result.delivery_records[0].usable_step), (0, 1))

    def test_old_received_receiver_without_speed_stays_unknown(self):
        env = fixture_env()
        env.reset()
        env.step(RiskSchedulingAction(uplinks=(UplinkAction(3, 1),)))
        obs = env.step(RiskSchedulingAction()).next_observation
        self.assertTrue(math.isinf(predicted_object_distance_to_task(
            obs, DownlinkAction(1, 2), obs.tasks[0], env._risk_config)))
        self.assertFalse(any(c.key[0] == "D" for c in build_safe_step(obs, env._risk_config).candidates))

    def test_received_receiver_ages_without_rewriting_original_snapshot(self):
        env = received_receiver_env()
        obs = env.current_observation
        receiver = next(s for s in obs.rsu_snapshots if s.object_id == 1)
        self.assertEqual((receiver.source_id, receiver.observation_step, receiver.received_step,
                          receiver.velocity_previous_observation_step, receiver.velocity_basis),
                         (3, 1, 2, 0, "local_history"))
        obj = next(s for s in obs.rsu_snapshots if s.object_id == 2)
        prediction = predict_pair(receiver, obj, obs.tasks[0].conflict_point,
                                  obs.elapsed_seconds, obs.step_times, env._risk_config)
        self.assertIs(prediction.risk, True)
        self.assertAlmostEqual(prediction.ego_arrival_seconds, 1.)
        self.assertEqual(receiver.z, -15.)
        self.assertAlmostEqual(predicted_object_distance_to_task(obs, DownlinkAction(1, 2), obs.tasks[0],
                                                               env._risk_config), 0.)

    def test_vehicle_keeps_self_motion_outside_the_rsu_interface(self):
        class SelfVisibility(FixedVisibility):
            def visible_object_ids(self, owner, frame):
                return (2,) if owner == 1 else super().visible_object_ids(owner, frame)

        env = fixture_env()
        env._sensor = SelfVisibility()
        obs = env.reset()
        self.assertTrue(callable(getattr(env, "world_diagnostics", None)), "缺少独立世界诊断接口")
        while obs.step_index < 3:
            obs = env.step(RiskSchedulingAction()).next_observation
        world = env.world_diagnostics()
        own = next(v.self_snapshot for v in world.active_vehicles if v.vehicle_id == 1)
        self.assertEqual(own.source_id, 1)
        self.assertAlmostEqual(own.velocity_z, 10.)
        self.assertNotIn(1, obs.rsu_known_object_ids)
        result = env.step(RiskSchedulingAction())
        self.assertEqual(result.risk_decisions[0][1].outcome, "risk")
        self.assertEqual(result.feedback.task_scores[0].task_loss, 0.)

    def test_b_public_view_does_not_publish_crossing_step(self):
        env = fixture_env()
        self.assertIsNone(env.reset().tasks[0].crossing_step)
        self.assertEqual(fixture_env(mode="A").reset().tasks[0].crossing_step, 4)

    def test_b_policies_do_not_even_read_crossing_step(self):
        class BTask:
            task_id = "poison-crossing"
            vehicle_id = 1
            conflict_point_id = 0
            conflict_point = Vector3(0, 0, 0)
            decision_step = 3

            @property
            def crossing_step(self):
                raise AssertionError("B 读取了 crossing_step")

        env = received_receiver_env()
        obs = replace(env.current_observation, tasks=(BTask(),))
        self.assertAlmostEqual(predicted_object_distance_to_task(obs, DownlinkAction(1, 2), obs.tasks[0],
                                                               env._risk_config), 0.)
        self.assertTrue(any(c.key[0] == "D" for c in build_safe_step(obs, env._risk_config).candidates))

    def test_window_forecast_accepts_legal_snapshots_without_a_replay_reference(self):
        params = inspect.signature(predict_downlink_window).parameters
        self.assertNotIn("replay_history", params, "窗口预测仍可读取完整世界历史")
        self.assertIn("rsu_snapshots", params)
        env = received_receiver_env(prediction=True)
        obs = env.current_observation
        forecast = next(f for receiver, obj, f in obs.link_forecasts if (receiver, obj) == (1, 2))
        self.assertTrue(forecast.available)
        self.assertEqual(forecast.predicted_steps[0], obs.step_index + env._risk_config.decision_lead_steps)

    def test_window_prediction_unknown_receiver_is_unavailable(self):
        env = fixture_env(prediction=True)
        env.reset()
        obs = env.step(RiskSchedulingAction()).next_observation
        forecast = next(f for receiver, obj, f in obs.link_forecasts if (receiver, obj) == (1, 2))
        self.assertFalse(forecast.available, "窗口预测借用了未上报的接收车运动")
        self.assertEqual(forecast.reachable_steps, ())
        self.assertIsNone(forecast.first_unreachable_step, "未知不能写成已知断链")

    def test_projection_uses_original_snapshot_age_and_rejects_missing_motion(self):
        snapshot = RiskSnapshot(1, 3, 0, 0, 0, -20, 999, 0, 0, 10, 1, None, "local_history")
        self.assertEqual(project_snapshot_position(snapshot, 1., (0., .5, 1.)), Vector3(0, 0, -10))
        self.assertIsNone(project_snapshot_position(replace(snapshot, velocity_z=None), 1., (0., .5, 1.)))
        self.assertIsNone(project_snapshot_position(replace(snapshot, observation_step=3), 1., (0., .5, 1.)))

    def test_window_prediction_ages_known_blockers_without_future_times(self):
        class Channel:
            def reachable(self, start, end, sender_id, blockers):
                # Current time=1, old receiver@0 and blocker@.5. First predicted
                # time=1.5, then 2; only the second known-blocker geometry passes.
                return end == Vector3(2, 0, 0) and blockers == (Vector3(7, 0, 2),)

        env = fixture_env()
        cfg = replace(env._risk_config, decision_lead_steps=1, prediction_horizon_steps=2)
        receiver = RiskSnapshot(1, 3, 0, 0, 0, 0, 999, 90, 1, 0, 1)
        blocker = RiskSnapshot(2, 0, 1, 4, 0, 2, 999, 90, 2, 0, None)
        kwargs = dict(snapshot=blocker, receiver_id=1, current_step=2,
                      rsu_snapshots=(receiver, blocker), step_times=(0., .5, 1., 10000.), config=cfg,
                      communication_model=Channel(), current_time_seconds=1., fallback_step_seconds=.3)
        forecast = predict_downlink_window(**kwargs)
        self.assertEqual(forecast.predicted_steps, (3, 4))
        self.assertEqual(forecast.reachable_steps, (4,))
        self.assertEqual(forecast.first_unreachable_step, 3)
        kwargs["rsu_snapshots"] = (receiver, replace(blocker, velocity_x=None))
        unknown = predict_downlink_window(**kwargs)
        self.assertFalse(unknown.available)
        self.assertIsNone(unknown.first_unreachable_step)

    def test_unreported_blocker_motion_does_not_enter_window_prediction(self):
        class Channel:
            def evaluate(self, start, end, blockers=(), **kwargs):
                return SimpleNamespace(reachable=not any(p.x > 500 for p in blockers))

        forecasts = []
        for helper_x in (30., 1000.):
            env = fixture_env(prediction=True, private_helper_x=helper_x)
            env._communication = SnapshotCommunication(Channel(), env._runtime.config.resource)
            env.reset()
            for _ in range(2):
                env.step(RiskSchedulingAction(uplinks=(UplinkAction(3, 1),)))
            obs = env.current_observation
            self.assertNotIn(3, obs.rsu_known_object_ids)
            forecast = next(f for r, o, f in obs.link_forecasts if (r, o) == (1, 2))
            self.assertTrue(forecast.available)
            self.assertEqual(forecast.reachable_steps, forecast.predicted_steps)
            forecasts.append(forecast)
        self.assertEqual(*forecasts)

    def test_model_metadata_records_rsu_permission_separately_from_vehicle_model(self):
        from skylines_demo.risk_learning.metadata import contract
        from skylines_demo.risk_learning.training import TrainingSpec
        value = contract(TrainingSpec("B", False, 7, 2, "smoke"), {})
        self.assertIn("rsu_information_contract", value)
        self.assertEqual(value["rsu_information_contract"]["version"], "rsu-legal-knowledge-v1")

    def test_new_freeze_records_the_authorized_transition_and_rejects_drift(self):
        from skylines_demo.risk_learning.metadata import freeze_manifest
        from skylines_demo.risk_learning.source_contract import (
            LEGACY_FROZEN_SOURCE_FINGERPRINT, validate_frozen_sources,
        )
        manifest = freeze_manifest(ROOT / "data/config/train.yaml", ROOT / "data/config/risk.yaml")
        self.assertEqual(manifest["source_contract"]["parent_frozen_source_fingerprint"],
                         LEGACY_FROZEN_SOURCE_FINGERPRINT)
        self.assertNotEqual(manifest["source_contract"]["frozen_source_fingerprint"],
                            LEGACY_FROZEN_SOURCE_FINGERPRINT)
        changed = dict(manifest["frozen_source_hashes"])
        changed["DEMO/src/skylines_demo/wireless/log_distance.py"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_frozen_sources(changed)

    def test_oracle_entry_checks_the_same_versioned_freeze_without_solving(self):
        sys.path.insert(0, str(ROOT / "oracle_experiments"))
        from experiment import manifest as oracle_manifest
        from skylines_demo.risk_learning.metadata import freeze_manifest

        env = RiskCommEnv(load_runtime(ROOT / "data/config/train.yaml"),
                          load_risk_config(ROOT / "data/config/risk.yaml"))
        oracle = oracle_manifest(env)
        learning = freeze_manifest(ROOT / "data/config/train.yaml", ROOT / "data/config/risk.yaml")
        self.assertEqual(oracle["source_contract"], learning["source_contract"])
        self.assertEqual(oracle["source_fingerprint"], learning["source_fingerprint"])

    def test_new_untrained_checkpoint_roundtrip_and_strict_rejection(self):
        from uuid import uuid4
        from skylines_demo.risk_learning.evidence import ExclusiveWriter
        from skylines_demo.risk_learning.metadata import freeze_manifest, load_risk_policy, save_checkpoint
        from skylines_demo.risk_learning.policy import RiskCandidatePolicy
        from skylines_demo.risk_learning.training import TrainingSpec, initialize_seed

        initialize_seed(19)
        spec = TrainingSpec("B", False, 19, 2, "smoke")
        manifest = freeze_manifest(ROOT / "data/config/train.yaml", ROOT / "data/config/risk.yaml")
        writer = ExclusiveWriter(ROOT / "outputs" / ("rsu_untrained_checkpoint_check_" + uuid4().hex))
        model = writer.directory / "episode_0000.pt"
        policy = RiskCandidatePolicy()
        save_checkpoint(model, policy, spec, manifest, completed_episodes=0, updates=0,
                        optimizer_state={}, rng_state={})
        loaded = load_risk_policy(model, spec, manifest)
        for key, tensor in policy.state_dict().items():
            self.assertTrue(tensor.equal(loaded.state_dict()[key]))
        with self.assertRaises(FileExistsError):
            save_checkpoint(model, policy, spec, manifest, completed_episodes=0, updates=0,
                            optimizer_state={}, rng_state={})
        changed_manifest = dict(manifest, rsu_information_contract={"version": "wrong"})
        with self.assertRaises(ValueError):
            load_risk_policy(model, spec, changed_manifest)
        writer.json("verification.json", dict(passed=True, training_started=False, optimizer_updates=0,
                                               strict_reload=True, overwrite_and_contract_drift_rejected=True))

    def test_historical_smoke_models_are_rejected_without_changing_their_sidecars(self):
        import hashlib
        from skylines_demo.risk_learning.metadata import load_risk_policy, read_spec

        history = ROOT / "outputs/risk_learning_20261010_112722_812903"
        models = [history / f"smoke_{mode}_{relay}_seed7/episode_0002.pt"
                  for mode in ("B", "A") for relay in ("off", "on")]
        if not all(path.exists() for path in models):
            self.skipTest("未附带历史烟雾模型；新合同不提供旧模型兼容入口。")
        for path in models:
            sidecar = path.with_suffix(".metadata.json")
            before = (hashlib.sha256(path.read_bytes()).hexdigest(), sidecar.read_bytes())
            with self.assertRaises(ValueError) as error:
                load_risk_policy(path, read_spec(path), {})
            self.assertIn("RSU运动权限合同不兼容", str(error.exception.__cause__))
            self.assertEqual(before, (hashlib.sha256(path.read_bytes()).hexdigest(), sidecar.read_bytes()))

    def test_existing_oracle_plans_keep_physical_execution_decisions_and_scores(self):
        def normalize(value):
            return json.loads(json.dumps(value))

        runtime = load_runtime(ROOT / "data/config/train.yaml")
        for relay, expected in ((False, 12.10), (True, 11.14)):
            directory = ROOT / f"outputs/offline_oracle_B_{'on' if relay else 'off'}_20261015_01/certified_best"
            if not (directory / "replay.jsonl").exists():
                self.skipTest("未附带历史 oracle 动作；不启动求解。")
            env = RiskCommEnv(runtime, replace(load_risk_config(ROOT / "data/config/risk.yaml"),
                                               enable_v2v_relay=relay))
            env.reset()
            scores = []
            for line in (directory / "replay.jsonl").read_text(encoding="utf-8").splitlines():
                old = json.loads(line)
                action = old["action"]
                result = env.step(RiskSchedulingAction(
                    tuple(UplinkAction(**c) for c in action["uplinks"]),
                    tuple(DownlinkAction(**c) for c in action["downlinks"]),
                    tuple(RelayAction(**c) for c in action["relays"])))
                self.assertFalse(result.invalid_action)
                self.assertEqual(normalize([asdict(r) for r in result.delivery_records]), old["delivery_records"])
                self.assertEqual(normalize([asdict(e) for e in result.event_results]), old["event_results"])
                self.assertEqual(normalize([{"task_id": k, "decision": asdict(d)}
                                            for k, d in result.risk_decisions]), old["risk_decisions"])
                scores.extend(asdict(s) for s in result.feedback.task_scores)
            self.assertIsNone(env.current_observation)
            self.assertEqual(len(scores), 46)
            self.assertEqual(json.loads(json.dumps(scores)),
                             json.loads((directory / "task_scores.json").read_text(encoding="utf-8")))
            self.assertAlmostEqual(env.objective_summary["objective_sum"], expected)

    def test_oracle_extra_truth_selects_only_environment_legal_snapshots(self):
        env = fixture_env()
        obs = env.reset()
        action = select_diagnostic_oracle_action(obs, env._risk_config, env.catalog,
                                                replay=env._replay, communication=env._communication)
        self.assertIn(DownlinkAction(1, 2), action.downlinks)
        result = env.step(action)
        self.assertFalse(result.invalid_action)
        self.assertEqual(result.delivered_downlinks[0].source_id, 0)
        before = tuple(env._rsu_knowledge.known_object_ids)
        invalid = env.step(RiskSchedulingAction(downlinks=(DownlinkAction(1, 999),)))
        self.assertTrue(invalid.invalid_action)
        self.assertEqual(invalid.feedback.send_count, 0)
        self.assertEqual(before, env._rsu_knowledge.known_object_ids)

    def test_same_step_uplink_cannot_create_a_downlink_or_relay_source(self):
        for relay in (False, True):
            env = fixture_env(relay=relay)
            env.reset()
            action = RiskSchedulingAction(uplinks=(UplinkAction(3, 1),),
                                          downlinks=(DownlinkAction(2, 1),))
            result = env.step(action)
            self.assertTrue(result.invalid_action)
            self.assertEqual(result.feedback.send_count, 0)
        env = fixture_env(relay=True)
        env.reset()
        result = env.step(RiskSchedulingAction(downlinks=(DownlinkAction(3, 2),),
                                               relays=(RelayAction(3, 1, 2),)))
        self.assertTrue(result.invalid_action)
        self.assertEqual(result.feedback.send_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
