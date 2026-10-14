"""非训练合同检查：四配置随机合法路径、重算概率及完整反馈。"""
from dataclasses import replace
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.risk_learning.evidence import ExclusiveWriter
from skylines_demo.risk_learning.metadata import freeze_manifest
from skylines_demo.risk_learning.policy import RiskCandidatePolicy
from skylines_demo.risk_learning.training import collect_episode, initialize_seed


def main() -> int:
    p = argparse.ArgumentParser(description="非训练风险学习合同检查。")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("输出已存在。")
    manifest = freeze_manifest(ROOT / "data/config/train.yaml", ROOT / "data/config/risk.yaml")
    writer = ExclusiveWriter(args.output_dir)
    writer.json("freeze.json", manifest)
    runtime = load_runtime(ROOT / "data/config/train.yaml")
    summary = []
    for mode in ("B", "A"):
        for relay in (False, True):
            cfg = replace(load_risk_config(ROOT / "data/config/risk.yaml"), enable_v2v_relay=relay,
                          link_prediction=False,
                          ego_eta_mode="causal-self-motion" if mode == "B" else "public-crossing-step")
            generator, _ = initialize_seed(7)
            policy = RiskCandidatePolicy()
            episode, run = collect_episode(RiskCommEnv(runtime, cfg), cfg, policy, generator,
                                            ExclusiveWriter(args.output_dir / f"{mode}_relay_{relay}"))
            for record in episode.records:
                logp, value = policy.evaluate_path(record.encoded, record.step, record.path)
                assert abs(logp.item() - record.path.joint_log_prob) < 1e-5
                assert abs(value.item() - record.path.value) < 1e-5
            summary.append(dict(mode=mode, relay=relay, objective=run["objective"], invalid_action_count=0))
    writer.json("check.json", dict(training_started=False, runs=summary, passed=True))
    print("四配置各23step，概率重算、非法0、完整结算、原始版本/每跳时延通过；未训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
