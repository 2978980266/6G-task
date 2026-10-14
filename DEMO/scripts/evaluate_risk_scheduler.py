"""严格重载完整回放；R屏蔽只允许独立诊断并保留父哈希。"""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skylines_demo.bootstrap import load_runtime
from skylines_demo.config import load_risk_config
from skylines_demo.env.risk_comm_env import RiskCommEnv
from skylines_demo.risk_learning.evidence import ExclusiveWriter, fingerprint
from skylines_demo.risk_learning.metadata import file_hash, freeze_manifest, load_risk_policy, read_spec
from skylines_demo.risk_learning.training import collect_episode, initialize_seed


def main() -> int:
    p = argparse.ArgumentParser(description="完整同场景评测，不证明泛化；greedy重复两次。")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--metadata-path", type=Path)
    p.add_argument("--decoding", choices=("greedy", "stochastic"), required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "data/config/train.yaml")
    p.add_argument("--risk-config", type=Path, default=ROOT / "data/config/risk.yaml")
    p.add_argument("--mode", choices=("B", "A"))
    p.add_argument("--disable-relay-diagnostic", action="store_true")
    p.add_argument("--global-cap", type=int, choices=(0, 20, 30, 40, 47))
    args = p.parse_args()
    try:
        if args.output_dir.exists():
            raise FileExistsError("输出目录已存在。")
        if args.repeat != (2 if args.decoding == "greedy" else 10):
            raise ValueError("greedy核验2遍；随机推理补充10个既定seed。")
        spec = read_spec(args.model_path, args.metadata_path)
        if args.mode is not None and args.mode != spec.mode:
            raise ValueError("A/B模式与metadata不匹配。")
        manifest = freeze_manifest(args.config, args.risk_config)
        policy = load_risk_policy(args.model_path, spec, manifest)
        if args.disable_relay_diagnostic and not spec.relay:
            raise ValueError("R屏蔽要求父模型开启中继。")
        config = replace(load_risk_config(args.risk_config), link_prediction=False,
                         ego_eta_mode="causal-self-motion" if spec.mode == "B" else "public-crossing-step",
                         enable_v2v_relay=spec.relay and not args.disable_relay_diagnostic)
        writer = ExclusiveWriter(args.output_dir)
        writer.json("evaluation_manifest.json", dict(training_spec=asdict(spec), freeze=manifest,
                    parent_model_sha256=file_hash(args.model_path), decoding=args.decoding,
                    risk_config=asdict(config), global_cap=args.global_cap,
                    only_diagnostic_change="enable_v2v_relay: true -> false" if args.disable_relay_diagnostic else None))
        runtime = load_runtime(args.config)
        results = []
        for index in range(args.repeat):
            seed = 1000007 + index if args.decoding == "stochastic" else spec.train_seed
            generator, _ = initialize_seed(seed)
            _, run = collect_episode(RiskCommEnv(runtime, config), config, policy, generator,
                                     ExclusiveWriter(args.output_dir / f"repeat_{index+1:02d}"),
                                     "greedy" if args.decoding == "greedy" else "sample", args.global_cap)
            results.append(run)
        if args.decoding == "greedy" and fingerprint(results[0]) != fingerprint(results[1]):
            raise AssertionError("重载贪心完整回放不一致。")
        writer.json("evaluation_summary.json", dict(mode=spec.mode, relay=config.enable_v2v_relay,
                    repeats=[dict(objective=r["objective"], legacy=r["legacy"], fingerprint=fingerprint(r),
                                  invalid_action_count=r["invalid_action_count"]) for r in results],
                    greedy_identical=args.decoding == "greedy", independent_samples=False))
        print("完整回放验证通过；同场景内评测，不宣称性能收益。")
    except (ValueError, FileExistsError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
