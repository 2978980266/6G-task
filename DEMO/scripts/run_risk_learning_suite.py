"""显式固定矩阵；不默认运行、不接续烟雾模型、不选择胜出checkpoint。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skylines_demo.risk_learning.evidence import ExclusiveWriter
from skylines_demo.risk_learning.metadata import freeze_manifest
from skylines_demo.risk_learning.training import TrainingSpec, run_training


def main() -> int:
    p = argparse.ArgumentParser(description="固定训练矩阵，仅在用户明确授权固定预算后执行。")
    p.add_argument("--modes", nargs="+", choices=("B", "A"), required=True)
    p.add_argument("--relay-modes", nargs="+", choices=("off", "on"), required=True)
    p.add_argument("--train-seeds", nargs="+", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--phase", choices=("fixed",), required=True)
    p.add_argument("--domain", default="receiver-risk-domain-v1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "data/config/train.yaml")
    p.add_argument("--risk-config", type=Path, default=ROOT / "data/config/risk.yaml")
    args = p.parse_args()
    try:
        if args.output_dir.exists():
            raise FileExistsError("输出已存在。")
        if (args.modes not in (["B"], ["B", "A"]) or args.relay_modes != ["off", "on"]
                or args.train_seeds != [7, 17, 27, 37, 47]):
            raise ValueError("只接受预设B先/A分表、关开中继、5个完整训练seed。")
        specs = [TrainingSpec(mode, relay == "on", seed, args.episodes, args.phase, args.domain)
                 for mode in args.modes for relay in args.relay_modes for seed in args.train_seeds]
        targets = [args.output_dir / spec.mode / f"relay_{'on' if spec.relay else 'off'}" / f"seed_{spec.train_seed}"
                   for spec in specs]
        if any(t.exists() for t in targets):
            raise FileExistsError("训练矩阵目标已存在。")
        manifest = freeze_manifest(args.config, args.risk_config)
        writer = ExclusiveWriter(args.output_dir)
        writer.json("suite_manifest.json", dict(specs=[asdict(s) for s in specs], freeze=manifest,
                    requested_world_steps=sum(s.world_steps for s in specs),
                    checkpoint_selection="episode_0200_only", fresh_initialization=True,
                    automatic_resume=False, A_executed="A" in args.modes))
        for spec, target in zip(specs, targets):
            run_training(spec, target, args.config, args.risk_config)
    except (ValueError, FileExistsError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
