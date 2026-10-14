"""显式有限预算训练入口，无默认训练或自动续训。"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skylines_demo.risk_learning.training import TrainingSpec, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="独立风险S0训练；fixed预算仅在用户授权后调用。")
    parser.add_argument("--mode", choices=("B", "A"), required=True)
    parser.add_argument("--relay", choices=("off", "on"), required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--phase", choices=("smoke", "fixed"), required=True)
    parser.add_argument("--domain", default="receiver-risk-domain-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config/train.yaml")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "data/config/risk.yaml")
    args = parser.parse_args()
    try:
        spec = TrainingSpec(args.mode, args.relay == "on", args.train_seed, args.episodes, args.phase, args.domain)
        if args.output_dir.exists():
            raise FileExistsError("输出目录已存在，拒绝覆盖。")
        run_training(spec, args.output_dir, args.config, args.risk_config)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
