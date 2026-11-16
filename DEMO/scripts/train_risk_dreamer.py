"""Explicit bounded Dreamer training; importing this module never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the RSU S0 Dreamer with an explicit episode budget.")
    parser.add_argument("--mode", choices=("B", "A"), required=True)
    parser.add_argument("--relay", choices=("off", "on"), required=True)
    parser.add_argument("--train-seed", "--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--size", choices=("smoke", "default"))
    parser.add_argument("--phase", choices=("smoke", "fixed"), help="Compatibility alias: smoke/fixed -> smoke/default.")
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--prefill-episodes", type=int, default=1)
    parser.add_argument("--threads", "--cpu-threads", type=int, default=1)
    parser.add_argument("--domain", default="receiver-risk-domain-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config/train.yaml")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "data/config/risk.yaml")
    args = parser.parse_args(argv)
    alias = {"smoke": "smoke", "fixed": "default"}.get(args.phase)
    if args.size is not None and alias is not None and args.size != alias:
        parser.error("--size and --phase select different architectures.")
    try:
        from skylines_demo.risk_dreamer.training import TrainingSpec, run_training

        spec = TrainingSpec(mode=args.mode, relay=args.relay == "on", train_seed=args.train_seed,
                            episodes=args.episodes, size=args.size or alias or "default",
                            updates_per_episode=args.updates_per_episode, prefill_episodes=args.prefill_episodes,
                            domain=args.domain, threads=args.threads)
        result = run_training(spec, args.output_dir, args.config, args.risk_config)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
