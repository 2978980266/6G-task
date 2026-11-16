"""Dreamer checkpoint reload and two complete deterministic greedy evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Dreamer checkpoint and repeat complete greedy evaluation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--decoding", choices=("greedy",), default="greedy")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--mode", choices=("B", "A"))
    parser.add_argument("--relay", choices=("off", "on"))
    parser.add_argument("--threads", "--cpu-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config/train.yaml")
    parser.add_argument("--risk-config", type=Path, default=ROOT / "data/config/risk.yaml")
    args = parser.parse_args(argv)
    try:
        from skylines_demo.risk_dreamer.training import run_evaluation

        result = run_evaluation(args.model_path, args.output_dir, args.config, args.risk_config,
                                metadata_path=args.metadata_path, repeat=args.repeat, mode=args.mode,
                                relay=None if args.relay is None else args.relay == "on", threads=args.threads)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
