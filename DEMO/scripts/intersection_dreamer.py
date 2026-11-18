"""Analyze, train, or evaluate identity-aware continuous junction recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'train', 'evaluate'))
    parser.add_argument('--config', type=Path, default=Path('DEMO/data/config/intersection.yaml'))
    parser.add_argument('--risk-config', type=Path, default=Path('DEMO/data/config/risk_pair_events.yaml'))
    parser.add_argument('--relay', choices=('on', 'off'), default='off')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--capacity', type=int, default=80)
    parser.add_argument('--size', choices=('smoke', 'default'), default='default')
    parser.add_argument('--gap-steps', type=int, default=4)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args()
    from skylines_demo.risk_dreamer.dynamic_training import (
        prepare_data, data_report, run_training, load_model, evaluate, dump, initialize_seed)
    try:
        if args.command == 'train':
            if args.output_dir is None:
                parser.error('train requires --output-dir')
            result = run_training(args.config, args.risk_config, args.output_dir,
                                  episodes=args.episodes, seed=args.seed, relay=args.relay == 'on',
                                  capacity=args.capacity, size=args.size, threads=args.threads,
                                  gap_steps=args.gap_steps)
        elif args.command == 'evaluate':
            if args.model_path is None:
                parser.error('evaluate requires --model-path')
            initialize_seed(args.seed, args.threads)
            agent, runtime, risk, entries, capacity = load_model(args.model_path)
            result = evaluate(agent, runtime, risk, entries, capacity)
            repeated = evaluate(agent, runtime, risk, entries, capacity)
            if result != repeated:
                raise AssertionError('Repeated greedy evaluation differs.')
        else:
            runtime, risk, entries = prepare_data(args.config, args.risk_config,
                                                args.relay == 'on', args.gap_steps)
            result = data_report(runtime, risk, entries)
        if args.command != 'train' and args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            dump(args.output_dir / 'result.json', result)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
