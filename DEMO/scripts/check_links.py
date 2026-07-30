from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "train.yaml"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skylines_demo.bootstrap import build_environment
from skylines_demo.config import load_config
from skylines_demo.policies import MaxTaskRewardGainPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect V2I/V2V candidate links in the DEMO scene.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the scenario config YAML.",
    )
    parser.add_argument("--steps", type=int, default=5, help="How many non-empty steps to print.")
    parser.add_argument("--candidates", type=int, default=4, help="How many candidate samples to print per step.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the scenario config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed channel, role, and resource diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    env = build_environment(args.config, seed=seed)
    print(f"seed={seed}")
    policy = MaxTaskRewardGainPolicy(
        config.resource,
        max_links=config.environment.max_active_links,
        allow_multi_v2i_from_rsu=config.environment.allow_multi_v2i_from_rsu,
    )
    observation = env.reset()
    shown = 0

    while observation is not None and shown < args.steps:
        schedule_sample = []
        if observation.candidate_links:
            scheduling_action = policy.select_action(observation)
            action_set = set(scheduling_action.links)
            schedule_sample = [
                candidate
                for candidate in observation.candidate_links
                if candidate.action in action_set
            ]
            print(
                f"step={observation.step_index} frame={observation.simulation_frame} "
                f"time={observation.elapsed_seconds:.3f}s "
                f"dt={observation.delta_seconds:.3f}s "
                f"vehicles={len(observation.active_vehicles)} "
                f"candidates={len(observation.candidate_links)} "
                f"(v2i={observation.v2i_candidate_count}, "
                f"v2v={observation.v2v_candidate_count}) "
                f"selected={len(schedule_sample)}"
            )
            if schedule_sample:
                sample_text = ", ".join(
                    f"{candidate.action.link_type.upper()}:{candidate.action.transmitter_id}->{candidate.action.receiver_id}"
                    for candidate in schedule_sample
                )
                print(f"  schedule={sample_text}")
            for candidate in observation.candidate_links[: args.candidates]:
                action = candidate.action
                budget = candidate.link_budget
                if args.verbose:
                    print(
                        "  "
                        f"{action.link_type.upper()} tx={action.transmitter_id} "
                        f"rx={action.receiver_id} "
                        f"dist={budget.distance_m:.2f}m los={budget.los} "
                        f"raw_rate={budget.raw_rate_mbps:.2f}Mbps "
                        f"effective_rate={budget.rate_mbps:.2f}Mbps "
                        f"blocks={candidate.resource_block_count} "
                        f"role={'AV' if candidate.receiver_is_autonomous else 'HUM'} "
                        f"task_progress={candidate.task_progress_gain} "
                        f"task_reward_gain={candidate.task_reward_gain:.3f} "
                        f"packets={candidate.selected_packet_ids} "
                        f"delivered_packets={int(candidate.delivered_packet_count)}"
                    )
                else:
                    print(
                        "  "
                        f"{action.link_type.upper()} "
                        f"{action.transmitter_id}->{action.receiver_id} "
                        f"rate={budget.rate_mbps:.2f}Mbps "
                        f"blocks={candidate.resource_block_count} "
                        f"packets={candidate.selected_packet_ids} "
                        f"progress={candidate.task_progress_gain} "
                        f"reward_gain={candidate.task_reward_gain:.3f}"
                    )
            shown += 1

        result = env.step(policy.select_action(observation))
        observation = result.next_observation


if __name__ == "__main__":
    main()
