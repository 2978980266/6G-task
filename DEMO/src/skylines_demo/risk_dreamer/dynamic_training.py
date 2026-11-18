"""Dynamic-traffic Dreamer rollouts; identities and labels stay outside tensors."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..bootstrap import load_runtime
from ..config import load_risk_config
from ..data.episodes import split_windows
from ..env.risk_comm_env import RiskCommEnv
from ..risk.pair_events import build_pair_events
from ..risk.queries import build_query_catalog
from ..risk_learning.domain import build_safe_step
from ..types import RiskSchedulingAction
from .config import DreamerConfig
from .dynamic_adapter import DynamicCatalogue
from .learning import DreamerAgent
from .training import Episode, Replay, initialize_seed, json_safe

CHECKPOINT_VERSION = 'dynamic-pair-queries-v1'
ENCODING_VERSION = 'causal-visit-slots-pair67-node16-v1'


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, allow_nan=False, indent=2)


def prepare_data(config_path, risk_path, relay=False, gap_steps=4):
    runtime = load_runtime(config_path)
    if len({value['session_id'] for value in runtime.replay.identities.values()}) > 1:
        raise ValueError('Analyze/train each recording session separately; do not cut windows across sessions.')
    risk = replace(load_risk_config(risk_path), enable_v2v_relay=relay)
    if risk.risk_mode != 'pair-events' or risk.link_prediction:
        raise ValueError('Dynamic training requires pair-events and link_prediction=false.')
    windows = split_windows(runtime.replay, gap_steps=gap_steps)
    event_cache = {}
    entries = []
    for window in windows:
        if window.split not in event_cache:
            event_cache[window.split] = build_pair_events(window.label_context, risk)
        catalog = build_query_catalog(window.replay, window.label_context, risk, event_cache[window.split])
        entries.append((window, catalog, event_cache[window.split]))
    return runtime, risk, tuple(entries)


def environment(runtime, risk, entry):
    window, _, events = entry
    return RiskCommEnv(replace(runtime, replay=window.replay), risk,
                       label_replay=window.label_context, pair_events=events)


def data_report(runtime, risk, entries):
    rows = []
    for window, catalog, events in entries:
        scored = [task for task in catalog.tasks if task.evaluable]
        rows.append(dict(split=window.split, start=window.start, stop=window.stop,
                         steps=len(window.replay), vehicles=len(window.replay.vehicle_ids),
                         queries=len(catalog.tasks), scored=len(scored),
                         unknown=len(catalog.tasks) - len(scored),
                         positive=sum(task.true_risk for task in scored),
                         negative=sum(not task.true_risk for task in scored),
                         partition_pair_events=len(events)))
    all_ids = {split: {i for w, _, _ in entries if w.split == split
                       for i in w.replay.vehicle_ids} for split in ('train', 'validation', 'test')}
    return dict(frames=len(runtime.replay), records=sum(len(f.vehicles) for f in runtime.replay.frames),
                visits=len(runtime.replay.vehicle_ids),
                full_recording_pair_events=len(build_pair_events(runtime.replay, risk)),
                seconds=runtime.replay.frames[-1].elapsed_seconds - runtime.replay.frames[0].elapsed_seconds,
                risk_config=asdict(risk), windows=rows,
                train_validation_shared_visits=len(all_ids['train'] & all_ids['validation']),
                train_test_shared_visits=len(all_ids['train'] & all_ids['test']),
                scope='One recording and one junction; partitions share traffic, not independent scenes.',
                negative_scope='Recorded continuous visits only; unrecorded road users are outside ground truth.')


@torch.no_grad()
def collect(env, risk, agent, capacity, *, greedy=False):
    slots = DynamicCatalogue(capacity, risk.enable_v2v_relay)
    obs = env.reset()
    prior = None
    observations, actions, rewards, continues = [], [], [], []
    while obs is not None:
        encoded = {k: v.copy() for k, v in slots.encode(obs, risk).items()}
        tensors = {k: torch.from_numpy(v).unsqueeze(0) for k, v in encoded.items()}
        state, output = agent.observe(tensors, prior, greedy)
        slots.validate_selection(encoded, output.selections[0])
        result = env.step(slots.to_action(output.selections[0]))
        if result.invalid_action:
            raise AssertionError('Dynamic actor selected an invalid real action.')
        observations.append(encoded)
        actions.append(output.actions[0].cpu().numpy().copy())
        rewards.append(float(result.reward))
        continues.append(float(result.next_observation is not None))
        prior = agent.world.transition(state, output.actions, sample=not greedy)
        obs = result.next_observation
    observations.append({k: np.zeros_like(v) for k, v in observations[-1].items()})
    episode = Episode(observations, np.asarray(actions, np.float32),
                      np.asarray(rewards, np.float32), np.asarray(continues, np.float32), None)
    episode.validate()
    summary = env.objective_summary
    if (not summary['evaluation_complete'] or summary['pending_task_count']
            or not math.isclose(sum(rewards), -summary['objective_sum'], abs_tol=1e-5)):
        raise AssertionError('Rollout reward/score mismatch.')
    return episode, summary


def baseline_action(obs, risk, name):
    if name == 'no-communication':
        return RiskSchedulingAction()
    # Give rules the exact same version/endpoint/slot constraints as the actor.
    # Pair mode retains uncertain and predicted-safe downlinks as well.
    safe = build_safe_step(obs, risk)
    candidates = sorted(safe.candidates, key=lambda c: (
        0 if c.key[0] == 'D' else 1 if c.key[0] == 'R' else 2,
        c.features[8], -c.features[10], c.key))
    if name == 'pair-prediction':
        candidates.sort(key=lambda c: (-c.features[17], c.features[8], -c.features[10], c.key))
    elif name != 'freshness':
        raise ValueError('Unknown baseline.')
    occupied, selected = set(), []
    counts = dict(U=0, D=0, R=0)
    for c in candidates:
        kind, sender, receiver, _ = c.key
        endpoints = {i for i in (sender, receiver) if i}
        if endpoints & occupied or counts[kind] >= dict(U=2, D=2, R=int(risk.enable_v2v_relay))[kind]:
            continue
        occupied.update(endpoints)
        counts[kind] += 1
        selected.append(c.action)
    return RiskSchedulingAction(tuple(a for a in selected if a.__class__.__name__ == 'UplinkAction'),
                                tuple(a for a in selected if a.__class__.__name__ == 'DownlinkAction'),
                                tuple(a for a in selected if a.__class__.__name__ == 'RelayAction'))


def summarize(rows):
    keys = ('catalog_task_count', 'catalog_positive_count', 'catalog_negative_count',
            'positive_success_count', 'positive_failure_count', 'negative_false_positive_count',
            'negative_abstention_count', 'send_count', 'objective_sum', 'unscored_query_count')
    totals = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    totals['J_per_scored_query'] = (totals['objective_sum'] / totals['catalog_task_count']
                                    if totals['catalog_task_count'] else None)
    return totals


def evaluate(agent, runtime, risk, entries, capacity):
    results = {}
    for split in ('train', 'validation', 'test'):
        split_entries = [entry for entry in entries if entry[0].split == split]
        rows = []
        for entry in split_entries:
            _, score = collect(environment(runtime, risk, entry), risk, agent, capacity, greedy=True)
            rows.append(score)
        results[split] = summarize(rows)
    return results


def evaluate_baselines(runtime, risk, entries):
    results = {}
    for name in ('no-communication', 'freshness', 'pair-prediction'):
        by_split = {}
        for split in ('train', 'validation', 'test'):
            scores = []
            for entry in entries:
                if entry[0].split != split:
                    continue
                env = environment(runtime, risk, entry)
                obs = env.reset()
                while obs is not None:
                    result = env.step(baseline_action(obs, risk, name))
                    if result.invalid_action:
                        raise AssertionError('Baseline selected an invalid action.')
                    obs = result.next_observation
                scores.append(env.objective_summary)
            by_split[split] = summarize(scores)
        results[name] = by_split
    return results


def run_training(config_path, risk_path, output, *, episodes=4, seed=7, relay=False,
                 capacity=80, size='default', threads=1, gap_steps=4):
    if type(episodes) is not int or episodes < 2 or capacity < 1 or size not in ('smoke', 'default'):
        raise ValueError('Require at least two episodes, positive capacity, and smoke/default size.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    rng = initialize_seed(seed, threads)
    runtime, risk, entries = prepare_data(config_path, risk_path, relay, gap_steps)
    for split in ('train', 'validation', 'test'):
        if not any(t.evaluable for w, cat, _ in entries if w.split == split for t in cat.tasks):
            raise ValueError(f'{split} has no evaluable queries; record more context or revise the split.')
    if max(len(e[0].replay.vehicle_ids) for e in entries) > capacity:
        raise ValueError('Slot capacity is insufficient; increase --capacity.')
    config = DreamerConfig.smoke() if size == 'smoke' else DreamerConfig()
    config = replace(config, batch_size=2, replay_capacity=16, imagination_batch=4)
    agent = DreamerAgent(config, DynamicCatalogue(capacity, relay).catalogue)
    output.mkdir(parents=True)
    dump(output / 'data_report.json', data_report(runtime, risk, entries))
    before = evaluate(agent, runtime, risk, entries, capacity)
    dump(output / 'before.json', before)
    dump(output / 'baselines.json', evaluate_baselines(runtime, risk, entries))
    replay = Replay(config.replay_capacity)
    train_entries = [entry for entry in entries if entry[0].split == 'train'
                     and any(t.evaluable for t in entry[1].tasks)]
    updates = 0
    with (output / 'metrics.jsonl').open('x', encoding='utf-8') as log:
        for i in range(episodes):
            entry = train_entries[i % len(train_entries)]
            episode, score = collect(environment(runtime, risk, entry), risk, agent, capacity)
            replay.add(episode)
            metrics = {}
            if i:
                metrics = agent.update(replay.sample(config.batch_size, rng))
                updates += 1
            row = dict(episode=i + 1, window_start=entry[0].start, score=score, **metrics)
            log.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(dict(episode=i + 1, updates=updates,
                                 J=score['objective_sum'], model_loss=metrics.get('model_loss'))), flush=True)
    after = evaluate(agent, runtime, risk, entries, capacity)
    metadata = dict(version=CHECKPOINT_VERSION, encoding=ENCODING_VERSION,
                    model_config=asdict(config), capacity=capacity, relay=relay,
                    risk_config=asdict(risk), config_path=str(config_path), risk_path=str(risk_path),
                    gap_steps=gap_steps, seed=seed, episodes=episodes, updates=updates,
                    scenario=json_safe(asdict(runtime.config)))
    torch.save(dict(metadata=metadata, model=agent.state_dict()), output / 'model.pt')
    dump(output / 'metadata.json', metadata)
    loaded, loaded_runtime, loaded_risk, loaded_entries, loaded_capacity = load_model(output / 'model.pt')
    repeated = evaluate(loaded, loaded_runtime, loaded_risk, loaded_entries, loaded_capacity)
    if json_safe(after) != json_safe(repeated):
        raise AssertionError('Greedy evaluation changed after checkpoint reload.')
    summary = dict(before=before, after=after, updates=updates, steps=episodes * 23,
                   checkpoint_reload_equal=True, invalid_actions=0,
                   model_path=str(output / 'model.pt'))
    dump(output / 'summary.json', summary)
    return summary


def load_model(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    if metadata['version'] != CHECKPOINT_VERSION or metadata['encoding'] != ENCODING_VERSION:
        raise ValueError('Incompatible dynamic checkpoint.')
    capacity = metadata['capacity']
    agent = DreamerAgent(DreamerConfig(**metadata['model_config']),
                         DynamicCatalogue(capacity, metadata['relay']).catalogue)
    agent.load_state_dict(payload['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in agent.parameters()):
        raise ValueError('Nonfinite checkpoint tensors.')
    runtime, risk, entries = prepare_data(metadata['config_path'], metadata['risk_path'],
                                        metadata['relay'], metadata['gap_steps'])
    if asdict(risk) != metadata['risk_config'] or json_safe(asdict(runtime.config)) != metadata['scenario']:
        raise ValueError('Evaluation configuration differs from checkpoint metadata.')
    agent.eval()
    return agent, runtime, risk, entries, capacity
