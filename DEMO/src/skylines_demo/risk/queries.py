"""Periodic query schedule; truth and validity never enter public task views."""
from __future__ import annotations

from .conflicts import ConflictCatalog, RiskTask
from ..types import RiskTaskView, Vector3


def query_steps(length, config):
    return range(config.query_warmup_steps, length, config.query_interval_steps)


def public_queries(active_ids, step, length, config):
    # Publish the same calendar for every active vehicle, regardless of labels,
    # departure times, future paths, or whether an offline query is evaluable.
    return tuple(RiskTaskView(f'query-{vehicle_id}-{deadline}', vehicle_id, -1,
                             Vector3(0., 0., 0.), None, deadline)
                 for deadline in query_steps(length, config) if deadline >= step
                 for vehicle_id in active_ids)


def build_query_catalog(replay, context, config, events=None):
    from .pair_events import build_pair_events, label_query
    if events is None:
        events = build_pair_events(context, config)
    context_indices = {frame.simulation_frame: i for i, frame in enumerate(context.frames)}
    tasks = []
    for step in query_steps(len(replay.frames), config):
        frame = replay.frames[step]
        index = context_indices[frame.simulation_frame]
        for vehicle_id in sorted(frame.vehicles):
            label = label_query(context, events, vehicle_id, index, config)
            tasks.append(RiskTask(f'query-{vehicle_id}-{step}', vehicle_id, -1,
                                  Vector3(0., 0., 0.), frame.elapsed_seconds, -1,
                                  step, label.true_risk, label.related_vehicle_ids,
                                  label.evaluable, label.reason))
    return ConflictCatalog((), (), tuple(tasks))
