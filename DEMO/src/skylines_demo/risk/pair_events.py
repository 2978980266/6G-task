"""Offline geometric crossing labels for identity-mapped, recorded ROI visits.

There is no spatial clustering or extrapolation. Missing observations never form
segments. Evaluability describes recording coverage, independently of the label:
even an observed positive is unevaluable when the required context is incomplete.
Explicit visit metadata defines first/last presence as entry/exit from the recorded
ROI. Missing frames inside a visit are unknown; outside that visit the vehicle is
outside the labeling domain. Legacy IDs without visit metadata require presence
throughout the context. Labels make no claim about entirely unrecorded traffic or
motion between an ROI boundary and the first/last sampled position.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from statistics import median
from typing import Iterable

from ..config import RiskCommConfig
from ..types import Vector3, VehicleReplay
from .conflicts import _build_trajectory_segments, _segment_intersection, _TrajectorySegment

_EPS = 1e-9
_DEDUP_TOLERANCE = 1e-7


@dataclass(frozen=True)
class PairEvent:
    first_id: int
    second_id: int
    point: Vector3
    first_time: float
    second_time: float
    heading_difference_deg: float
    time_gap_seconds: float


@dataclass(frozen=True)
class QueryLabel:
    """Observed label; ``true_risk=False`` is certified only if evaluable."""

    true_risk: bool
    related_vehicle_ids: tuple[int, ...]
    evaluable: bool
    reason: str


@dataclass(frozen=True)
class _BoundedSegment:
    segment: _TrajectorySegment
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    heading: float


def build_pair_events(
    replay: VehicleReplay, config: RiskCommConfig,
) -> tuple[PairEvent, ...]:
    """Return distinct-pair intersections satisfying angle, height and time bounds.

    Times are interpolated along consecutive recorded steps. Events use ascending
    visit IDs and deterministic chronological order. A temporal sweep and cached
    bounding boxes reject unrelated segments before exact intersection arithmetic.
    Only duplicate representations at shared vertices are merged; revisits remain.
    """
    candidates: list[_BoundedSegment] = []
    maximum_interval = _maximum_frame_interval(replay)
    for segments in _build_trajectory_segments(replay).values():
        for segment in segments:
            if (segment.end_step != segment.start_step + 1
                    or segment.end_time - segment.start_time > maximum_interval + _EPS
                    or segment.length_sq <= 1e-12):
                continue
            start, end = segment.start.position, segment.end.position
            candidates.append(_BoundedSegment(
                segment, min(start.x, end.x), max(start.x, end.x),
                min(start.z, end.z), max(start.z, end.z), segment.heading_deg,
            ))
    candidates.sort(key=lambda value: (
        value.segment.start_time, value.segment.vehicle_id, value.segment.start_step,
    ))
    gap = config.conflict_time_gap_seconds
    active: list[_BoundedSegment] = []
    events_by_pair: dict[tuple[int, int], list[PairEvent]] = {}
    for current in candidates:
        active = [other for other in active
                  if other.segment.end_time + gap + _EPS >= current.segment.start_time]
        for other in active:
            first, second = other.segment, current.segment
            if first.vehicle_id == second.vehicle_id:
                continue
            if (other.max_x < current.min_x - _EPS or current.max_x < other.min_x - _EPS
                    or other.max_z < current.min_z - _EPS or current.max_z < other.min_z - _EPS):
                continue
            difference = abs(other.heading - current.heading) % 360.0
            difference = min(difference, 360.0 - difference)
            if not (config.crossing_heading_min_deg - _EPS <= difference
                    <= config.crossing_heading_max_deg + _EPS):
                continue
            intersection = _segment_intersection(first, second)
            if intersection is None:
                continue
            first_fraction, second_fraction, point = intersection
            first_y = first.start.position.y + first_fraction * (
                first.end.position.y - first.start.position.y)
            second_y = second.start.position.y + second_fraction * (
                second.end.position.y - second.start.position.y)
            if abs(first_y - second_y) > config.event_max_height_difference_m + _EPS:
                continue
            first_time = first.start_time + first_fraction * (first.end_time - first.start_time)
            second_time = second.start_time + second_fraction * (second.end_time - second.start_time)
            time_gap = abs(first_time - second_time)
            if time_gap > gap + _EPS:
                continue
            if first.vehicle_id > second.vehicle_id:
                first, second = second, first
                first_time, second_time = second_time, first_time
            event = PairEvent(first.vehicle_id, second.vehicle_id, point,
                              first_time, second_time, difference, time_gap)
            pair_events = events_by_pair.setdefault((event.first_id, event.second_id), [])
            if not any(
                abs(prior.first_time - event.first_time) <= _DEDUP_TOLERANCE
                and abs(prior.second_time - event.second_time) <= _DEDUP_TOLERANCE
                and prior.point.distance_to(event.point) <= _DEDUP_TOLERANCE
                for prior in pair_events
            ):
                pair_events.append(event)
        active.append(current)
    return tuple(sorted(
        (event for pair in events_by_pair.values() for event in pair),
        key=lambda event: (min(event.first_time, event.second_time), event.first_id,
                           event.second_id, event.first_time, event.second_time),
    ))


def label_query(
    replay: VehicleReplay,
    events: Iterable[PairEvent],
    vehicle_id: int,
    decision_step: int,
    config: RiskCommConfig,
) -> QueryLabel:
    """Label ego crossings in inclusive ``[now, now + horizon]``.

    ``decision_step`` indexes ``replay.frames``. Other passage times may lie up
    to the configured gap before or after ego's crossing, including before now.
    Eligibility requires global coverage of ``[now-gap, now+horizon+gap]``,
    continuous ego presence through the horizon, and complete other-visit context.
    Boundary interpolation needs the bracketing recorded frames. An unevaluable
    negative means unknown, never evidence of safety. Eligibility and its reason
    never inspect events or the observed truth value.
    """
    frames = replay.frames
    if not 0 <= decision_step < len(frames):
        return QueryLabel(False, (), False, "invalid_decision_step")
    now = frames[decision_step].elapsed_seconds
    horizon_end = now + config.event_horizon_seconds
    related: set[int] = set()
    for event in events:
        if event.first_id == vehicle_id:
            ego_time, other_id = event.first_time, event.second_id
        elif event.second_id == vehicle_id:
            ego_time, other_id = event.second_time, event.first_id
        else:
            continue
        if now - _EPS <= ego_time <= horizon_end + _EPS:
            related.add(other_id)

    reason = _coverage_reason(replay, vehicle_id, decision_step, horizon_end, config)
    return QueryLabel(bool(related), tuple(sorted(related)), reason == "complete", reason)


def _coverage_reason(
    replay: VehicleReplay, vehicle_id: int, decision_step: int,
    horizon_end: float, config: RiskCommConfig,
) -> str:
    frames = replay.frames
    if vehicle_id not in frames[decision_step].vehicles:
        return "ego_absent"
    now = frames[decision_step].elapsed_seconds
    context_start = now - config.conflict_time_gap_seconds
    context_end = horizon_end + config.conflict_time_gap_seconds
    times = tuple(frame.elapsed_seconds for frame in frames)
    if times[0] > context_start + _EPS or times[-1] < context_end - _EPS:
        return "recording_context_incomplete"
    start = max(0, bisect_right(times, context_start + _EPS) - 1)
    end = min(len(frames) - 1, bisect_left(times, context_end - _EPS))
    context = frames[start:end + 1]
    maximum_interval = _maximum_frame_interval(replay)
    if any(second.step_index != first.step_index + 1
           or second.elapsed_seconds <= first.elapsed_seconds
           or second.elapsed_seconds - first.elapsed_seconds > maximum_interval + _EPS
           for first, second in zip(context, context[1:])):
        return "recording_context_incomplete"
    ego_end = min(len(frames) - 1, bisect_left(times, horizon_end - _EPS))
    if any(vehicle_id not in frame.vehicles for frame in frames[decision_step:ego_end + 1]):
        return "ego_future_incomplete"
    # Bounds use the whole partition: a gap at the context edge may still be
    # interior to a recorded visit, including a visit absent from this context.
    bounds: dict[int, tuple[int, int]] = {}
    for index, frame in enumerate(frames):
        for other_id in frame.vehicles:
            if other_id != vehicle_id:
                bounds[other_id] = (bounds.get(other_id, (index, index))[0], index)
    for other_id, (first, last) in bounds.items():
        if last < start or first > end:
            continue
        if other_id in replay.identities:
            required_start, required_end = max(start, first), min(end, last)
        else:
            required_start, required_end = start, end
        if any(other_id not in frame.vehicles
               for frame in frames[required_start:required_end + 1]):
            return "other_context_incomplete"
    return "complete"


def _maximum_frame_interval(replay: VehicleReplay) -> float:
    """Infer cadence from this partition; a missing sample must not be bridged."""
    intervals = [second.elapsed_seconds - first.elapsed_seconds
                 for first, second in zip(replay.frames, replay.frames[1:])
                 if second.elapsed_seconds > first.elapsed_seconds]
    return 1.5 * median(intervals) if intervals else 0.0
