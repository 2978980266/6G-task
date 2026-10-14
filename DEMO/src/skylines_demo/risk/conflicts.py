from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable

from ..config import RiskCommConfig
from ..types import Vector3, VehicleReplay, VehicleSnapshot


@dataclass(frozen=True)
class ConflictPoint:
    point_id: int
    position: Vector3
    raw_intersection_count: int


@dataclass(frozen=True)
class VehiclePassage:
    point_id: int
    vehicle_id: int
    crossing_time: float
    crossing_step: int
    heading_deg: float


@dataclass(frozen=True)
class RiskTask:
    task_id: str
    vehicle_id: int
    conflict_point_id: int
    conflict_point: Vector3
    crossing_time: float
    crossing_step: int
    decision_step: int
    true_risk: bool
    related_vehicle_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConflictCatalog:
    points: tuple[ConflictPoint, ...]
    passages: tuple[VehiclePassage, ...]
    tasks: tuple[RiskTask, ...]


@dataclass(frozen=True)
class _TrajectorySegment:
    vehicle_id: int
    start: VehicleSnapshot
    end: VehicleSnapshot
    start_step: int
    end_step: int
    start_time: float
    end_time: float

    @property
    def dx(self) -> float:
        return self.end.position.x - self.start.position.x

    @property
    def dz(self) -> float:
        return self.end.position.z - self.start.position.z

    @property
    def length_sq(self) -> float:
        return self.dx * self.dx + self.dz * self.dz

    @property
    def heading_deg(self) -> float:
        return _heading_from_vector(self.dx, self.dz)


@dataclass(frozen=True)
class _RawIntersection:
    point: Vector3
    first: _TrajectorySegment
    second: _TrajectorySegment
    first_fraction: float
    second_fraction: float

    @property
    def first_time(self) -> float:
        return _interpolate(
            self.first.start_time,
            self.first.end_time,
            self.first_fraction,
        )

    @property
    def second_time(self) -> float:
        return _interpolate(
            self.second.start_time,
            self.second.end_time,
            self.second_fraction,
        )


@dataclass(frozen=True)
class _Cluster:
    point: Vector3
    raw_intersections: tuple[_RawIntersection, ...]


@dataclass(frozen=True)
class _PassageCandidate:
    vehicle_id: int
    point: Vector3
    crossing_time: float
    heading_deg: float


@dataclass(frozen=True)
class _PassageInterval:
    start_time: float
    end_time: float
    crossing_time: float
    heading_deg: float
    distance_m: float
    moving: bool
    distance_sq_exact: Fraction


def build_conflict_catalog(
    replay: VehicleReplay,
    config: RiskCommConfig,
) -> ConflictCatalog:
    frame_times = tuple(frame.elapsed_seconds for frame in replay.frames)
    segments = _build_trajectory_segments(replay)
    raw_intersections = _find_raw_intersections(segments)
    clusters = _cluster_intersections(raw_intersections, config.conflict_cluster_radius_m)

    points = tuple(
        ConflictPoint(
            point_id=index,
            position=cluster.point,
            raw_intersection_count=len(cluster.raw_intersections),
        )
        for index, cluster in enumerate(clusters)
    )

    passages: list[VehiclePassage] = []
    tasks: list[RiskTask] = []
    for point, cluster in zip(points, clusters):
        candidates = _find_passage_candidates(
            segments,
            cluster.point,
            config.conflict_cluster_radius_m,
        )
        deduplicated = _deduplicate_passages(candidates)
        point_passages: list[VehiclePassage] = []
        for candidate in deduplicated:
            crossing_step = _crossing_step(frame_times, candidate.crossing_time)
            point_passages.append(
                VehiclePassage(
                    point_id=point.point_id,
                    vehicle_id=candidate.vehicle_id,
                    crossing_time=candidate.crossing_time,
                    crossing_step=crossing_step,
                    heading_deg=candidate.heading_deg,
                )
            )

        point_passages.sort(key=lambda passage: (passage.crossing_time, passage.vehicle_id))
        passages.extend(point_passages)
        for passage_index, passage in enumerate(point_passages):
            decision_step = passage.crossing_step - config.decision_lead_steps
            if not 0 <= decision_step < len(replay.frames):
                continue
            if passage.vehicle_id not in replay.frames[decision_step].vehicles:
                continue
            related_vehicle_ids = tuple(
                sorted(
                    {
                        other.vehicle_id
                        for other in point_passages
                        if other.vehicle_id != passage.vehicle_id
                        and abs(other.crossing_time - passage.crossing_time)
                        <= config.conflict_time_gap_seconds + 1e-9
                        and _heading_difference(passage.heading_deg, other.heading_deg)
                        >= config.crossing_heading_min_deg - 1e-9
                        and _heading_difference(passage.heading_deg, other.heading_deg)
                        <= config.crossing_heading_max_deg + 1e-9
                    }
                )
            )
            tasks.append(
                RiskTask(
                    task_id=f"risk-{passage.vehicle_id}-{point.point_id}-{passage_index}",
                    vehicle_id=passage.vehicle_id,
                    conflict_point_id=point.point_id,
                    conflict_point=point.position,
                    crossing_time=passage.crossing_time,
                    crossing_step=passage.crossing_step,
                    decision_step=decision_step,
                    true_risk=bool(related_vehicle_ids),
                    related_vehicle_ids=related_vehicle_ids,
                )
            )

    passages.sort(key=lambda passage: (passage.point_id, passage.crossing_time, passage.vehicle_id))
    tasks.sort(key=lambda task: (task.decision_step, task.vehicle_id, task.conflict_point_id))
    return ConflictCatalog(
        points=points,
        passages=tuple(passages),
        tasks=tuple(tasks),
    )


def _build_trajectory_segments(
    replay: VehicleReplay,
) -> dict[int, tuple[_TrajectorySegment, ...]]:
    observations: dict[int, list[tuple[int, float, VehicleSnapshot]]] = {}
    for frame in replay.frames:
        for vehicle_id, snapshot in frame.vehicles.items():
            observations.setdefault(vehicle_id, []).append(
                (frame.step_index, frame.elapsed_seconds, snapshot)
            )

    result: dict[int, tuple[_TrajectorySegment, ...]] = {}
    for vehicle_id, points in observations.items():
        segments: list[_TrajectorySegment] = []
        for first, second in zip(points, points[1:]):
            if second[1] <= first[1]:
                continue
            segments.append(
                _TrajectorySegment(
                    vehicle_id=vehicle_id,
                    start=first[2],
                    end=second[2],
                    start_step=first[0],
                    end_step=second[0],
                    start_time=first[1],
                    end_time=second[1],
                )
            )
        result[vehicle_id] = tuple(segments)
    return result


def _find_raw_intersections(
    segments_by_vehicle: dict[int, tuple[_TrajectorySegment, ...]],
) -> tuple[_RawIntersection, ...]:
    raw: list[_RawIntersection] = []
    vehicle_ids = sorted(segments_by_vehicle)
    for first_index, first_vehicle_id in enumerate(vehicle_ids):
        for second_vehicle_id in vehicle_ids[first_index + 1 :]:
            for first in segments_by_vehicle[first_vehicle_id]:
                for second in segments_by_vehicle[second_vehicle_id]:
                    intersection = _segment_intersection(first, second)
                    if intersection is None:
                        continue
                    first_fraction, second_fraction, point = intersection
                    raw.append(
                        _RawIntersection(
                            point=point,
                            first=first,
                            second=second,
                            first_fraction=first_fraction,
                            second_fraction=second_fraction,
                        )
                    )
    return tuple(raw)


def _segment_intersection(
    first: _TrajectorySegment,
    second: _TrajectorySegment,
) -> tuple[float, float, Vector3] | None:
    p_x = first.start.position.x
    p_z = first.start.position.z
    r_x = first.dx
    r_z = first.dz
    q_x = second.start.position.x
    q_z = second.start.position.z
    s_x = second.dx
    s_z = second.dz
    denominator = r_x * s_z - r_z * s_x
    if abs(denominator) <= 1e-12:
        return None

    q_minus_p_x = q_x - p_x
    q_minus_p_z = q_z - p_z
    first_fraction = (q_minus_p_x * s_z - q_minus_p_z * s_x) / denominator
    second_fraction = (q_minus_p_x * r_z - q_minus_p_z * r_x) / denominator
    if not (-1e-9 <= first_fraction <= 1.0 + 1e-9):
        return None
    if not (-1e-9 <= second_fraction <= 1.0 + 1e-9):
        return None

    fraction_a = min(max(first_fraction, 0.0), 1.0)
    fraction_b = min(max(second_fraction, 0.0), 1.0)
    first_y = _interpolate(first.start.position.y, first.end.position.y, fraction_a)
    second_y = _interpolate(second.start.position.y, second.end.position.y, fraction_b)
    return (
        fraction_a,
        fraction_b,
        Vector3(
            x=p_x + fraction_a * r_x,
            y=(first_y + second_y) / 2.0,
            z=p_z + fraction_a * r_z,
        ),
    )


def _cluster_intersections(
    intersections: Iterable[_RawIntersection],
    radius_m: float,
) -> tuple[_Cluster, ...]:
    values = tuple(intersections)
    if not values:
        return ()

    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first in enumerate(values):
        for second_index in range(first_index + 1, len(values)):
            second = values[second_index]
            if first.point.horizontal_distance_to(second.point) <= radius_m + 1e-9:
                union(first_index, second_index)

    grouped: dict[int, list[_RawIntersection]] = {}
    for index, value in enumerate(values):
        grouped.setdefault(find(index), []).append(value)

    clusters: list[_Cluster] = []
    for group in sorted(grouped.values(), key=lambda group: _point_sort_key(group[0].point)):
        point = Vector3(
            x=sum(value.point.x for value in group) / len(group),
            y=sum(value.point.y for value in group) / len(group),
            z=sum(value.point.z for value in group) / len(group),
        )
        clusters.append(_Cluster(point=point, raw_intersections=tuple(group)))
    return tuple(clusters)


def _find_passage_candidates(
    segments_by_vehicle: dict[int, tuple[_TrajectorySegment, ...]],
    point: Vector3,
    radius_m: float,
) -> tuple[_PassageCandidate, ...]:
    candidates: list[_PassageCandidate] = []
    for vehicle_id, segments in sorted(segments_by_vehicle.items()):
        intervals: list[_PassageInterval] = []
        for segment in segments:
            interval = _circle_passage_interval(segment, point, radius_m)
            if interval is not None:
                intervals.append(interval)
        intervals.sort(key=lambda interval: (interval.start_time, interval.end_time))
        for interval_group in _merge_passage_intervals(intervals):
            best_interval = min(
                interval_group,
                key=lambda interval: (interval.distance_sq_exact, interval.crossing_time),
            )
            # A stationary minimum retains its time, but uses the nearest
            # moving portion of this visit for the existing heading rule.
            heading_interval = best_interval
            if not best_interval.moving:
                heading_interval = min(
                    (interval for interval in interval_group if interval.moving),
                    key=lambda interval: (abs(interval.crossing_time - best_interval.crossing_time),
                                          interval.crossing_time),
                    default=best_interval,
                )
            candidates.append(
                _PassageCandidate(
                    vehicle_id=vehicle_id,
                    point=point,
                    crossing_time=best_interval.crossing_time,
                    heading_deg=heading_interval.heading_deg,
                )
            )
    return tuple(candidates)


def _circle_passage_interval(
    segment: _TrajectorySegment, point: Vector3, radius_m: float,
) -> _PassageInterval | None:
    """Clip a segment to the actual horizontal circle, including dwell time."""
    offset_x = segment.start.position.x - point.x
    offset_z = segment.start.position.z - point.z
    fraction = _closest_segment_fraction(segment, point)
    distance_m = math.hypot(offset_x + fraction * segment.dx,
                            offset_z + fraction * segment.dz)
    moving = segment.length_sq > 1e-12
    if not moving:
        if distance_m > radius_m:
            return None
        entry, exit_ = 0.0, 1.0
    else:
        projection = offset_x * segment.dx + offset_z * segment.dz
        c = offset_x * offset_x + offset_z * offset_z - radius_m * radius_m
        discriminant = projection * projection - segment.length_sq * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        entry = max(0.0, (-projection - root) / segment.length_sq)
        exit_ = min(1.0, (-projection + root) / segment.length_sq)
        if entry > exit_:
            return None
    return _PassageInterval(
        start_time=_interpolate(segment.start_time, segment.end_time, entry),
        end_time=_interpolate(segment.start_time, segment.end_time, exit_),
        crossing_time=_interpolate(segment.start_time, segment.end_time, fraction),
        heading_deg=segment.heading_deg if moving else segment.start.angle_y,
        distance_m=distance_m, moving=moving,
        distance_sq_exact=_segment_distance_sq_exact(segment, point, moving),
    )


def _segment_distance_sq_exact(
    segment: _TrajectorySegment, point: Vector3, moving: bool,
) -> Fraction:
    """Compare actual binary input geometry without a distance tie tolerance.

    Reversing identical endpoints gives exactly the same squared distance;
    arbitrarily close but distinct distances remain distinct. Float clipping,
    crossing times and the existing stationary threshold are unchanged.
    """
    offset_x = Fraction(segment.start.position.x) - Fraction(point.x)
    offset_z = Fraction(segment.start.position.z) - Fraction(point.z)
    start_distance_sq = offset_x * offset_x + offset_z * offset_z
    if not moving:
        return start_distance_sq
    dx = Fraction(segment.end.position.x) - Fraction(segment.start.position.x)
    dz = Fraction(segment.end.position.z) - Fraction(segment.start.position.z)
    length_sq = dx * dx + dz * dz
    projection = -(offset_x * dx + offset_z * dz)
    if projection <= 0:
        return start_distance_sq
    if projection >= length_sq:
        return (offset_x + dx) ** 2 + (offset_z + dz) ** 2
    return start_distance_sq - projection * projection / length_sq


def _merge_passage_intervals(
    intervals: Iterable[_PassageInterval],
) -> tuple[tuple[_PassageInterval, ...], ...]:
    groups: list[list[_PassageInterval]] = []
    for interval in intervals:
        if not groups or interval.start_time > max(i.end_time for i in groups[-1]) + 1e-9:
            groups.append([interval])
        else:
            groups[-1].append(interval)
    return tuple(tuple(group) for group in groups)


def _deduplicate_passages(
    candidates: Iterable[_PassageCandidate],
) -> tuple[_PassageCandidate, ...]:
    result: list[_PassageCandidate] = []
    for candidate in sorted(candidates, key=lambda value: (value.vehicle_id, value.crossing_time)):
        duplicate = any(
            existing.vehicle_id == candidate.vehicle_id
            and abs(existing.crossing_time - candidate.crossing_time) <= 1e-7
            for existing in result
        )
        if not duplicate:
            result.append(candidate)
    return tuple(result)


def _closest_segment_fraction(segment: _TrajectorySegment, point: Vector3) -> float:
    if segment.length_sq <= 1e-12:
        return 0.0
    delta_x = point.x - segment.start.position.x
    delta_z = point.z - segment.start.position.z
    fraction = (delta_x * segment.dx + delta_z * segment.dz) / segment.length_sq
    return min(max(fraction, 0.0), 1.0)


def _crossing_step(frame_times: tuple[float, ...], crossing_time: float) -> int:
    step = bisect_left(frame_times, crossing_time - 1e-9)
    return min(max(step, 0), len(frame_times) - 1)


def _heading_from_vector(delta_x: float, delta_z: float) -> float:
    return math.degrees(math.atan2(delta_x, delta_z)) % 360.0


def _heading_difference(first: float, second: float) -> float:
    difference = abs((first - second) % 360.0)
    return min(difference, 360.0 - difference)


def _interpolate(first: float, second: float, fraction: float) -> float:
    return first + (second - first) * fraction


def _point_sort_key(point: Vector3) -> tuple[float, float, float]:
    return (round(point.x, 9), round(point.z, 9), round(point.y, 9))
