from __future__ import annotations

from dataclasses import dataclass
import math

from ..types import BuildingSnapshot, Vector3


@dataclass(frozen=True)
class OrientedFootprint:
    center_x: float
    center_z: float
    half_x: float
    half_z: float
    yaw_rad: float
    min_y: float
    max_y: float


def normalize_yaw(angle_value: float) -> float:
    if abs(angle_value) > (2.0 * math.pi + 1e-6):
        return math.radians(angle_value)
    return angle_value


def building_to_footprint(
    building: BuildingSnapshot,
    fallback_cell_size_m: float = 8.0,
) -> OrientedFootprint:
    size_x = building.size_x if building.size_x > 0 else max(building.width_cells, 1) * fallback_cell_size_m
    size_z = building.size_z if building.size_z > 0 else max(building.length_cells, 1) * fallback_cell_size_m
    return OrientedFootprint(
        center_x=building.position.x + building.center_offset_x,
        center_z=building.position.z + building.center_offset_z,
        half_x=max(size_x / 2.0, 0.5),
        half_z=max(size_z / 2.0, 0.5),
        yaw_rad=normalize_yaw(building.angle_y),
        min_y=building.min_y,
        max_y=building.max_y,
    )


def _to_local_frame(x: float, z: float, footprint: OrientedFootprint) -> tuple[float, float]:
    dx = x - footprint.center_x
    dz = z - footprint.center_z
    cos_yaw = math.cos(footprint.yaw_rad)
    sin_yaw = math.sin(footprint.yaw_rad)
    local_x = dx * cos_yaw + dz * sin_yaw
    local_z = -dx * sin_yaw + dz * cos_yaw
    return local_x, local_z


def _clip_axis(p: float, q: float, u1: float, u2: float) -> tuple[bool, float, float]:
    if abs(p) < 1e-12:
        return (q >= 0.0), u1, u2

    ratio = q / p
    if p < 0.0:
        if ratio > u2:
            return False, u1, u2
        return True, max(u1, ratio), u2

    if ratio < u1:
        return False, u1, u2
    return True, u1, min(u2, ratio)


def _segment_intersection_interval_aabb(
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> tuple[float, float] | None:
    dx = x1 - x0
    dz = z1 - z0
    u1 = 0.0
    u2 = 1.0

    checks = (
        (-dx, x0 - min_x),
        (dx, max_x - x0),
        (-dz, z0 - min_z),
        (dz, max_z - z0),
    )

    for p, q in checks:
        visible, u1, u2 = _clip_axis(p, q, u1, u2)
        if not visible:
            return None

    if u1 > u2:
        return None
    return u1, u2


def segment_intersection_interval(
    start: Vector3,
    end: Vector3,
    footprint: OrientedFootprint,
) -> tuple[float, float] | None:
    start_x, start_z = _to_local_frame(start.x, start.z, footprint)
    end_x, end_z = _to_local_frame(end.x, end.z, footprint)
    return _segment_intersection_interval_aabb(
        x0=start_x,
        z0=start_z,
        x1=end_x,
        z1=end_z,
        min_x=-footprint.half_x,
        max_x=footprint.half_x,
        min_z=-footprint.half_z,
        max_z=footprint.half_z,
    )


def segment_intersects_footprint(
    start: Vector3,
    end: Vector3,
    footprint: OrientedFootprint,
) -> bool:
    return segment_intersection_interval(start, end, footprint) is not None
