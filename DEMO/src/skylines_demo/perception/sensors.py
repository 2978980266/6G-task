from __future__ import annotations

import math

from ..config import RiskCommConfig
from ..types import FrameRecord, Vector3
from ..wireless.interfaces import LineOfSightEvaluator


class SensorModel:
    """Deterministic range/FOV/geometry visibility for risk observations."""

    def __init__(
        self,
        rsu_position: Vector3,
        los_evaluator: LineOfSightEvaluator,
        config: RiskCommConfig,
    ) -> None:
        self._rsu_position = rsu_position
        self._los_evaluator = los_evaluator
        self._config = config

    def visible_object_ids(self, observer_id: int, frame: FrameRecord) -> tuple[int, ...]:
        if observer_id == 0:
            observer_position = self._rsu_position
            range_m = self._config.rsu_sensor_range_m
            fov_deg = self._config.rsu_sensor_fov_deg
            observer_heading_deg: float | None = None
        else:
            observer = frame.vehicles.get(observer_id)
            if observer is None:
                return ()
            observer_position = observer.position
            range_m = self._config.vehicle_sensor_range_m
            fov_deg = self._config.vehicle_sensor_fov_deg
            observer_heading_deg = observer.angle_y

        visible: list[int] = []
        for target_id, target in sorted(frame.vehicles.items()):
            if target_id == observer_id:
                continue
            if observer_position.horizontal_distance_to(target.position) > range_m + 1e-9:
                continue
            if not _within_field_of_view(
                observer_position,
                observer_heading_deg,
                target.position,
                fov_deg,
            ):
                continue
            if not self._los_evaluator.has_line_of_sight(
                observer_position,
                target.position,
            ):
                continue
            visible.append(target_id)
        return tuple(visible)


def _within_field_of_view(
    observer_position: Vector3,
    observer_heading_deg: float | None,
    target_position: Vector3,
    fov_deg: float,
) -> bool:
    if fov_deg >= 360.0 - 1e-9 or observer_heading_deg is None:
        return True

    delta_x = target_position.x - observer_position.x
    delta_z = target_position.z - observer_position.z
    horizontal_distance = math.hypot(delta_x, delta_z)
    if horizontal_distance <= 1e-12:
        return True

    heading_rad = math.radians(observer_heading_deg)
    heading_x = math.sin(heading_rad)
    heading_z = math.cos(heading_rad)
    cosine = (heading_x * delta_x + heading_z * delta_z) / horizontal_distance
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return angle_deg <= fov_deg / 2.0 + 1e-9
