from __future__ import annotations

from .footprints import building_to_footprint, segment_intersection_interval
from ..types import BuildingSnapshot, Vector3


class BuildingLineOfSight:
    def __init__(
        self,
        buildings: list[BuildingSnapshot],
        fallback_cell_size_m: float = 8.0,
        height_clearance_m: float = 1.0,
    ) -> None:
        self._footprints = [
            building_to_footprint(building, fallback_cell_size_m=fallback_cell_size_m)
            for building in buildings
        ]
        self._height_clearance_m = height_clearance_m

    def has_line_of_sight(self, start: Vector3, end: Vector3) -> bool:
        for footprint in self._footprints:
            interval = segment_intersection_interval(start, end, footprint)
            if interval is None:
                continue
            entry_t, exit_t = interval
            entry_y = start.y + entry_t * (end.y - start.y)
            exit_y = start.y + exit_t * (end.y - start.y)
            lowest_link_y = min(entry_y, exit_y)
            if lowest_link_y <= footprint.max_y + self._height_clearance_m:
                return False
        return True
