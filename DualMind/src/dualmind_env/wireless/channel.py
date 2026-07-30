from __future__ import annotations

import math

from ..config import ChannelModelConfig
from ..geometry.primitives import building_to_footprint, segment_intersects_footprint
from ..types import BuildingSnapshot, LinkState, Vector3, VehicleSnapshot


class ChannelEvaluator:
    def __init__(
        self,
        config: ChannelModelConfig,
        rsu_position: Vector3,
        buildings: list[BuildingSnapshot],
    ) -> None:
        self._config = config
        self._rsu_position = rsu_position
        self._footprints = [
            building_to_footprint(building, fallback_cell_size_m=config.building_cell_size_m)
            for building in buildings
        ]

    def has_line_of_sight(self, vehicle_position: Vector3) -> bool:
        for footprint in self._footprints:
            if segment_intersects_footprint(vehicle_position, self._rsu_position, footprint):
                return False
        return True

    def evaluate(self, vehicle: VehicleSnapshot) -> LinkState:
        distance_m = max(
            vehicle.position.horizontal_distance_to(self._rsu_position),
            self._config.min_distance_m,
        )
        los = self.has_line_of_sight(vehicle.position)

        path_loss_db = (
            self._config.path_loss_offset_db
            + 10.0 * self._config.path_loss_exponent * math.log10(distance_m)
        )
        if not los:
            path_loss_db += self._config.nlos_extra_loss_db

        rx_power_dbm = self._config.tx_power_dbm - path_loss_db
        snr_db = rx_power_dbm - self._config.noise_power_dbm
        snr_linear = max(10.0 ** (snr_db / 10.0), 1e-12)
        rate_bps = self._config.bandwidth_hz * math.log2(1.0 + snr_linear)
        rate_mbps = rate_bps / 1_000_000.0
        reachable = rate_mbps >= self._config.min_rate_mbps

        return LinkState(
            vehicle_id=vehicle.vehicle_id,
            distance_m=distance_m,
            los=los,
            path_loss_db=path_loss_db,
            snr_db=snr_db,
            rate_mbps=rate_mbps,
            reachable=reachable,
        )
