from __future__ import annotations

import math

from ..config import ChannelConfig
from ..types import LinkBudget, Vector3
from .interfaces import LineOfSightEvaluator


class LogDistanceChannelModel:
    def __init__(self, config: ChannelConfig, los_evaluator: LineOfSightEvaluator) -> None:
        self._config = config
        self._los_evaluator = los_evaluator

    def evaluate(
        self,
        transmitter_position: Vector3,
        receiver_position: Vector3,
        blockers: tuple[Vector3, ...] = (),
        bandwidth_hz: float | None = None,
        tx_power_dbm: float | None = None,
        rate_limit_mbps: float | None = None,
    ) -> LinkBudget:
        distance_m = max(
            transmitter_position.distance_to(receiver_position),
            self._config.min_distance_m,
        )
        effective_bandwidth_hz = (
            self._config.bandwidth_hz if bandwidth_hz is None else bandwidth_hz
        )
        if effective_bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive.")
        effective_tx_power_dbm = (
            self._config.tx_power_dbm if tx_power_dbm is None else tx_power_dbm
        )
        los = self._los_evaluator.has_line_of_sight(transmitter_position, receiver_position)

        path_loss_db = (
            self._config.path_loss_offset_db
            + 10.0 * self._config.path_loss_exponent * math.log10(distance_m)
        )
        if not los:
            path_loss_db += self._config.nlos_extra_loss_db

        vehicle_blocker_count = 0
        vehicle_blockage_loss_db = 0.0
        if self._config.enable_vehicle_blockage and blockers:
            vehicle_blocker_count = sum(
                1
                for blocker in blockers
                if _is_vehicle_blocker(
                    transmitter_position,
                    receiver_position,
                    blocker,
                    corridor_radius_m=self._config.vehicle_blockage_radius_m,
                    blocker_height_m=self._config.vehicle_blockage_height_m,
                )
            )
            if vehicle_blocker_count > 0:
                vehicle_blockage_loss_db = self._config.vehicle_blockage_loss_db
                path_loss_db += vehicle_blockage_loss_db

        path_loss_db += self._config.implementation_margin_db
        rx_power_dbm = effective_tx_power_dbm - path_loss_db
        noise_power_dbm = self._noise_power_dbm(effective_bandwidth_hz)
        snr_db = rx_power_dbm - noise_power_dbm
        snr_linear = max(10.0 ** (snr_db / 10.0), 1e-12)
        spectral_efficiency = (
            math.log2(1.0 + snr_linear) * self._config.rate_efficiency_factor
        )
        max_spectral_efficiency = self._config.max_spectral_efficiency_bps_hz
        if max_spectral_efficiency is not None:
            spectral_efficiency = min(spectral_efficiency, max_spectral_efficiency)
        raw_rate_mbps = (
            effective_bandwidth_hz * max(spectral_efficiency, 0.0) / 1_000_000.0
        )
        rate_mbps = raw_rate_mbps
        if rate_limit_mbps is not None:
            rate_mbps = min(rate_mbps, max(rate_limit_mbps, 0.0))
        reachable = rate_mbps >= self._config.min_rate_mbps

        return LinkBudget(
            distance_m=distance_m,
            los=los,
            path_loss_db=path_loss_db,
            snr_db=snr_db,
            rate_mbps=rate_mbps,
            reachable=reachable,
            vehicle_blocked=vehicle_blocker_count > 0,
            vehicle_blocker_count=vehicle_blocker_count,
            vehicle_blockage_loss_db=vehicle_blockage_loss_db,
            allocated_bandwidth_hz=effective_bandwidth_hz,
            raw_rate_mbps=raw_rate_mbps,
        )

    def _noise_power_dbm(self, bandwidth_hz: float) -> float:
        density = self._config.noise_density_dbm_per_hz
        if density is None:
            return self._config.noise_power_dbm
        return (
            density
            + 10.0 * math.log10(bandwidth_hz)
            + self._config.receiver_noise_figure_db
        )


def _is_vehicle_blocker(
    transmitter_position: Vector3,
    receiver_position: Vector3,
    blocker_position: Vector3,
    corridor_radius_m: float,
    blocker_height_m: float,
) -> bool:
    delta_x = receiver_position.x - transmitter_position.x
    delta_z = receiver_position.z - transmitter_position.z
    horizontal_length_sq = delta_x * delta_x + delta_z * delta_z
    if horizontal_length_sq <= 1e-9:
        return False

    blocker_x = blocker_position.x - transmitter_position.x
    blocker_z = blocker_position.z - transmitter_position.z
    projection = (blocker_x * delta_x + blocker_z * delta_z) / horizontal_length_sq
    if projection <= 0.0 or projection >= 1.0:
        return False

    closest_x = transmitter_position.x + projection * delta_x
    closest_z = transmitter_position.z + projection * delta_z
    lateral_distance = math.hypot(blocker_position.x - closest_x, blocker_position.z - closest_z)
    if lateral_distance > corridor_radius_m:
        return False

    link_height_y = transmitter_position.y + projection * (receiver_position.y - transmitter_position.y)
    blocker_top_y = blocker_position.y + blocker_height_m
    return blocker_top_y >= link_height_y - 0.5
