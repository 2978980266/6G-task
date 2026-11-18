"""Causal straight-line pair prediction; no map point or true future input."""
from __future__ import annotations

import math

from .model import PairPrediction, predict_passage
from ..types import Vector3


def predict_crossing(ego, obj, now, times, config):
    if ego is None:
        return PairPrediction(None, 'missing_ego_state')
    # Reuse the snapshot/time validation of the historical passage predictor.
    for snapshot in (ego, obj):
        check = predict_passage(snapshot, Vector3(0., 0., 0.), now, times)
        if check.reason:
            return PairPrediction(None, check.reason)
        if not math.isfinite(snapshot.y):
            return PairPrediction(None, 'invalid_height')
    av = (ego.velocity_x, ego.velocity_z)
    bv = (obj.velocity_x, obj.velocity_z)
    if math.hypot(*av) <= 1e-9 or math.hypot(*bv) <= 1e-9:
        return PairPrediction(None, 'stationary_future_motion_unknown', stationary=True)
    age_a, age_b = now - times[ego.observation_step], now - times[obj.observation_step]
    ax, az = ego.x + av[0] * age_a, ego.z + av[1] * age_a
    bx, bz = obj.x + bv[0] * age_b, obj.z + bv[1] * age_b
    angle = math.degrees(math.acos(max(-1., min(1.,
        (av[0] * bv[0] + av[1] * bv[1]) / (math.hypot(*av) * math.hypot(*bv))))))
    if not config.crossing_heading_min_deg <= angle <= config.crossing_heading_max_deg:
        return PairPrediction(False, 'non_crossing_heading', heading_difference_deg=angle)
    if abs(ego.y - obj.y) > config.event_max_height_difference_m:
        return PairPrediction(False, 'different_height', heading_difference_deg=angle)
    determinant = av[0] * bv[1] - av[1] * bv[0]
    if abs(determinant) <= 1e-10:
        return PairPrediction(False, 'parallel_paths', heading_difference_deg=angle)
    dx, dz = bx - ax, bz - az
    eta_a = (dx * bv[1] - dz * bv[0]) / determinant
    eta_b = (dx * av[1] - dz * av[0]) / determinant
    gap = abs(eta_a - eta_b)
    risk = (-1e-9 <= eta_a <= config.event_horizon_seconds + 1e-9
            and gap <= config.conflict_time_gap_seconds + 1e-9)
    return PairPrediction(risk, 'predicted_pair_crossing' if risk else 'outside_pair_time_window',
                          distance_m=0., ego_distance_m=0., object_arrival_seconds=eta_b,
                          ego_arrival_seconds=eta_a, heading_difference_deg=angle,
                          temporal_gap_seconds=gap)
