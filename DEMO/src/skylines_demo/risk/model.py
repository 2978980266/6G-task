"""Crossing approximation with explicit own-schedule or causal-motion permission."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

from ..config import RiskCommConfig
from ..types import RiskSnapshot, Vector3

MODEL_VERSION = "crossing-area-heading-eta-v3"
CAUSAL_MODEL_VERSION = "crossing-area-heading-eta-v3-causal-self-motion-v1"


def model_version(config: RiskCommConfig) -> str:
    return CAUSAL_MODEL_VERSION if config.ego_eta_mode == "causal-self-motion" else MODEL_VERSION


def own_information_permission(config: RiskCommConfig) -> str:
    return ("current_self_state_and_legal_past_motion_only_for_eta"
            if config.ego_eta_mode == "causal-self-motion"
            else "offline_real_own_passage_step_with_past_interval_eta")


def own_schedule_eta(
    config: RiskCommConfig, crossing_step: int, current_step: int, interval_seconds: float,
) -> float | None:
    """A receives offline-derived own passage step; B never derives ETA from it."""
    if config.ego_eta_mode == "causal-self-motion":
        return None
    return max(crossing_step - current_step, 0) * interval_seconds


@dataclass(frozen=True)
class PassagePrediction:
    distance_m: float | None
    arrival_seconds: float | None
    stationary: bool
    reason: str | None = None


@dataclass(frozen=True)
class PairPrediction:
    risk: bool | None
    reason: str
    distance_m: float | None = None
    ego_distance_m: float | None = None
    object_arrival_seconds: float | None = None
    ego_arrival_seconds: float | None = None
    heading_difference_deg: float | None = None
    temporal_gap_seconds: float | None = None
    stationary: bool = False


def predict_passage(
    snapshot: RiskSnapshot,
    point: Vector3,
    now: float,
    times: Sequence[float],
) -> PassagePrediction:
    """Signed closest-passage time using measured speed and observed current yaw.

    The signed time can describe a recent passage before the decision. This is
    inferred from the snapshot, never from another vehicle's trajectory history.
    A stationary observation models persistent occupancy at its current position.
    """
    if snapshot.velocity_x is None or snapshot.velocity_z is None:
        return PassagePrediction(None, None, False, "missing_velocity")
    if not all(math.isfinite(v) for v in (snapshot.velocity_x, snapshot.velocity_z)):
        return PassagePrediction(None, None, False, "invalid_velocity")
    if not 0 <= snapshot.observation_step < len(times):
        return PassagePrediction(None, None, False, "invalid_observation_step")
    if any(not math.isfinite(v) for v in times) or any(
        b <= a for a, b in zip(times, times[1:])
    ):
        return PassagePrediction(None, None, False, "invalid_time_axis")
    if not math.isfinite(now):
        return PassagePrediction(None, None, False, "invalid_current_time")
    if now < times[snapshot.observation_step]:
        return PassagePrediction(None, None, False, "future_observation")
    if not all(math.isfinite(v) for v in (
        snapshot.x, snapshot.z, snapshot.heading_deg, point.x, point.z,
    )):
        return PassagePrediction(None, None, False, "invalid_position_or_heading")
    speed = math.hypot(snapshot.velocity_x, snapshot.velocity_z)
    if not math.isfinite(speed):
        return PassagePrediction(None, None, False, "invalid_velocity")
    heading = math.radians(snapshot.heading_deg)
    dx, dz = math.sin(heading), math.cos(heading)
    age = now - times[snapshot.observation_step]
    x = snapshot.x + speed * dx * age
    z = snapshot.z + speed * dz * age
    ox, oz = point.x - x, point.z - z
    if not all(math.isfinite(v) for v in (ox, oz)):
        return PassagePrediction(None, None, False, "invalid_extrapolation")
    if speed <= 1e-9:
        return PassagePrediction(math.hypot(ox, oz), 0.0, True)
    along = ox * dx + oz * dz
    distance = math.hypot(ox - along * dx, oz - along * dz)
    arrival = along / speed
    if not math.isfinite(distance) or not math.isfinite(arrival):
        return PassagePrediction(None, None, False, "invalid_extrapolation")
    return PassagePrediction(distance, arrival, False)


def predict_pair(
    ego: RiskSnapshot | None,
    obj: RiskSnapshot,
    point: Vector3,
    now: float,
    times: Sequence[float],
    config: RiskCommConfig,
    scheduled_ego_arrival_seconds: float | None = None,
) -> PairPrediction:
    """Predict nearby crossing passages with compatible arrival times.

    Spatial membership uses the existing truth visit area (cluster radius), not
    the legacy 5m core. A public own schedule supplies ETA with a causal interval.
    Without that schedule, ego must reach the prediction circle within the
    configured horizon. The other passage may be earlier/later by the event gap.
    Stationary occupancy persists over that window; future acceleration/turning
    and unobserved objects are not certified safe by this approximation.
    """
    if config.ego_eta_mode == "causal-self-motion":
        # Enforce the permission even if a caller accidentally passes a schedule.
        scheduled_ego_arrival_seconds = None
    other = predict_passage(obj, point, now, times)
    if other.reason is not None:
        return PairPrediction(None, other.reason)
    if ego is None:
        return PairPrediction(None, "missing_ego_state", distance_m=other.distance_m)
    if not 0 <= ego.observation_step < len(times) or now < times[ego.observation_step]:
        return PairPrediction(None, "future_or_invalid_ego_state", distance_m=other.distance_m)
    if scheduled_ego_arrival_seconds is None:
        own = predict_passage(ego, point, now, times)
        if own.reason is not None:
            return PairPrediction(None, f"ego_{own.reason}", distance_m=other.distance_m)
    else:
        # A explicitly receives a step derived from the offline real own passage.
        # Past intervals estimate elapsed ETA; they do not remove that permission.
        if (not math.isfinite(scheduled_ego_arrival_seconds)
                or scheduled_ego_arrival_seconds < 0 or not math.isfinite(ego.heading_deg)):
            return PairPrediction(None, "invalid_ego_schedule")
        own = PassagePrediction(None, scheduled_ego_arrival_seconds, False)
    angle = abs((obj.heading_deg - ego.heading_deg + 180.0) % 360.0 - 180.0)
    assert own.arrival_seconds is not None and other.arrival_seconds is not None
    if other.stationary:
        gap = 0.0  # The object's observed occupancy is assumed to persist.
    elif own.stationary:
        # Without a public schedule the ego occupies only the modeled window.
        gap = max(0.0, -other.arrival_seconds,
                  other.arrival_seconds - config.extrapolation_seconds)
    else:
        gap = abs(other.arrival_seconds - own.arrival_seconds)
    diagnostics = PairPrediction(None, "unresolved",
        distance_m=other.distance_m, ego_distance_m=own.distance_m,
        object_arrival_seconds=other.arrival_seconds,
        ego_arrival_seconds=own.arrival_seconds,
        heading_difference_deg=angle, temporal_gap_seconds=gap,
        stationary=other.stationary,
    )
    # A scheduled own task exists, but a straight-line model may fail to describe
    # it during a turn, a stop, or at the 5m/8m region boundary. Abstain explicitly.
    own_unpredictable = scheduled_ego_arrival_seconds is None and (
            own.distance_m is None or own.distance_m > config.conflict_cluster_radius_m + 1e-9
            or not -1e-9 <= own.arrival_seconds <= config.extrapolation_seconds + 1e-9)
    if config.ego_eta_mode == "causal-self-motion" and own_unpredictable:
        return replace(diagnostics, reason="ego_passage_not_predictable")
    if not config.crossing_heading_min_deg - 1e-9 <= angle <= config.crossing_heading_max_deg + 1e-9:
        return replace(diagnostics, risk=False, reason="non_crossing_heading")
    if own_unpredictable:
        return replace(diagnostics, reason="ego_passage_not_predictable")
    assert other.distance_m is not None
    if other.distance_m > config.conflict_cluster_radius_m + 1e-9:
        return replace(diagnostics, risk=False, reason="outside_event_area")
    if gap > config.conflict_time_gap_seconds + 1e-9:
        return replace(diagnostics, risk=False, reason="arrival_time_mismatch")
    return replace(diagnostics, risk=True, reason="compatible_crossing_passages")
