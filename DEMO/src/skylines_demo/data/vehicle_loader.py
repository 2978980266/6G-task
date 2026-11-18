from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
from typing import Sequence, Tuple

from ..types import FrameRecord, Vector3, VehicleReplay, VehicleSnapshot

_REQUIRED_COLUMNS = {
    "recordTimeUtc",
    "simulationFrame",
    "gameTime",
    "vehicleId",
    "x",
    "y",
    "z",
    "angleX",
    "angleY",
    "speed",
    "prefabName",
}
_IDENTITY_COLUMNS = {"sessionId", "trackId", "visitId"}
_Identity = Tuple[str, int, int]


def _require_columns(fieldnames: Sequence[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(_REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"Vehicle trajectory CSV is missing columns: {missing}")


def _parse_float(value: str | None, column: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Vehicle trajectory {column} must be a finite number.") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Vehicle trajectory {column} must be finite.")
    return parsed


def _parse_integer(value: str | None, column: str) -> int:
    try:
        parsed = Decimal(value or "")
    except InvalidOperation as error:
        raise ValueError(f"Vehicle trajectory {column} must be a finite integer.") from error
    if not parsed.is_finite():
        raise ValueError(f"Vehicle trajectory {column} must be finite.")
    if parsed != parsed.to_integral_value():
        raise ValueError(f"Vehicle trajectory {column} must be an integer.")
    return int(parsed)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return _parse_float(stripped, "sessionSeconds")


def _ingest_csv(
    csv_path: Path,
    buckets: dict[int, dict[str, object]],
    identities: dict[_Identity, dict[str, str | int]],
    tracks: dict[tuple[str, int], tuple[int, str]],
    identity_mode: bool | None,
) -> bool:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames)
        identity_columns = _IDENTITY_COLUMNS.intersection(reader.fieldnames or [])
        if identity_columns and identity_columns != _IDENTITY_COLUMNS:
            raise ValueError("Vehicle trajectory identity requires sessionId, trackId and visitId.")
        has_identity = bool(identity_columns)
        if identity_mode is not None and identity_mode != has_identity:
            raise ValueError("Cannot mix identity-aware and legacy vehicle trajectory CSV files.")

        for row in reader:
            frame_id = _parse_integer(row["simulationFrame"], "simulationFrame")
            vehicle_id = _parse_integer(row["vehicleId"], "vehicleId")
            prefab_name = row["prefabName"].strip()
            identity: int | _Identity = vehicle_id
            session_id = None
            if has_identity:
                if any(not (row.get(column) or "").strip() for column in _IDENTITY_COLUMNS):
                    raise ValueError(f"Vehicle trajectory identity is incomplete at frame {frame_id}.")
                session_id = row["sessionId"].strip()
                track_id = _parse_integer(row["trackId"], "trackId")
                visit_id = _parse_integer(row["visitId"], "visitId")
                identity = (session_id, track_id, visit_id)
                track = (session_id, track_id)
                metadata = (vehicle_id, prefab_name)
                if tracks.setdefault(track, metadata) != metadata:
                    raise ValueError(
                        f"Identity conflict for track {track}: vehicleId or prefabName changed."
                    )
                identities.setdefault(identity, {
                    "session_id": session_id,
                    "track_id": track_id,
                    "visit_id": visit_id,
                    "original_vehicle_id": vehicle_id,
                })
            vehicle = VehicleSnapshot(
                vehicle_id=vehicle_id,
                position=Vector3(
                    x=_parse_float(row["x"], "x"),
                    y=_parse_float(row["y"], "y"),
                    z=_parse_float(row["z"], "z"),
                ),
                speed=_parse_float(row["speed"], "speed"),
                angle_x=_parse_float(row["angleX"], "angleX"),
                angle_y=_parse_float(row["angleY"], "angleY"),
                prefab_name=prefab_name,
            )

            parsed_session = _parse_optional_float(row.get("sessionSeconds"))
            frame_metadata = {
                "game_time": row["gameTime"].strip(),
                "record_time_utc": row["recordTimeUtc"].strip(),
                "session_seconds": parsed_session,
                "session_id": session_id,
            }
            bucket = buckets.setdefault(
                frame_id,
                {**frame_metadata, "vehicles": {}},
            )
            if bucket["session_id"] != session_id:
                raise ValueError(f"Sessions overlap at simulation frame {frame_id}.")
            for column in ("game_time", "record_time_utc", "session_seconds"):
                # Legacy exporters timestamped individual vehicles separately.
                if not has_identity and (
                    column == "record_time_utc"
                    or (column == "session_seconds" and (
                        parsed_session is None or bucket[column] is None
                    ))
                ):
                    continue
                if bucket[column] != frame_metadata[column]:
                    raise ValueError(
                        f"Conflicting frame metadata at frame {frame_id}: {column}."
                    )
            bucket["record_time_utc"] = frame_metadata["record_time_utc"]
            if parsed_session is not None:
                bucket["session_seconds"] = parsed_session
            vehicles = bucket["vehicles"]
            assert isinstance(vehicles, dict)
            if identity in vehicles:
                raise ValueError(f"Duplicate frame identity {identity} at frame {frame_id}.")
            if has_identity and any(v.vehicle_id == vehicle_id for v in vehicles.values()):
                raise ValueError(
                    f"Identity conflict: vehicleId {vehicle_id} has multiple identities "
                    f"at frame {frame_id}."
                )
            vehicles[identity] = vehicle
    return has_identity


def load_vehicle_replay(path: str | Path, fallback_step_seconds: float = 1.0) -> VehicleReplay:
    if not math.isfinite(fallback_step_seconds) or fallback_step_seconds <= 0:
        raise ValueError("fallback_step_seconds must be positive and finite.")
    source_path = Path(path)
    buckets: dict[int, dict[str, object]] = {}
    identities: dict[_Identity, dict[str, str | int]] = {}
    tracks: dict[tuple[str, int], tuple[int, str]] = {}
    identity_mode: bool | None = None

    if source_path.is_dir():
        csv_files = sorted(item for item in source_path.glob("*.csv") if item.is_file())
        if not csv_files:
            raise ValueError(f"Vehicle trajectory directory contains no CSV files: {source_path}")
    else:
        csv_files = [source_path]
    for csv_file in csv_files:
        identity_mode = _ingest_csv(csv_file, buckets, identities, tracks, identity_mode)

    # Allocate once across the entire input so row order and file boundaries do
    # not change the mapping. A separate visit always receives a separate ID.
    internal_ids = {identity: i for i, identity in enumerate(sorted(identities), start=1)}
    replay_identities = {internal_ids[key]: value for key, value in identities.items()}
    sessions = {key[0] for key in identities}
    closed_sessions: set[str] = set()
    previous_session: str | None = None
    last_identity: dict[tuple[str, int], _Identity] = {}
    closed_identities: set[_Identity] = set()
    for frame_id in sorted(buckets):
        bucket = buckets[frame_id]
        if not identity_mode:
            continue
        session_id = bucket["session_id"]
        assert isinstance(session_id, str)
        if len(sessions) > 1 and bucket["session_seconds"] is None:
            raise ValueError("Multiple sessions require nonoverlapping sessionSeconds timelines.")
        if session_id != previous_session:
            if session_id in closed_sessions:
                raise ValueError("Sessions have overlapping simulation frame ranges.")
            if previous_session is not None:
                closed_sessions.add(previous_session)
            previous_session = session_id
        vehicles = bucket["vehicles"]
        assert isinstance(vehicles, dict)
        remapped = {}
        for identity, vehicle in vehicles.items():
            source = (session_id, vehicle.vehicle_id)
            previous_identity = last_identity.get(source)
            if previous_identity is not None and previous_identity != identity:
                closed_identities.add(previous_identity)
            if identity in closed_identities:
                raise ValueError(f"Identity conflict: visits or tracks interleave for {source}.")
            last_identity[source] = identity
            internal_id = internal_ids[identity]
            remapped[internal_id] = replace(vehicle, vehicle_id=internal_id)
        bucket["vehicles"] = remapped

    ordered_buckets: list[tuple[int, dict[str, object], float | None, float]] = []
    previous_elapsed: float | None = None
    for frame_id in sorted(buckets):
        bucket = buckets[frame_id]
        session_seconds = bucket["session_seconds"]
        assert session_seconds is None or isinstance(session_seconds, float)

        if session_seconds is not None:
            elapsed_seconds = session_seconds
            if elapsed_seconds < 0.0:
                raise ValueError(
                    f"Vehicle trajectory sessionSeconds must be non-negative at frame {frame_id}."
                )
            if previous_elapsed is not None and elapsed_seconds <= previous_elapsed:
                raise ValueError(
                    "Vehicle trajectory sessionSeconds must increase between frames: "
                    f"frame {frame_id} has {elapsed_seconds}, previous elapsed time was "
                    f"{previous_elapsed}."
                )
        else:
            elapsed_seconds = (
                0.0
                if previous_elapsed is None
                else previous_elapsed + fallback_step_seconds
            )

        ordered_buckets.append(
            (frame_id, bucket, session_seconds, float(elapsed_seconds))
        )
        previous_elapsed = float(elapsed_seconds)

    frames: list[FrameRecord] = []
    for step_index, (
        frame_id,
        bucket,
        session_seconds,
        elapsed_seconds,
    ) in enumerate(ordered_buckets):
        vehicles = bucket["vehicles"]
        assert isinstance(vehicles, dict)
        delta_seconds = fallback_step_seconds
        if step_index + 1 < len(ordered_buckets):
            next_elapsed_seconds = ordered_buckets[step_index + 1][3]
            delta_seconds = next_elapsed_seconds - elapsed_seconds

        frames.append(
            FrameRecord(
                step_index=step_index,
                simulation_frame=frame_id,
                game_time=str(bucket["game_time"]),
                record_time_utc=str(bucket["record_time_utc"]),
                session_seconds=session_seconds,
                elapsed_seconds=float(elapsed_seconds),
                delta_seconds=float(delta_seconds),
                vehicles=dict(sorted(vehicles.items())),
            )
        )

    if not frames:
        raise ValueError(f"Vehicle trajectory source is empty: {source_path}")

    return VehicleReplay(frames=tuple(frames), identities=dict(sorted(replay_identities.items())))
