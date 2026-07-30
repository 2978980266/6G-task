from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

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


def _require_columns(fieldnames: Sequence[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(_REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"Vehicle trajectory CSV is missing columns: {missing}")


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _ingest_csv(csv_path: Path, buckets: dict[int, dict[str, object]]) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames)

        for row in reader:
            frame_id = int(float(row["simulationFrame"]))
            vehicle_id = int(float(row["vehicleId"]))
            vehicle = VehicleSnapshot(
                vehicle_id=vehicle_id,
                position=Vector3(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                ),
                speed=float(row["speed"]),
                angle_x=float(row["angleX"]),
                angle_y=float(row["angleY"]),
                prefab_name=row["prefabName"].strip(),
            )

            bucket = buckets.setdefault(
                frame_id,
                {
                    "game_time": row["gameTime"].strip(),
                    "record_time_utc": row["recordTimeUtc"].strip(),
                    "session_seconds": _parse_optional_float(row.get("sessionSeconds")),
                    "vehicles": {},
                },
            )
            bucket["game_time"] = row["gameTime"].strip()
            bucket["record_time_utc"] = row["recordTimeUtc"].strip()
            parsed_session = _parse_optional_float(row.get("sessionSeconds"))
            if parsed_session is not None:
                bucket["session_seconds"] = parsed_session
            vehicles = bucket["vehicles"]
            assert isinstance(vehicles, dict)
            vehicles[vehicle.vehicle_id] = vehicle


def load_vehicle_replay(path: str | Path, fallback_step_seconds: float = 1.0) -> VehicleReplay:
    source_path = Path(path)
    buckets: dict[int, dict[str, object]] = {}

    if source_path.is_dir():
        csv_files = sorted(item for item in source_path.glob("*.csv") if item.is_file())
        if not csv_files:
            raise ValueError(f"Vehicle trajectory directory contains no CSV files: {source_path}")
        for csv_file in csv_files:
            _ingest_csv(csv_file, buckets)
    else:
        _ingest_csv(source_path, buckets)

    ordered_buckets: list[tuple[int, dict[str, object], float | None, float]] = []
    previous_elapsed: float | None = None
    for step_index, frame_id in enumerate(sorted(buckets)):
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

    return VehicleReplay(frames=tuple(frames))
