from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

from ..types import Vector3, VehicleSnapshot


@dataclass(frozen=True)
class ReplayFrame:
    simulation_frame: int
    game_time: str
    record_time_utc: str
    vehicles: dict[int, VehicleSnapshot]


@dataclass(frozen=True)
class TrajectoryReplay:
    frames: tuple[ReplayFrame, ...]

    @property
    def vehicle_ids(self) -> tuple[int, ...]:
        seen: set[int] = set()
        for frame in self.frames:
            seen.update(frame.vehicles.keys())
        return tuple(sorted(seen))

    def __len__(self) -> int:
        return len(self.frames)


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


def _require_columns(fieldnames: list[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(_REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"Vehicle trajectory CSV is missing columns: {missing}")


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
                    "vehicles": {},
                },
            )
            bucket["game_time"] = row["gameTime"].strip()
            bucket["record_time_utc"] = row["recordTimeUtc"].strip()
            vehicles = bucket["vehicles"]
            assert isinstance(vehicles, dict)
            vehicles[vehicle.vehicle_id] = vehicle


def load_vehicle_replay(path: str | Path) -> TrajectoryReplay:
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

    frames: list[ReplayFrame] = []
    for frame_id in sorted(buckets):
        bucket = buckets[frame_id]
        vehicles = bucket["vehicles"]
        assert isinstance(vehicles, dict)
        frames.append(
            ReplayFrame(
                simulation_frame=frame_id,
                game_time=str(bucket["game_time"]),
                record_time_utc=str(bucket["record_time_utc"]),
                vehicles=dict(sorted(vehicles.items())),
            )
        )

    if not frames:
        raise ValueError(f"Vehicle trajectory source is empty: {source_path}")

    return TrajectoryReplay(frames=tuple(frames))
