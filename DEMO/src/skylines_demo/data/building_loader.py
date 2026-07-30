from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from ..types import BuildingSnapshot, Vector3

_REQUIRED_COLUMNS = {
    "recordTimeUtc",
    "simulationFrame",
    "gameTime",
    "buildingId",
    "x",
    "y",
    "z",
    "angleY",
    "widthCells",
    "lengthCells",
    "sizeX",
    "sizeY",
    "sizeZ",
    "centerOffsetX",
    "centerOffsetY",
    "centerOffsetZ",
    "minY",
    "maxY",
    "prefabName",
}


def _require_columns(fieldnames: Sequence[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(_REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"Building snapshot CSV is missing columns: {missing}")


def _load_building_csv(csv_path: Path) -> list[BuildingSnapshot]:
    buildings: list[BuildingSnapshot] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames)

        for row in reader:
            buildings.append(
                BuildingSnapshot(
                    building_id=int(float(row["buildingId"])),
                    position=Vector3(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=float(row["z"]),
                    ),
                    angle_y=float(row["angleY"]),
                    width_cells=int(float(row["widthCells"])),
                    length_cells=int(float(row["lengthCells"])),
                    size_x=float(row["sizeX"]),
                    size_y=float(row["sizeY"]),
                    size_z=float(row["sizeZ"]),
                    center_offset_x=float(row["centerOffsetX"]),
                    center_offset_y=float(row["centerOffsetY"]),
                    center_offset_z=float(row["centerOffsetZ"]),
                    min_y=float(row["minY"]),
                    max_y=float(row["maxY"]),
                    prefab_name=row["prefabName"].strip(),
                )
            )
    return buildings


def load_buildings(path: str | Path) -> list[BuildingSnapshot]:
    source_path = Path(path)

    if source_path.is_dir():
        csv_files = sorted(item for item in source_path.glob("*.csv") if item.is_file())
        if not csv_files:
            raise ValueError(f"Building snapshot directory contains no CSV files: {source_path}")
        buildings: list[BuildingSnapshot] = []
        for csv_file in csv_files:
            buildings.extend(_load_building_csv(csv_file))
        return buildings

    return _load_building_csv(source_path)
