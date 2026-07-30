from __future__ import annotations

from typing import Protocol

from ..types import LinkBudget, Vector3


class LineOfSightEvaluator(Protocol):
    def has_line_of_sight(self, start: Vector3, end: Vector3) -> bool:
        ...


class LinkEvaluator(Protocol):
    def evaluate(
        self,
        transmitter_position: Vector3,
        receiver_position: Vector3,
        blockers: tuple[Vector3, ...] = (),
        bandwidth_hz: float | None = None,
        tx_power_dbm: float | None = None,
        rate_limit_mbps: float | None = None,
    ) -> LinkBudget:
        ...
