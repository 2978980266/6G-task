from __future__ import annotations

from statistics import median
from typing import Sequence


def estimate_step_seconds(history_times: Sequence[float], fallback_seconds: float) -> float:
    """Median positive historical interval, with an explicit configured fallback."""
    intervals = [later - earlier for earlier, later in zip(history_times, history_times[1:])
                 if later > earlier]
    return float(median(intervals)) if intervals else fallback_seconds
