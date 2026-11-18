"""Disjoint temporal partitions; label context is restricted to its partition."""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..types import VehicleReplay


@dataclass(frozen=True)
class ReplayWindow:
    split: str
    start: int
    stop: int
    replay: VehicleReplay
    label_context: VehicleReplay


def subset(replay, start, stop):
    return VehicleReplay(tuple(replace(frame, step_index=i)
                               for i, frame in enumerate(replay.frames[start:stop])),
                         replay.identities)


def split_windows(replay, episode_steps=23, gap_steps=4, train_fraction=.6, validation_fraction=.2):
    if (type(episode_steps) is not int or episode_steps < 2
            or type(gap_steps) is not int or gap_steps < 0
            or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1
            or train_fraction + validation_fraction >= 1):
        raise ValueError('Invalid temporal split or episode length.')
    available = len(replay.frames) - 2 * gap_steps
    train_size = int(available * train_fraction)
    val_size = int(available * validation_fraction)
    sizes = (train_size, val_size, available - train_size - val_size)
    if min(sizes) < episode_steps:
        raise ValueError('Recording is too short for train/validation/test windows and gaps.')
    windows = []
    cursor = 0
    for name, size in zip(('train', 'validation', 'test'), sizes):
        context = subset(replay, cursor, cursor + size)
        for start in range(cursor, cursor + size - episode_steps + 1, episode_steps):
            windows.append(ReplayWindow(name, start, start + episode_steps,
                                        subset(replay, start, start + episode_steps), context))
        cursor += size + gap_steps
    return tuple(windows)
