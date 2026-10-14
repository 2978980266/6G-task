"""固定尺度和变长padding；键只用于组合关系，绝不进入数值embedding。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import CandidateKey, SafeStep

ENCODING_VERSION = "risk-safe-32-6-v1"
FEATURE_NAMES = (
    "U", "D", "R", "source_age", "source_known", "rsu_age", "rsu_known",
    "target_age", "target_known", "source_minus_rsu", "source_minus_target",
    "target_tasks", "target_deadline", "target_deadline_known", "U_receivers",
    "U_deadline", "U_deadline_known", "D_risk", "D_no_risk", "D_unknown",
    "D_distance", "D_distance_known", "D_ego_eta", "D_ego_eta_known",
    "D_object_eta", "D_object_eta_known", "D_gap", "D_gap_known",
    "D_source_rsu", "D_velocity_known", "rsu_local_visible", "U_current_local",
)
SCALES = (1., 1., 1., 23., 1., 23., 1., 23., 1., 23., 23., 46., 23., 1.,
          16., 23., 1., 1., 1., 1., 120., 1., 1.5, 1., 1.5, 1., 1.5, 1., 1., 1., 1., 1.)
GLOBAL_NAMES = ("t_over_23", "historical_interval_over_0.3", "remaining_U_over_2",
                "remaining_D_over_2", "remaining_R_over_1", "future_public_task")


@dataclass(frozen=True)
class EncodedStep:
    candidate_features: Any
    global_features: Any
    keys: tuple[CandidateKey, ...]
    padding_mask: Any


def _readonly(array: Any) -> Any:
    array.setflags(write=False)
    return array


def encode_step(step: SafeStep) -> EncodedStep:
    import numpy as np
    features = np.asarray([c.features for c in step.candidates], dtype=np.float32).reshape(-1, 32)
    features = np.clip(np.nan_to_num(features, nan=0., posinf=0., neginf=0.) / np.asarray(SCALES),
                       -10., 10.).astype(np.float32)
    global_features = np.asarray(step.global_features, dtype=np.float32)
    if not np.isfinite(global_features).all():
        raise ValueError("全局安全输入含非有限值。")
    return EncodedStep(_readonly(features), _readonly(global_features),
                       tuple(c.key for c in step.candidates),
                       _readonly(np.zeros(len(step.candidates), dtype=bool)))


def pad_batch(steps: tuple[EncodedStep, ...]) -> tuple[EncodedStep, ...]:
    import numpy as np
    size = max((len(s.keys) for s in steps), default=0)
    result = []
    for step in steps:
        rows = np.zeros((size, 32), dtype=np.float32)
        rows[:len(step.keys)] = step.candidate_features[:len(step.keys)]
        mask = np.arange(size) >= len(step.keys)
        result.append(EncodedStep(_readonly(rows), step.global_features, step.keys, _readonly(mask)))
    return tuple(result)
