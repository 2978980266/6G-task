from __future__ import annotations

from dataclasses import dataclass

from .conflicts import RiskTask
from .executor import RiskDecision, RiskObjectDecision


@dataclass(frozen=True)
class RiskObjectResult:
    object_id: int
    known: bool
    true_conflict: bool
    predicted_risk: bool
    communication_error: bool
    model_error: bool
    observation_step: int | None = None
    distance_m: float | None = None
    enters_risk_radius: bool | None = None
    reason: str | None = None
    diagnostic: RiskObjectDecision | None = None


@dataclass(frozen=True)
class RiskEventResult:
    task_id: str
    true_risk: bool
    outcome: str
    correct: bool
    missed: bool
    false_positive: bool
    timeout: bool
    communication_error: bool
    model_error: bool
    receiver_id: int | None = None
    decision_step: int | None = None
    mixed_error: bool = False
    object_results: tuple[RiskObjectResult, ...] = ()
    insufficient: bool = False
    abstained: bool = False
    decision_reason: str | None = None


def evaluate_decision(
    task: RiskTask,
    decision: RiskDecision,
) -> RiskEventResult:
    predicted_risk = decision.outcome == "risk"
    missed = task.true_risk and not predicted_risk
    false_positive = not task.true_risk and predicted_risk
    timeout = decision.outcome == "timeout"
    known_object_ids = set(decision.known_object_ids)
    related_object_ids = set(task.related_vehicle_ids)
    unknown_related = related_object_ids - known_object_ids
    known_related = related_object_ids & known_object_ids
    mixed_error = missed and bool(unknown_related) and bool(known_related)
    communication_error = missed and bool(unknown_related) and not known_related
    model_error = false_positive or (missed and not unknown_related)
    correct = (task.true_risk and predicted_risk) or (
        not task.true_risk and decision.outcome == "no_risk"
    )
    predicted_object_ids = set(decision.predicted_object_ids)
    diagnostics = {item.object_id: item for item in decision.object_results}
    object_results: list[RiskObjectResult] = []
    for object_id in sorted(known_object_ids | related_object_ids):
        known = object_id in known_object_ids
        true_conflict = object_id in related_object_ids
        predicted = object_id in predicted_object_ids
        diagnostic = diagnostics.get(object_id)
        object_results.append(RiskObjectResult(
            object_id=object_id,
            known=known,
            true_conflict=true_conflict,
            predicted_risk=predicted,
            communication_error=true_conflict and not known,
            model_error=(true_conflict and known and not predicted)
            or (not true_conflict and predicted),
            observation_step=diagnostic.observation_step if diagnostic is not None else None,
            distance_m=diagnostic.distance_m if diagnostic is not None else None,
            enters_risk_radius=diagnostic.enters_risk_radius if diagnostic is not None else None,
            reason=diagnostic.reason if diagnostic is not None else (
                "unknown_object" if not known else None
            ),
            diagnostic=diagnostic,
        ))
    return RiskEventResult(
        task_id=task.task_id,
        true_risk=task.true_risk,
        outcome=decision.outcome,
        correct=correct,
        missed=missed,
        false_positive=false_positive,
        timeout=timeout,
        communication_error=communication_error,
        model_error=model_error,
        receiver_id=task.vehicle_id,
        decision_step=task.decision_step,
        mixed_error=mixed_error,
        object_results=tuple(object_results),
        insufficient=decision.outcome == "insufficient",
        abstained=decision.outcome in {"timeout", "insufficient"},
        decision_reason=decision.reason,
    )
