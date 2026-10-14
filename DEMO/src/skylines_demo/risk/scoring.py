"""Offline objective and finite-episode feedback, never policy observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum
from typing import Sequence

from .conflicts import RiskTask
from .evaluator import RiskEventResult


@dataclass(frozen=True)
class RiskObjectiveWeights:
    positive_failure: float = 5.0
    negative_false_positive: float = 1.0
    negative_abstention: float = 0.25
    send: float = 0.01
    gamma: float = 1.0


OBJECTIVE_WEIGHTS = RiskObjectiveWeights()
OBJECTIVE_VERSION = "true-object-loss-5-1-0.25-send-0.01-v1"


@dataclass(frozen=True)
class RiskTaskScore:
    task_id: str
    decision_step: int
    true_risk: bool
    outcome: str
    positive_success: bool
    positive_failure: bool
    wrong_object_only_alarm: bool
    positive_abstention: bool
    negative_false_positive: bool
    negative_abstention: bool
    negative_no_risk: bool
    true_alarm_object_ids: tuple[int, ...]
    extra_alarm_object_ids: tuple[int, ...]
    positive_failure_loss: float
    negative_false_positive_loss: float
    negative_abstention_loss: float
    task_loss: float


def score_event(event: RiskEventResult) -> RiskTaskScore:
    """Use truth only after the deadline decision; retain legacy binary fields."""
    if event.decision_step is None or event.outcome not in {
        "risk", "no_risk", "timeout", "insufficient",
    }:
        raise ValueError("Scoring requires a deadline and a supported outcome.")
    alarm = event.outcome == "risk"
    true_ids = tuple(sorted({item.object_id for item in event.object_results
                             if alarm and item.predicted_risk and item.true_conflict}))
    extra_ids = tuple(sorted({item.object_id for item in event.object_results
                              if alarm and item.predicted_risk and not item.true_conflict}))
    success = event.true_risk and bool(true_ids)
    failure = event.true_risk and not success
    abstained = event.outcome in {"timeout", "insufficient"}
    false_positive = not event.true_risk and alarm
    negative_abstention = not event.true_risk and abstained
    positive_loss = OBJECTIVE_WEIGHTS.positive_failure * failure
    false_positive_loss = OBJECTIVE_WEIGHTS.negative_false_positive * false_positive
    abstention_loss = OBJECTIVE_WEIGHTS.negative_abstention * negative_abstention
    return RiskTaskScore(
        event.task_id, event.decision_step, event.true_risk, event.outcome,
        success, failure, event.true_risk and alarm and not success,
        event.true_risk and abstained, false_positive, negative_abstention,
        not event.true_risk and event.outcome == "no_risk", true_ids, extra_ids,
        positive_loss, false_positive_loss, abstention_loss,
        positive_loss + false_positive_loss + abstention_loss,
    )


@dataclass(frozen=True)
class RiskStepFeedback:
    step_index: int
    task_scores: tuple[RiskTaskScore, ...]
    positive_failure_loss: float
    negative_false_positive_loss: float
    negative_abstention_loss: float
    task_loss_sum: float
    uplink_count: int
    downlink_count: int
    relay_count: int
    send_count: int
    uplink_cost: float
    downlink_cost: float
    relay_cost: float
    send_cost_sum: float
    objective_sum: float
    task_reward: float
    send_reward: float
    reward: float


class RiskScoreLedger:
    """Settle exactly once at each deadline; never synthesize truncated results.

    Actual send counts come from execution, including unused/repeated sends.
    A shared send is passed once regardless of how many tasks it may serve.
    """

    def __init__(self, tasks: Sequence[RiskTask], episode_steps: int) -> None:
        if episode_steps < 1:
            raise ValueError("An episode must contain at least one step.")
        self._tasks = {task.task_id: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise ValueError("Task IDs must be unique.")
        if any(not 0 <= task.decision_step < episode_steps for task in tasks):
            raise ValueError("Every task deadline must be inside the real episode.")
        self._episode_steps = episode_steps
        self._scores: dict[str, RiskTaskScore] = {}
        self._feedback: list[RiskStepFeedback] = []

    def settle_step(
        self, step_index: int, events: Sequence[RiskEventResult], *,
        uplink_count: int, downlink_count: int, relay_count: int,
    ) -> RiskStepFeedback:
        if step_index != len(self._feedback) or step_index >= self._episode_steps:
            raise ValueError("Settlement steps must be consecutive and occur only once.")
        if any(type(count) is not int or count < 0
               for count in (uplink_count, downlink_count, relay_count)):
            raise ValueError("Actual send counts must be non-negative integers.")
        due = {task_id for task_id, task in self._tasks.items()
               if task.decision_step == step_index}
        if {event.task_id for event in events} != due or len(events) != len(due):
            raise ValueError("Settle every due task exactly once, and no other task.")
        for event in events:
            task = self._tasks[event.task_id]
            if event.decision_step != step_index or event.true_risk != task.true_risk:
                raise ValueError("Event deadline/label does not match the registered task.")
            if event.task_id in self._scores:
                raise ValueError("This task has already settled.")
        # Validate and compute the whole step before mutating the ledger.
        scores = tuple(score_event(event) for event in events)
        positive_loss = fsum(score.positive_failure_loss for score in scores)
        false_positive_loss = fsum(score.negative_false_positive_loss for score in scores)
        abstention_loss = fsum(score.negative_abstention_loss for score in scores)
        task_loss = fsum(score.task_loss for score in scores)
        send_count = uplink_count + downlink_count + relay_count
        send_cost = OBJECTIVE_WEIGHTS.send * send_count
        objective = task_loss + send_cost
        feedback = RiskStepFeedback(
            step_index, scores, positive_loss, false_positive_loss, abstention_loss,
            task_loss, uplink_count, downlink_count, relay_count, send_count,
            OBJECTIVE_WEIGHTS.send * uplink_count,
            OBJECTIVE_WEIGHTS.send * downlink_count,
            OBJECTIVE_WEIGHTS.send * relay_count, send_cost, objective,
            -task_loss, -send_cost, -objective,
        )
        self._scores.update((score.task_id, score) for score in scores)
        self._feedback.append(feedback)
        return feedback

    def summary(self) -> dict[str, object]:
        scores = tuple(self._scores.values())
        task_count = len(self._tasks)
        positive_count = sum(task.true_risk for task in self._tasks.values())
        negative_count = task_count - positive_count
        complete = len(self._feedback) == self._episode_steps and len(scores) == task_count

        def rate(numerator: int | float, denominator: int) -> float | None:
            return numerator / denominator if complete and denominator else None

        positive_success = sum(score.positive_success for score in scores)
        positive_abstention = sum(score.positive_abstention for score in scores)
        negative_abstention = sum(score.negative_abstention for score in scores)
        false_positive = sum(score.negative_false_positive for score in scores)
        uplinks = sum(step.uplink_count for step in self._feedback)
        downlinks = sum(step.downlink_count for step in self._feedback)
        relays = sum(step.relay_count for step in self._feedback)
        sends = uplinks + downlinks + relays
        positive_loss = fsum(score.positive_failure_loss for score in scores)
        false_positive_loss = fsum(score.negative_false_positive_loss for score in scores)
        abstention_loss = fsum(score.negative_abstention_loss for score in scores)
        loss = fsum(score.task_loss for score in scores)
        cost = OBJECTIVE_WEIGHTS.send * sends
        initial = self._feedback[0] if self._feedback else None
        return {
            "objective_version": OBJECTIVE_VERSION,
            "weights": asdict(OBJECTIVE_WEIGHTS),
            "catalog_task_count": task_count,
            "catalog_positive_count": positive_count,
            "catalog_negative_count": negative_count,
            "settled_task_count": len(scores),
            "pending_task_count": task_count - len(scores),
            "pending_task_ids": tuple(sorted(self._tasks.keys() - self._scores.keys())),
            "task_settlement_complete": len(scores) == task_count,
            "evaluation_complete": complete,
            "steps_run": len(self._feedback),
            "episode_steps": self._episode_steps,
            "positive_success_count": positive_success,
            "positive_success_rate": rate(positive_success, positive_count),
            "positive_failure_count": sum(score.positive_failure for score in scores),
            "wrong_object_only_alarm_count": sum(score.wrong_object_only_alarm for score in scores),
            "positive_abstention_count": positive_abstention,
            "negative_false_positive_count": false_positive,
            "negative_false_positive_rate": rate(false_positive, negative_count),
            "negative_abstention_count": negative_abstention,
            "negative_abstention_rate": rate(negative_abstention, negative_count),
            "negative_no_risk_count": sum(score.negative_no_risk for score in scores),
            "negative_no_risk_rate": rate(sum(score.negative_no_risk for score in scores), negative_count),
            "decision_coverage": rate(task_count - positive_abstention - negative_abstention, task_count),
            "true_alarm_object_count": sum(len(score.true_alarm_object_ids) for score in scores),
            "extra_alarm_object_count": sum(len(score.extra_alarm_object_ids) for score in scores),
            "tasks_with_extra_alarm_count": sum(bool(score.extra_alarm_object_ids) for score in scores),
            "positive_failure_loss": positive_loss,
            "negative_false_positive_loss": false_positive_loss,
            "negative_abstention_loss": abstention_loss,
            "task_loss_sum": loss,
            "uplink_count": uplinks,
            "downlink_count": downlinks,
            "relay_count": relays,
            "send_count": sends,
            "uplink_cost": OBJECTIVE_WEIGHTS.send * uplinks,
            "downlink_cost": OBJECTIVE_WEIGHTS.send * downlinks,
            "relay_cost": OBJECTIVE_WEIGHTS.send * relays,
            "send_cost_sum": cost,
            "objective_sum": loss + cost,
            "mean_task_loss": rate(loss, task_count),
            "mean_objective": rate(loss + cost, task_count),
            "sends_per_task": rate(sends, task_count),
            "total_task_reward": fsum(step.task_reward for step in self._feedback),
            "total_send_reward": fsum(step.send_reward for step in self._feedback),
            "total_reward": fsum(step.reward for step in self._feedback),
            "initial_task_loss_sum": initial.task_loss_sum if initial else 0.0,
            "initial_task_ids": tuple(score.task_id for score in initial.task_scores) if initial else (),
        }
