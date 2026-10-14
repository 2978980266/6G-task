"""S0同域规则适配；保留原排序，仅显式共享候选范围。"""
from __future__ import annotations

from dataclasses import replace
import random

from ..config import RiskCommConfig
from ..risk_learning.actions import append_choice, initial_state, legal_mask, to_environment_action
from ..risk_learning.domain import DOMAIN_ID, build_safe_step
from ..types import DownlinkAction, RelayAction, RiskObservation, RiskSchedulingAction, UplinkAction
from .risk_public_uplink import select_deadline_stopped_action, select_public_uplink_action
from .risk_receiver_policy import select_receiver_value_action
from .risk_relay_policy import select_receiver_value_with_relay_action


def select_control(name: str, observation: RiskObservation, config: RiskCommConfig,
                   domain_id: str = DOMAIN_ID, generator: random.Random | None = None) -> RiskSchedulingAction:
    step = build_safe_step(observation, config, domain_id)
    if name == "no-communication":
        return RiskSchedulingAction()
    if name == "random-legal-stop":
        if generator is None:
            raise ValueError("随机控制必须提供显式generator。")
        state = initial_state(step)
        while not state.stopped:
            legal = [i for i, valid in enumerate(legal_mask(step, state)) if i and valid]
            choice = 0 if not legal or generator.random() < .2 else generator.choice(legal)
            state = append_choice(step, state, choice)
        return to_environment_action(step, state)
    allowed = {c.action for c in step.candidates}
    restricted = replace(observation,
                         uplink_candidates=tuple(c for c in observation.uplink_candidates
                                                 if isinstance(c, UplinkAction) and c in allowed),
                         downlink_candidates=tuple(c for c in observation.downlink_candidates
                                                   if isinstance(c, DownlinkAction) and c in allowed),
                         relay_candidates=tuple(c for c in observation.relay_candidates
                                                if isinstance(c, RelayAction) and c in allowed))
    base = select_receiver_value_with_relay_action if config.enable_v2v_relay else select_receiver_value_action
    if name == "receiver-value":
        return base(restricted, config)
    if name == "receiver-value-stop":
        return select_deadline_stopped_action(restricted, config, selector=base)
    if name in {"version-uplink", "demand-uplink"}:
        return select_public_uplink_action(restricted, config,
                                          mode="version" if name == "version-uplink" else "demand",
                                          with_relay=config.enable_v2v_relay)
    raise ValueError("未知S0控制；age等需要另名同域适配。")
