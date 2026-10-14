"""共享候选网络及动态合法路径概率；Torch只在实例化时加载。"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any

from .actions import SelectionState, append_choice, initial_state, legal_mask
from .domain import SafeStep
from .encoding import EncodedStep

POLICY_VERSION = "risk-autoregressive-ppo-v1"
NETWORK_SPEC = {"candidate_mlp": [32, 64, 64], "activation": "Tanh",
                "pool": "unpadding_mean_or_zero", "actor_context": 137,
                "critic_input": 70, "stop_initial_probability": .2,
                "hidden_state": False, "id_embedding": False}


@dataclass(frozen=True)
class ActionPath:
    indices: tuple[int, ...]
    masks: tuple[tuple[bool, ...], ...]
    prefixes: tuple[tuple[int, ...], ...]
    joint_log_prob: float
    value: float
    stop_reason: str


class RiskCandidatePolicy:
    def __init__(self) -> None:
        # 当前Anaconda NumPy与用户Torch共用OpenMP；保留已验证的加载顺序。
        # 仍只在实例化学习器时加载，不影响规则/旧任务入口。
        import_module("numpy")
        import torch
        from torch import nn
        self.network = nn.ModuleDict({
            "candidate": nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()),
            "actor": nn.Linear(201, 1),
            "gate": nn.Sequential(nn.Linear(137, 64), nn.Tanh(), nn.Linear(64, 1)),
            "critic": nn.Sequential(nn.Linear(70, 64), nn.Tanh(), nn.Linear(64, 1)),
        }).to(device="cpu")
        gate = self.network["gate"]
        assert isinstance(gate, nn.Sequential)
        gate_head = gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(gate_head.bias, math.log(.2 / .8))
        self.network.train()
        # 保存/优化的网络均为CPU，没有episode隐藏状态。
        assert next(self.network.parameters()).device == torch.device("cpu")

    def parameters(self) -> Any:
        return self.network.parameters()

    def state_dict(self) -> Any:
        return self.network.state_dict()

    def load_state_dict(self, state: Any) -> None:
        self.network.load_state_dict(state, strict=True)

    def distribution(self, encoded: EncodedStep, step: SafeStep,
                     state: SelectionState) -> tuple[Any, Any]:
        import torch
        k = len(step.candidates)
        if encoded.keys != tuple(c.key for c in step.candidates):
            raise ValueError("编码键与候选快照不匹配。")
        if (encoded.candidate_features.shape[1:] != (32,) or len(encoded.padding_mask) < k
                or any(encoded.padding_mask[:k]) or not all(encoded.padding_mask[k:])):
            raise ValueError("非法padding布局。")
        # 从真实前缀切片并复制，padding无论值多大都不会进入网络。
        x = torch.tensor(encoded.candidate_features[:k], dtype=torch.float32)
        global_features = torch.tensor(encoded.global_features, dtype=torch.float32)
        if global_features.shape != (6,) or not torch.isfinite(x).all() or not torch.isfinite(global_features).all():
            raise ValueError("安全数值输入不合法。")
        embeddings = self.network["candidate"](x)
        pool = embeddings.mean(dim=0) if k else torch.zeros(64)
        value = self.network["critic"](torch.cat((pool, global_features))).squeeze()
        selected = (embeddings[[i - 1 for i in state.selected_indices]].mean(dim=0)
                    if state.selected_indices else torch.zeros(64))
        remaining = torch.tensor([state.remaining[0] / 2., state.remaining[1] / 2.,
                                  float(state.remaining[2])])
        context = torch.cat((pool, global_features, remaining, selected))
        mask = torch.tensor(legal_mask(step, state), dtype=torch.bool)
        if not mask[0]:
            raise ValueError("STOP后的状态没有分布。")
        if not torch.isfinite(value):
            raise FloatingPointError("策略产生非有限value。")
        if not mask[1:].any():
            return torch.cat((torch.ones(1), torch.zeros(k))), value
        logits = self.network["actor"](torch.cat((embeddings, context.expand(k, -1)), dim=1)).squeeze(-1)
        gate_logit = self.network["gate"](context).squeeze()
        if not torch.isfinite(logits).all() or not torch.isfinite(gate_logit) or not torch.isfinite(value):
            raise FloatingPointError("策略产生非有限logit/value。")
        stop = torch.sigmoid(gate_logit)
        conditional = torch.softmax(logits.masked_fill(~mask[1:], -torch.inf), dim=0)
        return torch.cat((stop.reshape(1), (1. - stop) * conditional)), value

    def _path(self, encoded: EncodedStep, step: SafeStep, generator: Any = None,
              indices: tuple[int, ...] | None = None, greedy: bool = False) -> ActionPath:
        import torch
        state = initial_state(step)
        choices: list[int] = []
        masks: list[tuple[bool, ...]] = []
        prefixes: list[tuple[int, ...]] = []
        logp = 0.
        initial_value = 0.
        reason = ""
        with torch.no_grad():
            while not state.stopped:
                probs, value = self.distribution(encoded, step, state)
                mask = legal_mask(step, state)
                forced = not mask[1:].any()
                if not choices:
                    initial_value = float(value)
                if indices is not None:
                    if len(choices) >= len(indices):
                        raise ValueError("路径必须以STOP结束。")
                    choice = indices[len(choices)]
                elif forced:
                    choice = 0
                elif greedy:
                    best = probs.max().item()
                    tied = [i for i, prob in enumerate(probs) if prob.item() == best]
                    choice = min(tied, key=lambda i: ("", 0, 0, 0) if i == 0 else step.candidates[i - 1].key)
                else:
                    choice = int(torch.multinomial(probs, 1, generator=generator).item())
                prefixes.append(state.selected_indices)
                masks.append(tuple(bool(v) for v in mask))
                state = append_choice(step, state, choice)
                choices.append(choice)
                if not forced:
                    logp += float(torch.log(probs[choice]))
                if choice == 0:
                    reason = "forced_no_legal" if forced else "voluntary"
        if indices is not None and len(choices) != len(indices):
            raise ValueError("STOP后仍有路径索引。")
        return ActionPath(tuple(choices), tuple(masks), tuple(prefixes), logp, initial_value, reason)

    def sample(self, encoded: EncodedStep, step: SafeStep, generator: Any) -> ActionPath:
        return self._path(encoded, step, generator)

    def greedy(self, encoded: EncodedStep, step: SafeStep) -> ActionPath:
        return self._path(encoded, step, greedy=True)

    def path_from_indices(self, encoded: EncodedStep, step: SafeStep,
                          indices: tuple[int, ...]) -> ActionPath:
        return self._path(encoded, step, indices=indices)

    def evaluate_path(self, encoded: EncodedStep, step: SafeStep,
                      path: ActionPath) -> tuple[Any, Any]:
        import torch
        if (not path.indices or path.indices[-1] != 0
                or len(path.indices) != len(path.masks) or len(path.indices) != len(path.prefixes)):
            raise ValueError("路径长度/STOP不完整。")
        state = initial_state(step)
        logp: Any = torch.zeros(())
        initial_value: Any = None
        for index, saved_mask, saved_prefix in zip(path.indices, path.masks, path.prefixes):
            mask = legal_mask(step, state)
            if tuple(bool(v) for v in mask) != saved_mask or state.selected_indices != saved_prefix:
                raise ValueError("路径保存掩码/前缀不一致。")
            probs, value = self.distribution(encoded, step, state)
            if initial_value is None:
                initial_value = value
                logp = value * 0.
            forced = not mask[1:].any()
            state = append_choice(step, state, index)
            if not forced:
                logp = logp + torch.log(probs[index])
        if not torch.isfinite(logp) or not torch.isfinite(initial_value):
            raise FloatingPointError("路径概率/value非有限。")
        return logp, initial_value
