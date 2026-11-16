"""Conditional message pointers over shared node/pair encodings."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..risk_learning.domain import CandidateKey
from .adapter import NODE_WIDTH, PAIR_WIDTH
from .distributions import twohot_loss, twohot_mean
from .relations import message_masks, relay_receivers


@dataclass(frozen=True)
class ActorOutput:
    actions: Tensor
    selections: tuple[tuple[CandidateKey, ...], ...]
    log_prob: Tensor
    entropy: Tensor


class MessageActor(nn.Module):
    def __init__(self, config, catalogue) -> None:
        super().__init__()
        hidden = config.hidden
        feature_dim = config.deter + config.stoch * config.classes
        self.max_messages = 4 + int(catalogue.relay)
        self.vehicle_ids = catalogue.vehicle_ids
        self.unimix = config.unimix
        self.register_buffer("slot_limits", torch.tensor((2, 2, int(catalogue.relay)), dtype=torch.long))
        self.node = nn.Sequential(nn.Linear(NODE_WIDTH, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.candidate = nn.Sequential(nn.Linear(PAIR_WIDTH + 2 * hidden, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(feature_dim + 2 * hidden + 9, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.kind_embedding = nn.Embedding(3, hidden)
        self.kind_head = nn.Linear(hidden, 3)
        self.first_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.object_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.receiver_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.message = nn.Sequential(nn.Linear(3 * hidden + 3, hidden), nn.SiLU())
        self.stop_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, math.log(.2 / .8))

    def _pick(self, scores: Tensor, mask: Tensor, greedy: bool) -> tuple[int, Tensor, Tensor]:
        if not mask.any():
            raise ValueError("Conditional pointer has no legal completion.")
        probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        uniform = mask.to(scores.dtype) / mask.sum()
        probabilities = (1. - self.unimix) * probabilities + self.unimix * uniform
        choice = probabilities.argmax() if greedy else torch.multinomial(probabilities, 1).squeeze(0)
        logs = probabilities.clamp_min(torch.finfo(scores.dtype).tiny).log()
        return int(choice), logs[choice], -(probabilities * logs).sum()

    def forward(self, feature: Tensor, observation: dict[str, Tensor], greedy: bool = False) -> ActorOutput:
        return self.sample(feature, observation, greedy)

    def sample(self, feature: Tensor, observation: dict[str, Tensor], greedy: bool = False) -> ActorOutput:
        feature = feature.detach()
        observation = {name: value.detach() for name, value in observation.items()}
        candidates, legal, nodes = observation["candidates"], observation["legal"], observation["nodes"]
        batch, count = nodes.shape[:2]
        if legal.dtype != torch.bool or legal.shape != (batch, count, count, 3):
            raise ValueError("Require three boolean legal channels per node pair.")
        if count != len(self.vehicle_ids):
            raise ValueError("Node count differs from the current row correspondence.")
        embeddings = self.node(nodes)
        left = embeddings[:, :, None, :].expand(-1, -1, count, -1)
        right = embeddings[:, None, :, :].expand(-1, count, -1, -1)
        pairs = self.candidate(torch.cat((candidates, left, right), -1))
        actions = feature.new_zeros(batch, self.max_messages, 4)
        selections, log_probs, entropies = [], [], []
        for row in range(batch):
            obs = {name: value[row] for name, value in observation.items()}
            occupied = legal.new_zeros(count)
            remaining = self.slot_limits.clone()
            selected_sum = embeddings[row].sum(0) * 0.
            log_prob = selected_sum.sum() * 0.
            entropy = log_prob.clone()
            selected = []
            for step in range(self.max_messages + 1):
                uplink, downlink, relay, contacts = message_masks(obs, occupied, remaining)
                masks = (uplink, downlink, relay)
                kinds = torch.stack([mask.any() for mask in masks])
                if not kinds.any():
                    break
                context = self.context(torch.cat((feature[row], embeddings[row].mean(0),
                    selected_sum / max(len(selected), 1),
                    remaining.to(feature.dtype) / self.slot_limits.clamp_min(1), obs["global"]), -1))
                # Independent stop/continue gate. Greedy decoding never compares
                # one STOP probability against thousands of individual messages.
                stop_logit = self.stop_head(context).squeeze(-1)
                gate = torch.stack((stop_logit, stop_logit * 0.))
                stop, lp, ent = self._pick(gate, kinds.new_ones(2), greedy)
                log_prob, entropy = log_prob + lp, entropy + ent
                if stop == 0:
                    break
                kind, lp, ent = self._pick(self.kind_head(context), kinds, greedy)
                log_prob, entropy = log_prob + lp, entropy + ent
                query = context + self.kind_embedding.weight[kind]
                matrix = masks[kind]
                first_scores = self.first_head(torch.cat((embeddings[row], query.expand(count, -1)), -1)).squeeze(-1)
                first, lp, ent = self._pick(first_scores, matrix.any(-1), greedy)
                log_prob, entropy = log_prob + lp, entropy + ent
                object_scores = self.object_head(torch.cat((embeddings[row], pairs[row, first],
                    (query + embeddings[row, first]).expand(count, -1)), -1)).squeeze(-1)
                obj, lp, ent = self._pick(object_scores, matrix[first], greedy)
                log_prob, entropy = log_prob + lp, entropy + ent
                sender, receiver = (first, -1) if kind == 0 else (-1, first)
                if kind == 2:
                    sender = first
                    receiver_scores = self.receiver_head(torch.cat((embeddings[row], pairs[row, :, obj],
                        (query + embeddings[row, first] + embeddings[row, obj]).expand(count, -1)), -1)).squeeze(-1)
                    receiver, lp, ent = self._pick(receiver_scores, relay_receivers(obs, contacts, sender, obj), greedy)
                    log_prob, entropy = log_prob + lp, entropy + ent
                actions[row, step] = actions.new_tensor((kind + 1, sender + 1, receiver + 1, obj + 1))
                ids = self.vehicle_ids
                selected.append((("U", "D", "R")[kind], ids[sender] if sender >= 0 else 0,
                                 ids[receiver] if receiver >= 0 else 0, ids[obj]))
                zero = torch.zeros_like(embeddings[row, 0])
                types = feature.new_zeros(3)
                types[kind] = 1.
                message = self.message(torch.cat((embeddings[row, sender] if sender >= 0 else zero,
                    embeddings[row, receiver] if receiver >= 0 else zero, embeddings[row, obj], types), -1))
                selected_sum = selected_sum + message
                occupied = occupied.clone()
                for endpoint in (sender, receiver):
                    if endpoint >= 0:
                        occupied[endpoint] = True
                remaining = remaining.clone()
                remaining[kind] -= 1
            selections.append(tuple(selected))
            log_probs.append(log_prob)
            entropies.append(entropy)
        return ActorOutput(actions.flatten(1).detach(), tuple(selections), torch.stack(log_probs), torch.stack(entropies))


class SymlogValue(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden
        self.network = nn.Sequential(
            nn.Linear(config.deter + config.stoch * config.classes, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, config.bins))
        self.register_buffer("bins", torch.linspace(-config.symlog_limit, config.symlog_limit, config.bins))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, feature: Tensor) -> Tensor:
        return self.network(feature)

    def logits(self, feature: Tensor) -> Tensor:
        return self(feature)

    def value(self, feature: Tensor) -> Tensor:
        return twohot_mean(self(feature), self.bins)

    def loss(self, feature: Tensor, target: Tensor) -> Tensor:
        return twohot_loss(self(feature), target, self.bins)
