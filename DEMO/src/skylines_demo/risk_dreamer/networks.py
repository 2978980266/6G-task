"""Shared node RSSMs and quadratic pair reconstruction, without ID one-hots."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .config import DreamerConfig
from .adapter import PAIR_WIDTH, NODE_WIDTH, KNOWLEDGE_KNOWN, KNOWLEDGE_LOCAL
from .distributions import (
    balanced_kl, categorical_logits, categorical_sample, symlog, symexp,
    twohot_loss, twohot_mean,
)


def mlp(input_size: int, hidden: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         nn.Linear(hidden, output_size))


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor
    logits: Tensor

    def detach(self) -> RSSMState:
        return RSSMState(self.deter.detach(), self.stoch.detach(), self.logits.detach())


class WorldModel(nn.Module):
    def __init__(self, config: DreamerConfig, catalogue: object) -> None:
        super().__init__()
        self.config = config
        self.node_count = catalogue.node_count
        self.max_messages = 4 + int(catalogue.relay)
        self.register_buffer("bins", torch.linspace(-config.symlog_limit, config.symlog_limit, config.bins))
        hidden = config.hidden
        self.action_size = catalogue.action_size
        self.feature_size = config.deter + config.stoch * config.classes
        self.node_encoder = mlp(NODE_WIDTH, hidden, hidden)
        self.candidate_encoder = mlp(PAIR_WIDTH + 3 + 2 * hidden, hidden, hidden)
        self.encoder = mlp(3 * hidden + 6, hidden, config.embed)
        self.initial_deter = nn.Parameter(torch.zeros(config.deter))
        self.message_encoder = mlp(3 * self.feature_size + 3, hidden, hidden)
        self.role_encoders = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.recurrent_input = mlp(2 * config.stoch * config.classes + hidden, hidden, hidden)
        self.recurrent = nn.GRUCell(hidden, config.deter)
        self.prior = mlp(config.deter, hidden, config.stoch * config.classes)
        self.posterior_net = mlp(config.deter + config.embed, hidden, config.stoch * config.classes)
        self.decoder_context = mlp(self.feature_size, hidden, hidden)
        self.candidate_decoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, PAIR_WIDTH + 3))
        self.node_decoder = nn.Linear(hidden, NODE_WIDTH)
        self.global_decoder = mlp(self.feature_size, hidden, 6)
        self.reward_head = mlp(self.feature_size, hidden, config.bins)
        self.continue_head = mlp(self.feature_size, hidden, 1)
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.reward_head[-1].bias)

    def feature(self, state: RSSMState) -> Tensor:
        return self.node_feature(state).mean(dim=1)

    def node_feature(self, state: RSSMState) -> Tensor:
        return torch.cat((state.deter, state.stoch.flatten(-2)), -1)

    def _state(self, deter: Tensor, logits: Tensor, sample: bool) -> RSSMState:
        logits = categorical_logits(logits.reshape(*deter.shape[:-1], self.config.stoch, self.config.classes),
                                    self.config.unimix)
        return RSSMState(deter, categorical_sample(logits, sample=sample), logits)

    def initial(self, batch_size: int, sample: bool = True) -> RSSMState:
        deter = torch.tanh(self.initial_deter).expand(batch_size, self.node_count, -1)
        return self._state(deter, self.prior(deter), sample)

    def transition(self, state: RSSMState, action: Tensor, sample: bool = True) -> RSSMState:
        batch, count = state.deter.shape[:2]
        messages = action.reshape(batch, self.max_messages, 4)
        if (not torch.isfinite(messages).all() or (messages != messages.round()).any()
                or (messages[..., 0] < 0).any() or (messages[..., 0] > 3).any()
                or (messages[..., 1:] < 0).any() or (messages[..., 1:] > count).any()):
            raise ValueError("Actions must contain categorical kinds and valid one-based node rows.")
        kinds, endpoints = messages[..., 0].long(), messages[..., 1:].long()
        features = self.node_feature(state)
        # Row integers only index tensors; their magnitudes never enter an MLP.
        padded = torch.cat((features.new_zeros(batch, 1, self.feature_size), features), dim=1)
        rows = torch.arange(batch, device=features.device)[:, None]
        role_features = [padded[rows, endpoints[..., role]] for role in range(3)]
        types = functional.one_hot((kinds - 1).clamp_min(0), 3).to(features.dtype)
        message = self.message_encoder(torch.cat((*role_features, types), dim=-1))
        effects = features.new_zeros(batch, count + 1, self.config.hidden)
        for role, encoder in enumerate(self.role_encoders):
            present = (kinds > 0) & (endpoints[..., role] > 0)
            values = encoder(message) * present.unsqueeze(-1)
            effects = effects.scatter_add(1, endpoints[..., role, None].expand(-1, -1, values.shape[-1]), values)
        stoch = state.stoch.flatten(-2)
        pooled = stoch.mean(1, keepdim=True).expand(-1, count, -1)
        recurrent_input = self.recurrent_input(torch.cat((stoch, pooled, effects[:, 1:]), -1))
        deter = self.recurrent(recurrent_input.flatten(0, 1), state.deter.flatten(0, 1)).reshape(batch, count, -1)
        return self._state(deter, self.prior(deter), sample)

    def encode(self, observation: dict[str, Tensor]) -> Tensor:
        candidates, legal, nodes = observation["candidates"], observation["legal"], observation["nodes"]
        batch, count = nodes.shape[:2]
        node_rows = self.node_encoder(symlog(nodes))
        left = node_rows[:, :, None, :].expand(-1, -1, count, -1)
        right = node_rows[:, None, :, :].expand(-1, count, -1, -1)
        pairs = self.candidate_encoder(torch.cat((symlog(candidates), legal.to(nodes.dtype), left, right), -1))
        globals_ = symlog(observation["global"])[:, None, :].expand(-1, count, -1)
        return self.encoder(torch.cat((node_rows, pairs.mean(2), pairs.mean(1), globals_), -1))

    def posterior(self, prior: RSSMState, observation: dict[str, Tensor],
                  sample: bool = True) -> RSSMState:
        return self._state(prior.deter,
                           self.posterior_net(torch.cat((prior.deter, self.encode(observation)), -1)), sample)

    def decode(self, state: RSSMState) -> dict[str, Tensor]:
        context = self.decoder_context(self.node_feature(state))
        count = context.shape[1]
        left = context[:, :, None, :].expand(-1, -1, count, -1)
        right = context[:, None, :, :].expand(-1, count, -1, -1)
        rows = self.candidate_decoder(torch.cat((left, right), -1))
        return dict(candidate_symlog=rows[..., :PAIR_WIDTH], legal_logits=rows[..., PAIR_WIDTH:],
                    node_symlog=self.node_decoder(context),
                    global_symlog=self.global_decoder(self.feature(state)))

    def imagined_observation(self, state: RSSMState) -> dict[str, Tensor]:
        prediction = self.decode(state)
        nodes = symexp(prediction["node_symlog"].clamp(-3., 3.))
        for index in (7, 9, 10, 11, 12, 15):
            nodes[..., index] = prediction["node_symlog"][..., index].sigmoid()
        candidates = symexp(prediction["candidate_symlog"].clamp(-2.3978953, 2.3978953))
        for index in (KNOWLEDGE_KNOWN, KNOWLEDGE_LOCAL):
            candidates[..., index] = prediction["candidate_symlog"][..., index].sigmoid()
        candidates[..., 64] = candidates[..., 64].clamp_min(0.)
        return {"candidates": candidates,
                "legal": prediction["legal_logits"] > 0., "nodes": nodes,
                "global": symexp(prediction["global_symlog"].clamp(-3., 3.))}

    def reward(self, state: RSSMState) -> Tensor:
        return twohot_mean(self.reward_head(self.feature(state)), self.bins)

    def continuation(self, state: RSSMState) -> Tensor:
        return self.continue_head(self.feature(state)).squeeze(-1).sigmoid()

    def reconstruction(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
        legal = target["legal"].float()
        truth = target["candidates"]
        errors = (prediction["candidate_symlog"] - symlog(truth)).square()
        candidate_loss = sum((errors[..., 32*i:32*(i+1)].mean(-1) * legal[..., i]).sum((1, 2))
                             / legal[..., i].sum((1, 2)).clamp_min(1.) for i in (0, 1))
        known = truth[..., KNOWLEDGE_KNOWN]
        age_loss = (errors[..., 64] * known).sum((1, 2)) / known.sum((1, 2)).clamp_min(1.)
        bits = functional.binary_cross_entropy_with_logits(
            prediction["candidate_symlog"][..., 65:67], truth[..., 65:67], reduction="none").mean((1, 2, 3))
        candidate_loss = candidate_loss + age_loss + bits
        mask_error = functional.binary_cross_entropy_with_logits(prediction["legal_logits"], legal, reduction="none")
        positive = (mask_error * legal).sum((1, 2)) / legal.sum((1, 2)).clamp_min(1.)
        negative = (mask_error * (1. - legal)).sum((1, 2)) / (1. - legal).sum((1, 2)).clamp_min(1.)
        binary = (7, 9, 10, 11, 12, 15)
        continuous = (0, 1, 2, 3, 4, 5, 6, 8, 13, 14)
        node_error = (prediction["node_symlog"][..., continuous] - symlog(target["nodes"][..., continuous])).square()
        weights = torch.ones_like(node_error)
        weights[..., :8] = target["nodes"][..., 11:12]
        node_loss = (node_error * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1.)
        node_presence = functional.binary_cross_entropy_with_logits(
            prediction["node_symlog"][..., binary], target["nodes"][..., binary], reduction="none").mean((1, 2))
        global_loss = (prediction["global_symlog"] - symlog(target["global"])).square().mean(-1)
        return dict(candidate=candidate_loss, legal=.5 * (positive + negative).mean(-1),
                    node=node_loss + node_presence, global_=global_loss)

    def loss(self, batch: dict[str, object]) -> tuple[Tensor, RSSMState, dict[str, float]]:
        observations = batch["observations"]
        actions, rewards, continues = batch["actions"], batch["rewards"], batch["continues"]
        observed = batch["observed"]
        batch_size, length = rewards.shape
        prior = self.initial(batch_size)
        states = []
        terms: dict[str, list[Tensor]] = {}
        for index in range(length + 1):
            target = {name: value[:, index] for name, value in observations.items()}
            posterior = self.posterior(prior, target)
            available = observed[:, index]
            state = RSSMState(*(torch.where(available.reshape(batch_size, *([1] * (value.ndim - 1))), value, other)
                                for value, other in zip((posterior.deter, posterior.stoch, posterior.logits),
                                                        (prior.deter, prior.stoch, prior.logits))))
            if index < length:
                if not available.all():
                    raise ValueError("Replay requires complete, unpadded episode observations.")
                states.append(state)
                dyn, rep = balanced_kl(posterior.logits, prior.logits, self.config.free_nats)
                reconstruction = self.reconstruction(self.decode(state), target)
                for name, value in dict(reconstruction, dynamics=dyn, representation=rep).items():
                    terms.setdefault(name, []).append(value.mean())
            if index:
                terms.setdefault("reward", []).append(twohot_loss(
                    self.reward_head(self.feature(state)), rewards[:, index - 1], self.bins).mean())
                terms.setdefault("continuation", []).append(functional.binary_cross_entropy_with_logits(
                    self.continue_head(self.feature(state)).squeeze(-1), continues[:, index - 1]))
            if index < length:
                prior = self.transition(state, actions[:, index])
        averaged = {name: torch.stack(values).mean() for name, values in terms.items()}
        loss = sum(value * (self.config.dyn_scale if name == "dynamics" else
                            self.config.rep_scale if name == "representation" else 1.)
                   for name, value in averaged.items())
        starts = RSSMState(*(torch.stack([getattr(state, name) for state in states], 1).flatten(0, 1).detach()
                             for name in ("deter", "stoch", "logits")))
        metrics = {name + "_loss": value.detach().item() for name, value in averaged.items()}
        return loss, starts, metrics
