"""World-model learning and discrete actor-critic updates in latent imagination."""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .behavior import MessageActor, SymlogValue
from .config import DreamerConfig
from .distributions import ReturnEMA, lambda_returns
from .networks import RSSMState, WorldModel


ALGORITHM_VERSION = "rsu-dreamer-pair-node-imagination-v2"


class DreamerAgent(nn.Module):
    def __init__(self, config: DreamerConfig, catalogue: object) -> None:
        super().__init__()
        self.config = config
        self.world = WorldModel(config, catalogue)
        self.actor = MessageActor(config, catalogue)
        self.critic = SymlogValue(config)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.return_scale = ReturnEMA()
        self.model_optimizer = torch.optim.Adam(self.world.parameters(), lr=config.model_lr, eps=1e-8)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, eps=1e-5)

    def observe(self, observation: dict[str, Tensor], prior: RSSMState | None = None,
                greedy: bool = False) -> tuple[RSSMState, object]:
        if prior is None:
            prior = self.world.initial(observation["global"].shape[0], sample=not greedy)
        state = self.world.posterior(prior, observation, sample=not greedy)
        return state, self.actor.sample(self.world.feature(state), observation, greedy=greedy)

    def _optimize(self, loss: Tensor, optimizer: torch.optim.Optimizer,
                  parameters: list[nn.Parameter]) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Dreamer loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, self.config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError("Non-finite Dreamer parameter after optimizer update.")
        optimizer.zero_grad(set_to_none=True)
        return float(norm.detach())

    def imagine(self, starts: RSSMState) -> dict[str, Tensor]:
        indices = torch.randperm(starts.deter.shape[0], device=starts.deter.device)[:self.config.imagination_batch]
        state = RSSMState(*(getattr(starts, name)[indices].detach() for name in ("deter", "stoch", "logits")))
        features, log_probs, entropies, rewards, continuations = [], [], [], [], []
        for _ in range(self.config.imagination_horizon):
            with torch.no_grad():
                feature = self.world.feature(state)
                observation = self.world.imagined_observation(state)
            choice = self.actor.sample(feature.detach(), observation)
            features.append(feature.detach())
            log_probs.append(choice.log_prob)
            entropies.append(choice.entropy)
            with torch.no_grad():
                state = self.world.transition(state, choice.actions.detach())
                rewards.append(self.world.reward(state))
                continuations.append(self.world.continuation(state))
        features.append(self.world.feature(state).detach())
        return dict(features=torch.stack(features, 1), log_probs=torch.stack(log_probs, 1),
                    entropies=torch.stack(entropies, 1), rewards=torch.stack(rewards, 1),
                    continuations=torch.stack(continuations, 1))

    def update(self, batch: dict[str, object]) -> dict[str, float]:
        self.train()
        model_loss, _, metrics = self.world.loss(batch)
        metrics["model_loss"] = float(model_loss.detach())
        metrics["model_grad_norm"] = self._optimize(model_loss, self.model_optimizer, list(self.world.parameters()))
        with torch.no_grad():
            _, starts, _ = self.world.loss(batch)
        trajectory = self.imagine(starts)
        features = trajectory["features"]
        with torch.no_grad():
            values = self.critic.value(features)
            discounts = self.config.discount * trajectory["continuations"]
            returns = lambda_returns(trajectory["rewards"], values, discounts, self.config.lambda_return)
            scale = self.return_scale(returns)
            advantages = (returns - values[:, :-1]) / scale
            weights = torch.cumprod(torch.cat((torch.ones_like(discounts[:, :1]), discounts[:, :-1]), 1), 1)
        actor_loss = -(weights * (trajectory["log_probs"] * advantages +
                                 self.config.entropy_scale * trajectory["entropies"])).mean()
        metrics["actor_loss"] = float(actor_loss.detach())
        metrics["actor_grad_norm"] = self._optimize(actor_loss, self.actor_optimizer, list(self.actor.parameters()))
        with torch.no_grad():
            slow_target = self.slow_critic.value(features[:, :-1])
        critic_loss = (weights * (self.critic.loss(features[:, :-1], returns) +
                                 self.critic.loss(features[:, :-1], slow_target))).mean()
        metrics["critic_loss"] = float(critic_loss.detach())
        metrics["critic_grad_norm"] = self._optimize(critic_loss, self.critic_optimizer, list(self.critic.parameters()))
        with torch.no_grad():
            for slow, online in zip(self.slow_critic.parameters(), self.critic.parameters()):
                slow.lerp_(online, self.config.slow_fraction)
        metrics.update(imagined_return=float(returns.mean()), return_scale=float(scale),
                       entropy=float(trajectory["entropies"].detach().mean()),
                       imagined_continuation=float(trajectory["continuations"].mean()),
                       imagination_steps=float(self.config.imagination_horizon * features.shape[0]))
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("Non-finite Dreamer diagnostics.")
        return metrics

    def optimizer_state(self) -> dict[str, object]:
        return {name: getattr(self, name + "_optimizer").state_dict() for name in ("model", "actor", "critic")}
