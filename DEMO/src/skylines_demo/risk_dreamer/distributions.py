from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def symlog(tensor: Tensor) -> Tensor:
    positive = tensor >= 0
    magnitude = torch.where(positive, tensor, -tensor)
    transformed = torch.log1p(magnitude)
    return torch.where(positive, transformed, -transformed)


def symexp(tensor: Tensor) -> Tensor:
    positive = tensor >= 0
    magnitude = torch.where(positive, tensor, -tensor)
    transformed = torch.expm1(magnitude)
    return torch.where(positive, transformed, -transformed)


def _validate_bins(bins: Tensor) -> None:
    if bins.ndim != 1 or bins.numel() < 2 or not bins.is_floating_point():
        raise ValueError("bins must be a floating tensor with at least two ordered entries.")
    if not torch.isfinite(bins).all() or not (bins[1:] > bins[:-1]).all():
        raise ValueError("bins must be finite and strictly increasing.")


def _working_logits(logits: Tensor) -> Tensor:
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def twohot(target: Tensor, bins: Tensor) -> Tensor:
    """Encode scalar targets on a symlog support, preserving target dimensions."""
    _validate_bins(bins)
    bins = _working_logits(bins)
    target = target.to(device=bins.device, dtype=bins.dtype)
    transformed = symlog(target).clamp(bins[0], bins[-1])
    upper = torch.searchsorted(bins, transformed.contiguous()).clamp(1, bins.numel() - 1)
    lower = upper - 1
    upper_weight = (transformed - bins[lower]) / (bins[upper] - bins[lower])
    encoded = bins.new_zeros((*target.shape, bins.numel()))
    encoded.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
    encoded.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return encoded


def twohot_loss(logits: Tensor, target: Tensor, bins: Tensor) -> Tensor:
    logits = _working_logits(logits)
    if logits.shape[-1] != bins.numel():
        raise ValueError("The last logits dimension must match the number of bins.")
    if target.shape == (*logits.shape[:-1], 1):
        target = target.squeeze(-1)
    if target.shape != logits.shape[:-1]:
        raise ValueError("target dimensions must match the logits batch dimensions.")
    encoded = twohot(target, bins.to(device=logits.device, dtype=logits.dtype))
    return -(encoded * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def twohot_mean(logits: Tensor, bins: Tensor) -> Tensor:
    _validate_bins(bins)
    logits = _working_logits(logits)
    if logits.shape[-1] != bins.numel():
        raise ValueError("The last logits dimension must match the number of bins.")
    bins = bins.to(device=logits.device, dtype=logits.dtype)
    return symexp((F.softmax(logits, dim=-1) * bins).sum(dim=-1))


def categorical_logits(logits: Tensor, unimix: float) -> Tensor:
    if not math.isfinite(unimix) or not 0.0 <= unimix <= 1.0:
        raise ValueError("unimix must be in [0, 1].")
    log_probs = F.log_softmax(_working_logits(logits), dim=-1)
    if unimix == 0.0:
        return log_probs
    uniform = torch.full_like(log_probs, math.log(unimix) - math.log(logits.shape[-1]))
    if unimix == 1.0:
        return log_probs * 0.0 + uniform
    return torch.logaddexp(log_probs + math.log1p(-unimix), uniform)


def categorical_sample(logits: Tensor, sample: bool = True) -> Tensor:
    probs = F.softmax(_working_logits(logits), dim=-1)
    if sample:
        indices = torch.distributions.Categorical(probs=probs).sample()
    else:
        indices = probs.argmax(dim=-1)
    onehot = F.one_hot(indices, num_classes=logits.shape[-1]).to(probs.dtype)
    return onehot + (probs - probs.detach())


def balanced_kl(
    posterior_logits: Tensor, prior_logits: Tensor, free_nats: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Return dynamics and representation KLs over stochastic categories and classes."""
    if posterior_logits.shape != prior_logits.shape or posterior_logits.ndim < 2:
        raise ValueError("Posterior and prior must match and have category and class dimensions.")
    if not math.isfinite(free_nats) or free_nats < 0.0:
        raise ValueError("free_nats must be finite and non-negative.")
    posterior = F.log_softmax(_working_logits(posterior_logits), dim=-1)
    prior = F.log_softmax(_working_logits(prior_logits), dim=-1)
    stopped = posterior.detach()
    dyn = (stopped.exp() * (stopped - prior)).sum(dim=(-2, -1))
    rep = (posterior.exp() * (posterior - prior.detach())).sum(dim=(-2, -1))
    return dyn.clamp_min(free_nats), rep.clamp_min(free_nats)


def lambda_returns(
    rewards: Tensor, values: Tensor, discounts: Tensor, lambda_: float = 0.95
) -> Tensor:
    if rewards.ndim != 2 or discounts.shape != rewards.shape:
        raise ValueError("rewards and discounts must both have shape [batch, horizon].")
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("values must have shape [batch, horizon + 1].")
    if not math.isfinite(lambda_) or not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1].")
    carry = values[:, -1]
    returns = []
    for step in range(rewards.shape[1] - 1, -1, -1):
        carry = rewards[:, step] + discounts[:, step] * (
            (1.0 - lambda_) * values[:, step + 1] + lambda_ * carry
        )
        returns.append(carry)
    return torch.stack(returns[::-1], dim=1) if returns else rewards.clone()


class ReturnEMA(nn.Module):
    """Track finite return quantiles and provide a detached scalar normalization scale."""

    def __init__(self, rate: float = 0.01) -> None:
        super().__init__()
        if not math.isfinite(rate) or not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1].")
        self.rate = rate
        self.register_buffer("lower", torch.tensor(0.0))
        self.register_buffer("upper", torch.tensor(0.0))

    @property
    def scale(self) -> Tensor:
        return (self.upper - self.lower).clamp_min(1.0).detach()

    @torch.no_grad()
    def update(self, targets: Tensor) -> Tensor:
        samples = targets.detach().flatten()
        samples = samples[torch.isfinite(samples)]
        if samples.numel():
            dtype = torch.float64 if samples.dtype == torch.float64 else torch.float32
            samples = samples.to(device=self.lower.device, dtype=dtype)
            quantiles = torch.quantile(samples, samples.new_tensor([0.05, 0.95]))
            if torch.isfinite(quantiles).all():
                lower, upper = quantiles.to(dtype=self.lower.dtype).unbind()
                if torch.isfinite(lower) and torch.isfinite(upper):
                    self.lower.lerp_(lower, self.rate)
                    self.upper.lerp_(upper, self.rate)
        return self.scale

    def forward(self, targets: Tensor, update: bool = True) -> Tensor:
        return self.update(targets) if update else self.scale
