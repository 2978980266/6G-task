from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class DreamerConfig:
    hidden: int = 128
    deter: int = 128
    stoch: int = 16
    classes: int = 16
    embed: int = 128
    bins: int = 255
    symlog_limit: float = 20.0
    unimix: float = 0.01
    free_nats: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    discount: float = 1.0
    lambda_return: float = 0.95
    entropy_scale: float = 3e-4
    slow_fraction: float = 0.02
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    grad_clip: float = 100.0
    imagination_horizon: int = 15
    imagination_batch: int = 16
    batch_size: int = 4
    replay_capacity: int = 100

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "hidden", "deter", "stoch", "classes", "embed", "bins",
            "imagination_horizon", "imagination_batch", "batch_size", "replay_capacity",
        ):
            value = getattr(self, name)
            minimum = 2 if name in ("classes", "bins") else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in (
            "symlog_limit", "unimix", "free_nats", "dyn_scale", "rep_scale",
            "discount", "lambda_return", "entropy_scale", "slow_fraction",
            "model_lr", "actor_lr", "critic_lr", "grad_clip",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
            if name == "discount":
                if value != 1.0:
                    raise ValueError("discount must equal 1.0 to preserve the original gamma=1 objective.")
            elif name in ("unimix", "lambda_return", "slow_fraction"):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1].")
            elif name in ("free_nats", "dyn_scale", "rep_scale", "entropy_scale"):
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative.")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive.")

    @classmethod
    def smoke(cls) -> DreamerConfig:
        return cls(
            hidden=64,
            deter=64,
            stoch=4,
            classes=4,
            embed=64,
            imagination_horizon=5,
            imagination_batch=4,
            batch_size=2,
        )
