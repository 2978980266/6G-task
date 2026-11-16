"""Conditional exact message masks with quadratic peak working storage."""
from __future__ import annotations

import torch
from torch import Tensor

from .adapter import KNOWLEDGE_AGE, KNOWLEDGE_KNOWN, NODE_ACTIVE, NODE_FUTURE


def message_masks(observation: dict[str, Tensor], occupied: Tensor,
                  remaining: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """U(s,o), D(r,o), relay objects(s,o), contacts(s,r), for one scene.

    Sender-wise reduction avoids a sender/receiver/object cube. Arithmetic
    is worst-case cubic, but working storage and neural features are quadratic.
    """
    legal, pairs, nodes = observation["legal"], observation["candidates"], observation["nodes"]
    n = nodes.shape[0]
    diagonal = torch.eye(n, dtype=torch.bool, device=nodes.device)
    free = ~occupied & (nodes[:, NODE_ACTIVE] > .5)
    uplink = legal[..., 0] & free[:, None] & ~diagonal & (remaining[0] > 0)
    downlink = legal[..., 1] & free[:, None] & ~diagonal & (remaining[1] > 0)
    contacts = (legal[..., 2] & free[:, None] & free[None, :] & ~diagonal
                & (nodes[:, NODE_FUTURE] > .5)[None, :] & (remaining[2] > 0))
    known, ages = pairs[..., KNOWLEDGE_KNOWN] > .5, pairs[..., KNOWLEDGE_AGE]
    target_age = torch.where(known, ages, torch.full_like(ages, torch.inf))
    objects = torch.zeros_like(contacts)
    for sender in range(n):
        allowed = contacts[sender, :, None] & ~diagonal
        oldest = target_age.masked_fill(~allowed, -torch.inf).amax(dim=0)
        objects[sender] = known[sender] & (ages[sender] < oldest)
        objects[sender, sender] = False
    return uplink, downlink, objects, contacts


def relay_receivers(observation: dict[str, Tensor], contacts: Tensor,
                    sender: int, obj: int) -> Tensor:
    pairs = observation["candidates"]
    mask = contacts[sender] & ((pairs[:, obj, KNOWLEDGE_KNOWN] <= .5)
                               | (pairs[sender, obj, KNOWLEDGE_AGE] < pairs[:, obj, KNOWLEDGE_AGE]))
    mask = mask.clone()
    mask[obj] = False
    return mask
