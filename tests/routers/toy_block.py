"""A miniature decoder block and a builder for the SMEAR-family routers.

Shared by the tests of praxis/routers/{smear,vear,distance}.py and the routers
registry sweep. Not collected: the name does not match ``test_*.py``.
"""

import torch
import torch.nn as nn

from praxis.routers.smear import SMEAR


class Opaque(nn.Module):
    """Stands in for PEER: routes itself, so it opts out of merging."""

    MERGE_OPAQUE = True

    def __init__(self, d=32):
        super().__init__()
        self.big = nn.Linear(d, d)

    def forward(self, x):
        return self.big(x)


class Block(nn.Module):
    """A miniature of the real decoder block's shape. Mixes nothing across
    positions or rows, so any such mixing a test sees came from the router."""

    def __init__(self, d=32):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(d, d)
        self.attn.output = nn.Linear(d, d)
        self.attn.kappa = nn.Parameter(torch.randn(2, d) * 0.02)
        self.attn_norm = nn.LayerNorm(d)
        self.ffn = Opaque(d)
        self.ffn_norm = nn.LayerNorm(d)

    def forward(
        self,
        inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        current_depth=0,
        block_ids=None,
        router_weights=None,
        positions=None,
    ):
        h = self.attn_norm(inputs)
        h = self.attn.output(self.attn.qkv(h) * self.attn.kappa[0])
        h = self.ffn(self.ffn_norm(inputs + h))
        return h, past_key_values, current_state, 0.0


class Cfg:
    hidden_size = 32
    depth = 6
    num_experts = 1


def make(cls=SMEAR, n=4, profile="all", depth=6, dropout=0.0, **kwargs):
    """A router of ``cls`` over a fresh ``Block``; returns ``(router, block)``.

    Expert dropout is stochastic and most assertions want the router's own
    coefficients rather than a particular draw, so it is off unless asked for.
    """
    cfg = Cfg()
    cfg.depth = depth
    block = Block(cfg.hidden_size)
    router = cls(
        cfg,
        block=block,
        num_experts=n,
        target_profile=profile,
        verbose=False,
        **kwargs,
    )
    router.EXPERT_DROPOUT = dropout
    return router, block


def router_args(block, x, depth=0):
    """The seven positional arguments LocalLayer passes a router."""
    return (block, x, None, None, None, depth, None)
