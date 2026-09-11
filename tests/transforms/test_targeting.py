"""Modular SMEAR: the paper's granularity, over a shared block plus deviations.

Not a new method - SMEAR (arxiv 2306.03745) applied the way the paper applies
it. Pinned here:

  * targets are discovered per MODULE, not per block, and each gets its own
    coefficient row (the paper puts a router on each inserted adapter);
  * Linear targets route per position on the running mean of the prefix (the
    paper's per-example pooling, made causal), while elementwise targets take
    the input-free depth prior (the paper does not treat layernorm parameters
    as experts either), so no position reads a later one and no row reads
    another;
  * expert dropout is present, because that is the paper's load-balancing
    mechanism and without it every target collapses to one-hot;
  * ``MERGE_OPAQUE`` subtrees and reference-tied parameters are never merged -
    the two structural exclusions PEER and the Titans memory rely on;
  * the router is EXACTLY identity at init, so a config swap is a clean A/B;
  * the merge really is the paper's merge in a base-plus-deviation basis, i.e.
    it equals the convex combination of the implied experts;
  * the shared trunk receives full gradient whatever the routing does, which is
    the property VEAR's dead experts did not have.
"""

import pytest
import torch
import torch.nn as nn

from praxis import registry
from praxis.routers.smear import SMEAR
from praxis.transforms.targeting import discover_targets


class Opaque(nn.Module):
    """Stands in for PEER: routes itself, so it opts out of merging."""

    MERGE_OPAQUE = True

    def __init__(self, d=32):
        super().__init__()
        self.big = nn.Linear(d, d)

    def forward(self, x):
        return self.big(x)


class Block(nn.Module):
    """A miniature of the real decoder block's shape."""

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


def make(cls=SMEAR, n=4, profile="all", depth=6, dropout=0.0):
    cfg = Cfg()
    cfg.depth = depth
    block = Block(cfg.hidden_size)
    router = cls(cfg, block=block, num_experts=n, target_profile=profile, verbose=False)
    # Dropout is stochastic; most assertions below want the router's own
    # coefficients rather than a particular draw, so it is off unless asked for.
    router.EXPERT_DROPOUT = dropout
    return router, block


# --- targeting ---------------------------------------------------------------


def test_opaque_subtree_is_never_targeted():
    _, block = make()
    groups, skipped = discover_targets(block, registry.lookup("target_profiles", "all"))
    names = {g.name for g in groups}
    assert not any(n.startswith("ffn.") or n == "ffn" for n in names)
    assert skipped["opaque"] == 2  # Opaque.big weight + bias


def test_tied_parameters_are_merged_at_most_once():
    _, block = make()
    block.attn.output.weight = block.attn.qkv.weight  # tie by reference
    groups, skipped = discover_targets(block, registry.lookup("target_profiles", "all"))
    flat = [p for g in groups for p in g.params]
    assert flat.count("attn.qkv.weight") + flat.count("attn.output.weight") == 1
    assert skipped["shared"] == 1


def test_frozen_parameters_are_skipped():
    _, block = make()
    block.attn_norm.weight.requires_grad_(False)
    _, skipped = discover_targets(block, registry.lookup("target_profiles", "all"))
    assert skipped["frozen"] == 1
