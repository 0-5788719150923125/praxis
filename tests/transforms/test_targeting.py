"""Tests for praxis/transforms/targeting.py: which parameters a target profile
reaches. ``MERGE_OPAQUE`` subtrees, reference-tied parameters and frozen
parameters are the structural exclusions PEER and the Titans memory rely on."""

import torch
import torch.nn as nn

from praxis import registry
from praxis.transforms.targeting import discover_targets


class Opaque(nn.Module):
    """Stands in for PEER: routes itself, so it opts out of merging."""

    MERGE_OPAQUE = True

    def __init__(self, d=32):
        super().__init__()
        self.big = nn.Linear(d, d)


class Block(nn.Module):
    """A miniature of the real decoder block's parameter layout."""

    def __init__(self, d=32):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(d, d)
        self.attn.output = nn.Linear(d, d)
        self.attn.kappa = nn.Parameter(torch.randn(2, d) * 0.02)
        self.attn_norm = nn.LayerNorm(d)
        self.ffn = Opaque(d)
        self.ffn_norm = nn.LayerNorm(d)


def _discover(block):
    return discover_targets(block, registry.lookup("target_profiles", "all"))


def test_opaque_subtree_is_never_targeted():
    groups, skipped = _discover(Block())
    names = {g.name for g in groups}
    assert not any(n.startswith("ffn.") or n == "ffn" for n in names)
    assert skipped["opaque"] == 2  # Opaque.big weight + bias


def test_tied_parameters_are_merged_at_most_once():
    block = Block()
    block.attn.output.weight = block.attn.qkv.weight  # tie by reference
    groups, skipped = _discover(block)
    flat = [p for g in groups for p in g.params]
    assert flat.count("attn.qkv.weight") + flat.count("attn.output.weight") == 1
    assert skipped["shared"] == 1


def test_frozen_parameters_are_skipped():
    block = Block()
    block.attn_norm.weight.requires_grad_(False)
    _, skipped = _discover(block)
    assert skipped["frozen"] == 1
