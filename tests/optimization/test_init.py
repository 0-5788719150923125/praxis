"""Muon param-splitting + CompositeOptimizer (Muon body / secondary head).

The invariants here guard the two things that historically broke Muon: it
must never orthogonalize an embedding or the LM head (the vocab-facing
params route to the secondary/AdamW), and the secondary's learning rate must
survive the cosine scheduler's per-group flattening as a fixed ratio.
"""

from types import SimpleNamespace

import torch.nn as nn

from praxis.optimization import (
    CompositeOptimizer,
    _create_muon,
    _split_muon_params,
    get_optimizer_profile,
)


class TinyLM(nn.Module):
    """Minimal LM-shaped model: an embedding and head share a vocab dimension;
    the two interior linears are the only Muon-eligible matrices."""

    def __init__(self, vocab=50, hidden=16, tie=False):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab, max_position_embeddings=128)
        self.embed = nn.Embedding(vocab, hidden)
        self.h1 = nn.Linear(hidden, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def forward(self, x):
        return self.lm_head(self.norm(self.h2(self.h1(self.embed(x)))))


def _names(model, params):
    by_id = {id(p): n for n, p in model.named_parameters()}
    return sorted(by_id[id(p)] for p in params)


# --------------------------------------------------------------------------
# Param split: only interior >=2D matrices reach Muon
# --------------------------------------------------------------------------


def test_split_keeps_embeddings_and_head_off_muon():
    model = TinyLM()
    muon, adamw = _split_muon_params(model)
    assert _names(model, muon) == ["h1.weight", "h2.weight"]
    # embeddings, head (vocab dim), norm weight/bias, and linear biases all go
    # to the secondary/AdamW group.
    adamw_names = _names(model, adamw)
    for n in ("embed.weight", "lm_head.weight", "norm.weight", "norm.bias"):
        assert n in adamw_names


def test_split_routes_tied_weight_to_adamw():
    # A tied embed/head shares one tensor with a vocab dimension: it must land
    # on AdamW, never Muon.
    model = TinyLM(tie=True)
    muon, adamw = _split_muon_params(model)
    assert "embed.weight" in _names(model, adamw)
    assert all("embed" not in n and "lm_head" not in n for n in _names(model, muon))


# --------------------------------------------------------------------------
# _create_muon: internal AdamW vs composite secondary
# --------------------------------------------------------------------------


def test_create_muon_internal_adamw_without_secondary():
    profile, _ = get_optimizer_profile("Muon")
    profile["secondary_optimizer"] = None
    opt = _create_muon(TinyLM(), **profile)
    assert type(opt).__name__ == "Muon"


def test_create_muon_builds_composite_with_secondary():
    profile, _ = get_optimizer_profile("Muon")  # secondary_optimizer="Lion"
    opt = _create_muon(TinyLM(), **profile)
    assert isinstance(opt, CompositeOptimizer)
    assert type(opt.primary).__name__ == "Muon"
    assert type(opt.secondary).__name__ == "Lion"
