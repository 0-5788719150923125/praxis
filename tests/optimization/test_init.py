"""Muon param-splitting + CompositeOptimizer (Muon body / secondary classifier).

The invariants here guard the two things that historically broke Muon: it
must never orthogonalize an embedding or the classifier (the vocab-facing
params route to the secondary/AdamW), and the secondary's learning rate must
survive the cosine scheduler's per-group flattening as a fixed ratio.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.optimization import (
    CompositeOptimizer,
    _create_muon,
    _split_muon_params,
    get_optimizer_profile,
)


class TinyLM(nn.Module):
    """Minimal LM-shaped model: an embedding and classifier share a vocab dimension;
    the two interior linears are the only Muon-eligible matrices."""

    def __init__(self, vocab=50, hidden=16, tie=False):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab, max_position_embeddings=128)
        self.embed = nn.Embedding(vocab, hidden)
        self.h1 = nn.Linear(hidden, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.classifier.weight = self.embed.weight

    def forward(self, x):
        return self.classifier(self.norm(self.h2(self.h1(self.embed(x)))))


def _names(model, params):
    by_id = {id(p): n for n, p in model.named_parameters()}
    return sorted(by_id[id(p)] for p in params)


# --------------------------------------------------------------------------
# Param split: only interior >=2D matrices reach Muon
# --------------------------------------------------------------------------


def test_split_keeps_embeddings_and_classifier_off_muon():
    model = TinyLM()
    muon, adamw = _split_muon_params(model)
    assert _names(model, muon) == ["h1.weight", "h2.weight"]
    # embeddings, classifier (vocab dim), norm weight/bias, and linear biases all go
    # to the secondary/AdamW group.
    adamw_names = _names(model, adamw)
    for n in ("embed.weight", "classifier.weight", "norm.weight", "norm.bias"):
        assert n in adamw_names


def test_split_routes_tied_weight_to_adamw():
    # A tied embed/classifier shares one tensor with a vocab dimension: it must land
    # on AdamW, never Muon.
    model = TinyLM(tie=True)
    muon, adamw = _split_muon_params(model)
    assert "embed.weight" in _names(model, adamw)
    assert all("embed" not in n and "classifier" not in n for n in _names(model, muon))


class ByteClassifier(nn.Module):
    """A classifier sized to its own vocabulary, as a byte-latent encoder's is."""

    def __init__(self, vocab=24, hidden=16, period=24):
        super().__init__()
        self.vocab_size = vocab
        self.hidden_size = hidden
        self.centers = nn.Parameter(torch.randn(vocab, hidden))
        self.scorer = nn.Linear(hidden, vocab, bias=False)
        # Shares the vocabulary's size without facing it (a sequence period).
        self.field = nn.Parameter(torch.randn(period, 4))


class ByteLM(nn.Module):
    def __init__(self, config_vocab=50, vocab=24, hidden=16):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=config_vocab, hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.bank = nn.EmbeddingBag(32, hidden)
        self.h1 = nn.Linear(hidden, hidden)
        self.classifier = ByteClassifier(vocab, hidden)


def test_split_finds_a_classifier_sized_to_its_own_vocabulary():
    """The config's vocab_size is not the byte classifier's: its scorers still
    leave the matrix path, and a tensor that merely shares the size stays."""
    model = ByteLM()
    muon, adamw = _split_muon_params(model)
    adamw_names = _names(model, adamw)
    for n in (
        "classifier.centers",
        "classifier.scorer.weight",
        "embed.weight",
        "bank.weight",
    ):
        assert n in adamw_names, n
    assert _names(model, muon) == ["classifier.field", "h1.weight"]


class _Halve(nn.Module):
    """Stores a weight as half its rows, as the ghost transforms do."""

    def forward(self, x):
        return torch.cat([x, x], dim=0)

    def right_inverse(self, w):
        return w[: w.shape[0] // 2]


def test_split_reads_parametrized_weights_by_their_logical_shape():
    from torch.nn.utils import parametrize

    model = ByteLM()
    parametrize.register_parametrization(model.classifier.scorer, "weight", _Halve())
    parametrize.register_parametrization(model.embed, "weight", _Halve())
    muon, adamw = _split_muon_params(model)
    adamw_names = _names(model, adamw)
    assert "classifier.scorer.parametrizations.weight.original" in adamw_names
    assert "embed.parametrizations.weight.original" in adamw_names
    assert _names(model, muon) == ["classifier.field", "h1.weight"]


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
