"""ParallelHead: gated parallel branches + namespaced per-branch dashboards."""

from functools import partial
from types import SimpleNamespace

import torch

from praxis.heads import HEAD_REGISTRY, ParallelHead
from praxis.heads.harmonic import HarmonicHead
from praxis.metrics.descriptions import get_metric_descriptions


def _cfg(**over):
    base = dict(
        hidden_size=16,
        vocab_size=32,
        max_position_embeddings=64,
        encoder_type="",
        loss_func="cross_entropy",
        crystal_n=None,
        crystal_label_smoothing=None,
        tie_word_embeddings=False,
        embed_size=16,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _parallel(cfg, n=2):
    field = partial(HarmonicHead, amp_modulation="learned", build_classifier=False)
    return ParallelHead(cfg, branches=[field for _ in range(n)])


def _stub(head):
    return SimpleNamespace(
        head=head, contrastive_isotropy=None, tasker=None, encoder=False
    )


def test_transform_preserves_shape_and_gate_normalizes():
    torch.manual_seed(0)
    head = _parallel(_cfg())
    x = torch.randn(2, 8, 16)
    out = head.transform(x)
    assert out.shape == x.shape
    w = torch.softmax(head.gate(x), dim=-1)
    assert torch.allclose(w.sum(-1), torch.ones(2, 8), atol=1e-5)
    assert head._gate_mean is not None and len(head._gate_mean) == 2


def test_gate_is_learned_and_receives_gradient():
    torch.manual_seed(0)
    head = _parallel(_cfg())
    x = torch.randn(2, 8, 16)
    head.transform(x).sum().backward()
    assert head.gate.weight.grad is not None
    assert head.gate.weight.grad.abs().sum() > 0


def test_prismatic_forward_logits_shape():
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic"](_cfg(), encoder=None)
    logits = head(torch.randn(2, 8, 16))
    assert logits.shape == (2, 8, 32)


def test_prismatic_repr_is_nested():
    head = HEAD_REGISTRY["prismatic"](_cfg(), encoder=None)
    assert repr(head) == (
        "Parallel(Sequential(HarmonicField), "
        "Sequential(HarmonicField, CrystalClassifier))"
    )


def test_prismatic_descriptions_namespaced_and_attributed():
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic"](_cfg(), encoder=None)
    descs = get_metric_descriptions(_stub(head))

    for i in (0, 1):
        key = f"p{i}_harmonic_amplitudes_norm"
        assert key in descs, key
        assert descs[key]["caller"] == "HarmonicField"
        assert descs[key]["chart"]["title"].endswith(f"#{i}")

    assert descs["gate_entropy"]["caller"] == "ParallelHead"
    assert descs["gate_weight_0"]["chart"]["series_group"] == "parallel_gate"


def test_training_metrics_namespaced_with_gate():
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic"](_cfg(), encoder=None)
    head(torch.randn(2, 8, 16))  # populate gate stats
    m = head.training_metrics()
    assert {"gate_weight_0", "gate_weight_1", "gate_entropy"} <= set(m)
    assert any(k.startswith("p0_harmonic") for k in m)
    assert any(k.startswith("p1_harmonic") for k in m)


def test_crystal_harmonic_descriptions_unchanged():
    # Regression guard for the SequentialHead.all_metric_descriptions override:
    # the single-field profile must still surface bare (unprefixed) keys.
    torch.manual_seed(0)
    head = HEAD_REGISTRY["crystal_harmonic"](_cfg(), encoder=None)
    descs = head.all_metric_descriptions()
    assert "harmonic_amplitudes_norm" in descs
    assert not any(k.startswith("p0_") for k in descs)
