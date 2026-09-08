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
    return SimpleNamespace(head=head, reg=[], tasker=None, encoder=False)


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
        "Parallel(arms=[Sequential(HarmonicField), "
        "Sequential(HarmonicField, CrystalClassifier)])"
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


def test_prismatic3_three_arms_and_identity_third_branch():
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic3"](_cfg(), encoder=None)
    assert len(head.branches) == 3
    x = torch.randn(2, 6, 16)
    out = head(x)
    assert out.shape == (2, 6, 32)
    # The pure arm's field is identity at init: its transform returns x as-is.
    torch.testing.assert_close(head.branches[2].transform(x), x)
    # And it reads as pure variance once its strands carry any energy.
    fld = head.branches[2].heads[0].field
    assert fld.amp_modulation == "pure"


# ── the blueprint repr ─────────────────────────────────────────────────────


def test_every_leaf_head_names_its_readout():
    """`compose_repr` is what the blueprint tab renders, and the base default
    falls back to the CLASS name. Two leaves never overrode it, so prismatic6-9
    rendered as `[CrystalClassifier, ForwardHead, HaloClassifier]` - one arm
    naming its class where the others name their function, which reads like a
    passthrough or a leftover default instead of the linear readout that is the
    deliberate control arm."""
    import inspect

    import praxis.heads as heads_pkg
    from praxis.heads.base import BaseHead

    import importlib
    import pkgutil

    seen = set()
    for info in pkgutil.iter_modules(heads_pkg.__path__):
        m = importlib.import_module(f"praxis.heads.{info.name}")
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (
                issubclass(obj, BaseHead)
                and obj is not BaseHead
                and not inspect.isabstract(obj)
            ):
                seen.add(obj)

    missing = [c.__name__ for c in seen if c.compose_repr is BaseHead.compose_repr]
    assert not missing, f"leaf heads falling back to their class name: {missing}"
    assert seen, "no head classes discovered"


def test_the_prismatic_arms_read_as_three_classifiers():
    """Geometric, direct, hyperspherical - and the shared harmonic stem in
    front of them, which is the part that looked missing."""
    import torch

    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    for head, geometric in (
        ("prismatic8", "CrystalClassifier"),
        ("prismatic9", "CrystalClassifier"),
    ):
        cfg = PraxisConfig(
            vocab_size=1024,
            hidden_size=64,
            embed_size=64,
            num_heads=2,
            depth=2,
            max_length=512,
            decoder_type="sequential",
            head_type=head,
            loss_func="halo",
            tokenizer_type="byte_level",
        )
        torch.manual_seed(0)
        r = repr(PraxisForCausalLM(cfg).head)
        # Keyword style, like every other module in the blueprint - no invented
        # arrow notation, which no torch repr produces.
        assert "->" not in r, r
        assert "stem=HarmonicField" in r, r
        assert f"arms=[{geometric}, LinearClassifier, " in r, r
        # And honest about the wiring: the HALO arm branches ABOVE the stem.
        assert "HaloClassifier(reads_trunk=True)" in r, r
        assert "ForwardHead" not in r, r


def test_the_stem_does_not_feed_every_arm():
    """The old `HarmonicField -> [...]` notation implied it did. An arm with
    `reads_trunk` branches above the stem and scores the raw trunk hidden
    states - HALOLoss scores those same features, and a transform in front
    would train one feature space and score another."""
    import torch

    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        max_length=512,
        decoder_type="sequential",
        head_type="prismatic8",
        loss_func="halo",
        tokenizer_type="byte_level",
    )
    torch.manual_seed(0)
    head = PraxisForCausalLM(cfg).head
    trunk = torch.randn(2, 6, 64)
    stemmed = head._stem_out(trunk)
    reads = {
        type(b).__name__: head._branch_input(b, stemmed, trunk) is stemmed
        for b in head.branches
    }
    assert reads["CrystalHead"] and reads["ForwardHead"]
    assert not reads["HaloHead"], "HALO must score trunk features, not the stem"
