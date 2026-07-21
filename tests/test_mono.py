"""Tests for mono-forward graph cutting in the sequential decoder
(praxis/decoders/mono.py)."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.modeling import PraxisForCausalLM, PraxisModel


def _config(mono_type, depth=6, **kwargs):
    kwargs.setdefault("decoder_type", "sequential")
    return PraxisConfig(
        vocab_size=1000,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        num_layers=3,
        depth=depth,
        mono_type=mono_type,
        **kwargs,
    )


def _trunk_forward(model, ids):
    """Run the PraxisModel trunk (decoder included) with labels supplied."""
    return PraxisModel.forward(model, input_ids=ids, labels=ids)


def test_layer_mode_cuts_graph_and_scores():
    """The flat regime: every expert call ends in a detach, so the decoder
    output carries NO graph (the head trains alone), while the accumulated
    goodness loss carries gradient into every trunk layer."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_config("layer")).train()
    ids = torch.randint(0, 1000, (2, 16))
    out = _trunk_forward(model, ids)

    assert "mono" in out.losses
    assert not out.last_hidden_state.requires_grad  # the graph is cut

    mono = out.losses.get_loss("mono")
    assert torch.isfinite(mono) and mono.requires_grad
    mono.backward()
    grads = [
        p.grad
        for layer in model.decoder.locals
        for p in layer.parameters()
        if p.requires_grad
    ]
    assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)


def test_no_mono_leaves_graph_intact():
    torch.manual_seed(0)
    model = PraxisForCausalLM(_config(None)).train()
    ids = torch.randint(0, 1000, (2, 16))
    out = _trunk_forward(model, ids)
    assert "mono" not in out.losses
    assert out.last_hidden_state.requires_grad


@pytest.mark.parametrize(
    "mono_type,depth,expected_cuts",
    [("layer", 6, 6), ("cycle", 6, 2), ("final", 6, 1)],
)
def test_cut_schedules(mono_type, depth, expected_cuts):
    """layer = every call; cycle = each full pass through the num_layers pool;
    final = once after the stack."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_config(mono_type, depth=depth)).train()
    ids = torch.randint(0, 1000, (2, 16))
    _trunk_forward(model, ids)
    metrics = model.decoder.mono.metrics()
    assert metrics["mono/cuts"] == expected_cuts
    assert f"mono/goodness_d{expected_cuts - 1}" in metrics


def test_eval_mode_is_a_noop():
    torch.manual_seed(0)
    model = PraxisForCausalLM(_config("layer")).eval()
    ids = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out = _trunk_forward(model, ids)
    assert "mono" not in out.losses


def test_parallel_decoder_rejects_mono():
    with pytest.raises(ValueError, match="sequential"):
        PraxisForCausalLM(_config("layer", decoder_type="parallel_mean"))


def test_byte_latent_latent_goodness():
    """Encoder path: goodness is the latent form (predict the next patch's
    input embedding), the graph still cuts, and the full training step stays
    finite end to end."""
    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_harmonic_serpent",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        byte_level=True,
        head_type="prismatic4",
        mono_type="layer",
    )
    model = PraxisForCausalLM(config).train()
    assert model.decoder.mono.latent  # patch-space goodness, not vocab CE
    ids = torch.randint(4, 260, (2, 24))
    # Latent goodness is self-sufficient (targets come from the decoder's own
    # input stream), so no labels are needed to exercise it.
    out = PraxisModel.forward(model, input_ids=ids)
    assert "mono" in out.losses
    assert not out.last_hidden_state.requires_grad  # graph cut in patch space
    mono = out.losses.get_loss("mono")
    assert torch.isfinite(mono) and mono.requires_grad
    mono.backward()
    grads = [
        p.grad
        for layer in model.decoder.locals
        for p in layer.parameters()
        if p.requires_grad
    ]
    assert grads and any(g is not None and torch.isfinite(g).all() for g in grads)
