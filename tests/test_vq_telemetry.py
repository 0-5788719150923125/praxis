"""VQ telemetry: reset/dead-code/perplexity metrics flowing from the quantizer
buffers up through the abstractinator encoder to the dashboard cards."""

import torch

from praxis.encoders.quantization.vector_quantizer import (
    MultiStageResidualVQ,
    VectorQuantizer,
)


def test_quantizer_accumulates_reset_telemetry():
    torch.manual_seed(0)
    vq = VectorQuantizer(
        K=32,
        D=8,
        reset_interval=2,
        max_codes_to_reset_pct=0.5,
        stale_after=0,
    ).train()
    for _ in range(8):
        vq(torch.randn(2, 4, 8))
    # Most of a fresh 32-code book is unused, so with resets every 2 steps and
    # no staleness gate some sampled codes must have been replaced.
    assert int(vq.reset_count_total) > 0
    assert 0 < int(vq.dead_code_count) <= 32
    # Telemetry is checkpoint state, not ephemeral.
    assert "reset_count_total" in vq.state_dict()
    assert "dead_code_count" in vq.state_dict()


def test_rvq_telemetry_keys_per_stage():
    torch.manual_seed(0)
    rvq = MultiStageResidualVQ(
        K=16, D=8, depth=2, reset_interval=2, stale_after=0
    ).train()
    for _ in range(4):
        rvq(torch.randn(2, 4, 8))
    t = rvq.telemetry()
    for s in range(2):
        assert t[f"vq_perplexity_s{s}"] >= 1.0
        assert 0.0 <= t[f"vq_dead_frac_s{s}"] <= 1.0
        assert t[f"vq_resets_s{s}"] >= 0.0


def test_encoder_metrics_and_cards_end_to_end():
    """The -d-shaped harmonic-bottleneck stack emits VQ metrics through
    model.encoder.training_metrics() (the DynamicsLogger route), and every
    emitted key has a chart description."""
    from praxis import PraxisConfig
    from praxis.encoders.abstractinator.encoder import AbstractinatorEncoder
    from praxis.modeling import PraxisForCausalLM

    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=2,
        encoder_type="abstractinator_harmonic_serpent",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        byte_level=True,
        head_type="prismatic4",
    )
    model = PraxisForCausalLM(cfg).train()
    ids = torch.randint(4, 260, (2, 24))
    model(input_ids=ids, labels=ids[..., 1:].contiguous())

    metrics = model.encoder.training_metrics()
    assert "vq_perplexity" in metrics
    assert any(k.startswith("vq_perplexity_s") for k in metrics)
    assert any(k.startswith("vq_dead_frac_s") for k in metrics)
    assert any(k.startswith("vq_resets_s") for k in metrics)
    for key, value in metrics.items():
        assert key in AbstractinatorEncoder.metric_descriptions, key
        assert value == value  # not NaN
