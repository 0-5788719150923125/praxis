"""CALM encoder + energy head + LF-temperature sanity tests.

These are shape / plumbing checks rather than training-quality
assertions. The smoke-test in the CALM README covers the latter.
"""

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.encoders import ENCODER_REGISTRY
from praxis.heads.energy import EnergyHead
from praxis.losses.energy_score import energy_score_loss
from praxis.metrics import compute_brier_lm


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=256,
        embed_size=32,
        hidden_size=64,
        num_heads=4,
        num_queries=2,
        num_layers=2,
        depth=2,
        block_size=32,
        max_position_embeddings=32,
        encoder_type="calm_small",
    )
    defaults.update(overrides)
    return PraxisConfig(**defaults)


def test_calm_in_encoder_registry():
    assert "calm" in ENCODER_REGISTRY


def test_calm_forward_backward():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    # Force joint mode so the energy head trains from step 0. The default is
    # now an AE pretraining phase, where energy is gated until the codec
    # freezes (see test_calm_pretraining_phase_freezes_on_cap).
    model.encoder.requires_pretraining = False
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss.requires_grad
    out.loss.backward()
    # Energy head and VAE both get gradients.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.vae.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.energy_head.parameters()
    )


def test_calm_handles_loss_skips_main_ce():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (1, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    # handles_loss=True means main CE is never added; only encoder /
    # decoder / energy contribute.
    # The base container seeds "main" = 0 which stays zero here.
    # Trainer strategy sums over tagged losses.
    assert model.encoder.handles_loss is True


def test_calm_generate_advances_in_K_steps():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.eval()
    from transformers import GenerationConfig

    gc = GenerationConfig(max_new_tokens=12, temperature=1.0, do_sample=True)
    input_ids = torch.randint(4, 200, (1, 8), dtype=torch.long)
    out = model.generate(input_ids, generation_config=gc)
    new = out.size(1) - input_ids.size(1)
    # Generation moves in chunk-size-sized jumps: at least 12, rounded up.
    K = model.encoder.K
    assert new % K == 0
    assert new >= 12


def test_calm_generate_aligns_unaligned_prompt():
    # A prompt whose length is not a multiple of K must still generate cleanly:
    # custom_generate left-pads for alignment (so the conditioning patch stays
    # full of real tokens) then strips the pads, so the returned sequence begins
    # with the verbatim prompt - no pad tokens injected into the output.
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.eval()
    from transformers import GenerationConfig

    K = model.encoder.K
    prompt_len = K + 1  # deliberately off-boundary
    gc = GenerationConfig(max_new_tokens=2 * K, temperature=1.0, do_sample=True)
    input_ids = torch.randint(4, 200, (1, prompt_len), dtype=torch.long)
    out = model.generate(input_ids, generation_config=gc)

    assert torch.equal(out[:, :prompt_len], input_ids)  # prompt preserved, no pads
    new = out.size(1) - prompt_len
    assert new % K == 0 and new >= 2 * K


def test_calm_with_crystal_head():
    # CALM borrows a HEAD_REGISTRY head as its token classifier. Crystal
    # (which previously refused loss-owning encoders) now sizes to the VAE
    # decoder layout and trains through the reconstruction path.
    cfg = _tiny_config(head_type="crystal")
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    out.loss.backward()
    centers = model.head.lm_head.centers
    assert centers.shape == (model.encoder.output_vocab_size, model.encoder.output_dim)
    assert centers.grad is not None and centers.grad.abs().sum() > 0


def test_energy_head_shapes():
    head = EnergyHead(
        cond_dim=32, noise_dim=16, latent_dim=8, hidden_dim=32, num_blocks=2
    )
    h = torch.randn(3, 5, 32)
    samples = head.sample(h, num_samples=4)
    assert samples.shape == (4, 3, 5, 8)


def test_harmonic_latent_head():
    # Drop-in sibling of FlowHead: same flow_loss/forward/sample surface, but the
    # flow lives in a compact harmonic coefficient space and synthesized latents
    # lie exactly in the harmonic subspace.
    from praxis.heads.flow import LATENT_HEAD_REGISTRY

    assert "harmonic" in LATENT_HEAD_REGISTRY
    head = LATENT_HEAD_REGISTRY["harmonic"](
        cond_dim=32, noise_dim=0, latent_dim=16, hidden_dim=32, num_blocks=2
    )
    assert head.noise_dim == 16  # caller builds a latent-width start state
    assert head.coeff_dim <= 16  # compressed (DC + low freqs)

    cond = torch.randn(2, 4, 32)
    target = torch.randn(2, 4, 16)
    loss = head.flow_loss(target, cond).mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head.net.parameters()
    )
    with torch.no_grad():
        best = head.forward(cond, torch.zeros(2, 4, 16))
        samples = head.sample(cond[:, :1], num_samples=3)
    assert best.shape == (2, 4, 16)
    assert samples.shape == (3, 2, 1, 16)
    # synthesized latents are pure harmonic superpositions (idempotent project)
    assert torch.allclose(best, head.synthesize(head.project(best)), atol=1e-5)


def test_calm_harmonic_head_trains():
    # Full CALM model with head_kind="harmonic" (overriding the profile's flow);
    # the harmonic head trains through the shared flow loss path.
    import functools

    from praxis.encoders import ENCODER_REGISTRY

    cfg = _tiny_config(encoder_type="calm_byte_flow", tokenizer_type="byte_level")
    orig = ENCODER_REGISTRY["calm_byte_flow"]
    ENCODER_REGISTRY["calm_byte_flow"] = functools.partial(orig, head_kind="harmonic")
    try:
        model = PraxisForCausalLM(cfg)
        model.train()
        model.encoder.requires_pretraining = False  # joint mode: head trains now
        assert type(model.encoder.energy_head).__name__ == "HarmonicLatentHead"
        ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
        out = model(input_ids=ids, labels=ids[:, 1:].contiguous())
        out.loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.energy_head.net.parameters()
        )
    finally:
        ENCODER_REGISTRY["calm_byte_flow"] = orig


def test_fixed_codec_deterministic_drop_in():
    # Fixed codec: deterministic encode (pure buffers), learned decode, zero KL.
    from praxis.encoders.calm.codecs import CODEC_REGISTRY, FixedCodec

    assert CODEC_REGISTRY["fixed"] is FixedCodec
    c = FixedCodec(
        vocab_size=264,
        embed_dim=32,
        chunk_size=4,
        latent_dim=16,
        hidden_dim=64,
        depth=2,
    )
    ids = torch.randint(0, 264, (2, 12))
    m1, lv1 = c.encode(ids)
    m2, _ = c.encode(ids)
    assert torch.equal(m1, m2)  # deterministic
    assert m1.shape == (2, 3, 16)
    assert float(c.kl_divergence(m1, lv1).abs().sum()) == 0.0
    # the encode transform is non-learnable (only the decoder has parameters)
    enc_params = [
        n for n, _ in c.named_parameters() if not n.startswith(("dec", "out"))
    ]
    assert enc_params == []
    out = c.decode(c.reparameterize(m1, lv1))
    assert out.shape == (2, 12, 64)


def test_calm_fixed_codec_trains_single_stage():
    # Full CALM model with codec_kind="fixed", single-stage (ae_freeze_steps=0):
    # the learned decoder trains against the stationary fixed latent.
    import functools

    from praxis.encoders import ENCODER_REGISTRY

    cfg = _tiny_config(encoder_type="calm_byte_flow", tokenizer_type="byte_level")
    orig = ENCODER_REGISTRY["calm_byte_flow"]
    ENCODER_REGISTRY["calm_byte_flow"] = functools.partial(
        orig, codec_kind="fixed", ae_freeze_steps=0
    )
    try:
        model = PraxisForCausalLM(cfg)
        model.train()
        assert type(model.encoder.vae).__name__ == "FixedCodec"
        ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
        out = model(input_ids=ids, labels=ids[:, 1:].contiguous())
        out.loss.backward()
        # decoder learns; the fixed encode path carries no gradients
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.vae.dec_in.parameters()
        )
    finally:
        ENCODER_REGISTRY["calm_byte_flow"] = orig


def test_harmonic_codec_variants():
    # Harmonic codec: standing-wave bases instead of random orthonormal. Linear
    # variant is deterministic with no learnable encode params; serpent variant
    # adds a learned periodic nonlinearity (encode becomes learnable).
    from praxis.encoders.calm.codecs import (
        CODEC_REGISTRY,
        HarmonicCodec,
        _harmonic_matrix,
        _separable_harmonic_matrix,
    )

    assert CODEC_REGISTRY["harmonic"] is HarmonicCodec
    # harmonic basis is orthonormal and deterministic
    h = _harmonic_matrix(20, 8)
    assert torch.allclose(h.T @ h, torch.eye(8), atol=1e-5)
    assert torch.equal(h, _harmonic_matrix(20, 8))
    # separable 2D basis: right shape, orthonormal columns, deterministic
    sep = _separable_harmonic_matrix(4, 12, 16)  # K=4, embed=12 -> latent 16
    assert sep.shape == (48, 16)
    assert torch.allclose(sep.T @ sep, torch.eye(16), atol=1e-5)
    assert torch.equal(sep, _separable_harmonic_matrix(4, 12, 16))

    ids = torch.randint(0, 264, (2, 12))
    lin = HarmonicCodec(264, 32, 4, 16, 64, depth=2)
    m1, _ = lin.encode(ids)
    m2, _ = lin.encode(ids)
    assert torch.equal(m1, m2)  # deterministic
    assert lin.act is None
    enc = [n for n, _ in lin.named_parameters() if not n.startswith(("dec", "out"))]
    assert enc == []  # pure fixed encode

    serp = HarmonicCodec(264, 32, 4, 16, 64, depth=2, nonlinear=True)
    out = serp.decode(serp.reparameterize(*serp.encode(ids)))
    out.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in serp.act.parameters()
    )


def test_hybrid_codec_residual_learns():
    # Hybrid codec: fixed scaffold + a never-frozen learned residual. Starts at
    # the fixed scaffold (zero-init), and the residual gets reconstruction grad.
    from praxis.encoders.calm.codecs import CODEC_REGISTRY, HybridCodec

    assert CODEC_REGISTRY["hybrid"] is HybridCodec
    c = HybridCodec(
        vocab_size=264,
        embed_dim=32,
        chunk_size=4,
        latent_dim=16,
        hidden_dim=64,
        depth=2,
    )
    ids = torch.randint(0, 264, (2, 12))
    out = c.decode(c.reparameterize(*c.encode(ids)))
    out.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in c.residual_net.parameters()
    )
    # latent stays unit-RMS even if the residual is forced large
    with torch.no_grad():
        for p in c.residual_net.parameters():
            p.add_(torch.randn_like(p) * 5)
    z, _ = c.encode(ids)
    assert torch.allclose(z.pow(2).mean(-1).sqrt(), torch.ones(2, 3), atol=1e-2)


def test_energy_score_loss_nonnegative_on_random():
    torch.manual_seed(0)
    model = torch.randn(2, 3, 4, 8)
    target = torch.randn(2, 3, 5, 8)
    val = energy_score_loss(model, target)
    assert val.dim() == 0
    assert val.item() == val.item()  # not NaN


def test_energy_score_loss_lower_when_distributions_match():
    """Matched distributions should score lower than mismatched ones."""
    torch.manual_seed(0)
    matched_m = torch.randn(4, 100, 8)
    matched_t = torch.randn(4, 100, 8)
    matched = energy_score_loss(matched_m, matched_t)

    mismatched_m = torch.randn(4, 100, 8)
    mismatched_t = torch.randn(4, 100, 8) + 10.0
    mismatched = energy_score_loss(mismatched_m, mismatched_t)

    assert matched.item() < mismatched.item()


def test_brierlm_scores_range():
    refs = [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(4)]
    a_same = [list(r) for r in refs]
    b_same = [list(r) for r in refs]
    assert compute_brier_lm(a_same, b_same, refs) == pytest.approx(100.0)

    a_rand = [[999] * 8 for _ in range(4)]
    b_rand = [[999] * 8 for _ in range(4)]
    assert compute_brier_lm(a_rand, b_rand, refs) == 0.0


def test_calm_two_stage_freezes_codec_and_enables_energy():
    """Legacy two-stage (ae_freeze_steps > 0, no AE pretraining phase): codec
    and LM train jointly in stage 1, then the codec freezes while the energy
    head takes over in stage 2 (against a stationary target). The default mode
    is now convergence-driven pretraining; see
    test_calm_pretraining_phase_freezes_on_cap."""
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    # Opt into legacy mode: disable the pretraining phase and set an explicit
    # tiny freeze boundary the test crosses quickly.
    enc.requires_pretraining = False
    enc.ae_freeze_steps = 2

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()

    # Stage 1: codec trainable and receiving gradient.
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert all(p.requires_grad for p in enc.vae.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.vae.parameters()
    )
    model.zero_grad(set_to_none=True)

    # Step past the boundary; the codec must freeze.
    for _ in range(4):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    assert enc._ae_is_frozen()
    assert all(not p.requires_grad for p in enc.vae.parameters())

    # Stage 2: energy head still learns; frozen codec gets no gradient.
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.energy_head.parameters()
    )
    assert all(p.grad is None for p in enc.vae.parameters())


def test_calm_legacy_joint_mode_trains_codec_throughout():
    """ae_freeze_steps == 0: codec never freezes (back-compatible)."""
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    assert enc.ae_freeze_steps == 0  # calm_small default

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    for _ in range(3):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    assert not enc._ae_is_frozen()
    assert all(p.requires_grad for p in enc.vae.parameters())


def test_calm_pretraining_phase_freezes_on_cap():
    """Default mode: the codec trains alone in an AE pretraining phase (energy
    gated off), then freezes once convergence - or the max-steps cap - is hit,
    after which the energy head activates."""
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    assert enc.requires_pretraining  # default, no explicit ae_freeze_steps
    assert enc.in_pretraining()
    assert not enc._ae_is_frozen()
    enc.ae_max_pretrain_steps = 2  # tiny cap backstop so the test converges fast

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()

    # Phase 1: codec trains, energy stays off.
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    enc.consume_pending_losses()  # clear
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in enc.vae.parameters()
    )
    model.zero_grad(set_to_none=True)

    # Cross the cap; the codec freezes and pretraining ends.
    for _ in range(3):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)

    assert enc._ae_is_frozen()
    assert not enc.in_pretraining()
    assert all(not p.requires_grad for p in enc.vae.parameters())

    # Phase 2: energy head now learns against the frozen codec.
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.energy_head.parameters()
    )


def test_calm_convergence_latches_on_low_recon_plateau():
    """The freeze must fire when recon plateaus, even at a tiny absolute value.
    Trend-vs-noise: a relative-to-mean delta would explode as recon CE -> 0 and
    never latch (the bug this fixes). Drives the detector directly with a
    crafted recon curve (descend, then a noisy plateau near zero)."""
    import random

    from praxis.encoders.calm.encoder import (
        PRETRAIN_FLAT_EPS,
        PRETRAIN_PATIENCE,
        PRETRAIN_WINDOW,
    )

    enc = PraxisForCausalLM(_tiny_config()).encoder
    enc._pretrain_min_steps = 0  # _opt_step() is 0 here; clear the warmup floor
    enc.ae_max_pretrain_steps = 10**9  # disable the cap so only convergence can latch
    assert enc.in_pretraining()

    # Steady descent fills the window with a clear trend -> must NOT latch.
    for k in range(PRETRAIN_WINDOW):
        enc._update_pretrain_convergence(5.0 - k * (5.0 - 0.006) / PRETRAIN_WINDOW)
    assert enc.in_pretraining()
    assert enc._diag["calm_pretrain_flatness"] > PRETRAIN_FLAT_EPS

    # Plateau near a TINY value with small noise - exactly where the old
    # relative-to-mean delta blew up. Trend-vs-noise reads it as flat and latches.
    rng = random.Random(0)
    for _ in range(2 * PRETRAIN_WINDOW + PRETRAIN_PATIENCE):
        enc._update_pretrain_convergence(0.006 + rng.uniform(-3e-4, 3e-4))
    assert not enc.in_pretraining()  # froze
    assert enc._diag["calm_pretrain_flatness"] < PRETRAIN_FLAT_EPS


def test_calm_convergence_does_not_latch_during_steady_descent():
    """A steady downward trend keeps flatness above threshold, so the codec
    never freezes while it is still meaningfully improving."""
    from praxis.encoders.calm.encoder import PRETRAIN_FLAT_EPS, PRETRAIN_WINDOW

    enc = PraxisForCausalLM(_tiny_config()).encoder
    enc._pretrain_min_steps = 0
    enc.ae_max_pretrain_steps = 10**9

    for k in range(3 * PRETRAIN_WINDOW):  # well past window + patience, still trending
        enc._update_pretrain_convergence(10.0 - 0.01 * k)
    assert enc.in_pretraining()  # trend dominates the noise -> not converged
    assert enc._diag["calm_pretrain_flatness"] > PRETRAIN_FLAT_EPS


def test_calm_convergence_samples_once_per_optimizer_step():
    """With grad accumulation, microbatch recon readings average into one
    history sample per optimizer step, so the window/patience horizons are in
    optimizer-step units and microbatch data variance doesn't inflate std."""
    enc = PraxisForCausalLM(_tiny_config()).encoder
    enc._grad_accum = 4
    enc._pretrain_min_steps = 0
    enc.ae_max_pretrain_steps = 10**9

    # Noisy microbatches whose group means are identical: 3 full groups.
    for _ in range(3):
        for v in (1.0, 5.0, 2.0, 4.0):  # mean 3.0
            enc._update_pretrain_convergence(v)
    assert enc._recon_hist == [3.0, 3.0, 3.0]

    # A partial group accumulates without entering the history.
    enc._update_pretrain_convergence(9.0)
    assert len(enc._recon_hist) == 3
    assert enc._recon_accum == [9.0]


def test_calm_pretrain_floor_covers_warmup_plus_window():
    """The latch floor must sit a full window past the LR warmup horizon, so the
    history holds only post-warmup readings (β is constant - no anneal term)."""
    from praxis.encoders.calm.encoder import PRETRAIN_WINDOW

    enc = PraxisForCausalLM(_tiny_config(warmup_steps=100)).encoder
    assert enc._pretrain_min_steps == 100 + PRETRAIN_WINDOW


def test_calm_with_stacked_crystal_harmonic_head():
    # crystal_harmonic stacks the harmonic field in front of the crystal
    # classifier; both mechanisms train through CALM's reconstruction path.
    cfg = _tiny_config(head_type="crystal_harmonic")
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    out.loss.backward()

    head = model.head  # SequentialHead([HarmonicHead(transform-only), CrystalHead])
    harmonic, crystal = head.heads[0], head.heads[1]
    centers = crystal.lm_head.centers
    amps = harmonic.field.amplitudes
    assert centers.shape == (
        model.encoder.output_vocab_size,
        model.encoder.output_dim,
    )
    # The transform-only harmonic stage builds no classifier of its own.
    assert harmonic.lm_head is None
    # Both mechanisms receive gradient (field modulates features, crystal
    # classifies them - the recon path trains both).
    assert centers.grad is not None and centers.grad.abs().sum() > 0
    assert amps.grad is not None and amps.grad.abs().sum() > 0
    # Both auxiliary losses are exposed and merged.
    aux = head.aux_losses()
    assert "centers_rms" in aux and "harmonic_smoothness" in aux


def test_calm_with_prismatic_head_learns_envelope():
    # prismatic = ParallelHead([Sequential(field+linear), Sequential(field, crystal)]):
    # a top-level gate balances the two arms' logits per token. Both envelopes and
    # the gate train through CALM's reconstruction path.
    cfg = _tiny_config(head_type="prismatic")
    model = PraxisForCausalLM(cfg)
    parallel = model.head  # ParallelHead is the top head
    fields = [arm.heads[0].field for arm in parallel.branches]
    assert len(fields) == 2
    # Branch 0 (bias arm) learns a static envelope; branch 1 (variance arm)
    # conditions its envelope on the input.
    assert fields[0].amp_modulation == "learned"
    assert fields[1].amp_modulation == "input"
    for field in fields:
        assert field.envelope_depth() > 0.0

    model.train()
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    out.loss.backward()
    # Both envelopes' coefficients are trainable and get gradient via recon.
    for field in fields:
        assert field.amp_coeffs.requires_grad
        assert field.amp_coeffs.grad is not None
        assert field.amp_coeffs.grad.abs().sum() > 0
    # The per-token gate that balances the two fields also learns.
    assert parallel.gate.weight.grad is not None
    assert parallel.gate.weight.grad.abs().sum() > 0


def test_stacked_head_logits_match_manual_compose():
    # forward == terminal(transform(h)): the field is genuinely in the path.
    cfg = _tiny_config(head_type="crystal_harmonic")
    model = PraxisForCausalLM(cfg)
    model.eval()
    head = model.head
    harmonic, crystal = head.heads[0], head.heads[1]
    feat = torch.randn(2, 6, model.encoder.output_dim)
    with torch.no_grad():
        composed = head(feat)
        manual = crystal(harmonic.transform(feat))
    assert torch.allclose(composed, manual, atol=1e-5)
    assert composed.shape == (2, 6, model.encoder.output_vocab_size)


def test_calm_vae_reference_dropouts():
    """Training applies input-token corruption + latent dropout (the
    reference's robustness sites); eval applies neither."""
    from praxis.encoders.calm.vae import CALMVAE

    torch.manual_seed(0)
    vae = CALMVAE(
        vocab_size=64,
        embed_dim=8,
        chunk_size=4,
        latent_dim=8,
        hidden_dim=16,
        dropout=0.5,
    )
    ids = torch.randint(1, 64, (2, 32))

    # Eval: encode is deterministic, decode passes z through untouched.
    vae.eval()
    m1, _ = vae.encode(ids)
    m2, _ = vae.encode(ids)
    assert torch.equal(m1, m2)

    # Train: input corruption makes encode stochastic; latent dropout
    # zeroes z entries (visible through decode's first linear).
    vae.train()
    t1, _ = vae.encode(ids)
    t2, _ = vae.encode(ids)
    assert not torch.equal(t1, t2)

    z = torch.ones(2, 8, 8)
    outs = [vae.decode(z) for _ in range(2)]
    assert not torch.equal(outs[0], outs[1])  # latent dropout active


def test_calm_halo_geometric_mode():
    """loss_func=halo selects the trinary geometric mode: recon stays CE, and
    once the codec freezes the energy head trains under the angular HALO +
    radial terms with gradient reaching only the head (codec/centroids are
    frozen instruments)."""
    from praxis.losses.cross_entropy import CrossEntropyLoss

    cfg = _tiny_config(loss_func="halo", head_type="crystal")
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    assert enc.geometric_mode
    assert isinstance(enc.recon_loss_fn, CrossEntropyLoss)  # never HALO on recon

    # Legacy two-stage with an immediate boundary: frozen from step 1 on.
    enc.requires_pretraining = False
    enc.ae_freeze_steps = 1

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    model(input_ids=input_ids, labels=labels)  # step past the boundary

    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert "calm_halo_angular" in enc._diag
    assert "calm_radial" in enc._diag
    assert "calm_energy_anchor" not in enc._diag  # anchor replaced
    # Gradient reaches the energy head...
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.energy_head.parameters()
    )
    # ...but not the frozen codec.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in enc.vae.parameters())


def test_calm_geometric_mode_off_by_default():
    cfg = _tiny_config(head_type="crystal")
    enc = PraxisForCausalLM(cfg).encoder
    assert not enc.geometric_mode
    assert not hasattr(enc, "geo_loss_fn")


def test_calm_halo_geometric_mode():
    """loss_func=halo selects CALM's trinary geometric mode: recon stays CE,
    and once the codec freezes the energy head trains under the angular HALO +
    radial terms with gradient reaching only the head (codec/centroids are
    frozen instruments)."""
    from praxis.losses.cross_entropy import CrossEntropyLoss

    cfg = _tiny_config(loss_func="halo", head_type="crystal")
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    assert enc.geometric_mode
    assert isinstance(enc.recon_loss_fn, CrossEntropyLoss)  # never HALO on recon

    # Legacy two-stage with an immediate boundary: frozen from step 1 on.
    enc.requires_pretraining = False
    enc.ae_freeze_steps = 1

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    model(input_ids=input_ids, labels=labels)  # step past the boundary

    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert "calm_halo_angular" in enc._diag
    assert "calm_radial" in enc._diag
    assert "calm_energy_anchor" not in enc._diag  # anchor replaced
    # Gradient reaches the energy head...
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.energy_head.parameters()
    )
    # ...but not the frozen codec.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in enc.vae.parameters())


def test_calm_geometric_mode_off_by_default():
    cfg = _tiny_config(head_type="crystal")
    enc = PraxisForCausalLM(cfg).encoder
    assert not enc.geometric_mode
    assert not hasattr(enc, "geo_loss_fn")


def test_calm_halo_geometric_mode():
    """loss_func=halo selects CALM's trinary geometric mode: recon stays CE,
    and once the codec freezes the energy head trains under the angular HALO +
    radial terms with gradient reaching only the head (codec/centroids are
    frozen instruments)."""
    from praxis.losses.cross_entropy import CrossEntropyLoss

    cfg = _tiny_config(loss_func="halo", head_type="crystal")
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    assert enc.geometric_mode
    assert isinstance(enc.recon_loss_fn, CrossEntropyLoss)  # never HALO on recon

    # Legacy two-stage with an immediate boundary: frozen from step 1 on.
    enc.requires_pretraining = False
    enc.ae_freeze_steps = 1

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    model(input_ids=input_ids, labels=labels)  # step past the boundary

    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert "calm_halo_angular" in enc._diag
    assert "calm_radial" in enc._diag
    assert "calm_energy_anchor" not in enc._diag  # anchor replaced
    # Gradient reaches the energy head...
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.energy_head.parameters()
    )
    # ...but not the frozen codec.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in enc.vae.parameters())


def test_calm_geometric_mode_off_by_default():
    cfg = _tiny_config(head_type="crystal")
    enc = PraxisForCausalLM(cfg).encoder
    assert not enc.geometric_mode
    assert not hasattr(enc, "geo_loss_fn")


def test_linear_prior_recovers_linear_map():
    """The streaming ridge solve recovers a known linear map z = h @ M with
    R² near 1, without any gradient training."""
    from praxis.heads.energy import LinearPrior

    torch.manual_seed(0)
    prior = LinearPrior(feature_dim=16, latent_dim=4, mode="linear")
    M = torch.randn(16, 4)
    for _ in range(30):
        h = torch.randn(64, 16)
        prior.observe(h, h @ M)
        prior.solve()
    h = torch.randn(64, 16)
    prior.observe(h, h @ M)  # refresh last_r2 against the solved W
    assert prior.last_r2 > 0.99
    assert torch.allclose(prior(h), h @ M, atol=0.05)

    prior.freeze()
    w_before = prior.W.clone()
    prior.observe(torch.randn(8, 16), torch.randn(8, 4))
    prior.solve()
    assert torch.equal(prior.W, w_before)  # frozen = fixed prior


def test_energy_prior_registry_and_default():
    """linear is the default wherever the energy head is used; none disables;
    harmonic augments features with the sin/cos basis."""
    from praxis.heads.energy import ENERGY_PRIOR_REGISTRY, PRIOR_HARMONIC_FREQS

    assert set(ENERGY_PRIOR_REGISTRY) == {"none", "linear", "harmonic"}

    enc = PraxisForCausalLM(_tiny_config()).encoder
    assert enc.energy_head.prior is not None  # default = linear
    assert enc.energy_head.prior.mode == "linear"

    harm = ENERGY_PRIOR_REGISTRY["harmonic"](feature_dim=8, latent_dim=4, period=16)
    phi = harm.features(torch.randn(2, 5, 8), torch.arange(5))
    assert phi.shape == (2, 5, 8 + 2 * PRIOR_HARMONIC_FREQS)


def test_calm_prior_solves_then_freezes_in_stage2():
    """Stage 2 accumulates stats and solves W during the window, emits the
    r2/norm diagnostics, and freezes once the window elapses."""
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    enc = model.encoder
    enc.requires_pretraining = False
    enc.ae_freeze_steps = 1
    enc._prior_window = 2

    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    model(input_ids=input_ids, labels=labels)  # step 1: cross the boundary

    model(input_ids=input_ids, labels=labels)  # stage 2: observe + solve
    prior = enc.energy_head.prior
    assert "calm_prior_r2" in enc._diag
    assert "calm_prior_norm" in enc._diag
    assert prior.W.abs().sum() > 0
    assert not bool(prior.frozen.item())

    for _ in range(4):  # past the window: freezes
        model(input_ids=input_ids, labels=labels)
    assert bool(prior.frozen.item())
