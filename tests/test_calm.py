"""CALM encoder + energy head + LF-temperature sanity tests.

These are shape / plumbing checks rather than training-quality
assertions. The smoke-test in the CALM README covers the latter.
"""

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.encoders import ENCODER_REGISTRY
from praxis.generation.lf_temperature import (
    lf_temperature_sample_batched,
    lf_temperature_sample_exact,
)
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


def test_lf_temperature_base_case_T1():
    torch.manual_seed(0)

    def sampler(n):
        return torch.randn(n, 4)

    z = lf_temperature_sample_batched(sampler, temperature=1.0, num_candidates=16)
    assert z.shape == (4,)


def test_lf_temperature_low_T_prefers_high_density():
    """At low T, locally-fair sampling should bias toward cluster centres."""
    torch.manual_seed(0)

    def clustered_sampler(n):
        # 80% near origin, 20% far away.
        base = torch.randn(n, 2) * 0.1
        mask = torch.rand(n) < 0.2
        base[mask] += 5.0
        return base

    dists = []
    for _ in range(64):
        z = lf_temperature_sample_batched(
            clustered_sampler, temperature=0.1, num_candidates=32
        )
        dists.append(z.norm().item())
    mean_dist = sum(dists) / len(dists)

    dists_t1 = []
    for _ in range(64):
        z = lf_temperature_sample_batched(
            clustered_sampler, temperature=1.0, num_candidates=32
        )
        dists_t1.append(z.norm().item())
    mean_dist_t1 = sum(dists_t1) / len(dists_t1)

    # Mode-seeking at T<1 should land closer to the dense cluster.
    assert mean_dist < mean_dist_t1


def test_lf_temperature_exact_falls_back_gracefully():
    torch.manual_seed(0)

    def sampler(n):
        return torch.randn(n, 3)

    z = lf_temperature_sample_exact(
        sampler, temperature=0.5, num_candidates=16, max_tries=8
    )
    assert z.shape == (3,)


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
