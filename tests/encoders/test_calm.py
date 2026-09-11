"""CALM encoder + energy generator + LF-temperature sanity tests.

These are shape / plumbing checks rather than training-quality
assertions. The smoke-test in the CALM README covers the latter.
"""

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM, registry


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


def test_calm_forward_backward():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    # Force joint mode so the generator trains from step 0. The default is
    # now an AE pretraining phase, where energy is gated until the codec
    # freezes (see test_calm_pretraining_phase_freezes_on_cap).
    model.encoder.requires_pretraining = False
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss.requires_grad
    out.loss.backward()
    # Generator and VAE both get gradients.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.vae.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.generator.parameters()
    )


def test_calm_generate_advances_in_K_steps():
    """The vote emits a whole K-token patch per trunk forward - it predicts a
    LATENT, and the VAE turns that into K tokens at once."""
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.eval()
    from transformers import GenerationConfig

    K = model.encoder.K
    # A budget that is a whole number of patches, so nothing is trimmed and the
    # K-at-a-time mechanism is what the length reports.
    gc = GenerationConfig(max_new_tokens=2 * K, temperature=1.0, do_sample=True)
    input_ids = torch.randint(4, 200, (1, 8), dtype=torch.long)
    out = model.generate(input_ids, generation_config=gc)
    new = out.size(1) - input_ids.size(1)
    assert new % K == 0
    assert new == 2 * K


def test_calm_generate_honors_an_off_patch_budget():
    """`max_new_tokens` means the same number here as on every other path.

    The patch is still the unit of PREDICTION - the loop cannot vote for half a
    latent - but the caller's budget is a hard bound, so the tail of the last
    patch is trimmed rather than overshot. It has to be: `Generator` sizes the
    prompt so prompt + max_new_tokens fits `max_position_embeddings`, and an
    overshoot past that is exactly the positional overflow that sizing exists
    to prevent. Transformers' own MaxLengthCriteria is what enforces it now.
    """
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.eval()
    from transformers import GenerationConfig

    K = model.encoder.K
    budget = K + 1  # deliberately not a whole number of patches
    input_ids = torch.randint(4, 200, (1, 8), dtype=torch.long)
    out = model.generate(
        input_ids,
        generation_config=GenerationConfig(
            max_new_tokens=budget, temperature=1.0, do_sample=True
        ),
    )
    assert out.size(1) - input_ids.size(1) == budget


def test_calm_generate_aligns_unaligned_prompt():
    # A prompt whose length is not a multiple of K must still generate cleanly:
    # vote_decoding left-pads for alignment (so the conditioning patch stays
    # full of real tokens) then strips the pads, so the returned sequence begins
    # with the verbatim prompt - no pad tokens injected into the output.
    #
    # The pads are also why the halt scan runs against the UNPADDED view: the
    # criteria were sized from the caller's prompt length and know nothing
    # about them, so a budget measured on the padded sequence would come up
    # short by exactly pad_n.
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
    assert new % K == 0 and new == 2 * K


def test_calm_with_crystal_classifier():
    # CALM borrows a ``classifiers`` registry entry as its token classifier. Crystal
    # (which previously refused loss-owning encoders) now sizes to the VAE
    # decoder layout and trains through the reconstruction path.
    cfg = _tiny_config(classifier_type="crystal")
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    out.loss.backward()
    centers = model.classifier.scorer.centers
    assert centers.shape == (model.encoder.output_vocab_size, model.encoder.output_dim)
    assert centers.grad is not None and centers.grad.abs().sum() > 0


def test_calm_harmonic_generator_trains():
    # CALM with generator_type="harmonic": the harmonic generator trains through the
    # shared flow loss path.
    cfg = _tiny_config(encoder_type="calm_byte_harmonic", tokenizer_type="byte_level")
    model = PraxisForCausalLM(cfg)
    model.train()
    model.encoder.requires_pretraining = False  # joint mode: generator trains now
    assert type(model.encoder.generator).__name__ == "HarmonicLatentGenerator"
    ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    out = model(input_ids=ids, labels=ids[:, 1:].contiguous())
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.generator.net.parameters()
    )


def test_fixed_codec_deterministic_drop_in():
    # Fixed codec: deterministic encode (pure buffers), learned decode, zero KL.
    from praxis.encoders.calm.codecs import FixedCodec

    assert registry.lookup("codecs", "fixed") is FixedCodec
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
    # CALM with codec_kind="fixed", single-stage (ae_freeze_steps=0): the
    # learned decoder trains against the stationary fixed latent.
    cfg = _tiny_config(encoder_type="calm_byte_fixed", tokenizer_type="byte_level")
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


def test_harmonic_codec_variants():
    # Harmonic codec: standing-wave bases instead of random orthonormal. Linear
    # variant is deterministic with no learnable encode params; serpent variant
    # adds a learned periodic nonlinearity (encode becomes learnable).
    from praxis.encoders.calm.codecs import (
        HarmonicCodec,
        _harmonic_matrix,
        _separable_harmonic_matrix,
    )

    assert registry.lookup("codecs", "harmonic") is HarmonicCodec
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
    from praxis.encoders.calm.codecs import HybridCodec

    assert registry.lookup("codecs", "hybrid") is HybridCodec
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


def test_calm_two_stage_freezes_codec_and_enables_energy():
    """Legacy two-stage (ae_freeze_steps > 0, no AE pretraining phase): codec
    and LM train jointly in stage 1, then the codec freezes while the energy
    generator takes over in stage 2 (against a stationary target). The default mode
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

    # Stage 2: generator still learns; frozen codec gets no gradient.
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.generator.parameters()
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
    after which the energy generator activates."""
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

    # Phase 2: generator now learns against the frozen codec.
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.generator.parameters()
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


def test_calm_with_stacked_crystal_harmonic_classifier():
    # crystal_harmonic stacks the harmonic field in front of the crystal
    # classifier; both mechanisms train through CALM's reconstruction path.
    cfg = _tiny_config(classifier_type="crystal_harmonic")
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    out.loss.backward()

    # SequentialClassifier([HarmonicClassifier(transform-only), CrystalClassifier])
    classifier = model.classifier
    harmonic, crystal = classifier.stages[0], classifier.stages[1]
    centers = crystal.scorer.centers
    amps = harmonic.field.amplitudes
    assert centers.shape == (
        model.encoder.output_vocab_size,
        model.encoder.output_dim,
    )
    # The transform-only harmonic stage builds no scorer of its own.
    assert harmonic.scorer is None
    # Both mechanisms receive gradient (field modulates features, crystal
    # classifies them - the recon path trains both).
    assert centers.grad is not None and centers.grad.abs().sum() > 0
    assert amps.grad is not None and amps.grad.abs().sum() > 0
    # Both auxiliary losses are exposed and merged.
    aux = classifier.aux_losses()
    assert "centers_rms" in aux and "harmonic_smoothness" in aux


def test_calm_with_prismatic_classifier_learns_envelope():
    # prismatic = ParallelClassifier([Sequential(field+linear),
    # Sequential(field, crystal)]): a top-level gate balances the two arms' logits per token. Both envelopes and
    # the gate train through CALM's reconstruction path.
    cfg = _tiny_config(classifier_type="prismatic")
    model = PraxisForCausalLM(cfg)
    parallel = model.classifier  # ParallelClassifier is the top classifier
    fields = [arm.stages[0].field for arm in parallel.branches]
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


def test_patch_vae_perturbs_both_reference_sites_in_training_only():
    """PatchVAE (AbstractinatorCALM's codec) drops input features in encode and
    the sampled latent in decode, the reference's ae_dropout, so the decoder
    learns to map a NEIGHBOURHOOD of z - the latent the generator predicts at
    generation - to the right features. Eval applies neither."""
    from praxis.encoders.calm.vae import PatchVAE

    torch.manual_seed(0)
    vae = PatchVAE(feature_dim=32, latent_dim=32, hidden_dim=32, dropout=0.15)
    h, z = torch.randn(2, 8, 32), torch.randn(2, 8, 32)
    vae.eval()
    assert torch.equal(vae.encode(h)[0], vae.encode(h)[0])
    assert torch.equal(vae.decode(z), vae.decode(z))
    vae.train()
    assert not torch.equal(vae.encode(h)[0], vae.encode(h)[0])
    assert not torch.equal(vae.decode(z), vae.decode(z))


def test_calm_halo_geometric_mode():
    """loss_func=halo selects CALM's trinary geometric mode: recon stays CE,
    and once the codec freezes the generator trains under the angular HALO +
    radial terms with gradient reaching only the generator (codec/centroids are
    frozen instruments)."""
    from praxis.losses.cross_entropy import CrossEntropyLoss

    cfg = _tiny_config(loss_func="halo", classifier_type="crystal")
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
    # Gradient reaches the generator...
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.generator.parameters()
    )
    # ...but not the frozen codec.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in enc.vae.parameters())


def test_calm_geometric_mode_off_by_default():
    cfg = _tiny_config(classifier_type="crystal")
    enc = PraxisForCausalLM(cfg).encoder
    assert not enc.geometric_mode
    assert not hasattr(enc, "geo_loss_fn")


def test_energy_prior_registry_and_default():
    """linear is the default wherever the energy generator is used; none disables;
    harmonic augments features with the sin/cos basis."""
    from praxis.generators.energy import PRIOR_HARMONIC_FREQS

    assert set(registry.namespace("energy_priors")) == {"none", "linear", "harmonic"}

    enc = PraxisForCausalLM(_tiny_config()).encoder
    assert enc.generator.prior is not None  # default = linear
    assert enc.generator.prior.mode == "linear"

    harm = registry.lookup("energy_priors", "harmonic")(
        feature_dim=8, latent_dim=4, period=16
    )
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
    prior = enc.generator.prior
    assert "calm_prior_r2" in enc._diag
    assert "calm_prior_norm" in enc._diag
    assert prior.W.abs().sum() > 0
    assert not bool(prior.frozen.item())

    for _ in range(4):  # past the window: freezes
        model(input_ids=input_ids, labels=labels)
    assert bool(prior.frozen.item())
