"""The abstractinator encoders (praxis/encoders/abstractinator).

Most of this file covers AbstractinatorCALM: a continuous CALM arm beside the
discrete RVQ arm. The thesis is that the DISCRETE arm can pay for the
CONTINUOUS one. CALM's energy score is a weak, high-variance signal that needs
far more tokens than this line can afford; an RVQ code is a dense,
low-variance, mode-seeking target, and predicting the next code from the same
conditioning hidden the energy generator reads is what should concentrate its
conditional. These tests pin the mechanics, not the thesis - the run decides
that.
"""

import pytest
import torch
from transformers import GenerationConfig

import praxis.encoders.abstractinator.calm as calm
from praxis import PraxisConfig
from praxis.encoders.abstractinator.encoder import AbstractinatorEncoder
from praxis.modeling import PraxisForCausalLM

PROFILE = "abstractinator_v1_calm"
PARENT = "abstractinator_v1"


def _config(encoder=PROFILE, d=64):
    return PraxisConfig(
        vocab_size=1024,
        hidden_size=d,
        embed_size=d,
        num_heads=2,
        depth=2,
        max_length=512,
        decoder_type="sequential",
        classifier_type="forward",
        encoder_type=encoder,
        tokenizer_type="byte_level",
        codebook_size=256,
    )


def _build(encoder=PROFILE, d=64):
    torch.manual_seed(0)
    return PraxisForCausalLM(_config(encoder, d)).train()


def _batch(b=2, t=128):
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (b, t))
    return ids, ids[:, 1:].contiguous()


def _step(m):
    ids, labels = _batch()
    return m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def _registered(m):
    """Run a training step, then re-run both training hooks on a fresh ``h``:
    the step's own registrations were drained into the model's loss."""
    _step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    m.encoder._register_calm_losses(h)
    return h


# ── the parent encoder ─────────────────────────────────────────────────────


def test_encoder_metrics_and_cards_end_to_end():
    """The harmonic-bottleneck stack emits VQ metrics through
    model.encoder.training_metrics() (the DynamicsLogger route), and every
    emitted key has a chart description."""
    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=2,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
    )
    model = PraxisForCausalLM(config).train()
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


# ── the CALM arm's mechanics ───────────────────────────────────────────────


def test_the_discrete_arm_is_untouched():
    """Only the continuous arm is added; the quantizer the parent runs is the
    same one, so a parent -> CALM comparison attributes to the new arm."""
    m, parent = _build(), _build(PARENT)
    assert type(m.encoder.quantizer) is type(parent.encoder.quantizer)
    assert m.encoder.quantizer.analysis.shape == parent.encoder.quantizer.analysis.shape


def test_every_new_component_receives_gradient():
    """Every CALM-side module trains, including the VAE decoder (the half that
    makes the latent mean something) and the learned arm gate. The gate is
    shape [1], not 0-dim: 0-dim parameters break schedule_free's swap."""
    m = _build()
    _step(m).loss.backward()
    enc = m.encoder
    for name, mod in (
        ("vae", enc.vae),
        ("vae.dec_blocks", enc.vae.dec_blocks),
        ("generator", enc.generator),
        ("code_classifiers", enc.code_classifiers),
    ):
        live = [
            p for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert live, f"{name} received no gradient"
    assert enc.arm_gate.shape == (1,)
    assert enc.arm_gate.grad is not None and enc.arm_gate.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_no_grad_training_forward_is_safe():
    """Lazy-module init runs `model.train()` under `no_grad` - the same shape
    that crashed the prismatic9 arm surgery."""
    m = _build()
    ids, labels = _batch()
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert torch.isfinite(out.loss)


def test_the_two_codecs_emit_one_latent_per_patch():
    """Two encoders side by side into one trunk is only well-posed because the
    patching is STATIC: both emit exactly one latent per patch, so z_q + z_c is
    an alignable merge. The continuous latent is unit-RMS by the VAE's own
    contract, so the energy target space is stationary without a correction at
    the loss."""
    m = _build()
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)
    z, _ = enc._post_downsample(h, torch.zeros(()))
    assert z.shape == h.shape
    assert enc._last_code.shape[:2] == h.shape[:2]
    rms = enc._last_code.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_the_reference_constants():
    """Fidelity to github.com/shaochenze/calm. The noise is narrower than the
    latent (reference: noise_size 64 against latent_size 128), since noise as
    wide as the target lets the generator satisfy the score without the
    conditioning; the codec carries the reference's ae_dropout."""
    m = _build()
    assert m.encoder.generator.noise_dim < m.encoder.generator.latent_dim
    assert calm.ENERGY_SAMPLES_N == 8  # config.num_samples
    assert calm.ENERGY_SAMPLES_M == 100  # n_y, hardcoded in energy_score()
    assert calm.FREE_BITS == 0.5  # kl_clamp
    assert calm.VAE_DROPOUT == 0.15  # ae_dropout
    assert m.encoder.vae.dropout_p == calm.VAE_DROPOUT


# ── the vote ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "samples,temperature", [(500, 0.5), (1, 0.5), (16, 0.5), (500, 1.0)]
)
def test_the_vote_codes_each_decoded_proposal(monkeypatch, samples, temperature):
    """Temperature is realized as a draw COUNT (n = round(1/T)), so any sample
    count and T=1 (the reference's no-vote case) must run. Each proposal's
    code is read off its DECODED feature: `_quantize_to_codes` applies the
    analysis rotation, which is defined on patch features, not latents."""
    enc = _build().encoder
    enc.vote_samples = samples
    seen = {}

    def spy(owner, name):
        real = getattr(owner, name)

        def wrapped(x, *args, **kwargs):
            out = real(x, *args, **kwargs)
            seen[name] = (x, out)
            return out

        monkeypatch.setattr(owner, name, wrapped)

    spy(enc.generator, "sample")
    spy(enc.vae, "decode")
    spy(enc, "_quantize_to_codes")

    z = enc.vote_next_latent(torch.randn(3, 64), temperature=temperature)
    assert z.shape == (3, 64) and torch.isfinite(z).all()
    proposals = seen["sample"][1].permute(1, 0, 2).reshape(-1, 64)
    assert torch.equal(seen["decode"][0], proposals)
    assert seen["_quantize_to_codes"][0] is seen["decode"][1]
    assert 0.0 < enc._calm_diag["calm_vote_margin"] <= 1.0
    if samples > 1:
        assert enc._calm_diag["calm_vote_lift"] >= 0.0


def test_vote_uses_codes_as_the_equivalence_classes():
    """The cheap part: CALM decodes every candidate to a K-token patch to
    compare them; the RVQ already defines that partition, so a
    nearest-neighbour lookup replaces a decode."""
    enc = _build().encoder
    z = torch.randn(7, 64)
    codes = enc._quantize_to_codes(z)
    assert codes.shape == (7,)
    assert codes.dtype in (torch.int64, torch.long)
    # Deterministic AND side-effect free. The quantizer's forward MUTATES (EMA
    # updates, replacement buffer, dead-code resets), so voting through it
    # would push 500 candidate latents per generated patch into the live
    # codebook, collapsing distinct latents onto one code.
    assert torch.equal(codes, enc._quantize_to_codes(z))
    before = enc.quantizer.quantizer.stage_codebook(0).clone()
    for _ in range(5):
        enc._quantize_to_codes(torch.randn(64, 64))
    assert torch.equal(before, enc.quantizer.quantizer.stage_codebook(0))


def test_metric_cards_exist_for_every_diagnostic():
    """Every CALM diagnostic and every training metric (the learned balance
    weights included) has a card."""
    m = _build()
    _step(m)
    m.encoder.vote_next_latent(torch.randn(2, 64))
    descs = type(m.encoder).metric_descriptions
    metrics = m.encoder.training_metrics()
    for key in set(m.encoder._calm_diag) | set(metrics):
        assert key in descs, f"{key} has no card"
    assert {"calm_weight_energy", "calm_weight_code_ce", "calm_weight_kl"} <= set(
        metrics
    )


def test_the_winner_is_a_real_proposal_not_an_average():
    """The reference votes in TOKEN space and simply emits the winner - it
    never averages. Averaging inside the winning cell reintroduces the
    conditional-mean estimator the energy score exists to avoid: with a coarse
    codebook most proposals land in one cell and that average IS the global
    mean."""
    enc = _build().encoder
    torch.manual_seed(3)
    h = torch.randn(4, 64)
    # Force a spread-out, non-degenerate proposal cloud.
    with torch.no_grad():
        enc.generator.final_layer.linears[-1].weight.normal_(0, 0.5)
    torch.manual_seed(7)
    z = enc.vote_next_latent(h)

    # The vote draws its proposals before anything else touches the RNG, so
    # re-seeding replays the same pool exactly.
    torch.manual_seed(7)
    with torch.no_grad():
        pool = enc.generator.sample(h, num_samples=enc.vote_samples)
    nearest = (pool.permute(1, 0, 2) - z.unsqueeze(1)).norm(dim=-1).min(dim=1).values
    assert torch.all(nearest < 1e-5), nearest


def test_selection_is_the_weighted_cascade_not_an_argmax():
    """`C(count, n)` weighting IS the temperature. A plain modal argmax is this
    algorithm's T -> 0 limit and deletes the only sampling control there is."""
    pick = calm.AbstractinatorCALM._cascade_pick
    vals = torch.tensor([10, 20, 30])
    counts = torch.tensor([50, 40, 10])

    # n=1: every cell with >=1 vote is eligible, so the rare cell is reachable.
    picks = {int(pick(vals, counts, 1)) for _ in range(400)}
    assert len(picks) > 1, "n=1 must stay stochastic"
    assert 30 in picks, "a low-count cell must remain reachable at n=1"

    # Lowering the temperature (raising n) concentrates on the best-supported.
    hot = [int(pick(vals, counts, 1)) for _ in range(400)]
    cold = [int(pick(vals, counts, 40)) for _ in range(400)]
    assert cold.count(10) / len(cold) > hot.count(10) / len(hot)


def test_cascade_descends_when_no_cell_is_supported_enough():
    """n_initial above every count must fall through rather than fail."""
    vals = torch.tensor([7, 8])
    counts = torch.tensor([2, 1])
    for _ in range(20):
        assert int(calm.AbstractinatorCALM._cascade_pick(vals, counts, 500)) in (7, 8)


# ── the arm ceiling ────────────────────────────────────────────────────────


def test_the_arm_cannot_exceed_the_ceiling():
    """Left as a bare `z = z_q + z_c` with only a 1e-3 KL opposing it, the
    continuous arm grew to 20-35x the discrete one and took the codebook, the
    code CE and `val_byte_nll_bits` with it. The bound is structural."""
    m = _build()
    enc = m.encoder
    # Force the codec wide open - far past anything training could reach.
    with torch.no_grad():
        enc.vae.to_params.weight.normal_(0, 5.0)
        enc.vae.to_params.bias.normal_(0, 5.0)
        enc.arm_gate.fill_(20.0)  # gate saturated at the ceiling
    _step(m)
    assert enc._calm_diag["calm_arm_ratio"] <= calm.ARM_CEILING + 1e-3


def test_the_cap_is_soft_so_the_silent_start_survives():
    """The arm starts silent through the GATE: the cap is soft, so below the
    ceiling z_c passes through untouched rather than being scaled up to it.
    The log-variance is nowhere near the clamp floor, so the KL - which
    penalizes a too-small variance as hard as a too-large one - is not paid to
    keep the arm quiet."""
    m = _build()
    _step(m)
    enc = m.encoder
    assert enc._calm_diag["calm_arm_ratio"] < 0.05
    assert enc._calm_diag["calm_arm_gate"] < 0.01
    _, logvar = enc._last_posterior
    assert float(logvar.detach().mean()) > -4.0


def test_the_cap_does_not_let_the_arm_widen_by_shrinking_z_q(monkeypatch):
    """`z_q` is detached inside the cap. Otherwise the cheapest way to raise
    the allowance is to collapse the discrete arm - the exact failure mode."""
    m = _build()
    enc = m.encoder
    # A small z_q, so the cap binds and the contribution is scaled by it.
    z_q = (0.01 * torch.randn(2, 8, m.config.hidden_size)).requires_grad_()
    monkeypatch.setattr(
        AbstractinatorEncoder, "_post_downsample", lambda self, h, aux: (z_q, aux)
    )
    h = torch.randn(2, 8, m.config.hidden_size)
    z, _ = enc._post_downsample(h, torch.zeros(()))
    assert enc._calm_diag["calm_arm_ratio"] < 1.0  # the cap is binding
    (z - z_q).sum().backward()  # the arm's contribution alone
    assert z_q.grad is None or not z_q.grad.any()


# ── the CALM losses and their balance ──────────────────────────────────────


def test_every_calm_loss_goes_through_the_balance():
    """With the arm capped to 0.25% the run still degraded, so the damage was
    the CALM losses' GRADIENT on the trunk, not the arm's contribution to it:
    every term the training hooks register is weighted by the learned balance.
    The VAE carries its own reconstruction objective (1.0 is predict-zero) and
    the conditioning gap is reported beside the energy."""
    m = _build()
    _registered(m)
    enc = m.encoder
    assert set(enc._pending) == {"calm_kl", "calm_code_ce", "calm_energy", "calm_recon"}
    assert set(enc.loss_balance.log_var) == set(enc._pending)
    assert enc._calm_diag["calm_recon_rel"] > 0.0
    assert "calm_energy_cond_gap" in enc._calm_diag


def test_the_kl_does_not_leak_into_validation():
    """`consume_pending_losses` drains into the shared container on EVERY
    forward, so a KL registered in eval inflates `val_loss` and breaks the
    comparability the experiment rests on. Checked on the hook directly: after
    a full forward `_pending` is empty because it was drained."""
    m = _build().eval()
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    assert m.encoder._pending == {}, m.encoder._pending


def test_code_ce_is_normalized_by_chance():
    """Dividing by ln(K) makes the term dimensionless - 1.0 is chance - so its
    scale stops being an accident of codebook size (unnormalized, an untrained
    code classifier sits at ln 256 = 5.5)."""
    m = _build()
    _registered(m)
    ce = m.encoder._pending.get("calm_code_ce")
    assert ce is not None
    assert float(ce) < 2.0


def test_the_balance_starts_neutral():
    """Weight exactly 1 at step 0, so the run is an A/B and not a reroll."""
    for w in _build().encoder.loss_balance.weights().values():
        assert w == pytest.approx(1.0)


def test_the_kl_is_not_a_thousand_nats_at_init():
    """Driving the log-variance bias to the floor to silence the arm MAXIMIZES
    the KL (3.50 nats per dimension), and summing over the width turns that
    into hundreds of nats. The KL is small at init and does not grow with
    model width."""
    enc = _build().encoder
    enc._post_downsample(torch.randn(2, 8, 64), torch.zeros(()))
    kl = float(enc._pending["calm_kl"])
    assert kl < 2.0, kl
    wide = _build(d=256).encoder
    wide._post_downsample(torch.randn(2, 8, 256), torch.zeros(()))
    assert float(wide._pending["calm_kl"]) == pytest.approx(kl, rel=0.1)


def test_training_draws_go_through_the_generators_sampler(monkeypatch):
    """The energy score's model draws come from `EnergyGenerator.sample`, the
    reference's uniform [-0.5, 0.5] noise, at the reference's N."""
    m = _build()
    enc = m.encoder
    _step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))
    calls = []
    real = enc.generator.sample

    def sample(h_cond, num_samples, **kwargs):
        calls.append(num_samples)
        return real(h_cond, num_samples, **kwargs)

    monkeypatch.setattr(enc.generator, "sample", sample)
    enc._register_calm_losses(h)
    assert calls == [calm.ENERGY_SAMPLES_N]


def test_the_energy_target_is_centred_on_the_mean_not_a_draw(monkeypatch):
    """`z_c` already carries one posterior draw, so centring the target cloud
    on it applies the noise twice, around a centre that moves every step. The
    target cloud sits on the normalized posterior MEAN of the next patch."""
    m = _build()
    _step(m)
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)

    # In eval the codec emits its mean, so the two coincide.
    enc.eval()
    enc._post_downsample(h, torch.zeros(()))
    assert torch.allclose(enc._last_code, enc._last_code_mean, atol=1e-5)

    # In training the stashed code is a draw, and the target ignores it.
    enc.train()
    enc._post_downsample(h, torch.zeros(()))
    assert not torch.allclose(enc._last_code, enc._last_code_mean, atol=1e-5)
    mu, logvar = enc._last_posterior
    assert torch.allclose(enc._last_code_mean, enc.vae.normalize_latent(mu))

    # Collapse the posterior spread so every target draw IS its centre.
    enc._last_posterior = (mu, torch.full_like(logvar, -40.0))
    targets = []
    real = calm.energy_score_loss

    def capture(proposals, target, *args, **kwargs):
        targets.append(target)
        return real(proposals, target, *args, **kwargs)

    monkeypatch.setattr(calm, "energy_score_loss", capture)
    enc._register_calm_losses(h)
    centre = enc._last_code_mean[:, 1:].unsqueeze(2).expand_as(targets[0])
    assert torch.allclose(targets[0], centre, atol=1e-6)


def test_the_energy_loss_ignores_the_quantized_latent():
    """The energy target is the VAE latent, not the trunk input `z_q + z_c`:
    blowing up the quantized half cannot move the energy term at all."""
    m = _build()
    enc = m.encoder
    _step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))

    torch.manual_seed(5)
    enc._register_calm_losses(h)
    e1 = float(enc._pending["calm_energy"])
    enc._last_latent = enc._last_latent * 1000.0
    torch.manual_seed(5)
    enc._register_calm_losses(h)
    assert float(enc._pending["calm_energy"]) == e1


# ── generation: the vote, selectable at RUNTIME ────────────────────────────


def _gen(model, n=16, b=1):
    torch.manual_seed(2)
    return model.generate(
        torch.randint(0, 256, (b, 32)),
        generation_config=GenerationConfig(max_new_tokens=n, do_sample=False),
    )


def _build_mode(mode):
    c = _config()
    c.generation_mode = mode
    torch.manual_seed(0)
    return PraxisForCausalLM(c).eval()


def test_default_is_the_byte_loop_and_the_vote_is_off():
    """The default mode declares no decoding method, so generation runs the
    byte-level loop."""
    m = _build().eval()
    assert m.encoder.generation_mode == "standard"
    assert m.encoder.decoding_method() is None
    assert _gen(m).shape == (1, 48)


def test_vote_mode_owns_generation():
    m = _build_mode("vote")
    assert m.encoder.generation_mode == "vote"
    assert _gen(m).shape == (1, 48)
    # The vote's diagnostics only exist if the vote actually ran.
    assert "calm_vote_margin" in m.encoder._calm_diag
    assert "calm_vote_lift" in m.encoder._calm_diag


def test_both_modes_are_the_same_weights():
    """Runtime switch, not a different architecture: one checkpoint reads out
    either way, which is the only way to attribute a difference to the decoder."""
    a, b = _build().encoder, _build_mode("vote").encoder
    assert {n: tuple(p.shape) for n, p in a.named_parameters()} == {
        n: tuple(p.shape) for n, p in b.named_parameters()
    }


def test_vote_generation_never_mutates_the_codebook():
    """`_trunk_input` calls the quantizer's forward, which runs EMA updates and
    dead-code resets - both gated on `self.training`. No stage's codebook may
    move while the vote generates."""
    m = _build_mode("vote")
    core = getattr(m.encoder.quantizer, "quantizer", m.encoder.quantizer)
    stages = range(m.encoder.vq_depth)
    before = [core.stage_codebook(s).clone() for s in stages]
    _gen(m, n=8)
    for s in stages:
        assert torch.equal(before[s], core.stage_codebook(s)), s


def test_vote_generation_requires_the_latent_and_logits_seams():
    """A patch the model just PREDICTED has no bytes behind it, so
    `base_forward` cannot reach it; and the classifier belongs to the model, not the
    encoder. Both arrive per call - derived from `model` via `trunk_hooks`, or
    supplied explicitly by a non-standard driver - rather than as stored
    back-references, so the loop cannot run with neither."""
    m = _build_mode("vote")
    with pytest.raises(ValueError, match="trunk"):
        m.encoder.vote_decoding(None, torch.randint(0, 256, (1, 8)))
    # The encoder still names the loop; the seams are what it lacks.
    assert m.encoder.decoding_method().__func__ is type(m.encoder).vote_decoding


def test_vote_generation_honors_return_dict_in_generate():
    """`return_dict_in_generate` arrives as a KWARG on model.generate (see
    DecodeBackend), not on the config, and the caller then reads `.sequences`."""
    m = _build_mode("vote")
    ids = torch.randint(0, 256, (1, 32))
    gc = GenerationConfig(max_new_tokens=16, do_sample=False)
    assert isinstance(m.generate(ids, generation_config=gc), torch.Tensor)
    out = m.generate(ids, generation_config=gc, return_dict_in_generate=True)
    assert hasattr(out, "sequences") and out.sequences.shape == (1, 48)


def test_vote_generation_survives_a_stop_string_format(monkeypatch):
    """Under `prose` (stop strings only) transformers builds StopStringCriteria
    from the tokenizer, so the tokenizer has to survive the hand-off to the
    vote's decoding method. See PraxisForCausalLM._extract_generation_mode_kwargs."""
    from transformers.generation.stopping_criteria import StopStringCriteria

    from praxis.tokenizers import create_tokenizer
    from praxis.tokenizers.chat_templates import chat_format_of

    tokenizer = create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format="prose"
    )
    stops = list(chat_format_of(tokenizer).stop_strings())

    m = _build_mode("vote")
    seen = {}

    def capture(self, model, input_ids=None, **kwargs):
        seen.update(kwargs)
        return input_ids

    monkeypatch.setattr(type(m.encoder), "vote_decoding", capture)
    m.generate(
        torch.randint(0, 256, (1, 32)),
        generation_config=GenerationConfig(
            max_new_tokens=16, do_sample=False, stop_strings=stops
        ),
        tokenizer=tokenizer,
    )
    assert seen["tokenizer"] is tokenizer
    assert any(isinstance(c, StopStringCriteria) for c in seen["stopping_criteria"])


def test_vote_generation_honors_the_deadline():
    """Queued generations decode INSIDE the training loop, so a loop that
    ignores the caller's stopping criteria stalls the run. transformers never
    runs the criteria list for a loop it does not own."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Halt(StoppingCriteria):
        def __call__(self, seq, scores, **kwargs):
            return torch.ones(seq.shape[0], dtype=torch.bool)

    m = _build_mode("vote")
    out = m.generate(
        torch.randint(0, 256, (1, 32)),
        generation_config=GenerationConfig(max_new_tokens=256, do_sample=False),
        stopping_criteria=StoppingCriteriaList([Halt()]),
    )
    assert out.shape[1] < 32 + 256, out.shape


def test_vote_generation_actually_samples():
    """BrierLM's estimator is `1{a=y} + 1{b=y} - 1{a=b}` over two i.i.d.
    samples. Two byte-identical draws pin the self-match term at 1, force
    every order non-positive and floor the metric at exactly 0, whatever the
    model has learned. Greedy decoding stays deterministic."""
    from praxis.metrics.brier import compute_brier_lm_with_orders

    m = _build_mode("vote")
    ids = torch.randint(0, 256, (1, 32))
    sampled = GenerationConfig(max_new_tokens=16, do_sample=True, temperature=1.0)
    a = m.generate(ids, generation_config=sampled)
    b = m.generate(ids, generation_config=sampled)
    assert not torch.equal(a, b), "two sampled draws must differ"

    # An untrained model matches nothing, but it must not ANTI-match: a -1
    # order is the a == b signature, not a statement about the model.
    ref = torch.randint(0, 256, (16,)).tolist()
    _, per = compute_brier_lm_with_orders(
        [a[0, 32:].tolist()], [b[0, 32:].tolist()], [ref]
    )
    for n, v in per.items():
        if v is not None:
            assert v > -1.0 + 1e-9, (n, v)

    greedy = GenerationConfig(max_new_tokens=16, do_sample=False)
    assert torch.equal(
        m.generate(ids, generation_config=greedy),
        m.generate(ids, generation_config=greedy),
    ), "greedy must stay deterministic"


def test_the_vote_temperature_is_not_the_sampler_temperature(monkeypatch):
    """The vote's temperature is a draw COUNT (n = round(1/T)); the sampler's
    is a logit scale. It is read from `calm_vote_temperature` or the encoder's
    default, never from `generation_config.temperature`, so a caller asking for
    hotter text cannot silently disable the vote's cascade."""
    m = _build_mode("vote")
    enc = m.encoder
    seen = []
    vote = enc.vote_next_latent

    def record(h_cond, temperature):
        seen.append(temperature)
        return vote(h_cond, temperature=temperature)

    monkeypatch.setattr(enc, "vote_next_latent", record)
    ids = torch.randint(0, 256, (1, 32))
    gc = GenerationConfig(max_new_tokens=8, do_sample=True, temperature=0.1)
    m.generate(ids, generation_config=gc)
    assert seen == [enc.vote_temperature]

    seen.clear()
    gc.calm_vote_temperature = 1.0
    m.generate(ids, generation_config=gc)
    assert seen == [1.0]


# ── the next-code objective (abstractinator_v3) ───────────────────────────


def _next_code(m):
    """Quantize a fresh ``h`` and register the objective on it, as decode does."""
    torch.manual_seed(2)
    h = torch.randn(2, 8, m.config.hidden_size, requires_grad=True)
    m.encoder._post_downsample(h, torch.zeros(()))
    m.encoder._register_next_code(h)
    return h, m.encoder.consume_pending_losses()


def test_next_code_trains_the_trunk_and_nothing_at_eval():
    m = _build("abstractinator_v3")
    out = _step(m)
    assert torch.isfinite(out.loss)
    h, pending = _next_code(m)
    loss = pending["next_code_halo"]
    (grad,) = torch.autograd.grad(loss, h, allow_unused=True)
    assert grad is not None and float(grad.abs().sum()) > 0
    # Position p predicts patch p+1's code, so the last position has no target.
    assert float(grad[:, -1].abs().sum()) == 0.0
    m.eval()
    with torch.no_grad():
        _step(m)
    assert m.encoder.consume_pending_losses() == {}


def test_next_code_is_divided_by_log_k():
    """The term's size must not grow with the codebook: at init it sits at a
    few units of log K, not at tens of nats."""
    m = _build("abstractinator_v3")
    _, pending = _next_code(m)
    assert 0.5 < float(pending["next_code_halo"]) < 5.0


def test_next_code_is_off_on_the_parent():
    m = _build(PARENT)
    assert m.encoder.next_code is None
    _step(m)
    assert "next_code_acc" not in m.encoder.training_metrics()
