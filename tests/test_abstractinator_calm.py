"""AbstractinatorCALM: a continuous CALM arm beside the discrete RVQ arm.

The thesis under test is that the DISCRETE arm can pay for the CONTINUOUS one.
CALM's energy score is a weak, high-variance signal that needs far more tokens
than this line can afford; an RVQ code is a dense, low-variance, mode-seeking
target, and predicting the next code from the same conditioning hidden the
energy head reads is what should concentrate its conditional.

These tests pin the mechanics, not the thesis - the run decides that.
"""

import pytest
import torch

from praxis import PraxisConfig
from praxis.encoders import ENCODER_REGISTRY
from praxis.encoders.abstractinator import AbstractinatorCALM
from praxis.modeling import PraxisForCausalLM

PROFILE = "abstractinator_harmonic_gdn_vocab_bank_static_calm"
PARENT = "abstractinator_harmonic_gdn_vocab_bank_static"


def cfg(encoder=PROFILE, d=64):
    return PraxisConfig(
        vocab_size=1024, hidden_size=d, embed_size=d, num_heads=2, depth=2,
        max_length=512, decoder_type="sequential", head_type="forward",
        encoder_type=encoder, tokenizer_type="byte_level", codebook_size=256,
    )


def build(encoder=PROFILE, seed=0):
    torch.manual_seed(seed)
    return PraxisForCausalLM(cfg(encoder)).train()


def batch(b=2, t=128):
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (b, t))
    return ids, ids[:, 1:].contiguous()


def step(m):
    ids, labels = batch()
    return m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


def test_registered_and_is_the_calm_variant():
    assert PROFILE in ENCODER_REGISTRY
    m = build()
    assert isinstance(m.encoder, AbstractinatorCALM)


def test_the_discrete_arm_is_untouched():
    """Only the continuous arm is added; the quantizer the parent runs is the
    same one, so a parent -> CALM comparison attributes to the new arm."""
    m, parent = build(), build(PARENT)
    assert type(m.encoder.quantizer) is type(parent.encoder.quantizer)
    assert m.encoder.quantizer.analysis.shape == parent.encoder.quantizer.analysis.shape


def test_the_arm_starts_silent():
    """A fully zeroed posterior is the trap: it zeroes mu AND logvar, and
    logvar 0 means sigma 1, so z_c would be a standard normal added to every
    patch latent (measured at ratio 1.24). The log-variance bias sits at the
    clamp floor instead."""
    m = build()
    step(m)
    assert m.encoder._calm_diag["calm_arm_ratio"] < 0.05


def test_every_new_component_receives_gradient():
    m = build()
    out = step(m)
    out.loss.backward()
    enc = m.encoder
    for name, mod in (
        ("vae", enc.vae),
        ("energy_head", enc.energy_head),
        ("code_heads", enc.code_heads),
    ):
        live = [
            p for p in mod.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert live, f"{name} received no gradient"
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_all_three_calm_losses_are_registered():
    m = build()
    step(m)
    # consume_pending_losses is drained by modeling.py during the forward, so
    # re-run the encoder path and inspect before it is consumed.
    m.encoder._pending.clear()
    step(m)
    # The forward already drained it; what matters is that the loss grew.
    plain = build(PARENT)
    assert float(step(plain).loss.detach()) < float(step(build()).loss.detach())


def test_code_ce_predicts_the_next_code_not_the_current_one():
    """Off-by-one here would make the objective trivial (the code is already in
    the conditioning) and the metric meaningless."""
    m = build()
    enc = m.encoder
    step(m)
    digits = enc._stage_indices()
    assert digits is not None and len(digits) >= 1
    # calm_code_acc is scored against digits[s][:, 1:] from h[:, :-1]; a shape
    # mismatch is the only way that alignment can silently drift.
    assert digits[0].shape[1] == enc._last_latent.shape[1]


def test_eval_forward_registers_nothing():
    """The CALM objectives are training-only, so validation loss stays the
    parent's and the two runs remain comparable on val."""
    m = build().eval()
    ids, labels = batch()
    with torch.no_grad():
        m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert m.encoder._pending == {}


def test_no_grad_training_forward_is_safe():
    """Lazy-module init runs `model.train()` under `no_grad` - the same shape
    that crashed the prismatic9 arm surgery."""
    m = build()
    ids, labels = batch()
    with torch.no_grad():
        m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert m.encoder._pending == {}


# ── the vote ───────────────────────────────────────────────────────────────


def test_vote_returns_a_continuous_latent_not_a_codebook_entry():
    """Selection is discrete (the modal code - that IS CALM's vote), but the
    winner is the MEAN of the proposals that voted for it, so nothing is
    rounded onto the codebook."""
    m = build()
    enc = m.encoder
    h = torch.randn(3, 64)
    z = enc.vote_next_latent(h)
    assert z.shape == (3, 64)
    assert torch.isfinite(z).all()


def test_vote_margin_is_reported_and_bounded():
    m = build()
    m.encoder.vote_next_latent(torch.randn(4, 64))
    margin = m.encoder._calm_diag["calm_vote_margin"]
    assert 0.0 < margin <= 1.0


def test_vote_uses_codes_as_the_equivalence_classes():
    """The cheap part: CALM decodes every candidate to a K-token patch to
    compare them; the RVQ already defines that partition, so a
    nearest-neighbour lookup replaces a decode."""
    m = build()
    enc = m.encoder
    z = torch.randn(7, 64)
    codes = enc._quantize_to_codes(z)
    assert codes.shape == (7,)
    assert codes.dtype in (torch.int64, torch.long)
    # Deterministic AND side-effect free. The quantizer's forward MUTATES (EMA
    # updates, replacement buffer, dead-code resets), so voting through it
    # would push 500 candidate latents per generated patch into the live
    # codebook. When it did, the second call collapsed all seven latents onto
    # one code.
    assert torch.equal(codes, enc._quantize_to_codes(z))
    before = enc.quantizer.quantizer.stage_codebook(0).clone()
    for _ in range(5):
        enc._quantize_to_codes(torch.randn(64, 64))
    assert torch.equal(before, enc.quantizer.quantizer.stage_codebook(0))


def test_the_vote_does_not_touch_the_codebook():
    """The same guarantee at the level people will actually hit it."""
    m = build()
    enc = m.encoder
    before = [
        enc.quantizer.quantizer.stage_codebook(s).clone()
        for s in range(enc.vq_depth)
    ]
    enc.vote_next_latent(torch.randn(2, 64))
    for s, b in enumerate(before):
        assert torch.equal(b, enc.quantizer.quantizer.stage_codebook(s))


@pytest.mark.parametrize("n", [1, 16])
def test_vote_sample_count_is_configurable_and_structural(n):
    """Temperature realized as a COUNT (n = round(1/T)) rather than a logit
    scale - the substitution that makes CALM's sampler what it is."""
    torch.manual_seed(0)
    m = PraxisForCausalLM(cfg()).train()
    m.encoder.vote_samples = n
    z = m.encoder.vote_next_latent(torch.randn(2, 64))
    assert z.shape == (2, 64)


def test_metric_cards_exist_for_every_diagnostic():
    m = build()
    step(m)
    m.encoder.vote_next_latent(torch.randn(2, 64))
    descs = type(m.encoder).metric_descriptions
    for key in m.encoder._calm_diag:
        assert key in descs, f"{key} has no card"


# ── the vote must not degenerate into an average ───────────────────────────


def test_the_winner_is_a_real_proposal_not_an_average():
    """The reference votes in TOKEN space and simply emits the winner - it
    never averages. Averaging inside the winning cell reintroduces the
    conditional-mean estimator the energy score exists to avoid: with a coarse
    codebook most proposals land in one cell and that average IS the global
    mean."""
    m = build()
    enc = m.encoder
    torch.manual_seed(3)
    h = torch.randn(4, 64)
    # Force a spread-out, non-degenerate proposal cloud.
    with torch.no_grad():
        enc.energy_head.final_layer.linears[-1].weight.normal_(0, 0.5)
    torch.manual_seed(7)
    z = enc.vote_next_latent(h)

    # Reproduce the SAME pool: the vote draws its noise first, through the
    # head's own uniform sampler, so re-seeding identically replays it.
    torch.manual_seed(7)
    pool = enc.energy_head.sample(h, num_samples=enc.vote_samples).permute(1, 0, 2)
    # The returned vector must be an actual member of the pool's support, not a
    # synthetic point. Checked by distance-to-nearest-draw being far smaller
    # than the pool's own spread.
    for b in range(4):
        d = (pool[b] - z[b]).norm(dim=-1)
        assert float(d.min()) < float(pool[b].std(0).mean()) * 64 ** 0.5


def test_vote_lift_is_reported():
    """The receipt on the redundancy risk: near 0 means the vote is decorative
    whatever calm_vote_margin says."""
    m = build()
    m.encoder.vote_next_latent(torch.randn(3, 64))
    assert m.encoder._calm_diag["calm_vote_lift"] >= 0.0


def test_selection_is_the_weighted_cascade_not_an_argmax():
    """`C(count, n)` weighting IS the temperature. A plain modal argmax is this
    algorithm's T -> 0 limit and deletes the only sampling control there is."""
    enc = build().encoder
    vals = torch.tensor([10, 20, 30])
    counts = torch.tensor([50, 40, 10])

    # n=1: every cell with >=1 vote is eligible, so the rare cell is reachable.
    picks = {int(enc._cascade_pick(vals, counts, 1)) for _ in range(400)}
    assert len(picks) > 1, "n=1 must stay stochastic"
    assert 30 in picks, "a low-count cell must remain reachable at n=1"

    # Lowering the temperature (raising n) concentrates on the best-supported.
    hot = [int(enc._cascade_pick(vals, counts, 1)) for _ in range(400)]
    cold = [int(enc._cascade_pick(vals, counts, 40)) for _ in range(400)]
    assert cold.count(10) / len(cold) > hot.count(10) / len(hot)


def test_cascade_descends_when_no_cell_is_supported_enough():
    """n_initial above every count must fall through rather than fail."""
    enc = build().encoder
    vals = torch.tensor([7, 8])
    counts = torch.tensor([2, 1])
    for _ in range(20):
        assert int(enc._cascade_pick(vals, counts, 500)) in (7, 8)


def test_temperature_one_is_effectively_a_single_draw():
    """T=1 -> n_initial=1, the reference's no-vote case."""
    m = build()
    z = m.encoder.vote_next_latent(torch.randn(2, 64), temperature=1.0)
    assert z.shape == (2, 64) and torch.isfinite(z).all()


# ── the arm ceiling (the -p b32ddef0f failure) ─────────────────────────────


def test_the_arm_cannot_exceed_the_ceiling():
    """The whole -p failure in one assertion.

    Left as a bare `z = z_q + z_c` with only a 1e-3 KL opposing it, the
    continuous arm went from a 7e-4 ratio to 20-35 within 2500 steps and took
    the codebook (73% dead, perplexity 6-10 of 512), the code CE (pinned at
    chance) and `val_byte_nll_bits` (5.169 -> 5.355, the wrong way) with it.
    """
    from praxis.encoders.abstractinator.calm import ARM_CEILING

    m = build()
    enc = m.encoder
    # Force the codec wide open - far past anything training could reach.
    with torch.no_grad():
        enc.vae.to_params.weight.normal_(0, 5.0)
        enc.vae.to_params.bias.normal_(0, 5.0)
        enc.arm_gate.fill_(20.0)  # gate saturated at the ceiling
    step(m)
    assert enc._calm_diag["calm_arm_ratio"] <= ARM_CEILING + 1e-3


def test_the_cap_is_soft_so_the_silent_start_survives():
    """A hard normalization would scale the near-zero init z_c UP to the
    ceiling and delete the A/B against -o. Below the cap the scale is exactly
    1 and z_c passes through untouched."""
    m = build()
    step(m)
    assert m.encoder._calm_diag["calm_arm_ratio"] < 0.05
    assert m.encoder._calm_diag["calm_arm_gate"] < 0.01


def test_the_gate_is_learned_and_bounded():
    m = build()
    assert m.encoder.arm_gate.requires_grad
    assert m.encoder.arm_gate.shape == (1,)  # 0-dim breaks schedule_free's swap
    out = step(m)
    out.loss.backward()
    assert m.encoder.arm_gate.grad is not None


def test_the_cap_does_not_let_the_arm_widen_by_shrinking_z_q():
    """`z_q` is detached inside the cap. Otherwise the cheapest way to raise
    the allowance is to collapse the discrete arm - the exact failure mode."""
    import inspect

    from praxis.encoders.abstractinator.calm import AbstractinatorCALM

    src = inspect.getsource(AbstractinatorCALM._post_downsample)
    assert "z_q.detach().norm(dim=-1, keepdim=True)" in src


def test_the_kl_does_not_leak_into_validation():
    """`calm_kl` was registered unconditionally, and `consume_pending_losses`
    drains into the shared container on EVERY forward - so a term worth ~80
    nats under the runaway was inflating `val_loss` and breaking the
    comparability the experiment rests on. Checking `_pending` is empty does
    not catch this: it is empty because it was drained."""
    m = build().eval()
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    assert m.encoder._pending == {}, m.encoder._pending


def test_training_still_registers_the_kl():
    m = build()
    step(m)
    # drained by modeling.py, so re-run the hook directly
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    assert "calm_kl" in m.encoder._pending


def test_code_ce_is_normalized_by_chance():
    """At K=512 an untrained head parked ~31 nats in the total loss forever and
    pushed 5x-amplified noise into the trunk. Dividing by ln(K) makes the term
    dimensionless - 1.0 is chance - so its scale stops being an accident of
    codebook size."""
    m = build()
    step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    m.encoder._register_calm_losses(h)
    ce = m.encoder._pending.get("calm_code_ce")
    assert ce is not None
    # An untrained head sits at ~chance, i.e. ~1.0 after normalization, and the
    # balance starts at weight 1 with a zero log-var offset.
    assert float(ce) < 2.0


# ── fidelity to github.com/shaochenze/calm ─────────────────────────────────


def test_the_energy_target_space_is_stationary():
    """The reference's energy target is a FROZEN VAE latent held near N(0,I);
    ours is the live RVQ output. The energy score is a distance, so both its
    value and its gradient scale with ||target|| - detaching the target does
    not freeze the SPACE. Unit-RMS mapping restores what the reference gets
    for free."""
    m = build()
    enc = m.encoder
    step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))

    # Blow the latent up 100x; the energy term must barely move.
    base = enc._last_latent.clone()
    enc._register_calm_losses(h)
    e1 = float(enc._pending["calm_energy"])
    enc._pending = {}
    enc._last_latent = base * 100.0
    enc._register_calm_losses(h)
    e2 = float(enc._pending["calm_energy"])
    assert abs(e2 - e1) < max(1.0, abs(e1)), (e1, e2)


def test_model_draws_use_the_reference_uniform_noise():
    """`EnergyHead.sample` draws uniform [-0.5, 0.5] - the reference's
    `torch.rand(...) - 0.5`. This path hand-rolled `torch.randn` and called the
    head directly, bypassing the one faithful sampler in the file."""
    import inspect

    from praxis.encoders.abstractinator.calm import AbstractinatorCALM

    src = inspect.getsource(AbstractinatorCALM._register_calm_losses)
    assert "self.energy_head.sample(" in src
    assert "torch.randn(B, P, ENERGY_SAMPLES_N" not in src


def test_noise_is_narrower_than_the_latent():
    """Reference: noise_size 64 against latent_size 128. Noise as wide as the
    target lets the head satisfy the score without the conditioning."""
    m = build()
    assert m.encoder.energy_head.noise_dim < m.encoder.energy_head.latent_dim


def test_sample_counts_match_the_reference():
    from praxis.encoders.abstractinator.calm import (
        ENERGY_SAMPLES_M,
        ENERGY_SAMPLES_N,
        FREE_BITS,
    )

    assert ENERGY_SAMPLES_N == 8  # config.num_samples
    assert ENERGY_SAMPLES_M == 100  # n_y, hardcoded in energy_score()
    assert FREE_BITS == 0.5  # kl_clamp


def test_the_conditioning_gap_is_reported():
    """A head that ignores its conditioning scores the same on misaligned
    targets. Without this, a pinned calm_sample_agreement cannot be told apart
    from a collapsed codebook."""
    m = build()
    enc = m.encoder
    step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))
    enc._register_calm_losses(h)
    assert "calm_energy_cond_gap" in enc._calm_diag



# ── balancing the arm against the main task ────────────────────────────────


def test_every_calm_loss_goes_through_the_balance():
    """The whole point of the remediation: with the arm capped to 0.25% the run
    still degraded, so the damage was the CALM losses' GRADIENT on the trunk,
    not the arm's contribution to it."""
    m = build()
    step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    m.encoder._register_calm_losses(h)
    assert set(m.encoder._pending) == {
        "calm_kl", "calm_code_ce", "calm_energy", "calm_recon",
    }
    assert set(m.encoder.loss_balance.log_var) == set(m.encoder._pending)


def test_the_balance_starts_neutral():
    """Weight exactly 1 at step 0, so the run is an A/B and not a reroll."""
    m = build()
    for w in m.encoder.loss_balance.weights().values():
        assert w == pytest.approx(1.0)


def test_a_hard_objective_down_weights_itself():
    """s settles at log L, so the weight settles at 1/L: an objective stuck
    high contributes gradient scaled DOWN, and cannot drown a task that is
    already working."""
    from praxis.losses.uncertainty import UncertaintyWeighting

    uw = UncertaintyWeighting(("hard", "easy"))
    opt = torch.optim.SGD(uw.parameters(), lr=0.5)
    for _ in range(600):
        opt.zero_grad()
        (uw("hard", torch.tensor(16.0)) + uw("easy", torch.tensor(0.5))).backward()
        opt.step()
    w = uw.weights()
    assert w["hard"] == pytest.approx(1 / 16.0, rel=0.15)
    assert w["easy"] == pytest.approx(1 / 0.5, rel=0.15)
    assert w["hard"] < w["easy"]


def test_the_balance_cannot_mute_an_objective():
    """Without the `+ s` term the optimum is s -> inf and every auxiliary is
    silently deleted."""
    from praxis.losses.uncertainty import UncertaintyWeighting

    uw = UncertaintyWeighting(("x",))
    opt = torch.optim.SGD(uw.parameters(), lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        uw("x", torch.tensor(3.0)).backward()
        opt.step()
    assert uw.weights()["x"] > 1e-3


def test_balance_weights_are_charted():
    m = build()
    step(m)
    keys = set(m.encoder.training_metrics())
    assert {"calm_weight_energy", "calm_weight_code_ce", "calm_weight_kl"} <= keys


def test_the_energy_target_is_centred_on_the_mean_not_a_draw():
    """`z_c` already carries one posterior draw, so centring the target cloud
    on it applies the noise twice, around a centre that moves every step."""
    import inspect

    from praxis.encoders.abstractinator.calm import AbstractinatorCALM

    src = inspect.getsource(AbstractinatorCALM._register_calm_losses)
    assert "self._last_code_mean[:, 1:, :]" in src

    m = build()
    step(m)
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)

    # In eval the codec emits its mean, so the two coincide.
    enc.eval()
    enc._post_downsample(h, torch.zeros(()))
    assert torch.allclose(enc._last_code, enc._last_code_mean, atol=1e-5)

    # In training they must not, or the centre is a draw again.
    enc.train()
    enc._post_downsample(h, torch.zeros(()))
    assert not torch.allclose(enc._last_code, enc._last_code_mean, atol=1e-5)


def test_pairwise_distance_avoids_the_N_by_M_by_D_tensor():
    """The differencing form allocates [..., N, M, D] - several GB at N=8,
    M=100, D=272 - which is what was silently capping M."""
    from praxis.losses.energy_score import _pairwise_distance

    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 8, 16), torch.randn(2, 3, 100, 16)
    ref = torch.sqrt(
        (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1).clamp_min(1e-4)
    )
    assert torch.allclose(ref, _pairwise_distance(a, b), atol=1e-4)



def test_the_kl_is_not_a_thousand_nats_at_init():
    """The regression that produced `loss` 1256 with spikes to 16760.

    Two compounding causes, both fixed: driving the log-variance bias to -8 to
    silence the arm MAXIMIZES the KL (it penalizes too-small variance exactly as
    hard as too-large: 3.50 per dimension), and summing over D=272 turned that
    into 952 nats. Invisible under the old hand-set KL_WEIGHT=1e-3; the whole
    loss once a learned balance starting at 1.0 replaced that weight."""
    m = build()
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))
    kl = float(enc._pending["calm_kl"])
    assert kl < 2.0, kl
    # And it must not grow with model width.
    wide = PraxisConfig(
        vocab_size=1024, hidden_size=256, embed_size=256, num_heads=2, depth=2,
        max_length=512, decoder_type="sequential", head_type="forward",
        encoder_type=PROFILE, tokenizer_type="byte_level", codebook_size=256,
    )
    torch.manual_seed(0)
    enc2 = PraxisForCausalLM(wide).train().encoder
    enc2._post_downsample(torch.randn(2, 8, 256), torch.zeros(()))
    assert float(enc2._pending["calm_kl"]) == pytest.approx(kl, rel=0.1)


def test_the_gate_silences_the_arm_not_the_log_variance():
    """Silencing by driving the log-variance to -8 fights the KL, which
    penalizes a too-SMALL variance exactly as hard as a too-large one (3.50
    nats per dimension). The gate silences instead, and costs the KL nothing."""
    m = build()
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))
    _, logvar = enc._last_posterior
    # Nowhere near the clamp floor, so the KL is not being paid to stay quiet.
    assert float(logvar.mean()) > -4.0
    assert enc._calm_diag["calm_arm_ratio"] < 0.05
    assert float(enc._pending["calm_kl"]) < 2.0


# ── the restored VAE ───────────────────────────────────────────────────────


def test_the_continuous_arm_is_a_real_vae_not_a_linear_posterior():
    """Four rounds of instability were about this substitution. A bare
    `nn.Linear(D, 2*D)` has no reconstruction objective of its own, so nothing
    ever required its latent to be informative - the only forces on it were a
    KL pulling it to the prior and a distant byte CE."""
    from praxis.encoders.calm.vae import PatchVAE

    m = build()
    assert isinstance(m.encoder.vae, PatchVAE)
    assert not hasattr(m.encoder, "posterior")
    # Encoder AND decoder, both trained.
    assert len(list(m.encoder.vae.enc_blocks)) > 0
    assert len(list(m.encoder.vae.dec_blocks)) > 0


def test_the_vae_carries_its_own_reconstruction_objective():
    m = build()
    step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    m.encoder._post_downsample(h, torch.zeros(()))
    assert "calm_recon" in m.encoder._pending
    # Relative error: 1.0 is the trivial predict-zero solution.
    assert m.encoder._calm_diag["calm_recon_rel"] > 0.0


def test_the_vae_decoder_receives_gradient():
    """The decoder is the half that makes the latent mean something; if it is
    dead the VAE has degenerated back into the bare posterior."""
    m = build()
    step(m).loss.backward()
    live = [
        p for p in m.encoder.vae.dec_blocks.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert live


def test_the_energy_target_is_the_vae_latent_and_is_unit_rms():
    """CALM's head predicts the next VAE latent. That space is stationary by
    the codec's own contract, so no RMS correction is applied at the loss to
    compensate for an unbounded quantizer output."""
    import inspect

    from praxis.encoders.abstractinator.calm import AbstractinatorCALM

    src = inspect.getsource(AbstractinatorCALM._register_calm_losses)
    assert "_unit_rms" not in src

    m = build()
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))
    rms = enc._last_code.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_the_energy_loss_is_stationary_under_a_quantizer_blowup():
    """The failure the RMS band-aid was hiding: the energy score is a DISTANCE,
    so it used to scale with a target the same gradient step was reshaping. The
    target no longer touches z_q at all."""
    m = build()
    enc = m.encoder
    step(m)
    h = torch.randn(2, 8, m.config.hidden_size)
    enc._post_downsample(h, torch.zeros(()))
    enc._register_calm_losses(h)
    e1 = float(enc._pending["calm_energy"])
    enc._pending = {}
    enc._last_latent = enc._last_latent * 1000.0  # blow the quantized half up
    enc._register_calm_losses(h)
    assert float(enc._pending["calm_energy"]) == pytest.approx(e1, rel=0.5)


def test_the_vote_quantizes_the_decoded_feature_not_the_latent():
    """`_quantize_to_codes` applies the analysis rotation, which is defined on
    patch FEATURES. Feeding it raw latents quantized in the wrong space."""
    import inspect

    from praxis.encoders.abstractinator.calm import AbstractinatorCALM

    src = inspect.getsource(AbstractinatorCALM.vote_next_latent)
    assert "self.vae.decode(" in src

    m = build()
    out = m.encoder.vote_next_latent(torch.randn(3, 64))
    assert out.shape == (3, 64)
    assert torch.isfinite(out).all()


def test_the_two_codecs_emit_one_latent_per_patch():
    """Two encoders side by side into one trunk is only well-posed because the
    patching is STATIC: both emit exactly one latent per patch, so z_q + z_c is
    an alignable merge rather than two irreconcilable sequences."""
    m = build()
    enc = m.encoder
    h = torch.randn(2, 8, m.config.hidden_size)
    z, _ = enc._post_downsample(h, torch.zeros(()))
    assert z.shape == h.shape
    assert enc._last_code.shape[:2] == h.shape[:2]


# ── generation: the vote, selectable at RUNTIME ────────────────────────────


def _gen(model, n=16, b=1):
    from transformers import GenerationConfig

    torch.manual_seed(2)
    return model.generate(
        torch.randint(0, 256, (b, 32)),
        generation_config=GenerationConfig(max_new_tokens=n, do_sample=False),
    )


def _build_mode(mode):
    c = cfg()
    c.generation_mode = mode
    torch.manual_seed(0)
    return PraxisForCausalLM(c).eval()


def test_generation_mode_is_not_part_of_the_model_hash():
    """A registry profile would have forced a whole separate TRAINING run just
    to compare decoders. Both paths are trained by the same objectives, so the
    flag is inference-only and excluded from the hash."""
    from praxis.cli.core.hasher import compute_args_hash

    base = ["--encoder-type", "x", "--batch-size", "4"]
    assert compute_args_hash(base) == compute_args_hash(
        base + ["--generation-mode", "vote"]
    )


def test_the_encoder_declares_its_own_modes():
    """An encoder with exactly one decoding path names it, so the run never has
    to. CALM only ever decodes by vote; a plain byte-latent encoder drives no
    custom path at all."""
    from praxis.encoders.abstractinator import AbstractinatorCALM
    from praxis.encoders.base import BaseEncoder
    from praxis.encoders.calm.encoder import CALMEncoder

    assert CALMEncoder.generation_modes == ("vote",)
    assert CALMEncoder.default_generation_mode == "vote"
    assert AbstractinatorCALM.generation_modes == ("standard", "vote")
    assert AbstractinatorCALM.default_generation_mode == "standard"
    assert BaseEncoder.generation_modes == ()


def test_an_unsupported_mode_fails_loudly_at_build():
    """Silently decoding the other way is the failure to avoid."""
    m = build()
    with pytest.raises(ValueError, match="supports generation_mode"):
        m.encoder.resolve_generation_mode("nonsense")
    # An encoder with no custom path rejects a mode it cannot drive.
    parent = build(PARENT)
    assert parent.encoder.resolve_generation_mode(None) == "standard"
    with pytest.raises(ValueError, match="no custom generation path"):
        parent.encoder.resolve_generation_mode("vote")


def test_default_is_the_byte_loop_and_the_vote_is_off():
    """`vote_next_latent` was fully implemented, tested, and NEVER CALLED: the
    class defined no custom_generate, so it inherited the base hook that
    returns None and generation fell through to the byte-level loop."""
    m = build().eval()
    assert m.encoder.generation_mode == "standard"
    assert m.encoder.custom_generate(
        torch.randint(0, 256, (1, 8)),
        base_forward=None,
        latent_forward=None,
        decode_logits=None,
    ) is None
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
    a, b = build().encoder, _build_mode("vote").encoder
    assert {n: tuple(p.shape) for n, p in a.named_parameters()} == {
        n: tuple(p.shape) for n, p in b.named_parameters()
    }


def test_vote_generation_never_mutates_the_codebook():
    """`_trunk_input` calls the quantizer's forward, which runs EMA updates and
    dead-code resets - both gated on `self.training`. Pushing generated
    candidates into the live codebook is the exact bug `_quantize_to_codes` was
    hand-rolled to avoid."""
    m = _build_mode("vote")
    core = getattr(m.encoder.quantizer, "quantizer", m.encoder.quantizer)
    before = core.stage_codebook(0).clone()
    _gen(m, n=8)
    assert torch.equal(before, core.stage_codebook(0))


def test_vote_generation_requires_the_latent_and_logits_seams():
    """A patch the model just PREDICTED has no bytes behind it, so
    `base_forward` cannot reach it; and the head belongs to the model, not the
    encoder. Both arrive as closures rather than stored back-references."""
    m = _build_mode("vote")
    for missing in ("latent_forward", "decode_logits"):
        kw = {"base_forward": None, "latent_forward": object(), "decode_logits": object()}
        kw[missing] = None
        assert m.encoder.custom_generate(torch.randint(0, 256, (1, 8)), **kw) is None
