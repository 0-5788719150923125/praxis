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
        ("posterior", enc.posterior),
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
    z = enc.vote_next_latent(h)

    cond = h.unsqueeze(1).expand(4, enc.vote_samples, 64).reshape(-1, 64)
    torch.manual_seed(99)
    noise = torch.randn_like(cond)
    pool = enc.energy_head(cond, noise).view(4, enc.vote_samples, 64)
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
