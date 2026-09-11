"""praxis/heads/harmonic.py: HarmonicField.

Covers the amplitude envelope modes (off, static, learned, input, pure), the
separable field evaluation and its phase table, the fast-weight overlay, cached
decode, and the bias/variance capacity split.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.attention.cache import PraxisCache
from praxis.heads.harmonic import (
    FAST_EPS,
    FAST_REPR_INTERVAL,
    FAST_SEGMENT,
    HarmonicField,
)


def _field(
    mode="off", *, hidden=16, max_positions=64, fast=False, live_input=False, seed=0
):
    """A seeded field. ``live_input`` randomizes the zero-init input projection
    so the conditional envelope of the input/pure modes is live."""
    torch.manual_seed(seed)
    f = HarmonicField(
        hidden_dim=hidden,
        max_positions=max_positions,
        amp_modulation=mode,
        fast_weights=fast,
    )
    if live_input:
        with torch.no_grad():
            f.amp_input.weight.normal_(std=0.5)
    return f


def _live_fast_field():
    """A fast-weight field whose overlay is live in every factor."""
    f = _field(hidden=32, max_positions=256, fast=True)
    with torch.no_grad():
        for mod in (f.fast_u, f.fast_v, f.fast_qkv):
            mod.weight.normal_(0, 0.3)
    return f


# ── Envelope modes ──────────────────────────────────────────────────────────


def test_off_is_identity_envelope():
    f = _field("off")
    assert f._envelope() is None
    assert f.envelope_depth() == 0.0
    # Effective grid is exactly the raw grid when modulation is off.
    assert torch.equal(f.effective_amplitudes(), f.amplitudes.detach())


def test_static_modulates_but_does_not_learn():
    f = _field("static")
    # The envelope is real and non-flat...
    assert f.envelope_depth() > 0.0
    assert not torch.equal(f.effective_amplitudes(), f.amplitudes.detach())
    # ...but its coefficients are a buffer, not a trainable parameter.
    param_names = {n for n, _ in f.named_parameters()}
    assert "amp_coeffs" not in param_names
    assert "amp_coeffs" in dict(f.named_buffers())
    # Same amplitude init as "off" (same seed), so the envelope alone moves the output.
    off = _field("off")
    torch.testing.assert_close(off.amplitudes, f.amplitudes)
    x = torch.randn(1, 8, 16)
    assert not torch.allclose(off(x), f(x))


def test_learned_envelope_is_trainable_and_gets_gradient():
    f = _field("learned")
    assert f.envelope_depth() > 0.0
    param_names = {n for n, _ in f.named_parameters()}
    assert "amp_coeffs" in param_names
    # Coefficient count is derived from the grid: a complete sine basis per axis.
    assert f.amp_coeffs.numel() == f.F_t + f.F_d
    assert f.amp_K == f.F_t + f.F_d

    x = torch.randn(2, 8, 16, requires_grad=True)
    f(x).sum().backward()
    assert f.amp_coeffs.grad is not None
    assert f.amp_coeffs.grad.abs().sum() > 0


def test_static_and_learned_match_at_init():
    # Same formula, same init (single oscillation) -> identical field at step 0.
    s, l = _field("static"), _field("learned")
    torch.testing.assert_close(s._envelope(), l._envelope())


# ── Separable evaluation and the phase table ───────────────────────────────


def _irfft2_reference(field, scaled, seq_len):
    """The old full-T irfft2 construction, kept as the ground truth the
    separable evaluation must reproduce."""
    rfft_D = field.D // 2 + 1
    batched = scaled.dim() == 3
    if not batched:
        scaled = scaled.unsqueeze(0)
    spec = torch.zeros(scaled.shape[0], field.T, rfft_D, dtype=torch.complex64)
    spec[:, 1 : field.F_t + 1, 1 : field.F_d + 1] = scaled
    spec[:, field.T - field.F_t : field.T, 1 : field.F_d + 1] = scaled.flip(1).conj()
    out = torch.fft.irfft2(spec, s=(field.T, field.D), norm="ortho")[:, :seq_len]
    return out if batched else out[0]


def test_separable_field_matches_irfft2():
    """_eval_field == the ortho irfft2 of the Hermitian-extended spectrum,
    unbatched and batched, including the Nyquist column (F_d == D//2)."""
    f = _field("off")
    assert f.F_d == f.D // 2  # Nyquist weight path is exercised
    phase = torch.complex(f.spec_real, f.spec_imag)

    scaled = phase * f.amplitudes
    got = f._eval_field(scaled, 24, torch.device("cpu"))
    want = _irfft2_reference(f, scaled, 24)
    assert torch.allclose(got, want, atol=1e-5)

    batched = phase.unsqueeze(0) * torch.randn(3, f.F_t, f.F_d)
    got_b = f._eval_field(batched, 24, torch.device("cpu"))
    want_b = _irfft2_reference(f, batched, 24)
    assert got_b.shape == (3, 24, f.D)
    assert torch.allclose(got_b, want_b, atol=1e-5)


def test_separable_field_wraps_past_period():
    """Positions past T wrap: the field is T-periodic by construction."""
    f = _field("off", max_positions=32)
    scaled = torch.complex(f.spec_real, f.spec_imag) * f.amplitudes
    long = f._eval_field(scaled, 2 * f.T, torch.device("cpu"))
    assert torch.allclose(long[: f.T], long[f.T :], atol=1e-5)


@pytest.mark.parametrize(
    "max_positions, seq_len", [(128, 1), (128, 16), (128, 64), (128, 128), (32, 70)]
)
def test_phase_table_matches_on_the_fly_computation(max_positions, seq_len):
    """Cached phases equal the formula, including past one period (seq_len > T),
    where the lookup falls off the precomputed table."""
    field = _field(max_positions=max_positions)
    cos_a, sin_a = field._phase_table(seq_len, torch.device("cpu"))
    assert cos_a.shape == (seq_len, field.F_t)
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    f_t = torch.arange(1, field.F_t + 1, dtype=torch.float32)
    ang = 2 * math.pi * t * f_t / field.T
    assert torch.allclose(cos_a, torch.cos(ang), atol=1e-6)
    assert torch.allclose(sin_a, torch.sin(ang), atol=1e-6)


# ── The input-conditional envelope (input, pure) ────────────────────────────


def test_pure_is_identity_at_init():
    """No static spectrum: zero field before the input projection learns."""
    f = _field("pure")
    x = torch.randn(2, 8, 16)
    torch.testing.assert_close(f(x), x)
    # Strands: bias is identically zero, nothing separated yet.
    d = f.field_strands()
    assert max(d["bias_energy"]) == 0.0
    assert d["separated"] == 0.0


def test_pure_field_is_input_conditional_and_reads_as_variance():
    f = _field("pure", live_input=True)
    param_names = {n for n, _ in f.named_parameters()}
    assert "amp_gain" in param_names and "amp_input.weight" in param_names
    assert "amp_coeffs" not in param_names  # no static base envelope
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        assert not torch.allclose(f(x), x)  # field is live once the projection is
    # Strands read as pure variance: zero bias, all energy conditional.
    d = f.field_strands()
    assert max(d["bias_energy"]) == 0.0
    assert max(d["var_energy"]) > 0.0
    assert d["separated"] == 1.0


@pytest.mark.parametrize("mode, gain", [("input", "amplitudes"), ("pure", "amp_gain")])
def test_input_conditional_field_is_trainable(mode, gain):
    f = _field(mode, live_input=True)
    x = torch.randn(2, 30, 16, requires_grad=True)
    f(x).sum().backward()
    assert f.amp_input.weight.grad.abs().sum() > 0
    assert getattr(f, gain).grad is not None


# ── The input-conditional envelope is causal, per position ──────────────────


def test_input_envelope_matches_static_field_at_init():
    """Zero-init projection: every position reads the static (bias) field."""
    f = _field("input", max_positions=512)
    x = torch.randn(2, 200, 16)
    with torch.no_grad():
        got = f(x)
        static = x * (1.0 + f._field(200, x.device, x.dtype))
    torch.testing.assert_close(got, static)


def test_input_envelope_does_not_read_the_future():
    """The pool used to be the mean over the WHOLE window - a leak. Now a
    position's field depends only on its causal prefix: perturb a token, and
    nothing before it may change, while it and everything after must."""
    for mode in ("input", "pure"):
        f = _field(mode, max_positions=512, live_input=True)
        x = torch.randn(1, 200, 16)
        x2 = x.clone()
        x2[:, 100] += 5.0
        with torch.no_grad():
            o1, o2 = f(x), f(x2)
        torch.testing.assert_close(o1[:, :100], o2[:, :100])
        # Every later position pools the perturbed state, so its field moved.
        moved = (o1[:, 101:] - o2[:, 101:]).abs().amax(dim=-1)  # [1, 99]
        assert (moved > 0).all(), mode


def test_input_envelope_conditions_short_windows_too():
    """A window shorter than any bank/segment length is conditioned as fully
    as a long one - the fix must not switch the variance axis off at the
    curriculum's short tiers."""
    f = _field("input", max_positions=512, live_input=True)
    x = torch.randn(3, 16, 16)
    with torch.no_grad():
        got = f(x)
        static = x * (1.0 + f._field(16, x.device, x.dtype))
    assert not torch.allclose(got[:, 1:], static[:, 1:])


def test_input_envelope_matches_whole_window_evaluation_per_position():
    """Position t's field equals the full-window field built from position
    t's own coefficient set - the per-position contraction is exact."""
    f = _field("input", max_positions=512, live_input=True)
    x = torch.randn(2, 40, 16)
    with torch.no_grad():
        prefix = torch.cumsum(x, 1) / torch.arange(1, 41).view(1, -1, 1)
        coeffs = f.amp_coeffs + f.amp_input(prefix)  # [B, T, K]
        field = f._field_conditional(x)
        for t in (0, 7, 39):
            amps = f.amplitudes * f._env_from_coeffs(coeffs[:, t])  # [B, F_t, F_d]
            whole = f._build_field(amps, 40, x.device)
            torch.testing.assert_close(field[:, t], whole[:, t], atol=1e-5, rtol=1e-4)


def test_envelope_factorizes_and_reduces_to_ft_only_when_fd_is_zero():
    f = _field("learned")
    coeffs = torch.randn(f.amp_K)
    coeffs[f.F_t :] = 0.0
    env = f._env_from_coeffs(coeffs)  # [F_t, F_d]
    # Constant along f_d, equal to the f_t factor.
    e_t, e_d = f._env_factors(coeffs)
    torch.testing.assert_close(e_d, torch.ones_like(e_d))
    torch.testing.assert_close(env, e_t.unsqueeze(-1).expand_as(env))


# ── Fast-weight overlay ─────────────────────────────────────────────────────


def test_fast_weights_are_identity_at_init():
    # fast_u is zero-init, so the per-token overlay is exactly zero and the field
    # matches the no-fast field, for every modulation mode.
    for mode in ["learned", "input", "pure"]:
        x = torch.randn(2, 80, 16)
        base = _field(mode, max_positions=256)  # same seed -> same amplitude grid
        f = _field(mode, max_positions=256, fast=True)
        torch.testing.assert_close(base(x), f(x))
        assert f(x).shape == x.shape


def test_fast_weights_gradient_reaches_overlay():
    f = _field("learned", max_positions=256, fast=True)
    names = {n for n, _ in f.named_parameters()}
    assert {"fast_qkv.weight", "fast_u.weight", "fast_v.weight"} <= names
    x = torch.randn(2, 80, 16, requires_grad=True)
    f(x).sum().backward()
    assert f.fast_u.weight.grad.abs().sum() > 0
    assert float(f._fast_repr.norm()) == 0.0  # overlay zero at init -> zero repr


def test_fast_weights_overlay_is_causal():
    # The delta-rule bank is built from PRIOR segments only, so perturbing a token
    # in a later segment must not change an earlier segment's output. The base
    # "learned" field is position-only, so any change would be a future leak.
    f = _field("learned", max_positions=256, fast=True)
    with torch.no_grad():
        f.fast_u.weight.normal_(std=0.5)  # make the overlay live
    L = 200  # 4 segments at FAST_SEGMENT=64, the last one ragged
    x = torch.randn(1, L, 16)
    x2 = x.clone()
    x2[:, 190] += 5.0  # perturb a token in the last segment
    with torch.no_grad():
        o1, o2 = f(x), f(x2)
    earlier = slice(FAST_SEGMENT, 2 * FAST_SEGMENT)  # an earlier segment
    torch.testing.assert_close(o1[:, earlier], o2[:, earlier])
    assert not torch.allclose(o1[:, 190], o2[:, 190])  # its own token did change


def test_fast_weights_overlay_reads_as_variance():
    f = _field("learned", max_positions=256, fast=True)
    with torch.no_grad():
        f.fast_u.weight.normal_(std=0.5)
    f(torch.randn(2, 80, 16))  # populate the live representative
    d = f.field_strands()
    assert max(d["var_energy"]) > 0.0 and d["separated"] > 0.0
    assert float(f._fast_repr.norm()) > 0.0


def _reference_delta_loop(field, hs):
    """The plain sequential delta-rule loop the vectorized _fast_retrieve must
    reproduce. Kept here as the ground truth for the closed-form rewrite.

    Two halves per segment: the compressed bank of everything before it, and a
    causally-masked read over its own tokens."""
    q, k, v = field.fast_qkv(hs.float()).split(field.fast_mem, dim=-1)
    sig_q, sig_k = F.elu(q) + 1.0, F.elu(k) + 1.0
    b, L, _ = q.shape
    mem = q.new_zeros(b, field.fast_mem, field.fast_mem)
    z = q.new_zeros(b, field.fast_mem, 1)
    reads = []
    for s in range(0, L, FAST_SEGMENT):
        e = min(s + FAST_SEGMENT, L)
        sq, sk, sv = sig_q[:, s:e], sig_k[:, s:e], v[:, s:e]
        # Read = compressed bank of PRIOR segments + this segment's causal
        # prefix, sharing one z normalizer. The prefix half is what stops a
        # token being blind from the last segment boundary up to itself.
        span = e - s
        causal = torch.ones(span, span, dtype=torch.bool).tril()
        scores = (sq @ sk.transpose(-2, -1)).masked_fill(~causal, 0.0)
        num = sq @ mem + scores @ sv
        den = sq @ z + scores.sum(dim=-1, keepdim=True)
        reads.append(num / (den + FAST_EPS))
        retrieved = (sk @ mem) / (sk @ z + FAST_EPS)
        mem = mem + sk.transpose(-2, -1) @ (sv - retrieved)
        z = z + sk.sum(dim=1, keepdim=True).transpose(-2, -1)
    return torch.cat(reads, dim=1)


def test_fast_retrieve_matches_sequential_loop():
    # The vectorized affine-recurrence form must equal the naive per-segment loop
    # to float precision, across exact-multiple, ragged-tail, and sub-segment L.
    # Covers both halves of the read: bank and within-segment causal prefix.
    for L in [50, 64, 200, 256, 513]:
        f = _field("learned", max_positions=1024, fast=True, seed=L)
        with torch.no_grad():
            f.fast_qkv.weight.normal_(std=0.7)  # non-trivial memory
        x = torch.randn(3, L, 16)
        got, want = f._fast_retrieve(x), _reference_delta_loop(f, x)
        assert got.shape == (3, L, f.fast_mem)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0.0)


# ── Fast-weight overlay: the causal prefix and the single-segment case ──────


def test_every_token_sees_every_earlier_token():
    """The read covers the whole causal prefix, not whole prior segments, and
    nothing later: perturbing token t moves the read of token t first and of
    every token after it, never of one before it."""
    field = _live_fast_field()
    seq_len = FAST_SEGMENT * 2
    x = torch.randn(1, seq_len, 32)
    base = field._fast_retrieve(x)
    noise = 1e-4  # fp32 cross-talk floor is ~1e-6 of the perturbed magnitude

    for t in (
        0,
        1,
        FAST_SEGMENT // 2,
        FAST_SEGMENT - 1,
        FAST_SEGMENT,
        seq_len - 2,
        seq_len - 1,
    ):
        y = x.clone()
        y[0, t] += 5.0
        delta = (field._fast_retrieve(y) - base).abs().sum(-1)[0]
        moved = (delta > noise).nonzero().flatten()
        assert moved.numel() > 0, f"token {t} influenced nothing"
        assert (
            moved.min().item() == t
        ), f"token {t} first moved read {moved.min().item()}"
        assert moved.max().item() == seq_len - 1


def test_overlay_is_live_at_a_single_segment():
    """The whole reason for the within-segment term: a short sequence used to
    read an empty bank and produce exactly nothing."""
    field = _live_fast_field()

    x1 = torch.randn(2, FAST_SEGMENT, 32)
    x2 = torch.randn(2, FAST_SEGMENT, 32)
    assert field._fast_retrieve(x1).abs().max().item() > 0.0
    overlay1, overlay2 = field._field_fast(x1), field._field_fast(x2)
    assert (overlay1 - overlay2).abs().max().item() > 0.0, "not input-dependent"
    assert overlay1.std(dim=1).max().item() > 0.0, "not varying across tokens"


def test_single_segment_early_out_keeps_gradients_attached():
    """Zero grad, not absent grad: the optimizer must see what it saw before."""
    field = _field(hidden=32, max_positions=256, fast=True)
    x = torch.randn(2, 16, 32, requires_grad=True)
    field._field_fast(x).sum().backward()

    grad = field.fast_qkv.weight.grad
    assert grad is not None, "fast_qkv detached from the graph"
    assert grad.abs().max().item() == 0.0


def test_fast_repr_refreshes_on_a_cadence():
    """The snapshot-only readout holds for a full period, then refreshes."""
    field = _live_fast_field()
    x = torch.randn(2, 16, 32)

    field._field_fast(x)
    first = field._fast_repr.clone()
    for _ in range(FAST_REPR_INTERVAL - 1):  # the rest of the period
        with torch.no_grad():
            field.fast_u.weight.normal_(0, 0.3)
        field._field_fast(x)
        assert torch.equal(field._fast_repr, first), "refreshed inside the period"
    field._field_fast(x)  # first call of the next period
    assert not torch.equal(field._fast_repr, first), "never refreshed"


# ── Cached decode: chunked == full-sequence ─────────────────────────────────


class _FakeCache:
    """Stands in for PraxisCache: past_length() is what the trunk has written
    (including the current chunk, as at head time), plus the head side-store."""

    def __init__(self):
        self.length = 0
        self.head_states = {}

    def past_length(self):
        return self.length

    def get_head_state(self, key):
        return self.head_states.get(key)

    def set_head_state(self, key, state):
        self.head_states[key] = state


def _chunked_equals_full(f, x, chunks):
    with torch.no_grad():
        full = f(x)
        cache = _FakeCache()
        outs = []
        pos = 0
        for size in chunks:
            chunk = x[:, pos : pos + size]
            cache.length = pos + size  # trunk wrote this chunk before the head runs
            f.decode_cache = cache
            try:
                outs.append(f(chunk))
            finally:
                f.decode_cache = None
            pos += size
        got = torch.cat(outs, 1)
    torch.testing.assert_close(got, full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["static", "learned", "input", "pure"])
def test_cached_decode_matches_full_forward(mode):
    """Prefill of 70, then single tokens and a multi-token chunk that straddles
    a FAST_SEGMENT boundary: every chunk must equal its slice of the full
    forward. Before this the head evaluated every chunk at positions 0.. and,
    for the conditional modes, pooled the suffix alone."""
    f = _field(mode, max_positions=512, fast=True, live_input=mode in ("input", "pure"))
    with torch.no_grad():
        f.fast_u.weight.normal_(std=0.5)
    x = torch.randn(2, 150, 16)
    _chunked_equals_full(f, x, [70, 1, 1, 60, 1, 17])


def test_cached_decode_without_cache_is_full_recompute():
    """Cache-less attention leaves past_length at 0 and feeds the whole
    sequence every call; the head must then behave exactly as untethered."""
    f = _field("input", fast=True, live_input=True)
    x = torch.randn(2, 30, 16)
    cache = _FakeCache()  # length stays 0
    with torch.no_grad():
        want = f(x)
        f.decode_cache = cache
        try:
            got = f(x)
        finally:
            f.decode_cache = None
    torch.testing.assert_close(got, want)


def test_cached_decode_positions_past_one_period():
    """The on-the-fly phase path with an offset: chunks past T still match."""
    f = _field("learned", max_positions=32)
    x = torch.randn(1, 50, 16)
    _chunked_equals_full(f, x, [30, 5, 15])


# ── Cached decode through a real model and PraxisCache ──────────────────────


@pytest.mark.parametrize("head_type", ["crystal_harmonic", "prismatic6"])
def test_harmonic_head_cached_logits_match(head_type):
    """The harmonic field is anchored to absolute position and (prismatic6's
    stem) carries a causal prefix mean and a fast-weight bank across tokens.
    Under cached decode the head sees only the suffix, so without the head-side
    cache state it evaluated every chunk at positions 0.. and pooled the suffix
    alone. Prefill + one-token steps must match the full forward, logit for
    logit, across a FAST_SEGMENT boundary."""
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        device="cpu",
        block_type="transformer",
        max_position_embeddings=256,
        attention_type="vanilla",
        embeddings="positional",
        encoding="nope",
        head_type=head_type,
        block_size=128,
    )
    model = PraxisForCausalLM(cfg).eval()
    ids = torch.randint(0, 200, (1, 70))  # crosses the 64-token segment fold
    with torch.no_grad():
        full = model(input_ids=ids).logits
        cache = PraxisCache()
        prefill = model(input_ids=ids[:, :60], past_key_values=cache).logits
        steps = [
            model(input_ids=ids[:, i : i + 1], past_key_values=cache).logits
            for i in range(60, 70)
        ]
    torch.testing.assert_close(prefill, full[:, :60], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.cat(steps, 1), full[:, 60:], rtol=1e-4, atol=1e-5)
    # The head actually left state behind (it is not falling back silently).
    assert any(k.startswith("HarmonicField:") for k in cache.head_states)


# ── Capacity split ──────────────────────────────────────────────────────────


def test_variance_share_excludes_dormant_headroom():
    """The bias/variance split of the written field, dormant out of it.

    ``harmonic_capacity_variance`` divides by a saturation ceiling, so on a
    concentrated field it reads far below the actual delta share. This asserts
    the two are genuinely different numbers and that the new one is the one
    with the honest denominator.
    """
    f = _field("input", max_positions=256, fast=True, live_input=True)
    with torch.no_grad():
        f.fast_u.weight.normal_(std=0.5)
    f(torch.randn(2, 80, 16))  # populate _last_input_coeffs and _fast_repr
    c = f.capacity_split()

    share = c["harmonic_variance_share"]
    bias, var = c["harmonic_capacity_bias"], c["harmonic_capacity_variance"]
    assert 0.0 <= share <= 1.0
    # Same ratio, dormant removed from the denominator.
    assert share == pytest.approx(var / (bias + var), rel=1e-5)
    # Dormant is real headroom here, so the honest share is strictly larger.
    assert c["harmonic_capacity_dormant"] > 0.0
    assert share > var


def test_variance_share_is_zero_for_a_static_field():
    """No conditional path -> the field is pure bias, and the share says so."""
    f = _field("static", max_positions=128)
    assert f.capacity_split()["harmonic_variance_share"] == pytest.approx(0.0)
