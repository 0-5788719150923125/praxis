"""Amplitude modulation envelope on the harmonic field (off|static|learned)."""

import torch
import torch.nn.functional as F

from praxis.heads.harmonic import AMP_MOD_BASIS_K, FAST_EPS, FAST_SEGMENT, HarmonicField


def _field(mode):
    torch.manual_seed(0)
    return HarmonicField(hidden_dim=16, max_positions=64, amp_modulation=mode)


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


def test_learned_envelope_is_trainable_and_gets_gradient():
    f = _field("learned")
    assert f.envelope_depth() > 0.0
    param_names = {n for n, _ in f.named_parameters()}
    assert "amp_coeffs" in param_names
    assert f.amp_coeffs.numel() == AMP_MOD_BASIS_K

    x = torch.randn(2, 8, 16, requires_grad=True)
    f(x).sum().backward()
    assert f.amp_coeffs.grad is not None
    assert f.amp_coeffs.grad.abs().sum() > 0


def test_static_and_learned_match_at_init():
    # Same formula, same init (single oscillation) -> identical field at step 0.
    s, l = _field("static"), _field("learned")
    torch.testing.assert_close(s._envelope(), l._envelope())


def test_modulation_changes_the_field():
    off, stat = _field("off"), _field("static")
    # Same amplitude init (same seed); the envelope must change the output.
    torch.testing.assert_close(off.amplitudes, stat.amplitudes)
    x = torch.randn(1, 8, 16)
    assert not torch.allclose(off(x), stat(x))


def test_forward_shape_preserved():
    x = torch.randn(3, 8, 16)
    assert _field("learned")(x).shape == x.shape


def test_head_type_keys_compose_sequential_heads():
    # The single-field harmonic+crystal keys are functools.partial over
    # SequentialHead, composing [HarmonicHead(mode, transform-only), CrystalHead]
    # dynamically - no bespoke subclass. The mode lives in the harmonic builder's
    # keywords.
    import functools

    from praxis.heads import HEAD_REGISTRY
    from praxis.heads import CrystalHead, HarmonicHead
    from praxis.heads.stacked import SequentialHead

    for key, mode in [
        ("crystal_harmonic", "off"),
        ("crystal_harmonic_static", "static"),
    ]:
        entry = HEAD_REGISTRY[key]
        assert isinstance(entry, functools.partial)
        assert entry.func is SequentialHead
        harmonic_spec, crystal_spec = entry.keywords["heads"]
        assert crystal_spec is CrystalHead
        assert harmonic_spec.func is HarmonicHead
        assert harmonic_spec.keywords["amp_modulation"] == mode
        assert harmonic_spec.keywords["build_classifier"] is False


def test_prismatic_is_top_level_parallel_split():
    # prismatic is a top-level Parallel of two arms balancing bias vs variance:
    #   Parallel(Sequential(HarmonicField), Sequential(HarmonicField, CrystalClassifier))
    import functools

    from praxis.heads import HEAD_REGISTRY, CrystalHead, HarmonicHead, ParallelHead
    from praxis.heads.stacked import SequentialHead

    entry = HEAD_REGISTRY["prismatic"]
    assert isinstance(entry, functools.partial) and entry.func is ParallelHead
    arm0, arm1 = entry.keywords["branches"]
    assert arm0.func is SequentialHead and arm1.func is SequentialHead

    # arm 0 (bias): a single harmonic field with its own linear readout.
    (field0,) = arm0.keywords["heads"]
    assert field0.func is HarmonicHead
    assert field0.keywords["amp_modulation"] == "learned"
    assert field0.keywords["build_classifier"] is True

    # arm 1 (variance): a transform-only field feeding the crystal classifier.
    field1, crystal = arm1.keywords["heads"]
    assert field1.func is HarmonicHead
    assert field1.keywords["build_classifier"] is False
    assert crystal is CrystalHead


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
    torch.manual_seed(0)
    f = HarmonicField(hidden_dim=16, max_positions=64, amp_modulation="off")
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
    torch.manual_seed(0)
    f = HarmonicField(hidden_dim=16, max_positions=32, amp_modulation="off")
    scaled = torch.complex(f.spec_real, f.spec_imag) * f.amplitudes
    long = f._eval_field(scaled, 2 * f.T, torch.device("cpu"))
    assert torch.allclose(long[: f.T], long[f.T :], atol=1e-5)


def test_pure_is_identity_at_init():
    """No static spectrum: zero field before the input projection learns."""
    f = _field("pure")
    x = torch.randn(2, 8, 16)
    torch.testing.assert_close(f(x), x)
    # Strands: bias is identically zero, nothing separated yet.
    d = f.field_strands()
    assert max(d["bias_energy"]) == 0.0
    assert d["separated"] == 0.0


def test_pure_field_is_input_conditional_and_trainable():
    f = _field("pure")
    param_names = {n for n, _ in f.named_parameters()}
    assert "amp_gain" in param_names and "amp_input.weight" in param_names
    assert "amp_coeffs" not in param_names  # no static base envelope

    with torch.no_grad():
        f.amp_input.weight.add_(0.5)
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = f(x)
    assert not torch.allclose(out, x)  # field is live once the projection is
    out.sum().backward()
    assert f.amp_gain.grad is not None
    assert f.amp_input.weight.grad.abs().sum() > 0

    # Strands now read as pure variance: zero bias, all energy conditional.
    d = f.field_strands()
    assert max(d["bias_energy"]) == 0.0
    assert max(d["var_energy"]) > 0.0
    assert d["separated"] == 1.0


def _fast_field(mode, max_positions=256):
    torch.manual_seed(0)
    return HarmonicField(
        hidden_dim=16,
        max_positions=max_positions,
        amp_modulation=mode,
        fast_weights=True,
    )


def test_fast_weights_are_identity_at_init():
    # fast_u is zero-init, so the per-token overlay is exactly zero and the field
    # matches the no-fast field, for every modulation mode.
    for mode in ["learned", "input", "pure"]:
        x = torch.randn(2, 80, 16)
        torch.manual_seed(0)  # same seed as _fast_field -> identical amplitude grid
        base = HarmonicField(hidden_dim=16, max_positions=256, amp_modulation=mode)
        f = _fast_field(mode)
        torch.testing.assert_close(base(x), f(x))
        assert f(x).shape == x.shape


def test_fast_weights_gradient_reaches_overlay():
    f = _fast_field("learned")
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
    f = _fast_field("learned")
    with torch.no_grad():
        f.fast_u.weight.normal_(std=0.5)  # make the overlay live
    L = 200  # 3 segments at FAST_SEGMENT=64
    x = torch.randn(1, L, 16)
    x2 = x.clone()
    x2[:, 190] += 5.0  # perturb a token in the last segment
    with torch.no_grad():
        o1, o2 = f(x), f(x2)
    earlier = slice(FAST_SEGMENT, 2 * FAST_SEGMENT)  # an earlier segment
    torch.testing.assert_close(o1[:, earlier], o2[:, earlier])
    assert not torch.allclose(o1[:, 190], o2[:, 190])  # its own token did change


def test_fast_weights_overlay_reads_as_variance():
    f = _fast_field("learned")
    with torch.no_grad():
        f.fast_u.weight.normal_(std=0.5)
    f(torch.randn(2, 80, 16))  # populate the live representative
    d = f.field_strands()
    assert max(d["var_energy"]) > 0.0 and d["separated"] > 0.0
    assert float(f._fast_repr.norm()) > 0.0


def _reference_delta_loop(field, hs):
    """The plain sequential delta-rule loop the vectorized _fast_retrieve must
    reproduce. Kept here as the ground truth for the closed-form rewrite."""
    q, k, v = field.fast_qkv(hs.float()).split(field.fast_mem, dim=-1)
    sig_q, sig_k = F.elu(q) + 1.0, F.elu(k) + 1.0
    b, L, _ = q.shape
    mem = q.new_zeros(b, field.fast_mem, field.fast_mem)
    z = q.new_zeros(b, field.fast_mem, 1)
    reads = []
    for s in range(0, L, FAST_SEGMENT):
        e = min(s + FAST_SEGMENT, L)
        sq, sk, sv = sig_q[:, s:e], sig_k[:, s:e], v[:, s:e]
        reads.append((sq @ mem) / (sq @ z + FAST_EPS))
        retrieved = (sk @ mem) / (sk @ z + FAST_EPS)
        mem = mem + sk.transpose(-2, -1) @ (sv - retrieved)
        z = z + sk.sum(dim=1, keepdim=True).transpose(-2, -1)
    return torch.cat(reads, dim=1)


def test_fast_retrieve_matches_sequential_loop():
    # The vectorized affine-recurrence form must equal the naive per-segment loop
    # to float precision, across exact-multiple, ragged-tail, and sub-segment L.
    for L in [50, 64, 200, 256, 513]:
        torch.manual_seed(L)
        f = _fast_field("learned", max_positions=1024)
        with torch.no_grad():
            f.fast_qkv.weight.normal_(std=0.7)  # non-trivial memory
        x = torch.randn(3, L, 16)
        got, want = f._fast_retrieve(x), _reference_delta_loop(f, x)
        assert got.shape == (3, L, f.fast_mem)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0.0)
