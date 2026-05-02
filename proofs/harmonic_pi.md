# Harmonic head expressiveness, 2D Weyl-seeded, feature-space, multiplicative

## Setup

For input `x` and position `t`, the model emits

```
out[t, d]    = h(x)[t, d] * (1 + b[t, d])
logits[t, v] = (out @ W)[t, v]
```

with `h(x) ∈ R^{T × D}` from the upstream model, `W ∈ R^{D × V}` learnable, and `b ∈ R^{T_max × D}` a 2D real field built from a sparse complex spectrum:

```
S[f_t, f_d] = a[f_t, f_d] * exp(i * 2π * frac(f_t * π + f_d * e)) / sqrt(f_t² + f_d²)^α
b           = irfft2(zero_pad(S), s=(T_max, D), norm="ortho")
```

`a ∈ R^{F_t × F_d}` is learnable, the phase grid is fixed, `α` sets the 2D radial 1/f decay.

## The measure metaphor

A token is a measure, not a note. The T-axis indexes which measure; the D-axis indexes beat-within-measure - the local structure carried inside the token's feature vector. The 2D field is the meter: "beat 3 of measure 7" is phase-related to both "beat 3 of measure 8" (cross-measure persistence on T) and "beat 4 of measure 7" (intra-measure continuity on D). High `F_d` resolves fine internal structure within each measure.

This matches what hidden states actually are - high-dim vectors carrying many learned properties simultaneously, not point events.

## Claim

For any target `p(v | x, t) = q(v, t | x) + ε(x, t, v)` with `q` factorizable as `f(v | x) ⊙ (1 + g(t, d))` for `g` band-limited at `(B_t ≤ F_t, B_d ≤ F_d)` and `||ε||_∞ < δ`, there exist `h, a, W` realizing `p` within `O(δ + 1/sqrt(F_t * F_d))` in KL.

## Sketch

1. **2D Weyl equidistribution.** `{1, π, e}` are linearly independent over Q (both π and e are transcendental, and no nontrivial integer combination of them with 1 vanishes). The 2D Weyl theorem then gives equidistribution of `((f_t α + f_d β) mod 1)` on the unit interval as `(f_t, f_d)` ranges over `N²`, for any such `α, β`. So the unscaled spectrum is a deterministic, well-spread complex basis - 1D Weyl one dimension up.

2. **Effective signal length.** The field lives in an `F_t × F_d`-dim subspace of length-`T_max × D` 2D real signals. With `T_max = 8192, D = 384`, that's ~3.1M field cells parameterized by ~73K amplitudes. The "long localized sequence" the head optimizes over is the entire 2D meter; tokens index measures, but the field exists at every (t, d) cell between them.

3. **Multiplicative coupling forces head participation.** Under additive bias `out = h + b`, the upstream can emit `h'(x) = h_target(x) - b(t)` and route around the bias entirely - the head is a no-op gate. Under multiplicative coupling `out = h * (1 + b)`, recovering `out = h_target` requires `h = h_target / (1 + b)`, which is content-dependent (`b` varies with `t`) and cannot be precomputed upstream. The field is in the gradient path of every output.

4. **Learnable `W` is necessary under multiplicative coupling.** A frozen kernel `W` defines a fixed direction-to-vocab map. Multiplicative modulation produces content-dependent feature shifts that cannot align with any fixed map; the projection must move with the modulation. (Under additive coupling, a frozen `W` could absorb `b @ W` as additive logit perturbations - hence why the previous frozen-`W` design worked for *additive* bias. It does not survive the change.)

## What this does *not* prove

- **Convergence.** Existence ≠ reachability under SGD.
- **Necessity of π and e specifically.** Any pair of irrationals linearly independent with 1 over Q works; π and e are the parameter-free, universal choice.
- **Non-factorizable targets.** Targets where the `(t, d, v)` structure does not factor as `f(v|x) ⊙ (1 + g(t, d))` remain bounded by upstream capacity alone.

## Why 2D coupling matters

A 1D bias on time alone gives D independent waves, one per feature dim. The model can mute the entire bias one feature at a time. The 2D field couples every (t, d) pair - the same `F_t × F_d` amplitudes that lock measures to one another also lock beats within a measure. Disabling the field requires zeroing the whole grid simultaneously, which is a much harder optimization barrier than zeroing D independent tracks.

## Implementation pointer

`praxis/heads/harmonic.py` - 2D Weyl spectrum, learnable amplitudes, `irfft2` each forward, multiplicative coupling, learnable `lm_head`.
