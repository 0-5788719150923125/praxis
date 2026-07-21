# Memory basis zoo: candidate function classes for new neural memory arms

Status: living list (opened 2026-07-21, spline arm implemented same day).
Companions: [predictive_memory.md](predictive_memory.md) (what the test-time
memory learns), [harmonic_memory_velocity.md](harmonic_memory_velocity.md)
(spectral-attractor conjecture), and the band-smear machinery in
`praxis/memory/surfacings.py`.

## The frame

The memory net is any `DENSE_REGISTRY` variant (`praxis/memory/models.py`), and
`MemoryBandSmear` holds N of them as opposed function-class regimes under one
floored inverse-surprise bandit. That makes the bandit itself the measurement
instrument: adding an arm is cheap (a dense module + a profile entry), and the
`memory_blend_*` cards plus the regime river ARE the experiment - a regime that
earns share above the floor is pulling forecast weight off the others on the
same NextLat target. No per-experiment tuning; the floor protects a new arm
while it matures.

Current arms (abstractinator-d, `mal_energy_quad`): A = serpent-energy MLP
(harmonic/exponential regime), B = EML tree (log-minus-exponent), C = geometric-
grid KAN (fixed multi-scale radial cascade), D = learned-knot spline (adaptive
resolution). Expensive grid cores fire on staggered sparse phases.

A second axis worth remembering: every parameter of a memory net is a FAST
WEIGHT (meta-learned init, surprise-updated at test time). So a basis whose
*shape parameters* are nn.Parameters gets test-time adaptation of the basis
itself - serpent's frequencies retune online (arm A), the spline's knots move
online (arm D). When evaluating a candidate basis, always ask what its shape
parameters mean as fast weights.

## Implemented

### Spline (learned knots) - arm D of mal_energy_quad, 2026-07-21

Piecewise-linear hat basis `max(0, 1 - |x - k| / h)` per feature; knot
positions AND widths are Parameters (`praxis/dense/spline.py`). As fast
weights, the surprise update re-knots the basis online: coarse where the
sequence is smooth, fine where it is complex - adaptive resolution as a
test-time axis. Deliberate contrasts with the KAN arm: learned placement vs
frozen geometric grid (same basis count, 6, so the bandit compares placement
policy, not capacity), and compact support vs Gaussian tails (a local edit
cannot bleed across the axis). Verdict card: `memory_blend_d` vs
`memory_blend_c`.

## Logged candidates

### Rational functions (Padé approximants)

Per-feature `P(x) / Q(x)`. Handles poles and near-singular behaviour that
polynomial and RBF bases can only ring around (Gibbs); if the latent sequence
has sharp transitions, discontinuities, or phase boundaries, a rational basis
catches them with far fewer terms. Implementation note: the denominator must
be structurally positive (Pade Activation Unit trick, `Q = 1 + |q1 x + q2 x^2
+ ...|`) or training will actually visit the pole. Honest caution for the
memory setting: near-pole regions make the surprise gradient spiky, and the
fast-weight update is a bare Adam-style rule with no trust region - a rational
arm probably wants a gradient clamp before it is bandit-safe. Good candidate
for a phase-boundary-heavy corpus (code, structured logs).

### Chebyshev expansions

Orthogonal basis on [-1, 1] via the recurrence `T_{n+1} = 2x T_n - T_{n-1}`;
minimal max-oscillation, much better conditioned than raw monomials. Natural
fit here because the stream the memory reads is RMS-normalized (bounded-ish
dynamics); a tanh squash maps it onto the interval cleanly. Relation to
existing code: `praxis/dense/poly.py` is the raw-polynomial cousin
(monomial degrees + cross terms) - a Chebyshev variant is the numerically
stable version of the same bet and could live as a `cheb` dense or a poly
mode. Shape parameters as fast weights are weak here (coefficients only, the
basis itself is rigid) - that rigidity is the point (stability), but it means
no test-time basis adaptation story.

### Fourier memory

Explicit spectral regression: features `sin(w_i x + phi_i)`, `cos(w_i x)` with
learnable frequencies, linear readout. IMPORTANT distinction from what exists:
arm A (serpent MLP) is a time-domain net with a periodic activation - it is
harmonic-flavoured, not a spectral decomposition. A Fourier arm would store
content AS amplitudes over an explicit frequency set, which is the
[harmonic_memory_velocity.md](harmonic_memory_velocity.md) / Koopman frame
made literal: fixed (or slowly-adapted) eigenbasis, fast-weight amplitudes as
the state. Risk: closest of the five to redundant with arm A - which is
exactly the kind of question the bandit answers cheaply. If a Fourier arm
cannot earn share off serpent, the explicit basis adds nothing over the
learned one; if it can, the Koopman reading gains a concrete leg.

### Wavelet memory

Multi-scale decomposition; each bank owns a frequency band, so coarse bands
carry slow context and fine bands carry transients. Two shapes it could take:
(1) one arm whose basis is a small learned filter cascade (Haar/db2-init,
filters as fast weights = test-time re-tuned bands); (2) more radically, NOT
one arm - let the band-smear split SEVERAL wavelet arms by scale, so the
bandit allocates forecast weight ACROSS TIMESCALES per sequence. Option 2
turns the blend chart into a legible "which timescale is predictive right now"
card, and pairs naturally with the phase-locking story (WaveletLM block
exists as `block_type: wavelet`; this would be its memory-side sibling).
Probably the most interesting post-spline candidate.

## Suggested order

1. ~~Spline~~ (done; watch `memory_blend_d` on -d before adding anything else -
   one new regime at a time keeps the river readable).
2. Wavelet-by-scale (option 2 above) - new information per card, ties to
   phase-locking.
3. Fourier - cheap to build, answers the "is serpent already the harmonic
   memory" question directly.
4. Chebyshev - stability play; try if poly-family regimes ever look promising.
5. Rational/Pade - highest ceiling on sharp-transition corpora, but needs the
   surprise-gradient guard first.
