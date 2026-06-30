# Harmonic memory as spectral-attractor convergence (open conjecture)

Status: **conjecture; first measurement null on the length/damping cut.** Captured
honestly with open questions stated plainly - not an established claim. Reframed
from "length extrapolation" to spectral-attractor convergence (see Provenance).
Companion: [[project_titans_memory]], [[project_harmonic_amp_modulation]],
[[project_harmonic_latent_koopman]], and the bias/variance conjecture
(`praxis/pillars/conjectures/bias-variance-decoupling.yml`).

## The idea (current framing)

The representation may be **timeless** in a precise sense: it lives in a *fixed,
length-independent spectral basis* (the Koopman frozen phases), so the eigenbasis
does not grow with the sequence. Computation is then a **fixed-point iteration** -
recurrent depth + halting are the *step budget* - that converges to a **spectral
attractor**: a cluster of amplitudes/frequencies in that fixed basis. The
attractor is a *superposition of eigenmodes* - several modes present at once, the
"overlaid geometries / phantoms," which the model learns to classify. This is
Deep Equilibrium Models meeting Koopman: iterate a fixed operator to an attractor,
read off the spectral state.

Under this frame **length extrapolation is a corollary, not the headline**: a
fixed basis extends to any sequence length by construction, so "length" is the
wrong axis. The live quantities are the spectral content (which modes are active)
and the compute budget to resolve them - "no length, only a budget of steps."

## Provenance (and an honest caution)

This started as length extrapolation with a damping "entry velocity" reading. The
*linear* form of that reading came back null - but the null was scoped too
narrowly (a straight-line fit cannot see clustered/sparse/periodic structure), and
re-measuring the right shape found the **sparse, clustered, low-dim geometry is
actually there** (below). So the reframe to spectral-attractor convergence is
partly *earned by data* (the structural half), not a pure post-null retreat. The
discipline still holds on what remains unmeasured: the periodic-jump half is
untested, and "timelessness/phantoms" predict nothing as words - the reframe keeps
its place only by predicting the battery below. The one part that is *not*
recoverable as a measurement is the cosmic scale ("an Euler-derived timescale ≈
the length of the full universe"): the period is real and measurable (the β's, the
recurrent depth), but "the length of the universe" is not a quantity in the
network. Keep the measurable core; let the cosmic scale be the muse, not the claim.

## What is measured (batch 7165) - one null, one positive

The first pass tested for a *linear* shape and was scoped too narrowly (a low R²
on a line says "not a line," not "no structure"):

- **Null (linear ramp):** Serpent params γ/β/α vs feature index, R² 0.00-0.08;
  activation amplitude vs patch position, trend R² 0.015. Kills the smooth
  "amplitude rises right-to-left / dampening" reading - but *only* that shape.

Re-measured on the shapes the theory actually predicts (sparse, clustered,
periodic), the picture flips on the structural half:

- **Positive (sparse + clustered):** crystal bank centers are low-dimensional and
  tightly clustered - effective dim (participation ratio) **1.0, 1.3, 7.5, 7.4**
  out of 128 across the four experts; nearest-neighbor / mean distance **0.05-0.62**.
  Experts 0-1 are essentially a *line* (eff. dim ≈ 1) - the "line in the bias"
  Servant intuition, unbidden. This is a sparse, clustered geometric index, the
  structural half of the conjecture. Caveat: eff. dim 1.0 is genuine structure OR
  partial center-collapse; the non-harmonic control (test 4) distinguishes them.
- **Untested (periodic jumps):** β frequencies span scales (range/std 7.6, up to
  ~6), but a static FFT of one activation across patches is near-flat (top freq
  1.28x mean - no dominant period). "Jumps between clusters" is a property of the
  *dynamics across compute steps*, not a static activation, so this is untested,
  not disproven (see test 1).

Named prior art the conjecture maps onto: Kanerva **Sparse Distributed Memory**
(vast address space, sparse population, proximity retrieval), the **Platonic
Representation Hypothesis** (approximately-universal shared space), and the
**grokking "clock" circuits** (periodic activations folding inputs modularly).

## Falsification battery (the cash value of "timelessness")

1. **Convergence + discrete jumps within budget.** Hold the input fixed, run the
   iteration, and track *which cluster is active* across recurrent-depth steps. Two
   predictions in one: it should reach a fixed point within the budget (the literal
   "computes to a cluster"), and the path there should be *discrete hops* between
   populated clusters, not a smooth drift - the right-shaped test for the
   periodic-jump mechanism the static FFT could not see. Gate to any *paper* claim.
2. **Compute scales with spectral complexity, not length.** Steps-to-converge
   (halting depth) should track the input's spectral entropy and be *flat* in
   sequence length - regress steps against complexity vs against length.
3. **Low-rank superposition ("phantoms").** The attractor should be a superposition
   of few modes: low effective rank in the eigenbasis, approximately linear under
   mode superposition.
4. **Non-harmonic control.** A non-harmonic baseline that does none of (1)-(3) is
   what turns description into evidence - and specifically tells the measured
   low-dim centers (eff. dim ≈ 1) apart as *organized geometry* vs *degenerate
   collapse*: structure should survive the control, collapse should appear in it too.

## Open questions (unresolved - stated, not papered over)

1. **Basis vs attractor length-invariance.** Is the *basis* length-invariant (nearly
   free - a fixed eigenbasis trivially does this), or the *attractor itself*
   length-invariant (the strong, surprising claim - the same attractor regardless
   of how much sequence is fed)? This decides whether test #2 is the easy or the
   hard version. Open - not yet answered.
2. **Which view** shows the damping gradient, if any? Not the encoder
   param-vs-feature axis, not the patch-position envelope (null above). Unmeasured
   candidates: a dashboard derivative/spectrum plot; the activation derivative
   `dy/dx = 1 + sin(2αx)·α²/(α²+ε²) + γβ·cos(βx)`; a non-raw feature ordering; other
   checkpoints.
3. **Which module?** Serpent lives in the encoder/codec MLPs (`config.activation =
   serpent`). The claim is about the Titans memory - confirm whether the memory
   carries Serpent or whether encoder ↔ memory is being conflated.
4. **"Backward propagation through time"** - BPTT (trivial, every model) or
   anti-causal inference at *inference* time (strong, surprising, testable)?

## Servant - the activation (BUILT 2026-06-30, repurposed)

The original Servant proposal (a parametric *line* over the feature index,
`γ(k) = γ₀ + γ₁·k`, to make Serpent's K params extrapolate) was **set aside**. The
name now carries a different idea we actually built: a **test-time-modulated**
Serpent. `praxis/activations/servant.py`.

Keeps Serpent's harmonic form but lets the per-feature frequency `a` *breathe at
inference* with each token's own energy:

    s     = rms(x over features)            # live per-token energy (detached: a measurement)
    m     = tanh(log s - log_s_ref)         # centered test-time signal in (-1, 1)
    a_eff = a * (1 + MOD_MAX * tanh(v) * m) # frequency breathes -> a learnable chirp
    y     = x + sin^2(a_eff x)/a_eff + g sin(b x)

Why this is the same "velocity" idea, kept honest:
- A frequency that varies across the signal *is* a chirp (paper Definitions:
  frequency = angular velocity). The watchable velocity is now `v`, the per-feature
  coupling, not a line slope.
- By Parseval the per-token RMS is the token's total spectral power, so the
  modulation is driven by a genuinely *spectral* quantity - coupled to harmonic
  principles, self-contained (reads only `x`, reduced over features -> causal,
  instance-local), no plumbing, drops into any activation slot.
- `v` is **zero-init** => `a_eff == a` => Servant *is* Serpent at init. So it is a
  strict generalization that anneals into test-time dependence. `tanh(v)` and
  `MOD_MAX=0.5` bound the swing; the `1/a_eff` floor (Serpent's INV_FLOOR_EPS)
  stops a near-zero modulated frequency from exploding.

**The null is the point.** If `v` stays at zero, the live energy signal carries no
usable information - a clean result, not a failure. Watch `v` in the blueprint.

Staged as **`experiments/calm-d-2.yml`** (`extends: calm-d-1`, `activation:
servant`). calm-d-1 inherits `activation: serpent` from calm-a, so calm-d-2 starts
bit-identical and isolates exactly the test-time-chirp contribution. Validated:
registered, == Serpent at init, `v` gets nonzero finite gradient, bf16 stable,
config resolves. Untested on a real run.

## Status

Stays in `next/`, out of the paper, until at least convergence-within-budget
(test 1) has a plot. The null first look is a feature, not a failure: it moved the
question from "does the amplitude dampen" to "does the iteration converge to a
fixed spectral attractor, and is that attractor length-invariant."
