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

This started as length extrapolation through harmonic memory of past amplitudes,
with a damping "entry velocity" reading of the Serpent modulation. That damping
reading came back null (below). The reframe to spectral-attractor convergence is
better aligned with the Koopman thesis - but it was made *after* a null, which is
exactly when a falsifiable claim can quietly become an unfalsifiable one.
"Timelessness" and "phantoms" predict nothing as words. The reframe earns its keep
only because it predicts *more* measurements than the length version did (the
battery below); held to those, it is a sharpening, not a retreat. If it ever
floats free of the battery, it has become the sealed box and should be cut.

## What is measured (null, batch 7165)

The damping-line sub-reading is falsified on the obvious cut:

- Encoder Serpent params, γ/β/α vs feature index (linear fit): **R² 0.00-0.08.**
  No line in the bias.
- Activation amplitude vs patch position: thirds 0.291 / 0.304 / 0.299, trend
  **R² 0.015.** No envelope.

This kills "amplitude rises right-to-left / clear dampening" in those views. It
says nothing yet about spectral-attractor convergence, which predicts other things.

## Falsification battery (the cash value of "timelessness")

1. **Convergence within budget.** Hold the input fixed, run the iteration, track
   the spectral state across compute steps; it should reach a fixed point within
   the budget. Falsified if it never settles. This is the literal "computes to a
   cluster," and is the gate to any *paper* claim.
2. **Compute scales with spectral complexity, not length.** Steps-to-converge
   (halting depth) should track the input's spectral entropy and be *flat* in
   sequence length - regress steps against complexity vs against length.
3. **Low-rank superposition ("phantoms").** The attractor should be a superposition
   of few modes: low effective rank in the eigenbasis, approximately linear under
   mode superposition.
4. **Non-harmonic control.** A non-harmonic baseline that does none of (1)-(3) is
   what turns description into evidence.

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

## Servant - the proposed activation

Serpent learns K *independent* per-feature `(α, β, γ)`. **Servant** makes those a
*structured function* of the feature/position index - simplest a line,
`γ(k) = γ₀ + γ₁·k` ("a line in the bias"). Three consequences: the slope γ₁ is an
explicit, watchable velocity; it costs 2 params not K; and a parametric envelope
*extrapolates to unseen lengths where a per-position table cannot* - the length
corollary above, made mechanical. A line extrapolates linearly; the exponential
version needs an exp envelope or the scaling conjecture's interference argument.
Ambiguity to resolve before building: a line *across features* vs a learned line
*direction in K-space* are different activations.

## Status

Stays in `next/`, out of the paper, until at least convergence-within-budget
(test 1) has a plot. The null first look is a feature, not a failure: it moved the
question from "does the amplitude dampen" to "does the iteration converge to a
fixed spectral attractor, and is that attractor length-invariant."
