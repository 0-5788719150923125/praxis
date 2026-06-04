# Forced Computation: capacity as compute, domain-respecting math, shared geometry

Status: exploratory (2026-06-04). The "why a tiny model can still be a world
model" companion to [world_models.md](world_models.md) (which is the "how to
distribute it"). Grounded/speculative split kept explicit, like
[the_dial.md](the_dial.md). Touches the scaling conjecture in `research/main.tex`,
[exponentials.md](exponentials.md), [harmony.md](harmony.md).

## The spine (falsifiable)

**Capacity is paid in computation, not parameters.** Parameter count sets a
representational ceiling; recurrent depth and harmonic coupling spend *compute*
to reach effective capacity well above what the parameter count alone would
suggest. Praxis is built to be useful at the scale where this trade is visible
and cheap to test: sub-100M parameters, currently ~1.26M. That small scale is a
deliberate test condition, not a limitation - the bet only means something if it
holds where there are not enough parameters to simply memorize.

This is the sharp, testable form of the paper's existing claim that what a model
can learn is set by the geometry of its representation more than by its size. The
formal version is already in the paper: the scaling conjecture (consensus $O(N)$,
interference $O(\log N)$). This note adds the operational reading - the gears, not
the parameters, are where the capacity lives - and the test: **param-efficiency
at tiny scale.** Does the recurrent/harmonic stack beat a same-parameter baseline
that lacks the forced structure? A negative result kills the thesis cheaply.

## "Forcing" = importing a domain's mathematical structure as inductive bias

We force harmonics, continuous latents, autoregression, recurrence. Each forcing
imports the *structure* of a mathematical domain - and, where we can name it, an
actual provable property. Calling these "guarantees" is fair only for the
properties we can write down; the rest is aspiration. Cashing it out honestly:

- **Autoregression** -> an exact factorized likelihood (chain rule). This one is
  a real guarantee: a proper normalized joint, exactly.
- **Continuous latents (CALM/VAE)** -> a differentiable, smooth latent space with
  a reconstruction bound (the ELBO). The guarantee is the bound and the
  smoothness, not "understanding."
- **Harmonic analysis** -> superposition and band-limited/smoothness structure;
  Weyl-equidistributed phases give a provably non-degenerate basis. The guarantee
  is the spectral structure, not semantic correctness.
- **Recurrence** -> iterative refinement toward a fixed point. If the depth map is
  contractive it converges; that is a convergence property, conditional on
  contraction, not unconditional.

So the honest statement is: **each forcing makes the model compute the way a
particular domain's mathematics computes**, inheriting that domain's structure
(and a small, nameable set of guarantees) while remaining stochastic in how it
manifests. "Mathematics designed to compute in those areas confers guarantees in
those areas" is true exactly to the extent we can name the property - and that
list above is the test of the claim, not a flourish on top of it.

## Recurrence as refinement of the input distribution

With recurrent steps the same physical layers are revisited (the watch turned
into the depth dimension - see the paper's recurrent section). Read as
computation: each step re-forms the field rather than enumerating positions,
gradually fine-tuning the whole input distribution and pushing the decision into
latent space, where the KL halting gate reads it out at a calibrated confidence
(the trinary gate: continue / commit / abstain). Capacity-as-compute is literally
this: more turns of the same small mesh, not more teeth.

## Resolution vs orientation: two different "thirds"

A tempting slogan is that the continuous value between 0 and 1 is a third state -
trinary computing. It is not. A continuum is infinitely-valued, the opposite of
three discrete states; calling the spectrum on the 0-1 line a "third bit" is a
category slip. There are two genuinely different moves, and they are orthogonal:

- **Resolution along an axis** - the continuous value. Finer gradation on *one*
  axis: the continuous latent, the float-as-dial of [the_dial.md](the_dial.md).
- **A new axis** - the *third point*. Two points fix a line; three non-collinear
  points fix a plane (the paper's `lemma-three-plane`), and a phase needs that
  plane. The third point adds *orientation*, not resolution.

Praxis spends both, and they are not the same expenditure: continuous latents buy
resolution on each axis; the trinary halting gate (continue / commit / abstain -
a real discrete ternary decision, and a legitimate nod to balanced ternary) buys
the third point that lifts the line into the plane where phase, and therefore
harmony, first becomes possible. The reason "the ternary is the important part"
is exactly that it is *off the line* - a third value crammed onto the 0-1 axis
would add nothing; a third point opening a second dimension adds everything.

And the dimension does not stop at two - this is where the ascent is honest. The
plane the third point fixes is *configuration* space: it says where the phase
sits. But a phase has a velocity and a direction (the harmonic is turning), and
position-and-motion together is a *phase space* - for a two-dimensional plane, a
**four-dimensional** one. So the fourth dimension is real and earned: it is where
harmonic *motion*, not just harmonic position, becomes representable. The line is
one dimension; the third point lifts it to the plane, two; and the turning that
plane was built to carry doubles it to the four the dynamics actually inhabit.
Each new point is a rung. This is the grounded reading of "the ascent"
([goad.md](goad.md)): not a digit smuggled into a constant, but the climb from a
position, to a line, to the plane where phase lives, to the phase space where
phase *moves* - the velocity-and-direction the third point was always pointing at.

## Geometry as a shared symbolic substrate (speculative)

If the computation has a shape, the shape shows up in geometry - the crystal
centers and harmonic fields we already render. The conjecture: the space of
shapes is *finite but practically infinite* (a vast basis of distinguishable
configurations), and the model does **not** learn to represent every shape. It
learns the **symbolic meaning** of each shape and the transform from that shape
to an output modality (text, or whatever we ask it to produce). The learning
space is then *shared across modalities* - one geometric latent, many readouts -
with the per-domain guarantees above riding along innately because they come from
the math, not the data.

This is the aspirational horizon and is flagged as such. It connects to
[the_dial.md](the_dial.md) (any finite bit sequence is a position to classify by
geometry) and to world_models.md (the streamed, partial world model). It is not a
result; the falsifiable down-payments are below.

## What would make this real (in order of cheapness)

1. **Param-efficiency at tiny scale.** The spine. Recurrent/harmonic stack vs a
   structure-free baseline at matched parameters; does forced computation buy
   capacity? Testable now.
2. **Per-forcing property checks.** Verify each named guarantee actually holds in
   the implementation (AR likelihood normalization, ELBO behavior, harmonic
   spectral structure, recurrence contraction/convergence). Turns "guarantees"
   from rhetoric into a checklist.
3. **Shape <-> symbol correspondence.** Does a recurring geometric shape map to a
   stable symbolic meaning across inputs? (Probe the crystal/harmonic geometry.)
4. **Shared latent across modalities.** Later: one geometric latent, multiple
   readouts - the world-model claim, only worth attempting after 1-3.

## What stays manifesto

"Monitor all five domains of science and you have a world model" is the framing,
not a result - keep it as direction. The paper gets only the spine (capacity as
compute at tiny scale, stated as the deliberate test regime) and the honest
inductive-bias reading; the symbolic-geometry and world-model claims live here,
attached to their tests, per the discipline that removed the fake figure.
