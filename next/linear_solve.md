# Linear solve: the radical version

Status: deferred. Gated on the shipped version proving out first.

## What shipped (2026-06-06)

`LinearPrior` (praxis/heads/energy.py): a closed-form streaming-ridge readout
beside the CALM energy head's MLP. `z = W*h + f(h, noise)`; W is solved from
EMA sufficient statistics during the post-freeze warmup window, then frozen.
Backprop runs in parallel the whole time and trains only the residual. W never
touches model weights. Watch `calm_prior_r2` - it decides whether any of this
thread matters.

## The radical version (this note)

Instead of a parallel solved term, the solver writes INTO the model's weights:

- For X steps at the start of stage 2, suppress backprop on the energy head
  entirely. Each step, least-squares-solve the head's final linear layer (or a
  low-rank factor of it) directly against the batch: ELM / reservoir-style
  initialization of `f` itself, not a bypass around it.
- At window end, the solve decays away (blend the solved weights toward
  trainable ones over a short ramp) and backpropagation takes over, starting
  from a head that already solves the linear part of the conditional.
- Harmonic variant: solve in the harmonic feature basis, so the warm-start is
  literally fitted Fourier coefficients - "converge onto harmonic sequences
  quickly" as an initialization, not a hope.

Why it might beat the parallel version: the MLP's nonlinear layers then shape
features ON TOP of a correct linear readout from step one, rather than learning
a residual around an external term they can't influence.

Why it might lose: backprop has to take over from an initialization it didn't
choose; early gradient noise can destroy the solved structure before the
nonlinear layers learn to support it (the parallel version is immune to this
by construction - W is a buffer).

## Decision rule

Only build this if the shipped version shows `calm_prior_r2` is high but the
fluency gain is marginal - i.e. the linear structure exists and the residual
framing is the bottleneck. If r2 is ~0, neither version matters: the backbone
does not linearize the sequence and the problem is upstream.

Credit: both versions trace to the 16-year-old on Reddit who believed ML was
a linear solve. See the LinearPrior docstring.
