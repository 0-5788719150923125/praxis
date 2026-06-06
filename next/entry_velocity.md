# Entry Velocity: Amplitude as the Learned Quantity

> Status: **active** (2026-06-06). Two implementations landed this pass;
> the velocity channel and the depth envelope are the open threads.
> Companion to [oscillatory_axes.md](oscillatory_axes.md) (the substrate
> framing) and [harmony.md](harmony.md) (the head engineering log).

## The intuition

Forward propagation is a wave over steps, and the wave lives in the
gradients too: whatever amplitude a step applies forward, the same
amplitude scales the gradient flowing back through it. So "how much force
should I apply at this step" is one learnable knob controlling both
directions at once - and the safe initial answer is *none*. Data should
enter dampened, and the model should learn its own entry velocity,
irrespective of sequence length (the knob lives on the depth axis, shared
by all tokens).

## What we did

Two instances of the same trick - zero-at-init amplitude, learned ramp -
on two different axes:

- **The pure-variance arm (`prismatic3`, calm-f).** A harmonic field with
  no static spectrum: the field is the input-conditional delta alone,
  exactly zero at init, behind a learnable per-band gain
  (`amp_modulation="pure"`). Variance enters from the opposite end of the
  bias/variance decomposition and can occupy frequency bins the bias arms
  never touch. The strand card shows it as red erupting from a point.
- **ReZero over depth (`residual_type: rezero`).** Per-depth zero-init
  gains on every block branch: `out = residual + alpha[d] * branch`. The
  whole stack is the identity at entry; the model learns the force of each
  step. Crucially the gain is indexed by `current_depth`, so in a
  recurrent stack the *same* block applies a different learned force at
  each loop step - the loop's amplitude profile over depth is itself a
  trained object.

Both follow the no-tuning rule: nothing to set, the ramp is endogenous.

## What to try next

- **Run calm-f.** The decisive readout is the third strand card: does the
  pure arm's gain actually ramp, and in which bands? Compare val
  bits/byte against calm-d; watch the gate share the pure arm earns.
- **A rezero run.** Cheapest probe: any existing config plus
  `residual_type: rezero`. Watch where the alpha profile over depth
  settles - flat (ReZero is just stabilization), rising (momentum-like
  build), or banded (depth has structure worth an envelope).
- **SandwichNorm ablation.** Recurrent depth + bias needed SandwichNorm
  for stability; zero-init entry gains are the other classical fix for
  that instability class. Does rezero let us relax it?
- **Depth envelope basis.** If the alpha profile shows structure, replace
  free per-depth scalars with the existing `_envelope_basis` (K
  low-frequency modes over depth instead of f_t) - the harmonic treatment
  of the depth axis, matching the head's amplitude story.
- **The velocity channel.** The full second-order form (momentum residual
  nets): `v_{d+1} = mu * v_d + f(x_d)`, `x_{d+1} = x_d + alpha[d] * v_{d+1}`.
  Only worth the state cost if the rezero profile suggests inertia.
- **Dashboard.** A depth-gain envelope card (alpha vs depth, one line per
  residual site), and overlay with momentum-grad cosine from the optimizer
  suite: does learned forward inertia align with the optimizer's backward
  inertia? That figure does not exist anywhere yet.
