# Hash-Gated Frozen-Anchor Weight Replacement

> Status: **foundation landing** (2026-05-29). A variant *action* for the
> weight-RL controller (calm-c), and the grounded form of the float-precision
> thread parked in [oscillatory_axes.md](oscillatory_axes.md).

## The reframe that makes it work

The earlier objection to "harvest float precision error" was that the error is
content-free - a fixed property of IEEE-754, not an axis the model writes to.
This design sidesteps that: precision-breakdown is **not the signal**, it's an
**addressing function** - a hash that decides *which* weights get touched. A
hash doesn't need to carry task information; it needs to be deterministic and
well-distributed. So content-free is fine for this role.

## The mechanism

Two copies of the editable weights:
- **live**: the normally-optimized weights.
- **anchor**: a frozen snapshot, never optimized ("we don't optimize the
  frozen copy").

A **gate mask** selects, per element of a localized chunk (a weight row),
whether to keep the live value or replace it with the anchor value:

```
mask = selector(row_len, action, seed)         # boolean over the row
w[row][mask] = anchor[row][mask]                # pull selected elements back
```

This replaces the RL controller's harmonic *modulation* action with a
*selection* action: instead of `w *= 1 + alpha*sin(...)`, the policy chooses a
gate that swaps a structured subset of the row back to the frozen reference.
As the live weights drift from the anchor during training, gating becomes a
selective, structured reset-toward-anchor (a plasticity / shrink-and-perturb
flavored move) - and which elements reset is hash-derived, not uniform-random.

## Selectors (pluggable - this is the whole ablation)

- **`sinusoidal`** (default foundation): `mask[i] = sin(omega*i + phi) > thresh`.
  The "sin over the row" idea made literal; the policy's `(thresh, omega, phi)`
  shapes the mask. Structured, log/periodic in index.
- **`uniform_hash`** (controllable baseline): a seeded PRNG marks a `density`
  fraction. Fully deterministic and reproducible. The honest baseline that
  isolates "does *structure* in the gate help, vs random selection at the same
  density?"
- **`precision_hash`** (experimental, not yet built): derive the mask from
  where float precision breaks down as a scalar is driven around the
  representable grid by repeated multiplication (the "spiral"), sampled with a
  delay. Slots into `build_gate_mask` as one more branch.

## Why precision might be worth it - and why to test it last

A clean PRNG already gives "hash-derived selection" with full control, so the
only reasons to prefer the precision route:
1. **Content-dependence.** Keyed on real value magnitudes, *where* precision
   breaks down depends on the model's own state - so the gate becomes
   state-coupled rather than fixed-random. A PRNG can't do that.
2. **Structured distribution.** The selection pattern is log-periodic (per
   binade), not uniform - the same structure `sinusoidal` imposes by hand.

The honest snag: float-precision behavior is **not reproducible** across
GPU/CPU, float32/64, compile/eager, fast-math. A hash must be deterministic;
the precision route quietly isn't. So: build and validate with the controllable
selectors first, then ablate `precision_hash` vs `uniform_hash` vs `sinusoidal`
at matched gate density. That comparison is the actual test of whether the
spiral is special or just pretty.

## Anchor staleness (known knob)

A single early snapshot goes stale (gating resets toward old weights). The
foundation snapshots once at `rl_warmup_steps`; later, consider an EMA/slow
copy (SWA-flavored) or periodic refresh so the anchor tracks "recent-good"
rather than "first-decent". Left as a knob, not solved here.

## Where it lives

`praxis/policies/harmonic_weight_rl.py` (`map_gate_action`, `build_gate_mask`)
and the callback (`rl_edit_mode: anchor_gate`, `rl_selector`, frozen anchor).
Enable in calm-c with `rl_edit_mode: anchor_gate` (+ optional `rl_selector`).
Default stays `harmonic`, so calm-c is unchanged until flipped.
