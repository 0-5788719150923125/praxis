# Mixture-of-Widths: deep recurrence as a population of narrow voters

Status: Phase 0 implemented and runnable (2026-06-05). The plumbing is in -
`praxis/width/` registry, `--width-type`, the helical deflation policy, the
per-depth profile metric, and a "Width Profile" dashboard card. What is *not* yet
done is the part that makes it pay: real matmul shrinkage and a learned schedule.
Sibling to [forced_computation.md](forced_computation.md),
[the_fifth_dimension.md](the_fifth_dimension.md),
[architecture_separation.md](architecture_separation.md),
[deep_recurrent_cost.md](deep_recurrent_cost.md),
[oscillatory_axes.md](oscillatory_axes.md), [the_dial.md](the_dial.md).

## The idea

Recurrent depth already lets the same small mesh turn many times. Mixture-of-
widths adds a second axis to that turning: the *width* of each turn breathes.
Spec the model at full capacity, then deflate by default - each recurrent step
runs at a low inner rank, and the rank rises and falls on a schedule over depth.
The residual stream stays full width; what deflates is the rank *inside* each
block (the GLU's inner channels, eventually the attention heads). So each step
contributes a low-rank additive update to the full stream, and a deep stack
becomes a **population of narrow voters whose consensus over depth recovers a
full-rank computation**. That is the route to extreme recurrent depth: not wider
gears, more turns of a cheaper gear.

This is the same reframe the scaling conjecture rests on (linear consensus vs
logarithmic interference): a single full-rank pass is one loud voter; many
low-rank passes are a chorus whose interference pattern carries the capacity. The
clock turns, and the width of the hands is itself a coordinate.

## The helix

A fixed low-rank prefix would always exercise the same channels. Instead the
active window precesses with depth by a golden-ratio (Weyl) stride, so the kept
block winds around the full width like a helix and the union over depth covers
everything while any single step stays cheap. The Weyl stride is the same trick
the harmonic head uses for its frozen phases - a deterministic, well-spread index
walk, no resonance, no learned parameters. "Helical unrolling of weight
dimensions over time" is literal: which dimensions participate rotates as the
gears turn.

## The arch (what Phase 0 shows)

The active fraction follows a skewed raised-sine arch over depth: it inflates
from a floor at the first step up to a crest near the front of the stack, then
decays back toward the floor through the tail. Features grow early and thin out
late - biased to run narrow, and so to exit, as depth grows. This is the
deterministic Phase-0 schedule, chosen so the dynamics are *visible*: the Width
Profile card plots the arch directly (`width/active_d{depth}`), and you can see
inflate-at-front, decay-at-tail at a glance. `--width-type` presets tune it:
`helical` crests early (peak 0.3), `helical_late` mid-stack, `helical_steady` a
gentle constant breathing, `helical_tight` an aggressive floor.

## What stays open (the phases)

- **Phase 0 (done):** helical mask + arch, applied as a forward pre-hook that
  zeros inactive inner channels (full matmul, zeroed input). Proves the
  capacity-fluctuation dynamics and the metric. No speedup yet.
- **Phase 1 (real savings) - done for FFN and attention:** `helical_sparse`
  (`praxis/width/sparse.py`) runs two recipes per step, both over the precessing
  helical window, and gradients train only what ran:
  - **FFN** - a generic resizer keyed by the GLU `up`/`down` name convention
    slices both weights, so the intermediate matmul shrinks. A per-channel
    parametric activation (Serpent's a/b/g, Snake, PReLU) is sliced to the same
    window so it stays aligned - including ArcGLU's per-depth activation list;
    the slice is deferred only while such a param is still lazy
    (uninitialized), so it materializes at full width first (the model's
    lazy-init pass handles that). Retargeting another convention is a one-line
    `PAIR_NAMES` change.
  - **Attention** (the dominant compute) - whole KV heads and their GQA query
    groups are dropped via the attention's own `head_budget` context
    (`CausalAttention.head_budget`), which slices the fused QKV, output, gate,
    per-depth QKV bias, and the Infini `betas`/`init_mem`/`init_z` in lockstep
    and drops the head counts the views read. The dataflow-entangled slicing
    lives in the module that owns it - a name-keyed external resizer can't do it
    safely (q/k/v split, RoPE, per-depth and per-head state sit between qkv and
    output) - so the policy only chooses how many heads survive. The per-depth
    *output* bias is per-hidden, not per-head, so it stays whole.

  The budget is a function of depth, so only one shape occurs per recurrent step
  - recompiles are bounded and predictable (a multiplier over the steps), which
  is the whole discipline; unbounded continuous width stays a non-goal.

  **Eager-only.** The runtime monkeypatching is untraceable by Dynamo, so width
  no-ops under `torch.compile` (guarded via `is_compiling`); it is an eager-mode
  optimization. Stacking compile and width would need a native budget *argument*
  the forward reads (no monkeypatch) + bucketed ranks so Dynamo specializes one
  graph per bucket. See [deep_recurrent_cost.md](deep_recurrent_cost.md).
- **Phase 2 (learned schedule):** replace the deterministic arch with a width
  gate - a per-depth base (an `nn.Embedding(depth, 1)` like the learnable RoPE
  theta) plus an input-conditional delta - trained with a budget loss whose
  default pull is toward the floor (deflate by default), upward pressure earned
  only when it lowers task loss. This mirrors the Taxus asymmetric budget loss
  and the KL halting "bias to exit early." Straight-through on the threshold for
  gradients. With Matryoshka/slimmable training (sample the rank during
  training), each prefix becomes individually valid and the voters genuinely
  span the space. Once it learns, the Width Profile card stops being a static
  arch and starts moving over training - the cosine the dynamics want to become.

## Why this is a note, not paper content yet

The voter/low-rank-consensus framing is compelling and ties cleanly to the
scaling conjecture, but it is unproven - Phase 0 only demonstrates the mechanism,
not that narrow-voter recurrence reaches what a full-rank model reaches. It earns
a paper paragraph when Phase 1/2 land and there is a depth-vs-cost curve to show.
Until then the honest home is here, next to forced_computation's "capacity paid
in computation, not parameters" - of which this is now a concrete second
instrument.
