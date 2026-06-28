# Mixture-of-Widths: deep recurrence as a population of narrow voters

Status: Phase 0 implemented and runnable (2026-06-05). The plumbing is in -
`praxis/width/` registry, `--width-type`, the helical deflation policy, the
per-depth profile metric, and a "Width Profile" dashboard card. What is *not* yet
done is the part that makes it pay: real matmul shrinkage and a learned schedule.
Sibling to [forced_computation.md](forced_computation.md),
[the_fifth_dimension.md](the_fifth_dimension.md),
[architecture_separation.md](architecture_separation.md),
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
  graph per bucket. See the cost question below.
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

## Does trimming embedding width help speed? (the cost question)

Status: analysis, parked until profiled. Mixture-of-widths (`helical_sparse`) cut
the per-step FLOPs and it feels faster, but a deep recurrent model is still slower
than a shallow one. This records *why*, and why reducing the embedding/output
width is a small, orthogonal win - not the lever.

Trimming embedding/head width shrinks the gradient **tensors** (fewer elements),
not the per-element **magnitude**, so it is a parameter-count win - a cheaper Lion
step (O(num_params)) and less optimizer-state memory, plus the fixed per-forward
cost of the byte-latent encoder/embeddings/head - not a gradient-conditioning win.
No optimizer hooks are needed: `embed_size` is already decoupled from
`hidden_size` (96 vs 128 in the calm-c blueprint, projected up), so trimming it is
a static config change. Hooks would only be needed to vary width *dynamically* per
step, which hits the same residual-stream invariant this note respects - you
cannot shrink the stream the layers add into, so a dynamic embedding width does
not compose. (Width gets around it by deflating each block's *inner* rank and
scattering a low-rank update back into the full stream.)

It will not fix the slowness, because the embedding and head run **once per
forward**, amortized over every depth step. A deep recurrent model is slower than
a shallow one because of **depth-sequential latency**: N steps that must run in
sequence, each paying kernel-launch + Python overhead. Width cut per-step FLOPs
(the felt speedup) but left the step *count* and per-step *overhead* untouched,
and trimming the embedding touches neither. The real levers, in order:
`torch.compile` (the slow runs are `--no-compile`, so every step is eager - but it
does not compose with width, see the eager-only note above; pick the regime per
run), per-step fixed overhead (Infini's segment loop and the memory
retrieve/update are Python-looped and paid `depth` times, plausibly bigger than
the embedding at small hidden sizes), and fewer steps (halting already trims
depth; the arch's bias-to-exit compounds it).

Honest recommendation: profile before trimming. For this model the head is small
(byte vocab 264; crystal/HALO, not a large softmax), so the embedding/head is
probably not hot - the depth loop and eager mode are. Test `helical_sparse` as is,
turn compile back on and re-measure to separate eager overhead from real compute,
profile a step to see whether the encoder/head or the per-step attention loop
dominates, and only then consider a smaller `embed_size` - as an
optimizer-step/memory win, not a latency fix.

## Why this is a note, not paper content yet

The voter/low-rank-consensus framing is compelling and ties cleanly to the
scaling conjecture, but it is unproven - Phase 0 only demonstrates the mechanism,
not that narrow-voter recurrence reaches what a full-rank model reaches. It earns
a paper paragraph when Phase 1/2 land and there is a depth-vs-cost curve to show.
Until then the honest home is here, next to forced_computation's "capacity paid
in computation, not parameters" - of which this is now a concrete second
instrument.
