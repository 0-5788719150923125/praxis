# Where the Cost Is in a Deep Recurrent Model (and whether trimming embedding width helps)

Status: analysis / parked until profiled (2026-06-05). Mixture-of-widths
(`helical_sparse`) cut the per-step FLOPs and it feels faster, but a deep
recurrent model is still slower than a shallow one. This note records *why*, and
why reducing the embedding/output width is a small, orthogonal win - not the
lever. Sibling to [mixture_of_widths.md](mixture_of_widths.md),
[forced_computation.md](forced_computation.md).

## The question

Would reducing embedding width help speed, "so that output gradients would be
smaller"? Does it work that way, or would it need optimizer-level hooks?

## The mechanics (gradient *size* vs *magnitude*)

Trimming the embedding/head width shrinks the gradient **tensors** (fewer
elements), not the per-element gradient **magnitude**. What that actually buys:

- a cheaper **optimizer step** - Lion's update is O(num_params), so fewer params
  means less work and less optimizer-state memory;
- the fixed **per-forward** cost of the byte-latent encoder, the embeddings, and
  the head.

It does *not* change the magnitude of the signal flowing back, so it is a
parameter-count win, not a gradient-conditioning win. No optimizer hooks are
needed: `embed_size` is already decoupled from `hidden_size` (96 vs 128 in the
calm-c blueprint, with a projection up), so trimming it is a static config
change. Hooks would only be needed to vary width *dynamically* per step, and that
hits the same residual-stream invariant mixture-of-widths already respects - you
cannot shrink the stream the layers add into, so a dynamic embedding width does
not compose. (Mixture-of-widths gets around it by deflating each block's *inner*
rank and scattering a low-rank update back into the full stream.)

## Why it will not fix the slowness

The embedding and head run **once per forward**, amortized over every recurrent
depth step. The reason a deep recurrent model is slower than a shallow one is
**depth-sequential latency**: N steps that must run in sequence (no parallelism
across depth), each paying kernel-launch + Python overhead. Mixture-of-widths cut
the per-step FLOPs (FFN + attention heads), which is the felt speedup - but the
step *count* and per-step *overhead* are untouched, and trimming the embedding
touches neither.

The real levers for deep-recurrent speed, roughly in order:

- **`torch.compile`** - the slow runs are `--no-compile`, so every step is eager.
  Compiled is far faster per step, BUT (measured 2026-06-05) it does **not**
  compose with mixture-of-widths: the sparse policy monkeypatches `module.forward`
  and mutates `_parameters`/`_buffers` per step, which Dynamo cannot trace - under
  `torch.compile` (whole-model, `fullgraph=False, dynamic=True`) it raised a hard
  internal error rather than graph-breaking. So width is now guarded to **no-op
  while compiling** (`praxis/width/helical.is_compiling`): the compiled model runs
  at full width; width slicing only applies in eager. The two are alternatives,
  not stacked:
  - `--no-compile` + `width_type: helical_sparse` -> cheaper FLOPs per step,
    eager overhead (width's home regime).
  - compile (no `--no-compile`) -> faster per step, but width disabled.

  To get *both* would need a compile-friendly path: a native width/head **budget
  argument** the forward reads (no runtime monkeypatching), with the rank
  quantized to a few buckets so Dynamo specializes one graph per bucket. That is
  the real Phase-2 of this - the current monkeypatch version is eager-only by
  construction. Until then, pick the regime per run.
- **Per-step fixed overhead** - InfiniAttention's segment loop and the memory
  retrieve/update are Python-looped and paid `depth` times. This is plausibly a
  bigger cost than the embedding at small hidden sizes.
- **Fewer steps** - halting already trims depth; the width arch's bias-to-exit
  reinforces it. Cheaper *and* fewer steps compound.

## The honest recommendation

Profile before trimming. For this model the head is small (byte vocab 264;
crystal/HALO instead of a large softmax), so the embedding/head is probably not
hot - the depth loop and eager mode are. Order of operations:

1. Test the current `helical_sparse` (FFN + attention) as is.
2. Turn `torch.compile` back on and re-measure - separate eager overhead from
   real compute.
3. Profile a step to see whether the encoder/head or the per-step attention loop
   dominates. Only then consider a smaller `embed_size`, and treat it as an
   optimizer-step / memory win, not a latency fix.

If profiling does flag the head, the contained win is a smaller `embed_size`
(static, no hooks) or leaning on the already-loaded `cut_cross_entropy` path for
the vocab projection - both reduce the per-forward fixed cost, neither addresses
the depth-sequential latency that is the actual gap to a shallow model.
