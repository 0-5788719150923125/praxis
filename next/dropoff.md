# Dropoff: forcing error instead of absorbing it

> Status: **vision / unscoped** (2026-06-03). A hypothesis, not a result. The
> kernel is sound and partly grounded in what already runs; the mechanism is
> deliberately left open. Companion to [world_models.md](world_models.md),
> [oscillatory_axes.md](oscillatory_axes.md), and the attention-sink and
> the-seed-dissolves material in `research/main.tex`.

## The anchor we missed: ghostmax is about precision

Ghostmax (softmax1, [Miller 2023](https://www.evanmiller.org/attention-is-off-by-one.html))
is usually told as "a head that can attend to nothing." Its actual motivation was
**quantization**. A softmax forced to sum to one cannot express "no match," so heads
that want to be no-ops instead dump their weight somewhere and produce enormous
activation outliers (the attention-sink dimension). Those outliers are precisely
what break int8/4-bit quantization. The implicit `exp(0)=1` in the denominator is an
escape valve: the head down-weights everything, the outliers vanish, the model
quantizes cleanly. The sink is a **precision-preserving, stabilizing** device.

Our paper's `attention-sink` framing currently tells the harmonics-dampening story.
That is downstream and true, but the origin is precision, and we should say so.

## Dropoff: the matched dual

If ghostmax is a _sink_ - a zero-logit, zero-value ghost that lets total attention
fall to zero and bounds the output - then its dual is a device that does the
opposite on purpose: it **injects** error rather than absorbing it, and it weans the
model off the causal tip.

The sharpened shape: **a sink at the tip, not the start.** Ghostmax's ghost is
positionless (a pristine zero, masked always-accessible at index 0); dropoff gives
the sink a position - the most-recent token, the causal focus - so attention there
can fall into a no-op. Two precisions make this hold water rather than rhyme:

- **The feature-dependence rides the value, not the weight.** Attention weights are
  per-head scalars; there is no per-feature attention weight. So dropoff is a
  feature-dependent **warp on the value** that sinks the tip: an envelope, zero at
  the last position and recovering backward at a per-feature rate, so attending to
  the tip injects ~0 per feature. It is the value-side dual of ghostmax's
  zero-value ghost, moved from the start to the tip.
- **"Backwards" is the depth/gradient axis, not the forward sequence.** A causal
  forward pass never flows future to past. What flows backward is the gradient and,
  in a recurrent loop, the re-derivation across beats: sink the tip on one beat and
  the surrounding beats learn, backward over the shared-weight interval, to set up
  for it and recorrect after. The "lingering" is across depth, not across the
  sequence.

Where ghostmax improves quantizability by removing outliers, dropoff would **force
quantization error to manifest** - as a training pressure, not a defect.

Implemented as a two-mode ablation (`praxis/attention/causal.py:_maybe_dropoff`):
`shift` (a crude uniform K/V delay) and `warp` (the feature-dependent tip-value
sink above). calm-c runs `warp` at step 6 of 8. Caveat: a tip-sink only touches
the most-recent query per forward, so its per-step effect is small - it relies
entirely on the recurrent loop to amplify it across beats.

## The schedule was never really applied (2026-09-02)

**Dropoff has not been measured, by any run, because the schedule it shipped
with is very nearly inert.**

`dropoff_step = depth - num_layers` fires only when `current_depth` reaches it.
Under KL halting the **training** depth budget is *sampled*, not fixed:
`KLHalting.get_depth` returns `_sample_loop_count() * num_layers` from a
log-normal Poisson while training. (`check` is the inference-only half; reading
that one first is what hid this.) At `-h`'s settings - depth 6, `num_layers` 1,
so `max_loops` 6 and `r_bar = (6-1)/2 = 2.5`, sigma 0.5 - the loop count comes
out:

| loops | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| share | 15.4% | 24.9% | 23.6% | 17.6% | 11.5% | **6.97%** |

Step 5 is reached only at 6 loops, so one-beat dropoff fires on **7.0% of
training steps** and on **2.3% of the passes that actually execute** (`E[loops]`
is 3.06, not 6). calm-c's `dropoff_step: 6` of 8 has the same problem.

The realized rate is therefore set by an unrelated sampling distribution, not by
the ablation's design. Any run that looked flat on dropoff was measuring almost
nothing.

## The two schedules

`dropoff_every` (profiles `*_dropoff_always`) applies the sink at every executed
pass: 3.06 firings per step against 0.07, ~**44x** the exposure. That is the
first schedule under which dropoff is a real intervention.

Arguments for it beyond the exposure:

- `depth - num_layers` is a hand-picked beat, and this repo does not otherwise
  set schedules by hand ([[feedback_no_hyperparameter_tuning]]). Every-pass has
  no knob.
- Ghostmax, the device this is the dual of, is applied at every pass. Not the
  argument it looks like, though: ghostmax is **permissive** (it adds a freedom
  the model may decline) while dropoff is **privative** (it removes information
  the model did not choose to give up). Always-on is obviously right for a
  capability and not automatically right for a deprivation.

Against, stated so a null result stays readable:

- The tip is then permanently absent from the value path at every depth, which
  is a recency **prior**, not an ablation.
- The envelope is anchored to the current forward's `T`, so under a sequence
  curriculum its shape moves with the batch.
- With `dropoff_every` and Kaleidoscope's per-depth facets both active,
  `kaleido_facet_depth_specialization` partly reads compensation for a constant
  per-depth perturbation rather than genuine depth structure.

Measured scale, so neither side is argued in the abstract: the warp touches
exactly **6 positions** whatever `T` is (only the tip is fully zeroed; the
envelope is back to 98.9% six positions back). At T=256 that is 2.3% of
positions and **0.6% of total V mass**.

`experiments/abstractinator-i.yml` runs the every-pass schedule.
`arc_single_dropoff_always_nomem` is the control that isolates the schedule from
the attention core, if a delta needs attributing.

## Fixed 2026-09-02: dropoff was firing at inference

`_maybe_dropoff` had no `self.training` gate, so at generation it fired whenever
the loop reached `dropoff_step` - sinking the value of the most recent token,
which is exactly what the decode conditions on. Now gated in
`CausalAttention._maybe_dropoff`, which arc, single-head arc and kaleidoscope
all inherit.

**The gate is a necessity, not a preference.** The warp is anchored to the
current forward's `T`, and a cached decode passes `T = 1`, so `dist = [0]` and
`warp = 1 - exp(0) = 0`: the token's value is multiplied by **zero**.
`_adjust_kv` runs before the cache write (`InfiniAttention.forward`), so every V
entering the cache would be exactly zero and the attention branch would output
zero at every decode step. Verified. Not a distribution shift - a dead branch.

The train/inference asymmetry is nonetheless real and uncorrected. Dropout gets
away with the same asymmetry by applying its *expectation* at test time; a
multiplicative envelope in `[0, 1]` has no analogous rescaling. Two coherent
designs, and the code implements the first:

1. **Train-only** (current). An uncorrected shift, and a much larger one under
   `dropoff_every`, where every pass now trains under the envelope.
2. **On at both.** Requires re-anchoring the envelope to *absolute* sequence
   position so a one-token forward knows where the tip is. Not merely a
   consistency fix: under `dropoff_every` the model would never read the most
   recent token's value at any depth while generating, which is a different
   model.

If `-i` shows a train/eval gap that a one-beat arm does not, (2) is the first
thing to try.
