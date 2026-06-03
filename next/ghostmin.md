# Ghostmin: forcing error instead of absorbing it

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

## Ghostmin: the matched dual

If ghostmax is a _sink_ - a zero-logit, zero-value ghost that lets total attention
fall to zero and bounds the output - then its dual is a device that does the
opposite on purpose: it **injects** error rather than absorbing it, and it weans the
model off the causal tip.

The shape, kept open: a ghost that **caps** attention on the most-recent token (the
causal focus) and routes the surplus into a **delayed** pathway, so the model cannot
lean on the tip and must integrate lagged structure instead. Make the cap **periodic**
and you get a backwards-flowing, pulsing dampening - the recent token is repeatedly
denied, and reliance shifts to delayed computation (the same lagged regime as the
remote-expert swarm; see the "Neither halting nor blocking" section).

Where ghostmax improves quantizability by removing outliers, ghostmin would **force
quantization error to manifest** - as a training pressure, not a defect.

## Why force error: velocity per feature

The bet: deliberately quantize a parameter (or its seed) so error appears, then let
each feature **learn a velocity** that rides it. "Velocity" is not a metaphor here -
it is a phase-rate. Praxis already learns angular velocity per depth (the learnable
per-depth RoPE theta); push it to per-feature and drive it with injected quant noise,
and each feature learns the phase-rate at which its generated wave is **robust** to
the noise on its seed. Error becomes a smoothness prior: the geometry learns to be
the shape that survives being rounded.

## Parameters as seeds; the shape is the parameter

This only coheres if the locus of "parameter" moves. The harmonic field already does
it: the stored amplitudes are seeds, the Weyl phase is **keyed by index** `(f_t, f_d)`,
and the reconstructed field is the effective parameter (this is the-seed-dissolves
made literal). Generalize it: an on-disk scalar at index `i` is a low-precision
**seed**, not a value; combined with `i` it generates a wave of adjustments to a
prior; the wave is the real parameter. You can quantize the seed hard _because_ the
wave absorbs it - weights as a function of their own index, hypernetwork/INR-style,
turned inward. Quantization stops being a cost paid at deployment and becomes the
substrate the structure is trained against.

## Falsifiable predictions

1. **Forced seed quantization + learnable per-feature velocity beats a fixed-velocity
   baseline** under aggressive post-training quantization, at equal capacity.
2. **A tip-cap (or periodic tip-mask) shifts attention mass measurably toward delayed
   positions** without collapsing generation coherence - the model finds the lagged
   pathway rather than failing.
3. **Sensitivity to seed rounding falls over training** when velocity is learnable and
   does not when it is frozen - the wave is learning to absorb the error.

## Smallest testable kernel

- Add per-feature learnable phase-rate (extend the per-depth theta to per-feature).
- During training, inject quantization noise on the seed and measure post-quant loss
  vs the frozen-velocity control (prediction 1, 3).
- Separately, an attention ablation: cap the max attention weight (or periodically
  mask `kv_idx = T`), measure the mass-shift to earlier positions and generation
  coherence (prediction 2). This needs no new parameters and isolates the tip claim.

## What this does _not_ claim

- That ghostmin, in any specific form, helps. Forcing error could simply degrade; that
  is what prediction 1 is for.
- That "velocity" is anything but per-feature phase-rate until measured.
- That seeds-as-parameters is novel in general (hypernetworks, INRs, Fourier features
  all live here). The claim is narrower: that _quantizing the seed and training the
  wave against it_ is a usable pressure in this architecture.
