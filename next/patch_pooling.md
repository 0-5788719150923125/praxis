# Patch pooling and patch boundaries are two different knobs

> Status: **pooling thread PARKED behind the VQ work** (2026-08-08). The `avg`
> experiment RAN and was killed - result in section 1 below. `abstractinator-i`
> was then repurposed onto the VQ bottleneck (codebook 16384 -> 1024, Serpent
> out) and `-j` drafts a GDN compander behind it. Pick this thread back up once
> -i/-j report, because a pooling A/B measured against a starving codebook tells
> you about the codebook. The boundary axis is untouched and still `space`.
> Sibling to [information_density.md](information_density.md), which supplies the
> density claim that the positional-pooling idea below actually tests.

## The distinction that keeps getting lost

Two independent choices, and almost every confusion in this thread came from
treating them as one:

- **Boundaries** - WHERE the byte stream is cut. `space` (today), `entropy`,
  `static`, `byte`. Lives in `praxis/encoders/byte_latent/patcher.py`.
- **Pooling** - HOW the bytes inside one patch collapse to a single vector.
  `max` (today), `avg`, `min`, `topk:N`. Lives in
  `praxis/encoders/byte_latent/encoder.py::pooling_downsample`.

They compose freely. A pooling change is a registry line; a boundary change is
usually a project.

## What is established (measured, this session)

- **Max-pool has a length bias.** `E[max of n]` grows like `sqrt(2 ln n)`. Under
  `space` patching n runs from 1 (a lone "a", a collapsed punctuation run, any
  control byte the patcher force-cuts on via `tokens < OFFSET`) to 10+.
  Measured on a 6/1/5-byte patch triple: `max` gives per-patch norms
  3.93/1.41/3.07, `avg` gives 1.07/1.41/1.39.
- **Magnitude is not the argument.** `avg` carries the INVERSE bias (~1/sqrt(n)),
  and `HarmonicResidualVQ.forward` RMS-normalizes onto the sphere anyway
  (`z * rsqrt(z.pow(2).mean(-1) + 1e-5)`), so only DIRECTION reaches the
  codebook. Any magnitude effect - damping or amplification - is erased there.
- **The real difference is direction stability.** A mean is an unbiased estimate
  of the patch's content direction at every n (only its variance depends on n).
  A max is an order statistic whose selected coordinates, and therefore its
  direction, shift systematically as n grows. That is what the sphere projection
  cannot undo.
- **All current pooling modes are permutation-invariant** over the bytes in a
  patch (`scatter_reduce` over dim=1). None of them can express WHERE in the
  patch a byte sat. This is the opening the positional idea below exploits.

## Pooling candidates

### 1. `avg` - RAN, KILLED, and it falsified the hypothesis behind it

The prediction written into the config was that removing max's order-statistic
direction drift would LOWER `vq_dead_frac` and raise perplexity. The opposite
happened:

| | -h `max` @ 10151 | -i `avg` @ 6475 |
|---|---|---|
| `vq_dead_frac_s0` | **0.27**, falling | **0.81**, rising |
| `vq_dead_frac_s1` | **0.097**, falling | **0.85**, rising |

BPB tracked -h, but inference coherence was visibly worse and the dead-fraction
trends point in OPPOSITE directions - a comparison that survives the step
mismatch.

**The mechanism runs backwards from the one this note originally argued.**
Mean-pooling shrinks patch-vector variance (~1/sqrt(n)), so after the RMS
projection the patch directions cluster more tightly on the sphere and most
codebook entries end up far from any data. Max-pooling's extreme selection
SPREADS them. The length-dependent direction drift flagged as a defect is
functioning as a **dispersion mechanism**, and a codebook needs spread more than
it needs an unbiased mean.

**Keep this as the lens for everything below:** the operative question for a
pooling scheme in this stack is *does it disperse or concentrate patch
directions*, not *is it unbiased*. Anything that concentrates will starve the
bank, and BPB will not tell you - the local byte decoder carries too much of it.

### 2. positional / decay-weighted pooling - the best idea here, needs code

`framing.tex` (fig:mtp-window caption) claims, in print: *"Density is drawn per
patch - heaviest at a patch's first byte, decaying through its predictable
tail."* No permutation-invariant statistic can represent that. A pooling that
weights by within-patch offset - first-byte selection, or `exp(-t/tau)` on the
offset, the same shape as the `decay_bias` field - is a **direct, falsifiable
test of a claim the paper already makes**, which makes it a better experiment
than any further shuffling of order statistics.

Implementation: `patch_ids` plus a within-patch offset (`arange(seq) - ` the
patch's start index, available from `cumsum(patch_lengths)`) gives the weights;
then a weighted `scatter_reduce` sum. Small, but real code.

**When:** after -i, regardless of -i's outcome. This one is worth running on its
own merits because it tests the paper, not the pooling.

### 3. `topk:N` - already implemented, correct, but subtler than it looks

`topk_mean_pooling` takes the mean of the top-k values per patch. Attractive in
principle: with fixed k the count is **length-independent**, so it removes the
`sqrt(2 ln n)` bias while KEEPING extreme selection - arguably the best of both.

Short patches ARE handled (verified): `valid_mask = iter_range < counts_exp` and
the divisor is `num_valid.clamp(min=1)`, so a patch with n < k bytes averages its
n real values instead of averaging in `NEG_INF`.

But read what that means. For n < k it degenerates to a **plain mean**; for
n >= k it is a **top-k mean**. So the statistic itself changes identity with
patch length - a different length dependence than max's, not an absence of one.
Under `space` patching with mean ~5-6 bytes, a k of 4 would put a large fraction
of patches on each side of that switch. That is worth knowing before reading any
result from it.

Caveat: it is a Python loop over k with a `scatter_reduce` and an `h` clone per
iteration, so it is slow.

**When:** only if you want extreme-selection without max's length bias AND -i
suggests extremes were carrying real information. Pick k below the 5th
percentile of patch length if you want one statistic rather than two.

### 4. `avg_min_max` - NOT a pooling swap, a capacity increase

Multi-mode pooling concatenates (`cat` gains an entry per mode), so this is
`3*dim` and `byte_config.dim_token_emb` must widen or `token_proj` shape-errors.
Parameter count changes, so a CE win is confounded with the extra width.

The justification is real though: min and max together are a **signed** spread,
preserving the direction a variance statistic would destroy - and if the deviation
from the standing wave is the signal, a mean alone discards it.

**When:** only if -i REGRESSES, i.e. max's extreme selection was carrying
information a mean averages away. Then run it against -i, in its own config,
and expect to argue about the width.

### 5. `min` alone - don't

`min` is `-max(-x)`: the most NEGATIVE coordinate, not the smallest-amplitude
one. Same order-statistic length bias, mirrored. It is not damping and cannot be
- see the RMS point above. Whether it even differs from `max` depends on the
skew of the pre-pool activations, which sandwich norm makes roughly
zero-centered. No mechanism, no experiment.

### 6. variance pooling - don't

Second-order and sign-blind: `E[(x-mu)^2]` keeps the envelope and discards the
direction. In a coupled-wave model phase IS the information, so this is the
wrong projection. Use min+max (signed) if you want spread.

## From the CV pooling literature (surveyed 2026-08-08)

A filter to apply before importing any of these: they are almost all designed
for a **fixed 2D grid downsampled by a stride**. Our problem is a
**variable-length 1D set with content-adaptive boundaries**, collapsed by
`scatter_reduce`. Methods whose content is the 2D geometry do not transfer;
methods whose content is *what statistic to take* or *how to handle variable
size* often do.

### 7. spectral pooling over WITHIN-PATCH position - strongest import

DFT along the byte axis inside a patch, keep the lowest k coefficients as the
patch vector. Three properties make this the best item on the list:

- **Position-aware.** Every current mode is permutation-invariant over the bytes
  in a patch. A DFT is not, so this is the one import that can represent the
  paper's own "heaviest at a patch's first byte, decaying through its
  predictable tail" claim (`research/framing.tex`, fig:mtp-window). It
  subsumes the decay-weighted idea in section 2 and is strictly more general.
- **A strict generalization of `avg`.** The k=1 (DC) coefficient IS the mean, so
  k>1 adds positional structure on top of the mean rather than replacing it.
- **On-theme.** Coefficients over within-patch position are literally a
  standing-wave decomposition of the patch interior, which is the same object
  the bottleneck already builds over the FEATURE axis.

Machinery partly exists: `praxis/encoders/basis.py::separable_harmonic_matrix`
already builds a 2D standing-wave basis over (K position, E feature). It is
currently used only by the CALM codec, and it assumes a FIXED chunk size, so
variable patch length needs either resampling to a common k or per-length
handling. That is the real work.

Open question, and it matters given section 1: does a low-pass over position
disperse or concentrate? Keeping k coefficients is a summary, and summaries
concentrate. The dispersion may have to come from k being large enough to keep
the discriminative structure. **Instrument `vq_dead_frac` from step one.**

### 8. SPP-style multi-scale positional bins - conceptually the right shape

Spatial Pyramid Pooling exists to produce a **fixed-length output from a
variable-size input**, which is precisely the patch problem stated in CV terms.
The 1D version: pool the whole patch, then halves, then quarters, and
concatenate. Position-aware (the bins ARE positional), variable-length-native by
construction, and much simpler to implement than the DFT.

Cost is the same as `avg_min_max`: concatenation widens the output, so
`byte_config.dim_token_emb` must widen or `token_proj` shape-errors. And short
patches degenerate - a 1-byte patch has nothing to bisect - so it inherits the
`topk` problem of the statistic changing identity with length.

**When:** as the cheap stand-in for section 7 if the DFT looks like too much
work. Same falsifier.

### 9. MAM (max-average-min) - prior art for section 4

The survey's "MAM pooling" is `avg_min_max` under another name, which is useful
confirmation the combination is not eccentric. It also strengthens the case
relative to section 1's finding: max contributes the dispersion, avg contributes
the unbiased direction, and min makes the spread signed. Same capacity caveat as
section 4.

### 10. rank-based / weighted pooling - the family, not a method

"Assign weights inside the window rather than treating activations uniformly" is
the general family that CONTAINS the decay-weighted idea (section 2), SPP
(section 8), and **BLT's own cross-attention pooling** - learned per-byte
importance, which is what the reference implementation actually ships and what
`cross_attn_encoder=False` / the `cross_attn_mask()` stub leave unimplemented
here. If any of this thread is worth real engineering, that is the target,
because it is both the general case and the published one.

### 11. compact bilinear pooling - plausible on dispersion, expensive

Second-order statistics via outer products. Note this escapes the objection in
section 6: cross terms `x_i * x_j` carry sign, so unlike variance it is not
phase-blind. Higher-order statistics are more discriminative, which is the
dispersion property section 1 says the bank needs.

Against it: even the compact (Count/Tensor Sketch) forms project to thousands of
dimensions, which is absurd against `hidden_size` 111 and `latent_dim` 55. And
it is still permutation-invariant, so it does nothing for the positional gap.
**Low priority.**

### 12. blur / max-blur pooling, strip pooling - do not transfer

Blur pooling's anti-aliasing argument presumes downsampling by a **regular
stride**; our boundaries are content-adaptive and the reduction is a scatter,
so there is no aliasing structure to correct. Strip pooling's elongated `1xN` /
`Nx1` kernels are 2D-specific and have no 1D analogue. Skip both.

## Boundary candidates

- **`entropy`** - the principled endpoint and BLT's own best result. But it is a
  project, not a swap: an entropy model trained jointly (`calculate_entropies`),
  a per-batch threshold search (`_find_safe_threshold`), and it is the one mode
  the test suite does not exercise. Do it when boundaries are the week's subject.
- **`static`** - fixed stride. FIXED as of 2026-08-07 (`_static_patching` ignored
  `include_next_token` and asserted on every forward). Now available for
  ablation, unused. Note the literature is against it - SpaceByte (arXiv:2404.14408)
  exists precisely because fixed boundaries cut mid-word, and that mechanism is
  scale-independent, so the "they tested overparameterized models" escape hatch
  probably does not hold.
- **`max_patch_length` capping** - `PatcherConfig` has the field and
  `_postprocess_patch_lengths` honours it, but `ByteLatentEncoder` does not
  appear to pass it through. Cheap middle ground: keep content-aligned cuts,
  bound the tail.
- **minimum-patch-length merging** - merge 1-byte patches into a neighbour.
  Attacks the "a" problem directly while keeping content alignment. Not
  implemented; no prior art checked.

## What to read

CE against -h at equal steps is the headline, but the VQ telemetry is where the
mechanism either earns its explanation or loses it. If the length bias was
consuming codebook capacity, `vq_perplexity` should RISE and `vq_dead_frac_s*`
should FALL. **CE improving while that telemetry sits flat means the win came
from somewhere else and the story is wrong even though the number is better.**
`vq_resets_s*` slope is the stability read.

## Ranked, if this thread is resumed

1. **Section 7** (spectral over within-patch position) - the only import that
   tests a claim the paper already makes in print.
2. **Section 10** (cross-attention pooling) - the general case AND what BLT
   actually ships; the biggest engineering, the strongest prior.
3. **Section 8** (SPP bins) - cheap stand-in for 7.
4. **Section 4 / 9** (`avg_min_max` / MAM) - only against a *healthy* codebook,
   and only after the capacity confound is handled.

Everything else on the list is either rejected above or subsumed.

## Falsifier for the whole line of thinking

If `avg`, positional, and `topk` all land within noise of `max`, then how bytes
collapse into a patch vector is simply not where this model's capacity is going,
and the pooling axis should be closed. The honest next place to look would be the
boundary axis (entropy) or the bottleneck's spectral budget
(`bottleneck_ratio`, currently 0.5), not further pooling variants.

One caveat on running ANY of this soon: section 1's result was measured against
a codebook that was 50% of the model and under 3% occupied. A pooling A/B under
those conditions is measuring the codebook. Re-baseline after -i/-j.

## Anchors

`praxis/encoders/byte_latent/encoder.py::pooling_downsample`,
`patch_reduce`, `topk_mean_pooling`; `praxis/encoders/quantization/harmonic_bottleneck.py`;
`research/framing.tex` (per-patch density, fig:mtp-window);
[information_density.md](information_density.md) (the hum/chirp claim is about
the CONTEXT window, not the patch interior - do not transfer it);
`experiments/abstractinator-i.yml`.
