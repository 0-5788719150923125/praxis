# The knight's move: shear channels as cheap content scoring

> Status: **design note only** (2026-09-11). Nothing built, no run. Every number
> below is an offline fit against synthetic targets, not a training result.
> Owns the open question [kaleidoscope.md](kaleidoscope.md) records as "selection
> stays structural ... induction wants a discrete pointer" - how that block gets
> *some* content into its scores without paying `T^2 * d` for a QK product.
> Sibling to [surrogate_geometry.md](surrogate_geometry.md), where `const [t, t]`
> started, and to [differential_mirrors.md](differential_mirrors.md), which owns
> the other half of the same question: this note adds content to the logits,
> that one changes how logits become weights.

## Where this came from

The shape of a knight's move in chess, and the question of whether an L-shaped
offset in the score matrix could do something a diagonal one could not. The
guess attached to it was:

> a delta rule shifted down by two and to the right by one is both causal while
> ALSO encoding the DIFFERENCE in that change over time, in a way that a simple
> 1 down and 1 right shift would not (because that would be invariant)

That guess is correct, and "invariant" is the right word. The diagonal shift is
exactly the symmetry of relative position: a matrix constant along its diagonals
is a Toeplitz matrix, and a Toeplitz matrix *is* a relative-position bias. So a
recurrence built on the `(1, 1)` offset sits inside the family of things
ALiBi, RoPE-bias, Toeplitz Neural Networks ([arXiv:2305.04749](https://arxiv.org/abs/2305.04749))
and this repo's own frozen mirrors already supply for free. The knight offset is
the smallest one that does not. The rest of this note is that statement made
precise, measured, and turned into three builds.

## The one line

Compute part of the score matrix by **integrating along a sheared ray** instead
of by a pairwise product. One scalar feature per token per channel, swept along
a `(p, q)` direction with a decay, gives a full-rank content-dependent score
matrix at `O(C * T^2)` - no `d` factor - and the choice of `(p, q)` decides
*which kind* of content structure it can hold.

```
per channel c:   a_c = w_a^c . x        b_c = w_b^c . x        (two scalars per token)
shear scan:      G_c[i, j] = alpha_c * G_c[i - p, j - q] + a_c[i] * b_c[j]
unrolled:        G_c[i, j] = sum_n alpha_c^n * a_c[i - p n] * b_c[j - q n]
scores:          S[i, j] = sum_k w_k(x_i) M_k[i, j]  +  sum_c v_c(x_i) G_c[i, j]
                           \___ frozen mirrors ___/     \___ live channels ___/
```

Every index the scan touches is `<= i`, so it is causal. `(p, q) = (1, 1)` is
the diagonal; `(2, 1)` and `(1, 2)` are the knight's moves.

## Why the offset vector decides everything

Fix `a`, `b` and look at what the family `G = sum_n alpha^n Shift_{(p,q)}^n (a (x) b)`
can be.

**(i) Axis-aligned offsets buy nothing.** For `(p, 0)`,
`Shift^n (a (x) b) = (Z^{pn} a) (x) b`, so `G = (sum_n alpha^n Z^{pn} a) (x) b`
is still **rank 1** - it is a rank-1 score wearing a scan. Same for `(0, q)`.
Measured rank at `T = 90`: `(1,0) -> 1`, `(0,1) -> 1`, `(3,0) -> 1`.

**(ii) The diagonal breaks rank but leaks into relative position.**
`Shift_{(1,1)}` fixes exactly the Toeplitz matrices, so the recurrence's
invariant subspace *is* the relative-position family. Concretely
`G[i, j]` is an EMA, along time, of the product `a_t b_{t-l}` at fixed lag
`l = i - j` - a running autocorrelation, which drifts toward a stationary
function of lag. Measured: rank 88 of 90, but **41% of its energy in the
back half of the sequence is exactly a function of lag**, i.e. redundant with
a positional bias.

**(iii) The knight breaks both.** `Shift_{(p,q)}` fixes the functions
`f(i, j) = g(q i - p j)`. For `(2, 1)` that is `g(i - 2j)`: constant along rays
of slope `1/2`, a *parallel* family, where the mirrors' ratio structure is a
*concurrent* family through the origin. Setting `i = j` forces `g` constant, so
**the knight's invariant subspace meets the Toeplitz family only in constants.**
Measured non-Toeplitz energy of the invariant subspace: `(1,1) -> 0.0000`
(entirely Toeplitz, as claimed), `(2,1) -> 0.9718`, `(1,2) -> 0.9728`.
Measured non-stationary energy of `G` itself: `(1,1) -> 0.59`, `(2,1) -> 0.986`.

**(iv) The knight's moves are the minimal offsets with both properties.** At
`|p| + |q| <= 2` the only candidates are `(1,0)`, `(0,1)` and `(1,1)`, covered
above. At weight 3, `(3,0)` and `(0,3)` are axis-aligned, leaving exactly `(2,1)`
and `(1,2)`. There is nothing smaller. This is the whole content of the
intuition that started the note.

**(v) The price.** The knight's rays are half as many, so its rank ceiling is
`ceil(T/2)` where the diagonal's is `T`. Measured 48 vs 88 at `T = 90`. The
knight spends less rank, but spends all of it on structure the frozen dictionary
cannot already express.

## What a shear channel actually measures

A `(p, q)` scan correlates the `a` stream against the `b` stream **under a
rational time dilation `q/p`**. The diagonal (`1/1`) is correlation under pure
translation - autocorrelation, the Fourier/convolution axis. The knight (`1/2`)
is correlation under a 2:1 dilation - the Mellin/wavelet axis, self-similarity
across *scale* rather than across *position*.

Two consequences worth taking seriously:

- **The offsets are indexed by the rationals.** Primitive `(p, q)` ordered by
  `p + q` is the Stern-Brocot enumeration: `1/1`, then `1/2` and `2/1` (the two
  knights), then `1/3`, `3/1`, `2/3`, `3/2`. A dictionary of shear channels is a
  dictionary of **tempo ratios**, and simple integer ratios are exactly the
  consonant ones. `2:1` is the octave; `3:2` is the fifth. This is not a pun
  imported from [harmony.md](harmony.md) - it is the same arithmetic arriving in
  the score matrix, and it means "which shears" has a principled answer
  (low-denominator first) rather than a hand-picked one.
- **`(r, 1)` is the cross-level alignment operator for a hierarchy.** An
  Abstractinator level running at compression ratio `r` relates to the level
  below by exactly this dilation. A shear channel at `r` is the operator that
  says "attend to the position in the lower stream that corresponds to mine".
  A single-level model with shear channels can discover a compression ratio it
  was never given.

## Idea 1 (the build): `kaleido_knight` - live entries in a frozen dictionary

Kaleidoscope's dictionary is `N` frozen mirrors blended by a free signed
per-token turn. Add `C` **shear channels** as extra dictionary entries. They
differ from mirrors in one respect only: a mirror is a fixed function of
position, a shear channel is a scan over content. The same turn blends both, so
nothing about the routing story changes.

```
turn:     w = beta_d + m * tanh(W_turn x)            now [B, T, H, N + C]
mirrors:  einsum("bihk,kij->bhij", w[..., :N], M)     frozen, positional
shear:    einsum("bihc,bcij->bhij", w[..., N:], G)    live, content
```

In the metaphor: the mirrors are the frozen glass, and this is the object
chamber - the loose beads that actually move.

**Which shears.** `(1, 1)`, `(2, 1)`, `(1, 2)`, two channels each. The diagonal
is **not optional** - see the measurement below, it is what holds the
previous-token head.

**Cost.** Each `(shift, channel, step)` triple costs one `T^2/2` outer product,
against `d * T^2/2` for `QK^T`. Six channels on the full recurrence is 6 terms
against `d = 90`: **15x cheaper than a QK product**, 1.5x the existing mirror
einsum. The truncated ladder below is 18 terms: 5x cheaper than QK, 4.5x the
mirrors. Parameters: `2 * C * hidden` for the `a`/`b` projections, plus `C`
decays. At `-i` dimensions that is under 3.5k, against 77,714 for the block.

**How to materialize it.** Three options, in increasing order of effort:

1. **Truncated unroll**: `K` shifted outer products per shift. Fully parallel,
   no scan, ~10 lines. Reaches lag `K * p` only.
2. **Dyadic ladder** (recommended first build): shifts `(2^{s+1}, 2^s)` for
   `s = 0..log2(T)`, each unrolled `K = 3` steps. All rungs share the invariant
   `i - 2j` (since `2^s (i - 2j)` is the same null space), so the whole ladder is
   one dilation-2 channel sampled at every scale. Reaches the full sequence,
   stays parallel, and mirrors the `zoom_ladder` the block already has.
3. **Full chunked scan**: exact, `O(C T^2)`, log depth. The naive
   `cumsum(c / alpha^n) * alpha^n` overflows on long rays; chunk it, the same
   shape `HarmonicField._fast_retrieve` already uses for its segmented delta
   memory.

**The mod-3 trap, and it is real.** The two causal knight moves generate the
index-3 sublattice `{(p, q) : p + q = 0 mod 3}`. Verified: starting from the
origin and applying `{(2,1), (1,2)}` only ever reaches cells with
`(i + j) mod 3 == 0`. A recurrence driven by knight moves alone therefore splits
the score matrix into **three non-communicating interleaved lattices**. Adding
`(1, 1)` restores all three classes (verified). This is a second, independent
reason the diagonal channel has to be in the dictionary.

**Causality, including the objection.** The backward `(2,1)` ray from a cell can
leave the causal triangle - from `(i, i)` the predecessor `(i-2, i-1)` has its
key ahead of its query. This does not matter. The unrolled form only ever reads
`a` at `i - 2n` and `b` at `j - n`, both `<= i`, so the score is a function of
the prefix regardless of the path the *cell* index takes. Mask once at the end,
and treat out-of-triangle cells as 0 rather than `-inf` inside the scan.

**The bonus: this gives kaleidoscope the decode path it does not have.** The
note currently records "No KV cache path. The scores are recomputed from the
frozen dictionary each forward, so decode is `O(T^2)` per step." The recurrence
form fixes that for the live half: row `i` is
`alpha * shift(row_{i-2}) + a_i * b`, so caching the last two score rows plus the
`b` stream gives `O(T)` work and `O(C T)` state per step - **against `O(T d)`
for an ordinary KV cache**, roughly 20x smaller at `C = 6, d = 90`. Verified
numerically that the row-at-a-time recurrence reproduces the unrolled form
exactly (max abs err 1.9e-6 at `T = 40`).

**Identity at init, and the two-stage unlock.** Zero-init the shear half of
`turn_static` and `W_turn`, random-init `a`/`b`. Then `S` is exactly zero at step
0 - the block's existing identity state - and `dL/dv_c = <dL/dS, G_c>` is
nonzero, so the blend moves first and `a`/`b` unlock once it has. Identical
asymmetry to `facet_u`/`facet_v`, and a gradient audit will flag it the same way.

**The falsifier is one number.** `kaleido_shear_share`, the share of blend
magnitude on the live entries. At 0 the model declined the content term, the
frozen dictionary was sufficient, and that is a publishable finding about this
architecture rather than a failed experiment. Report it alongside
`kaleido_turn_modes` computed over `N + C`.

## Idea 2 (do this first - it is an afternoon, and it decides the rest)

Before building anything, **measure whether real attention in this repo is
stationary.** The whole argument above is that the frozen mirrors already own
the Toeplitz part of the score matrix and a content term should spend itself
elsewhere. That is an assumption, and it is directly measurable on an existing
`-h` / arc run.

On a sample of realized attention logit matrices `S`, report:

- `attn_stationary_share` - energy of the best fit `h(i - j)` over total energy
  on the causal triangle. This is the fraction of attention a pure
  relative-position bias could have produced. Report per depth.
- `attn_shear_share[r]` - the same quantity after projecting out the fit for
  dilation `r`, for `r` in `{1, 2, 1/2, 3/2}`. Which tempo ratio, if any, real
  heads already use.

This is `measure before you build`, the same discipline
[[project_objective_conflict_metric]] applies to gradient surgery. It costs one
forward pass and a least-squares fit. If `attn_stationary_share` comes back at
0.9, the mirrors are already doing almost everything and Idea 1 is a small
correction; if it comes back at 0.3, the missing content term is the binding
constraint and the note's premise is confirmed. Either way it is a figure.

## Idea 3 (nearly free): a shear coordinate for the mirror dictionary

`MIRROR_COORDS` currently offers `ratio` (query fraction, key fraction) and
`split` (half ratio, half query-fraction x log lag). Add `shear`: index the
second axis by `sign(u) * log1p(|u|)` with `u = i - r * j` and **`r` a learned
per-mirror scalar**.

- The existing lag coordinate is exactly `r = 1`, so this generalizes rather
  than replaces it, and `r` is *learned*, which keeps it on the right side of
  [[feedback_no_hyperparameter_tuning]] - no hand-picked ratio.
- `r != 1` gives the dictionary the parallel slope-`1/r` ray family, which
  neither the ratio nor the lag coordinate spans.
- The sign handling is the only new wrinkle: `u` is negative wherever
  `j > i / r`, which for `r = 2` is most of the near-diagonal band.
- Read `kaleido_shear_r`, the learned ratios. If they all sit at 1 the model
  asked for lag and the coordinate bought nothing.

This is a few dozen lines inside `_lag_grid` / `_resample` and does not touch the
routing at all. It is the cheapest way to test whether the *geometry* wants a
shear, independent of whether the *scoring* does.

## Idea 4 (considered, not recommended)

**Fold the sequence to a `W`-wide board and attend along knight offsets**
(`lag = 2W +- 1`, `W +- 2`). The honest arguments: at equal degree the knight
graph has roughly half the diameter of the king graph, so information mixes in
half as many layers; and knight offsets are precisely the ones a row/column
factorization cannot produce, which is what Sparse Transformer
([arXiv:1904.10509](https://arxiv.org/abs/1904.10509)) builds its strided pattern
out of. The reason not to build it: a *dilated* pattern reaches anywhere in
`O(1)` or `O(log T)` hops, so halving the diameter of a local pattern loses to
prior art that is already in every long-context model. The interesting residue
is only that knight distance is non-monotonic in `|i - j|` near zero, which is a
periodic prior in disguise - and the log-lag mirrors already give the block a
better-behaved version of that.

## The measurements

Offline fits, `T = 90`, relative squared residual on the causal triangle, lower
is better. "bilinear" is a rank-`C` `QK^T` fit by Adam **on the same mask**, not
an SVD, so the comparison is fair. Targets: an induction pattern (attend after
the previous occurrence of the current token), a previous-token head, a
half-rate alignment (`j = i / 2`), and a genuinely rank-8 `QK^T` matrix.

| method | params | induction | prev-token | half-rate | rank-8 QK |
| --- | --- | --- | --- | --- | --- |
| rank-4 bilinear | 8T | 0.410 | 0.422 | 0.642 | 0.273 |
| rank-8 bilinear | 16T | **0.073** | 0.197 | 0.429 | **0.000** |
| `(1,1)` x4 | 8T | 0.404 | **0.000** | 0.483 | 0.271 |
| `(2,1)` x4 | 8T | 0.404 | 0.220 | **0.000** | 0.272 |
| `(1,1)+(2,1)` x2 | 8T | 0.404 | 0.000 | 0.000 | 0.270 |
| **`(1,1)(2,1)(1,2)` x2** | 12T | 0.197 | **0.000** | **0.000** | 0.098 |
| dyadic ladder, 6 shifts x 3 steps | 12T | 0.196 | 0.065 | 0.209 | 0.094 |

What this says, stated against itself:

- **Shear is complementary to low-rank bilinear, not a replacement.** It wins by
  a mile on the structural patterns and loses to rank-8 on induction. Induction
  needs high-dimensional token identity matching, and `C` scalars per token
  cannot do exact matching however cleverly they are swept. Any claim that this
  replaces QK is wrong.
- **Every shear variant beats rank-4 bilinear on a genuinely bilinear target**
  (0.27 vs 0.27 for one shift, 0.098 for three). So the family is a competent
  generic approximator of real attention too, at comparable parameter count and
  a fraction of the FLOPs.
- **The `(1,1)` channel is mandatory.** Knight-only reaches 0.220 on the
  previous-token head, the single most load-bearing pattern in a language model,
  where the diagonal channel reaches 0.000. A pure-knight design would lose it.
- **The truncated dyadic ladder is close enough to the full scan to build
  first** (0.196 vs 0.197 on induction, 0.094 vs 0.098 on rank-8) at the cost of
  a real regression on `prev-token` (0.065 vs 0.000) and `half-rate` (0.209 vs
  0.000), because 3 steps per rung cannot reach every lag exactly.

These are offline fits with `a` and `b` free to be anything. In a model they are
projections of `x` and the blend is a per-token router, so treat the table as an
**upper bound on what the family can express**, not a prediction of what it will
learn.

## What would falsify the whole note

1. `attn_stationary_share` near 1 on the baseline (Idea 2). Then the frozen
   mirrors already cover attention and there is no content gap to fill.
2. `kaleido_shear_share` decaying to 0 in training. The model was offered
   content scoring at 1/15 the price of QK and declined it.
3. A `kaleido_knight` run that matches `kaleido_split` on loss. The shear
   channels cost FLOPs and parameters; parity means they bought nothing, and the
   honest conclusion is that the block's limit is the smooth router, not the
   absent content term - which is what `kaleidoscope.md` already suspects
   ("a smooth router over a frozen dictionary emits a smooth attention row
   however well the geometry resolves position, and induction wants a discrete
   pointer").
4. The induction column never improving. If `-i` plus shear still cannot do
   induction, the block needs a pointer mechanism, not a better score field, and
   that is a different note.

## Open questions

- **Is `alpha` per channel enough?** A learned decay per channel fixes the
  ray length globally. A per-token gate (`alpha_i = sigmoid(w . x_i)`) would make
  the scan a proper gated recurrence and put it in Gated DeltaNet's family
  ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)), at the cost of the
  closed-form unroll. Not obviously worth it at `C = 6`.
- **Which rationals, beyond the first three?** Stern-Brocot order says `3/2` and
  `2/3` next. Nothing here has measured whether they add anything over `2/1`,
  and a sweep is the calibration the note does not have.
- **Interaction with `MERGE_OPAQUE`.** The `a`/`b` projections are ordinary
  Linears inside a subtree the block declares opaque to SMEAR. Same known loss
  the note already records for `value`/`gate`/`output`.
- **Does the shear score compose with `block_ids`?** No, and neither does the
  existing block. Packed documents would bleed across boundaries through the
  scan in a way a masked softmax would not catch, because the scan runs before
  the mask. This is a harder version of the gap `kaleidoscope.md` lists, and it
  should be fixed at the same time as that one.
- **Prior art check came back empty on the specific construction.** Sparse
  Transformer's strided pattern, Toeplitz Neural Networks, Synthesizer, and
  DeltaNet's WY form are the neighbours; none of them builds the score matrix by
  a non-diagonal shifted scan, and none of them parameterizes a score family by
  a tempo ratio. Worth one more search against "sheared attention" and
  "dilated correlation attention" before any writing that claims novelty.
