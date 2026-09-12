# Differential mirrors: signed blending on the far side of the softmax

> Status: **design note only** (2026-09-11). Nothing built, no run. Every number
> here is an offline measurement on synthetic targets or a closed-form argument,
> not a training result. Companion to [knights_move.md](knights_move.md) (which
> owns the content-scoring half) and to [kaleidoscope.md](kaleidoscope.md)
> (which the code has since moved past: the zoom ladder, the N sweep and the
> `mix_norm` control are all in the registry and none of them is in that note).
>
> **The one thing to take from this note**, if nothing else: there is a
> closed-form reason why `kaleido_turn_modes` decays, it is a property of mixing
> *before* the softmax, and it gets worse exactly as sequences lengthen. That is
> the pattern `-l` measured. See "Why the router collapses" below. The
> reachable-set argument for the differential, by contrast, is small: see the
> correction in the section before it.

## Where this came from

Differential Transformer, and the question of whether its two-map subtraction is
novel inside kaleidoscope, plus four ways of crossing it with the knight's move:
a difference of two mirrors, two L-shapes on one mirror, a difference of the two
knight offsets, and "10 or 50 samples from the same matrix, selected at
log-normal random".

Short version: **two of the four crossings earn their place and two do not, the
differential itself buys less reach than it first appeared to, and the strongest
reason to build any of it is a gradient argument rather than an expressivity
one.** The two that earn it are a knight stencil applied inside each mirror
(best at zero cost) and log-normal tap offsets (which beat fresh mirrors on rate
structure and beat a regular ladder outright). The sampling idea also turns out
to have independent support in the literature, from a paper written specifically
to fix what differential attention breaks.

**Read the measurement sections rather than an earlier summary of them.** Two
tables in this note were rewritten after their first draft, both because a
log-space linearization was used to fit a class that is not linear, and both
reversals went against what the first draft claimed.

## Getting the reference right first, because it changes the answer

DIFF Transformer ([arXiv:2410.05258](https://arxiv.org/abs/2410.05258), ICLR
2025) subtracts **softmax outputs**, not logits:

```
DiffAttn(X) = ( softmax(Q1 K1^T / sqrt(d))  -  lambda * softmax(Q2 K2^T / sqrt(d)) ) V
lambda      = exp(l_q1 . l_k1) - exp(l_q2 . l_k2) + lambda_init
lambda_init = 0.8 - 0.6 * exp(-0.3 * (l - 1))          # l = layer index
head_i      = (1 - lambda_init) * RMSNorm(head_i)
```

Confirmed against `microsoft/unilm/Diff-Transformer/multihead_diffattn.py`,
which implements the split as **pairs of heads**: reshape the weights to
`(B, H, 2, T, T)` and subtract along the new axis. So DIFF is 2H heads combined
pairwise with a learned negative coefficient, and the halved head count is what
keeps the parameter count matched.

Two things from their own ablation table that matter more than the headline:

- The gain is **small**: 1.4B validation loss 3.087 (Transformer) -> 3.062.
- Without GroupNorm it is **3.122, which is worse than the baseline**. The
  normalization is not a detail; it is the difference between a win and a loss,
  because the two subtracted maps leave heads with wildly different statistics.

Read that as: the mechanism is real, the margin is thin, and it does not survive
being dropped into a block that has no output normalization. Kaleidoscope has
none - it has a SiLU gate, which can absorb a scale but not a per-head variance.

## The four crossings, measured properly

**The first pass of this section used a log-space proxy and got this wrong too.**
Refit with a per-row signed blend trained by Adam on the true objective
(relative L2 to the target distribution), `R = 64`, `T = 256`, `K = T/R = 4`.
All dictionaries have 8 entries except the 16-mirror control. Lower is better,
**bold is best among the 8-entry dictionaries**.

| dictionary | induction | prev-token | half-rate | local band |
| --- | --- | --- | --- | --- |
| 8 iid mirrors (the control) | 0.441 | 0.482 | 0.378 | 0.132 |
| 8 mirrors, each **knight-stencilled** at canonical scale | **0.434** | 0.492 | 0.237 | **0.117** |
| 1 mirror + 7 **log-normal** knight taps | 0.501 | **0.448** | **0.223** | 0.435 |
| 1 mirror + 7 **dyadic** knight taps | 0.554 | 0.717 | 0.331 | 0.400 |
| 8 **pairwise differences** of mirrors | 0.452 | 0.528 | 0.414 | 0.234 |
| 4 mirrors + 4 taps | 0.465 | 0.518 | 0.254 | 0.196 |
| *16 iid mirrors (scale control)* | *0.367* | *0.475* | *0.282* | *0.029* |

Taking the four crossings in turn:

**A difference of two different mirrors: no.** Worse than the mirrors it is built
from on every target. The turn is free and signed, so `w = [+1, -1, 0, ...]`
already *is* that difference; building the dictionary out of differences only
throws away a dimension (the cyclic difference map has a one-dimensional kernel)
and the numbers show exactly that cost.

**Two L-shapes on one mirror, at a regular ladder: mostly no.** The dyadic taps
lose on three of four targets, badly on prev-token (0.717 against 0.482).

**Two L-shapes on one mirror, at log-normal offsets: yes, and specifically
better than the regular ladder.** This is the one that was dismissed on the bad
proxy and should not have been. Log-normal taps beat **8 fresh iid mirrors** on
prev-token (0.448 against 0.482) and beat them by 41% on half-rate (0.223
against 0.378), while losing badly on induction and the local band. They beat
the dyadic ladder on three of four. The trade is interpretable: **a shifted copy
of a field IS a rate relation**, so a tap dictionary is good at alignment and
rate structure and bad at dense local structure, where fresh draws win. The
heavy tail is what makes it work - some taps land inside the resample's
correlation length and resolve fine structure, some land well outside it and
decorrelate. A regular ladder cannot cover both.

**A stencil on each mirror (the difference of two offsets on the same field):
yes, and this is the one to build.** Best or tied-best on three of four targets
among the 8-entry dictionaries, at **zero extra parameters and zero extra
dictionary slots** - it keeps `N` fresh draws and adds the shift relation on top
of each, rather than spending slots on taps. Half-rate 0.237 against 0.378 is
most of the tap dictionary's gain without its local-band collapse (0.117 against
0.435). The mixed 4+4 dictionary sits between on everything, which says the
right place to spend a shift is *inside* each entry, not *beside* it.

**One caveat that now carries weight.** The stencil measured here is the
canonical-scale one, `M - shift(M, 8K, 4K)`. The literal `(2, 1)` knight is a
different and worse operator at this resolution, for the reason the sharpening
section gives. The earlier "a stencil changes nothing" reading came from both
the bad proxy *and* a live-units offset.

**`N` still dominates**, but less uniformly than the proxy suggested: 16 iid
wins induction and the local band outright and is essentially tied on prev-token
(0.475 against 0.482), where a log-normal tap dictionary of half the size beats
it (0.448). Whatever limits prev-token is not dictionary size.

## The result that does not hold up, and the one that replaces it

**First pass, and it was wrong.** Fitting the two function classes

- pre-softmax, what the block does today: `softmax( sum_k w_k M_k[i, :] )`
- post-softmax, what DIFF does: `sum_k w_k softmax( M_k[i, :] )`

against the same targets appeared to show post-softmax reaching strictly further
at every `N`, with the gap widening. It does not. That table compared an **exact
least squares** for the post-softmax class (which is linear in `w`) against a
**log-space linearization** for the pre-softmax class (which is not), and the
linearization is a bad proxy: at `N=8` on the previous-token head it reported
0.957 where the true optimum is 0.486. Refitting both classes by gradient
descent on the identical objective reverses the ordering completely.

Relative L2 to the target distribution, per-row free signed blend, **both
classes fitted by Adam on the same loss**:

| N | target | pre-softmax | post-softmax | both together |
| --- | --- | --- | --- | --- |
| 4 | induction | 0.603 | 0.846 | **0.580** |
| 4 | prev-token | 0.852 | 0.960 | **0.850** |
| 4 | local band | 0.518 | 0.778 | **0.505** |
| 8 | induction | 0.494 | 0.740 | **0.445** |
| 8 | prev-token | 0.486 | 0.918 | **0.479** |
| 8 | local band | 0.131 | 0.623 | **0.126** |
| 16 | induction | 0.429 | 0.630 | **0.363** |
| 16 | prev-token | 0.470 | 0.849 | **0.448** |
| 16 | half-rate | 0.388 | 0.815 | **0.377** |
| 16 | local band | 0.028 | 0.447 | **0.027** |

Three things this actually says:

1. **`kaleidoscope.md` is right about pre-softmax, and by a wide margin.**
   Log-linear pooling over a frozen dictionary is far more expressive than a
   signed combination of the same dictionary's softmaxed rows. Post-softmax
   alone is roughly twice the residual on every target. The "gap in the
   argument" the first draft of this note claimed does not exist.
2. **The post-softmax term is worth having as an ADDITION, not a replacement.**
   `both` beats `pre` at every `N` and every target, consistently but modestly:
   1% to 15% of the remaining residual, largest on induction (0.429 -> 0.363 at
   `N=16`). That is the right order of magnitude for DIFF's own reported margin
   (3.087 -> 3.062), and it is what the mechanism should be expected to buy.
3. **A frozen dictionary reaches much further than the proxy suggested.** At
   `N=16` pre-softmax the local band is essentially solved (0.028) and induction
   is at 0.43. The earlier "no frozen dictionary can express these patterns"
   reading was an artifact of the same bad fit. `N` still dominates: induction
   0.603 -> 0.494 -> 0.429 going 4 -> 8 -> 16.

So the reachable-set case for the differential is **small and additive**, and
the honest build is `S_A` pre-softmax exactly as today plus a subtracted second
branch, never a replacement of the mixing rule. The stronger case for building
it at all is the next section, which is about gradients rather than reach.

## Why the router collapses, and why this fixes it

`abstractinator-m` records the measurement that motivates everything above:

```
kaleido_turn_modes   0-5k   5-10k  10-16k  16-24k  24k+
-i (T<=65)           3.157  2.860  2.979   2.821    -
-k (T<=65)           3.040  3.010  2.842   2.753    -
-l (T<=257)          3.422  2.873  2.691   2.439   2.360
```

The reading in that file is redundancy: four iid draws are too alike, so the
zoom ladder makes them differ by construction. That may well be true. **There is
also a closed-form mechanism that predicts this exact pattern and has nothing to
do with what the mirrors contain.**

Differentiate a pre-softmax blend. With `p = softmax(sum_k w_k M_k)`:

```
d p_j / d w_k  =  p_j * M_k[j]  -  p_j * (p . M_k)
```

As the blend sharpens and `p` concentrates on one key `j*`, every one of these
`N` gradient vectors becomes a multiple of the same vector, differing only by
the scalar `M_k[j*]`. The Jacobian of the attention row with respect to the
router's `N` weights **degenerates to rank one**, regardless of `N`, and its
magnitude vanishes at the same time. The mirrors stop being independently
identifiable, so the router cannot hold `N` distinct modes even when the
dictionary offers them.

Measured, `N = 8`, one query row, sweeping the blend norm:

| `\|\|w\|\|` | Jacobian eff. rank, pre | Jacobian eff. rank, post | attention support (keys) |
| --- | --- | --- | --- |
| 0.25 | 7.65 | 6.19 | 196.6 |
| 1.00 | 7.27 | 6.19 | 139.3 |
| 2.00 | 6.10 | 6.19 | 51.5 |
| 4.00 | 4.00 | 6.19 | 8.1 |
| 8.00 | 2.60 | 6.19 | 2.1 |
| 32.0 | 1.53 | 6.19 | 1.0 |

The post-softmax column is constant **by construction**: `d p / d w_k` is
`softmax(M_k)`, which does not depend on `w` at all. There is no sharpening
pressure on the router's conditioning because the router never passes through
the softmax.

Three things follow, and all three are checkable against logs that already
exist:

1. **`kaleido_turn_modes` should anti-correlate with `kaleido_turn_scale`
   within a run.** `turn_scale` is already logged and the block already
   documents it as the effective softmax temperature. If the two move against
   each other, this mechanism is operating and the redundancy story is at best
   secondary. **This is a plot, not an experiment, and it should be made before
   anything is built.**
2. **It should be worse at longer `T`,** because selecting one key among more
   requires a sharper blend. `-l` at `T<=257` falls to 2.36 where `-i`/`-k` at
   `T<=65` hold near 2.8. That is the measured pattern.
3. **`mix_norm` is a partial treatment already in the registry.** Dividing the
   mix by `sqrt(N)` holds the logit scale down, which holds the Jacobian's
   conditioning up. If `kaleido_12_norm_dropoff_always` shows higher
   `turn_modes` than `kaleido_12_dropoff_always`, that is this mechanism, not a
   capacity effect.

Honest counterweight: at low `||w||` the pre-softmax Jacobian is **better
conditioned** than the post-softmax one (7.65 against 6.19), because softmaxed
rows are non-negative and share a simplex, so they are partially correlated with
each other. Pre-softmax has the higher ceiling and no floor; post-softmax has a
lower ceiling and a hard floor. **That argues for keeping both, not for
replacing one with the other** - which is where the original "SMEAR, twice"
framing lands anyway.

## What the stencil reading is actually good for: sharpening, at one specific scale

The oracle test cannot see a fixed filter, but the softmax can. A mirror is
stored at `R = 64` and resampled to `T`, so the live field is smooth with a
correlation length of `T / R` cells. Differencing it against a shifted copy is a
directional derivative, and a derivative of a smooth field has *smaller*
amplitude, which flattens the softmax rather than sharpening it. Measured at
`T = 256`, `R = 64` (correlation length 4 cells), effective keys in the second
half of the sequence:

| stencil | amplitude vs base | effective keys |
| --- | --- | --- |
| base mirror | 1.00 | 127.0 |
| minus shift (2, 1) - the literal knight | 0.82 | 146.0 (**blurrier**) |
| minus shift (4, 2) | 1.27 | 105.1 |
| minus shift (8, 4) | 1.41 | **89.0** |
| minus shift (16, 8) | 1.38 | 91.9 |
| minus shift (64, 32) | 1.29 | 95.7 |

**The literal knight's move blurs the dictionary.** A shift of `(2, 1)` at
`T=256, R=64` lands inside the resample's own smoothing kernel, so the two taps
are correlated and the difference is small. Past about two correlation lengths
the taps decorrelate, the amplitude saturates at `sqrt(2)`, and the difference
sharpens the dictionary by **30% in effective key count**.

The design rule that falls out is concrete and easy to get wrong: **stencil
offsets must be expressed in canonical units, not live units.** A knight's move
on the canonical `[R, R]` grid is `(2 T/R, T/R)` live, and it moves with the
sequence curriculum. A hard-coded `(2, 1)` is a different operator at every
length, and at long lengths it is a no-op.

This also closes a TODO the code already carries: `MIRROR_ALPHA`'s comment says
"making mirror k band-limited to spatial frequency ~k would make the ordering
real". **A learned multi-tap stencil per mirror is a learned band-limit**, so
the pink envelope's arbitrary ordering could become a real one, with the taps
learned rather than hand-picked ([[feedback_no_hyperparameter_tuning]]).
`grid_sample` is already in `_resample`, so fractional (and therefore learnable)
offsets are nearly free to implement here.

## The sampling idea has independent support, from DIFF's own critics

"Take 10 samples, or 50, from the same QK matrix" is close to what the **Integral
Transformer** ([arXiv:2508.18387](https://arxiv.org/abs/2508.18387), 2025)
proposes, and it proposes it specifically as the repair for differential
attention:

```
phi_intg(X) = softmax( (1/S) * sum_{s=1..S} Q^s K^s^T )     # S = 8, head dim split S ways
```

Their argument against subtraction is empirical and sharp: **41% of DIFF's
attention weights on the `[BOS]` token are negative** (50% for Cog Attention,
[arXiv:2411.07176](https://arxiv.org/abs/2411.07176), which replaces softmax with
`sign(QK^T) * softmax(|QK^T|)`). Attention sinks are load-bearing, and
subtraction actively destroys them. They report 48.9% against DIFF's 47.6% and
vanilla's 47.2% at 1.2B, and find vanilla attention is better in the *lower*
layers.

**This is the single most important design constraint for building the
differential here, and it is specific to this block.** Kaleidoscope's ghost is
exactly a sink - a principled, always-available null atom that lets an early
query say "there is nothing back there", and `kaleido_ghost_share` is
length-dependent by construction for that reason. A naive post-softmax
subtraction would drive the sink negative and undo it. The fix is structural
rather than a tuning knob: the ghost enters as a **per-row scalar gate**
(`keep = sigmoid(logsumexp)`), applied outside the weights, so applying **branch
A's gate to the whole difference** keeps the sink strictly positive and outside
the subtraction:

```
out = keep_A * ( softmax(S_A) - lambda * softmax(S_B) ) @ V
```

Not `keep_A * softmax(S_A) - lambda * keep_B * softmax(S_B)`, which subtracts
branch B's ghost from branch A's and is the thing Integral Transformer indicts.

One honest note on their formula: `sum_s X W_Q^s (X W_K^s)^T = X (sum_s W_Q^s
W_K^{sT}) X^T`, and that sum has rank at most `S * (d_h/S) = d_h`, which is the
rank of a single head's `QK^T`. So the signal decomposition looks
**expressivity-equivalent to ordinary attention** and the gain has to come from
the `1/S` scaling (a temperature effect) or from optimization, not from a larger
function class. Worth resolving before borrowing it; it is exactly the kind of
neighbouring-cell check `kaleidoscope.md` does for Synthesizer.

## The build: `kaleido_differential`

Minimal, and deliberately three arms so the normalization is not confounded with
the mechanism (DIFF's own ablation says it would be):

1. `kaleido_split_norm_dropoff_always` - an RMSNorm on the attention output
   before the SiLU gate, nothing else. The control DIFF's Table 6 says is
   mandatory.
2. `kaleido_differential_dropoff_always` - arm 1 plus the differential.
3. Existing `kaleido_split_dropoff_always` as the base.

The mechanism:

```
turn        w = beta_d + m * tanh(W_turn x)          [B, T, H, 2N]   (split, DIFF's convention)
two scores  S_A = sum_k w[..., k]   M_k ,  S_B = sum_k w[..., N+k] M_k
lambda_d    per RECURRENT PASS, DIFF's schedule on the pass index:
            lambda_init(d) = 0.8 - 0.6 * exp(-0.3 * d)
weights     P = softmax(mask(S_A)) - lambda_d * softmax(mask(S_B))
ghost       out = keep_A * (P @ dropoff(V))          keep_A = sigmoid(logsumexp(S_A))
out         W_o( gamma * RMSNorm(out) ),   gamma = silu(W_gamma x)
```

- **Identity at init survives.** Both blends are zero-init, so both softmaxes are
  uniform and `P = (1 - lambda) * uniform`. Still uniform attention over the
  causal prefix, just scaled - and the SiLU gate absorbs the scale, which is why
  DIFF's `(1 - lambda_init)` output factor matters less here than it does there.
- **Cost is nothing.** Two blend einsums and two softmaxes instead of one. The
  score half is `2 N T^2` against `T^2 d`, still roughly 0.03x a QK product at
  `d = 90, N = 4`.
- **Parameters:** `W_turn` doubles its output width, plus four `lambda` vectors
  per depth. Under 1% of the block.
- **`lambda` per depth, not global.** The block is recurrent, so DIFF's layer
  index maps to the pass index, and `turn_static` is already per-depth. Report
  it: `kaleido_lambda` at 0 means the second branch was never used and this is
  the base block with a wasted einsum.

**Metrics, and what each falsifies:**

- `kaleido_turn_modes`, now over `2N`. **This is the primary read**, because the
  Jacobian argument predicts it should stop decaying. If it still decays from
  ~3.4 to ~2.4 the mechanism above is wrong and redundancy was the story after
  all.
- `kaleido_lambda` per depth against DIFF's schedule. Rising toward 1 means
  strong cancellation; pinned at `lambda_init` means the parameterization is
  inert.
- `kaleido_negative_mass` - the fraction of attention weight below zero, and
  **separately** the sink's share, so the Integral Transformer failure is
  visible rather than hidden inside an average.
- `kaleido_ghost_share` should be **unchanged** from the base arm by
  construction. If it moves, the ghost is inside the subtraction and the wiring
  is wrong.

## What would falsify all of this

1. `kaleido_turn_scale` and `kaleido_turn_modes` **not** anti-correlated in the
   existing `-i`/`-k`/`-l` logs. Then the Jacobian collapse is not what is
   happening and the redundancy story stands alone. This costs one plot.
2. `turn_modes` still decaying under the differential. Same conclusion, more
   expensively.
3. Parity on loss with `kaleido_split`. The measured reach gain is only 1-15%
   of the remaining residual, so parity is a **likely** outcome and is not by
   itself a refutation of the mechanism. Read `turn_modes` first: the claim this
   note makes is about the router's conditioning, and a run can fix the
   conditioning without moving byte NLL at this scale.
4. `kaleido_negative_mass` concentrating on the earliest positions. That is
   Integral Transformer's indictment reproduced, and the answer is to shrink
   `lambda` at low depth (their finding: vanilla in the lower layers) rather
   than to abandon the mechanism.

## Open questions

- **Does the differential compose with the shear channels of
  [knights_move.md](knights_move.md)?** They are orthogonal on paper: shear adds
  content to the logits, the differential changes how logits become weights.
  Both at once is two interventions and should not be the first run.
- **Is `2N` mirrors the right split, or should the two branches share one
  dictionary?** DIFF splits the head; sharing the dictionary and using two
  *turns* over the same `N` mirrors is cheaper and is arguably the more faithful
  translation, since the mirrors are the analogue of the model's fixed geometry
  and the turn is the analogue of Q/K. Untested either way.
- **The low-`||w||` regime.** The Jacobian table says pre-softmax is better
  conditioned when attention is soft. A schedule that starts pre-softmax and
  moves toward post-softmax is the obvious thought and is also exactly the kind
  of hand-tuned schedule this repo does not do. A learned per-depth mix between
  the two is the legitimate version.
- **Nothing here has been checked at `T` beyond 256**, and the sharpening
  measurement in particular depends on `T/R`.
