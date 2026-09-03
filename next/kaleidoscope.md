# Kaleidoscope: frozen attention geometries, turned by a router

Built 2026-09-02. `attention_type: kaleido`, `praxis/attention/kaleidoscope.py`,
33 tests in `tests/test_kaleidoscope.py`, wired as `experiments/abstractinator-i.yml`
(`--abstractinator-i`, one intervention off `-h`). Untrained.

Origin is the `const [t, t]` thread in
[surrogate_geometry.md](surrogate_geometry.md), from a Discord bot's name taken
seriously as a programming-language question.

## The one line

`N` full `[T, T]` mixing matrices, drawn once and **never trained**; everything
learned is *which combination of them to look through*, per token and per
recurrent pass. A kaleidoscope's mirrors never change - every pattern comes from
turning the tube, and none of them is stored inside it.

```
turn      w(x_i) = beta_d + m * tanh(W_turn x_i)      free, signed, per token
facets    M_k^(d) = A_k + s * tanh(u_{d,k} (x) v_{d,k})
scores    S[i, j] = sum_k w_k(x_i) * M_k^(d)[i, j]
O         = ghostmax(mask(S)) @ dropoff(V)
out       W_o (gamma * O),   gamma = silu(W_gamma x)
```

There is no Q and no K. Nothing is computed from content by a pairwise
comparison, so the projections have nothing to project.

**One head**, matching the rest of the lineage. `patch_config` corrects the
count to 1 and leaves `head_size` as the width, and Mega's Theorem 1
(arXiv:2209.10655) supplies the SiLU output gate that lets one head span what
several did - SiLU rather than sigmoid because the theorem needs a gate that can
amplify and flip sign, and `kaleido_gate_negative` reports whether that freedom
is used. The single head fits better here than it does for Arc: the dictionary
was already shared across heads, so multi-head kaleido would have been H
independent turns of one set of mirrors - a router widening, not new geometry.

## Ghostmax and dropoff, both on

Both inherited from the arc line rather than reinvented, so `-i` differs from
`-h` in the attention core and not in the ablations around it. The registry
entry that carries dropoff is `kaleido_dropoff`, matching `arc_dropoff`.

**`ssog.py`'s reason for declining the ghost does not transfer, and it is worth
being precise about why.** That module left it out because "Softmax1's
always-visible zero-logit ghost would take roughly half of a Gaussian field's
mass." True there: its logits are log-*densities*, large and negative, so a
logit of zero dominates them - measured, 0.505 at T=256, **at every position**.
Kaleidoscope's logits are unit-scale blends of `N(0, 1)` mirrors, and the same
measurement over causal prefixes gives a mean of 0.054 at T=64, 0.018 at 256,
0.010 at 512. The objection was about a logit **scale**, not about ghostmax.

Those means are dominated by the start of the sequence, and that is the point.
Position 0 has exactly one key, so its ghost share is ~0.50 whatever the logits
do; at the tip it falls to 0.0034 at T=256. So the ghost is doing exactly the
job SSOG had to build a *learned null atom* for - letting a query near the start
say "there is nothing back there" - and costs nothing where there is something
to read. **`kaleido_ghost_share` is therefore length-dependent by construction**
and moves with the sequence curriculum; compare like lengths, and do not read a
fall as the model learning to attend.

It is applied without materializing a column: softmax1 is ordinary softmax
scaled by `Z/(1+Z)`, and `Z/(1+Z) = sigmoid(log Z)`, so one sigmoid on the
log-sum-exp the softmax already needs gives it exactly. Same identity
`ssog.py::_apply_null` uses. There is a test asserting it against the literal
append-a-zero-column construction.

**Dropoff is training-only, and the gate is forced rather than chosen.** The
warp is anchored to the current forward's `T`, and a cached decode passes
`T = 1`, so the envelope evaluates to exactly **zero** and every V entering the
cache would be zero - a dead attention branch, not a distribution shift.
Verified. The train/inference asymmetry is nonetheless real and uncorrected;
see [dropoff.md](dropoff.md) for the two coherent designs. **If `-i` shows a
train/eval gap, this is the first suspect.**

**`-i` runs `kaleido_dropoff_always`, and that is not the confound it first
looked like.** The one-beat schedule `-h` runs is very nearly inert: under KL
halting the *training* depth budget is sampled, so at depth 6 the loop reaches
step 5 on only 7.0% of steps - **2.3% of the passes that execute**. There was
no meaningful ablation to hold constant. Always-on is ~44x that exposure and is
the first schedule under which dropoff is a real intervention, so a null result
here should not be read as "dropoff does not matter" - nothing has tested it
yet. `arc_single_dropoff_always_nomem` isolates the schedule if a delta needs
attributing.

One interaction to keep in mind: with `dropoff_every` and the per-depth facets
both active, `kaleido_facet_depth_specialization` partly reads compensation for
a constant per-depth perturbation rather than genuine depth structure.

**Dropoff is `warp` only.** The `shift` mode shifts K as well as V, and there is
no K here to shift - the scores come from the frozen mirrors, not a key
projection - so a V-only shift would be a different ablation wearing the same
name. The envelope is imported from `CausalAttention._dropoff_warp_value` rather
than reimplemented, with a test asserting the two stay identical: it is one
idea, and a second copy would drift from what the arc configs run.

## Why it is not a known variant

Synthesizer (Tay et al., [arXiv:2005.00743](https://arxiv.org/abs/2005.00743))
asked exactly this question - can the attention matrix be *synthesized* rather
than computed - and filled in nearly every neighbouring cell:

| | the matrix | mixing weights | input-conditional? |
| --- | --- | --- | --- |
| Synthesizer (Fixed Random) | 1 frozen random `[L, L]` | - | no |
| Synthesizer (Random) | 1 trained | - | no |
| MLP-Mixer token-mixing | 1 trained | - | no (their own 2021 addendum: "Random Synthesizers **are** a form of MLP-Mixers") |
| Mixture of Synthesizers | N of them | `alpha_{i,h,l}` - **static learned scalars** | **no** |
| MixiT ([arXiv:2506.01115](https://arxiv.org/abs/2506.01115)) | 1 frozen random | - | no |
| **kaleido** | **N frozen random** | **router on x** | **yes** |
| QK attention | - | - | yes, full rank |

Their mixture weights are indexed by head and layer. They are parameters, not
functions of the input. The input-conditional blend is the one cell nobody
occupied.

**The bar is not fixed-random.** Their Fixed Random already reaches ~24 BLEU on
WMT EnDe against ~27.3 for a Transformer, and their *static* Mixture of
Random+Dense "performs comparably to vanilla Transformers." So a frozen basis is
not the interesting claim and beating it proves nothing. The question is whether
making the blend a function of the input beats making it a parameter.

## The turn is not a SMEAR router

Worth stating plainly, because two different fours meet here. `NUM_MIRRORS = 4`
is the dictionary size; `num_experts: 4` in `abstractinator-a.yml` is SMEAR's
expert count. The mirror count was set to match it and is therefore an
**arbitrary inheritance, not a calibration** - the reachable set is an
N-dimensional affine slice, so N is the expressivity knob and nothing here has
measured it.

The turn itself is a plain `nn.Linear` plus softmax. Nothing is imported from
`praxis/routers/smear.py`; what is borrowed is its `input_dependence` estimator
for the metric. So none of SMEAR's machinery is present - no expert dropout, no
utilization balancing, no `smear_*` metrics.

**It does take SMEAR's `base + deviations` form.** `beta_d` is a per-depth
learned blend - the static preference, "mirror 2 is generally useful at pass 3",
free and unbounded like `amplitudes` - and `m * tanh(W_turn x)` is the bounded
input-conditional deviation on top. Only the per-token half is capped, so it
cannot run away with the effective logit scale while the slow blend stays
foundational.
The static term is not optional bookkeeping. Without it the only way to express
a constant preference is through the input projection, which wastes it *and*
contaminates `kaleido_turn_static_share`: a learned constant smuggled through
`W_turn` reads as input dependence that is not there. Separating them is what
makes the metric measure what it claims.

Per-depth rather than a plain bias because it subsumes one (depth 1 degenerates
to a bias) and matches both Arc's per-depth bias table and the facets, which are
already per-depth. `kaleido_turn_static_share` reports which half does the work,
measured as variance *across mirrors* - a constant added to every mirror cancels
in the softmax, so a raw norm would be the wrong quantity. A share at 1.0 is
Synthesizer's Mixture with per-depth alphas, and is the honest way for the model
to say the dictionary needs no per-token selection.

**It also takes SMEAR's balancing.** Mirror dropout at 0.1 - SMEAR's own rate
and mechanism, not an auxiliary balance loss - because the softmax cut proved
that nothing otherwise stops one mirror monopolizing the blend. Dropping all `N`
is safe for the same reason it is safe there: `w` becomes zero and the score
falls back to uniform attention *exactly*, which is this module's identity
state. No inverted-dropout rescaling - these are coefficients on frozen
matrices, so scaling survivors up would change the softmax temperature rather
than preserve an expectation.

**Both halves zero-init, so the score matrix is exactly zero at step 0** and
attention opens uniform over the causal prefix. That is a cleaner identity start
than the softmax version gave, where a uniform blend still produced the
dictionary mean - an arbitrary random matrix the model had to unlearn.

The mechanism is SMEAR-*shaped* in the way that matters (merge N things by a
softmax and apply the merged object once, rather than run N and average their
outputs), and the reduction is **per token** - the strongest of the three
`smear.py` offers. That module states its own honest limit as: "Routing is per
EXAMPLE, never per TOKEN ... a per-token merge would need a distinct geometry
per position." A row of a `[T, T]` mixing matrix already *is* a geometry per
position, so the thing SMEAR cannot reach for a Linear target comes free here.

## It is `MERGE_OPAQUE` to the SMEAR target walker

`praxis/routers/targeting.py` walks the module tree and merges every
parameter-bearing module that does not already route itself. Run against a built
model, it targeted **`attn.turn.weight` itself** - wrapping the per-token turn in
a SMEAR `MergedLinear` routed per **example**, a coarser router around a finer
one - plus batch-mean merges on `facet_u`, `facet_v` and `turn_static`, all of
which are already per-depth conditioned.

The decisive objection is measurement, not cost. If SMEAR varies `turn.weight`
per example, `kaleido_turn_modes` reads variation caused by SMEAR's router
rather than by this one, and the first number the architecture is meant to be
judged on stops measuring what it claims.

The block therefore declares `MERGE_OPAQUE = True`, which is exactly the
condition that module's own docstring names: "a module that already routes its
own parameters per token gains nothing from a per-batch merge wrapped around
it." The frozen mirrors were never at risk - they are buffers, not parameters.

The flag covers the whole subtree, so `value`, `gate` and `output` are excluded
too. That is a real loss, since they are ordinary projections and routing them
is what SMEAR does for arc. Recovering them means moving the geometry machinery
into an opaque submodule, at the cost of changing parameter qualnames; worth
doing if the ablation ever matters, not worth doing mid-experiment.

## Three decisions that carry the design

**1. Mix before the softmax, and do not normalize the weights.** These are one
decision, and getting the second half wrong made the first half worthless.

Blending *after* the softmax is a convex combination of distributions, so
everything reachable lies inside the hull of the frozen patterns - pure
interpolation. Blending *logits* is log-linear pooling,
`softmax(aA + bB) ~ exp(A)^a exp(B)^b`, a product of experts: it puts mass where
mirrors *agree*, which is a pattern no single mirror contains. Verified both
directions - a spread blend lands outside the hull, and a one-hot blend lands
exactly *on* a mirror.

That second fact is the trap. **Pre-softmax mixing only synthesizes anything
while the weights are spread.** At one-hot the score IS one frozen random
matrix, i.e. Synthesizer's Fixed Random evaluated per token - the variant
measured to be worse than a trained one. The first cut of this block used
softmax weights and did exactly that: blend entropy 0.31, which is ~90% on the
top mirror, with utilization oscillating down to `1/N`. It was static attention
wearing a router.

Softmax is the wrong parameterization for two reasons. It confines the blend to
the convex hull - an `(N-1)`-simplex - and its exponential actively *pressures*
the weights toward the one-hot corner where the mechanism dies. This is the same
failure `praxis/routers/smear.py` records for itself: "nothing stopped one
deviation per target from monopolizing its coefficient, and on abstractinator-m
every one of the twelve targets duly saturated to near one-hot," which is why
sharpening is off by default there.

**Free signed weights give the linear span instead** - dimension `N`, not
`N-1` - and a negative weight *subtracts* a mirror, which no mixture of any
weighting can reach (in the product-of-experts reading, a negative exponent:
"attend where this mirror says *not* to"). It also hands the model its own
attention temperature, since with unit-scale mirrors the score is
`~N(0, ||w||^2)`.

This is what the harmonic head already does. `HarmonicField.amplitudes` is a
free real parameter over a frozen basis, not a distribution over it. A simplex
here was the inconsistent choice.

**2. Route per query position, not per sequence.** A pooled router makes the
matrix constant *within* a sequence, which barely moves off Fixed Random. Per
token, row `i` of the matrix is that token's own mixture: each token chooses
which fixed geometry to read the past through.

**3. The per-depth bias goes on the mirrors, not on the inputs** - and those are
different transformations, which is the question that prompted it. Arc adds
`nn.Embedding(depth, dim)` to the *projected inputs* (`praxis/attention/arc.py`); for a
linear map `W(x + b) = Wx + Wb` is a constant offset, the same shift for every
token. Biasing the operator gives `(W + B)x = Wx + Bx`, a correction that scales
with what it acts on. Here the operator *is* the score matrix and it passes
through a softmax, so an additive bias on a mirror is a **multiplicative**
reweighting of the geometry - the same coupling argument `HarmonicField` makes
for applying its field as `h * (1 + b)` rather than `h + b`: the upstream cannot
cancel it by emitting the difference.

The deformation must be **per mirror** or it collapses. Turn weights sum to one,
so a bias added to every mirror alike factors back out,
`sum_k w_k (A_k + B) = (sum_k w_k A_k) + B`, and buys only a per-depth score
bias. Per-(depth, mirror) rank-1 facets do not factor: each pass sees a
differently ground dictionary while the frozen core persists. There is a test
asserting exactly this (`test_facet_deformation_does_not_factor_out_of_the_mixture`).

## The cost, measured

At `abstractinator-i` dimensions (hidden 272, 1 head, head_size 90, depth 6,
N=4, R=64) - and independent of sequence length, which is the point:

- **77,714** parameters in the whole block, single head, gate and static blend
  included. No Q/K projections at all.
- **65.5 KB** of frozen mirrors, non-persistent (regenerated deterministically
  from a fixed seed, exactly as `HarmonicField` does its Weyl spectrum), so they
  never enter a checkpoint.
- The score half costs `N*T^2` against `T^2*d`: a **0.015x** FLOP ratio, ~68x
  cheaper. The `A @ V` half is unchanged. Do not read a wall-clock win into
  this before measuring - `single.py` records the flex path as fixed-overhead
  bound below ~1000 positions, and this block materializes `[B, 1, T, T]`
  rather than using flex at all. Cheaper in FLOPs is not the same as faster.

**The mirrors are length-free**, and this replaced an absolute-position design
that had a fixed span, sliced for short sequences and raised for long ones.

A mirror is a function on the unit square in *relative* position, stored at a
canonical `[R, R]` (R=64) and bilinearly resampled to the live `[T, T]` every
forward. `align_corners=True` pins the canonical corners to the sequence's, so
the whole distribution stretches or shrinks to fit rather than being cropped.
Any `T` works, including the `T = 1` of a cached decode step. Verified at
T = 1, 2, 7, 32, 64, 200, 4096.

**Why it matters more than tidiness.** Under a sequence curriculum `T` changes
every batch. An absolute-indexed dictionary hands the model a *different*
geometry at each length - a different corner slice of one big random matrix,
with no relationship between them - while a ratio-indexed one hands it the
*same* geometry resampled. Measured: the value at relative position
`(0.5, 0.25)` reads 0.866 / 0.819 / 0.797 / 0.788 at T = 128 / 256 / 512 / 1024,
converging as the resample gets finer. Absolute indexing would have given four
unrelated numbers.

It also makes the module a **continuous frozen basis evaluated at the positions
in use**, which is exactly what `HarmonicField` does with `_phase_table` rather
than storing a `[T, D]` table. The absolute version was the odd one out.

**The cost, measured rather than hand-waved.** Ratio structure survives exactly:
"attend to the start", "attend a third of the way back", the diagonal itself.
**Fixed-lag structure does not.** A one-cell canonical feature at column 1
lands on lag 62 spanning 4 positions at T=128, and lag 248 spanning 16 at
T=512. So a dictionary of ratio mirrors alone cannot express "the token
immediately before" at long lengths, and `R` is the knob trading
length-invariance against positional acuity.

### That was the uniform grid, not relative indexing - fixed in `coords="split"`

The paper first stated this as a property of relative indexing. It is not. It
is a property of the **uniform** stretch: `F.interpolate` scales both axes
evenly, so the tip smears exactly as hard as the head, and the block does *not*
inherit the hum-at-head/chirp-at-tip profile the harmonic latent claims for the
residual stream (`conjectures/information-density.yml`).

`MIRROR_COORDS = "split"` reads half the dictionary in (query fraction, **log
lag**), sampled at `log1p(i - j) / log1p(T - 1)`. The log is the whole device:
at T=512, R=64 it puts lags 0/1/2/3 on canonical columns **0/7/11/14**, where
the uniform grid puts them at 0/0.1/0.2/0.4 and cannot separate them. Measured,
a one-cell feature in this coordinate resolves **lag 1 to width 1** at T=256
and T=512.

**Half, not all.** A lag mirror cannot express a ratio any more than a ratio
mirror can express a lag - warping the whole dictionary swaps the limitation
rather than removing it. The split spans both and lets the router choose, which
is the block's own argument one level up. N=4 -> 2 ratio + 2 lag.

**The envelope ranks within each group.** Global ranking would hand the lag
mirrors k=3,4 for being stored second, so a pink run would suppress the new
coordinate system by an accident of ordering. Both groups get `[1.0, 0.5]`, and
`kaleido_envelope_fight` fits against `env_rank` rather than `1..N`. This is
why the -j/-k comparison is meaningful at all - and why the *magnitude* of
`envelope_fight` is not comparable across them, only its sign and movement.

**`kaleido_lag_share`** is the read: share of blend magnitude on the lag half,
even split so **0.5 is parity**. Near zero means acuity was not the binding
constraint and the ratio-only dictionary was right.

**One honesty note.** Normalizing by `log1p(T-1)` keeps the full lag range on
the grid at every length, at the cost of a fixed lag drifting slowly across it
(lag 1 at column 9 for T=128, column 7 for T=512). Logarithmic drift where the
uniform grid's is linear: weaker than exact lag-invariance, much stronger than
none.

Runs: `-i`/`-j` ratio-only (the ablation), `-k` split (`--abstractinator-k`).

Side benefit: the dictionary is now **65.5 KB** instead of 16.8 MB, the facets
live in canonical space (`D*N*2R`, not `D*N*2T`), and the whole block is
**77,714** parameters against 123,794.

## What to watch, in order, before loss

1. **`kaleido_turn_modes`** - effective mirrors in the blend, `1..N`. At 1 the
   score is one frozen matrix and this is Synthesizer's Fixed Random per token,
   the known-worse variant. **This is the failure the softmax cut actually
   hit**, so it is not hypothetical. Collapse here is *not* SMEAR's
   constant-router fixed point, which belongs to the batch reduction; this
   routes per token.
2. **`kaleido_turn_negative`** - fraction of weights below zero, the direct
   falsifier for leaving the simplex. Pinned near 0 means the free
   parameterization bought nothing a softmax could not have done.
3. **`kaleido_mirror_utilization`** - how many mirrors clear half the mean
   magnitude, `1/N` at collapse. Read with (1): together they say which way an
   N-sweep should go.
4. **`kaleido_turn_scale`** (`||w||`) - the effective softmax temperature, since
   the score is `~N(0, ||w||^2)`. Runaway growth sharpens attention onto a
   single key; that is why the per-token half is tanh-bounded.
5. **`kaleido_turn_static_share`** - at 1.0 the blend is a learned constant per
   depth and the input is ignored, which is Synthesizer's Mixture with per-depth
   alphas rather than the claim this makes.
6. **`kaleido_facet_depth_specialization`** - zero means every pass ground its
   mirrors identically and the depth axis is not earning its parameters, which
   would answer the weights-vs-inputs question in the negative.
7. **`kaleido_facet_strength`** - staying near 0 is a *finding*: the dictionary
   plus routing was enough and depth needed no deformation.
8. **`kaleido_gate_negative`** - pinned at 0 means the single head never needed
   to flip a sign and Arc's sigmoid gate would have served.
9. **`kaleido_ghost_share`** - length-dependent by construction, see above. Near
   1 is the attention branch switching itself off.

Every turn metric is **absent at init**, where the blend is identically zero and
the ratios are 0/0. Reporting them would read as collapse, which is the opposite
of an untouched identity start.

A gradient audit will flag the facets as dead at step 0. They are, twice over,
and both are structural: `dS/d(facet_k) = w_k`, so no facet moves until the
blend does, and within a facet `d/dv (u (x) v) = u` with `u` zero-init. The
chain is blend -> `u` -> `v`, and the blend has gradient from step 0, so it
unlocks immediately.

## Honest open questions

- **Random is the control, not the ceiling.** N iid random matrices span a
  random N-dimensional slice of a `T^2`-dimensional space. Fixed Random's 24
  BLEU says one random matrix is already useful, so the slice is not worthless,
  but a dictionary seeded with the patterns real heads converge to (local band,
  start-of-sequence sink, uniform prefix) would almost certainly be better.
  Hand-picking those is disqualified by [[feedback_no_hyperparameter_tuning]];
  a *learned* or *measured* dictionary is the legitimate follow-on.
- **N=4 is a guess** carried over from the SMEAR expert count. The reachable set
  is an N-dimensional affine slice, so N is the expressivity knob and nothing
  here has calibrated it.
- **No `block_ids` or `attention_mask` handling.** Matches the materialised path
  in `ssog.py`, which has the same gap. Fine for a first run; wrong for
  documents packed into one sequence.
- **No KV cache path.** The scores are recomputed from the frozen dictionary
  each forward, so decode is `O(T^2)` per step. Not a blocker for a training
  measurement, a real one for generation.
- **The interference claim is untested.** Pre-softmax mixing *can* produce
  patterns outside the hull of the mirrors; whether it *does* is measurable
  (compare the realized attention rows against their best convex approximation)
  and is not currently measured.

## The A/B

`experiments/abstractinator-i.yml`, `--abstractinator-i`. Extends `-h` and
changes `attention_type` alone, so any delta is the attention core.

Read `kaleido_turn_modes` first. If it sits at 1 the score is a single frozen
matrix, the perplexity number describes Synthesizer's Fixed Random rather than
this design, and the fix is a routing question, not an attention one.

The falsification table lives in the experiment file's header. The short
version: `turn_modes` at 1 is a result about routing collapse, not about
kaleidoscope; **parity is the interesting outcome**, because the block
holds no Q/K parameters and is ~68x cheaper on its score half, and the follow-up
would be an N-sweep since N=4 is inherited from the SMEAR expert count rather
than calibrated.

### A gradient audit will flag this and it is not a bug

`facet_v` receives no gradient at step 0. `d/dv (u (x) v) = u`, and `u` is
zero-initialised so the deformation starts at exactly zero; `u` moves first and
`v` joins once it has. Same asymmetry as `HarmonicField`'s `fast_u`/`fast_v`,
and asserted in `test_facet_v_gradient_unlocks_only_once_u_has_moved`.
