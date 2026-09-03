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
turn      w(x_i) = softmax(W_turn x_i + beta_d)      per token, per depth
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

**It does take SMEAR's `base + deviations` form** (added 2026-09-02). `beta_d`
is a per-depth learned blend - the static preference, "mirror 2 is generally
useful at pass 3" - and `W_turn x` is the input-conditional deviation on top.
The static term is not optional bookkeeping. Without it the only way to express
a constant preference is through the input projection, which wastes it *and*
contaminates `kaleido_turn_dependence`: a learned constant smuggled through
`W_turn` reads as input dependence that is not there. Separating them is what
makes the metric measure what it claims.

Per-depth rather than a plain bias because it subsumes one (depth 1 degenerates
to a bias) and matches both Arc's per-depth bias table and the facets, which are
already per-depth. `kaleido_turn_static_share` reports which half does the work,
measured as variance *across mirrors* - a constant added to every mirror cancels
in the softmax, so a raw norm would be the wrong quantity. A share at 1.0 is
Synthesizer's Mixture with per-depth alphas, and is the honest way for the model
to say the dictionary needs no per-token selection.

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
per example, `kaleido_turn_dependence` reads variation caused by SMEAR's router
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

**1. Mix before the softmax.** Blending after it is a convex combination of
distributions, so everything reachable lies inside the hull of the frozen
patterns - pure interpolation. Blending logits is log-linear pooling:
`softmax(aA + bB) ~ exp(A)^a exp(B)^b` is an *intersection*, and can put mass
where two mirrors agree and nowhere else, which is a pattern neither mirror
contains. That is the difference between blending geometry and synthesizing it.
Synthesizer also mixes inside the softmax.

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
N=4). `-i` inherits `max_position_embeddings: 4096`, which the `MIRROR_SPAN`
default caps at 1024:

- **123,794** parameters in the whole block, single head, gate and static
  blend included. No Q/K projections at all. The facets are `D*N*2*span`, so
  the span shows up in the parameter count as well as in memory; the static
  blend is 24 of them.
- **16.8 MB** of frozen mirrors, non-persistent (regenerated deterministically
  from a fixed seed, exactly as `HarmonicField` does its Weyl spectrum), so they
  never enter a checkpoint.
- The score half costs `N*T^2` against `T^2*d`: a **0.015x** FLOP ratio, ~68x
  cheaper. The `A @ V` half is unchanged. Do not read a wall-clock win into
  this before measuring - `single.py` records the flex path as fixed-overhead
  bound below ~1000 positions, and this block materializes `[B, 1, T, T]`
  rather than using flex at all. Cheaper in FLOPs is not the same as faster.

The span is the real constraint and it is inherent, not an oversight. Mirrors
are indexed by **absolute** `(query, key)` position - that is what lets them
express patterns a lag-indexed kernel cannot, like "attend to the start of the
sequence" - and absolute indexing means a fixed span, `N*T^2` floats: 4 MB at
512, 16 MB at 1024, 268 MB at 4096. Overrunning it raises rather than silently
truncating. Synthesizer hit the same wall and answered it by truncating to the
batch's length.

## What to watch, in order, before loss

1. **`kaleido_turn_dependence`.** If it decays to zero the router is a constant,
   the mixture is a fixed matrix, and this has silently become Fixed Random -
   the known-worse variant. Still the first number to read.

   This is a risk to watch, **not** an inherited trap. SMEAR's constant-router
   fixed point belongs to its **batch** reduction, where "the loss reaches the
   routing only through `probs.mean(0)`, so every example receives the identical
   routing gradient" - measured decaying to exactly 0 on `-m`. Kaleidoscope
   routes **per token**, which has a distinct gradient per position and does not
   have that fixed point. Ordinary router collapse is still possible, which is
   why (2) is paired with it.
2. **`kaleido_turn_entropy`** read *with* it. Falling entropy at zero dependence
   is collapse onto one mirror. Falling entropy at rising dependence is genuine
   per-token selection. The pair separates the two; neither number does alone.
3. **`kaleido_facet_depth_specialization`.** Zero means every pass ground its
   mirrors identically and the depth axis is not earning its parameters -
   which would answer the weights-vs-inputs question in the negative.
4. **`kaleido_facet_strength`.** Staying near 0 is a *finding*: the frozen
   dictionary plus routing was enough and depth needed no deformation.
5. **`kaleido_gate_negative`.** Pinned at 0 means the single head never needed
   to flip a sign and Arc's sigmoid gate would have served equally.
6. **`kaleido_mirror_utilization`.** Are the mirrors earning their keep - the
   fraction carrying more than half their fair share of the blend, the same
   estimator as `smear_expert_utilization`. `1/N` is total collapse onto one
   mirror. This is what says which direction an N-sweep should go: 1.0 with
   committed entropy argues for more mirrors, a falling value says the
   dictionary is already larger than the model can use.
7. **`kaleido_turn_static_share`.** At 1.0 the blend is a learned constant per
   depth and the input is ignored - which is Synthesizer's Mixture, not the
   claim this makes. Read with (1): they answer the same question from the two
   sides of the sum.
8. **`kaleido_ghost_share`.** Length-dependent, see above. Near 1 is the
   attention branch switching itself off.

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

Read `kaleido_turn_dependence` first. If it is at zero the perplexity number
means nothing, because the model being measured is not the model that was
designed - and the fix would be a routing question, not an attention one.

The falsification table lives in the experiment file's header. The short
version: `turn_dependence` at 0 is a result about SMEAR routers in this stack,
not about kaleidoscope; **parity is the interesting outcome**, because the block
holds no Q/K parameters and is ~68x cheaper on its score half, and the follow-up
would be an N-sweep since N=4 is inherited from the SMEAR expert count rather
than calibrated.

### A gradient audit will flag this and it is not a bug

`facet_v` receives no gradient at step 0. `d/dv (u (x) v) = u`, and `u` is
zero-initialised so the deformation starts at exactly zero; `u` moves first and
`v` joins once it has. Same asymmetry as `HarmonicField`'s `fast_u`/`fast_v`,
and asserted in `test_facet_v_gradient_unlocks_only_once_u_has_moved`.
