# Holding the Live Regime: Instruments First, Then Two Structures

> Status: **assessment + one shipped-ready finding, 2026-08-30.** Prompted by
> `../rift/books/north-star/chapters/15-roguelike.md` and by
> `abstractinator-e` producing unusually coherent output while sitting in a
> measurably un-collapsed state. Companion to
> [magnetism.md](magnetism.md), [structural_variance.md](structural_variance.md),
> [information_density.md](information_density.md).

## The one line

The chirp-then-heat-death claim is **already instrumented in this repo**, the
instruments say the collapse is real and monotone on every run - and one of the
things driving it is a **regularizer we are running on purpose whose consumer
does not exist**.

## What the instruments say

Median over steps 2000-5000 ("early") vs 8000-11400 ("late"), matched windows:

| metric | -a early→late | -b early→late | -e early→late |
|---|---|---|---|
| `repr_anisotropy` | 0.502 → 0.252 | 0.536 → 0.218 | 0.546 → 0.244 |
| `repr_nematic` | 0.584 → 0.383 | 0.553 → 0.379 | 0.550 → **0.403** |
| `repr_dimensions` | 0.044 → 0.069 | 0.047 → 0.069 | 0.048 → 0.065 |
| `stem_harmonic_capacity_dormant` | 0.322 → 0.569 | 0.333 → 0.307 | 0.299 → **0.123** |
| `gate_entropy` | 0.142 → 0.139 | 0.456 → **0.177** | 0.392 → **0.399** |

Read together: magnetization down, the nematic (second-moment) order down,
effective dimensionality **up** - i.e. the direction distribution is getting
more uniform and the spectrum flatter. That is the chapter's "every feature
beating the same beat, the spectrum gone flat", in numbers, on every run.

`-e` is the outlier on two of them: dormant capacity **falls** to 0.12 where -a
rose to 0.57, and gate entropy **holds** at 0.40 where -b collapsed to 0.18.
Whatever else the stitched memory is doing, the model in that run is using more
of its harmonic capacity and more of its head mixture than any sibling. That is
the regime the chapter is about, observed live rather than remembered.

## The finding worth acting on first

`repr_anisotropy` **is** the squared magnetization (pinned in
[magnetism.md](magnetism.md)). It falls by half on every run. It is *supposed
to*: `contrastive_isotropy` is in the active regularizer list and its entire job
is to push distinct token representations apart toward isotropy. Per the
magnetism note's own correction, **an isotropy loss is a demagnetizer**.

So: if collapse-to-the-mean is the thing to prevent, we are running a term whose
gradient points at it.

It was adopted for SimCTG-style contrastive-search decoding. **Nothing in
`praxis/` references `penalty_alpha` (0 files).** The regularizer runs on every
step; the decode path it was meant to enable is not wired.

**This is the cheapest experiment in the whole area and it is one key:** a run
with `regularizers: [harmonic_kl]`. If anisotropy and nematic order stop halving
and the coherent regime lasts longer, the demagnetizer was the mechanism. If
nothing changes, collapse is intrinsic and the structural ideas below are the
only route. Either answer is worth more than either idea is, right now.

## Idea 1: input-conditional weight bounds (hypernetwork-ish)

**Proposal.** A router emits an upper and lower scalar per input; remap a linear
layer's weight distribution into that range before multiplying.

**Where it is right.** It is a *learned per-input variance dial*, which is
exactly the axis. And there is precedent in-tree: `Servant` already does
input-conditional modulation of an activation's frequency
(`a_eff = a*(1 + MOD_MAX*tanh(v)*m)`).

**Where it fails as stated, and it is a hard failure.** An affine remap of a
weight matrix into `[lo, hi]` is `W' = aW + b`, so `W'x = a(Wx) + b(1ᵀx)1`: a
scalar gain plus a **rank-1** term. A scalar gain multiplies every singular
value equally - it changes the *temperature* of the layer and cannot change the
*shape* of its spectrum. Collapse is a spectral-shape phenomenon. So this
version cannot prevent it, however well the router is trained.

**The version that could.** Make the bounds **per-output-channel** (a vector,
not a scalar), or per-singular-direction. Then the modulation reshapes the
spectrum instead of scaling it, which is the quantity that actually flattens.
Still cheap: one vector per layer per input, not a matrix.

**Carry this scar in.** [project_energy_signal_saturation] - Servant's live
energy signal `tanh(log s - ref)` **pinned at 1.0** and the modulation silently
became a constant. Any router driving these bounds needs a running z-score on
its input signal, and a metric that would show saturation. Build the metric
first; the failure is invisible in magnitude statistics.

## Idea 2: degaussing by direction

**Proposal.** Read sequences front-to-back or back-to-front (not both at once),
possibly with per-direction embeddings, possibly alternating across batches; let
the positional geometry curve differently per direction, like refraction.

**Where it is right, and it is more right than idea 1.** Degaussing is not a
metaphor here. You demagnetize iron with an **alternating field of decaying
amplitude**; alternating the reading direction across batches is literally an
alternating drive on the sequence axis. The magnetism note concluded the magnet
analogy was "mechanically empty" - that verdict was about a *static* isotropy
term. It does not cover an alternating drive, which is the actual operation the
physics names. The analogy earns its keep here.

The anti-collapse mechanism is concrete and does not depend on the metaphor:
**two objectives that cannot both be satisfied by the mean.** A feature that is
optimal for left-to-right is not optimal for right-to-left, so the flat average
of the corpus stops being a fixed point. That is the structural version of
"weakness is strength" - a strain the model cannot relieve by converging.

**Cheapest honest test.** Reverse the byte stream for a fraction of rows. The
patcher is `patch_size: 8` static, so reversal is clean and cheap, and
`row_continues` already proves a per-row channel can reach the model. Separate
embeddings per direction and a direction-conditional phase warp (ArcHoPE already
warps; [project_learnable_rope_theta] already makes theta learnable) are the
second and third steps, not the first.

**Honest risks.** Reversed text is a different distribution, so some capacity
goes to modelling it - though at 3.2M params, on this project's own thesis, a
model that cannot memorize either direction may be forced to twist rather than
copy, which is the point. And the gain may be indistinguishable from ordinary
augmentation; the falsifier is that augmentation should *not* specifically slow
the decay of `repr_nematic` and `capacity_dormant`, and this should.

## Sequencing

1. **Drop `contrastive_isotropy`.** One key. It is the only candidate mechanism
   currently identified *inside* the model, and it is running unopposed.
2. **Log the collapse trace through a death.** The metrics above already exist;
   what is missing is watching them across the transition rather than at two
   snapshots. Which flattens *first* discriminates the two ideas: if the
   spectrum shape goes before the magnetization, idea 1's per-channel version is
   aimed correctly; if direction-asymmetric features die first, idea 2 is.
3. **Direction augmentation**, as the cheapest structural change.
4. **Per-channel conditional bounds**, only with the saturation metric built first.

## What this note does not claim

That the coherent-then-dead trajectory is *caused* by any of this. Three runs
showing monotone flattening is a pattern, not a mechanism, and `-e`'s two
outlying metrics are one run. The chapter's claim is currently better supported
than any explanation of it.

## Addendum: harmony vs interference (2026-08-30)

**`harmonic_kl` is not the harmony term.** It is a readout trust region:
KL(EMA-of-the-classifier's-own-parameters || live readout), measuring how fast
the output basis moves. It says nothing about phase relationships between
features, so "all features phase-lock, hence the loss is near zero" is aimed at
the wrong object.

And near-zero is its **pre-registered success condition**, not a failure. From
`praxis/losses/harmonic_kl.py`: *"If the basis really is constitutive, the
penalty should sit near zero without being paid for."* It sits at 2.5e-5 (-a),
2.1e-5 (-b). The claim it was built to falsify survived.

One number underneath it is worth keeping, though: `harmonic_drift` is 0.00050
on -a, 0.00042 on -b and **0.00699 on -e** - a 14x wider readout excursion on
the run that is also holding gate entropy and burning down dormant capacity.
That is a fourth independent instrument on which -e is the outlier.

**The phase-locking intuition is real, but it lives one module over.**
`harmonic_capacity_dormant` is literally "share of spectral capacity sitting
dormant" (`praxis/heads/harmonic.py:365`) and it **rises to 0.57 on -a**. That
is mode death in the harmonic basis - features collapsing onto a shared periodic
shape and leaving the rest of the spectrum unused. It is the thing an
interference/repulsion term would fight, and it is already measured.

**The 50/50 blend implies a tension that does not exist.** The two terms act on
different axes:

| | acts on | over |
|---|---|---|
| `harmonic_kl` | the readout's parameters | time (live vs its own past) |
| a repulsion term | features' frequencies | one forward |

They are orthogonal. There is no dial between them - you can run both, and a
blend weight would only be trading two unrelated things against each other.

**If it gets built, the shape already exists in-tree.** `mtp_field_distinctness`
is exactly this quantity in *measurement* form (1 - mean pairwise |cosine| of
the depths' Serpent frequency spectra), deliberately left unpenalised so it
would report rather than enforce. VEAR carries the *penalty* form, and
`praxis/routers/bank.py:300-316` documents the hazard that forced it to be
**parameter-only** - which is the constraint any frequency repulsion here should
inherit before it is written.

Sequencing note: this is still behind item 1. `contrastive_isotropy` is a term
we are actively paying that pushes toward the flat state; a repulsion term is a
new term that might pull away from it. Removing a known demagnetizer is a
cheaper and cleaner first measurement than adding an opposing force on top of it.
