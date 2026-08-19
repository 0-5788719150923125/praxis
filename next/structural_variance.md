# Structural Variance: Not Collapsing the Distributions We Already Build

> Status: **parked design, later pass** (2026-08-17). Nothing here is
> implemented. Explicitly gated behind `ssog` being trained and measured at
> least once - see [Sequencing](#sequencing) at the bottom. Companion to
> [harmonic_koopman.md](harmonic_koopman.md),
> [prismatic.md](prismatic.md), [multirate_vote.md](multirate_vote.md),
> and the paper's Section 2 (`research/body.tex`, `sec:manifold`).

## The one line

Praxis parameterizes probability distributions in half a dozen places and then
**collapses every one of them to an expectation** before the value moves
downstream. The axis worth exploring is not "represent variance" - we already do
that - it is **stop taking the mean at the readout**.

## The observation that started this

Ask "is there a concept in ML for mapping structured variance inside the model,
not just at the classification head?" and the honest answer is that the
interesting distinction is not represent-vs-not. It is
**represent-vs-collapse**. Once you look for it, the pattern is everywhere in
our own code:

| Site | What it builds | What it emits |
| --- | --- | --- |
| `attention/ssog.py` | a real mixture density over lag: `mu`, `sigma`, `lambda` per atom | `softmax(logits) @ V` - a **weighted mean of V** |
| `heads/harmonic.py` | input-conditional amplitude delta `Δ_φ(context)` | one scalar per cell - a **point estimate** |
| crystal head | a discrete set of centers | a soft assignment - an **expectation over centers** |
| standard attention | a distribution over positions | a **weighted mean of values** |
| `heads/parallel.py` | two branches, bias arm and variance arm, on the same input | `sum_i softmax(gate)_i * branch_i(h)` - a **weighted mean of branches** |
| `routers/smear.py` | a router distribution over N experts | a softmax-weighted **merge of expert parameters** |

SSOG is the sharpest case, because the density is completely explicit. The
Gaussians shape *which* mean gets taken, and then a mean gets taken. It is a
structured-variance **parameterization** with a mean-based **readout**.

The classic illustration of why that can be the wrong summary is the mixture
density network failure mode (Bishop, 1994): if the target is bimodal, the mean
lands between the modes. The mean of "swerve left" and "swerve right" is "hit
the obstacle."

ParallelHead and SMEAR are worth calling out separately, because in both cases
the collapse is **deliberate and well-motivated**, not an oversight. SMEAR
merges expert parameters by a softmax specifically so that routing is
differentiable without sampling - that is the paper's entire contribution. The
point is not that these are bugs. The point is that we have chosen the mean
everywhere, for good local reasons, and never once priced what the choice costs.

The one place in the stack that does **not** collapse is the CALM patch vote -
mode of N decoded candidates rather than argmax of an averaged distribution.
That is why it feels like a different regime. It is the only non-expectation
readout we have.

## What "multimodal" means here (three senses of "mode")

Disambiguation first, because "mode" is now doing three separate jobs in this
codebase and two of them already collide in the paper.

1. **Statistical mode** - a peak of a probability distribution. A distribution
   is *multimodal* when it has more than one peak. This is the sense used
   throughout this note, and the sense in "mode of 500 samples."
2. **Frequency mode** - one component of the harmonic basis. This is the
   glossary sense in `research/body.tex`. Section 2 already leans on the
   collision with sense 1 on purpose: "mode collapse" is true in both readings
   at once, since a spectrum concentrating on one cell *is* a distribution
   collapsing to one peak.
3. **Modality** - text, image, audio. Different input types. **Not used in this
   note at all.**

The operational test for sense 1, which is the only one that matters here:

> **A distribution is multimodal when its mean is not a valid sample.**

The mean of "swerve left" and "swerve right" is "hit the obstacle." The mean of
`the` and `a` is not a word. Where the mean is not in the support, taking the
mean is not a summary - it is a fabrication, and a vote is doing something a
mean structurally cannot.

## Two paths over one input: a mechanism, not a definition

The natural question is whether running the same input down two pathways -
standard LM embeddings and CALM embeddings, say - and combining them makes the
model "multimodal." In sense 3, yes, loosely. In sense 1, **not by itself**, and
the distinction is the whole thesis of this note.

Two pathways are a *mechanism that can produce* a multimodal predictive
distribution: when the paths disagree about the next token, the combined
distribution has a peak per path. That is a real and cheap source of structured
variance, and it is worth having.

But **how you combine them decides whether any of it survives.** Cross-attention
between the paths, a learned gate, or a weighted sum all take an expectation,
and an expectation over a bimodal mixture lands in the valley between the peaks.
We already have exactly this architecture and it already collapses:
`heads/parallel.py` runs a bias arm and a variance arm on the same input and
emits `sum_i softmax(gate)_i * branch_i(h)`. The `prismatic` profile is two
paths over one input, merged by a mean.

So the requirement is not "accept the same input through different paths." It is
**keep the paths distinguishable at a readout that does not average them.** A
mixture is only a mixture until something integrates over it. Concretely, that
means one of:

- carry both paths' distributions to the final readout and **vote**, rather than
  gating them into one representation mid-stack;
- train the combination with a proper scoring rule (CRPS / energy score) so the
  loss rewards covering both peaks instead of splitting the difference;
- keep the gate, but make it **sample** a path rather than blend paths, and pay
  for that with the quantile machinery below rather than with REINFORCE.

There is a cheap diagnostic here that needs no new architecture: on an existing
`prismatic` run, log the **per-token gate entropy** and the **disagreement
between branch 0 and branch 1 logits**. Tokens where the branches disagree
sharply *and* the gate is near 0.5 are exactly the tokens whose true predictive
distribution is bimodal and whose blended output is a fabrication. If that set
is empty, the branches have converged onto the same function and the parallel
split is not buying variance at all - which would be a finding about
`prismatic` independent of everything else in this note.

## Where multimodality should actually appear

Candidate sites, ordered by how likely a real second peak is and how cheap it is
to check. All of these are measurable on runs we already have.

- **Byte-level positions at a word boundary.** Inside a word the next byte is
  near-deterministic; at a boundary the model is choosing among words, which is
  genuinely multimodal. A pure-byte model therefore *alternates* between
  unimodal and multimodal positions on a regular schedule. This is the cleanest
  natural experiment available to us and it costs one logging hook.
- **CALM patch decoding.** A patch spans several bytes, so its distribution is a
  joint over several decisions, which is far more likely to be multimodal than
  any single byte. This is the strongest a-priori reason the patch vote might be
  doing real work, and it is what the check below tests.
- **MTP draft positions.** Multimodality should grow monotonically with draft
  distance. The adaptive draft width already tracks an accepted-run EMA, which
  is an *indirect* measurement of exactly this - a direct one would tell us
  whether width should key on measured modality instead.
- **Halting depth.** A token that could plausibly halt at depth 2 or depth 5 is
  bimodal in depth. KL halting currently reads a convergence ratio, which is a
  unimodal summary of a possibly-bimodal quantity.
- **Router assignments.** A genuinely ambiguous token should produce a flat or
  bimodal router distribution. SMEAR merges it either way.

## Rendered text as images: adjacent, and a separate thread

Writing training text into images and mixing it alongside the byte stream is
sense-3 multimodality. It is a real line of work - PIXEL (Rust et al. 2023)
trains a language model on rendered text specifically to escape the vocabulary
bottleneck, and the optical-compression results treat a page of rendered text as
a cheaper carrier than its tokens.

It would give two views of the same content, and disagreement between views is a
variance source, so the instinct connects. But it does not test anything in this
note more cheaply than two embedding paths over the same bytes do, and it drags
in a renderer, an image encoder, and a resolution/fidelity axis we would then
have to tune. **Verdict: worth its own note if the compression angle is what
appeals, not a component of this one.**

## Prior art, so we do not reinvent it under a new name

Ordered by how much is directly stealable. None of this is ours; the thing that
would be ours is doing it in a **harmonic** basis, where the moments are closed
form.

- **Distributional RL** - Bellemare et al. 2017 (C51), Dabney et al. 2018
  (QR-DQN, IQN). The precedent that matters most. They replaced `E[Z]` with the
  full return distribution `Z` throughout the Bellman operator, and the
  **mean-greedy policy got better anyway**. That is the exact empirical shape of
  the claim we would be making: carrying the distribution improves the *point*
  prediction, so it is a representational win, not just an uncertainty-reporting
  win. They also solved the gradient problem (below).
- **Mixture density networks** - Bishop 1994. Direct ancestor of SSOG's
  parameterization, and the source of the bimodal-mean failure mode.
- **Variational attention** - Deng et al. 2018, "Latent Alignment and
  Variational Attention." Attention weights as latent random variables that are
  sampled rather than averaged, motivated by exactly the observation that soft
  attention is the *mean* of a distribution over alignments. The closest
  existing thing to "SSOG that does not take the mean."
- **Distributed distributional codes / probabilistic population codes** - Sahani
  & Dayan 2003, Vertes & Sahani 2018, Ma et al. 2006. A population whose
  *activity pattern* encodes a full posterior with no sampling. The formal
  object closest to "map variance in an interference pattern," and the
  theoretical grounding for the 500-vote intuition.
- **Sample-free moment propagation** - assumed density filtering, natural
  parameter networks, Gast & Roth 2018, Wu et al. 2019. Propagate a distribution
  through the stack in closed form. This is the piece that makes any of it
  affordable at our scale.

Three of these are now in `praxis/pillars/citations.bib` (`sobol1993`,
`mezic2005`, `bellemare2017distributional`); the rest go in when something
actually cites them. **Verify each against the source before it lands in a
fragment** - the list above was written from memory.

## The proposal, concretely: phase concentration on the harmonic field

This is the piece worth building, and it is one parameter grid.

### What we have

`heads/harmonic.py` builds

```
b = IRFFT2( a[f_t, f_d] * exp(i * phi[f_t, f_d]) / f**alpha )
```

with `phi` **frozen** at Weyl-seeded values. Frozen phase means the field is
deterministic by construction. In the paper's own terms, that is why the base
architecture is all bias: there is no second moment anywhere in the field.

### What we would add

Give each cell a **phase distribution** instead of a phase value: von Mises with
mean direction the existing Weyl `phi[f_t, f_d]`, and a new learnable
concentration grid `kappa[f_t, f_d]`.

The useful fact is that the first moment is closed form:

```
E[ cos(2*pi*f*x + phi) ]  =  A(kappa) * cos(2*pi*f*x + phi)
A(kappa) = I_1(kappa) / I_0(kappa)          # mean resultant length
```

So the **mean field is exactly today's field, damped per cell by `A(kappa)`**.
The second moment is also closed form, as is the covariance between two modes'
interference terms, so we get a variance field without sampling anything inside
the stack.

### Why this shape fits how we build things

- **Identity at initialization.** `kappa -> inf` gives `A -> 1`, which is
  literally the current model. Same discipline as the prismatic3 pure-variance
  arm and the zero-init `Δ_φ` envelope: the new axis starts inert and the model
  opens it itself.
- **No sampling in the forward pass.** Closed-form moments, so no 500-way
  anything at every layer. Cost is one extra `[f_t, f_d]` grid and a Bessel
  ratio, which is a cheap `tanh`-like function of `kappa` we can approximate
  directly rather than calling into `scipy`.
- **Differentiable throughout.** No straight-through estimator, no REINFORCE.
- **It makes the paper's central claim mechanical rather than rhetorical.**
  Right now Section 2 argues bias/variance orthogonality from "separate
  parameters with separate gradients," which is an argument about the optimizer.
  Under this construction bias and variance are the **first and second moments
  of one field**, which is orthogonality in the mathematics.
- **It lines up with the Koopman framing we already cite.** Point spectrum
  (discrete lines, eigenfunctions, recurring) = bias. Continuous spectrum (no
  eigenfunction, spread, mixing) = variance. `kappa` is the dial between them.

### Where it would live

- `praxis/heads/harmonic.py` - the `kappa` grid, the `A(kappa)` damping applied
  to the existing amplitude-times-phase product, and a variance-field branch.
- Registry, not a CLI flag - a head profile (per
  [feedback_registry_over_cli_args]), so the arm is selectable the way
  `prismatic3` is and the baseline is untouched.
- `training_metrics()` on the head itself (per the co-location rule), emitting
  the diagnostics below.

### What we should expect to see

Stated as falsifiers, because the interesting outcome is the one that kills it.

1. **`kappa` runs to infinity and `A(kappa) -> 1` everywhere.** This is the
   default expected failure, and the reason is not subtle: **cross-entropy
   rewards confident means.** If everything downstream consumes the mean field,
   the model correctly concludes that spread is pure loss and closes it. If we
   build this and see `mean(A(kappa)) -> 1.0` within a few thousand steps, the
   experiment is *done* and the answer is "the readout has to pay for variance
   first." Which leads to:
2. **Nothing works until some readout is not an expectation.** The `kappa` grid
   is necessary but not sufficient. Something downstream has to be trained by a
   proper scoring rule (CRPS / energy score) or has to vote. This is the real
   dependency, and it is why this note is parked rather than queued.
3. **New diagnostic: line-to-continuum ratio, per grid.** Hoyer concentration
   measures how tightly mass sits in a few cells; it does not distinguish atomic
   mass from spread. The measure this argument wants is the ratio of line mass
   to continuum mass, computed **separately on the static baseline and on the
   delta**. The decoupling claim predicts they move in *opposite* directions. If
   they move together, the axes were never separate and the paper's Section 2
   needs weakening.
4. **The win condition, borrowed from distributional RL:** the *deterministic
   mean readout gets better* with `kappa` learnable than with it pinned at
   infinity, at matched parameter count. If carrying the second moment only
   helps when you also change the readout, that is a weaker and more expensive
   result. If it helps the mean, the axis is real.

## The cheap measurement that should come first

Before any of the above, there is a few-hour check that decides whether the
whole framing has legs, and it uses runs we already have. (The prismatic
gate-entropy / branch-disagreement probe described earlier is cheaper still and
independent of this one - it asks whether a two-path split is producing
multimodality, while this one asks whether an existing vote is consuming any.)

**Is the CALM sample distribution actually multimodal?** The claim "mode of 500
is voting over the variance, not a mean over the bias" is only true if the
samples are multimodal. If they are unimodal, mode is approximately mean and the
vote is just a variance-reduced mean estimator - the *same* regime, done more
expensively.

Test: for each patch, compare `mode of N samples` against
`argmax of averaged logits`. Ask whether they differ more often than sampling
noise explains, and whether the disagreement correlates with correctness. A dip
test or a plain modality count on the sample set answers it directly.

- **Multimodal** -> we have a measured result that the vote is a different
  regime, and everything above is worth building.
- **Unimodal** -> CALM's advantage was ensembling, the "different regime"
  language comes out of the paper, and the honest remaining claim is the narrow
  one: we parameterize distributions and consume them as expectations. Still a
  real observation, still points at the same fix, but much smaller.

Note that CALM is shelved as of 2026-07-16 (see `project_calm_direction`); this
is a measurement on archived runs, not a reason to resume that thread.

## The gradient problem, and the answer

Voting and mode-taking are not differentiable, and N samples per site is not
affordable. Distributional RL solved both with the same move: **do not sample,
carry quantiles.**

Represent a population as N deterministic outputs interpreted as quantile
locations, train with the pinball / quantile-Huber loss, take the mode or a vote
only at the final readout. The forward pass stays deterministic, gradients flow
normally, the population is real, and N=8 is plenty (QR-DQN used 32 for full
Atari). This composes cleanly with the `kappa` proposal: closed-form moments
give the distribution, quantiles give a differentiable handle on its shape, and
a vote gives the non-expectation readout that makes `kappa` worth opening.

## Sequencing

Hard ordering. Do not skip.

1. **Train and measure `ssog` at least once.** It is implemented (2026-08-16)
   and untrained. Everything above is a claim about what mixture densities buy,
   and we have not yet seen one run. If SSOG is a dud as a *mean-based* mixture,
   that is important information about whether the non-mean version is worth it.
2. **The CALM multimodality check** on archived runs. Cheap, decides the framing.
3. **The prismatic gate probe** (gate entropy + branch logit disagreement). Also
   cheap, also on existing runs, and independent of 1 and 2 - it says whether
   the two-path structure we already ship is generating multimodality or whether
   the branches have converged. Can run in parallel with 2.
4. Only then: the `kappa` grid, and only alongside a readout that pays for
   spread.

## Paper status

Landed 2026-08-17, ahead of any of the above:

- `research/body.tex` glossary, "Bias and variance, in these terms" - one
  sentence naming which variance we mean (over the input distribution, not over
  training-set resamples). This was a genuine ambiguity: under the classical
  reading, a model that responds *more* to its input has *lower bias*, so a
  reader applying the wrong definition reads Section 2 backwards.
- `praxis/pillars/framing/manifold-which-variance.yml` - unconditional fragment,
  section `manifold`, order 30. Carries the functional-ANOVA grounding, the
  Koopman point/continuous spectrum reading, the line-to-continuum diagnostic,
  and the represent-vs-collapse question stated as a direction with a falsifier.

The paper deliberately does **not** claim the `kappa` construction. That stays
here until something is measured.
