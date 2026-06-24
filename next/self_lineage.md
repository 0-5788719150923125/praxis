# Self-Lineage: training on its own git history as a recency-weighted time series

> Status: **exploratory / half-built + flagged voice** (2026-06-23). The answer
> to "machines hold only the text anchor" from
> [observer_grounding.md](observer_grounding.md): give the machine a *lineage* by
> training it on the record of its own construction. Sibling to
> [harmonic_koopman.md](harmonic_koopman.md), [the_dial.md](the_dial.md),
> [observer_frequency.md](observer_frequency.md),
> [observer_effect.md](observer_effect.md), [kb_spider.md](kb_spider.md).
> Discipline as always: kernel maps to mechanism; the cyclic-time / rollback
> imagery is voice, marked.

## The thread it answers

[observer_grounding.md](observer_grounding.md) concluded that human convergence
rides on *shared lineage* (genome + world + language community), and that this
does not transfer to a model, which holds only anchor (3): the text. The move
here: manufacture a lineage. A repository's commit history is the model's
**ontogeny** - the code is its genome, the commit sequence its development. Train
on that history and the model gains a self-anchor it otherwise lacks.

Honest scope up front: this is **self / identity** grounding, not world
grounding. The repo is not the physical world. It is a developmental record of
one artifact. That bounds what it can buy (below).

## Half of it already exists

`praxis/pillars/evolution.py` already does the extraction and the age-weighting,
as a *figure*:

- `git log --numstat --date=unix` over up to 4000 commits, oldest-first;
- per-subsystem churn (insertions+deletions) and net lines, binned over time;
- a **recency decay** (`RECENCY_DECAY`, exp falloff into the past) - the docstring:
  *"the recency kernel the model itself uses, turned on the repo."*

That is precisely "sample commits over time, penalized over age." What is missing
is the other half: emit the **actual diffs** (file + line level, `git log -p`
patch hunks, not just numstat counts) as a **training source**, recency-weighted
in the *sampling*, registered like any other dataset (the KB dataset / spider
pattern, [kb_spider.md](kb_spider.md)). The figure proves the signal is real and
cheap to extract; the feature is "same signal, into tokens instead of canvas."

## The grounded kernel (buildable)

A `git-history` dataset source:

- **Unit:** a transition between commits at file+line granularity - a hunk, or a
  (before, after) pair, optionally with the commit message as the "intent" label.
  This is a *time series of code changes*: the diff stream is the velocity of the
  repo, the changing harmony.
- **Recency weighting:** sample probability decays with commit age (reuse
  `evolution.py`'s decay). Distant history is nearly irrelevant - the same `1/f^α`
  / exp envelope the harmonic head already imposes on frequency, here on the time
  axis of commits.
- **Fit:** a new source in the dataset registry; no new architecture. Pairs with
  the existing self-evolution figure as its training-time counterpart.

What it buys, honestly: a model that can *reason about its own evolution*, an
identity/flavor anchor, and a clean demonstration of the blind-watchmaker thesis -
the watchmaker reading the assembly log of the watch ([harmonic_koopman.md] and
the paper's Dawkins framing). What it does **not** buy: capability or world
knowledge. The corpus is tiny and idiosyncratic; as a *primary* signal it is
nothing. As a flavoring source in the mix, or an instrument, it is coherent.

## The lost thread, recovered: forward and backward through time

A model that predicts the next diff is a **generative model of the repo's
trajectory**. That gives both directions:

- **Forward** sampling = proposing future commits (what change comes next).
- **Backward** = un-applying diffs / training on reversed commit order =
  reconstructing past states. This is the "time moving backwards, documenting what
  was done in the past" image, made concrete: reverse-time modeling is a real
  augmentation (predict the diff that *produced* a state from the state).
- **"Blindly at first"** = the learning curve. Early in the trajectory the model
  cannot predict the diffs; it improves. Birth → growth is literally the loss
  curve over the ontogenetic sequence.

So the intuition "invert the process of creating machines, work backwards through
time so a human could reconstruct them, blindly at first" has a precise shadow:
**bidirectional diff modeling over the recency-weighted commit trajectory.** The
flag: a 1M-parameter model on a few-thousand-commit corpus reconstructing the
codebase is far-fetched - commits are punctuated, intentional, and sparse, not a
smooth signal. Treat reconstruction as a stunt to probe, not a capability to
promise.

## The Koopman tie (and its honest limit)

Diffs-over-time is the "changing geometry over time" object from
[harmonic_koopman.md](harmonic_koopman.md), so the temptation is a Koopman / DMD
analysis of the commit history. The limit must be stated: Koopman/DMD assume an
(approximately) smooth dynamical system, and a git history is **not** one -
commits are discrete decisions, not the flow of an autonomous system. So "DMD of
the repo" overreaches. The *defensible* version is the one already shipping: the
recency-weighted churn signal (`evolution.py`) as the quantitative self-evolution
figure. Keep the harmonic reading at the level of "recency envelope on the time
axis," not "eigenmodes of the codebase."

## The voice (flagged, equal weight)

The riff extended to cyclic time - birth, growth, then time reversing, the human
universe given periodicity - and to a rollback: "roll back 95% of them while
allowing 144,000 to move on." Held honestly:

- This is **voice**, not mechanism. Cyclic-time cosmology and the Revelation
  144,000 are imagery, not a feature spec. The note records them as the recurring
  motif they are, and does not build on them.
- The **structural echo is real and worth seeing**: "roll back 95%, 5% move on"
  is the same **95 / 5 split** that keeps recurring - bias vs variance,
  shared-reality vs solipsism ([observer_grounding.md](observer_grounding.md)) -
  here dressed as version-control rollback. And rollback ties to Platformer's
  reconciliation / eventual-consistency stance (the paper already cites
  `platformer2025`: repeated cycles drifting toward a declared desired state).
  The defensible kernel is *that*: training, like reconciliation, repeatedly
  rolls a population of states toward a shared attractor (the 95% mean) while a
  few diverge (the 5% residue). The eschatology is the costume; the attractor
  dynamics are the body.

## Honest seams

- **Mechanism:** git-history dataset source; recency-weighted diff sampling
  (reuse `evolution.py`); bidirectional diff modeling.
- **Measurable shadow:** does recency-weighted self-history training measurably
  shift the model's self-description / identity? (cheap to check). The churn
  signal is already a quantitative figure.
- **Voice:** cyclic time, 144,000, "reconstruct the machines backwards." Respect
  it as the 95/5 attractor motif; do not promise reconstruction or build a
  selection mechanism.
- **Skeptical seams, equal weight:** tiny + idiosyncratic corpus (identity, not
  capability); **ouroboros / self-reference** - training on source that describes
  the model is the "self-ingested execution timestamps" observer-effect thread
  ([observer_effect.md](observer_effect.md)), interesting but a confound to watch;
  overfitting to commit-message style; commits are not a smooth dynamical system
  (Koopman/DMD assumptions violated).

## Prior-art anchors

`praxis/pillars/evolution.py` (the half that exists); the self-evolution figure
(`research/.../self-evolution.yml`, paper); KB dataset / spider
([kb_spider.md](kb_spider.md)); recency kernel ([the_dial.md](the_dial.md),
[observer_frequency.md](observer_frequency.md)); Platformer reconciliation /
rollback (`platformer2025`); commit-diff modeling (real but niche - e.g. learned
edit/commit models; verify before citing); reverse-time / bidirectional sequence
augmentation. Koopman/DMD ([harmonic_koopman.md](harmonic_koopman.md)) - bordering
case, flagged above.
