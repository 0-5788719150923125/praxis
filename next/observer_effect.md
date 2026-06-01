# The Observer Effect: Does Measuring the Harmonic Field Change It?

> Status: **ablation design** (2026-06-01). Three falsifiable experiments asking
> whether the act of *observing* a harmonic model perturbs what it learns -
> the measurement-level companion to the paper's architecture-level claim that
> "attention constructs which patterns manifest" (`research/main.tex`, §6, §8).
> Companion to [oscillatory_axes.md](oscillatory_axes.md), [harmony.md](harmony.md),
> and the swarm-dynamics roadmap entry.

## The question, stated honestly

Quantum measurement is the motivating analogy: the act of watching changes the
watched. The paper already cites it (Heisenberg, alongside attention-modulates-
cortex and predictive processing) as a *family* of "observation participates in
the observed" ideas, and flags the physics connection as speculative. There is
no wavefunction in the network. But the analogy points at mechanisms that **are**
real here, and the intuition that drove the question is sound: if the harmonic
field is a standing wave over the full (position, feature) computation - first
feature to last, across the whole sequence - then it is **non-local**. A
perturbation injected by a measurement at one cell does not stay local; it rides
the wave to every other cell. So a model whose representation is harmonic should
be *more* sensitive to impure observation than one whose features are
independent, not less.

The strong (quantum) form is metaphor. The weak form is measurable today, and it
resolves into three concrete artifacts - the first two real now, the third the
one this codebase is uniquely set up to test.

## Mechanism 1: non-pure measurement (real; already observed)

The clearest "observer changes the observed" event in the codebase is the
calm-c step-441 crash ([project_calm_stage2_eval_race]): the dashboard's
`/api/activation_curves` route flipped the shared model into `eval()` *mid-
forward* from a background thread, and because CALM stage-2 grad terms gate on
`self.training`, the act of observing the model fatally changed its computation.
That is a literal observer effect, and a mundane one: the difference between a
side-effect-free read and one that mutates state. Harmonics make it worse exactly
as the intuition predicts - the mutation is not contained to the observed cell;
it propagates along the field.

**Experiment.** Diff two runs from the same seed: one that snapshots the spectrum
/ samples the pool / reads activation curves on a fixed cadence, and one that
never observes. A *pure* read (detached, no mode flip, no in-place op) must
produce zero divergence in the trained field (spectrum concentration, the
amplitude grid, `harmonic_delta_norm`). Any measured drift means the instrument
has a side effect - and the magnitude of the drift, relative to a non-harmonic
control, measures how much the field's non-locality amplifies it. Falsifiable and
cheap: it is a regression test for measurement purity.

## Mechanism 2: precision artifacts on the exponential edge (real; most paper-relevant)

The paper's substrate is that the model is "kept numerically alive only by
normalization holding activations in the narrow band where the exponential is
both sensitive and stable" (§6). Observation forces dtype boundaries that a
pure forward pass would not: dashboard snapshots cast to float32/CPU, the
`irfft2` with ortho-norm is sensitive in its low-order bits, the JS swarm experts
are float64 while torch is float32. On the exponential edge, observation-induced
rounding can **amplify rather than wash out** - and that is precisely the regime
where the harmonic field is most plastic. This is the same physics the
[oscillatory_axes.md](oscillatory_axes.md) and [hash_gated_anchor.md](hash_gated_anchor.md)
threads turn into a feature (float-precision error as an addressing function, not
noise); here we ask whether it leaks in as an artifact of *watching*.

**Experiment.** Does the act of measuring introduce a reproducible floor or drift
in `harmonic_delta_norm` or `val_codec_bpb` that is absent when the same run
never casts-for-measurement? Compare snapshot-every-step vs never-snapshot; if a
floor appears only under measurement, that is an imprecision artifact of
observation, localized to the bits the snapshot path touches and spread by the
field. The honest null is that fp32 headroom swamps it and there is nothing to
see - which is itself worth recording.

## Mechanism 3: the swarm observer/participant ablation (the new one)

The orchestration pool (`praxis/orchestration/`) is, by construction right now, a
**passive observer**: it trains tiny experts on the model's real batches but
their gradients never cross back into the model (the `RemoteLayer` /
`ExpertPoolLayer` contribute-back seam is stubbed - see those modules). This is
not a limitation. It is a ready-made **control arm**. The observer effect becomes
a clean two-arm ablation, with the stub as the switch:

- **Observer arm (today):** the swarm watches - samples the model's batches,
  trains locally, reports - but does not participate in the model's forward.
- **Participant arm (draw the stub):** the swarm's vote folds back into the main
  forward as a structured-variance term (the bias/variance framing - the swarm
  supplies variance the resident prior lacks).

The observer effect is then any divergence in the *main model's* trained geometry
(spectrum concentration, delta-norm, loss path) attributable **solely** to
whether being-watched also means being-coupled. If the two arms' fields diverge
by more than measurement noise, watching-with-feedback demonstrably changes what
manifests - the paper's "attention constructs which patterns manifest" thesis
made measurable at the swarm level rather than the attention-head level. The null
(no divergence beyond noise) is equally informative: it would say the contribute-
back coupling is too weak to matter at this scale, and bound how large the swarm
must be before it does.

## Where this sits relative to the paper

This is the paper's "tractability is observer-relative" (§5.x) and "you exist in
others' realities through their geometry" (§8) carried down one level: from the
observer's *basis* (which sets what is tractable) to the observer's *act of
reading* (which the strong analogy says sets what is true). The honest,
paper-shaped position: the quantum form is a generative metaphor; the weak form
is three falsifiable experiments. Mechanisms 1 and 2 are measurement-hygiene
claims with a harmonic twist (non-locality amplifies impure reads); mechanism 3
is a genuine architectural ablation the codebase is already wired for. None
require physics to explain - they require a diff between a watched run and an
unwatched one, and a control that is not harmonic. That is the bar §7 sets: state
the idea, draw the real-vs-conjecture line, hand the reader the falsifiable form.
