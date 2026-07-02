# Information density breaks the boxes-and-arrows picture

> Status: **in the paper** (2026-07-01) - three paragraphs in "The outer
> representation" (`research/body.tex`), the two-strip density figure
> (`fig:density`, gaussian white/blue blur matching `fig:masks` block style),
> and a standing conjecture
> (`praxis/pillars/conjectures/information-density.yml`, order 19). The
> metric is still open - this note remains the spec for it, **but see
> [information_geometry.md](information_geometry.md) (2026-07-01): the
> falsifier as printed reads deviation in norm, and a silent bit flip
> (geometry changes, magnitude doesn't) may dodge it; the metric should be
> specified in symbol-occupancy coordinates as well.** Sibling to
> [recurrent_depth_concentration.md](recurrent_depth_concentration.md) (which
> predicts the deviation gets *sparser* across depth; this note predicts
> *where* it goes) and to the "outer representation" subsection of
> `sec:harmonic` in `research/body.tex`, which is the natural home for it in
> the paper.

## The picture

Every hundredth paper redraws the same figure: tokens in boxes, rendered
sequentially ("The quick brown fox jumps..."), little arrows hopping from box
to box. The figure encodes an assumption that is rarely stated: **uniform
information density**. Each token, at the embedding layer, is a self-contained
semantic unit, equally represented - to the best of the mean's ability. The
next-token objective reinforces it: the same loss lands at every position.

The claim: a harmonic model breaks that picture *structurally*. Information is
pushed toward the extremes of the window. At the back, a slow, echoing,
powerful hum - the low-frequency modes that span the whole context, the bias
axis, the attention sink. At the front, a chirp in otherwise perfect bias -
the high-frequency, input-conditional deviation riding the leading edge.
Hidden states are extremely stable at the start of the sequence and unstable
at the tip. "Information density" stops being a per-token constant and becomes
a *shape* - one that also shifts across depth, not just across position.

## The sharpening: emergent vs. constitutive

The strong form ("GPT cannot do this") is falsified by the literature before
it leaves the room: vanilla transformers *do* break the uniform picture -
attention sinks on the first token (StreamingLLM), massive activations, the
first-token norm explosion. Gradient descent finds this structure on its own,
crudely, in every large causal model.

That is not a defeat; it is the strongest available evidence *for* the claim,
restated: gradient descent is visibly straining to build a hum-and-chirp
density profile inside an architecture that gives it no coordinates for one.
The sink it improvises is a bug the architecture tolerates, not a feature it
provides. The harmonic model makes the same structure **constitutive**:
ghostmax is the head sink by construction, dropoff is the tip sink by
construction, the frozen phases fix which modes are slow and which are fast,
and the monotonic + periodic basis gives "back = hum, front = chirp" an
address. Encode the algorithm into the architecture and the model builds
solutions gradient descent would otherwise miss - or find only as an
unstable, undirected approximation.

Why gradient descent alone misses it: the gradient cannot tell bias from
variance - they are indistinguishable in a single scalar loss unless the
architecture separates their parameters (the cross-curvature conjecture,
`research/conjectures.tex`). It detects the significance of the split and
overshoots it. The distinction is architectural or it is nothing.

## Falsification

**Prediction: hidden-state variance increases with recurrent depth, with a
monotone positional gradient - early positions settle, tip positions keep
moving.** This extends through the Titans recurrence: harmonize anything and
the same profile should appear.

Operationalized:

- Per position $t$, per recurrent-depth step $d$: the update norm
  $\|h_t^{(d+1)} - h_t^{(d)}\|$ (or dispersion across inputs at fixed $t,d$).
- Predicted: (a) at fixed $d$, the profile rises from head to tip; (b) the
  profile *steepens* as $d$ grows - early positions converge toward a fixed
  point while the tip stays live.
- **Refuted** if the profile is flat in position, or if variance does not
  grow with recurrent depth. Either kills the density-shape reading outright.

Free corroboration already instrumented: the KL halting distribution is
per-position. If the picture is right, halting should fire early at early
positions and late (or never) at the tip - the halting profile *is* the
information-density profile, read through compute. Same for reading
`harmonic_delta_norm` / `concentration()` per depth step
([recurrent_depth_concentration.md](recurrent_depth_concentration.md)): that
note predicts the deviation concentrates; this one predicts the concentration
has a location, the rim, and mostly the tip.

## For the paper

This is the missing *figure* for the outer-representation subsection: redraw
the canonical boxes-and-arrows strip, then under it the harmonic version - a
density curve over the same strip, heavy and slow at the head, quiet through
the interior, sharp at the tip. Interior-running / interior-filling / rim
(Figure `fig:masks`) already makes the argument in mask space; this makes it
in density space, and it is the version every reader has prior exposure to,
because they have all seen the first panel a hundred times.

## Prior-art anchors

Attention sinks: StreamingLLM (Xiao et al. 2023); massive activations (Sun et
al. 2024). Internal: `sec:harmonic` "The outer representation" and the
ghostmax/dropoff sink pair (`research/body.tex`, `research/ghostmax.tex`);
cross-curvature conjecture (`research/conjectures.tex`);
[harmonic_koopman.md](harmonic_koopman.md) (stable/changing split);
[dropoff.md](dropoff.md); [oscillatory_axes.md](oscillatory_axes.md).
