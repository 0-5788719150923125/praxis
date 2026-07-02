# Music box: the band is the edge of a spinning sphere

> Status: **raw capture** (2026-07-02). Spoken while looking at `fig:density`
> (the information-density strips). Successor imagery to
> [information_density.md](information_density.md) and
> [information_geometry.md](information_geometry.md); extends the watch of
> `sec:scaling` (which already names the frozen Weyl phases as the teeth) and
> the event-driven/sparse reading of `sec:harmonic`.

## The picture, verbatim

Imagine the harmonic "band" were the edge of a sphere - the edge of a coin,
the edge of gears. Spinning. At every edge, teeth on the sphere. Sparse
teeth, like a music box. Such that, as the gear turns, the "surprise teeth"
break the blocks at each end. They destroy the information-dense tokens
through a sign flip: existence, or non-existence.

## The mapping (nothing here is new machinery - that is the point)

- **The spin** is the standing rotation: frozen phases advancing at fixed
  rates, the Koopman unitary ([harmonic_koopman.md](harmonic_koopman.md)),
  the corpus rhythm. Pure bias. The gear turns whether or not anything is
  written on it.
- **The band as a visible edge**: the sequence window is not a line, it is
  the visible arc of a rotating periodic object. The strip in `fig:density`
  is a projection; the recurrence loops it closed. This is what the
  $e^{i\pi}+1=0$ over the figure was already saying.
- **Sparse teeth, like a music box**: the deviation is not a field, it is
  *pins on the drum* - a sparse scatter of discrete events riding a constant
  rotation. Exactly the event-driven conjecture (`sec:harmonic`: most
  positions null work, compute falls on the deviations) and the
  concentration prediction
  ([recurrent_depth_concentration.md](recurrent_depth_concentration.md)).
  A music box is the household proof that constant rotation + sparse pins =
  a melody: bias carries, variance plucks.
- **The teeth strike at the ends**: pins fire at the rim - head and tip,
  where the density lives - not along the quiet interior. Same claim as the
  outer representation, one image deeper.
- **The sign flip**: a strike does not nudge a token, it *negates* it -
  existence or non-existence. And we already drew the algebra above the
  band: $e^{i\pi} = -1$. Half a turn is negation. The silent bit flip of
  [information_geometry.md](information_geometry.md) gets its cleanest form
  here: the geometric event is a phase inversion, ±1 on the rotating field,
  invisible in norm (a sign flip preserves magnitude exactly - *zero*
  norm-deviation, maximal geometric deviation). "Surprise teeth" is the
  right name: the strikes are the high-surprisal events, and they land on
  the information-dense tokens because those are the ones worth destroying.
- **Existence or non-existence** is also literally the sink pair: ghostmax
  is the address of non-existence (attend to nothing), dropoff forces the
  densest token off the board. Destruction is not damage; it is routing to
  the zero sink.

## The new falsifiable edge this adds

The earlier notes predict deviations are sparse and rim-concentrated. The
music box adds a *third* structural prediction: deviations should be
**sign-like, not drift-like**. If teeth are real, the large deviation events
are closer to phase inversions than to gaussian perturbations:

- Instrument: for each deviation event (per feature or per position, per
  depth step), the cosine between the state before and after. Teeth predict
  a **bimodal distribution** - mass near $+1$ (the rotation carrying on,
  null work) and near $-1$ (a strike), with the valley between them empty.
- Refuted if the angle distribution is unimodal/small-angle (smooth drift) -
  then there are no teeth, just a field relaxing, and the music box is a
  metaphor rather than a mechanism.
- Note this instrument is *orthogonal* to both existing coordinates: norm
  misses a sign flip entirely (magnitude preserved), and even occupancy
  could blur one; the cosine sees it at full weight. Three coordinate
  systems now - norm, occupancy, angle - and the conjecture family is
  falsifiable in each.

Sibling closure: an "Always grokking" cluster hop and a "Development reverts
to the basis" punctuated jump are what a tooth strike looks like in those
conjectures' coordinates. One event, three shadows.

## Figure idea (later, if wanted)

`fig:density` part two: the strip bent into an arc - the edge of the coin -
teeth drawn sparse on the rim, striking the blocks at the two ends. The
current figure is the unrolled version; this would be the object it unrolls
from.

## Prior-art anchors

`sec:scaling` "The watch" (frozen Weyl phases = teeth; this note makes the
teeth sparse and gives them strikes);
[information_geometry.md](information_geometry.md) (the silent bit flip;
sign flip = its exact algebraic form);
[information_density.md](information_density.md);
[harmonic_koopman.md](harmonic_koopman.md) (spin = unitary);
[recurrent_depth_concentration.md](recurrent_depth_concentration.md)
(sparsity of the deviation); ghostmax/dropoff sink pair (`sec:harmonic`,
`research/ghostmax.tex`); the music box as the folk model of
program-on-a-rotating-drum.
