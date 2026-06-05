# The Snake on the Dial: a single sequence as a radial loop

Status: prototype (2026-06-04). Web-only proof of concept (Dynamics card
"Sequence Snake (on the dial)"); no LaTeX yet - prove the structure first.
Sibling to [the_dial.md](the_dial.md) (the dial / phase / circles-the-origin
geometry) and the clock-face / quadrant idea.

## What it is

One of the very few per-single-example readouts we have. The harmonic field over
a single sequence, sampled at **12 fixed fractional positions** (the same 12 no
matter the sequence length), drawn on a **clock-face of four quadrants**:

- **angle** = the field's PCA-2D phase at that position (the data decides where on
  the dial each sample lands),
- **radius** = **time**, collapsing toward the **origin** - newest sample at the
  center, so the sequence sinks inward as 12 discrete blocks (the vortex),
- the blocks are connected tail (oldest, outer) to head (newest, center): the
  snake.

## The falsifiable part: does the snake circle the origin?

The whole point is that we do **not** force the loop. We gate on the claim "the
snake circles the origin" and read whether the data actually produced it:

> winding = (1/2pi) * sum of the wrapped phase steps of the field's PCA-2D angle
> over the full sequence.

`circles_origin` is true only when `|winding| >= 1` - the field's dominant mode
completes a genuine full turn over the sequence. The card prints the number, so a
snake that coils without enclosing the origin (winding 0.9, say) reads as a
falsified loop, not a forced one. Whether it circles is a fact about the learned
transformations, not the visualization.

## Why radius = time (not magnitude)

Per the design call: time-to-center makes this a per-example structural probe. A
trained model that has learned a traveling-wave structure over position should
show the phase advancing monotonically as the blocks sink in - a clean inward
spiral that winds. A model that hasn't will show the blocks scattered in angle, no
winding, no circle. The radius carries the sequence axis; the angle carries the
learned phase; the winding tests their relationship.

## Grounded vs speculative

Grounded: the PCA-2D trajectory of the field (already the `spiral()` snapshot),
the phase, the winding number - all computed, all real. The clock-face/quadrants
and "vortex" are presentation. Speculative: that the *symbolic* content of a
sequence is legible in whether/how the snake winds - that is the conjecture the
card exists to let us look at, not a claim. Shares the dial thesis from
[the_dial.md](the_dial.md): a sequence as a position (here, a path) on a dial.

## Not done yet (deferred)

- Bias/variance radial coloring of the quadrant frame ("gradient cut in straight
  lines down the valley") - the snake is currently colored by time. A refinement.
- LaTeX figure (deferred until the concept holds on real trained runs).
- Per-example capture beyond the field's input-conditional state (the snake reads
  the field as conditioned by the latest forward).
