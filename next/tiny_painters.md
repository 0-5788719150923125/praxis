# Tiny Painters: the Head's Plane as a Canvas for Time

> Status: **noted** (2026-06-07). A design intuition from the platformer's
> sequential-walk idea, carried over to the dashboard. Companion to
> [symbolic_chunks.md](symbolic_chunks.md) and the linear-solve lineage in
> praxis/heads/energy.py (LinearPrior).

Over in `../platformer` there is a notion of walking an architecture's
lifecycle sequentially - forward modes and recovery modes, stepped through
one at a time rather than summarized into a single aggregate curve. The
dashboard wants the same gesture. A metrics card today shows you the present
with a thin tail of history; what it should offer is the walk itself: cycle
backward and forward through the time history of one metric, one weight
shape, one geometry, the way you scrub a film strip. And the domain where
that scrubbing is natural is vision - which, on this dashboard, is every
domain. Everything we chart is already a picture.

The harmonic head hands us the canvas for free. Its field has a continuous
2D PCA plane - the cross-section the spiral and snake cards already project
onto - and that plane is the one surface in the framework where the model's
own sense of position and feature is laid flat. So let it be the ground:
a flat plane onto which the geometric structures of the architecture are
projected over time. Attention heads, optimizer state, latent clusters,
weight shapes - each gets drawn as a structure on the head's plane, frame
by frame across training, and the walk through the lifecycle becomes a walk
through pictures of the same place changing.

The main thing, though, is not the plotting. It is the learnable function -
the linear solve, again - and where it lives. It lives in the comparison
between a parent map and a child map, and the crucial fact is that the two
are not the same kind of map. The parent is the head's 2D PCA projection:
coarse, blocky, a handful of principal directions flattening a
high-dimensional field. The child is asked for pixel-detail output - a
small image in a format the parent never speaks. This is the human-level
move: a person comparing a subway diagram to a street map is not checking
pixels against pixels; they are holding two representations of one
territory, in different spectrums, and the understanding is exactly the
translation between them. The child cannot copy the parent. It has to
*twist* it - cross resolutions, cross formats, hallucinate detail the
blocky input only implies - and that twist is the learnable function. What
gets learned is not fidelity within a format but a correspondence across
formats: a twist in the internal world model, the same way a mind's map of
a place is not at the scale of any map it was built from.

The renderer is therefore a probe, not decoration, with the same logic as
the energy head's LinearPrior - solve what is solvable, and let the
residual tell you what the cheap map cannot capture. If a low-rank
convolution over a pixel-rebasing map can paint the fine picture from the
blocky coordinates, then the detail was already implied by the head's
geometry; the twist was learnable and the structure lives inside the plane.
Where the tiny painter fails - the pixels it keeps getting wrong, frame
after frame - is exactly where the architecture holds structure the
parent's spectrum does not span at any resolution. The reconstruction
error over time is itself the metric: a film of what the model knows about
itself, and a residual map of what it does not.

There is a pleasing recursion in it. We would be training miniature models
whose entire world is pictures of a larger model, judged by how faithfully
they can dream its portrait. The dashboard stops being instrumentation that
reads the model and becomes a small population that learns it - and the
cards we scrub through are their drawings, dated, hung in order down the
hallway of the run.

What this needs when it becomes work: a frame store (geometry snapshots per
cadence, the existing snapshot routes nearly suffice), the pixel-rebasing
map (a fixed, cheap rasterization of the PCA plane so the convolution sees
a stable raster), the low-rank solve itself (streaming ridge over frames,
the LinearPrior machinery generalizes), and the card chrome for walking
time - which the deck's momentum carousel already knows how to feel like.
