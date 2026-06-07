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
the linear solve, again. The head's 2D projection map is only the input
side. The output side is a tiny renderer: a low-rank convolution over a
pixel-rebasing map, solved or trained to produce a small image from the
projected geometry. Tiny models predicting tiny images, trying to mimic the
head. The renderer is not decoration; it is a probe with the same logic as
the energy head's LinearPrior - solve what is solvable, and let the residual
tell you what the cheap map cannot capture. If a rank-k convolution can
reproduce the picture of a structure from the head's coordinates, that
structure lives inside the head's geometry; the image was already implied.
Where the tiny painter fails - the pixels it keeps getting wrong, frame
after frame - is exactly where the architecture holds structure the head's
plane does not span. The reconstruction error over time is itself the
metric: a film of what the model knows about itself, and a residual map of
what it does not.

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
