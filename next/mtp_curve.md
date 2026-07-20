# MTP turns the vector into a curve - the sliding-window fox

> Status: **in the paper** (2026-07-18, corrected 2026-07-20) - framing
> fragment `praxis/pillars/framing/mtp-vear-draft.yml` (gated on
> `mtp_type: vear` alone; the prose is K-generic since abstractinator-c
> raised K to 8, the figure is an explicit K=4 schematic), with its own
> figure (`fig:mtp-window`) - the sliding-window variant of the
> boxes-and-arrows fox strip. Sibling to
> [information_density.md](information_density.md), which owns the original
> fox chart (`fig:density`) and the density-as-shape reading this note extends
> to the generation scale.

## The observation

abstractinator-b's byte-level MTP (`mtp_type: vear`, K=4 bytes per two
passes) changes the unit of prediction. A next-token head emits a point
estimate per position. The MTP bank emits a short **curve**: four future
bytes read off one shared trunk state, through four sliding-window-merged
harmonic depth-transforms (depth k = uniform merge of experts [k, k+1, k+2],
cyclic; adjacent depths share two of three experts; repulsion keeps the
geometries distinct - `praxis/heads/mtp/vear.py`).

Because all K losses land on the same trunk state, supervision propagates
**backwards** within the window: the loss at position t+3 shapes the
representation at t. That is multi-token dependence - the property the fox
chart (`fig:density`) says the uniform next-token objective lacks, and the
property CALM bought with its unbounded, full-sequence continuous latent.

## The correction (2026-07-20, twice)

Two wrong drafts before the right one, both worth remembering:

1. **Disjoint blocks** (first figure): K-byte draft chunks tiling the
   sequence. That's the *inference* decode, not the hypothesis - speculative
   decoding tiles; training is dense.
2. **Windows as compressed clones** (second figure): each K=4 window
   carrying a miniature of the whole sequence, four ghost boxes tethered to
   the strip. Closer, but still treats the window as the object.

The right object: **each draft depth IS the full sequence, shifted.** Depth
k predicts t+k for *every* t at once, so its output is the sequence again,
shifted left by k - K near-duplicate full-sequence views, sharing
representation by construction (one trunk; adjacent depths share 2/3
experts), distinguished chiefly by their shift. Content compresses toward
the head as each shift retires the tip - information goes dense on the
left. Each depth is free to settle anywhere between the window's
granularity and the full sequence's span: multi-scale timeseries modeling.

The window then stops being a box of its own: a decode step is a **column**
through the stack, and one byte's K re-predictions (successive states
bolstering or refuting) are a **diagonal**. Both slices fall out of the same
lattice, which is what makes the stacked figure readable where the ghost
boxes weren't. The column's width carries the mode distinction: standard
byte-latent decoding is a one-cell column (one byte per forward); MTP
advances a **K-wide** front through every depth at once. And the top of the
figure keeps the frequency wave from the density chart, because the
representation is frequencies throughout.

Final round of refinements (also 2026-07-20):

- **Rows slide by K, not 1** - because MTP advances K per step, the rows are
  successive *steps* (t+K, t+2K, ...), each re-deriving the full sequence
  from K further along. Needed a longer sentence: "The quick brown fox
  jumps" (25 bytes). The outlined band became a fixed K-wide window the
  stream slides through, one column per step.
- **Spaces restored as teeth** - byte-latent patching cuts on them, so each
  space stays in the lattice as a byte-wide *white* cell: no content, pure
  entropy. Full width, not half - a space is a byte and occupies a real
  position, and full-width teeth keep the column grid aligned across rows
  shifted by K.
- **Density localized per patch** - heaviest at each patch's first byte
  (where the information arrives), decaying through the predictable tail;
  each step's row runs denser than the last, so density grows over T+N as
  context accumulates. Drawn as chained per-cell gradients (fig:density's
  technique) so each patch is one smooth gradient, times a ramp rising
  along T - later positions compress the full sequence so far, so every
  row darkens toward its tip - times the per-step multiplier. Every patch
  in every row decays to genuine WHITE (a white stretch where little
  information lives) - t+4K included, by explicit decision: it is the
  densest row with the narrowest white, but never fully saturated. Don't
  force color into it. Teeth stay white everywhere (they are boundaries,
  not gradient).
- **Entry/exit corner arrows** - both arrows kept, but only ONE label each
  side, on two label rows total (four rows read as clutter): top line is
  "the sequence..." left + "enters with velocity" right, above the
  horizontal wave; bottom line is "phase-locked at the head" left (with the
  exit arrow) + "pull K through each step" right. The "exits with
  stability" label is gone - the arrow carries it. Each arrow projects from
  just off the END of its own frequency band (entry off the horizontal
  wave's right end, exit off the vertical wave's bottom end), so both sit at
  matching offsets and read as the fields continuing out of frame.
  The diagonal is fixed by the figure, not by taste: rows
  slide left, so a byte's trajectory is top-right -> bottom-left; the
  strands run that way; density is heaviest bottom-left; and body.tex
  already states hidden states are "extremely stable at the head and
  unstable at the tip", so velocity belongs at the right and stability at
  the left. The mirrored placement (entry top-left / exit bottom-right) was
  proposed first and is wrong on all four counts.
- **The echo window** - a dotted clone of the K-wide band at 1.25x, anchored
  by its TOP-RIGHT CORNER at the centre of the band's top-right token box
  (cell 11 of the trunk row). That anchor is the point: the echo's angle
  nests inside the first window's angle, which is what makes the scaling
  legible - a centre-scale-plus-small-offset version was too subtle to
  read. Larger because it is nearer - the same window one scale out. Carries
  the exponential/multi-scale claim with no label at all: the window is a
  rung, not a fixed span, so a single K encodes the whole landscape's shape
  rather than one altitude. The lattice's own label row keeps its line just
  under the final row and the echo's dotted edge passes BEHIND it (labels
  are drawn after the echo) - the echo must not push the lattice's labels
  around. Only the title/subtitle sit clear below the echo's lower edge.
- **Vertical phase-locking** - a second frequency wave runs down the depth
  axis, left of the blocks and right of the t+K labels: the same field,
  read across the steps. Must be clearly periodic like the horizontal one:
  low-frequency hum at the trunk, high-frequency chirp into t+4K.
- **Strands replace the token diagonal** - wiggly lines (snake decoration;
  `decorations.pathmorphing` now loaded in `research/main.tex`) from the
  left edge of the K-wide window band at each row to the first token of the
  final sequence, drawn white with a black casing (like the glyph contours)
  and ABOVE all other elements so they stay visible. They attach at the
  center of the rightmost content byte inside the band per row (never a
  tooth or dashed cell), not the band's edge: every window connected to both the future and the past of
  every sequence - phase-locking, every feature contributing to every
  representation. (Not from the row tails - that was wrong.) Four strands,
  not five: the final row is what they all resolve into, so it draws none of
  its own - a strand from it to itself curved the wrong way, which was the
  tell. The strands encode their own direction as a chirp: wide slow wobble
  at the source, near-smooth through the middle, tightening into fast
  oscillation as they integrate at the target. Drawn as explicit parametric
  plots (not the uniform `snake` decoration, which reads directionless),
  with the offset taken perpendicular in *visual* space since the tikz axes
  are unequally scaled (x=0.36cm, y=0.44cm).
- **Characters melt, like fig:density's** - glyphs are not bound to their
  token boxes; per patch they pool toward the head with the density chart's
  power-law drift (`x = p + 0.3 + (L-0.6)*(i/L)^1.8`),
  spilling over cell walls and leaving the right of every window sparse.
  Box-centered glyphs sold the wrong idea - uniform density. Horizontal melt
  only: the vertical baseline drift the other charts use was distracting
  here, so glyphs stay vertically centered. The melt tightens per row - each
  patch's glyphs squeeze toward *its own* patch head (never a global pull
  toward the strip head; glyphs must stay inside their colored patch), until
  t+4K stacks each patch's bytes on a single position.

## The contrast with CALM

We sought the Abstractinator as an approximation of CALM, and this is the
axis on which the approximation earns its keep:

- **CALM**: multi-token dependence via an unbounded continuous latent - the
  full-sequence field. All coupling, everywhere, at once. A rocket.
- **abstractinator-b**: the same dependence over a **bounded, periodic**
  window of K=4 - local, cyclic, parameter-shared, riding a ~95% discrete
  substrate (byte-patch residual codes) with a ~5% continuous remainder (the
  fuzz). A rollercoaster.

"Gentler" is the design claim; whether gentler is *better* is an empirical
one, and the honest scoreboard is convergence behavior plus the per-depth
**MTP Draft Accuracy** cards. The accept-rate profile across draft depths is
the within-window information-density curve, measured - the fox chart's
argument recurring at the generation scale, this time with an instrument
already logged.

## Pointers

- `experiments/abstractinator-b.yml` - the run config and its MTP comment block
- `praxis/heads/mtp/vear.py` - the sliding-window bank
- `praxis/pillars/inlines/mtp-field-concentration.yml` - the Serpent-spectrum
  Hoyer inline that already fed the paper
- [information_density.md](information_density.md) - the parent reading
- [integration_backlog.md](integration_backlog.md) - where the rest of the
  2026-07-18 idea dump lives
