# MTP turns the vector into a curve - the sliding-window fox

> Status: **in the paper** (2026-07-18) - framing fragment
> `praxis/pillars/framing/mtp-vear-draft.yml` (gated on `mtp_type: vear`,
> `mtp_depth: 4`, so only an abstractinator-b-shaped run renders it), with its
> own figure (`fig:mtp-window`) - the sliding-window variant of the
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
