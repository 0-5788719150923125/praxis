# RLCT Landscape + weight-geometry terrains

> Status: **built** (2026-06). Source `../RLCT.md`. The first loss-geometry
> visualizations in Praxis. Sibling to [harmonic_koopman.md](harmonic_koopman.md)
> and [temperament_bias_variance.md](temperament_bias_variance.md).

Three terrain cards on the Dynamics tab, one `rlct` group, all fed by
`RLCTLandscapeCallback` (`praxis/callbacks/lightning/rlct.py`) which probes every
`period` steps on the TRAINING thread (never a snapshot recipe - SGLD needs a
grad-bearing forward that would race the eval-mode guard), stashes onto the
uncompiled model (`_orig_mod`), and is drained by `DynamicsLoggerCallback` (scalars)
+ merged into `/api/head_snapshots` (grids). Compute core: `praxis/metrics/rlct.py`
(owns the universal `RLCT_METRIC_DESCRIPTIONS`). Tests: `tests/test_rlct_landscape.py`
(10, green).

## The three cards

1. **RLCT Landscape** (`probe_landscape` → `rlct_mesh`). A 2D loss slice through
   the live weights along two fixed filter-normalized random directions + a
   best-effort SGLD λ̂. This is a *scalar field over a slice* - inherently a
   smooth bowl near a minimum (correct, not a bug; the dramatic mockup was
   artistic). Color: cyan basins / red high-curvature ridges (Laplacian) / grey
   hillshade. λ̂ read relatively (phase-transition drops).
2. **Parameter Manifold** (`compute_param_manifold` → `param_manifold`). PCA of a
   structured weight's ROWS → 2D; height = density (the cloud SHAPE = geometry:
   blob vs ring/arms), color = amplitude. Tracks `rlct_manifold_var` (planarity).
3. **Parameter Field** (`compute_param_field` → `param_field`). The LITERAL weight
   terrain: each cell a real parameter at its native index, height/color =
   `|value|`. Native res unless > `field_max_cells` (128/axis), then max-pooled.
   For a harmonic head this is the actual spectrum as mountains.

`_pick_manifold_weight` (shared by 2+3) is **structured-first**: largest among
harmonic/crystal/field-named weights if any, else largest overall.

## Rendering

Manifold + field share `terrainMesh(canvas, spec)` in `dynamics.js` (one place for
the camera): close + oblique (zoom ~0.55, tilt ~40°) so the plane overflows the
card, gentle spin, cell-aspect preserved for non-square weights, grid lines
dropped on dense (>4000-quad) fields. `rlct_mesh` left standalone (bespoke
water/curvature color). Edit `web/src/`, rebuild `build.py`.

## Key decisions / gotchas

- **Perf (the step-stall):** probe was G² full forwards at full `block_size`
  (4096) → ~60s+. Fixed by `probe_len=256` (truncate the probe sequence; CALM pads
  to K internally, plain models accept any length - the 16x lever), `grid` 17,
  `probe_seqs` 2, `chain_steps` 6, `sgld_backoff` 1.
- **No per-run tuning:** all constants in `RLCT_DEFAULTS`, model-agnostic.
- **VRAM:** landscape clones base+2 dirs (~3x params transient), guarded by
  `max_params` (150M); manifold/field are weight-only (cheap, run even when the
  landscape skips).
- **Gated** to `is_lightning_module` trainers (backprop only).

## Follow-ups Ryan may want

- Target a *specific* head weight (not the heuristic pick).
- Whole-model flattened fingerprint as an alternate Field mode.
- Hessian-eigenvector directions for an anisotropic *loss* slice (offered, not
  built) - the only thing that makes card 1 not-a-bowl.