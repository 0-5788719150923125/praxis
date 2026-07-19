# Multi-scale memory banks - the harmonic fingerprint

> Status: **idea, deliberately parked** (2026-07-18). Captured so it isn't
> lost; no implementation planned right now, by explicit decision. Sibling to
> [integration_backlog.md](integration_backlog.md) (the dump this was pulled
> from) and [harmonic_koopman.md](harmonic_koopman.md) (the fixed-eigenbasis
> frame it would live in).

## The mechanism (as first stated)

Memory banks of expanding dimensionality [2D, 3D, 4D, ...], controlled
through harmonics. Each scale is an overtone of the base frequency, and
higher dimensions are progressively dampened by harmonic filters:

- **2D bank**: full resolution, all features active - context.
- **3D bank**: keep only frequencies where `f mod 3 == 0`, ~67% dampened.
- **4D bank**: `f mod 4 == 0`, ~75% dampened.
- **Higher**: increasingly sparse, increasingly selective - abstraction.

The sparse features that survive each filter are the invariants at that
scale. Together the banks form a multi-scale harmonic hierarchy: the
full-resolution bank provides context, the dampened banks provide
compression.

## The refinement (the part that matters)

The first statement talks as if dimensionality is something the banks *add*.
It isn't. **A geometry is already multi-dimensional, regardless of its
matrix size.** A point landing close to where it belongs is still part of a
higher-dimensional shape - the shape exists whether or not any bank is wide
enough to render it. So the banks don't construct scales; they **read** the
scales that are already there, at different grains. Each harmonic filter is
a choice of which few coordinates to look through, and the surviving sparse
features still pin down the object - a fingerprint still defines the human.

That reframing changes what the experiment would test. Not "does adding
dimensions help" but: does the *same* geometry, read through progressively
stricter harmonic filters, stay identifiable? If yes, the invariants are
real and the hierarchy is a free multi-scale readout. If a dampened bank's
features stop tracking the full-resolution bank's geometry, the filter was
discarding signal, not noise, and the overtone story is wrong.

## If it ever gets built

- Natural home: the Titans memory stack (`praxis/memory`), where
  `mal_energy_dual` already runs two regimes side by side - a dampened
  overtone bank would be a third resident, not a new organ.
- The instrument exists: Serpent per-feature frequency spectra and the Hoyer
  concentration measure already read harmonic occupancy
  (`praxis/pillars/inlines/mtp-field-concentration.yml`).
- Registry discipline applies: bank count/filters as registry-profile
  constants, not new config scalars.
- Falsifier sketch: probe task identity (or patch identity) from each bank
  independently; the claim dies if decodability collapses faster than the
  dampening ratio predicts.
