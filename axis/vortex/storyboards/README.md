# Runbooks

A **runbook** is a user-authored score for a song: an ordered list of scenes the
visualizer plays in sequence, instead of the automatic novelty scheduler choosing
for you. This is **manual mode** - the first step toward orchestrating an entire
visualization by hand.

- **Auto mode** (no runbook): the `Director` picks scenes by novelty weighting and
  cuts on the music. This is the default.
- **Manual mode** (a runbook): the `Director` walks your `sequence` in order, and
  each entry decides how long it stays and what cue it exits on.

Run one:

```
godot --path axis/vortex -- --runbook default         # loads runbooks/default.json
godot --path axis/vortex -- --audio ~/track.wav --runbook default
```

(or pick it from the splash screen, in manual mode).

## Format

A runbook is a JSON object:

```json
{
  "name": "demo",
  "loop": true,
  "sequence": [ { "scene": "...", ... }, ... ]
}
```

| field      | type    | default | meaning                                                        |
|------------|---------|---------|----------------------------------------------------------------|
| `name`     | string  | file    | label shown in logs / the splash.                              |
| `loop`     | bool    | `true`  | restart the sequence from the top when it ends. `false` holds the final scene. |
| `sequence` | array   | -       | the ordered scenes (required, non-empty).                      |

### A `sequence` entry

| field      | type    | default        | meaning                                                                 |
|------------|---------|----------------|-------------------------------------------------------------------------|
| `scene`    | string  | -              | scene file in `scripts/scenes/` **without** `.gd` (e.g. `"planes"`). Required. |
| `behavior` | string  | `"drift"`      | motion behavior: `static` / `drift` / `fluid` (see `VortexScene`).      |
| `shot`     | string  | by framing     | camera framing: `centered` / `offset` / `push_in` / `pull_back` / `pan` / `canted`. Omit to let the scene's framing class choose. |
| `seed`     | int     | derived        | pin the scene's seed for an exact look. Omit for a song-derived seed.   |

**Exit rule** - how long the entry stays and what ends it. Pick one:

| field       | type   | meaning                                                                         |
|-------------|--------|---------------------------------------------------------------------------------|
| `hold`      | number | fixed seconds, then cut. Deterministic timing - ignores the music.              |
| `exit`      | string | land the cut on a musical cue: `beat` / `movement` / `lull`. Tune with `min_hold` / `max_hold` (seconds) - won't exit before `min_hold`, will exit by `max_hold` even if the cue never comes. |
| *(neither)* | -      | use the scene's own lifecycle: a **oneshot** (e.g. `shatter_glass` crumble, `rocks` crumble) plays its sequence and ends; a **loop** uses the Director's default beat cue. `min_hold` / `max_hold` may still be given as bounds. |

Scene names come from `scripts/scenes/*.gd`: `spectrum_ring`, `harmonic_lattice`,
`rooted_growth`, `fog_lights`, `strata`, `bloom`, `filaments`, `wire_solid`, `planes`,
`voxel_blocks`, `cityscape`, `shatter_glass`, `gaussian_landscape`, `rocks`,
`embers`, `metropolis`.

See `default.json` (the runbook Manual mode opens with) for a worked score mixing fixed holds, musical exits, and a
self-ending oneshot.

## Roadmap

Today a runbook is a linear sequence. The intended direction (see the project
README) is richer: per-entry parameter overrides, time-coded cues tied to the
song, blends chosen per transition, and eventually a full manual editor that
writes these files. The data spec is meant to grow toward that without breaking
the simple linear case.
