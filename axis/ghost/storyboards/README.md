# Storyboards

A **storyboard** is a user-authored score for a song: an ordered list of scenes the
visualizer plays in sequence, instead of the automatic novelty scheduler choosing
for you. This is **manual mode** - the first step toward orchestrating an entire
visualization by hand.

- **Auto mode** (no storyboard): the `Director` picks scenes by novelty weighting and
  cuts on the music. This is the default.
- **Manual mode** (a storyboard): the `Director` walks your `sequence` in order, and
  each entry decides how long it stays and what cue it exits on.

Run one:

```
godot --path axis/ghost -- --storyboard default         # loads storyboards/default.json
godot --path axis/ghost -- --audio ~/track.wav --storyboard default
```

(or pick it from the splash screen, in manual mode).

## Format

A storyboard is a JSON object:

```json
{
  "name": "demo",
  "loop": true,
  "sequence": [ { "scene": "...", ... }, ... ]
}
```

| field         | type    | default | meaning                                                        |
|---------------|---------|---------|----------------------------------------------------------------|
| `name`        | string  | file    | label shown in logs / the splash.                              |
| `loop`        | bool    | `true`  | restart the sequence from the top when it ends. `false` holds the final scene. |
| `transition`  | string  | `cut`   | default transition style for the whole storyboard (`cut` / `dip` / `fade`); a compatible morph always wins. |
| `sensitivity` | number  | `1.0`   | narrative **tempo**. Higher = **faster**: every scene's hold shrinks by this and each scene marches through its keyframe phases sooner (they are paced as fractions of the hold, so they always still land). It compresses ONLY the keyframe/cut clock - the **ambient animation** of the bodies (a prism's spin, an eye's saccades) is never sped up. Lower = slower, more deliberate. A per-entry `sensitivity` overrides it for one scene. |
| `sequence`    | array   | -       | the ordered scenes (required, non-empty).                      |

### A `sequence` entry

| field      | type    | default        | meaning                                                                 |
|------------|---------|----------------|-------------------------------------------------------------------------|
| `scene`       | string  | -              | scene file in `scripts/scenes/` **without** `.gd` (e.g. `"planes"`). Required. |
| `behavior`    | string  | `"drift"`      | motion behavior: `static` / `drift` / `fluid` (see `GhostScene`).      |
| `shot`        | string  | by framing     | camera framing: `centered` / `offset` / `push_in` / `pull_back` / `pan` / `canted`. Omit to let the scene's framing class choose. |
| `seed`        | int     | derived        | pin the scene's seed for an exact look. Omit for a song-derived seed.   |
| `sensitivity` | number  | storyboard's   | override the tempo for this one entry (see the top-level `sensitivity`). |

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

See `default.json` (the storyboard Manual mode opens with): the full "the-point" arc
(`eye → two_eyes → eye_prism → two_prisms → prism_swarm`) on timecoded fixed `hold`s, `loop:false`
so the final swarm holds for the end card. Consecutive entries **morph** into one another (each
scene's `morph_in` matches the previous scene's `morph_out`), so the eye and the prisms continue
across the cuts rather than being re-created. A fixed `hold` is honored exactly, above the auto-mode
silence/tail pacing gates - so a tightly-timed piece reaches its finale.

## Roadmap

Today a storyboard is a linear sequence. The intended direction (see the project
README) is richer: per-entry parameter overrides, time-coded cues tied to the
song, blends chosen per transition, and eventually a full manual editor that
writes these files. The data spec is meant to grow toward that without breaking
the simple linear case.
