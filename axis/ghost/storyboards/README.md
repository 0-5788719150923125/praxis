# Storyboards

A **storyboard** is a user-authored score for a song: an ordered list of scenes the
visualizer plays in sequence, instead of the automatic novelty scheduler choosing
for you. This is **manual mode** - orchestrating an entire visualization from data.

- **Auto mode** (no storyboard): the `Director` picks scenes by novelty weighting and
  cuts on the music. This is the default.
- **Manual mode** (a storyboard): the `Director` walks your `sequence` in order, and
  each entry decides how long it stays and what cue it exits on.

Run one:

```
godot --path axis/ghost -- --storyboard default         # loads storyboards/default.yaml
godot --path axis/ghost -- --audio ~/track.wav --storyboard default
```

(or pick it from the splash screen, in manual mode).

## Formats: YAML on disk, JSON for machines

A storyboard is **YAML** (the on-disk format - it holds comments) or **JSON** (what
the manual editor UI will emit); both parse to the same canonical structure, so
everything downstream is format-blind. A bare name resolves in this directory,
`.yaml` first, then `.json`; explicit paths are taken as-is.

The YAML dialect is a deliberately small subset (`scripts/yaml.gd`), and anything
outside it is **rejected with a line number**, never silently misparsed:

- `#` comments; block maps and lists nested by indentation (spaces only);
- single-line flow collections `[a, b]` / `{k: v}`, nestable;
- scalars: int, float, `true`/`false`, `null`/`~`, bare and quoted strings;
- NOT supported: anchors/aliases/tags, multi-document, block scalars (`|`, `>`),
  merge keys, flow collections spanning lines, tabs in indentation.

## Top level

| field         | type    | default | meaning                                                        |
|---------------|---------|---------|----------------------------------------------------------------|
| `name`        | string  | file    | label shown in logs / the splash.                              |
| `loop`        | bool    | `true`  | restart the sequence when it ends. `false` holds the final scene. |
| `transition`  | string  | `cut`   | default transition style (`cut` / `dip` / `fade`); a compatible morph always wins. |
| `sensitivity` | number  | `1.0`   | narrative **tempo**. Higher = faster: holds shrink and every scene's keyframe clock (including a stage `track`) compresses proportionally, so events still land. Ambient body animation never speeds up. Per-entry `sensitivity` overrides it. |
| `elastic`     | number  | `0`     | the timeline clock **breathes with the song**: energetic passages run a stage's keyframes up to `(1 + elastic)`x, quiet ones down to `(1 - elastic)`x. Zero-mean and endogenous (a fast energy EMA against a slow baseline), so the total stays anchored to the authored length - and with no audio the rate is exactly 1. Stamped into every entry; a per-entry `elastic` overrides. |
| `defs`        | map     | `{}`    | named reusable fragments - see [defs / use](#defs--use).       |
| `sequence`    | array   | -       | the ordered scenes (required, non-empty).                      |
| `tail`        | array   | `[]`    | entries the Director **cycles after a non-looping sequence ends**, instead of freezing the last frame - there is always something to roll into. Each cycles on its own exit rule until the song runs out (fixed `hold`s keep cycling even through quiet outros; cue exits pause in silence and near the song's end). Stage tails carry live actors like any other entry, so the cycle boundaries are invisible. |

### Sampled ranges ("cattle, not pets")

Anywhere a number is expected, `[lo, hi]` is accepted as a **range**, sampled once
per instance from the session seed. Same song, same show; a different seed rolls a
fresh variation of everything you left as a range. (A two-number array is therefore
always a range - positions are three-component `[x, y, z]`.)

### defs / use

`defs:` is a top-level map of named fragments. `use: <name>` inside any map
deep-merges that fragment UNDER the map's own keys (explicit keys win); a map that
is nothing but `{use: name}` becomes the fragment verbatim (so a whole `cast:` or
`track:` block can be a reference). Fragments can `use` other fragments (cycles are
caught). This is how a storyboard composes reusable pieces - camera rigs, gaze
tables, cast blocks - "a scene calling a scene".

## A `sequence` entry

| field      | type    | default        | meaning                                                                 |
|------------|---------|----------------|-------------------------------------------------------------------------|
| `scene`       | string  | -              | scene file in `scripts/scenes/` **without** `.gd`. Required. `stage` is the data-driven one below. |
| `behavior`    | string  | `"drift"`      | motion behavior: `static` / `drift` / `fluid` (see `GhostScene`).      |
| `shot`        | string  | by framing     | camera framing: `centered` / `offset` / `push_in` / `pull_back` / `pan` / `canted`. |
| `seed`        | int     | derived        | pin the scene's seed for an exact look. Omit for a song-derived seed.   |
| `sensitivity` | number  | storyboard's   | override the tempo for this one entry.                                  |
| `transition`  | string  | storyboard's   | transition style leaving this entry.                                    |

**Exit rule** - how long the entry stays and what ends it. Pick one:

| field       | type   | meaning                                                                         |
|-------------|--------|---------------------------------------------------------------------------------|
| `hold`      | number | fixed seconds, then cut. Deterministic timing - ignores the music.              |
| `exit`      | string | land the cut on a musical cue: `beat` / `movement` / `lull`. Tune with `min_hold` / `max_hold`. |
| *(neither)* | -      | the scene's own lifecycle (oneshots end themselves; loops use the beat cue).    |

## The `stage` scene: behavior as data

`scene: stage` is the **data-driven renderer**: the entry itself describes what is
on screen (`cast:`) and what happens over time (`track:`). Nothing about the show
lives in scene code - performers come from the **Cast registry**
(`scripts/cast.gd`), verbs from the **Actions registry** (`scripts/actions.gd`),
the timeline from `scripts/track.gd`.

```yaml
- scene: stage
  behavior: static
  hold: 4
  camera: {eye: [0, 0, 4.0], look: [0, 0, 0], fov: 48}
  cast:
    - {id: left, kind: eye, at: [0, 0, 0], radius: [0.30, 0.40]}
    - {id: right, kind: eye, at: [0.62, 0, 0], hidden: true}
  track:
    nominal: 4
    spans:
      - {from: 0, to: 4, action: look, target: [left, right], args: {use: gaze-calm}}
      - {at: 0.5, action: blink, target: left, args: {duration: [0.28, 0.40]}}
```

### Continuity (`carry`)

A stage declares `morph_in = morph_out = "stage"`, so **consecutive stage entries
morph**: live actors with **matching ids** continue across the cut - the same body,
pose, gaze, tendrils, and verb latches - while their slots come from the new entry's
cast. Actors that appear mid-entry are pre-declared `hidden: true` (verbs reveal
or replace; they never construct, so ids stay stable and sampling stays at build
time). Set `carry: false` on an entry to open with a clean cut instead.

### `cast:` - actors

Common fields: `id` (unique, required), `kind` (registry key), `at: [x, y, z]`
(world slot; the default camera at `z = 4`, fov 48 puts `x = ±0.62` in the two
side-by-side slots), `hidden`, `seed`.

| kind    | params | notes |
|---------|--------|-------|
| `eye`   | `radius`, `hue` | the photoreal `EyeBody`; gaze via the `look` verb (all targets share one focus = real vergence), eyelids via `blink`. |
| `prism` | `scale`, `hue`  | the living wireframe `PrismBody` (blue ≈ 0.6, red ≈ 0.0). |
| `swarm` | `count`, `count_red`, `spacing`, `head`, `size`, `helix`, `r_min`, `r_max`, `hue`, `hue_red`, `bank` (`left`/`right`/by seed), `lead` (actor id whose carried prism becomes member 0) | a GROUP actor: the formation-on-a-track from the finale. |

### `track:` - the timeline

`nominal` is the design length in seconds; span times are authored in it and scale
by `hold / nominal` (and `sensitivity`), so the timeline compresses as one piece.
Each span:

| field    | meaning |
|----------|---------|
| `at` / `from`+`to` | a point event, or a window (nominal seconds). |
| `action` | an Actions registry key (below). |
| `target` | actor id, list of ids, or `all` (default). |
| `args`   | the verb's config; ranges welcome. |
| `ease`   | `linear` / `smooth` / `in` / `out` / `spike` - shapes the window fraction. |
| `on`     | optional musical cue gate: `beat` / `movement` / `lull`. The span arms at its start and fires on the cue's next rising edge - the drop lands ON the drop. |
| `by`     | cue backstop (seconds after arming, default 1.5): fire anyway if the cue never comes. |
| `sustain` | `true` = after the window closes, keep applying at k = 1 for the rest of the scene instead of finishing. For motion that must never park: the finale's `fly` sustains, so the swarm keeps streaming while the end card holds. |

### Verbs (the Actions registry)

| verb | does | key args |
|------|------|----------|
| `look` | eyes track a wandering 3D focus (shared = vergence); depth-tier table with per-fixation ranges | `tiers`, `lateral`, `stray` |
| `blink` | close and reopen the eyelids once | `duration` |
| `split` | one eye divides into two identical ones easing to their slots | `into`, `slots`, `radius` |
| `crystallize` | an eye dissolves while a prism forms on its slot (form-flash at the crossover) | `into` |
| `tremble` | the riser: vibration + light building (latches) | - |
| `burst` | a prism bursts into being in a flash, optionally replacing an actor | `replaces` |
| `lock` | phase-lock pose to another prism (snap tie; latches) | `to`, `snap` |
| `desync` | break the phase-lock | - |
| `hold_still` | damp a prism's own spin to rest (latches) | - |
| `sway` | slip the anchor and drift weightlessly around it | `amp` |
| `specialize` | scale to `size` x, tempo to `tempo` x, latch a pulse | `size`, `tempo`, `pulse {amp, rate}` |
| `gather` | swarm members ease in one at a time | - |
| `fly` | the swarm flies its track; the stage camera follows a touch slower | `speed`, `follow` |
| `helix_split` | the track opens into a double helix; the red strand fades in | - |
| `lane_jump` | bank and leap to the other strand | - |
| `set` / `ramp` / `pulse` / `flash` | generic: set/ease any actor param (`scale`, `fade`, `hue`, `time_scale`, `drive`, `dilate`, `lid`), a scale pulse, a flash overlay | `param`, `value` / `from`, `to` |

## The boards here

- **`default.yaml`** - "the-point": the full 33s brief, five stage entries covering
  all fifteen timecoded beats, chained by carries, plus a two-entry `tail` that keeps
  the swarm streaming and weaving under the end card until the song runs out.
- **`stage-test.yaml`** - dev fixture: one entry per verb family, for render checks
  (`python build/scratchpad.py board stage-test ...`).

The five bespoke the-point scene files (`eye`, `two_eyes`, `eye_prism`,
`two_prisms`, `prism_swarm`) remain in the AUTO catalogue, but the storyboard path
runs entirely on `stage` data now.

## Roadmap

The stage spec is the foundation the manual editor writes to (JSON), and the
semi-automatic mode's dials are its sampled parameters surfaced live. Next rungs:
per-entry parameter dials in the Workspace, more actor kinds (rock, terrain,
layers-as-actors), verb-level `on:` cues throughout, and a `kind: stage` group
actor if a board ever needs true nesting.
