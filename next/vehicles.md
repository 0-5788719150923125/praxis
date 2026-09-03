# Vehicles: what carries the show

ghost has one presentation. Every mode that runs the Director (Auto, Manual, Synthesis,
Generative) paints ONE scene, full-bleed, edge to edge, and cuts to the next one. That is a
choice the code has never had to name, because there was only ever one of it.

A **vehicle** is that choice made addressable: the substrate the show is carried on. `full` is
today's behaviour, unchanged and default. `comic` renders the same scenes into the panels of a
comic page, and flies a real perspective camera over that page.

Not a mode. Modes decide what *drives* the show (a song, a storyboard, a voice). A vehicle
decides what the show is *presented as*, and every mode gets both.

## Why it is possible at all

Four things already in the codebase do almost all of the work:

- **Scenes are viewport-relative and aspect-agnostic.** `GhostScene.size` comes from
  `get_viewport_rect()`, geometry sizes off `unit() = min(size)`, and layers work in
  unit-fraction space "so they fill any aspect ratio without baring an edge". A scene rendered
  into a 512x384 panel is correct with no scene-side change whatsoever.
- **The stage is already a SubViewport, and the governor already freezes it.** `main._stage_host()`
  hands the Director a SubViewport; the stage governor already flips `render_target_update_mode`
  to `UPDATE_ONCE` and `process_mode` to `DISABLED` to spend less on the picture. The same two
  properties, per panel, are how a filled panel HOLDS ITS PICTURE for free - no readback, no
  `get_image()` (which is exactly the synchronous stall that cost Masking its frame rate).
- **`Lens3D` is a real perspective camera** projecting world points to centred unit-fraction
  screen space, and `Plane3D` is a real quad in that world. The page is a Plane3D. Rotating it on
  X, Y and Z is not a skew trick - it is the actual projection, which is why the user's instinct
  that "we have never laid a plane at other angles" is worth acting on.
- **`TriBatch` already submits TEXTURED triangle arrays** (`uvs` + `tex_rid` ->
  `canvas_item_add_triangle_array`). A panel is a subdivided grid of those triangles with the
  panel's own SubViewport texture on it.

## The mechanism

```
_stage (SubViewport)                     the composited picture, as today
└── ComicVehicle (Node2D)                draws the paper, the panels, the ink
     ├── slot 0 (SubViewport)  <- a GhostScene lives here, live or frozen
     ├── slot 1 (SubViewport)
     └── ...                              two pools of these (page A / page B)
```

Per frame the vehicle: eases the page's basis and the lens, projects the page quad, then for each
FILLED panel projects a subdivided grid of its rect in page-local space and submits it as one
textured draw call, UVs from the grid. Rounded corners are paper-coloured wedges painted back
over the square corners (exact under perspective, because they are computed in page space and
projected like everything else); the border is a rounded-rect polyline on top.

### One live panel, the rest held

The Director runs exactly one scene at a time (two during a transition). The comic keeps that:
the newest panel is live, and every panel behind it is a **frozen render target** - its viewport
stopped updating, its scene freed, its last frame still in VRAM. That is not a compromise dressed
up as a feature. A comic panel *is* a held moment; the page is a sequence of them; and the frame
is never static anyway because the camera and the page are always moving. It also means the comic
costs the same as the full vehicle, which is the only reason it can ship at all - the stage
governor already struggles with one heavy scene, and six live ones is not a thing that runs.

Phase 2 can round-robin a little life back into the held panels. Phase 1 does not need it.

### A cut is a panel, a full page is a page turn

Every Director cut advances to the next panel slot. When the page fills, the next cut turns the
page. So `Scene hold` and `Flourishes` keep meaning exactly what they mean today, and a burst of
quick cuts becomes a run of small panels filling in fast - which is what a comic does with a
fight.

Transitions survive, made panel-local: the OUTGOING alpha is held at 1 (the old panel must not
blank - it is already drawn), and the incoming fades up inside its own panel. A panel developing
on the paper. Morphs are kept untouched and are better here than anywhere: one eye in panel 3
splitting into two in panel 4 is continuity across a gutter, which is a thing comics actually do.
LAYER falls back to a fade (two scenes composited in one panel is just mud).

### Everything is sampled, nothing is drawn

Per the house rule. Page layouts are rolled from `hash([session_seed, "comic", page_index])`:
panel count (3-6), the row/column split, per-panel aspect (clamped 0.5..2.2 so a subject is never
cropped through), gutter width, corner radius, whether a panel is canted a couple of degrees,
whether this page has a splash panel. Page attitude - the X/Y/Z rotation - is a sampled rest pose
with slow ModBank oscillators around it, so the page breathes rather than sitting still.

### The camera over the page

A small seeded shot vocabulary, eased (never snapped), chosen per page:

- `read`  - tight on the newest panel; each cut is a slow pan across the gutter to the next one.
- `spread` - the whole page in frame, gently drifting, page tilted.
- `oblique` - page rotated hard on two or three axes, camera close, panels foreshortened.
- `flat`  - near square-on, a slow push in.

### The bookend

The whole-show fade from and to black currently rides `_current.modulate.a`. In a comic that
would fade one panel and leave the paper lit. The vehicle takes ownership of the bookend and
modulates its own root instead; the Director stops applying it when a vehicle claims it.

## What has to change in the Director

Small and countable. `_host.add_child(...)` at four sites becomes a router the vehicle answers,
`_choose_style` and `_transition_alphas` get a vehicle veto, and `_bookend_fade` gets an owner
flag. Nothing else. `full` returns the stage from the router and vetoes nothing, so its behaviour
is byte-identical to today's.

## Checklist

### Phase 0 - prove the two engine assumptions before building on them - DONE
- [x] `tests/vehicle_probe.gd`: a SubViewport set to `UPDATE_DISABLED` keeps its last frame after
      its child scene is freed. **Measured: drift 0.0000 after 8 frames with the scene freed.**
      The freeze is exact and free.
- [x] Same probe: a ViewportTexture RID draws through a projected, subdivided grid.
      **Measured: the affine (1x1) seam lands 28.5 px from the perspective-correct (8x8) one on a
      512 px frame**, so subdivision is not optional.

### Phase 1 - the vehicle axis - DONE
- [x] `scripts/vehicle.gd` - the base + `VEHICLE_REGISTRY` (`full`, `comic`), labels, blurbs.
- [x] `scripts/vehicles/full.gd` - the identity vehicle; every veto returns its argument.
- [x] Director: `_scene_host` router, `style_for` veto, `hold_outgoing`, `owns_bookend`,
      `begin_session`. Four call sites, no other control flow.
- [x] main: builds the vehicle, mounts it in the stage, passes it to every `Director.attach`,
      and rebuilds it when the setting changed (`_sync_vehicle`).
- [x] `Director.vehicle` + `set_vehicle()` + `resolved_vehicle()`, persisted to
      `[director] vehicle`, overridable with `--vehicle NAME`, forwarded by the exporter.
- [x] **Regression: `scene_mix_check` replays one seed and reproduces all 140 cuts exactly**, so
      rewriting `_choose_style` around the veto did not perturb the seeded stream.
      `sting_shape_check` also passes.

### Phase 2 - the comic page - DONE
- [x] `scripts/comic_page.gd` - the seeded page roll: 3-6 panels, rows then columns, uneven
      splits, gutters, margin, corner radius (weighted toward hard corners), the occasional cant.
      Rolls are REJECTED and retried when a panel aspect would crop a subject through.
- [x] `scripts/vehicles/comic.gd` - the two slot pools, the page basis, the lens, the draw.
- [x] Panel rendering: subdivided textured grid, aspect-matched render targets at constant AREA,
      half-texel UV inset, rounded corners by paper wedges, ink border.
- [x] Freeze-on-advance, page turn hinged on the SPINE, two-pool alternation.
- [x] Shot vocabulary (`read` / `spread` / `oblique` / `flat`) with an exactly-solved framing
      distance and eased attitude.

### Phase 3 - the surface - DONE
- [x] Vehicle picker in the Generative panel's "THE PICTURE" section, built off the registry.
- [x] `tests/comic_look_probe.gd` - drives a real comic session and writes PNGs; asserts only
      that the frame is not uniform (the failure most likely to ship silently).
- [x] `python docs.py` (`docs/vehicles.md`, the CLI flag, the layout block) + README section.

## Second pass (the first one was framed wrong)

The first build shipped a comic that worked and read badly, on three counts, all correct:

**"At least 60% of every page is dead zone, blackness."** It had a wide `spread` shot that
framed the whole page, and a portrait sheet in a landscape frame cannot fill it. There is now
NO wide shot at all. The camera is fitted by two constraints and takes the NEARER: close
enough that the panel spans the frame vertically, and close enough that the PAGE covers the
frame's width. So the panel is cropped by the screen and the sides are gutter and neighbours.
Behind all of it is a vignetted desk wash rather than black, with the sheet's own drop shadow
on it - black is not a background, it is the absence of one.

Two measured mistakes on the way there, both of which read as framing bugs and were not:
- `_fit`'s `fill` runs the other way round from how it reads - a SMALLER fill backs the camera
  off - so the first values (0.8-1.04) meant "mostly zoomed out".
- The near limit `CROP_MAX` came out FURTHER from the page than the covering distance did
  (1.66 against 1.24), so the floor, not the framing, decided every shot and vetoed exactly
  the push-in that fills the frame. A floor that binds in the ordinary case is the rule.

**"Multiple panels, available from the start, and all moving at the same time."** The page now
owns its cast: `Vehicle.owns_cast` / `take_over`, and `Director.mint_scene`. Every panel is
filled when the page turns (one per frame, so the build does not hitch), and a Director cut
means MOVE THE READING, not build a scene. Liveness follows the camera - panels in shot run,
panels out of shot are stopped targets - and the off-focus ones repaint every other frame at a
matching multiple of the step, so they run at the right speed with half the samples.

**"The pages rotate on a single axis, and just barely."** Attitude is now sampled at real
magnitude on all three axes and DRIFTS continuously, and the eye sits at a page-local azimuth
and elevation rather than on the page normal - an eye on the normal sees a flat rectangle
however the sheet is turned. Holding that station while the aim moves is what makes travel
between panels a move parallel to the paper.

### What it costs now

Three live scenes instead of one: **73 ms of active frame against the full vehicle's 41**, on
one seed with the same scenes. The earlier claim that the comic costs what the full frame
costs was true of the one-live-panel build and is not true of this one; the README says so.
The stage governor absorbs the difference, which is what it is for.

## Third pass: the camera is a vocabulary, not a target

Reported: "the camera just bounces around between frames, focusing on each one. But that
focus is the whole problem: a constant shot of a single frame should be an event, not the
rule." Correct, and the cause was structural - the camera EASED TOWARD A TARGET, which is a
spring settling, and a spring has exactly one behaviour.

A cut now picks a MOVE from a weighted bag and the camera TRAVELS it: a start state, an end
state, a duration, a curve. Travelling moves are ~70% of the bag and none of them settle -
`drift`, `track` (through the subject and out the far side, the shot that follows a car),
`sweep_h`, `sweep_v`, `push`, `pull`, `orbit`. The discontinuities and arrivals are the other
30% - `swoop` in low off the far side of the page, `whip`, jump `cut`, `dip` through black -
and `hold`, the one that just looks at a panel, is 3%.

Two things fell out of it:
- **The aim must stay on the sheet** (`AIM_MARGIN`). Sweeps and tracks deliberately run to
  the edges, and an aim past the trim is an aim at the desk, which no framing distance can
  fix - the covering solve has nothing left to solve.
- **Scale follows the panel the camera is OVER**, not the one being read. A travelling move
  spends most of its time between panels, and framing those seconds against a panel that is
  off screen puts the scale somewhere arbitrary.

### Rake is typed by the scene, not by the page

Reported alongside: fractal_zoom "warped in horrible ugly ways". Two findings, and only one
of them was mine:
- The diagonal banding in those panels is fractal_zoom's OWN float32 precision blocks, which
  the close framing magnifies. Checked, not assumed: going from a fixed 8x8 texture grid to
  an adaptive 6-40 changed those frames not at all. (The adaptive grid stayed anyway - the
  reasoning behind the fixed 8 no longer held at this framing - but it is not a repair.)
- The real finding is that **a field scene cannot take a hard rake**. ghost already types
  every scene `subject` or `field` and `Shots` already gives fields the gentle moves only;
  the same rule now applies to the page's rake (`FIELD_FLATTEN`). A hard angle reads as
  foreshortening when there is an object to foreshorten and as damage when there is not.

### Coverage is measured now, not eyed

"Is there dead space in the picture" took three passes to settle and every pass was judged by
looking. `ComicVehicle.page_coverage()` reports how much of the frame's width the sheet
spans, and the look probe prints it per frame and flags anything under 0.995. That is the
number to watch when touching the framing solve.

## Settings: one owner (2026-09-03)

Reported: "Vehicle and some of the other options are not serializing correctly. This happens
a lot; please ensure all options are just saved automatically, by default." Not a vehicle
bug - a shape the whole app had. Every remembered value was saved by whichever script owned
the control: the Director's debounce for the picture knobs, the Generative panel's for the
voice, the splash's for the last song, the deps panel's for its collapse. Five writers, five
debounces, each doing `ConfigFile.load()` -> set -> `save()` on the same file. That fails
three ways and all three were live:

- **Forgotten.** A control is persistent only if someone remembers to write save code. The
  Vehicle picker went in beside four sliders the DIRECTOR saves, in a panel that saves its
  own settings a different way, and nothing in the code said which applied.
- **Clobbered.** Read-modify-write from separate owners is safe only while nothing else
  holds a copy - and an export renders in a SECOND ghost against the same file.
- **Lost.** Every debounce flushed on quiet or at exit, so a kill threw away everything
  since the last pause. A debounce can also be starved indefinitely by a drag that never
  goes quiet.

`scripts/settings.gd` is now the one owner: one in-memory copy, one writer, `MAX_DIRTY_MS`
so an unflushed change lands within a couple of seconds whatever the control is doing, and
`_read_only` so a render or a test probe can never write the user's file.
`Settings.bind(control, section, key, default, keys)` is the part that answers "by default" -
it loads the stored value, connects the signal and writes on change, so a control cannot be
added without being saved. Two checks keep it that way: `docs.py check_settings_owner` and
`tests/settings_check.gd`, which also proves the kill case rather than assuming it.

Two things this turned up on the way:
- **Autoload order is load-bearing.** `Settings` must be FIRST; listed after `Director` it
  loaded the file after Director had already read its values, so every setting came back as
  its default - silently, because a default is a valid value. `vehicle_pick_check` now
  compares the picker against the FILE, which is what catches this.
- **A gate must not edit the config of whoever runs it.** Probes are read-only by default
  now; `settings_check` opts in explicitly because persistence is what it tests.

## Known, and deliberately left

- **Panels out of shot are stills.** Everything visible moves; the budget (`LIVE_MAX`) is what
  keeps that affordable. If a rake ever puts more of the page in shot than the budget allows,
  the ones that keep moving are the ones nearest the reading.
- **Dark scenes read as empty boxes.** Roughly half the catalogue is near-black by design (`bed`
  tops out at a mid tone), and against bright paper that is high contrast and fine - but a page
  of four dark panels is a page of four dark rectangles. This is a scene-palette question, not a
  vehicle one, and it is the first thing to look at if the comic feels flat.
- **The desk is a screen-space wash, not a surface in the world.** A quad big enough never to
  run out under a close raked camera has corners behind the eye, and the projection has to drop
  those - so the world-quad version simply never drew, at any size worth having. A defocused
  surface has nothing to rake anyway.
- **`get_image()` is never called and must not be.** Everything here holds pictures by stopping
  render targets. A future feature that wants a panel's PIXELS (a halftone filter, a thumbnail)
  must do it on the GPU or it reintroduces Masking's readback stall.
- **The GPU look probes core-dump at xvfb shutdown.** Pre-existing and unrelated: verified that
  `city_look_probe.gd`, untouched by this work, aborts identically after its own clean exit.

## Deferred (phase 4, explicitly not now)

- Ink / halftone / screentone filters over the panels ("make it look drawn").
- Round-robin liveness for held panels.
- Speech balloons and captions carrying the Synthesis/Generative narration.
- A real page turn with a bent page and a printed back.
- A page turn seen from the reading distance: it is currently a gesture designed to be watched
  from further back than the camera ever now sits.
- Adaptive `LIVE_MAX`, tied to what the stage governor is measuring.
