# vortex

A spectral music visualizer built with [Godot](https://godotengine.org/) 4.6. Point it at a `.wav`, and it draws geometry in response to the sound - rings, planes, harmonics, lattices, whatever - then loops through scenes like a video you can record full-screen.

It is the same move as the Arena and business-card generators elsewhere in this repo, pushed at audio: a **scene definition** (a typed bag of parameters) is passed through **typed transformations** that are modulated by audio features - amplitude, per-band energy, beat, velocity. Procedural, deterministic, and cheap. No generative AI in the render path.

## Why

Generating visuals for a song with an image/video model is slow and expensive. A procedural visualizer is free, runs in real time, and is infinitely customizable. Each play rolls a fresh random session seed (so you keep seeing new combinations); pass `--seed N` to pin one, which is exactly how the video exporter reproduces the session you just watched.

## The shape

```
  .wav ──▶ Spectrum ──▶ AudioFeatures ──▶ VortexScene ──▶ screen / video
           (analyzer)    (typed, per-frame)  │  (definition × behavior)
                                  movement ───┤
                                              ▼
                              Director: cut on the music, blend sometimes
```

- **`Spectrum`** (autoload) owns the `AudioStreamPlayer` and the `AudioEffectSpectrumAnalyzer` bus effect. Every frame it emits one **`AudioFeatures`**: overall `energy`, named bands (`bass`, `low_mid`, `mid`, `high`, `treble`), a smoothed `beat` pulse, the full `bands` array, `time`, and - new - `flux` and `movement` (a sliding-window measure of how much the spectrum is *changing*, used to time scene cuts). This is the single typed interface every scene reads - scenes never touch the audio engine directly.
- **`VortexScene`** (`class_name`, base script) is one visualizer. `build_params(rng)` rolls a **definition** from a seeded RNG (the typed parameter bag), `update(features, delta)` modulates that definition by the audio, and `_draw()` renders it through the scene's **view** (a centered camera: zoom / tilt / rotate / off-center). Add a scene by subclassing this.
- **Motion is its own axis.** *What* a scene draws (the definition) is separate from *how it moves* (a **behavior**). The base provides a **`ModBank`** of slow seeded oscillators pooled into named organic channels (`mod.value("sway")`) and a per-element `wobble(key, i)` offset.
- **Physics is a registry, not per-scene code.** The duplicated dynamics (scatter, gravity, springs) are extracted into **`Primitives`** - a registry of reusable **force** modules (`gravity`, `spring`, `drag`, `scatter`, `wind`, `pulse`, `orbit`, `wobble`), each a small class with constants baked in via its config, the Praxis registry move. A scene builds a **`ParticleSystem`** (a bag of `Particle`s = geometry) and **composes forces by key**; the substrate steps them and reports `settled()` for oneshots. The same `scatter` bursts glass, rocks, and embers - cross-contamination is free, and a new scene is mostly a parts list (see `embers`).
- **Behaviors** (`static` / `drift` / `fluid`) are typed presets of motion gain. `static` freezes the camera and per-element motion - the scene reacts to audio alone (the original, un-modulated look). `drift` adds gentle whole-scene camera breathing. `fluid` turns on independent per-element motion. The same geometry, kept as several options. Determinism is preserved: same song, same behaviors, every run.
- **Structure is the bias; motion is bounded variance** - for *flat* things. A flat subject (snowflake, glass pane) sways about a seeded rest pose rather than spinning, and the camera never rolls or shears flat 2D content (rolling/shearing a plane is fake 3D - `drift_view` does only zoom + pan). Genuinely 3D bodies (`Mesh3D`) are the exception: they rotate slowly and continuously, because that is how a real solid reveals its volume. **`Activation`** carries the same idea to elements: each gets a seeded threshold + gain through a soft nonlinearity and a fast-attack/slow-decay EMA, so with `sparsity > 0` some elements stay rooted (the static floor) while others bloom; `sparsity = 0` means everything moves. And the **camera eases** (the `SceneView` EMA-smooths toward its target), so every move is gentle.
- **Every scene declares its render kind.** The project has carried several rendering mechanisms forward (the "split"), so each scene now names *how* it draws on a typed axis - `canvas` (flat 2D), `mesh3d` (software 3D bodies projected onto the canvas), `particles`, `swarm`, or `scene3d` (the unified path below). Naming the divergence is the first step to converging on it; the kind rides along in every feedback record so a critique is tied to the renderer that produced it.
- **Real 3D where it matters.** `Mesh3D` is a software 3D primitive (icosphere / cube / octahedron / tetrahedron → depth-sorted flat-shaded faces). `Geo` holds the shared polygon helpers (convex split, fracture) that shatter the glass.
- **A unified 3D path (`scene3d`), the convergence target.** A real, positionable perspective camera (`Lens3D`: an eye looking at a target with a field of view) replaces the old fixed centred projector - so a scene can push in, orbit, and frame *in depth*, and a wide lens up close gives **forced perspective** (near geometry looms over far, the dimensional read a sheared 2D plane only fakes). `Plane3D` is a flat quad genuinely placed in 3D space (stack them for parallax, stand them as bars, tilt them into depth - geometry, not a sheared card). `Scene3D` is the base that owns the lens and a world of `Mesh3D` bodies + `Plane3D` quads, depth-sorts the lot back-to-front, and draws it - so bodies and planes correctly occlude each other under one camera. `wire_solid` is migrated onto it and `planes` is built on it; the rest migrate here over time.
- **Many items, local rules.** `Swarm` is a scalar field over a grid that evolves by *local* interaction - development creeping out from seeds (`GROW`), or injected pulses diffusing across the lattice (`WAVE`). It drives thousands of items without scripting each one (the `metropolis` city grows and pulses from one), and the same mechanism transfers to any abstract many-item grid - the cellular / ant-colony idea.
- **Sound drives colour, not scale.** Pulsing geometry *size* with amplitude reads as cheap throbbing - shapes hold their form. Instead, audio drives **colour, brightness, and glow** through `Lighting`: moving bright **hotspots** that sweep the frame (region-aware lighting / gradient swipes), a global **glow** that flares on beats and decays slowly, and a slow hue drift. A scene asks `light.at(pos)` for a local brightness boost and `light.glow()` for the global flare. This is the shared modulation surface a future unified renderer (2D or 3D under one control) will route everything through; for now scenes opt in.
- **Framing is a typed axis.** A scene declares a `framing` class and the **`Shots`** registry assigns a camera move from the matching pool: expressive for `subject` (offset / push / pan / canted), gentle for `field` fillers, square-on for a single `plane` (so a flat snowflake or pane never reads as a tumbling card). And planes can spawn in multiples - a few small ones don't look stranded the way one lone spinning plane does.
- **Lifecycle and exit cues are typed too.** A scene either **loops** until cut, or is a **oneshot** that plays one sequence and reports `finished()` (shatter glass settling, then ending). Once a scene is *eligible* to exit - a loop past its minimum hold, or a oneshot that finished - the **`Director`** waits for a chosen spectral **trigger** before actually cutting, so exits land on the music: usually a **beat** (rising edge), sometimes a **movement** (section change) or a **lull** (drop into quiet), with a maximum-hold backstop. Triggers are picked weighted per scene.
- **`Director`** (autoload) holds the registry of `{scene, behavior}` pairs, runs the lifecycle/trigger logic above, and performs the change - mostly a **dip to black** (the old scene fades out, a beat of true darkness, then the new one fades up, so scenes never overlap and the eye gets a clean gap), with the occasional hard **cut**. `--scene <name|N>` pins one scene for authoring.
- **Novelty-weighted scheduling.** Picking the next scene uniformly at random clusters - the same *kind* recurs while others go unseen. Instead each candidate is weighted by how long its kind has gone unshown, so long-unseen scenes are drawn far more often than recent duplicates (and never two of one kind back to back). A soft priority queue - seeded by the session seed (random per play, or pinned with `--seed N`).
- **Two ways to drive the show.** *Auto* is the above: the Director chooses and cuts on the music. *Manual* plays a **runbook** - a user-authored linear sequence of scenes (`runbooks/*.json`), each entry naming its scene, behavior, shot, and an exit rule (a fixed `hold`, a musical `beat`/`movement`/`lull` cue, or the scene's own lifecycle). The runbook is the first rung of orchestrating a whole visualization by hand; the data spec is meant to grow toward a full manual editor (see `runbooks/README.md`).
- **A feedback channel for authoring.** "This shape feels wrong" is hard to act on from a note; the `FeedbackConsole` (press `` ` ``) captures the scene on screen - its typed descriptor (name / kind / behavior / shot / seed / params / the audio frame) plus your typed query *and a screenshot* - into `feedback/NNNN.{json,png}`. The seed makes it reproducible; the image shows what "wrong" looked like.
- **`Splash`** is the start screen: import a song from disk (the last one is remembered in `user://vortex.cfg` and pre-selected), then click **Auto** (start the scheduler) or **Manual** (open the workspace) - the mode button *is* start, there is no separate one. CLI flags (`--audio` / `--scene` / `--runbook` / `--no-splash`) boot straight past it for authoring and automation.
- **`Workspace`** is the manual-mode surface (scaffolding): opened by the Manual button over a session running `runbooks/default.json`, a left-side panel lists the runbooks and clicking one switches the live show to it. This is the canvas the future hand-authoring tools (per-entry params, reordering, a timeline, save) will grow into.
- **Session lifecycle.** `main` owns it: splash → start a session (Auto or Manual) → and **when the song ends, tear the session down and return to the splash** (`Spectrum.song_finished` → `Director.detach()` + `Spectrum.stop()`). It also maps the global keys (next / full-screen / feedback / quit).

## Two ways to run

1. **Live (default).** Audio plays through the analyzer bus; scenes react in real time. Use this to author and preview, and screen-record it with OBS for a quick capture.
2. **Baked (for export).** The Export button runs two background processes: a **headless** `bake_runner` first analyzes the song into a spectrum timeline (`SpectrumBake`, cached per song), then a **Movie Maker** render (`--write-movie out.avi --fixed-fps 60 --bake-file …`) loads that timeline and drives the scenes from it instead of the live analyzer - frame-perfect, with synced audio, and unaffected by the fact that Movie Maker's offline audio would otherwise make the live analyzer unreliable. Keeping the analysis out of the render means the render never blocks on a grey frame. The live authoring session keeps using the real-time analyzer.

## Layout

- `project.godot` - Godot 4.6 project; autoloads `Spectrum` and `Director`; `main.tscn` is the entry scene.
- `scenes/main.tscn` / `scripts/main.gd` - root node; loads audio, full-screen (`F11`), next scene (`Space`), quit (`Esc`).
- `scripts/spectrum.gd` - `Spectrum` autoload: live analyzer + a baked-timeline backend (`--use-bake`), per-frame `AudioFeatures`, spectral-flux/movement detection.
- `scripts/bake.gd` - `SpectrumBake`: offline FFT of the song (via ffmpeg) into the same 64 log bands - the deterministic timeline the export render replays.
- `scripts/audio_features.gd` - `AudioFeatures`, the typed per-frame struct scenes consume.
- `scripts/vortex_scene.gd` - `VortexScene` base class: definition + view + behaviors + the `tick`/`drift_view`/`wobble` motion helpers.
- `scripts/mod_bank.gd` - `ModBank`, the seeded slow-oscillator pool behind organic motion.
- `scripts/activation.gd` - `Activation`, per-element sparse response gating with EMA decay.
- `scripts/scene_view.gd` - `SceneView`, the EMA-smoothed per-scene camera (zoom / tilt / rotate / off-center).
- `scripts/shots.gd` - `Shots`, the camera-framing registry (centered / offset / push / pull / pan / canted) with subject/field/plane pools.
- `scripts/particle.gd` / `scripts/primitives.gd` / `scripts/particle_system.gd` - the physics substrate: a `Particle`, the `Primitives` force **registry**, and the `ParticleSystem` that composes and steps them.
- `scripts/mesh3d.gd` / `scripts/geometry.gd` - the `Mesh3D` software-3D primitive (now with coherent fractal `warp`, planar `facet`, and a `rock()` factory) and `Geo` polygon helpers (split, fracture).
- `scripts/lens3d.gd` / `scripts/plane3d.gd` / `scripts/scene3d.gd` - the unified 3D path: the `Lens3D` perspective camera, the `Plane3D` quad primitive, and the `Scene3D` base that depth-sorts bodies + planes under one camera.
- `scripts/feedback.gd` - the `FeedbackConsole` (press `` ` ``): writes a scene record + screenshot to `feedback/` for authoring.
- `scripts/splash.gd` - the `Splash` start screen: song import (remembered), Auto / Manual.
- `scripts/workspace.gd` - the `Workspace`: manual-mode left panel listing runbooks over the live scene (scaffolding for hand-authoring).
- `runbooks/` - user-authored scene scores for Manual mode; `default.json` is the one Manual opens with, `runbooks/README.md` is the data spec.
- `scripts/swarm.gd` - `Swarm`, the grid field that spreads by local rules (growth fronts, pulse waves) for many-item scenes.
- `scripts/lighting.gd` - `Lighting`, audio-reactive colour: moving hotspots, beat glow, hue drift (the preferred channel over scale).
- `scripts/nonlinear.gd` / `scripts/flow.gd` / `scripts/filament.gd` - the organic layer: `Nonlinear` (shared response curves + asymmetric `flare`), `Flow2D` (curl-noise meander field), `Filament` (root / tendril / lightning / thread growth that crawls in along a front).
- `scripts/exporter.gd` / `scripts/bake_runner.gd` - the `Exporter` (persistent) renders to video in **two background processes**: first a **headless** `bake_runner` analyzes the song into a spectrum cache (no window, cached per song), then a Movie Maker render loads that cache (`--bake-file`) and draws immediately - so the render never freezes baking. Both are polled by PID; status ("Analyzing…" / "Rendering…" / "Exported ✓") shows in the main window.
- `scripts/director.gd` - `Director` autoload: `{scene, behavior}` registry, lifecycle + spectral-trigger timing, cut/blend transitions, `--scene` pin.
- `scripts/scenes/` - the visualizers (see below). Drop new ones here and register them in `Director.SCENES`.
- `audio/` - put `song.wav` here (git-ignored). Or pass `--audio /path/to/song.wav` on launch.

## Scenes

Each is a small combination of shapes; behavior decides how it moves.

- **`spectrum_ring`** - the spectrum bent into a circle; bars push outward per band. `static` (rigid wheel) and `fluid` (bars ripple as detached strokes).
- **`harmonic_lattice`** - a grid of cells breathing with a traveling spectral wave. `static` and `drift`.
- **`rooted_growth`** - crawling roots and tendrils (`Filament`) that spread from a seed, meandering on a curl-noise `Flow2D` field, branching and tapering, revealed along a growth front so they creep outward; they mature and regrow staggered, so something is always moving. Beats surge the growth (through a nonlinear `spike`); colour and glow carry the audio.
- **`filaments`** - the `Filament` primitive showcased, three lives by seed: slow **lightning** (forked bolts that strike on the beat, blaze, fade, re-strike), **neural** tendrils (a creeping, regrowing network), or flowing **threads**.
- **`fog_lights`** - glowing orbs tied to slices of the spectrum, diffused and occluded under big drifting blobs of fog.
- **`strata`** - translucent waveform planes stacked into tilted depth, near planes scrolling faster than far (parallax).
- **`bloom`** - elegant procedural rosette curves from the superformula (a few numbers → star / flower / gear / soft polygon), layered concentrically with a hue gradient and morphing *continuously* with the audio (energy sharpens the lobes through a nonlinear curve). Replaced the old hard-coded Koch stars, which stepped their recursion in visible jumps.
- **`wire_solid`** - a true 3D cube / octahedron / tetrahedron / icosahedron (`Mesh3D`): perspective, depth-sorted, translucent faces with bright edges. Rotates slowly and continuously, because a real solid reveals its volume by turning. Now on the unified `scene3d` path (a `Lens3D` camera, not the old centred projector).
- **`planes`** (`scene3d`) - the spectrum as a ring of genuine `Plane3D` quads standing on a ground plane, with a `Mesh3D` crystal tumbling at the centre, all projected through one `Lens3D` under forced perspective: near bars loom over far ones and the core passes behind and in front of them as the camera orbits. A deliberate 3D echo of `spectrum_ring`, and the first scene built natively on the unified path.
- **`voxel_blocks`** - an isometric heightfield equalizer. By seed a small centred `plot`, or a `city` of thousands of blocks spilling off every edge with the camera down among them - city blocks carry a structural base (a standing skyline) with the spectrum bouncing on top, and `Activation` sparsity keeps some columns rooted (still) while others move.
- **`cityscape`** - a layered skyline of rectangles; building heights track the bands and windows flicker on the beat.
- **`shatter_glass`** (`scene3d`) - a real pane of glass seen at a three-quarter angle through the `Lens3D` camera. On a beat it fractures into irregular angular shards (`Geo.fracture`, radiating from an impact - not pizza slices) that burst off the plane and **tumble through space**, each spiralling on its own 3D axis, depth-sorted so near shards occlude far ones. Seeds `loop` (burst → re-knit) or `oneshot` (burst → settle → end).
- **`gaussian_landscape`** - terrain built from Gaussian bumps (heights driven by the bands), drawn as receding ridgelines with translucent fog pooling in the valleys and flowing sideways in the wind.
- **`rocks`** - stones whose geometry is *generated*, not displayed as spheres: a subdivided icosphere given coherent multi-octave fractal mass (`Mesh3D.warp` - smooth correlated bulges and hollows, not per-vertex fuzz), shaved into angular fracture faces (`Mesh3D.facet` - the conchoidal flats of broken rock / the facets of a gem), then stretched into a natural non-round proportion. Real 3D, tumbling gently (rooted ones barely turn, via `Activation`). Style by seed: `plain` / `rough` / `crystal`. Mode by seed: `pulse` / `explode` / `crumble` (oneshot).
- **`embers`** - a warm cloud of sparks that flare on the beat and ride the wind. Almost no code of its own: `scatter` + `wind` + `spring` + `drag` from the registry - the cross-contamination demo.
- **`metropolis`** - a large isometric countryside of thousands of blocks running off every edge, driven by a `Swarm` field. By seed: `growth` (a city creeps out from a few seeds across the hills) or `pulse` (a built city with beat-injected colour fronts rippling through it), under flowing fog. Each block also bounces with its own spectral band (responsive, not a static slab), and hue/brightness gradients run with terrain height, position, development, and the colour pulse.

## Running it

Open `project.godot` in Godot 4.6 and press play, or from the command line:

```
godot --path axis/vortex                       # opens the splash: import a song, pick Auto/Manual
godot --path axis/vortex -- --audio ~/track.wav # skip the splash, boot straight in
godot --path axis/vortex -- --runbook default   # manual mode: play runbooks/default.json
godot --path axis/vortex -- --scene planes      # pin one scene for authoring
godot --path axis/vortex -- --no-splash         # auto mode, bundled/no audio, no splash
```

By default it opens the **splash** start screen (import a song from disk, choose Auto or Manual). Any of `--audio` / `--scene` / `--runbook` / `--no-splash` boots straight past it. `--audio` accepts `.wav`, `.mp3`, `.ogg`, and `.flac`. FLAC has no runtime loader in Godot, so it is transcoded to a temp WAV via `ffmpeg` (must be on `PATH`); the others load natively.

Controls: `Space` next scene · `F11` full-screen · `` ` `` send feedback · `Esc` quit.

The feedback key writes `feedback/NNNN.json` (the scene descriptor + your query) and `feedback/NNNN.png` (a screenshot of that frame) - a reproducible record of anything that looks off.

If no audio is found it still runs - scenes just animate on an idle clock with zeroed features, so you can develop a scene with no song loaded.

## Adding a scene

```gdscript
extends VortexScene

func build_params(rng: RandomNumberGenerator) -> Dictionary:
    return { "count": rng.randi_range(6, 24), "hue": rng.randf() }

func update(f: AudioFeatures, delta: float) -> void:
    tick(f, delta)            # advance organic motion (speed-scaled by behavior)
    drift_view(f)             # optional whole-scene camera drift (gated by behavior)
    queue_redraw()

func _draw() -> void:
    begin_draw()              # push the view transform; draw around (0,0) = center
    for i in int(params.count):
        var p := Vector2(0, -200 + i * 20)
        p += Vector2(wobble("dot", i), 0) * 40   # per-element drift (fluid only)
        draw_circle(p, 6, Color.from_hsv(params.hue, 0.7, 1.0))
```

Then add `{"script": preload("res://scripts/scenes/my_scene.gd"), "behavior": "fluid"}` to `Director.SCENES` - list it more than once with different behaviors to keep several looks. To make a oneshot, set `lifecycle = "oneshot"` in `build_params` and return `true` from `finished()` when its sequence ends. The contract: a seeded definition, modulated by audio, moved by a behavior, with a lifecycle, drawn through a view.

Set `render_kind` in `build_params` so the scene is typed (`"canvas"` is the default; use `"particles"` / `"swarm"` / `"mesh3d"` as appropriate). For a 3D scene, **extend `Scene3D` instead of `VortexScene`**: it sets `render_kind = "scene3d"`, gives you a `lens` ([Lens3D]), and renders a depth-sorted world of `Mesh3D` bodies (`add_body(...)`) and `Plane3D` quads (`add_plane(...)`) when you call `render_world()` from `_draw` - so you build geometry in real 3D space and fly the camera, rather than shearing 2D. See `planes.gd` and `wire_solid.gd`.

## Roadmap

- [x] Live analyzer → `AudioFeatures` → scene framework; fourteen scenes.
- [x] Behaviors (`static`/`drift`/`fluid`) - motion typed separately from geometry; originals kept as `static`.
- [x] Lifecycle (`loop`/`oneshot`) + spectral exit triggers (`beat`/`movement`/`lull`); exits land on the music; jump cuts with occasional blends.
- [x] **Primitive registry**: `Particle` / `ParticleSystem` + a `Primitives` force registry. Glass/rocks/embers composed from it.
- [x] **Gentle by construction**: bounded sway about rest poses (not unbounded spin), `Activation` sparse response with EMA decay, EMA-smoothed camera, typed framing pools (subject / field / plane), multi-instance planes.
- [x] **Real 3D**: `Mesh3D` software primitive; rocks and `wire_solid` are genuine 3D bodies. `Geo` fracture replaced the pizza-sliced glass.
- [x] **Swarm fields**: many-item scenes driven by local rules (`metropolis` city growth + colour pulses); transferable to any grid.
- [x] **Nonlinear / organic primitive layer**: a shared `Nonlinear` activation library, a curl-noise `Flow2D` field, and a `Filament` growth primitive - the source of "alive" (threshold/saturation, meander, crawling growth). `rooted_growth` rebuilt on it; a `filaments` scene (lightning / neural / thread) added.
- [x] **Colour over scale**: `Lighting` (moving hotspots, beat glow, hue drift) drives reactivity; shapes hold their form. Wired into ring / lattice / rocks / wire_solid so far.
- [ ] More `Swarm` rules: pheromone / ant-colony trails, reaction-diffusion, predator-prey - and abstract (non-city) many-item scenes that reuse the field.
- [x] **Render-kind typing**: every scene declares how it draws (`canvas` / `mesh3d` / `particles` / `swarm` / `scene3d`), making the carried-forward split explicit so it can converge.
- [x] **Forced-perspective 3D path**: a positionable `Lens3D` camera, a `Plane3D` quad primitive, and a `Scene3D` base that depth-sorts bodies + planes under one camera. `wire_solid` migrated, `planes` built native.
- [x] **Procedural geometry**: rocks are generated (coherent fractal `warp` + planar `facet` + stretch), not displayed spheres - the first "geometry from data" demonstrator.
- [x] **Novelty-weighted scheduling**: scenes least-recently shown are favored, so the show spreads across the catalogue instead of repeating.
- [x] **Authoring feedback console**: `` ` `` writes a reproducible scene record + screenshot to `feedback/`.
- [x] **Manual mode (runbooks)**: a JSON data spec mapping a song to a user-orchestrated linear sequence of scenes; a **splash** start screen imports audio and picks Auto/Manual.
- [ ] **Unified renderer, continued**: migrate the remaining 2D scenes onto `Scene3D` (planes for flat content) and route everything through one modulation surface (`Lighting` + materials), so any scene renders under one set of camera/light controls.
- [ ] **Richer runbooks**: per-entry parameter overrides, time-coded cues, per-transition blends, and eventually a manual editor that writes the files.
- [ ] **Procedural geometry kit**: extend `warp`/`facet` toward other "geometry from data" subjects (terrain, trees, crystals) feeding the `scene3d` world.
- [ ] Grow the kit (flow fields, collisions, soft-body links, emitters) and migrate more bespoke scenes onto particles / `Mesh3D`.
- [x] **Video export**: the Export button relaunches in Movie Maker mode (`--write-movie --use-bake`) to render the visualization + synced audio to a file, reproducing the session via `--seed` and driving reactivity from the offline bake. Runs as its own (background) render so the live session stays interactive.
- [x] **Bake mode**: `SpectrumBake` decodes the song (ffmpeg) and runs an offline FFT into the same 64 log bands, a deterministic per-frame timeline. The export render drives visuals from it (`--use-bake`) instead of the analyzer, so the recorded reactivity is correct under Movie Maker's offline audio *and* the render runs as fast as the machine allows. Live sessions stay on the real-time analyzer.
- [ ] Stronger beat/onset and tempo tracking; exits that snap to bars, not just beats.
- [ ] Definition / behavior / lifecycle presets per song (config-driven, like the other generators).

## Status

Working framework, live path end to end: import a song on the splash (or boot a CLI flag), watch seeded scenes react and change *with the music* - or hand-orchestrate them with a runbook. Geometry, motion (behaviors), lifecycle, exit cue, **and render kind** are independent typed axes; the physics is a composable **primitive registry**; and there is now a real **forced-perspective 3D path** (`Lens3D` / `Scene3D` / `Plane3D`) that the 2D scenes are migrating onto, with generated geometry (procedural rocks) as the first proof. An in-app **feedback console** (`` ` ``) captures a scene record + screenshot for authoring. Bake/Movie-Maker export, more scenes on the 3D path, and richer runbooks are next. Standalone; it does not import or depend on Praxis.
