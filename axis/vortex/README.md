# vortex

A spectral music visualizer built with [Godot](https://godotengine.org/) 4.6. Point it at a `.wav`, and it draws geometry in response to the sound - rings, planes, harmonics, lattices, whatever - then loops through scenes like a video you can record full-screen.

It is the same move as the Arena and business-card generators elsewhere in this repo, pushed at audio: a **scene definition** (a typed bag of parameters) is passed through **typed transformations** that are modulated by audio features - amplitude, per-band energy, beat, velocity. Procedural, deterministic, and cheap. No generative AI in the render path.

## Why

Generating visuals for a song with an image/video model is slow and expensive. A procedural visualizer is free, runs in real time, and is infinitely customizable. Each play rolls a fresh random session seed (so you keep seeing new combinations); pass `--seed N` to pin one, which is exactly how the video exporter reproduces the session you just watched.

## The idea: cattle, not pets

Most 3D work treats each object as a **pet**: you hand-model *this* eye, *this* rock - name it, tune it, love it. It doesn't scale and it never surprises you. vortex is built the other way - **cattle**: an object is a *recipe* of layered primitives whose parameters are **sampled from adjustable ranges**, so every instance is a fresh, naturally-occurring variation.

Take **the eye** (`EyeBody`) as the worked example. A pet eye is a bespoke mesh. But an eye is *really* just a few primitives stacked: a **sphere** for the ball, a thin pliable **lens** (a cornea dome) stretched across the front, a recessed **iris** cap, a **pupil** hole - plus colours, hues, gloss, and a light. Model *those layers* and *sample their ranges* - iris hue and saturation, pupil dilation, corneal curvature and gloss, eyeball size, how restless the gaze is - and you don't get one eye, you get the *space of all eyes*, occurring naturally. (Today's `EyeBody` is the first step: the layers are real 3D primitives, but several of their numbers are still hand-tuned. Lifting those into sampled ranges - turning the pet into cattle - is the [scene-spec pipeline](#roadmap) north star.) The same move is already in `rocks` (a sampled stack of geometry + texture + material) and `bloom` (a whole family of shapes from a handful of superformula numbers).

### Three ways to drive it

1. **Auto** - the autopilot. A random session seed rolls the whole show; the Director picks scenes by novelty and cuts on the music. Pure cattle: press play, watch something you've never seen (and will only see again with that seed).
2. **Manual** - a **storyboard**. You author the exact sequence and cues by hand (`storyboards/*.json`), the way "the-point" lays out eye → two_eyes → prism. Every pet placed deliberately.
3. **Semi-automatic** *(planned - the one to be excited about)*. Start from the **same autopilot seed**, then reach in and **pull levers, turn dials, move sliders** that influence the modulation at chosen points - nudge iris hue, gaze energy, how hard the beat splits the prisms - and watch the downstream scenes change in response, live. Not keyframing; *steering a living system*. The seed hands you a whole world for free; the controls bend it toward what you hear. The aim is for it to feel fast, responsive, and unlike anything you've used.

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
- **`Director`** (autoload) holds the registry of `{scene, behavior}` pairs, runs the lifecycle/trigger logic above, and performs the change. In auto mode it mostly **dips to black** (the old scene fades out, a beat of true darkness, then the new one fades up) with the occasional hard **cut**; a storyboard sets its own style (the-point forces 100% **cuts**). `--scene <name|N>` pins one scene for authoring.
- **Transition style is a hierarchy.** Highest wins: a compatible **morph** (below) → the storyboard **entry**'s `transition` → the **scene**'s own `transition_style` (set in build_params) → the storyboard's top-level default → the mode default (manual = `cut`, auto = the weighted dip/cut bag). So a storyboard can force "cut" for everything yet a single scene can still ask for a dip, and the-point gets 100% cuts except where a morph applies.
- **Content-aware transitions (typed morphs).** A scene declares the geometry it leaves (`morph_out`) and what it can grow in from (`morph_in`). When the next scene's `morph_in` matches the current scene's `morph_out`, the Director plays a **morph** - an instant swap where the incoming scene animates out of the outgoing shape - instead of a cut/dip, and hands over a typed `morph_payload()` so the transition is *continuous*: the single eye passes its colour/gaze/size to `two_eyes`, which starts as that exact eye and splits into two identical copies. Mismatched or empty types fall back to a cut, so a bespoke transition is only ever attempted between compatible geometries - it can't break.
- **Novelty-weighted scheduling.** Picking the next scene uniformly at random clusters - the same *kind* recurs while others go unseen. Instead each candidate is weighted by how long its kind has gone unshown, so long-unseen scenes are drawn far more often than recent duplicates (and never two of one kind back to back). A soft priority queue - seeded by the session seed (random per play, or pinned with `--seed N`).
- **Ways to drive the show** (see [Three ways to drive it](#three-ways-to-drive-it)). *Auto*: the Director chooses scenes by novelty and cuts on the music. *Manual*: plays a **storyboard** - a user-authored linear sequence (`storyboards/*.json`), each entry naming its scene, behavior, shot, and exit rule (a fixed `hold`, a musical `beat`/`movement`/`lull` cue, or the scene's own lifecycle); see `storyboards/README.md`. *Semi-automatic* (planned): the autopilot seed plus live levers/dials that steer the modulation. The storyboard is the first rung; the data spec grows toward a manual editor and then the semi-auto control surface.
- **A feedback channel for authoring.** "This shape feels wrong" is hard to act on from a note; the `FeedbackConsole` (press `` ` ``) captures the scene on screen - its typed descriptor (name / kind / behavior / shot / seed / params / the audio frame) plus your typed query *and a screenshot* - into `feedback/NNNN.{json,png}`. The seed makes it reproducible; the image shows what "wrong" looked like.
- **`Splash`** is the start screen: import a song from disk (the last one is remembered in `user://vortex.cfg` and pre-selected), then click **Auto** (start the scheduler) or **Manual** (open the workspace) - the mode button *is* start, there is no separate one. CLI flags (`--audio` / `--scene` / `--storyboard` / `--no-splash`) boot straight past it for authoring and automation.
- **`Workspace`** is the manual-mode surface (scaffolding): opened by the Manual button over a session running `storyboards/default.json`, a left-side panel lists the storyboards and clicking one switches the live show to it. This is the canvas the future hand-authoring tools (per-entry params, reordering, a timeline, save) will grow into.
- **Session lifecycle.** `main` owns it: splash → start a session (Auto or Manual) → and **when the song ends, tear the session down and return to the splash** (`Spectrum.song_finished` → `Director.detach()` + `Spectrum.stop()`). It also maps the global keys (next / full-screen / feedback / quit).

## Rendering: live & baked

(Distinct from the three *driving* modes above - this is how frames get produced.)

1. **Live (default).** Audio plays through the analyzer bus; scenes react in real time. Use this to author and preview, and screen-record it with OBS for a quick capture.
2. **Baked (for export).** The Export button runs two background processes: a **headless** `bake_runner` first analyzes the song into a spectrum timeline (`SpectrumBake`, cached per song), then a **Movie Maker** render (`--write-movie out.avi --fixed-fps 60 --bake-file …`) loads that timeline and drives the scenes from it instead of the live analyzer - frame-perfect, with synced audio, and unaffected by the fact that Movie Maker's offline audio would otherwise make the live analyzer unreliable. Keeping the analysis out of the render means the render never blocks on a grey frame. The live authoring session keeps using the real-time analyzer.

## Layout

- `project.godot` - Godot 4.6 project; autoloads `Boot` (hides the render window early in export mode), `Spectrum`, and `Director`; `main.tscn` is the entry scene.
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
- `scripts/mesh3d.gd` / `scripts/geometry.gd` - the `Mesh3D` software-3D primitive (icosphere/cube/octa/tetra + `dome` caps; coherent fractal `warp`, planar `facet`, `texturize`, smooth/Gouraud shading via `compute_normals`, a material with `gloss`/`roughness` specular + an `unlit` option, and a `rock()` factory) and `Geo` polygon helpers (split, fracture).
- `scripts/lens3d.gd` / `scripts/plane3d.gd` / `scripts/scene3d.gd` - the unified 3D path: the `Lens3D` perspective camera, the `Plane3D` quad primitive, and the `Scene3D` base that depth-sorts bodies + planes under one camera.
- `scripts/feedback.gd` - the `FeedbackConsole` (press `` ` ``): writes a scene record + screenshot to `feedback/` for authoring.
- `scripts/splash.gd` - the `Splash` start screen: song import (remembered), Auto / Manual.
- `scripts/workspace.gd` - the `Workspace`: manual-mode left panel listing storyboards over the live scene (scaffolding for hand-authoring).
- `scripts/prism_body.gd` / `scripts/eye_body.gd` - the `PrismBody` (browser Prism ported: wireframe tetra + living neural core) and `EyeBody` (a floating eyeball with saccadic gaze), reused across the "the-point" scenes.
- `storyboards/` - user-authored scene scores for Manual mode; `default.json` is the "the-point" sequence (eye → prism → prism_split, exits on the music), `storyboards/README.md` is the data spec.
- `scripts/swarm.gd` - `Swarm`, the grid field that spreads by local rules (growth fronts, pulse waves) for many-item scenes.
- `scripts/lighting.gd` - `Lighting`, audio-reactive colour: moving hotspots, beat glow, hue drift (the preferred channel over scale).
- `scripts/nonlinear.gd` / `scripts/flow.gd` / `scripts/filament.gd` - the organic layer: `Nonlinear` (shared response curves + asymmetric `flare`), `Flow2D` (curl-noise meander field), `Filament` (root / tendril / lightning / thread growth that crawls in along a front).
- `scripts/exporter.gd` / `scripts/bake_runner.gd` - the `Exporter` (persistent) renders to video in **two background processes**: first a **headless** `bake_runner` analyzes the song into a spectrum cache (no window, cached per song), then a Movie Maker render loads that cache (`--bake-file`) and draws immediately - so the render never freezes baking. Both are polled by PID; status ("Analyzing…  45%" / "Rendering…  72%" / "Exported ✓") shows in the main window. The button fades in after ~30s of playback (or partway through a shorter song).
- `scripts/director.gd` - `Director` autoload: `{scene, behavior}` registry, lifecycle + spectral-trigger timing, cut/blend transitions, `--scene` pin.
- `scripts/scenes/` - the visualizers (see below). Drop new ones here and register them in `Director.SCENES`.
- `audio/` - put `song.wav` here (git-ignored). Or pass `--audio /path/to/song.wav` on launch.

## Scenes

Each is a small combination of shapes; behavior decides how it moves.

- **`spectrum_ring`** - the spectrum bent into a circle; bars push outward per band. `static` (rigid wheel) and `fluid` (bars ripple as detached strokes).
- **`harmonic_lattice`** - a grid of cells breathing with a traveling spectral wave. `static` and `drift`.
- **`rooted_growth`** - crawling roots and tendrils (`Filament`) that spread from a seed, meandering on a curl-noise `Flow2D` field, branching and tapering, revealed along a growth front so they creep outward; they mature and regrow staggered, so something is always moving. Beats surge the growth (through a nonlinear `spike`); colour and glow carry the audio. Growth has a **timelapse twitch** - trunks hold steady while young tips tremble (strongest at the advancing front), so it reads as living growth, not a static drawing.
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
- **`rocks`** - stones assembled from a **sampled stack of composable layers**, not a fixed look: a geometry family (subdivided icosphere with coherent fractal mass `Mesh3D.warp` + angular `Mesh3D.facet` + non-round `stretch`; or `hybrid` - a crisp cube/octa/tetra with rock crusting over part of it via gaussian-masked `warp_masked`), a baked **surface texture** (`Mesh3D.texturize` mottles each face like real stone), and a **material** (`gloss` specular + `roughness`, so crystal looks wet/polished and rough looks matte). Style by seed: `plain` / `rough` / `crystal` / `hybrid`. Mode by seed: `pulse` / `explode` / `crumble` (oneshot).
- **`embers`** - a warm cloud of sparks that flare on the beat and ride the wind. Almost no code of its own: `scatter` + `wind` + `spring` + `drag` from the registry - the cross-contamination demo.
- **`metropolis`** - a large isometric countryside of thousands of blocks running off every edge, driven by a `Swarm` field. By seed: `growth` (a city creeps out from a few seeds across the hills) or `pulse` (a built city with beat-injected colour fronts rippling through it), under flowing fog. Each block also bounces with its own spectral band (responsive, not a static slab), and hue/brightness gradients run with terrain height, position, development, and the colour pulse.
- **`orbits`** - harmonograph curves that morph with the music: each is a damped sum of sines in x and y (the figure a harmonograph pen traces), with a slow global phase folding the trace through itself and the spectrum swelling the per-axis amplitudes. Incommensurate frequencies mean it rarely returns to the same shape.

The **"the-point"** scenes (from a planned video; the default storyboard plays them in sequence):
- **`eye`** (`scene3d`) - a **genuinely 3D** floating eyeball (`EyeBody`), and the worked example of [cattle, not pets](#the-idea-cattle-not-pets): discrete primitives stacked, each lit differently - a matte **sphere** sclera (**smooth/Gouraud-shaded** so it reads as a real ball, not facets), a recessed **iris** cap (`Mesh3D.dome`, concave, textured), a flat-black dilating **pupil**, and a clear glossy **cornea** dome (the pliable lens) that catches a wet specular highlight (a real reflection point on the cornea, projected through `Lens3D`). No lids/blink. It **looks around by rotating in 3D** in centre-preferring saccades (squared-radius bias toward the neutral gaze). Declares `morph_out = "eye"`. (Next step: lift its hand-tuned numbers into sampled ranges so each eye is naturally-occurring.)
- **`two_eyes`** (`scene3d`) - two 3D `EyeBody` eyes side by side with **conjugate (locked) gaze**: both track one shared focus, as real eyes do; occasionally one diverges - the async wander is the nonlinear deviation, not the default. Declares `morph_in = "eye"`: arriving from `eye` it plays the **split** (becomes that exact eye - same colour/gaze/size - at centre, then eases apart into two identical copies, shrinking to pair size); entered by a plain cut it simply opens already split.
- **`prism`** (`scene3d`) - a see-through wireframe tetrahedron with a living neural core (`PrismBody`, ported from the browser Prism): glowing edges only, tendrils flowing from the centre and surging with the audio, edges lighting up where tendrils reach them, slowly "looking around". Blue or red by seed.
- **`prism_split`** (`scene3d`) - one prism becoming two: a blue original from which a red one emerges, the pair separating left/right as energy drives the split.

## Toward a complete package: modeling the physical sciences

The long arc is simple to state and enormous to fill: **vortex should be able to model anything physical.** Every scene above is a recipe of sampled primitives ("cattle, not pets"); the goal is to keep growing the primitive kit until the catalogue spans the natural world - weather, light, crystals, terrain, structures, growth, fluids, the cosmos - so that pointing it at a song can summon *any* phenomenon, alone or in combination. The list below is the standing backlog of subjects to model: a domain-by-domain map of what a "complete package" contains. Most reuse primitives we already have (`Mesh3D`, `Swarm`, `Filament`, `Flow2D`, `Lighting`, `Lens3D`, the force registry); the work is composing them and lifting their numbers into sampled ranges.

### Next up (the immediate targets)

- **The snowflake, restored and multiplied.** We used to have a single Koch snowflake; bloom absorbed the old hard-coded Koch stars and the dedicated flake was lost. Bring it back as a proper six-fold dendrite primitive - and then let it manifest as **several dozen flakes at once**, varied in size, drifting as a field rather than a lone card. Motion by seed: all **spinning together** (locked phase), each **spinning at random** (independent), or **carried on the wind** (advected by `Flow2D` / the `wind` force) - the same flat-subject discipline as today (sway about a rest pose, square-on framing, no fake-3D tumbling), just many of them.
- **Weather effects.** Simple, legible atmospheric scenes: **falling snow** (drifting flakes settling and gusting), **rolling fog** (banks creeping across the frame and pooling in low ground - the `gaussian_landscape` / `metropolis` fog generalized into its own scene), **rain**, and **clouds**. Audio drives density, gust strength, and the colour/glow, not the snowflake's shape.
- **Light crossing terrain (real shadow).** A **moving light source** sweeping over a rolling landscape and **casting shadows** that travel with it - a day-arc across hills, the first scene where shadow (occlusion from a positioned light, not just `Lighting`'s hotspots) does the storytelling. Pairs the `gaussian_landscape` terrain with a shadow pass under `Lens3D`.
- **Block harmonics, breaking from the grid.** The large-scale block scenes (`voxel_blocks`, `metropolis`) still read as a regular lattice. Break from it: **plot the blocks on rolling landscape, like buildings on a countryside** - not a flat grid but structures following the terrain. Then make it **evolve**: start with a few small blocks and grow (via `Swarm`) into a **sprawling city of block shapes**, the skyline thickening with the music. (The `Swarm` `growth` mode is the seed of this; the new part is decoupling the plot from the grid and standing it on real terrain.)

### The domain map (the standing backlog)

- **Weather & atmosphere** - snow, rain, fog, clouds, wind streaks, hail, aurora, heat shimmer, lightning (the `Filament` lightning life is the start). Density / intensity ride the audio.
- **Light & shadow** - a positioned, moving light casting real shadows over geometry; day/night sweeps; god rays through fog; caustics; refraction and dispersion through the glass/prism. Beyond `Lighting`'s 2D hotspots toward true occlusion.
- **Crystals & symmetry** - snowflakes (six-fold dendrites), mineral crystals and lattices, growth by accretion, kaleidoscopic symmetry groups. Reuses `Mesh3D` + `Filament`-style dendritic growth.
- **Geology & terrain** - rolling hills (`gaussian_landscape`), erosion, sedimentary strata, canyons, plate motion, volcanoes. Terrain as the stage many other scenes stand on.
- **Structures & cities** - blocks on landscape, a city evolving from seeds to skyline, bridges, lattice frameworks, ruins. The "breaking from the grid" thread.
- **Botany & growth** - trees and branching (L-systems on `Filament`), leaves and ferns, vines, flowers (`bloom` superformula), forests as fields. Growth fronts that crawl in.
- **Fluids** - water surfaces and waves, ripples, smoke and vapour, curl-noise flow (`Flow2D`), whirlpools, splashes. Audio as the disturbance source.
- **Celestial & orbital** - planetary systems and n-body gravity (the `gravity`/`orbit` forces, `orbits` curves), moons and rings, galaxies and spiral arms, comets, constellations.
- **Particle physics & mechanics** - collisions, soft bodies and cloth, springs and chains, pendulums and harmonographs (`orbits`), shattering (`shatter_glass`), explosions (`embers`, `rocks`).
- **Biology & emergence** - cells and tissues, flocking and schooling (`Swarm`), reaction-diffusion patterns, predator-prey, ant-colony trails, slime molds. Many agents, local rules.
- **Waves & fields** - interference and standing waves, electromagnetic and gravitational fields, ripples on a membrane, the spectrum itself as a field (`harmonic_lattice`, `spectrum_ring`).
- **Chemistry & matter** - molecules and bonds, crystallization, phase changes (melt / freeze / boil), diffusion, combustion.

Filling any one row is a scene; filling the map is the package. The unifying mechanism is the [scene-spec pipeline](#roadmap) - one sampler that integrates these domains in adjustable ranges, so a single show can put snow on a city on a hillside under a moving sun, each layer a recipe rather than bespoke code.

## Running it

Open `project.godot` in Godot 4.6 and press play, or from the command line:

```
godot --path axis/vortex                       # opens the splash: import a song, pick Auto/Manual
godot --path axis/vortex -- --audio ~/track.wav # skip the splash, boot straight in
godot --path axis/vortex -- --storyboard default   # manual mode: play storyboards/default.json
godot --path axis/vortex -- --scene planes      # pin one scene for authoring
godot --path axis/vortex -- --no-splash         # auto mode, bundled/no audio, no splash
```

By default it opens the **splash** start screen (import a song from disk, choose Auto or Manual). Any of `--audio` / `--scene` / `--storyboard` / `--no-splash` boots straight past it. `--audio` accepts `.wav`, `.mp3`, `.ogg`, and `.flac`. FLAC has no runtime loader in Godot, so it is transcoded to a temp WAV via `ffmpeg` (must be on `PATH`); the others load natively.

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

- [x] Live analyzer → `AudioFeatures` → scene framework; a growing catalogue of scenes.
- [x] Behaviors (`static`/`drift`/`fluid`) - motion typed separately from geometry; originals kept as `static`.
- [x] Lifecycle (`loop`/`oneshot`) + spectral exit triggers (`beat`/`movement`/`lull`); exits land on the music; jump cuts with occasional blends.
- [x] **Primitive registry**: `Particle` / `ParticleSystem` + a `Primitives` force registry. Glass/rocks/embers composed from it.
- [x] **Gentle by construction**: bounded sway about rest poses (not unbounded spin), `Activation` sparse response with EMA decay, EMA-smoothed camera, typed framing pools (subject / field / plane), multi-instance planes.
- [x] **Real 3D**: `Mesh3D` software primitive; rocks and `wire_solid` are genuine 3D bodies. `Geo` fracture replaced the pizza-sliced glass.
- [x] **Swarm fields**: many-item scenes driven by local rules (`metropolis` city growth + colour pulses); transferable to any grid.
- [x] **Nonlinear / organic primitive layer**: a shared `Nonlinear` activation library, a curl-noise `Flow2D` field, and a `Filament` growth primitive - the source of "alive" (threshold/saturation, meander, crawling growth). `rooted_growth` rebuilt on it; a `filaments` scene (lightning / neural / thread) added.
- [x] **Composable geometry layers**: `Mesh3D` gained surface texture (`texturize`), a material (`gloss` / `roughness` specular), and a geometric+organic `hybrid` (gaussian-masked `warp_masked`); `Filament` gained the timelapse twitch (stable trunk, unstable tips). The idea: keep growing *complementary, sampleable modifiers* (geometry / texture / material / motion) per family.
- [ ] **Scene-spec pipeline** (the north star - "cattle, not pets"): a declarative spec that *samples a configuration* of geometry families + modifiers + materials + motion + lighting and composes them, so lifelike scenes emerge from randomly integrating many domains in adjustable ranges rather than from hand-crafted code. `rocks`/`bloom` already sample a small spec; generalize into a shared sampler across families, and lift the bodies (the `eye`'s hand-tuned numbers first) into sampled ranges.
- [ ] **Semi-automatic mode** (the third way to drive): start from the autopilot seed, then expose live **levers / dials / sliders** that influence the modulation at chosen points (iris hue, gaze energy, beat-split strength, …) and propagate downstream into the scenes in real time - steering a living system, not keyframing. Builds on the scene-spec pipeline (the dials are the spec's sampled parameters, surfaced and made tunable) and the `Workspace` control surface. Fast and responsive is the whole point.
- [x] **Colour over scale**: `Lighting` (moving hotspots, beat glow, hue drift) drives reactivity; shapes hold their form. Wired into ring / lattice / rocks / wire_solid so far.
- [ ] More `Swarm` rules: pheromone / ant-colony trails, reaction-diffusion, predator-prey - and abstract (non-city) many-item scenes that reuse the field.
- [x] **Render-kind typing**: every scene declares how it draws (`canvas` / `mesh3d` / `particles` / `swarm` / `scene3d`), making the carried-forward split explicit so it can converge.
- [x] **Forced-perspective 3D path**: a positionable `Lens3D` camera, a `Plane3D` quad primitive, and a `Scene3D` base that depth-sorts bodies + planes under one camera. `wire_solid` migrated, `planes` built native.
- [x] **Procedural geometry**: rocks are generated (coherent fractal `warp` + planar `facet` + stretch), not displayed spheres - the first "geometry from data" demonstrator.
- [x] **Novelty-weighted scheduling**: scenes least-recently shown are favored, so the show spreads across the catalogue instead of repeating.
- [x] **Authoring feedback console**: `` ` `` writes a reproducible scene record + screenshot to `feedback/`.
- [x] **Manual mode (storyboards)**: a JSON data spec mapping a song to a user-orchestrated linear sequence of scenes; a **splash** start screen imports audio and picks Auto/Manual.
- [ ] **Unified renderer, continued**: migrate the remaining 2D scenes onto `Scene3D` (planes for flat content) and route everything through one modulation surface (`Lighting` + materials), so any scene renders under one set of camera/light controls.
- [ ] **Richer storyboards**: per-entry parameter overrides, time-coded cues, per-transition blends, and eventually a manual editor that writes the files.
- [ ] **Procedural geometry kit**: extend `warp`/`facet` toward other "geometry from data" subjects (terrain, trees, crystals) feeding the `scene3d` world.
- [ ] **Restore the snowflake, as a field**: a six-fold dendrite primitive (lost when `bloom` absorbed the Koch stars), manifesting as several dozen flakes of varied size - spinning together, spinning independently, or carried on the wind - under the flat-subject discipline (sway, square-on, no fake tumble).
- [ ] **Weather scenes**: falling snow, rolling fog (the terrain fog generalized), rain, clouds - audio drives density / gust / colour, not shape.
- [ ] **Light crossing terrain (real shadow)**: a moving light source sweeping a rolling landscape and casting travelling shadows - true occlusion under `Lens3D`, beyond `Lighting`'s 2D hotspots.
- [ ] **Blocks breaking from the grid**: plot block harmonics on rolling landscape like buildings on a countryside (not a flat lattice), evolving via `Swarm` from a few small blocks into a sprawling city.
- [ ] **The complete package** (the horizon): grow the primitive kit until the catalogue spans the physical sciences - see [Toward a complete package](#toward-a-complete-package-modeling-the-physical-sciences).
- [ ] Grow the kit (flow fields, collisions, soft-body links, emitters) and migrate more bespoke scenes onto particles / `Mesh3D`.
- [x] **Video export**: the Export button relaunches in Movie Maker mode (`--write-movie --use-bake`) to render the visualization + synced audio to a file, reproducing the session via `--seed` and driving reactivity from the offline bake. Runs as its own (background) render so the live session stays interactive.
- [x] **Bake mode**: `SpectrumBake` decodes the song (ffmpeg) and runs an offline FFT into the same 64 log bands, a deterministic per-frame timeline. The export render drives visuals from it (`--use-bake`) instead of the analyzer, so the recorded reactivity is correct under Movie Maker's offline audio *and* the render runs as fast as the machine allows. Live sessions stay on the real-time analyzer.
- [ ] Stronger beat/onset and tempo tracking; exits that snap to bars, not just beats.
- [ ] Definition / behavior / lifecycle presets per song (config-driven, like the other generators).

## Status

Working framework, live path end to end: import a song on the splash (or boot a CLI flag), watch seeded scenes react and change *with the music* (Auto), or hand-orchestrate them with a storyboard (Manual). Geometry, motion (behaviors), lifecycle, exit cue, render kind, **and transitions** (typed morphs) are independent typed axes; the physics is a composable **primitive registry**; there's a real **forced-perspective 3D path** (`Lens3D` / `Scene3D` / `Plane3D`) that the 2D scenes are migrating onto; bodies like `rocks`, the `PrismBody`, and the 3D `EyeBody` are built from sampled/layered primitives ("cattle, not pets"); and **video export** renders the show + audio to a file via an offline bake. An in-app **feedback console** (`` ` ``) captures a scene record + screenshot for authoring.

The throughline is the **scene-spec pipeline**: keep growing complementary, sampleable primitive domains and let them integrate themselves, so scenes occur naturally rather than being hand-built. That pipeline is also what unlocks the **semi-automatic** third mode - the autopilot seed plus live dials that steer the modulation. Standalone; it does not import or depend on Praxis.
