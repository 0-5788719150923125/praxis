# ghost

A spectral audio visualizer built with [Godot](https://godotengine.org/) 4.6 - _spectral_ both ways: it draws from the audio **spectrum**, and the things it conjures (drifting fog, falling snow, fireflies, auroras) haunt the frame like apparitions. Point it at a `.wav`, and it draws geometry in response to the sound - rings, planes, harmonics, lattices, weather, whatever - then loops through scenes like a video you can record full-screen.

It is the same move as the Arena and business-card generators elsewhere in this repo, pushed at audio: a **scene definition** (a typed bag of parameters) is passed through **typed transformations** that are modulated by audio features - amplitude, per-band energy, beat, velocity. Procedural, deterministic, and cheap. No generative AI in the render path.

This README carries the philosophy; the **reference documentation is generated from the source** (doc comments, registries, the scene roster) into [docs/index.md](docs/index.md) - every script, scene, layer, force, Masking effect, and CLI flag, regenerated with `python docs.py` so it cannot drift.

## Why

Generating visuals for a song with an image/video model is slow and expensive. A procedural visualizer is free, runs in real time, and is infinitely customizable. The show is **spectrally deterministic**: the session seed is derived from the audio's own fingerprint, so the _same song always produces the same imagery_ - the visuals are a function of the sound, not a fresh random roll. `--seed N` overrides it (the exporter passes it so a render reproduces a session, and it is how you deliberately roll a _different_ show for the same song).

## The idea: cattle, not pets

Most 3D work treats each object as a **pet**: you hand-model _this_ eye, _this_ rock - name it, tune it, love it. It doesn't scale and it never surprises you. ghost is built the other way - **cattle**: an object is a _recipe_ of layered primitives whose parameters are **sampled from adjustable ranges**, so every instance is a fresh, naturally-occurring variation.

The discipline, taken to its limit: **every tunable constant is a candidate for sampling.** Wherever a number could be perturbed and the geometry / animation / shading would still read right, it should be drawn from an intelligent range (per instance, around a sensible centre) rather than baked as one value - so two things of a kind still differ, and the visualizer gains expression for free. `rocks` is the worked example: not one material per style, but a material _sampled per rock_; not a reveal flag, but a reveal _threshold sampled across a spectrum_.

Take **the eye** (`EyeBody`) as the worked example. A pet eye is a bespoke mesh. But an eye is _really_ just a few primitives stacked: a **sphere** for the ball, a thin pliable **lens** (a cornea dome) stretched across the front, a recessed **iris** cap, a **pupil** hole - plus colours, hues, gloss, and a light. Model _those layers_ and _sample their ranges_ - iris hue and saturation, pupil dilation, corneal curvature and gloss, eyeball size, how restless the gaze is - and you don't get one eye, you get the _space of all eyes_, occurring naturally. (Today's `EyeBody` is the first step: the layers are real 3D primitives, but several of their numbers are still hand-tuned. Lifting those into sampled ranges - turning the pet into cattle - is the [scene-spec pipeline](#roadmap) north star.) The same move is already in `rocks` (a sampled stack of geometry + texture + material) and `bloom` (a whole family of shapes from a handful of superformula numbers).

### Three ways to drive it

1. **Auto** - the autopilot. The session seed - derived from the song's own audio fingerprint - rolls the whole show; the Director picks scenes by novelty and cuts on the music. **Spectrally deterministic**: the same song always plays the same show (a different song, a different one), so the imagery belongs to the sound. `--seed N` overrides it to roll a fresh variant on purpose.
2. **Manual** - a **storyboard**. You author the exact sequence, cues, cast and choreography as data (`storyboards/*.yaml`, JSON also accepted), the way "the-point" lays out its full arc as five data-driven **stage** entries - actors from a registry, verbs on a timeline, live bodies carried across the cuts. Every scene described, not coded.
3. **Semi-automatic** _(the one to be excited about; first rung shipped)_. Start from the **same autopilot seed**, then reach in and **pull levers, turn dials, move sliders** that influence the modulation at chosen points - and watch the downstream scenes change in response, live. Not keyframing; _steering a living system_. The first instrument is the **Dial** (bottom-right of the Workspace): not a parameter knob but a **sculpting tool**. Turning it injects energy into seeded modulation *deposits* - each surges briefly, then decays into a smaller **standing pattern** the scene keeps, so turning is purely additive over the session. One revolution passes through **5 or 6 wedges**, each with its own seeded signature (which channels it bends - size, hue, tempo, drive, drift - at what frequencies, with what waveforms); crossing into the next revolution increments the **turn counter** and re-rolls the vocabulary. What a dial does is deliberately arbitrary, but **deterministic per song**: the signatures derive from the session seed, so the same song and the same gesture answer the same way. More dials, each with its own signature, come next; `--dial-demo` turns the first one hands-free for demos and renders.

## The shape

```
  .wav ──▶ Spectrum ──▶ AudioFeatures ──▶ GhostScene ──▶ screen / video
           (analyzer)    (typed, per-frame)  │  (definition × behavior)
                                  movement ──┤
                                             ▼
                              Director: cut on the music, blend sometimes
```

- **`Spectrum`** (autoload) owns the `AudioStreamPlayer` and the `AudioEffectSpectrumAnalyzer` bus effect. Every frame it emits one **`AudioFeatures`**: overall `energy`, named bands (`bass`, `low_mid`, `mid`, `high`, `treble`), a smoothed `beat` pulse, the full `bands` array, `time`, and - new - `flux` and `movement` (a sliding-window measure of how much the spectrum is _changing_, used to time scene cuts). This is the single typed interface every scene reads - scenes never touch the audio engine directly.
- **`GhostScene`** (`class_name`, base script) is one visualizer. `build_params(rng)` rolls a **definition** from a seeded RNG (the typed parameter bag), `update(features, delta)` modulates that definition by the audio, and `_draw()` renders it through the scene's **view** (a centered camera: zoom / tilt / rotate / off-center). Add a scene by subclassing this.
- **Motion is its own axis.** _What_ a scene draws (the definition) is separate from _how it moves_ (a **behavior**). The base provides a **`ModBank`** of slow seeded oscillators pooled into named organic channels (`mod.value("sway")`) and a per-element `wobble(key, i)` offset.
- **Physics is a registry, not per-scene code.** The duplicated dynamics (scatter, gravity, springs) are extracted into **`Primitives`** - a registry of reusable **force** modules (`gravity`, `spring`, `drag`, `scatter`, `wind`, `pulse`, `orbit`, `wobble`), each a small class with constants baked in via its config, the Praxis registry move. A scene builds a **`ParticleSystem`** (a bag of `Particle`s = geometry) and **composes forces by key**; the substrate steps them and reports `settled()` for oneshots. The same `scatter` bursts glass, rocks, and embers - cross-contamination is free, and a new scene is mostly a parts list (see `embers`).
- **Appearance is a registry too - the integration axis.** `Primitives` composes _physics_; **`Layer`** composes _appearance_ - the visual sibling registry. A **layer** is a self-contained visual component that seeds itself, advances on the audio, and draws itself onto the scene's canvas: `bed` (a colour wash with breathing pools), `fog` (rolling banks), `snow` (falling flakes, with procedural six-fold dendrites for the near ones), `rain`, `fireflies`, `stars` (twinkle + shooting stars), `aurora` (flowing curtains), `petals`, `dust` (motes in a light shaft), `bubbles`, `embers`. Any scene composes them through `add_layer(key, rng, cfg)` / `update_layers` / `draw_layers(z)` on the base. **This is the missing integration**: the same `snow` that is a scene on its own (`snowfall`) also falls over the `cityscape` skyline and the `gaussian_landscape` hills (stars drawn `z = "back"` behind the geometry, snow `"front"` over it) - weather is a component, not bespoke per-scene code. Layers draw in unit-fraction space and are handed the visible half-extents each frame, so they fill any aspect / fullscreen / 4K without baring an edge.
- **Behaviors** (`static` / `drift` / `fluid`) are typed presets of motion gain. `static` freezes the camera and per-element motion - the scene reacts to audio alone (the original, un-modulated look). `drift` adds gentle whole-scene camera breathing. `fluid` turns on independent per-element motion. The same geometry, kept as several options. Determinism is preserved: same song, same behaviors, every run.
- **Structure is the bias; motion is bounded variance** - for _flat_ things. A flat subject (snowflake, glass pane) sways about a seeded rest pose rather than spinning, and the camera never rolls or shears flat 2D content (rolling/shearing a plane is fake 3D - `drift_view` does only zoom + pan). Genuinely 3D bodies (`Mesh3D`) are the exception: they rotate slowly and continuously, because that is how a real solid reveals its volume. **`Activation`** carries the same idea to elements: each gets a seeded threshold + gain through a soft nonlinearity and a fast-attack/slow-decay EMA, so with `sparsity > 0` some elements stay rooted (the static floor) while others bloom; `sparsity = 0` means everything moves. And the **camera eases** (the `SceneView` EMA-smooths toward its target), so every move is gentle.
- **Every scene declares its render kind.** The project has carried several rendering mechanisms forward (the "split"), so each scene now names _how_ it draws on a typed axis - `canvas` (flat 2D), `mesh3d` (software 3D bodies projected onto the canvas), `particles`, `swarm`, or `scene3d` (the unified path below). Naming the divergence is the first step to converging on it; the kind rides along in every feedback record so a critique is tied to the renderer that produced it.
- **Real 3D where it matters.** `Mesh3D` is a software 3D primitive (icosphere / cube / octahedron / tetrahedron → depth-sorted flat-shaded faces). `Geo` holds the shared polygon helpers (convex split, fracture) that shatter the glass.
- **A unified 3D path (`scene3d`), the convergence target.** A real, positionable perspective camera (`Lens3D`: an eye looking at a target with a field of view) replaces the old fixed centred projector - so a scene can push in, orbit, and frame _in depth_, and a wide lens up close gives **forced perspective** (near geometry looms over far, the dimensional read a sheared 2D plane only fakes). `Plane3D` is a flat quad genuinely placed in 3D space (stack them for parallax, stand them as bars, tilt them into depth - geometry, not a sheared card). `Scene3D` is the base that owns the lens and a world of `Mesh3D` bodies + `Plane3D` quads, depth-sorts the lot back-to-front, and draws it - so bodies and planes correctly occlude each other under one camera. `wire_solid` is migrated onto it and `planes` is built on it; the rest migrate here over time.
- **Many items, local rules.** `Swarm` is a scalar field over a grid that evolves by _local_ interaction - development creeping out from seeds (`GROW`), or injected pulses diffusing across the lattice (`WAVE`). It drives thousands of items without scripting each one (the `metropolis` city grows and pulses from one), and the same mechanism transfers to any abstract many-item grid - the cellular / ant-colony idea.
- **Sound drives colour, not scale.** Pulsing geometry _size_ with amplitude reads as cheap throbbing - shapes hold their form. Instead, audio drives **colour, brightness, and glow** through `Lighting`: moving bright **hotspots** that sweep the frame (region-aware lighting / gradient swipes), a global **glow** that flares on beats and decays slowly, and a slow hue drift. A scene asks `light.at(pos)` for a local brightness boost and `light.glow()` for the global flare. This is the shared modulation surface a future unified renderer (2D or 3D under one control) will route everything through; for now scenes opt in.
- **Framing is a typed axis.** A scene declares a `framing` class and the **`Shots`** registry assigns a camera move from the matching pool: expressive for `subject` (offset / push / pan / canted), gentle for `field` fillers, square-on for a single `plane` (so a flat snowflake or pane never reads as a tumbling card). And planes can spawn in multiples - a few small ones don't look stranded the way one lone spinning plane does.
- **Lifecycle and exit cues are typed too.** A scene either **loops** until cut, or is a **oneshot** that plays one sequence and reports `finished()` (shatter glass settling, then ending). Once a scene is _eligible_ to exit - a loop past its minimum hold, or a oneshot that finished - the **`Director`** waits for a chosen spectral **trigger** before actually cutting, so exits land on the music: usually a **beat** (rising edge), sometimes a **movement** (section change) or a **lull** (drop into quiet), with a maximum-hold backstop. Triggers are picked weighted per scene.
- **`Director`** (autoload) holds the registry of `{scene, behavior}` pairs, runs the lifecycle/trigger logic above, and performs the change. In auto mode it mostly **dips to black** (the old scene fades out, a beat of true darkness, then the new one fades up) with the occasional hard **cut**; a storyboard sets its own style (the-point forces 100% **cuts**). `--scene <name|N>` pins one scene for authoring.
- **Transition style is a hierarchy.** Highest wins: a compatible **morph** (below) → the storyboard **entry**'s `transition` → the **scene**'s own `transition_style` (set in build_params) → the storyboard's top-level default → the mode default (manual = `cut`, auto = the weighted dip/cut bag). So a storyboard can force "cut" for everything yet a single scene can still ask for a dip, and the-point gets 100% cuts except where a morph applies.
- **Content-aware transitions (typed morphs).** A scene declares the geometry it leaves (`morph_out`) and what it can grow in from (`morph_in`). When the next scene's `morph_in` matches the current scene's `morph_out`, the Director plays a **morph** - an instant swap where the incoming scene animates out of the outgoing shape - instead of a cut/dip, and hands over a typed `morph_payload()` so the transition is _continuous_: the single eye passes its colour/gaze/size to `two_eyes`, which starts as that exact eye and splits into two identical copies. Mismatched or empty types fall back to a cut, so a bespoke transition is only ever attempted between compatible geometries - it can't break.
- **Novelty-weighted scheduling.** Picking the next scene uniformly at random clusters - the same _kind_ recurs while others go unseen. Instead each candidate is weighted by how long its kind has gone unshown, so long-unseen scenes are drawn far more often than recent duplicates (and never two of one kind back to back). A soft priority queue - seeded by the session seed (the audio fingerprint by default, so it is the same for a given song; `--seed N` to override).
- **Ways to drive the show** (see [Three ways to drive it](#three-ways-to-drive-it)). _Auto_: the Director chooses scenes by novelty and cuts on the music. _Manual_: plays a **storyboard** - a user-authored linear sequence (`storyboards/*.yaml`), each entry naming its scene, behavior, shot, and exit rule (a fixed `hold`, a musical `beat`/`movement`/`lull` cue, or the scene's own lifecycle); see `storyboards/README.md`. _Semi-automatic_ (planned): the autopilot seed plus live levers/dials that steer the modulation. The storyboard is the first rung; the data spec grows toward a manual editor and then the semi-auto control surface.
- **A feedback channel for authoring.** "This shape feels wrong" is hard to act on from a note; the `FeedbackConsole` (press `` ` ``) captures the scene on screen - its typed descriptor (name / kind / behavior / shot / seed / params / the audio frame) plus your typed query _and a screenshot_ - into `feedback/NNNN.{json,png}`. The seed makes it reproducible; the image shows what "wrong" looked like.
- **`Splash`** is the start screen: a list of all four modes, always visible - **Auto** (the seeded show), **Manual** (the workspace + storyboards), **Synthesis** (write a script, ghost speaks it), **Masking** (chroma-key a video) - each with a one-line description and what it consumes. One Import dialog fills the song and clip slots by extension (both remembered in `user://ghost.cfg` and shown side by side); the mode button _is_ start, there is no separate one. CLI flags (`--audio` / `--scene` / `--storyboard` / `--synth` / `--mask-edit` / `--no-splash`) boot straight past it for authoring and automation.
- **`Workspace`** is the manual-mode surface (scaffolding): opened by the Manual button over a session running `storyboards/default.yaml`, a left-side panel lists the storyboards and clicking one switches the live show to it. This is the canvas the future hand-authoring tools (per-entry params, reordering, a timeline, save) will grow into.
- **Session lifecycle.** `main` owns it: splash → start a session (Auto or Manual). When the song ends, an **Auto** session tears down and returns to the splash (`Spectrum.song_finished` → `Director.detach()` + `Spectrum.stop()`); a **Manual** session is **endless** - the audio loops in place (`Spectrum.replay()`) and the show stays up: the Dial's deposited modulations carry across the loop, and the sequence **restarts with the song by knowing the content itself** (see `Echo` below): the map spans exactly one hearing, so when the finished arc walks past its end the next content is necessarily the top, and the arc rolls over aligned - the eye returns the moment the song does. The same sections of the song replay the same scenes (position-keyed seeds), with live actors morph-carried across. It also maps the global keys (next / full-screen / feedback / quit).

## Rendering: live & baked

(Distinct from the three _driving_ modes above - this is how frames get produced.)

1. **Live (default).** Audio plays through the analyzer bus; scenes react in real time. Use this to author and preview, and screen-record it with OBS for a quick capture. The window stretches in **`canvas_items`** mode, so 2D is rasterized at the monitor's native resolution (`F11` fullscreen is crisp, not an upscale of the 1080p base) while the coordinate system scales _proportionally_ - UI and scene content keep their relative size and snap back exactly when the window returns to its original size. The export overrides this with an offscreen **`viewport`**-mode buffer so it can render above the physical display (true 4K on a 1080p monitor).
2. **Baked (for export).** The Export button first asks for a **quality** - **720p · 30 fps**, **1080p · 60 fps** (native), or **4K · 60 fps** (full resolution) - then runs two background processes: a **headless** `bake_runner` first analyzes the song into a spectrum timeline (`SpectrumBake`, cached per song), then a **Movie Maker** render (`--write-movie out.avi --fixed-fps <fps> --bake-file …`) loads that timeline and drives the scenes from it instead of the live analyzer - frame-perfect, with synced audio, and unaffected by the fact that Movie Maker's offline audio would otherwise make the live analyzer unreliable. Keeping the analysis out of the render means the render never blocks on a grey frame. The live authoring session keeps using the real-time analyzer. The chosen resolution is set via a transient **`override.cfg`** (the exporter writes it before the render and removes it after): Movie Maker locks its output resolution to the project's viewport size at engine startup, before any script can run, so `override.cfg` - which Godot reads from the project root at boot - is the only lever; it uses `viewport` stretch mode so the frames are an offscreen buffer of exactly that size, true 4K even on a 1080p display, and shrinks the OS window itself to a small floater (`window_*_override`) rather than minimizing it - **a minimized window stops rendering** (Godot skips drawing when `window_can_draw()` is false) and Movie Maker then records the last drawn frame over and over: seconds-long freezes, or on Wayland an entirely black video, while the audio keeps advancing. The exporter also repairs the scratch AVI's headers after a long render: past 4 GiB Godot's 32-bit RIFF/movi size fields wrap (the chunks themselves stay intact to EOF), and zeroing the two sizes ("unknown - read to end") keeps every demuxer walking the file sequentially instead of trusting the lie. CLI renders that bypass the exporter should transcode oversized AVIs to MP4 promptly for the same reason.

## Masking: the video masking editor

ghost carries a second app surface alongside the visualizer: **Masking**, a video chroma-key masking editor (`--mask-edit <video-or-session>`). A session is a stack of **markers** over a source video; each marker becomes a shader layer driving one effect - implemented in `shaders/mask_split.gdshader` and keyed volumetrically off the video's own colour. Sessions autosave under `masks/`, carry undo/redo and multi-track lanes with per-track trim/shift, and render to video headlessly. The control panel is data-driven: each effect declares which control groups it consumes (`MaskSession.EFFECT_CONTROLS`), so a slider that does nothing for the selected effect is never shown. See [docs/masking.md](docs/masking.md) for the data model, the full effect/control tables, and the headless marker-insertion tool.

## Layout

<!-- AUTODOC:LAYOUT:BEGIN -->

Top-level layout; the per-script map (every class, one line each) is [docs/index.md](docs/index.md).

- `project.godot` - Godot 4.6 project; autoloads `Boot`, `Spectrum`, `Director`; `scenes/main.tscn` is the entry scene.
- `scenes/` - The Godot entry scene (`main.tscn`). Everything else is code-built.
- `scripts/` - All GDScript. Per-script map in [docs/index.md](docs/index.md); the subsystem groups are described there too.
- `scripts/scenes/` - The visualizer scene catalogue - one class per scene. See [docs/scenes.md](docs/scenes.md).
- `shaders/` - The two GPU surfaces: `flame.gdshader` (fire layer), `mask_split.gdshader` (all Masking effects).
- `storyboards/` - Manual-mode scene scores (YAML; JSON accepted). [storyboards/README.md](storyboards/README.md) is the data spec.
- `masks/` - Saved Masking sessions, one directory per source video (runtime, git-ignored).
- `tests/` - Headless check scripts (`godot --headless --script tests/<x>.gd`).
- `reference/` - Reference imagery scenes were prototyped from.
- `docs/` - Generated documentation. Regenerate with `python docs.py`; do not edit by hand.
- `feedback/` - Feedback console output: `NNNN.json` + `NNNN.png` per report (runtime, git-ignored).
- `audio/` - Drop a `song.wav` here to bundle one (runtime, git-ignored); or use `--audio`.

<!-- AUTODOC:LAYOUT:END -->

## Scenes

Each scene is a small combination of shapes; behavior decides how it moves. The catalogue below is generated from each scene's own doc comment - [docs/scenes.md](docs/scenes.md) carries the full versions, plus each scene's morph types and composed layers.

<!-- AUTODOC:SCENES:BEGIN -->

43 scenes in the auto rotation (45 on disk). One line each - the full catalogue, with every scene's own documentation, is [docs/scenes.md](docs/scenes.md).

_Core catalogue_

- **`bloom`** (canvas, drift, static) - elegant procedural rosette curves (the koch replacement).
- **`cityscape`** (canvas, static) - a skyline of rectangles that grows with the music.
- **`clockwork`** (canvas, drift, static) - meshing gears under forced restraint, dramatic by physics not by clipart.
- **`embers`** (particles, drift) - a drift of sparks that twinkle and flare, each on its own.
- **`filaments`** (canvas, drift, static) - the procedural-growth primitive, showcased.
- **`fog_lights`** (canvas, drift) - soft lights breathing under a drifting cloud cover.
- **`furry`** (canvas, drift, fluid) - dense, thick, long tufts of fur/hair, magnetized rather than random.
- **`gaussian_landscape`** (canvas, drift) - rolling terrain with fog in the valleys.
- **`harmonic_lattice`** (canvas, drift, static) - a grid of cells that breathe with the spectrum.
- **`metropolis`** (swarm, drift) - a city of thousands of blocks growing over a countryside.
- **`planes`** (scene3d, drift, static) - the spectrum as a ring of real planes under a forced-perspective camera.
- **`rocks`** (mesh3d, drift, oneshot seeds) - faceted stones in real 3D, sampled from a small material/geometry spec.
- **`rooted_growth`** (canvas, drift) - crawling roots and tendrils that spread from a seed.
- **`shatter_glass`** (scene3d, drift, oneshot seeds) - a real pane of glass, shattering in true 3D.
- **`spectrum_ring`** (canvas, fluid, static) - the spectrum bent into a circle.
- **`strata`** (canvas, drift) - stacked waveform planes receding into depth.
- **`voxel_blocks`** (canvas, static) - an isometric heightfield equalizer.
- **`wire_solid`** (scene3d, drift) - a translucent polyhedron on the unified 3D path.

_Weather & atmosphere_

- **`aurora`** (canvas, drift) - slow curtains of light over a starlit night.
- **`bubbles`** (canvas, drift) - an underwater drift of rising bubbles in coloured depths.
- **`clouds`** (canvas, drift, static) - REAL 3D cloud masses drifting across the sky, lit by the sun.
- **`fire`** (canvas, drift) - a living flame attuned to the harmonics.
- **`fireflies`** (canvas, drift) - a dusk meadow sparkling with wandering lights.
- **`fog_bank`** (canvas, drift) - rolling coloured fog, light glowing from within.
- **`fog_volume`** (canvas, drift) - REAL 3D fog: a low, wide bank of soft gaussian puffs receding into depth, lit volumetrically (a brighter sunward edge fading into a dim core) and slowly drifting. A genuine haze with simulated dynamics, not a flat 2D wash. `bed` + `volumetric` (fog mode).
- **`motes`** (canvas, drift) - dust adrift in a shaft of light.
- **`petals`** (canvas, drift) - blossom or leaves drifting down on a soft breeze.
- **`rainfall`** (canvas, drift) - slanting rain over a brooding sky, fog rolling through it.
- **`snowfall`** (canvas, drift) - a quiet field of falling snow over a soft colour bed.
- **`snowflakes`** (canvas, drift) - a field of several dozen crystal dendrites, restored and multiplied.
- **`starfield`** (canvas, drift, static) - a deep night sky, twinkling, with the occasional shooting star.
- **`underwater`** (canvas, drift) - looking up through flowing water: shafts of light from the surface, bubbles rising, a deep blue-green wash. The submerged corner of the weather catalogue.

_Worlds & projections_

- **`projection`** (canvas, drift, static) - a PCA-style density map of a latent geometry, eye-shaped.
- **`spires`** (scene3d, drift) - a fractal metropolis of harmonic spires over a landscape.
- **`terrain`** (scene3d, drift, static) - real 3D landscapes built from the composable `Field` / `Terrain` foundation.
- **`terrain_city`** (scene3d, drift) - blocks rising as a city over real 3D terrain, growing nonlinearly.

_The-point scenes_

- **`eye`** (scene3d, static) - a single human eye in the black void (the-point, scene 1).
- **`eye_prism`** (scene3d, static) - Eye + prism - the right eye becomes its digital self (the-point, scenes 3-5).
- **`prism`** (scene3d, static) - a single living wireframe tetrahedron (from "the-point").
- **`prism_split`** (scene3d, static) - one prism strains, then breaks into two (from "the-point").
- **`prism_swarm`** (scene3d, drift) - the swarm forms, flies the track, splits into a helix, and jumps (the-point, scenes 12-15).
- **`two_eyes`** (scene3d, static) - the single eye split into two (the-point, scene 2).
- **`two_prisms`** (scene3d, static) - the pair, from the drop through specialization (the-point, scenes 6-11).

<!-- AUTODOC:SCENES:END -->

### Provenance: `spires`

<p align="center"><img src="./reference/arcbot.webp" width="480" alt="Reference photograph of ornate towered architecture that inspired the spires scene"></p>

Prototyped from the photograph above: procedural generation reading a real building's geometry, harmonics, and periodicity - the recursive tiering, the repeating turret motif, the way height and ornament scale together - and turning what one photograph fixed in a single instant into a generative rule that produces endless, never-repeating variations, driven by whatever song is playing.

### The "the-point" arc

The five `eye → two_eyes → eye_prism → two_prisms → prism_swarm` scenes come from a planned video - a continuous 33-second arc: an eye becoming its digital self, coming alive, then a swarm. Each is one object-composition, and consecutive scenes hand their **live bodies** across the cut through a content-aware morph (`morph_out` / `morph_in` + `morph_payload`), so an eye or a prism literally continues instead of being re-created. The bespoke files now serve the AUTO catalogue only: the default storyboard plays the same arc - all fifteen beats of the brief, including the blink the bespoke scenes never had - as data-driven `stage` entries (see `storyboards/`).

## Toward a complete package: modeling the physical sciences

The long arc is simple to state and enormous to fill: **ghost should be able to model anything physical.** Every scene above is a recipe of sampled primitives ("cattle, not pets"); the goal is to keep growing the primitive kit until the catalogue spans the natural world - weather, light, crystals, terrain, structures, growth, fluids, the cosmos - so that pointing it at a song can summon _any_ phenomenon, alone or in combination. The list below is the standing backlog of subjects to model: a domain-by-domain map of what a "complete package" contains. Most reuse primitives we already have (`Mesh3D`, `Swarm`, `Filament`, `Flow2D`, `Lighting`, `Lens3D`, the force registry); the work is composing them and lifting their numbers into sampled ranges.

### Next up (the immediate targets)

- ~~**The snowflake, restored and multiplied.**~~ **Done** - `snowflakes`: several dozen procedurally-generated six-fold dendrites of varied size, spinning together / independently / on the wind, under the flat-subject discipline. Built on the new `Layer.draw_flake` helper.
- ~~**Partial rocks (wireframe reveal).**~~ **Done** - a gaussian alpha mask reveals the wireframe lattice through the coat (sampled masking threshold, sub-face, back-faces culled); see the `rocks` entry. Still open: **photoreal** stone (texture / roughness / height-relief) to pair with it.
- ~~**Weather effects.**~~ **Done** - a whole `Layer` registry of composable visual components and a family of weather scenes (`snowfall`, `rainfall`, `fog_bank`, `fireflies`, `starfield`, `aurora`, `petals`, `bubbles`, `motes`), _and_ the same components composed over existing geometry (snow over `cityscape` / `gaussian_landscape`). Audio drives density / gust / colour, not shape. (Clouds + a lightning-storm scene are the remaining weather pieces.)
- **Light crossing terrain (real shadow).** A **moving light source** sweeping over a rolling landscape and **casting shadows** that travel with it - a day-arc across hills, the first scene where shadow (occlusion from a positioned light, not just `Lighting`'s hotspots) does the storytelling. Pairs the `gaussian_landscape` terrain with a shadow pass under `Lens3D`.
- **Block harmonics, breaking from the grid.** The large-scale block scenes (`voxel_blocks`, `metropolis`) still read as a regular lattice. Break from it: **plot the blocks on rolling landscape, like buildings on a countryside** - not a flat grid but structures following the terrain. Then make it **evolve**: start with a few small blocks and grow (via `Swarm`) into a **sprawling city of block shapes**, the skyline thickening with the music. (The `Swarm` `growth` mode is the seed of this; the new part is decoupling the plot from the grid and standing it on real terrain.)

### The domain map (the standing backlog)

- **Weather & atmosphere** - snow, rain, fog, fireflies, stars, aurora, petals, bubbles, dust (all shipped as `Layer` components + scenes); still open: clouds, wind streaks, hail, heat shimmer, a lightning storm (the `Filament` lightning life is the start). Density / intensity ride the audio.
- **Light & shadow** - a positioned, moving light casting real shadows over geometry; day/night sweeps; god rays through fog; caustics; refraction and dispersion through the glass/prism. Beyond `Lighting`'s 2D hotspots toward true occlusion.
- **Crystals & symmetry** - snowflakes (six-fold dendrites), mineral crystals and lattices, growth by accretion, kaleidoscopic symmetry groups. Reuses `Mesh3D` + `Filament`-style dendritic growth.
- **Geology & terrain** - real 3D heightfields from the `Field`/`Terrain` foundation (hills, mountains, valleys, canyons, mesas, islands + water; `terrain` scene); still open: erosion, rivers, plate motion, volcanoes. Terrain as the stage many other scenes stand on.
- **Structures & cities** - a `Swarm` city growing on real terrain, blocks oriented to the surface (`terrain_city`); still open: bridges, lattice frameworks, ruins, roads/districts. The "breaking from the grid" thread, realized.
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
godot --path axis/ghost                       # opens the splash: import a song, pick Auto/Manual
godot --path axis/ghost -- --audio ~/track.wav # skip the splash, boot straight in
godot --path axis/ghost -- --storyboard default   # manual mode: play storyboards/default.yaml
godot --path axis/ghost -- --scene planes      # pin one scene for authoring
godot --path axis/ghost -- --no-splash         # auto mode, bundled/no audio, no splash
```

By default it opens the **splash** start screen: all four modes listed - Auto, Manual, Synthesis, Masking - plus one Import for the song/clip slots. Any of `--audio` / `--scene` / `--storyboard` / `--synth` / `--mask-edit` / `--no-splash` boots straight past it. `--audio` accepts `.wav`, `.mp3`, `.ogg`, and `.flac`. FLAC has no runtime loader in Godot, so it is transcoded to a temp WAV via `ffmpeg` (must be on `PATH`); the others load natively.

Controls: `Space` next scene · `F11` full-screen · `` ` `` send feedback · `Esc` quit.

The feedback key writes `feedback/NNNN.json` (the scene descriptor + your query) and `feedback/NNNN.png` (a screenshot of that frame) - a reproducible record of anything that looks off.

If no audio is found it still runs - scenes just animate on an idle clock with zeroed features, so you can develop a scene with no song loaded.

## Adding a scene

```gdscript
extends GhostScene

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

To compose **weather / atmosphere**, add layers in build_params and drive them from update/\_draw - the appearance equivalent of composing forces:

```gdscript
func build_params(rng):
    framing = "field"
    add_layer("bed", rng, {"hue": 0.6})      # colour wash behind everything
    add_layer("snow", rng, {"count": 100})   # falling flakes over it
    return {}

func update(f, delta):
    tick(f, delta); update_layers(f, delta); queue_redraw()

func _draw():
    begin_draw(); draw_layers()               # or draw_layers("back") ... geometry ... draw_layers("front")
```

The same layer is reusable across scenes (that is the point); a geometry scene draws `draw_layers("back")` before its geometry (e.g. stars) and `draw_layers("front")` after (e.g. snow). See `snowfall.gd` (pure layers) and `cityscape.gd` (layers over geometry).

Set `render_kind` in `build_params` so the scene is typed (`"canvas"` is the default; use `"particles"` / `"swarm"` / `"mesh3d"` as appropriate). For a 3D scene, **extend `Scene3D` instead of `GhostScene`**: it sets `render_kind = "scene3d"`, gives you a `lens` ([Lens3D]), and renders a depth-sorted world of `Mesh3D` bodies (`add_body(...)`) and `Plane3D` quads (`add_plane(...)`) when you call `render_world()` from `_draw` - so you build geometry in real 3D space and fly the camera, rather than shearing 2D. See `planes.gd` and `wire_solid.gd`.

## Roadmap

- [x] Live analyzer → `AudioFeatures` → scene framework; a growing catalogue of scenes.
- [x] Behaviors (`static`/`drift`/`fluid`) - motion typed separately from geometry; originals kept as `static`.
- [x] Lifecycle (`loop`/`oneshot`) + spectral exit triggers (`beat`/`movement`/`lull`); exits land on the music; jump cuts with occasional blends.
- [x] **Primitive registry**: `Particle` / `ParticleSystem` + a `Primitives` force registry. Glass/rocks/embers composed from it.
- [x] **Gentle by construction**: bounded sway about rest poses (not unbounded spin), `Activation` sparse response with EMA decay, EMA-smoothed camera, typed framing pools (subject / field / plane), multi-instance planes.
- [x] **Real 3D**: `Mesh3D` software primitive; rocks and `wire_solid` are genuine 3D bodies. `Geo` fracture replaced the pizza-sliced glass.
- [x] **Swarm fields**: many-item scenes driven by local rules (`metropolis` city growth + colour pulses); transferable to any grid.
- [x] **Nonlinear / organic primitive layer**: a shared `Nonlinear` activation library, a curl-noise `Flow2D` field, and a `Filament` growth primitive - the source of "alive" (threshold/saturation, meander, crawling growth). `rooted_growth` rebuilt on it; a `filaments` scene (lightning / neural / thread) added.
- [x] **Composable geometry layers**: `Mesh3D` gained surface texture (`texturize`), a material (`gloss` / `roughness` specular), and a geometric+organic `hybrid` (gaussian-masked `warp_masked`); `Filament` gained the timelapse twitch (stable trunk, unstable tips). The idea: keep growing _complementary, sampleable modifiers_ (geometry / texture / material / motion) per family.
- [ ] **Scene-spec pipeline** (the north star - "cattle, not pets"): a declarative spec that _samples a configuration_ of geometry families + modifiers + materials + motion + lighting and composes them, so lifelike scenes emerge from randomly integrating many domains in adjustable ranges rather than from hand-crafted code. `rocks`/`bloom` already sample a small spec; generalize into a shared sampler across families, and lift the bodies (the `eye`'s hand-tuned numbers first) into sampled ranges. **First structural rung shipped**: the storyboard `stage` spec (cast + track + verbs, every number a range sampled through `Storyboard.sample`) is this pipeline at the choreography level - the remaining work is pushing the same spec DOWN into the bodies' own geometry/material numbers.
- [ ] **Semi-automatic mode** (the third way to drive): live controls that steer the modulation in real time - steering a living system, not keyframing. **First rung shipped: the [Dial]** (`scripts/dial.gd` engine + `scripts/dial_widget.gd` in the Workspace) - a seeded sculpting instrument: turns inject energy into per-(turn, wedge) modulation deposits that surge, decay, and leave standing patterns; 5-6 wedge signatures per revolution, re-rolled each turn; scenes read the summed bus via `Director.dial_value(slot, i)` (the stage applies it per actor: size / hue / tempo / drive / drift). Still open: more dials (each a unique signature), dial reach into auto-mode scenes and layers, and surfacing the scene-spec's sampled parameters as addressable controls.
- [x] **Colour over scale**: `Lighting` (moving hotspots, beat glow, hue drift) drives reactivity; shapes hold their form. Wired into ring / lattice / rocks / wire_solid so far.
- [ ] More `Swarm` rules: pheromone / ant-colony trails, reaction-diffusion, predator-prey - and abstract (non-city) many-item scenes that reuse the field.
- [x] **Render-kind typing**: every scene declares how it draws (`canvas` / `mesh3d` / `particles` / `swarm` / `scene3d`), making the carried-forward split explicit so it can converge.
- [x] **Forced-perspective 3D path**: a positionable `Lens3D` camera, a `Plane3D` quad primitive, and a `Scene3D` base that depth-sorts bodies + planes under one camera. `wire_solid` migrated, `planes` built native.
- [x] **Procedural geometry**: rocks are generated (coherent fractal `warp` + planar `facet` + stretch), not displayed spheres - the first "geometry from data" demonstrator.
- [x] **Novelty-weighted scheduling**: scenes least-recently shown are favored, so the show spreads across the catalogue instead of repeating.
- [x] **Spectral determinism (phase 1 - exact file)**: the auto session seed is derived from the audio's content fingerprint (`Spectrum._fingerprint` - sampled bytes + length, rename-proof) mixed with a tunable **`SEED_SALT`** constant, not a random roll, so the same song + same salt always yields the same show. `--seed N` overrides. (Logged at startup: `ghost: session seed N (audio fingerprint + salt …)`.)
- [x] **Launch-lottery salt**: `Director.SEED_SALT` (a named constant, starts at the digits of Pi) is mixed into the fingerprint seed, so changing it **re-rolls the entire auto show for a fixed song** without touching the audio. Tune it until the launch video looks best, then ship that value. The exporter reproduces whatever was live (it passes the final `session_seed()` verbatim).
- [ ] **Spectral determinism (phase 2 - perceptual signature)**: a fingerprint robust to re-encodes / lossy copies, so _like-sounding_ audio maps to the same imagery (an exact byte match can't - lossy compression changes the bytes). Sketch: a low-dimensional spectral signature (EMA-accumulated over a sliding window, decaying, amplitude-weighted) reduced to a locality-sensitive hash, so cosine-similar audio collapses to the same seed. **First rung shipped: `Echo` re-localization** - the perceptual descriptor (`HarmonicSignature`) already keeps a manual session's *cursor* honest against the content (loops, doubled tracks, cut-up copies re-converge onto the same scenes for the same sections); deriving the *session seed* itself perceptually is the remaining half.
- [x] **No edge-clipping**: scenes set a large canvas-item custom rect so content (especially big soft glows / lighting drifting off-frame) is never culled prematurely at the viewport edge - it eases off instead of popping out.
- [x] **Authoring feedback console**: `` ` `` writes a reproducible scene record + screenshot to `feedback/`.
- [x] **Manual mode (storyboards)**: a YAML/JSON data spec mapping a song to a user-orchestrated sequence of scenes; a **splash** start screen imports audio and picks Auto/Manual.
- [ ] **Unified renderer, continued**: migrate the remaining 2D scenes onto `Scene3D` (planes for flat content) and route everything through one modulation surface (`Lighting` + materials), so any scene renders under one set of camera/light controls.
- [x] **Richer storyboards (the `stage` data spec)**: a storyboard entry can now DESCRIBE a scene instead of naming one - `cast:` (actors from a registry, every number a sampleable range), `track:` (verbs on a timeline, in nominal seconds that compress with the hold, with musical `on:` cue gates and `sustain:` for motion that never parks), `defs`/`use` fragments, and automatic live-actor continuity across entries. A `tail:` section keeps a finished arc alive - the Director cycles its entries instead of freezing the last frame - and an `elastic` term breathes the timeline clock with the song (zero-mean, endogenous, so the authored length holds). On-disk boards are YAML (commented); JSON is kept as the future editor's format. `default.yaml` covers all fifteen beats of the-point brief this way. Still open: the manual editor itself, and per-entry dials in the Workspace.
- [ ] **Procedural geometry kit**: extend `warp`/`facet` toward other "geometry from data" subjects (terrain, trees, crystals) feeding the `scene3d` world.
- [x] **Partial rocks (wireframe reveal)**: a gaussian alpha mask (`Mesh3D.reveal_texture`) over the solid coat reveals the wireframe lattice beneath, sub-face (per-pixel via a texture with object-space UVs), back-faces culled so no interior shows; masking threshold sampled across a spectrum (sparse patches → half-and-half), drawn by `Mesh3D.draw_revealed`. Still open: **photoreal** stone (richer texture / roughness / height-relief shading) to pair with the reveal.
- [ ] **Sample every tunable, everywhere** (a standing principle, not one task): wherever a computation has a constant that the geometry / animation / shading would still look right with _perturbed_, lift it into a value sampled from an intelligent range (per instance, around a centre), rather than baking one number. `rocks` now samples its material and reveal per rock; extend the same discipline across the catalogue - every constant a candidate for sampled expression ("cattle, not pets" taken to its limit).
- [x] **Layer registry (component integration)**: a `Layer` registry - the visual sibling of `Primitives` - of reusable, composable visual components (`bed` / `fog` / `snow` / `rain` / `fireflies` / `stars` / `aurora` / `petals` / `dust` / `bubbles` / `embers`), added to any scene via `add_layer`/`update_layers`/`draw_layers(z)`. The same component is a scene on its own _and_ an overlay on geometry (snow over `cityscape` / `gaussian_landscape`).
- [x] **Restore the snowflake, as a field**: `snowflakes` - several dozen procedurally-generated six-fold dendrites (`Layer.draw_flake`), varied size, spinning together / independently / on the wind, under the flat-subject discipline. The hard-coded Koch flake is gone.
- [x] **Weather scenes**: `snowfall`, `rainfall`, `fog_bank` (coloured rolling fog), `fireflies`, `starfield`, `aurora`, `petals`, `bubbles`, `motes` - audio drives density / gust / colour, not shape. (Clouds + lightning weather still open.)
- [ ] **Light crossing terrain (real shadow)**: a moving light source sweeping a rolling landscape and casting travelling shadows - true occlusion under `Lens3D`, beyond `Lighting`'s 2D hotspots.
- [x] **Field / texture / terrain foundation**: a composable `Field` primitive (the universal texture-as-modulation), a `Palette`, and a `Terrain` that builds heightfields (hills / mountains / valleys / canyon / islands / mesa) with water, slope shading, and surface texture - drawn through a `Lens3D` (`terrain` scene). Sampled once (terrain is static), cheap thereafter.
- [x] **Blocks breaking from the grid (city on terrain)**: `terrain_city` - a `Swarm` city grows from seeds across a real 3D landscape, blocks standing on the surface and **oriented to the terrain normal** (leaning with the curvature), heights nonlinear in development × spectral band, some plots detaching. Foundation laid; the specs below extend it.
- [ ] **Terrain & city specs (next)** - the foundation is built; these are the directions to grow it:
  - _Texture as modulation, everywhere_ - reuse `Field` beyond terrain: drive a rock's surface, a layer's density, a mesh displacement, lighting. The whole point of the abstraction.
  - _Erosion & rivers_ - carve drainage by flowing the heightfield downhill (a cheap thermal/hydraulic pass on the grid); rivers find the valleys, water collects in basins.
  - _Vegetation & detail_ - scatter trees/rocks on the terrain by slope + height + a `Field` mask (the same placement the city uses), so a landscape is populated, not bare.
  - _Roads & districts_ - the city grows along the terrain's low-curvature contours; roads thread between districts; zoning (height/colour) by a `Field` over the plots.
  - _Real detachment & physics_ - detached districts drift/float with a gentle bob; on a beat, a block can lift off and re-settle; ruins where development has decayed.
  - _Smooth (Gouraud) terrain + better lighting_ - per-vertex colour and a moving sun (the light-crossing-terrain shadow idea) instead of flat facets.
  - _Day/weather over terrain_ - compose the `Layer` registry (snow on peaks, fog in valleys, rain) onto the terrain via `draw_layers`.
- [ ] **The complete package** (the horizon): grow the primitive kit until the catalogue spans the physical sciences - see [Toward a complete package](#toward-a-complete-package-modeling-the-physical-sciences).
- [ ] Grow the kit (flow fields, collisions, soft-body links, emitters) and migrate more bespoke scenes onto particles / `Mesh3D`.
- [x] **Video export**: the Export button relaunches in Movie Maker mode (`--write-movie --use-bake`) to render the visualization + synced audio to a file, reproducing the session via `--seed` and driving reactivity from the offline bake. Runs as its own (background) render so the live session stays interactive.
- [x] **Bake mode**: `SpectrumBake` decodes the song (ffmpeg) and runs an offline FFT into the same 64 log bands, a deterministic per-frame timeline. The export render drives visuals from it (`--use-bake`) instead of the analyzer, so the recorded reactivity is correct under Movie Maker's offline audio _and_ the render runs as fast as the machine allows. Live sessions stay on the real-time analyzer.
- [ ] Stronger beat/onset and tempo tracking; exits that snap to bars, not just beats.
- [ ] **Voice** (speech + subtitles, no generative AI, no recordings): source-filter synthesis - sound-font samples as the glottal source, rule-driven formant filters carrying the phonemes from a human-written script, a sampled population of renditions, sparse landmark labels generalized by harmonic content (the Echo/Dial move), and Fujisaki-style EMA prosody realization. Voice as a Cast actor; subtitles fall out of the Track spans. The full design and its falsifiable rungs: [next/voice.md](../../next/voice.md).
- [ ] Definition / behavior / lifecycle presets per song (config-driven, like the other generators).

## Status

Working framework, live path end to end: import a song on the splash (or boot a CLI flag), watch seeded scenes react and change _with the music_ (Auto), or hand-orchestrate them with a storyboard (Manual). Geometry, motion (behaviors), lifecycle, exit cue, render kind, **and transitions** (typed morphs) are independent typed axes; the physics is a composable **primitive registry** and the appearance is a composable **`Layer` registry** (weather as reusable components - snow/fog/stars/aurora/… - that stand alone as scenes _and_ overlay geometry); there's a real **forced-perspective 3D path** (`Lens3D` / `Scene3D` / `Plane3D`) that the 2D scenes are migrating onto; bodies like `rocks`, the `PrismBody`, and the 3D `EyeBody` are built from sampled/layered primitives ("cattle, not pets"); and **video export** renders the show + audio to a file via an offline bake. An in-app **feedback console** (`` ` ``) captures a scene record + screenshot for authoring.

The throughline is the **scene-spec pipeline**: keep growing complementary, sampleable primitive domains and let them integrate themselves, so scenes occur naturally rather than being hand-built. That pipeline is also what unlocks the **semi-automatic** third mode - the autopilot seed plus live dials that steer the modulation. Standalone; it does not import or depend on Praxis.
