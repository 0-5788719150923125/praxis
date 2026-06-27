# vortex

A spectral music visualizer built with [Godot](https://godotengine.org/) 4.6. Point it at a `.wav`, and it draws geometry in response to the sound - rings, planes, harmonics, lattices, whatever - then loops through scenes like a video you can record full-screen.

It is the same move as the Arena and business-card generators elsewhere in this repo, pushed at audio: a **scene definition** (a typed bag of parameters) is passed through **typed transformations** that are modulated by audio features - amplitude, per-band energy, beat, velocity. Procedural, deterministic, and cheap. No generative AI in the render path.

## Why

Generating visuals for a song with an image/video model is slow and expensive. A procedural visualizer is free, runs in real time, and is infinitely customizable - and because every scene is seeded, the same song always produces the same video.

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
- **Real 3D where it matters.** `Mesh3D` is a software 3D primitive (icosphere / cube / octahedron / tetrahedron → depth-sorted flat-shaded faces, perspective). Rocks and `wire_solid` are genuine 3D bodies, not sheared flat polygons. `Geo` holds the shared polygon helpers (convex split, fracture) that shatter the glass.
- **Many items, local rules.** `Swarm` is a scalar field over a grid that evolves by *local* interaction - development creeping out from seeds (`GROW`), or injected pulses diffusing across the lattice (`WAVE`). It drives thousands of items without scripting each one (the `metropolis` city grows and pulses from one), and the same mechanism transfers to any abstract many-item grid - the cellular / ant-colony idea.
- **Framing is a typed axis.** A scene declares a `framing` class and the **`Shots`** registry assigns a camera move from the matching pool: expressive for `subject` (offset / push / pan / canted), gentle for `field` fillers, square-on for a single `plane` (so a flat snowflake or pane never reads as a tumbling card). And planes can spawn in multiples - a few small ones don't look stranded the way one lone spinning plane does.
- **Lifecycle and exit cues are typed too.** A scene either **loops** until cut, or is a **oneshot** that plays one sequence and reports `finished()` (shatter glass settling, then ending). Once a scene is *eligible* to exit - a loop past its minimum hold, or a oneshot that finished - the **`Director`** waits for a chosen spectral **trigger** before actually cutting, so exits land on the music: usually a **beat** (rising edge), sometimes a **movement** (section change) or a **lull** (drop into quiet), with a maximum-hold backstop. Triggers are picked weighted per scene.
- **`Director`** (autoload) holds the registry of `{scene, behavior}` pairs, runs the lifecycle/trigger logic above, and performs the change - mostly clean **jump cuts**, occasionally a **blend** (fade / zoom-through / additive bleed). `--scene <name|N>` pins one scene for authoring.
- **`main`** wires it together: loads the audio, toggles full-screen, forces the next scene, quits.

## Two ways to run

1. **Live (default).** Audio plays through the analyzer bus; scenes react in real time. Use this to author and preview, and screen-record it with OBS for a quick capture.
2. **Baked (planned, for clean output).** Pre-analyze the `.wav` into a spectrum timeline once, then drive scenes from the baked frames and render with Godot's **Movie Maker mode** (`--write-movie out.avi --fixed-fps 60`). Movie Maker is frame-perfect and writes synced audio, but its capture driver makes the live analyzer unreliable - baking decouples analysis from playback so the preview and the recorded video are identical. See *Roadmap*.

## Layout

- `project.godot` - Godot 4.6 project; autoloads `Spectrum` and `Director`; `main.tscn` is the entry scene.
- `scenes/main.tscn` / `scripts/main.gd` - root node; loads audio, full-screen (`F11`), next scene (`Space`), quit (`Esc`).
- `scripts/spectrum.gd` - `Spectrum` autoload: analyzer setup, per-frame `AudioFeatures`, spectral-flux/movement detection.
- `scripts/audio_features.gd` - `AudioFeatures`, the typed per-frame struct scenes consume.
- `scripts/vortex_scene.gd` - `VortexScene` base class: definition + view + behaviors + the `tick`/`drift_view`/`wobble` motion helpers.
- `scripts/mod_bank.gd` - `ModBank`, the seeded slow-oscillator pool behind organic motion.
- `scripts/activation.gd` - `Activation`, per-element sparse response gating with EMA decay.
- `scripts/scene_view.gd` - `SceneView`, the EMA-smoothed per-scene camera (zoom / tilt / rotate / off-center).
- `scripts/shots.gd` - `Shots`, the camera-framing registry (centered / offset / push / pull / pan / canted) with subject/field/plane pools.
- `scripts/particle.gd` / `scripts/primitives.gd` / `scripts/particle_system.gd` - the physics substrate: a `Particle`, the `Primitives` force **registry**, and the `ParticleSystem` that composes and steps them.
- `scripts/mesh3d.gd` / `scripts/geometry.gd` - the `Mesh3D` software-3D primitive and `Geo` polygon helpers (split, fracture).
- `scripts/swarm.gd` - `Swarm`, the grid field that spreads by local rules (growth fronts, pulse waves) for many-item scenes.
- `scripts/director.gd` - `Director` autoload: `{scene, behavior}` registry, lifecycle + spectral-trigger timing, cut/blend transitions, `--scene` pin.
- `scripts/scenes/` - the visualizers (see below). Drop new ones here and register them in `Director.SCENES`.
- `audio/` - put `song.wav` here (git-ignored). Or pass `--audio /path/to/song.wav` on launch.

## Scenes

Each is a small combination of shapes; behavior decides how it moves.

- **`spectrum_ring`** - the spectrum bent into a circle; bars push outward per band. `static` (rigid wheel) and `fluid` (bars ripple as detached strokes).
- **`harmonic_lattice`** - a grid of cells breathing with a traveling spectral wave. `static` and `drift`.
- **`rooted_growth`** - trees/roots that surge outward into branching complexity on each beat, then decay back to a bare stub; the live edge buds and the structure twists as it grows. Pulses with decay of growth.
- **`fog_lights`** - glowing orbs tied to slices of the spectrum, diffused and occluded under big drifting blobs of fog.
- **`strata`** - translucent waveform planes stacked into tilted depth, near planes scrolling faster than far (parallax).
- **`koch_snowflake`** - fractal snowflakes whose recursion depth crystallizes finer with energy; usually one held square-on, sometimes several scattered, swaying about a rest angle.
- **`wire_solid`** - a true 3D cube / octahedron / tetrahedron / icosahedron (`Mesh3D`): perspective, depth-sorted, translucent faces with bright edges. Rotates slowly and continuously, because a real solid reveals its volume by turning (no more flat-wireframe illusion).
- **`voxel_blocks`** - an isometric Minecraft heightfield; columns of cubes rise and fall with the spectrum.
- **`cityscape`** - a layered skyline of rectangles; building heights track the bands and windows flicker on the beat.
- **`shatter_glass`** - a pane fractured into irregular angular shards (recursive splitting, not pizza slices) that burst and spin on the beat. Usually one pane held square-on; sometimes a few. Seeds `loop` (burst → recombine) or `oneshot` (burst → settle → end).
- **`gaussian_landscape`** - terrain built from Gaussian bumps (heights driven by the bands), drawn as receding ridgelines with translucent fog pooling in the valleys and flowing sideways in the wind.
- **`rocks`** - faceted stones in **real 3D** (`Mesh3D`), tumbling gently (rooted ones barely turn, via `Activation`). Style by seed: `plain` / `rough` / `crystal`. Mode by seed: `pulse` / `explode` / `crumble` (oneshot).
- **`embers`** - a warm cloud of sparks that flare on the beat and ride the wind. Almost no code of its own: `scatter` + `wind` + `spring` + `drag` from the registry - the cross-contamination demo.
- **`metropolis`** - a large isometric countryside of thousands of blocks running off every edge, driven by a `Swarm` field. By seed: `growth` (a city creeps out from a few seeds across the hills) or `pulse` (a built city with beat-injected colour fronts rippling through it), under flowing fog.

## Running it

Open `project.godot` in Godot 4.6 and press play, or from the command line:

```
godot --path axis/vortex                       # default: audio/song.wav if present
godot --path axis/vortex -- --audio ~/track.wav
```

Controls: `Space` next scene · `F11` full-screen · `Esc` quit.

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

## Roadmap

- [x] Live analyzer → `AudioFeatures` → scene framework; fourteen scenes.
- [x] Behaviors (`static`/`drift`/`fluid`) - motion typed separately from geometry; originals kept as `static`.
- [x] Lifecycle (`loop`/`oneshot`) + spectral exit triggers (`beat`/`movement`/`lull`); exits land on the music; jump cuts with occasional blends.
- [x] **Primitive registry**: `Particle` / `ParticleSystem` + a `Primitives` force registry. Glass/rocks/embers composed from it.
- [x] **Gentle by construction**: bounded sway about rest poses (not unbounded spin), `Activation` sparse response with EMA decay, EMA-smoothed camera, typed framing pools (subject / field / plane), multi-instance planes.
- [x] **Real 3D**: `Mesh3D` software primitive; rocks and `wire_solid` are genuine 3D bodies. `Geo` fracture replaced the pizza-sliced glass.
- [x] **Swarm fields**: many-item scenes driven by local rules (`metropolis` city growth + colour pulses); transferable to any grid.
- [ ] **Filaments** scene: a procedural growth primitive feeding lightning / tesla / snake / neural / roots / branches variants (replaces the removed `orbits`).
- [ ] More `Swarm` rules: pheromone / ant-colony trails, reaction-diffusion, predator-prey - and abstract (non-city) many-item scenes that reuse the field.
- [ ] Grow the kit (flow fields, collisions, soft-body links, emitters) and migrate more bespoke scenes onto particles / `Mesh3D`.
- [ ] **Bake mode**: offline FFT → spectrum-timeline resource + Movie Maker export, so recorded output is deterministic and frame-perfect.
- [ ] Stronger beat/onset and tempo tracking; exits that snap to bars, not just beats.
- [ ] Definition / behavior / lifecycle presets per song (config-driven, like the other generators).

## Status

Working framework, live path end to end: load a `.wav`, watch seeded scenes react and change *with the music*. Geometry, motion (behaviors), lifecycle, and exit cue are independent typed axes, and the physics is now a composable **primitive registry** rather than per-scene code - scenes are increasingly a parts list, not a pet. Bake/Movie-Maker export and a larger primitive kit are next. Standalone; it does not import or depend on Praxis.
