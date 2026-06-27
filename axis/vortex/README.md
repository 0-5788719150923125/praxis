# vortex

A spectral music visualizer built with [Godot](https://godotengine.org/) 4.6. Point it at a `.wav`, and it draws geometry in response to the sound - rings, planes, harmonics, lattices, whatever - then loops through scenes like a video you can record full-screen.

It is the same move as the Arena and business-card generators elsewhere in this repo, pushed at audio: a **scene definition** (a typed bag of parameters) is passed through **typed transformations** that are modulated by audio features - amplitude, per-band energy, beat, velocity. Procedural, deterministic, and cheap. No generative AI in the render path.

## Why

Generating visuals for a song with an image/video model is slow and expensive. A procedural visualizer is free, runs in real time, and is infinitely customizable - and because every scene is seeded, the same song always produces the same video.

## The shape

```
  .wav ──▶ Spectrum ──▶ AudioFeatures ──▶ VortexScene ──▶ screen / video
           (analyzer)    (typed, per-frame)  (definition + transforms)
                                              ▲
                                     Director picks / loops scenes
```

- **`Spectrum`** (autoload) owns the `AudioStreamPlayer` and the `AudioEffectSpectrumAnalyzer` bus effect. Every frame it emits one **`AudioFeatures`**: overall `energy`, named bands (`bass`, `low_mid`, `mid`, `high`, `treble`), a smoothed `beat` pulse, a full `bands` array, and `time`. This is the single typed interface every scene reads - scenes never touch the audio engine directly.
- **`VortexScene`** (`class_name`, base script) is one visualizer. It does two things: `build_params(rng)` rolls a **definition** from a seeded RNG (the typed parameter bag), and `update(features, delta)` modulates that definition by the audio and redraws. Add a scene by subclassing this; that is the whole extension point.
- **`Director`** (autoload) holds the registry of scenes, picks them randomly or sequentially, seeds each one from `song_hash + index`, and swaps/loops them on a timer. Determinism comes from the seed, exactly like Arena.
- **`main`** wires it together: loads the audio, toggles full-screen, advances scenes, quits.

## Two ways to run

1. **Live (default).** Audio plays through the analyzer bus; scenes react in real time. Use this to author and preview, and screen-record it with OBS for a quick capture.
2. **Baked (planned, for clean output).** Pre-analyze the `.wav` into a spectrum timeline once, then drive scenes from the baked frames and render with Godot's **Movie Maker mode** (`--write-movie out.avi --fixed-fps 60`). Movie Maker is frame-perfect and writes synced audio, but its capture driver makes the live analyzer unreliable - baking decouples analysis from playback so the preview and the recorded video are identical. See *Roadmap*.

## Layout

- `project.godot` - Godot 4.6 project; autoloads `Spectrum` and `Director`; `main.tscn` is the entry scene.
- `scenes/main.tscn` / `scripts/main.gd` - root node; loads audio, full-screen (`F11`), next scene (`Space`), quit (`Esc`).
- `scripts/spectrum.gd` - `Spectrum` autoload: analyzer setup + per-frame `AudioFeatures`.
- `scripts/audio_features.gd` - `AudioFeatures`, the typed per-frame struct scenes consume.
- `scripts/vortex_scene.gd` - `VortexScene` base class: `build_params(rng)` + `update(features, delta)`.
- `scripts/director.gd` - `Director` autoload: scene registry, selection, looping.
- `scripts/scenes/` - the visualizers. Ships with `spectrum_ring.gd` (radial bars) and `harmonic_lattice.gd` (a band-driven grid). Drop new ones here and register them in `Director.SCENES`.
- `audio/` - put `song.wav` here (git-ignored). Or pass `--audio /path/to/song.wav` on launch.

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
class_name MyScene

func build_params(rng: RandomNumberGenerator) -> Dictionary:
    return { "count": rng.randi_range(6, 24), "hue": rng.randf() }

func update(f: AudioFeatures, delta: float) -> void:
    _energy = f.energy        # stash what you need
    queue_redraw()

func _draw() -> void:
    # draw using self.params + the stashed features
    pass
```

Then add `preload("res://scripts/scenes/my_scene.gd")` to `Director.SCENES`. That is the entire contract: a seeded definition in, audio-modulated geometry out.

## Roadmap

- [x] Live analyzer → `AudioFeatures` → scene framework, two scenes, scene looping.
- [ ] **Bake mode**: offline FFT of the `.wav` → a spectrum-timeline resource; a baked `Spectrum` backend; Movie Maker export so recorded output is deterministic and frame-perfect.
- [ ] Beat/onset detection beyond the smoothed energy pulse.
- [ ] More scenes: 3D geometry, planes, particle fields, neural-net graphs.
- [ ] Scene transitions (crossfade) instead of hard cuts.
- [ ] Definition presets / per-scene parameter overrides (config-driven, like the other generators).

## Status

Early scaffold - the live path works end to end: load a `.wav`, watch two seeded scenes react and loop. Bake/Movie-Maker export and the richer scene library are next. Standalone; it does not import or depend on Praxis.
