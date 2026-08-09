# VOICE_PLAN: a generative path alongside the synthesizer

Draft, 2026-08-09. Nothing here is built. This is the design for adding a neural voice path to
ghost **beside** the existing formant synthesizer, not inside it.

Context for why: `next/voice_intelligibility.md` records a full measurement program against the
procedural engine. Its conclusion is that intelligibility in this class is reachable but naturalness
is not - DECtalk, the commercial gold standard of Klatt-lineage synthesis, scored 97% on the
Modified Rhyme Test against 97-99% for natural speech, and a listening comparison found it close to
what ghost already produces, including the same artifact class. The remaining gap to *natural* is a
property of the method, not of effort.

---

## 1. The shape of the decision

The constraint in `next/voice.md` is explicit: "no generative AI (nothing pretrained, nothing
trained on other people's voices, everything procedural and inspectable)". Adding a neural backend
retires that constraint for one path. It is worth stating plainly rather than letting it happen by
increments:

- The procedural engine stays, in full, as its own path. Nothing is deleted.
- The neural path is a second, separate mode with its own interaction model.
- The two are not assumed compatible. Where they can share, they share; where they cannot, they
  diverge cleanly rather than being forced together.

Two properties of the existing work survive the switch and are worth knowing before committing:
the phoneme table and the `[K AE T]` inline-phonetics override are good work that any backend
benefits from, and `measure_voice.py` measures a WAV without caring what produced it.

---

## 2. Architecture: a voice host, not a GDExtension

Godot has no first-class ONNX support. The obvious route - a GDExtension wrapping ONNX Runtime's C
API - is the expensive one: per-platform native binaries, a build story ghost does not currently
have (there is no `export_presets.cfg` at all), and a rebuild every time you want to try a
different model.

**Use a subprocess host instead.** ghost already does exactly this pattern: `mask_editor.gd`
bootstraps a private Python virtualenv at `user://ytdlp_venv` for the YouTube import, upgrades it,
and retries on failure. The machinery, the precedent and the failure handling all exist.

```
Godot (GDScript)                          voice host (Python)
────────────────                          ───────────────────
VoiceBackend (interface)   <── stdio ──>  host.py
  ├ ProceduralBackend                       ├ backends/piper.py
  └ NeuralBackend ─────────────────────>    ├ backends/kokoro.py
                                            └ backends/<yours>.py
```

- **Transport**: newline-delimited JSON on stdin/stdout for control, raw PCM on a separate pipe or
  a temp file for audio. Requests carry text plus a voice id plus parameters; responses carry PCM
  plus a word/phone timing map.
- **Why this and not a GDExtension**: swapping or adding a model becomes a Python file and a config
  entry, not a native rebuild. That is precisely the swappability requirement. The transport can be
  replaced by a GDExtension later without changing the interface, if inference latency ever
  justifies it.
- **Lifecycle**: the host starts with the mode, stays warm (model load is the slow part, not
  inference), and dies with it. A crashed host is a recoverable state, not a crashed game.
- **Cost**: one venv, bootstrapped on first use, cached under `user://`, never in git.

### The backend interface

Deliberately narrow, and deliberately the same shape the procedural engine already satisfies:

```
list_voices()                    -> [{id, name, model, sample_rate, license, notes}]
synthesize(text, voice, params)  -> {pcm, sample_rate, words:[{text,t0,t1}], phones:[...]}
capabilities()                   -> {streaming, phoneme_input, duration_control,
                                     pitch_control, reference_audio, singing}
```

`capabilities()` is the part that earns its keep. Backends differ enormously in what they expose,
and the UI should be built from this rather than from assumptions - a backend that cannot do
duration control simply does not show that slider.

---

## 3. Model selection

**Baseline: Piper.** VITS-based, MIT licensed, ONNX-native, roughly 20-60 MB per voice, and faster
than real time on a Raspberry Pi. It is the smallest thing that clears the "natural and
understandable without subtitles" bar, which is the entire point of this exercise.

One thing makes Piper unusually good for ghost specifically: it normally depends on eSpeak-NG for
phonemization, which is GPLv3 and would be viral for a shipped game. **ghost already has its own
G2P** - `phonemes.gd` plus a CMUdict-derived lexicon. The remaining work is mapping ARPAbet to the
IPA token set Piper's models expect, which is a real but bounded job, and it removes the licensing
problem that usually kills Piper for commercial use.

**Second backend to prove swappability: Kokoro.** 82M parameters, Apache 2.0, CPU-viable, better
quality than Piper. Roughly 320 MB at fp32 or ~80 MB int8. Adding it should require touching no
Godot code at all - that is the test of whether the interface is right.

**Deliberately not chosen**: XTTS and its descendants (licensing is non-commercial or unclear),
diffusion-based systems (too slow on CPU for a live path), and anything requiring a GPU.

> **Verify before committing.** These recommendations reflect knowledge with a mid-2026 cutoff and
> this field moves fast. The first task in Phase 1 is a current survey: check what is actually
> maintained, what the license on each *voice checkpoint* is (this differs from the code license
> and is where the real risk sits), and whether something better-suited has appeared.

### Weights are not code

- Never committed. Downloaded once into `user://voices/`, checksummed, gitignored by construction
  since `user://` is outside the repo.
- Per-voice license recorded in the manifest and shown in the UI. Piper voices derive from
  different datasets with different terms; some are CC-BY and require attribution, some are more
  restrictive. This must be tracked per voice, not per project.
- A first-run download flow that can fail gracefully and offline.

---

## 4. What transfers, what does not

From the architectural analysis in `next/voice_intelligibility.md`:

| Layer | Transfers? |
|---|---|
| **Post-buffer effects** - presence gain, distance lowpass, location bandpass and carrier, tanh ceiling, echo bus, broadcast chain (`voice_stream.gd:320-385`, `bake_location`) | **Yes, unchanged.** These operate on PCM. ~80 lines. |
| **Delivery** - worker thread, generator push, underrun telemetry, take writing | **Yes.** The threading model is backend-agnostic. |
| **Text front end** - `Phonemes`, `Phrasing`, `[K AE T]` overrides, text normalization | **Yes**, and it is needed by both paths. |
| **Timing map and subtitles** | **Yes**, if the backend returns durations. Piper's duration predictor can; otherwise force-align. |
| **Exporter, Chrome, Director, scenes, harmonic seeding** | **Yes.** They consume a WAV. |
| **`measure_voice.py`** | **Yes.** It measures any WAV. |
| **Per-sample synthesis DSP** - cascade, glottal source, frication, parallel branch | **No.** This *is* the synthesizer. |
| **The genome** - `ProsodyWalk`, `ProsodyField`, 16 PRIOR scalars, modulators, anchors, the toll | **No, not directly.** A neural model owns its own durations and f0. See below. |

### The genome, and why the game does not port

The fishing game's economy is defined over the procedural engine's parameter space: difficulty is
nearest-neighbour distance in trait-plus-genome space, the fit wheel is cosine similarity over an
18-dimensional vector, the toll anneals a genome toward the belt's forces. Piper exposes a speaker
id and three global scalars. You cannot host a 25-dimensional seed economy on that.

This is the concrete reason for a separate mode rather than a backend swap, and it is why the mode
choice happens at the *home screen* rather than inside the game.

That said, the *subject* of the game survives. What the fishing loop is really about is finding a
voice identity you like and keeping it. In a neural path that identity is a speaker embedding
rather than a genome - still a vector, still interpolable, still something you can throw, catch and
breed. Whether that makes a good game is an open question and should be answered by prototyping,
not by assuming the metaphor ports.

---

## 5. Prosody and singing on generated audio

You asked whether modulations can still be applied. Three tiers, in increasing order of ambition:

**Tier 1 - buffer effects (free).** Everything in the table above. Presence, distance, echo,
location colour, the broadcast chain. These work on day one.

**Tier 2 - post-hoc prosody (classic DSP, no ML).** Generate neutral speech, then impose the seed's
f0 contour and timing on it with PSOLA or a phase vocoder. This is 1980s technology, fully
procedural and inspectable, and it means `ProsodyWalk` can survive as a **post-process** rather than
a planner. The genome would then modulate any backend's output, which is a genuinely appealing
outcome: the part of ghost that encodes *performance* stays yours, and only the *timbre* becomes
learned.

Cheaper partial version: use whatever the backend natively exposes. Piper has `length_scale`,
`noise_scale` and `noise_w` - coarse, but real, and free.

**Tier 3 - singing.** TTS models do not sing, and asking them to is the wrong shape. Two better
routes:

- **Pitch-impose on generated speech** (an extension of Tier 2): quantize the f0 contour to a
  melody, add vibrato and sustain. Ghost already has all of this logic - `SUSTAIN_BAR`, `VIB_RATE`,
  the anchor shelf, the song trait. It would move from driving the synthesizer to driving a
  pitch-shifter.
- **Voice conversion over ghost's own output** - the interesting one. Ghost's formant synth *can*
  sing; what it lacks is human timbre. An RVC-style conversion model takes the synth's singing and
  re-timbres it. That inverts the usual arrangement: the procedural engine stays the performance
  instrument, and the neural model supplies only the voice quality. It preserves everything the
  fishing game does well, and it is the one configuration where both systems are doing what they
  are each best at.

Tier 3 is exploratory and should not gate anything else.

---

## 6. The mode split

Home screen "Synthesis" becomes a choice of two paths.

```
                    ┌─ Fishing (procedural)  -> synth_editor.gd, genome, belt, toll
Synthesis ──────────┤
                    └─ Generative (neural)   -> new editor, voices, embeddings
```

Both sit on a shared session layer so everything downstream is untouched:

```
VoiceSession (shared)
  text -> normalize -> [backend] -> PCM + timings
                                     ├ effects chain (presence/location/echo/broadcast)
                                     ├ VoiceStream delivery + subtitles sidecar
                                     └ Exporter + Director + scenes
```

Per `axis/ghost/CLAUDE.md`, shared furniture belongs in `chrome.gd`, never in a mode's branch of
`main.gd` - that rule exists because synthesis mode shipped twice without a feedback console. The
new mode gets export, assistant and feedback with zero wiring if it is built on `Chrome`.

**What the generative editor needs** (minimal first version): text box, voice picker with license
shown, whatever `capabilities()` reports as adjustable, generate, play, keep. No fishing metaphor
until prototyping says one fits.

---

## 7. Sequencing

Sized relatively. Items marked **(needed either way)** are worth doing even if the neural path is
abandoned.

```
P0  Text normalization + unicode front end        (needed either way) - medium
      numerals, ordinals, currency, abbreviations, curly quotes, em dashes,
      hyphenation. Today numerals are DELETED silently and curly quotes break
      dictionary lookup: 10.4% of tokens damaged on manuscript text. This is a
      hard blocker for 35 chapters regardless of synthesizer.

P1  Model survey + license audit                                      - small
      Verify the recommendations in section 3 against the current landscape.
      Audit per-voice checkpoint licenses. Decide ship-vs-download.

P2  Voice host + Piper backend, offline render only                   - medium
      Python venv bootstrap (mirror the ytdlp_venv pattern), stdio protocol,
      one backend, WAV out. No Godot UI yet. Verified by rendering a chapter
      and running measure_voice.py --wer over it.

P3  ARPAbet -> IPA mapping so ghost's own G2P feeds Piper             - small
      Removes the eSpeak-NG/GPL dependency and makes [K AE T] work on the
      neural path too. Do it here, not later; it changes the interface.

P4  Mode split + minimal generative editor                            - medium
      Home screen branch, VoiceSession abstraction, Chrome-based furniture.

P5  Effects chain + subtitles + export on the neural path             - small
      Mostly wiring; the code already exists and is backend-agnostic.

P6  Second backend (Kokoro)                                           - small
      The swappability test. Should touch no GDScript.

P7  Tier 2 prosody post-process (PSOLA / phase vocoder)               - medium
      Optional. Makes the genome modulate any backend.

P8  Singing exploration (pitch-impose, or RVC over the synth)         - large
      Exploratory. Gates nothing.
```

A useful checkpoint: **after P2 you can render a full chapter and judge it**, with no Godot work
done at all. That is the cheapest possible answer to "is this actually better," and it comes before
any UI investment.

---

## 8. Risks and open questions

- **Per-voice licensing** is the sharpest practical risk. Code licenses (MIT, Apache) are not the
  weights' licenses. This needs auditing before any voice is shipped or even chosen.
- **Web export** is likely infeasible for the neural path. Godot Web plus GDExtension is fragile,
  and a subprocess host does not exist in a browser at all. Note that ghost's *current* voice
  already cannot ship to web without cross-origin isolation for its worker thread, and that no
  export preset exists for any platform yet - so this is a pre-existing unknown, not a new one.
- **Determinism.** The fishing game guarantees a lineage reproduces forever. Neural inference is
  deterministic given a fixed seed on fixed hardware, but not necessarily across machines or
  runtime versions. If the generative mode ever gets a persistence model, decide early whether it
  promises reproducibility or just caches its outputs.
- **Latency and the live path.** Piper is faster than real time on CPU, so streaming at phrase
  granularity is feasible. But it is utterance-latency, not segment-latency: `retune()`, which
  bends timbre mid-sentence and which the reel mechanic depends on, has no neural equivalent.
- **Does the game port at all?** Open, and deliberately unanswered here. Prototype before deciding.
- **Disk and first-run experience.** A voice download is a first-run flow that can fail. It needs to
  fail gracefully and offline, the way the dataset fallback elsewhere in this repo does.

---

## 9. What this does not change

The procedural engine stays. It is the only path that can sing today, it is fully inspectable, it
has no dependencies and no weights, and after the work recorded in `next/voice_intelligibility.md`
its consonant inventory is measurably in-band for the first time. If the neural path turns out to
be a worse fit for the game, or the licensing sours, or a model disappears, ghost still has a voice
it owns outright.

That is the actual argument for building this as a second path rather than a replacement.
