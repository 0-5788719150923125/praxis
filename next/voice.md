# Voice: speech for ghost, from robotic sources

**Goal.** Ghost videos with spoken narration and subtitles over its 3D worlds - under two hard
constraints: **no generative AI** (nothing pretrained, nothing trained on other people's
voices, everything procedural and inspectable) and **no vocal recording sessions**. The human
writes the *script*; the machine has to find the *voice*. The bet: a realistic-enough human
voice can be built from very robotic, old-school audio sources plus careful modulation, sparse
human labels, tiny in-engine ML, and EMA convergence - the same machinery ghost already runs
on.

**What it is (the precise name).** Source-filter synthesis (the Klatt / eSpeak lineage) with a
sampled population of renditions, content-addressed sparse supervision, and a
Fujisaki-style prosody realization. Every part has a 1980s-DSP name; none of it is a black
box.

## The three-layer decomposition (this is what makes it possible)

Speech factors cleanly, and each factor has a different owner:

1. **Source** - the excitation: a periodic buzz plus noise. **This is where the 500
   sound-font samples live.** Any pitched, harmonically-rich sample (a cello note, a synth
   pad, a buzzer) can serve as the glottal source once it is pitch-tracked and looped. The
   source carries *timbre and identity* - a voice built on a bowed-string source will sound
   different from one built on a square wave, and that difference is a sampled axis
   ("cattle, not pets" over voices).
2. **Filter** - formant resonances (a handful of bandpass filters whose center frequencies
   move through time). **This is where intelligibility lives, and ONLY here.** A vowel is a
   formant configuration; a consonant is a transition plus noise shaping. The samples cannot
   carry this - no amount of stretching a beep produces an /a/ - so the filter layer is
   rule-driven from the script: per-phoneme formant targets from a small built-in table,
   interpolated through time. This is exactly what a talkbox or vocoder does: source from an
   instrument, phonemes from the filter.
3. **Prosody** - f0 contour, per-phoneme durations, pauses, energy envelope. **This is where
   "human" lives** - cadence and inflection, the difference between robotic and alive. It is
   low-dimensional (a few scalars per syllable), which is why tiny models are enough.

Robotic-but-intelligible is layers 1+2 alone and is the honest baseline. Everything
interesting happens in layer 3.

## The pipeline

1. **The human writes the script.** This kills the alignment problem by construction: we
   *synthesize* the audio, so every phoneme's position is known exactly - no ASR, no forced
   alignment, ever. It also softens grapheme-to-phoneme: the author can co-author
   pronunciation (a phonetic spelling convention in the script, or a small exceptions
   dictionary), so no G2P dependency is needed to start.
2. **Population render.** The script goes through the synth **500 times**, every tunable
   sampled from ranges: source sample, source pitch and loop character, formant scale (vocal
   tract length), f0 base and range, speaking rate, pause distribution, jitter/shimmer
   (micro-instability that reads as organic). The result is a **grounded distribution of
   candidate voices** - the eye move applied to a larynx. Same discipline as everywhere in
   ghost: seeded, deterministic, reproducible per (script, seed).
3. **Sparse landmark labels.** The human listens to the opening of a rendition and plants a
   few **landmarks** - typed pressure points where a careful modulation makes it sound
   *natural*: "pitch resets here", "lean on this syllable", "let this pause breathe",
   "de-emphasize this run". Mechanically these are **Dial deposits**: a typed channel
   (pitch / duration / energy / pause), a signed amount, a decay - a surge that settles into
   a standing pattern. The labeling gesture already exists in the Workspace; the feedback
   console already captures typed, reproducible records.
4. **Generalization by content, not position.** A landmark is keyed to the **harmonic /
   sequencing context where it was planted** (the `HarmonicSignature` move - the Echo
   discipline: corrections belong to what the audio *is*, never to where the playhead was).
   A small **nonlinearity** learns the activation of each pressure from context features
   (phrase position, stress pattern, phoneme class, local signature), so when the script's
   structures recur - and prose structure is periodic: phrase onsets, clause endings, stress
   feet - the pressures re-fire *there*, unlabeled. Labels at the start of the sequence
   propagate to the whole sequence exactly as far as prosodic structure actually repeats.
5. **EMA realization.** Targets jump discretely (per syllable); the realized contour
   converges smoothly. This is not just aesthetically consistent with ghost's
   convergence-over-lockstep commitment - it is *literally the accepted model of human f0*
   (Fujisaki, 1984: phrase and accent commands as impulses/steps smoothed through decaying
   second-order filters; declination is the slow EMA, accents are fast ones riding it). The
   EMAs are not a trick to make it sound acceptable; they are the physics of the thing being
   modeled. Same for the energy envelope and for formant transitions (coarticulation is
   target-undershoot under smoothing).

## Why the existing machinery carries almost all of this

- **Storyboard / Cast / Track**: a voice is an **actor**; an utterance is a **track span**
  with `on:` cues; **subtitles fall out for free** because the text spans and their timing
  are the same data. Timing against music uses the cue gates that already exist.
- **Dial**: the deposit mechanics (surge -> decay -> standing pattern, additive, seeded) are
  the labeling primitive, unchanged.
- **Echo / HarmonicSignature**: content-addressing for label generalization, unchanged in
  spirit.
- **Nonlinear**: the response-curve library is the activation shape of the pressure model.
- **Bake / export**: utterances render offline like everything else; determinism means the
  exported voice is byte-stable per (script, seed, labels).
- **Tiny ML in Godot**: inference is per-syllable (control rate), not audio rate - a few
  hundred to a few thousand parameters, evaluable in GDScript in microseconds. The audio-rate
  work is DSP (filters, overlap-add), done once per utterance into a cached WAV, not
  per-frame. Training on a handful of labels is small enough for in-engine SGD.

## Honestly graded

- **The cliff is intelligibility, not naturalness.** If layer 2 can't make a sound-font
  source say a recognizable word, prosody polish is irrelevant. So the rungs are ordered to
  hit the cliff first (below): one vowel, then one word, then one sentence. If formant
  filtering over exotic sources won't read, the fallback is a plain pulse-train source
  (pure Klatt) - still zero-AI, zero-recording, and the population/label/EMA machinery is
  untouched.
- **Sparse labels generalizing is the bet, not a given.** It is plausible for prosody
  because prosodic structure recurs; it is unproven for segment-level fixes. If landmarks
  from the opening don't transfer, the scheme degrades gracefully into a *performance
  instrument* - hand-planted deposits along the whole timeline, manual mode for the voice -
  which is already useful and is the same fallback the Dial made respectable.
- **The ceiling is stylized, not photoreal.** 500 modulated robotic renditions plus good
  prosody will not fool anyone into hearing a recording, and should not try - the uncanny
  valley is steepest from the photoreal side. The target is the ghost aesthetic: **alive,
  intentional, expressive**, unmistakably synthetic the way the eye is unmistakably
  procedural. "Truly natural" means natural *cadence*, not a cloned human.
- **No per-experiment tuning.** Constants fixed and model-agnostic; the only signals are
  endogenous (script structure, harmonic context) and the human's labels. The 500 is a
  population size, not a hyperparameter to sweep.
- **Dependency posture.** Nothing new to start: sources from samples we already have rights
  to, filters and pitch-tracking in-house (the FFT/bake machinery already exists), phoneme
  table in-house, pronunciation co-authored in the script. eSpeak-NG stays a *reference to
  read*, not a dependency to link.

## Rungs (each falsifiable before the next)

- [x] **Rung 0 - the vowel.** SHIPPED (pulse-train source first; sampled sound-font sources
  remain the open half): `Voice` synthesizes vowels through a three-formant cascade, and
  `tests/voice_check.gd` verifies them *objectively* - Goertzel probes show /AA/ dominating
  its F1 region by ~200x and /IY/ its F2 region by ~4x. Judged-by-ear still pending (the
  human half). Exotic sources move to a later rung: the source is behind one function.
- [x] **Rung 1 - the word.** SHIPPED: `Phonemes` (table + digraph/magic-e rules + exceptions
  + `[K AE T]` inline phonetics), full sentences render with monotonic word/phoneme timing
  maps, stops/fricatives/aspiration included. Robotic on purpose; ears will find the gaps.
- [~] **Rung 2 - the population.** Half shipped, now trait-shaped: **the speaker is a
  trait vector** (`Voice.TRAIT_KEYS`: pitch / lilt / tract / pace / breath / grit / drawl,
  each in [-1,1]) - the ZERO vector is the hand-curated default speaker (its centres live
  in `Spec.from_traits`, tune them there), a seed only initializes the sliders, and the
  vector itself is the replicable identity (autosaved; asserted byte-identical in
  `tests/voice_check.gd`). The N-render batch + side-by-side audition is still open.
- [x] **Real-time streaming (was: eliminate the pre-render wait).** Synthesis measured
  ~30x real time in GDScript, so `VoiceStream` runs a sliding window: a ~0.35s prebuffer
  is synthesized in the Speak click's own frame, pushed into an `AudioStreamGenerator` on
  Spectrum's analyzed bus (`Spectrum.begin_stream`), and a budgeted per-frame pump keeps
  ~1.2s of lead ahead of the playhead - onset is instant, scenes react to the voice as it
  is being made, subtitles grow live from the shared timing array, and the WAV + sidecar
  land in the background for export. Finished takes loop endlessly inside the stream
  (a generator never emits song_finished), with subtitle time wrapped by the take length.
  Output level is a fixed calibrated gain (`Voice.OUT_GAIN`; raw cascade peak ~3.3 across
  trait extremes) instead of retroactive normalization, so live == WAV == loop passes.
- [ ] **Rung 3 - landmarks.** Deposit-style labels in the Workspace on the opening; the
  nonlinear activation model; content-keyed re-firing over the full sequence; A/B against
  the unlabeled rendition. This is where the sparse-generalization bet is tested.
- [ ] **Rung 4 - the actor.** Voice as a Cast kind, utterances as Track spans, subtitles
  rendered from the same spans, export through the bake. A storyboard that *speaks*.
  (First slice shipped early, outside the Cast: the `--synth` editor plays a rendered take
  as a normal session - scenes react to the narration through Spectrum - with karaoke
  subtitles filling word-by-word on the exact synthesis timings, hue riding the live
  harmonic signature.)
