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
  ~30x real time in GDScript, so `VoiceStream` runs a sliding window: a ~0.3s prebuffer
  is synthesized in the triggering frame, pushed into an `AudioStreamGenerator` on
  Spectrum's analyzed bus (`Spectrum.begin_stream`), and a **worker thread** keeps ~0.6s
  of lead ahead of the playhead - production is fully decoupled from the render loop, so
  a lagging scene cannot break the audio. Onset is instant, scenes react to the voice as
  it is being made, subtitles grow live from mutex-drained timing snapshots, and the WAV
  + sidecar land in the background for export. Finished takes loop endlessly on the
  thread (a generator never emits song_finished), subtitle time wrapped by take length.
  Output level is a fixed calibrated gain (`Voice.OUT_GAIN`; raw cascade peak ~3.3 across
  trait extremes) instead of retroactive normalization, so live == WAV == loop passes.
- [x] **Implicit speaking (no Speak button).** The surface is an instrument: the loaded
  draft speaks on open, timbre traits (pitch / tract / breath / grit) **retune the
  running stream immediately** (`VoiceStream.retune` - an atomic spec swap the worker
  reads per chunk, landing ~0.6s later: the voice bends while it speaks), and structural
  changes (text, seed, and the plan-baked lilt / pace / drawl) **restart the stream in
  place** after an idle debounce - generator buffer cleared, karaoke clock rebased via
  `time_base`, the scene session untouched. `--say` applies immediately on boot.
- [x] **Rung 2.5 - the walk and the belt** (the "comes in hot, stays hot" fix). The
  reading is now STATEFUL: `ProsodyWalk`, a homeostatic word-by-word walk over sliding
  EMAs - **arousal** opens hot and settles (hot = fast, settled = slow: pace finally
  varies), **spent emphasis** is a windowed EMA that spaces emphases out (an emphasis
  suppresses the next until it decays; emphasized words get a pre-pause, stretch, rise),
  **breath debt** forces real pauses in unpunctuated text, and a seeded **motif
  vocabulary** (tilt / lean / gap gestures) recurs per sentence - habits, not wander.
  The genome is sampled through a **lineage**: a linear chain of seeds where the root
  samples every parameter and each later seed perturbs by 0.6^generation - Evolve
  refines a kept reading instead of re-rolling it. The editor's **belt** (7 slots)
  captures {lineage, traits}: capturing IS the labeling - a kept seed reproduces its
  reading exactly, and the lineage records how it was found. All deterministic; breath
  emergence and lineage byte-determinism are asserted in voice_check.
- [x] **Rung 2.75 - the inventory and the attention reward.** The belt became an active
  population with implicit reward. The signal is **hold time**: while a reading plays,
  every seed shaping it accrues listened seconds - nobody labels roboticness by hand,
  you vote by not switching it off. Each belt entry tracks plays, hold seconds, kept
  (restores), evolved (children spawned), caught (descendants captured); the
  **acceptance metric** is hold-per-play relative to the belt's collective average
  (1.0x = average, above = the belt favors it), shown per seed in the inventory rows.
  Each row has a **toggle**: synthesis blends 1+N genomes - a fixed population PRIOR
  (the range midpoints, outside anyone's influence - the regularizer) + the working
  lineage + every toggled seed, uniform mean, motif vocabularies pooled. Release prunes.
  All deterministic per (lineage set, text); blend determinism asserted in voice_check.
- [x] **Rung 2.9 - sparse activations over the prose** (motifs v2). Ghost's `Activation`
  discipline applied to words: four independent channels (stretch / pitch / echo /
  swell), each a thresholded nonlinearity over seeded drive with a fast-attack
  slow-decay refractory - events are sparse and self-spacing (~0-1 echo strikes per 44
  words, lineage-dependent). Firing feeds a damped **resonance ring** that colours the
  following seconds and invites neighbouring firings (events cluster, then die out).
  **Pitch attractors**: a pooled per-voice anchor shelf the melody continuously
  gravitates toward (genome `gravity`), with pitch activations jumping most of the way -
  the GLaDOS / Miku musical-quantization quality, earned honestly. **Echo** is a real
  feedback delay line in the synthesis output; only activated words are sent into it,
  and it rings through the pauses. All genome-carried: Evolve and the belt select over
  every one of these behaviours.
- [x] **Rung 2.95 - the throw loop.** The interface collapsed to one gesture: **Throw ->
  it grows or it doesn't -> Keep or don't.** A throw parents itself from the belt
  weighted by acceptance (sometimes wild), inheriting lineage + traits with
  generation-decaying jitter, and speaks immediately. The belt integrates itself: no
  toggles, no blend controls - ancestors accrue hold time from their descendants'
  listening, so the acceptance that picks parents reflects what a seed's LINE produces.
  Each kept seed is an inspectable property: its tooltip reads out the genome's
  implications (temperament, breath span, pitch gravity, ring, strike bar, pace range)
  plus the behavioral ledger. The user's whole job is the collecting, archiving, and
  preservation of lineage; the system's whole job is making every throw worth throwing.
  Cross-seed influence statistics (who lifts whom) deferred - go simple first.
- [x] **Rung 2.99 - reward profiles + catch mechanics.** The signal inverted on use: a
  QUICK catch means the ear knew immediately; a long listen means deliberating. So
  reward is profiled, user-selected per session: **Snap** (default - decision latency
  earns: 2.4·8/(8+t)), **Drift** (sitting with it earns), **Hunt** (distance earns) -
  and the profile also shapes the throws themselves (Hunt ranges wilder). Keep became a
  catch ATTEMPT: integration difficulty = nearest-neighbor distance from the candidate
  to the bank in trait+genome space; foreign seeds fight the ball (p = 1 - 0.75·d) and
  can break free - retry allowed, but in Snap the clock keeps running. A caught seed
  records its reward and difficulty; ancestors share the reward. Acceptance now reads
  20·reward + hold/30 per play - the profile shapes how value is earned, the reading is
  uniform. The catch is animated: a code-drawn orb in the seed's hue closes on the
  voice, wobbles (each wobble a contest), then settles with a ring or bursts into
  shards. The roll is decided before the orb appears; the orb tells you.
- [x] **Rung 2.995 - disfluency and wobble** (the "same-y endings" fix). Diagnosis
  confirmed: fixed terminal constants cloned every sentence closing. Now
  `sentence_end()` draws each ending fresh (lengthening 0.75-1.45x, falls of varied
  depth, 12% flat, questions rising by varied amounts). **Hesitation** joined the
  activation channels: sparse gaps or filled "um"s land BEFORE words (genome
  `hesit_bias` makes some voices fluent, some halting); the authored `%HESITATION`
  token (the ASR-transcript format from the Ephemeral Rift podcast) always renders one,
  shown as an ellipsis in the karaoke. **Swing**: every activation kicks a cadence
  offset (rush or drag by seeded coin) that folds back into the running pace and
  decays - the perturbation-into-the-approximation wiring. Export from synthesis mode
  fixed: streamed takes are eligible the moment they complete. Still open from the
  transcript's texture: word repeats and false starts ("welcome welcome", "the that").
- [x] **Rung 2.999 - bites and cards.** The catch got its temporal skill and its
  judgment step. **Bites**: the planner knows exactly when every strike will play, so
  the stream publishes its event times; a strike opens a detection window that decays
  (tau 2s) - catch odds are dominated by bite freshness, thin when the water is still.
  The effect is genome-carried regardless: land the seed at any moment and it strikes
  again forever. **Cards**: a successful catch presents the seed as a constellation -
  lines seeded random from the lineage, colour attuned to cosine similarity against the
  party's centre (kin warm, foreign cold; 18-dim trait+genome space). **Accept** folds
  it in and every member's colour re-attunes to the new party (how, you learn by
  doing); **Release** (or throwing again) changes nothing. Hold or fold. Layout: status
  under the gesture, card next, belt always last, orb anchored at the button.
- [x] **Rung 2.9999 - easy fishing and the adrenochrome.** The game slowed down to be
  playable: bites are LATENT - a strike latches an anchor that pulls for minutes (45s
  decay, occasionally leaving on its own), no reflexes required. The two axes are
  **Throw** and **Pull**: pull while it holds and the hook sets or is lost. A set hook
  begins the REEL - up to minutes long, scaled by integration difficulty and profile -
  during which the **adrenochrome** anneals: the candidate's genome steps through the
  party's force field (high-acceptance members attract, low-acceptance repel, the
  prior always gently pulls) under cooling noise, integrating with the belt and
  straining against it, then FREEZES. The frozen creature carries an explicit genome
  (`Spec.adrenochrome` overrides the walk; identity - motifs, anchors, gates - stays
  lineage-derived), joins the belt on Accept, and the working voice BECOMES it - you
  hear what you caught. Children inherit frozen genomes verbatim. The **HUD**: an LCD
  water panel between the text and the buttons - the constellation, the line bellying
  while something pulls, the taut thrum + progress arc + countdown during the reel -
  with the reward profiles as a three-toggle switchboard on its edge.
- [x] **Rung 2.99999 - scenes as reward (the metamorphosis).** In synthesis sessions
  the Director is GAME PACED: autonomous music-driven cuts stop entirely; a scene
  changes when a catch jumps it (each seed owns a scene, derived from its lineage -
  "haunts spires" in the ledger - and restoring a seed returns to its place), so every
  new scene reads as a reward earned, not weather. And the **metamorphosis**: while
  the reel runs, `Director.aura` contorts the CURRENT scene at the base-class level -
  motion tempo stretches, the frame zoom-breathes and pan-sweeps, scaled by reel
  progress times catch difficulty, so a foreign catch bends the world BIG before the
  jump releases it. Discipline-safe (zoom + pan only, flat content never rolls).
- [ ] **Rung 3.0 - the room (physical properties).** The hunch that physics is what's
  still missing. Candidates, roughly in order of reach: **room acoustics** - a seeded
  impulse-response-ish reverb (a large hall vs a quiet booth as a genome axis; the echo
  bus is the seed of this, generalized from one delay tap to a small tap cloud);
  **the crowd** - background voices speaking from DIFFERENT chunks of the same text,
  much quieter, lowpassed, unsynchronized (a second/third VoiceStream mixed low - the
  murmur of a room full of readers); **air** between voice and listener (distance =
  lowpass + reverb ratio). All genome-carried, all catchable. The fuzzy-radio ideal:
  the voice should sound like it comes from SOMEWHERE.
- [ ] **Rung 3 - landmarks.** Deposit-style labels in the Workspace on the opening; the
  nonlinear activation model; content-keyed re-firing over the full sequence; A/B against
  the unlabeled rendition. This is where the sparse-generalization bet is tested. (The
  belt is the coarse version: whole-reading labels. Landmarks are the fine version:
  within-reading labels. Same philosophy, two grains. The acceptance metric is the
  training signal both grains will eventually share.)
- [ ] **Rung 4 - the actor.** Voice as a Cast kind, utterances as Track spans, subtitles
  rendered from the same spans, export through the bake. A storyboard that *speaks*.
  (First slice shipped early, outside the Cast: the `--synth` editor plays a rendered take
  as a normal session - scenes react to the narration through Spectrum - with karaoke
  subtitles filling word-by-word on the exact synthesis timings, hue riding the live
  harmonic signature.)
