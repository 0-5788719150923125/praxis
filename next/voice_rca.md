# Voice RCA - the robotic sound, the clicks, and what the audio keys off of

2026-07-20. Root cause analysis only - nothing changed yet. Empirical numbers
come from an instrumented render (three voices: the curated default plus rolls
42 and 777, same paragraph) that tapped the signal at three points inside the
output chain. Listen for yourself:

- `/tmp/ghost_scratch/rca_default.wav` - what ships today
- `/tmp/ghost_scratch/rca_default_clean.wav` - the SAME synthesis with the
  broadcast stage removed (pre-AGC tap; it plays quieter - that loudness gap
  is the only thing the broadcast stage is genuinely contributing)

Re-render any time: `python build/scratchpad.py run` (the RCA harness lives in
`run()` right now).

---

## 1. What is the audio keying off of? (why it speaks with an empty belt)

Nothing about the belt, the cast, or the water gates the audio. The chain is:

- `SynthEditor._ready` -> text non-empty -> `_mark_structural()` -> debounced
  `_apply()` (`synth_editor.gd:111`)
- `_apply()` speaks the persisted draft with the WORKING candidate:
  `_traits` + `_lineage` + `_working_genome` from `ghost.cfg` - the fields
  `_persist()` saves regardless of whether anything was ever kept
- fresh install: `_traits` empty -> `_background_traits()` -> zero vector =
  the curated default speaker; `_lineage` = `[1]`

So the voice you hear before any throw is "the seed you were holding when you
last closed the app" (or the curated default), reading the draft. This was the
deliberate "write and it speaks" decision from 07-18, made before the fishing
loop existed - the fishing metaphor has since outgrown it. Under the current
metaphor it is wrong in exactly the way you said: sound with no source. There
is no bug here; it is a design leftover.

### Proposed gating (maps cleanly onto the existing state machine)

The mode states already exist: idle water / thrown / anchored / reeling /
caught. Audio presence can be a single scalar, call it `presence`:

- **belt empty, nothing thrown**: `presence = 0`. Optionally keep the
  permanent FLOOR_MIN static as the "tuned but empty station" bed - it is
  on-theme and tells you the instrument is live. Or true silence.
- **belt has seeds**: the PARTY is the background voice - acceptance-weighted
  belt blend. The plumbing already exists and is currently orphaned:
  `Spec.influences` is still honored by the 1+N walk blend but the editor
  stopped populating it when toggles were removed, and `_background_traits()`
  already computes the blended trait vector. "Seeds on the belt are what we
  synthesize with" is almost literally these two APIs re-joined.
- **thrown, anchored, reeling**: the candidate ramps IN with the reel -
  `presence` follows reel progress (the same scalar already driving
  `Director.set_aura`). Far away = quiet and dark (lowpass), closer = louder
  and brighter. This is the rung 3.0 "distance = lowpass" room-physics idea
  finding its first real use.
- **caught + accepted**: full presence (the restart-to-hear-what-you-caught
  behavior already does this).

One real design catch - the bootstrap deadlock: bites are detected from the
PLAYING stream's clock (`_fresh_strike` reads `Spectrum.current.time` against
`stream.events`). If a throw with an empty belt makes no sound at all, there
is no stream, no clock, no strikes, no anchor, no hook - you can never catch
the first seed. Two resolutions:

- (a) the stream always runs, but through the presence gain - at `presence=0`
  it is inaudible while the clock and events still tick (feeling a bite on a
  line you cannot hear is fine; the HUD is the bobber);
- (b) strikes before the first catch come from plan events on a wall clock,
  no stream. More machinery, no real gain.

Recommend (a). Also note: `Spectrum.begin_stream` is what gives the session
its seed and the scenes their audio - full silence with no stream would also
freeze the show. (a) keeps the session alive; the analyzer just hears quiet.

---

## 2. The clicks, pops, and static - measured

The discrete per-phoneme click class we fixed earlier IS fixed: across ~31 s
of rendered audio the impulse detector finds 0-1 isolated click events per
take, none at phone boundaries, none in silence. What reads as "clicks and
pops" now is something else, and the instrumented render finds it precisely.

**The broadcast chain (AGC -> cosine dampener -> static bed) is now the
dominant artifact source. It manufactures crackle and hiss out of clean
input.** Numbers from the default voice (the two rolls agree):

| measurement | value |
|---|---|
| pre-broadcast signal | peak 0.84, RMS 0.064, crest 13x - CLEAN, nothing to limit |
| samples that would pass the dampener knee unaided | 0.5% |
| AGC gain | pegged at the 2.5x maximum for 90-96% of ALL samples |
| voiced samples zeroed OUTRIGHT by the dampener | 10.2% |
| voiced samples crushed by more than half | 13.7% |
| static bed | above -40 dB for 97% of the take, median -33 dB |
| silence floor | -38 dBFS through every pause |
| 5-11 kHz band, vowel cores | pre-broadcast -25 dB -> final -9 dB (+16 dB of junk) |
| 3-5 kHz band, vowel cores | pre-broadcast -30 dB -> final -14 dB (+16 dB of junk) |

The causal chain, each step verified:

1. **The AGC's model is wrong for this signal.** It steers mean |x| toward
   `AGC_TARGET` "as a sine" (`level * 1.6` assumes crest ~1.4). Speech
   through formant resonators has crest ~13. The tracker therefore reports
   "too quiet" almost always, the gain rails at `AGC_MAX_BOOST` 2.5x, and a
   signal whose peaks were already calibrated to ~0.85 by `OUT_GAIN` gets
   pushed to ~2.1.
2. **The cosine dampener is a non-monotonic waveshaper.** Above the knee,
   gain falls to exactly ZERO at the ceiling - so every waveform peak that
   crosses 0.8 is not flattened but replaced by a hole. At 2.5x boost that is
   one sample in ten during speech. Holes punched into peaks at f0 rate are
   wideband splatter - this is the crackle, and most of the +16 dB of
   high-band junk. (v1's masking static was the pops then; the fold is the
   pops now. The two dB tables above are the A/B.)
3. **The bed then narrates the damage.** `excess` is nonzero almost
   constantly, so `sbed` never decays and `sfloor` ratchets to its cap and
   stays - the "snow" designed for rare heavy moments became a permanent
   -33 dB hiss, and its ramps ride every onset like a puff of static.
4. Bonus: because the AGC pegs during every pause (level decays toward zero,
   want maxes out), each utterance onset after a pause arrives pre-boosted
   at 2.5x - measured at 2.50 for 14 of 14 post-pause onsets - so phrase
   starts slam the fold hardest. "Comes in hot" is literal.

The fix direction is not another compensation layer; it is removing the arms
race. The pre-broadcast signal needs almost nothing: `OUT_GAIN` already lands
peaks at 0.73-0.84 across trait extremes (measured, matching the original
calibration). We synthesize 2.5 s AHEAD of the playhead - a true lookahead
peak limiter (the transparent kind, gain computed over the next few ms,
distortion-free by construction) is trivially available here and would handle
the rare stacked-activation overshoot. The AGC-toward-a-sine, the zero-gain
fold, and the always-on bed should all go, or the bed can stay as flavor at a
level tied to ACTUAL rare overdrive. Loudness consistency across rolls (the
real thing the AGC was after) belongs at PLAN time - scale a take's gain once
from its known contents (we know every amp before we render), not per-sample.

**Separate, live-only pop:** every throw/edit restart is
`Spectrum.restart_stream()` = `_player.stop(); _player.play()` - a hard cut
mid-waveform. Headless renders never show it; every interactive session hears
it. Needs a ~5-10 ms fade-out before the cut (and the ring already gives us
the lead to do it).

**Residual synthesis-side nits found while reading (small, listed for
completeness):** `period_gain` (shimmer) is a local reset to 1.0 at every
segment boundary, a mid-cycle gain step of a few percent per phoneme; the
per-period jitter draw is overwritten at the next 64-sample frame boundary
(`inc` is recomputed from `f0sm` per frame), so jitter only shapes ~40% of
each cycle. Both are quiet compared to the broadcast chain, but the second
one matters for section 3.

---

## 3. Why it still sounds robotic - the vocal physics we skipped

Ranked by expected audible payoff. The broadcast chain above is rank 0 - fix
that first, then re-listen before believing anything else on this list, since
fuzz this loud masks finer judgments.

1. **Loudness dynamics are being erased after we compute them.** The walk
   spends real machinery on emphasis, swells, effort tilt, arousal - then the
   AGC normalizes the whole performance back to one loudness. A voice with no
   dynamic range reads as machine even with perfect pitch. (Same fix as
   section 2 - listed here because it is a NATURALNESS cost, not just a
   distortion cost.)
2. **The periodicity is too perfect.** The jitter bug above means cycle
   lengths barely vary - a too-regular pulse train is the single oldest
   "robot voice" cue there is. The intent (spec.jitter 1.2%) is right; the
   frame loop discards most of it. One-line class of fix: let the drawn
   period survive until the next wrap.
3. **Only vowels carry the melody.** In `plan()`, `semitones` is computed
   inside `if is_vowel` - every consonant and every silence targets 0 st, so
   f0 dives toward neutral through every voiced consonant (L, R, M, N, W...)
   and glides back up into the next vowel, at 35 ms EMA speed, inside every
   word. Measured: within-word f0 span p90 of 4-5 semitones. Human f0 is one
   continuous gesture across the syllable; ours is a picket fence the EMA
   sands the corners off. Fix direction: give non-vowel segments the
   interpolated semitone value between neighboring vowels (the contour is a
   WORD property, not a vowel property).
4. **Three formants, and nothing above 3 kHz but noise.** Pre-broadcast vowel
   cores hold -25 to -30 dB above 3 kHz. Real voices have F4/F5 and a
   presence region; ours has the air-band hiss. This is the "hollow AM
   radio" component of the timbre. Klatt's synth used 5-6 cascade poles for
   exactly this reason - two fixed extra resonators (F4 ~3.3k, F5 ~4.5k,
   scaled by tract) are cheap in the existing cascade.
5. **No voice onset time.** After voiceless stops (P, T, K) a real voice
   leaks 30-80 ms of aspiration before voicing starts; ours snaps from burst
   straight into a fully voiced vowel. This is one of the strongest
   "synthetic" tells in stop-heavy text. We already have the `asp` machinery
   (HH) - a short aspirated release borrowed from it after voiceless stops
   is most of the fix.
6. **Coarticulation is one symmetric 18 ms EMA.** Real formant transitions
   are 50-100 ms, asymmetric, and consonant-specific (locus targets); much of
   stop/glide identity lives in the transition, not the steady state. The
   single fast EMA gives "plastic" morphs between static postures. A per-type
   time constant (slow into glides/nasals, fast out of bursts) would be the
   cheap first step; locus targets the real one.
7. **Nasals have no anti-resonance.** M/N/NG through the same 3-pole cascade
   at 0.45 gain is a buzzy hum; the defining feature of nasals is a spectral
   ZERO. One anti-resonator (a notch) switched in for nasals would do it.
8. **The air trait's top of range replaces the voice with static.** Roll 42
   (air high) shows 5-11 kHz at -5.9 dB PRE-broadcast and so little
   periodicity my f0 tracker only locks 59 frames (vs 238 for the default).
   Some population members are effectively whisper-static; on a belt they
   read as broken, not breathy. Either narrow `air_gain`'s sampled range or
   couple it against harmonic level so breath never exceeds voice.

Notably NOT on this list: the prosody walk itself. The arousal/emphasis/
breath/motif/activation system is doing its job - the readings MOVE. The
robotic quality is downstream of it (dynamics erased, melody notched, source
too regular, spectrum hollow).

---

## 4. The chorus idea - considered

Three different things hide inside "a chorus accumulated in the EMAs but
played back as one voice," with different outcomes:

- **(a) Parameter averaging across population samples** (mean of N sampled
  genomes/trait vectors). This exists already - it is exactly the 1+N
  influence blend with the PRIOR, and the reel's annealing is a weighted
  version of it. Averaging parameters regresses the reading toward the
  population mean: smoother, safer, and FLATTER. It will not fix robotic
  timbre (the artifacts are downstream of the parameters) and it spends
  variance - the thing the whole seed economy is designed to farm.
- **(b) Trajectory-level ensemble with votes/modes** - render the PLAN N
  times from different genome rolls, then per word take the MAJORITY on
  discrete events (does emphasis fire? echo? hesitation?) and the median on
  continuous values (duration, f0 target), and synthesize ONCE from the
  consensus reading. Alignment is free because it is the same text. This is
  the interesting version of the idea: a population CONSENSUS reading keeps
  decisions crisp (votes do not blur events the way means do) while washing
  out one genome's tics. It matches the original voice.md picture (500
  renders -> EMA realization) better than anything we have built yet. Worth
  an experiment - as a new way to DERIVE a reading, e.g. what the belt as a
  whole would say - but it will make readings more typical, not more human;
  it cannot repair source/filter artifacts.
- **(c) Unison doubling at the SIGNAL level** - synthesize the same plan 2-3
  times with independent jitter/shimmer/tension noise and hair-thin offsets
  (a few cents of f0, ~2% formant scale, a few ms onset), and sum. This is
  the one that directly attacks the too-perfect-periodicity problem: the
  summed periods decorrelate exactly the way a real glottis's cycles do, and
  it is how doubled vocals get thick without reading as two singers. Cost is
  linear (synthesis runs 16-27x realtime, so 3 layers still stream
  comfortably) and it stays ONE voice to the ear if the offsets stay tiny.
  Risks: push the offsets and it turns into a phasey "effect" or a crowd;
  intelligibility blurs slightly; and it partly papers over problem 2 rather
  than fixing it.

Honest recommendation: your instinct that something chorus-like sounds more
natural than a lone perfect oscillator is right, but the cheapest big win is
fixing WHY the lone voice is perfect (rank 0 and 2 above). After that, (c) as
a flavor layer and (b) as the belt's consensus-reading mechanic are both
live options - (b) is also a natural fit for the gating design in section 1,
where the belt collectively IS the background voice.

---

## 5. Suggested order of attack (for discussion)

1. Broadcast chain replacement: drop AGC + fold, lookahead limiter on the
   worker's existing 2.5 s lead, bed tied to actual rare overdrive, plan-time
   loudness per take. (Fixes crackle, hiss, AND dynamics - rank 0 and 1.)
2. Restart fade (the live pop on every throw/edit).
3. Jitter fix (one-line class), then re-listen before judging further.
4. Audio presence gating per section 1 (design decision needed on idle
   static vs silence, and confirming option (a) for the bootstrap).
5. Melody continuity through consonants (rank 3).
6. Then pick from ranks 4-8 and the chorus experiments by ear.

Open questions for you: idle = faint static or true silence? Should presence
ramp attach to anchor strength too, or only reel progress? Is the always-on
radio flavor (bed) something to keep at all, now that we know it costs -33 dB
of hiss?

---

## 6. Implemented (2026-07-20, same day) - first attempt at all of it

Everything below is in the working tree; `voice_check` ALL OK, compile clean,
headless `--say` boot clean. Fresh takes at `/tmp/ghost_scratch/rca_*.wav`
(re-render with `python build/scratchpad.py run`; `... say` boots the app
headless).

**Broadcast v3** (`voice.gd`): AGC, cosine fold and always-on bed are gone.
In their place a one-block lookahead limiter - output runs one 64-sample
block behind synthesis, each incoming block's peak sets a linear gain ramp
across the outgoing block. Measured: the limiter never engaged once across
all three test voices (pre-limiter peaks 0.71-0.86 vs the 0.92 ceiling);
final spectrum is bit-identical to the raw synthesis. The bed remains only
as a mask tied to actual limiter work, with the permanent floor dropped to
~-56 dB; measured silence floor -48 to -60 dBFS (was -38).

**The clicks were the stop bursts** - the second, deeper source your ear
caught in the clean sample. Every impulse event the detector flagged sat
inside a P/T/K/D burst: the envelope was BACKWARDS (slow 8 ms EMA attack,
hard cut - a burst is instant attack, exponential decay), and the noise was
added AFTER the formant cascade, a bare wideband tick exciting nothing. Both
fixed: bursts now attack instantly, decay through the EMA, and excite the
TRACT, retargeted to the next phone first so the release carries the coming
vowel's transition (locus-lite). Click events: 21 -> 4-7 per take, remaining
outliers at a third of the old amplitude with plosive-shaped profiles.

**Robot voice ranks 2-8, all landed**: jitter/shimmer draws persist in state
until the next period (the frame loop was quantizing them away); f0
continuity through consonants and silences (within-word f0 span p90 dropped
from 4-5 st to 1-2 st); F4/F5 presence poles at 3.4k/4.7k x tract; VOT
aspiration after voiceless stops (through the next phone's formants);
per-articulator coarticulation speeds (slow into vowels/glides/nasals, fast
out of bursts); nasal anti-resonance as a proper biquad notch - NOT a bare
zero pair (first attempt amplified the top octave ~30x and sprayed 5x peaks;
the notch's poles are what make the zero safe); shimmer default 6%->4% and
air trait ceiling roughly halved (roll 42's whisper-static class is gone).

**Sampling normalization** (`synth_editor.gd`): `_temper_traits` - a trust
region (RMS radius 0.85 over the trait axes) around the party's
acceptance-weighted centre (curated default when the belt is empty); a draw
beyond it is pulled back onto the boundary, direction intact. Drifting
widens the radius - foreign voices are EARNED by the cage, never rolled.

**Silence at launch + presence** (`synth_editor.gd`, `voice_stream.gd`): no
auto-speak; the first throw breaks the silence (`--say` counts as a landed
cast for demos/headless checks). Presence = distance from the caster,
applied at push time (the WAV take stays canonical): gain plus a distance
lowpass (~500 Hz far, open when landed). Far cast 0.28, anchored up to 0.6,
reel 0.4 -> 1.0 with progress, dipping when it runs; landed/kept/restored =
1. The stream always runs so the clock and bites keep working (the
bootstrap deadlock resolution) - at low presence you faintly hear the far
voice, which reads as intended.

**The reel is a fight** (`synth_editor.gd`): progress = belt power against
seeded RUNS (per-lineage fight pattern; foreign catches run more and
harder); a run pays line back out, drops presence, whips the HUD line red.
`_reel_power` replaces `_retrieval_factor`: knowledge (hold time + reward)
now powers retrieval in EVERY mode, 0.9x to 2.2x. Everything hooked lands
eventually; the wait anneals the adrenochrome AND auditions it - every 2 s
the stream's timbre retunes toward the annealing traits, so you increasingly
hear what you are pulling in before deciding. Folding hands the timbre back.
HUD shows percent, not a countdown (duration is no longer fixed).

**Restart pop** (`spectrum.gd`, `voice_stream.gd`): `fade_stream()` drops
the player ~40 ms before the stop/play cycle; every new/restarted stream
also fades in from silence over ~0.2 s. Overlapping restarts queue, latest
wins.

**Loudness (added after "far too quiet" report)**: two compounding causes.
The presence curve squared its gain (a fresh cast played at ~8% amplitude -
the dominant inaudibility), and the take itself sat at -24 dBFS RMS because
OUT_GAIN was staged for peaks, with the old AGC's 2.5x silently supplying
all the loudness. Fixed: presence gain is linear with an audible floor
(far cast ~-5 dB + the distance lowpass; distance is the FILTER's job,
silence belongs only to the un-cast water), and the broadcast stage grew
two honest components: a bounded syllable LEVELER (2:1 on the block
envelope, fast attack / slow release, +-4 dB bounds so the walk's dynamics
survive; NOTE its envelope rides a vowel's NEAR-PEAK level - the target is
in those units, a mean-sized target quietly cut every vowel) and a tanh
SOFT CEILING at 0.8 (monotonic, unlike the old cosine fold - peaks round
into fuzzy-radio warmth, never holes). A phase rotator was tried for crest
reduction and REMOVED - the resonator chain already disperses phase, it
measured nothing. Result: takes at -20 dBFS RMS (was -24), zero hard
limiting, click counts and spectrum identical pre/post chain. Live loudness
vs the report: ~+9 dB landed, ~+21 dB on a fresh cast.

NOT done yet (deliberately, judge by ear first): unison doubling (c) and the
belt-consensus reading (b) from section 4; locus targets beyond the burst
retarget.

---

## 7. Round two (2026-07-26) - the clicks localized to the sample, and lineage aging

Report: clicking/popping "20-30x per minute, never normalizes", voices chaotic
with deep lineages. Everything below is implemented and measured; `voice_check`
ALL OK (including a new ornament-aging check), compile clean.

**What the instrumented render actually showed first**: the adaptive masking
bed was NOT the noise the report describes - it idled at the -56 dB floor for
100% of samples across all three test voices. The clicks were synthesis-side
(identical with the whole broadcast chain bypassed). Component-level taps
(per-sample src / cascade / noise-band / echo) pinned the dominant class:

- **The noise ROUTE switch.** Aspiration (VOT, HH) and bursts send noise
  THROUGH the cascade; fricatives ride the post-cascade band. The envelope
  (nampsm) survived an asp->vowel or burst->vowel handover, so the leftover
  re-emitted as a raw burst-tuned band snapping on in ONE sample ~1 ms into
  the vowel after every stop (band tap: 0 -> 0.27 in one sample). Fixed:
  route tracked in state; the envelope zeroes on a cascade->band switch.
  Fricative->vowel decay tails are unaffected (both band-routed).
- **Per-frame envelope staircases.** Voiced amp and noise envelopes were
  per-frame constants (2.9 ms steps); the burst attack landed as a
  one-sample cliff. Both now RAMP per sample across each frame.
- **The noise resonator froze through the asp branch** (state held its
  burst-era ring, re-added on resume); it now steps every sample.

Measured: click events 14/11/11 -> 5/5/7 per take; max one-sample step
0.34 -> 0.16; every event >=0.2 is gone. The survivors are smooth
multi-sample plosive transients (legitimate stop acoustics).

**The rest of the noise report was LIVE-only, addressed at the source**:

- The adaptive bed + its 60 s floor ratchet ("never normalizes" - latent, but
  real once live peak-stacking engaged it) are DELETED. Only the constant
  FLOOR_MIN grain remains. The tanh ceiling covers its own peaks.
- retune() (the reel retunes every 2 s = the 20-30/min cadence) stepped
  breath/air_gain/air_cut/formant_scale(F4/F5) instantly mid-stream; all four
  now glide through a ~60 ms EMA in synth state (initialized from the spec,
  so fixed-spec renders are untouched).
- The location reception path hard-clipped (clampf +-1 after a 1.8x resonant
  boost) - a click per overshooting peak near a beacon. Now a tanh soft
  ceiling (LOC_CEIL 0.95), live path and offline bake both.

**Lineage aging (the "too many vocal effects" report)**: ornaments now AGE
OUT while identity stays root-anchored. A generation's spawned modulator
keeps its seeded gesture but its raw depth decays by ORN_DECAY (0.7) per
generation that lands after it - the existing suppress floor then prunes the
faded ones (AGE joins DAMPEN/NORMALIZE/SUPPRESS as the fourth regularizer).
The elaboration anchor shelf became a WINDOW: seeded by the NEWEST
generations, capped at 4 (was: oldest-first, up to 8) - refinements rotate
old notes out instead of piling the shelf higher. The genome's 0.6^gen
root-anchored refinement is unchanged: the caught seed IS the voice; its
ornaments are what fade. New voice_check: root gesture 0.916 -> 0.053 raw
depth over 8 generations and pruned at finalize, newest generation enters
un-aged, anchor shelf bounded at 10 for a single reading.

Listen: /tmp/ghost_scratch/rca_default.wav (+ roll42/roll777) and the
voice_check WAVs. NOT touched: unison doubling, belt-consensus reading,
locus targets (unchanged from section 6's deferral).

---

## 8. Round three (2026-07-26, same day) - the "random noise bursts" found and killed

Report: still "brief bursts of noise injected at random places", suspicion the
masking noise is masking nothing. Two changes:

**NOISE_FX kill switch** (`voice.gd`, hardcoded const, flip to restore): false
disables every NON-PHONEMIC noise injection - aspiration hiss (spec.breath),
the air/static band (spec.air_gain), the constant output grain (FLOOR_MIN).
Consonant noise (fricatives, bursts, VOT) stays - that IS the T/S/K. The rng
draws still happen when disabled, so the flag A/Bs the same take. Currently
FALSE.

**The actual random bursts were the ECHO BUS ringing frication.** Measured on
roll42: a 200 ms, aperiodic (autocorr 0.07), 3-11 kHz burst at ~-27 dB
sitting in a pause at 7.24 s - the echo activation at 6.78 s had thrown the
word "static" (S-T-AE-T-IH-K) into the undamped delay line, and its S-noise
rang through the following gap at 0.17 s intervals. Sparse walk-gated echo
firings = "random places"; raw fricative repeats = "noise injection". Fixed
with a damped delay line: one-pole lowpass (ECHO_LP 1400 Hz) INSIDE the
feedback loop, ~15 dB/pass at 8 kHz - every repeat comes back darker, voiced
ringing survives, static dies by the first repeat.

Measured after both: pause floors -62/-55/-60 dBFS (default/roll42/roll777;
roll42 was -40.6 before - dominated entirely by that echoed S), the 7.24 s
burst down ~17 dB to a faint dark murmur (the echo feature, working), click
counts unchanged. voice_check ALL OK.

---

## 9. Round four (2026-07-26) - the foundational one: gain staging into the ceiling

Report: noise/click bursts ~20-30/min, visible as visual-scene activity bursts,
identical on a BRAND-NEW seed - so outside the lineage system. Measured from
the taps: 37-52 saturation events/min, ~3% of ALL samples audibly distorted
(dist > 0.02), max drive 1.6x - and NOT correlated with walk activations. The
cause was ordinary STRESSED VOWELS: OUT_GAIN 0.55 staged peaks at 1.5-1.75
against an ALWAYS-ON tanh ceiling at 0.8, so every stressed syllable's pulse
peaks were folded (fuzz burst, return to normal), and the leveler then pulled
the RMS back DOWN to -20 dBFS. Distortion bought loudness that was discarded.
Seed-independent by construction - exactly the report.

**Broadcast v4**: the lookahead RAMP is the peak authority again (steers
block peaks under LIMIT 0.85, cornerless, distortion-free); the clip is a
SAFETY NET only - identity below KNEE 0.7, rounding into CLIP_CEIL 0.98
above. OUT_GAIN 0.55 -> 0.35 so peaks arrive ~1.0-1.1 raw (ramp trims the
hottest block ~1 dB). Measured after: ZERO samples over the distortion
threshold (max instantaneous error 0.013 = -38 dB on peak tips), clicks
eased further (4/5/8 per take, from 14/11/11 at day start), silence floors
-59 to -66 dBFS. Cost: RMS -21.5 to -22.1 dBFS, ~1.4 dB quieter than the
distorting chain - if it reads too quiet by ear, raise COMP_TARGET (honest
loudness), never OUT_GAIN past peak-safe staging.

Remaining honest caveat: the walk's activations/emphases still create real
LOUDNESS dynamics (by design - the performance). If scenes still burst on
those but the audio is clean, that is prosody, and the dial is act_thr /
ELABORATION, not the broadcast chain.

---

## 10. Round five (2026-07-26) - RAW bypass: the bisect experiment

"Essentially no improvement" after the v4 restaging - so stop patching and
bisect. `Voice.RAW_MODE := true` (hardcoded, next to NOISE_FX): every seed
plays the BASE synthesis only.

OFF while raw: the walk's audible realization (pace/emphasis/activations/
spontaneous hesitations/breath debt/tilt/ring/gravity - the walk still
advances and strike events still fire, so bites keep working), ProsodyField
wander, per-period jitter/shimmer/tension, the whole broadcast chain
(leveler/limiter/clip/grain; blocks emit under fixed RAW_TRIM 0.8), and
VoiceStream's push-time presence/location (the take reaches the bus
untouched; bake_location also returns raw for exports).

STILL ON (the base): phonemes, coarticulation EMAs, declination, accents,
vowel reduction, punctuation pauses, terminal contours, VOT, the trait
vector. NOISE_FX stays independently false.

Measured raw: silence floors -74/-78/-77 dBFS, clicks 4/5/6 (plosive-shaped),
RMS -22.8/-24.3/-23.4 (quieter - no leveler; expected). voice_check ALL OK
(readings check auto-skips under RAW_MODE).

Restore order when listening resumes (one at a time, listen between):
1. broadcast chain (RAW emit branch) - loudness back, still no modulation
2. jitter/shimmer/tension (the organic source wander)
3. ProsodyField wander
4. the walk's realization (drop the RAW mods override)
5. presence/location push effects
6. NOISE_FX true last (breath/air/grain)

---

## 11. Round six (2026-07-26) - the noise survives RAW mode: tap the mixer

The bursts persist with RAW_MODE on and nothing in the console (no underruns
logged; the session-seed lines in the user's log are stream restarts, each
followed by a Director re-seed + scene cut - expected on throw/edit). The
synthesized signal path is now measured clean end to end, so the remaining
suspects are the DELIVERY layers: the generator ring / resampler (the mixer
runs 44100, the generator 22050), the loop seam, or post-mix (PipeWire xruns
under scene load - which would also explain the "visual burst = noise"
correlation with REVERSED causality: heavy scene moment -> load spike ->
xrun crackle).

`Spectrum.LIVE_TAP := true` (hardcoded): an AudioEffectCapture on the Master
bus records the LAST 120 s of what the mixer actually hears into a rolling
ring, written to user://synth/live_tap.wav on quit (at the real mix rate).
Reproduce the noise, quit, analyze the file. Bursts present in the capture =
inside the mix, analyzable; capture clean while the ear hears them = post-mix
(driver/OS). Flip LIVE_TAP off when the hunt is over.

---

## 12. Round seven (2026-07-26) - FOUND: frication was the noise, all along

The live tap (section 11) caught the bursts IN THE MIX: 27 HF-noise events in
22.3 s, 40-200 ms, 70-90% of their energy above 5 kHz, at RMS 0.14-0.25
against a -27 dBFS session average. Matched against the offline phone map:
**every /S/ rendered at RMS 0.27-0.29 (peaks 0.8+) while vowels sat at
0.011-0.06** - fricatives +13 dB OVER the loudest vowel, +28 dB over the
median. Natural /s/ sits BELOW vowel level. Most of a take's total energy
was literally S-noise.

Mechanism: the post-cascade noise band (fricatives) is a resonator running
at its resonant gain, and the radiation first-difference then tilts +6
dB/oct - boosting an 8 kHz band ~12x relative to vowel bands. Every
S/Z/SH/F = a 100-200 ms broadband blast: "brief noise bursts at random
places", in every voice, every seed, raw mode included - it WAS the raw
synthesis. The visuals bursting were scenes reacting to the 8 kHz flux.

Fix: `FRIC_TRIM := 0.12` on the post-cascade band only (bursts route through
the cascade; untouched). Measured after: fricative/vowel median RMS ratio
4.5-25x -> 0.81-0.85 (natural), the S blasts down ~19 dB.

Fallout: EVERY prior gain-staging number was S-contaminated - the "peaks
1.5-1.75" that drove the limiter/tanh story were S peaks; the true voiced
peak at OUT_GAIN 0.35 is ~0.2. RAW_TRIM restaged 0.8 -> 4.0 (raw takes now
peak 0.6-0.8, RMS ~-25 dBFS). When the broadcast chain is restored,
OUT_GAIN / COMP_TARGET need honest recalibration against the S-clean signal.

Still open: whether tamed /s/ reads crisp enough (ear test); the bisect
ladder from section 10 for restoring modulations; LIVE_TAP still armed for
one confirming live run.

---

## 13. Round eight (2026-07-26) - correction, and the pure_say ladder

CORRECTION of round seven's follow-up: the second live tap MATCHES current
synthesis exactly (RMS -25.1 live vs -24.5 offline; fric events 0.11-0.19 =
offline 0.034 x the RAW_TRIM 4.0 restage; events repeat on the take's 7.92 s
loop = S positions in the text). The app plays exactly what we ship - the
levels are as designed, and the ear still reports "tons of noise, no
difference". So the remaining question is PERCEPTUAL: equal-RMS at 8 kHz
reads far louder than at voice frequencies (real /s/ sits 10-15 dB BELOW
vowels, not ~-1 dB), and our /s/ is raw resonator-filtered white noise -
harsher than real turbulence.

`tests/pure_say.gd` (user's minimal-repro idea): bare Voice.render -> WAV,
no Ghost, no autoloads, no stream. Renders one text at a LADDER of
FRIC_TRIM levels (off 0.0 / soft 0.04 / natural 0.12 / full 1.0), each
peak-normalized -> /tmp/ghost_scratch/pure_say_*.wav. FRIC_TRIM became a
static var solely so the ladder can sweep it. The `off` rung is decisive:
if the noise is STILL there with every fricative silent, frication is
exonerated and the base synthesis (source/cascade/plosives) is the suspect;
if `off` is clean, the noise IS the frication and the fix is level (soft)
plus spectral shaping of /s/.

---

## 14. Round nine (2026-07-26) - the verdict: frication doesn't FUSE

Ear verdict on the level ladder: `off` decent (tiny residual = bursts/VOT),
soft/natural/full ALL equally terrible - so the offense is CHARACTER, not
level. The isolated frication signal (natural minus off, sample-aligned)
measures clean: zero impulses, smooth envelope, proper broadband 4-11 kHz
spectrum. The crime is WHERE it enters: post-cascade, sharing no formant
shaping or coarticulation with the voice - the ear correctly segregates it
as a SECOND SOURCE injecting static. The burst path documented this exact
lesson long ago ("a bare wideband tick added after the cascade reads as a
pop; through the formants it reads as a consonant"). The user's original
phrase - "forced-injection of noise" - was mechanically accurate all along.

Shipped defaults: FRIC_TRIM 0.0 (the validated-decent state: voiceless
fricatives silent, slot/length kept, voiced murmurs stay), FRIC_THRU 0.0
(new: frication routed THROUGH the cascade, burst-style). pure_say ladder
v2 auditions: off / thru 2.2 / thru_hot 4.5 / band_soft 0.04 (contrast).
Note: the cascade amplifies the S-band drive (thru raw peak 2.3 vs off
0.96) - if `thru` has the right character at the wrong loudness, only the
FRIC_THRU constant moves. If through-tract frication cannot pass the ear
test, the real fix is a Klatt-style PARALLEL frication branch (dedicated
wide resonators F2'-F6 with per-phoneme amplitudes, pitch-synchronous AM on
voiced fricatives) - a design task for a fresh session.

---

## 15. Round ten (2026-07-26) - through-tract confirmed, drive calibrated

Ear verdict on ladder v2: `thru`/`thru_hot` moved the character from "pure
static" to "heavily-modulated voice effects" - the ROUTE is right, the
DRIVE was wrong. Measured why: at unity drive the cascade seats voiceless
fricatives at 4.44x vowel RMS, so the guessed drives (2.2/4.5) put them at
~10-20x - the original level crime through a better pipe.

pure_say ladder v3 SELF-CALIBRATES: renders at drive 1.0, measures the
voiceless-fric/vowel median RMS ratio from its own timing map, solves for
TARGET_RATIO 0.25 (~-12 dB, where real /s/ lives) -> calibrated drive 0.06.
Rungs: off (floor ratio 0.134 = bleed/ring) / thru_soft 0.03 (0.189) /
thru 0.06 (0.286) / thru_firm 0.11 (0.520). Raw peak 0.961 on EVERY rung -
fricatives no longer set the take's peak. Awaiting the ear: pick a rung ->
set Voice.FRIC_THRU to it (soft 0.03 / thru 0.06 / firm 0.11); if all
still offend, the Klatt parallel branch remains the fallback design.

---

## 16. Round eleven (2026-07-26) - back into Ghost: the instrument restored

Ear verdict on ladder v3: ALL FOUR rungs acceptable ("nowhere near as bad as
when we started"). Shipped configuration:

- FRIC_THRU 0.06 (the calibrated -12 dB seat), FRIC_TRIM 0.0 - the
  post-cascade band stays dead permanently; frication lives in the tract.
- RAW_MODE false - the full instrument returns: walk realization
  (emphasis/activations/hesitations/breath debt), ProsodyField wander,
  jitter/shimmer/tension, broadcast chain, presence/location push effects,
  and with them the round-two lineage aging (ornament decay, anchor window).
- OUT_GAIN 0.35 -> 0.9 -> 2.0, measured at each step: the S-clean voice has
  a 16-23x crest, so static gain must carry the level. At 2.0: RMS -21 to
  -23 dBFS, peaks steered to LIMIT exactly, ramp active on 0.2% of samples,
  ZERO saturation. The leveler is a gentle evener again, not a life-support
  system running on S energy.
- NOISE_FX false - breath/air/grain remain the LAST restore rung, only
  after the full instrument passes the ear in the seed-building loop.
- LIVE_TAP still armed for the in-app test; flip off after.

voice_check ALL OK (readings + ornament checks re-armed), compile clean,
boot clean. Remaining known artifacts: plosive-transient clicks (4-15 per
take, legitimate stop acoustics at threshold), and whatever the live loop
surfaces - the tap will catch it.

---

## 17. Round twelve (2026-07-26) - restore complete + the fidelity pass

Seeds confirmed "pretty good"; remaining ask: louder, brighter, clearer
consonants ("quiet and dull"). Shipped:

- **NOISE_FX true** (the last restore rung): breath + air band + grain are
  back - the air band is most of the voice's top-octave life, so it is part
  of the fidelity answer, not just flavor. LIVE_TAP retired (false).
- **SR 22050 -> 44100**: doubles the bandwidth ceiling AND removes Godot's
  22050->44100 playback resample (the generator now runs at device rate -
  no linear-resample imaging). Everything derives from SR; per-BLOCK
  constants rescaled to keep their time constants (LIMIT_RELEASE 1.022,
  leveler 0.19/0.012), voice_stream per-sample glides halved. Side effects
  measured: radiation tilt halves at speech frequencies (voice ~-6 dB, so
  frication and OUT_GAIN both re-seated), crest DOWN to 8-10x, clicks down
  to 2-8/take (all plosive-shaped, at boundaries).
- **Effort tilt floor raised** (0.3->0.45 base, 0.2->0.35 clamp): settled
  speech was being lowpassed into the reported dullness; the emphatic =
  brighter dynamic is intact.
- **FRIC_THRU 0.06 -> 0.18**: re-calibrated at 44.1k via pure_say
  (TARGET_RATIO now 0.35 - a step crisper, per the clarity request).
- **OUT_GAIN 2.0 -> 4.5**: re-staged by measurement against the 44.1k
  signal. Final: RMS -17.3/-18.8/-19.0 dBFS (4-6 dB louder), peaks steered
  to LIMIT, ramp work <=0.03% of samples, ZERO saturation.
- **Presence distance filter opened** (500 Hz floor -> 900, opens sooner):
  a not-fully-landed voice no longer strangled - distance still darkens.

voice_check ALL OK, compile clean, boot clean. NOTE: seeds will sound
noticeably different at 44.1k (per-sample rng sequences changed) - same
lineage still = same take, deterministically.

---

## 18. Round thirteen (2026-08-04) - the acoustic rebuild (Stage 1 of 3)

Prompted by a fresh report: dull and muddy, "Th" and "F" barely audible, "the
enunciation simply isn't there", too many words wrong in exported videos, and
the singing quality lost. A multi-agent investigation (7 dimensions, each
adversarially verified, plus an independent measurement pass) produced the
diagnosis; this section is the FIRST of three remediation stages. Stage 2 is
the text front end, Stage 3 is singing as a typed capability.

### What was actually wrong (measured, not inferred)

**The cascade was doing three jobs it should never have had.**

1. **Frication was routed through the vowel's formant cascade.** The `"fric"`
   branch was the only segment type that never called `_retarget()`, so during
   every fricative the 5-pole cascade held the PREVIOUS vowel's formants. The
   cascade is a DC-normalized all-pole chain topping out at 4.7 kHz, so it
   attenuated each fricative at its own noise band by 42 dB (/f/), 69 dB
   (/th/) and 85 dB (/s/). Measured consequence: the S/SH/F/TH log-spectra
   correlated at 0.99-1.00, three of four peaked in the SAME FFT bin (1367 Hz),
   and 6-10 kHz contrast against the neighbouring vowel was -0.2 dB. One sound
   wearing four labels. Both previously-tried routes were wrong for OPPOSITE
   reasons - the post-cascade band shared no tract shaping and read as injected
   static; the through-cascade path fused but was lowpassed into nothing.
2. **A DC pedestal held 43-52% of every take's power.** The Rosenberg pulse
   tables are unipolar (mean 0.404); every cascade pole passes DC at unity and
   the only rejection anywhere was the radiation zero at 1-0.96 = 0.04.
   Predicted output DC 0.0727, measured 0.0727. Inaudible, but it ate 0.07 of
   the 0.85 peak budget, was amplitude-modulated at the syllable rate, fed the
   leveler's envelope detector, and made every RMS staging number in the file
   ~2.4 dB optimistic.
3. **No top.** Five poles ending at 4.7 kHz put the -60 dB/oct cliff inside the
   speech band: 27 dB of drop between 3150 and 5000 Hz, and above 5 kHz the
   output was bit-for-bit the FLOOR_MIN dither. What brightness existed was
   breath/air hiss - 96-99% of a sustained vowel's 2-4 kHz energy.

Plus: obstruents had no tract posture at all, so /aba/, /ada/ and /aga/
produced bit-identical formant trajectories (no place cue); nasals shared one
fixed 1000 Hz zero, collapsing sum/sun/sung; fricative bandwidths were scaled
by vocal tract length alongside their centres, silently giving short tracts a
higher-Q /s/; and one `hiss` draw fed aspiration, the air band and the
constriction noise, so three nominally independent sources added coherently.

### Shipped

- **Parallel branch** (`phonemes.gd` `par`, `voice.gd` `FRIC_LEVEL`, `_tune_parallel`,
  `Reso.tune_peak`). Frication and bursts drive their own peak-normalized
  front-cavity resonators, summed with the cascade after the nasal zero and
  before radiation, signs alternating as Klatt's parallel branch does. This is
  the half of Klatt 1980 the file never had.
- **Mandatory tract posture on every TABLE row.** Fricatives and stops carry
  loci (labial 900, dental 1400, alveolar 1750, post-alveolar 2000, velar 1900
  Hz F2), so "a branch forgot to set the tract" is now unrepresentable.
- **Locus glide** (`LOCUS_TIME` 50 ms): a segment holds its own posture then
  bends toward the next one before the boundary. Retargeting only AT boundaries
  put the whole transition inside the following phone - which for a stop is the
  silent closure, so the listener never heard it.
- **Zero-mean pulse tables**, and the radiation coefficient derived from a named
  corner (`RAD_CORNER` 140 Hz) instead of a hardcoded 0.96 that silently
  doubled when SR went 22050 -> 44100.
- **F6 at 6000 Hz** and **Klatt 1980 per-formant bandwidths** (60/90/150/250/300/500)
  replacing `60 + F1*0.06` / `90 + F2*0.05`, which ran F2 at roughly twice
  natural width.
- **Per-phoneme nasal zeros** (M 950 / N 1800 / NG 3000).
- **Independent turbulence draw**; fricative bandwidths no longer tract-scaled.
- **Air band centre 0.07 -> 0.02**: it existed to give a topless cascade some
  brightness, and was measured MASKING /sh/ in its own 2-4 kHz band.
- **`FRIC_LEVEL` 0.05**, staged by measurement (see below).

### The calibrator was the reason this survived so long

`pure_say._ratio()` was broadband fricative/vowel RMS against `TARGET_RATIO`.
That number is orthogonal to a timbre collapse: run against the old synth it
converged, reported a healthy 0.35, and declared success on fricatives with
0.02-0.06% of their energy above 4 kHz. Two RCA rounds passed on a metric that
could not fail. Rebuilt as three gates that each catch a different failure:
in-band CONTRAST against the adjacent vowel (buried), broadband LEVEL (too
loud - the failure that recurred twice here), and pairwise spectral
DISTINCTNESS (collapsed). The contrast band is read from each phoneme's own
declared poles rather than fixed, and is gated only where it is diagnostic:
/sh/ sits ON F3/F4 by physical fact, so a threshold there would measure the
vowel, not the consonant - it is gated on level and distinctness instead,
where it scores -10.7 dB and -0.36 against /s/.

### Measured, before -> after (same prose, curated default speaker)

| measurement | before | after | natural |
|---|---|---|---|
| /s/ in-band vs adjacent vowel | -14.4 dB | **+19.9 dB** | +15..+25 |
| /f/ in-band vs adjacent vowel | -18.6 dB | **+11.7 dB** | +8..+15 |
| /th/ in-band vs adjacent vowel | -17.8 dB | **+14.1 dB** | +8..+15 |
| /s/,/f/,/th/ broadband vs vowel | - | -10.3 / -11.2 / -12.1 dB | -8..-15 |
| worst gated fricative pair correlation | 0.99-1.00 | **0.838** | - |
| S/SH correlation | ~0.94 | **-0.361** (anti-correlated) | - |
| 0-30 Hz share of take power | 39.0% | **0.0%** | ~0 |
| 2-4 kHz vs 5-8 kHz | +35.7 dB | **+5.0 dB** | +6..+10 |
| octave LTAS at 6k / 8k / 12k | -40.9 / -47.7 / -44.6 | -21.7 / -20.7 / -25.8 | -22 / -26 / -33 |
| /aba/ vs /ada/ vs /aga/ F2 track | bit-identical | **152-182 Hz mean abs delta** | - |
| /AA/ vs /IY/ vowel separation | ~x200 | **x8238** | - |

Gates: `pure_say` ALL OK, `voice_check` ALL OK, `sampler_check` ALL OK,
compile clean, headless boot clean.

### Known residuals (deliberately not chased further)

- 8-12 kHz still sits 5-7 dB above the natural reference - the broad /f/ and
  /th/ top poles. Narrowing them 3000 -> 2000 Hz bandwidth recovered ~4 dB at
  20 kHz and ~1 dB at 12 kHz; the rest would be tuning, not physics.
- 2-4k vs 5-8k lands at +5.0 dB against a +6..+10 natural window - close, and
  the residual is legitimate /s/ and /th/ energy in fricative-dense prose.
- 1 kHz is a few dB hot relative to the natural LTAS.

NOTE: every seed will sound different - the per-sample rng sequence changed
(an independent turbulence draw) and the acoustics moved substantially. Same
lineage still reproduces the same take, deterministically.

---

## 19. Round fourteen (2026-08-04) - "father" and "yours", and what they exposed

Ear report after §18 landed: better, but the F in "father" is completely
silent and the "rs" in "yours" is too. Two specific words turned out to
implicate FOUR separate defects, three of them introduced by §18 itself.
`tests/phone_dump.gd` (new) is the tool that made this tractable: it prints
every phoneme of an utterance with its position, duration, broadband RMS and
three band energies, so a word reported by ear becomes a measurement.

### The DSP half (three bugs, all mine from §18)

1. **The forward locus glide was erasing the cue it existed to create.** A
   voiceless fricative is silent through the tract, so gliding its posture
   toward the next vowel spends the transition where nobody can hear it and
   then hands the vowel over already sitting on its own target. Measured on
   "father": /AE/ began at F2 1720 (its own target) with no transition at all.
   Voiceless obstruents now HOLD their locus and let the following vowel carry
   the movement. After: /AE/ onset F2 1394 -> steady 1701, a real labial
   transition. For /f/ and /th/, whose own noise is weak, that transition IS
   the primary place cue - which is why "silent" was a fair description of a
   phoneme that was measurably present as hiss.
2. **The glide window ate short segments.** `min(LOCUS_TIME, n*0.5)` let a
   97 ms /r/ spend 48 ms travelling away from the low, close-spaced F2/F3 that
   defines an /r/, and with the 32 ms glide EMA lagging on top it never
   arrived. Capped at `LOCUS_SHARE` 0.35. After: /r/ reaches F2 1125 / F3 1378.
3. **Klatt's sign alternation was wrong for the diffuse fricatives.**
   Alternating the parallel poles stops a null forming between narrow,
   separated resonators (/s/, /sh/) but /f/'s poles at 1400 and 4800 Hz are
   1600 and 2800 Hz wide - they overlap almost entirely, so opposite signs made
   them CANCEL. Caught because raising /f/'s shoulder amplitude made it
   QUIETER. Sign is now data (a negative amplitude in the table), not a rule.
   /f/ 0-1 kHz improved 6 dB and the F/TH spectral correlation fell 0.911 ->
   0.615. Its low shoulder was also restored (0.15 -> 0.30) - the §18 level
   pass had thinned /f/ and /th/ into pure high hiss with no body.

### The front-end half (and the opening of Stage 2)

Both reported words were ALSO mispronounced, independently of any DSP:

    father -> F AE TH ER      (voiceless th)
    yours  -> Y AW R S        (wrong vowel, unvoiced suffix)
    comes  -> K AA M EH S     (phantom syllable)

Four rules shipped, all in `phonemes.gd`:

- **`-s` morphology**, one level: strip a final plural / third-person `-s`,
  pronounce the STEM, reattach with voicing assimilation (Z / S / syllabic
  IH Z). Going through the stem also lets the exceptions dictionary and
  magic-e do their work, so one pass fixes the vowel, the consonant and the
  spurious syllable together. The strip test is orthographic and deliberately
  conservative - a wrong strip invents a word, which is worse than missing one.
- **`th` voicing**: function words (`the`, `them`, `then`, `than`, `those`),
  intervocalic (`father`, `mother`, `together`), and the `-the$` spelling
  (`bathe`, `breathe`). Was unconditionally voiceless.
- **General silent final `e`**, which magic-e could not reach because it only
  inspects single letters and these have a digraph in the slot it checks.
  Soft c/g are now tested against the ORIGINAL spelling, since the dropped `e`
  is exactly the letter that softens them.
- **Word-final `y`** (AY in a monosyllable, IY otherwise) and **word-final `o`**,
  plus a `HARD_G` list for the common Germanic words the soft-g rule broke.

Measured:

    father -> F AE DH ER      mother -> M AA DH ER     together -> T AA G EH DH ER
    yours  -> Y AO R Z        comes  -> K AH M Z       gives    -> G IH V Z
    goes   -> G OW Z          wishes -> W IH SH IH Z   bathe    -> B AE DH
    large  -> L AA R D ZH     every  -> EH V ER IY     try      -> T R AY
    thin/path/both -> TH (correctly voiceless)   bus/gas/yes -> unstripped

### Known residuals

- No vowel reduction or stress, so `father` keeps AE where it wants AA and
  `village` takes a magic-e EY where it wants an unstressed IH. This is the
  syllabification + stress stage, still ahead.
- `change`/`danger` read the `ng` digraph and lose the soft g. Not guessed at:
  the same spelling is genuinely NG in `singer`, `longer`, `finger`, `anger`,
  so fixing one breaks the other without morphology.
- LIVE-ONLY RISK, unverified: `voice_stream.gd:359` applies a distance lowpass
  at `900 * 2^(4*presence)` Hz whenever presence < 0.995. At a half-landed
  presence that is ~3.6 kHz, which would remove /f/ and /s/ entirely while
  leaving vowels intact - exactly the reported symptom. It is bypassed at full
  presence and converges in about a second, so it is probably not what was
  heard, but any consonant report from inside the fishing loop should check it.

Gates: `pure_say`, `voice_check`, `sampler_check`, `loader_check` all OK;
compile clean; headless boot clean.

---

## 20. Round fifteen (2026-08-04) - the front end becomes data

Report: still only ~50% intelligible without subtitles. Correct - and the
measurement says the remaining barrier is the FRONT END, not the acoustics.
Graded against a hand-written reference over the vocabulary of a real chapter
(`rift/books/north-star/chapters/01-reruns.md`, 1530 tokens, 482 types), the
letter rules were mispronouncing words that carry most of the listening load.

### Why this became the architecture work too

The fix list was almost entirely DATA, not code: `we`/`me`/`he`/`she`/`be` all
read with EH, `head` as HH IY D, `cushion` as K AH SH IH AA N, `watched` as
W AE T SH EH D, `couldn't` as K AW L D N T. Letter rules cannot reach any of
these, and English's most frequent words are also its most irregular. So the
typed-pipeline question answered itself: the front end now loads its language
from disk.

**`data/english.yml`** - external, MiniYaml, three sections:
- `lexicon` (420 entries): the irregular high-frequency core, organised by the
  RULE CLASS each entry defeats (`ea` as EH, `ou` as AH, `oo` as UH before k/d,
  `ea`/`ai` before r as EH, unstressed `-or` as ER, `-all`/`-old`, syllabic
  `-le`, `-ion`, compounds). Grouped that way so the file reads as an argument
  rather than a word list, and so the next person can see what is missing.
- `clitics`: contractions split at the apostrophe, head pronounced through the
  same pipeline. Heads that change shape (`don't`, `won't`, `can't`) are whole
  lexicon entries instead - `do` + `n't` gives D UW AH N T.
- `suffixes`: `-ing`, `-ed`, `-ly`, `-ness`, `-ment`, `-ful`, with undoubling
  (`stopped -> stop`) and e-restoration (`hoping -> hope`).

The pipeline order is now lexicon -> clitic -> suffix -> letter rules, and the
suffix stage is the one that needed the typing: whether the `-ed` of `watched`
is a syllable, a /t/ or a /d/ depends on the last PHONE of the stem, so the
stem must be pronounced BEFORE the suffix can be chosen. A flat letter table
cannot express that dependency, which is exactly why it produced a phantom
syllable on every past-tense verb in the book.

### The measurement, which is the real deliverable

`tests/g2p_check.gd` grades against `data/reference.yml` - hand-written from
knowledge, independent of what the front end produces, never a snapshot of its
output - weighted by TOKEN frequency in the chapter. Type-level error rates
flatter the system badly: an error on `the` costs 130 times what an error on
`syndicate` costs.

    before this round:  token-weighted WER ~35% (measured in the §18 audit)
    after:              token-weighted WER 0.0% over the graded set
                        (110 types, 1054 of 1530 tokens = 69% of the text)

Also new: `tests/g2p_dump.gd` (every distinct word of a text with its phonemes,
ordered by corpus frequency) and `tests/phone_dump.gd` from §19. Between them a
word reported by ear becomes a measurement in one command.

### Honest limits

- The graded set is the top 110 types. The other 31% of tokens are the tail,
  where the error rate is real but unmeasured. Spot-checking it drove the last
  batch of lexicon entries (`good`, `air`, `swear`, `actor`, `history`,
  `sometimes`), and there will be more.
- **No stress and no vowel reduction yet.** Every syllable still gets its full
  vowel and roughly equal weight, so the reading chants. This is the single
  largest remaining intelligibility AND naturalness item: English listeners
  segment running speech by stress pattern, so flat stress costs word
  boundaries, not just naturalness. It is also what the user meant by "the same
  word might sound different at different positions".
- `change`/`danger` still lose their soft g to the `ng` digraph, unfixable
  without morphology (`singer`, `longer`, `finger` are genuinely NG).
- Two MiniYaml constraints the data file has to respect, both found the hard
  way: duplicate keys are rejected outright, and a bare key containing an
  apostrophe fails to scan - contraction keys must be quoted.

Gates: `g2p_check`, `pure_say`, `voice_check`, `sampler_check` all OK; compile
clean; headless boot clean. Listen: /tmp/ghost_scratch/chapter01.wav (the
chapter's opening through the current chain).

---

## 21. Round sixteen (2026-08-04) - CMUdict, and the stress that came with it

Report: no major difference from §20, still muddy. Correct, and the reason is
that §20 fixed WHICH phonemes get spoken without touching HOW LONG each one
lasts. Every syllable still arrived at roughly equal length and full vowel
quality, so the reading chanted - and when the phonemes are already right, that
is most of what "muddy" means.

The user also asked whether hand-building a lexicon was making this harder than
it needed to be. It was.

### CMUdict, vendored

`data/cmudict.dict` - 126,052 words, BSD-2-Clause, attribution in
`data/cmudict.LICENSE`. This is a DICTIONARY, not a model: the same kind of
1980s data table the phoneme inventory already is, and it does not touch the
no-generative-AI / no-recordings constraints. Measured cost: 3.3 MB, 117 ms to
read and index once at first speech.

It beat the hand-written lexicon on every word I spot-checked. But the reason
it matters is not coverage - it is the STRESS DIGITS:

    father    F AA1 DH ER0
    cushion   K UH1 SH AH0 N
    because   B IH0 K AO1 Z
    photograph  F OW1 T AH0 G R AE2 F
    photography F AH0 T AA1 G R AH0 F IY0

That last pair is the thing the user described as "the same word sounding
different at different positions" - the stress MOVES, and everything about the
rhythm follows it. A hand list could have supplied pronunciations; supplying
correct lexical stress for 126k words by hand was never going to happen, and
stress is the input the rhythm work needed.

Lookup order is now: `data/english.yml` overrides (where we disagree on
purpose) -> CMUdict -> clitics -> suffixes -> letter rules. The hand-written
lexicon stays as the override layer and as the fallback if the dict is absent.

### What the stress marks bought

- **Accent placement.** `stress_vowel()` returned the FIRST vowel of a content
  word; it now returns the PRIMARY-stressed one. The earlier audit measured the
  old behaviour wrong on ~32% of multisyllabic content tokens.
- **Vowel reduction, per syllable.** Unstressed vowels are now 0.62x duration,
  0.78x amplitude and 0.7 toward schwa; secondary stress 0.88x / 0.92x / 0.25.
  Reduction used to fire only on whole function WORDS.
- **A real schwa to reduce toward.** `_SCHWA` was `[640, 1190, 2390]` - byte
  identical to AH's own formant target - so reducing an AH toward schwa moved
  it nowhere, and AH is the vowel English reduces to. Now `[500, 1500, 2500]`,
  a genuinely central vowel.

Measured on the chapter's opening:

    stressed content vowels    195-393 ms at -12 to -16 dB
    reduced function vowels     72-79  ms at -21 to -25 dB

    vowel duration CV        0.55   (natural English running speech: 0.50-0.60)
    vowel duration p90/p10   3.4x   (natural: 3-4x)

Total take duration barely moved (89.2 s -> 89.0 s), which is the correct
result rather than a null one: English redistributes time between syllables at
a roughly constant rate, it does not spend more of it.

### Honest limits

- Take-level envelope dynamics moved only 7.6 -> 7.8 dB sd. The rhythm change
  is in DURATION, which that statistic does not capture; do not read it as the
  change having failed, and do not read the duration numbers as proof the ear
  will agree either. This one is for the ear.
- `yours` now reads Y UH R Z (CMUdict's rhotic form) against Y AO R Z in the
  reference; both are current English, and the gate passes at 0.4%.
- Still no PHRASE-level prosody: no phrase-final lengthening beyond the
  existing terminal stretch, no de-accenting of repeated information, no
  emphasis driven by the sentence's own structure. That is the next layer of
  "the same word sounds different at different positions" and it is not done.
- Everything here is LEXICAL stress. Sentence stress (which word in the clause
  carries the nuclear accent) is still the walk's seeded guess, not analysis.

Gates: `g2p_check` 0.4% token WER, `pure_say`, `voice_check`, `sampler_check`
all OK; compile clean; headless boot clean.
Listen: /tmp/ghost_scratch/chapter01_stress.wav against chapter01.wav (the same
excerpt before the stress work).

---

## 22. Round seventeen (2026-08-04) - the walk stops inventing prosody

The user's question, and it was the right one: use the dictionary's stress as
the BASELINE and apply the walk as a modulation on top, rather than letting the
walk generate the prosody itself.

That is exactly what was wrong. §21 wired lexical stress into duration and
reduction, but the pitch ACCENT was still this, in `ProsodyWalk.word()`:

    if _gate.randf() < 0.22 * appetite + (0.18 if frac > 0.7 else 0.0):
        emph = clampf(appetite * (1.0 - spent), 0.4, 1.4)

A coin flip that both INVENTED the accent and PLACED it. The same sentence
stressed different words on different seeds, and no seed stressed them where
English does. The walk was generating prosody when it should only have been
colouring it.

### `scripts/phrasing.gd` - sentence stress from the text

A new typed stage between [Phonemes] and `plan()`. Nothing in it is seeded: the
same text always gets the same phrasing, and the voice's temperament decides how
HARD it leans, never WHERE. Four parser-free rules:

1. **Content vs function** - function words carry no accent. The list moved to
   `data/english.yml` and grew from 33 entries to ~100; `has`, `will` and `we`
   were being read as content words and taking accents off the verb. Function-word
   CONTRACTIONS (`we're`, `it's`) resolve through their head.
   Deliberately NOT function words: the standalone possessives (`mine`, `yours`,
   `ours`) - they are predicates that carry focus, and listing them moved the
   nucleus of "the far end is YOURS" onto `end`.
2. **The nuclear accent** - the last content word before a boundary carries its
   phrase's main accent. The single most audible prosodic event in a clause, and
   it was entirely absent before.
3. **De-accenting given information** - a content word said in the last ~14 words
   is old news and drops to 0.45x. This is why "I bought a CAR. The car was RED"
   does not stress `car` twice.
4. **Stress clash** - two full accents in adjacent words is not English rhythm;
   the earlier one steps back.

Measured on the chapter's own sentences (`tests/phrasing_dump.gd`, new):

    in the BEGINNING was the [COUCH,] and the couch(0.31) was with [FATHER.]
    it's not the KIND of WRONG i can [EXPLAIN.]
    the FAR END is [YOURS.]

The second `couch` de-accents to 0.31 on its repeat; the nuclei land where a
reader would put them.

### The inversion in the walk

`word()` now takes `prominence` and returns `emph = prominence * appetite *
(1 - 0.55 * spent)`. The spent-emphasis EMA still spaces the big leans out, but
it can no longer DELETE an accent the sentence structure requires. The seeded
extra push survives as a multiplier on words that are already prominent - the
speaker's own reading of an emphatic word, not the invention of one. In `plan()`
the accent's SIZE now scales with prominence, so a nucleus lands hard and a
de-accented repeat barely rises; every content word taking the same full accent
is a list being read, not a sentence.

### Honest limits

- Take-level envelope statistics moved very little across all three rounds
  (7.6 -> 7.8 -> 7.4 dB sd; total duration 89.2 -> 89.0 -> 86.8 s). The change is
  in WHERE the length and pitch go, not how much there is of either, and this
  statistic cannot see that. It is not evidence the change worked; the phrasing
  dump and the ear are.
- **Contrastive focus is not attempted.** "I said the RED one" needs syntax or an
  author's mark. If a sentence's real focus is not its last content word, this
  stage will put the nucleus in the wrong place, confidently.
- No syntactic phrasing: boundaries come from punctuation only, so a long
  unpunctuated clause gets one nucleus at its end rather than being broken into
  intermediate phrases the way a reader would.
- Rule 3 keys on the exact surface word, so `couch`/`couches` and `is`/`was` do
  not count as repeats.

Gates: `g2p_check` 0.4% token WER, `pure_say`, `voice_check`, `sampler_check`
all OK; compile clean; headless boot clean.
Listen: /tmp/ghost_scratch/chapter01_phrased.wav against chapter01_stress.wav
(lexical stress only) and chapter01.wav (neither).

---

## 23. Round eighteen (2026-08-04) - the override layer was a silent regression

User's question: are `english.yml` and `reference.yml` redundant now that
CMUdict is in? Measured: **387 of 414 lexicon entries were byte-identical to
CMUdict**, and 105 of 110 reference entries were too. But the redundancy was
not the problem - it was the symptom.

### The bug the redundancy was hiding

`english.yml` is the OVERRIDE layer: it wins over CMUdict. Its entries carried
no stress digits, so they went through `_with_default_stress`, which marks the
FIRST vowel primary. For 56 of those words the primary stress is not on the
first vowel, so the override was actively mis-stressing them:

    about    should be AH0 B AW1 T   -> stressed A-bout
    because  should be B IH0 K AO1 Z -> stressed BE-cause
    before, between, again, another, believe, begin, afternoon, ... (56 total)

Every one is high frequency. §21 wired stress into the rhythm and §22 built the
phrasing on top of it, and this layer was quietly feeding both of them wrong
stress for the commonest polysyllables in the text. A redundant override is not
free; it is a regression with no symptom until something starts reading it.

### What `english.yml` keeps

Four lexicon entries: two homographs where the system must pick a reading
without part-of-speech tagging (`live`/`lives` as the verb, which is commoner in
prose than CMUdict's adjective), and two words CMUdict does not contain. The
suffix, clitic and function-word tables stay - CMUdict does not supply those.
Overrides may now carry stress digits, and must when polysyllabic.

**Reduced narrator forms were tried here and removed on principle.** CMUdict
lists function words in citation shape (`has` as HH AE1 Z), and the fix is to
strip the STRESS, not rewrite the vowel: `function_words` forces stress 0 and
the existing per-syllable reduction then centralizes and shortens the vowel at
synthesis time. Spelling `HH AH0 Z` into the lexicon would duplicate machinery
that already exists and freeze one degree of reduction for every speaker and
tempo.

### A second bug that fell out of testing it

Stress-flattening applied to ALL function words, so `about`, `between`, `under`
and `after` came back with no stressed vowel at all. English reduces the
MONOSYLLABIC function words; polysyllabic prepositions keep their internal
stress and simply take no sentence accent - which is Phrasing's job and already
handled. Now gated on syllable count.

### `reference.yml` repurposed

Grading dictionary words against a hand-written list mostly measured "did the
dictionary load". It now grades the paths CMUdict does NOT answer, in four
groups - `words` (a smoke set incl. our overrides), `derived` (suffixes,
clitics, plural voicing), `fallback` (out-of-dictionary), and `stress` (the
index of the primary-stressed vowel, `-1` for a reduced function word).

That last group is the important addition: **nothing in the suite tested stress
before**, and stress is what the whole rhythm now hangs off. It is also what
caught both bugs above. The token-weighted WER number is gone with the old
file; it had become a measure of CMUdict's coverage rather than of our work.

    g2p_check: words 9/9   derived 9/9   fallback 2/2   stress 17/17

`english.yml` 414 lexicon entries -> 4. Gates: `g2p_check`, `pure_say`,
`voice_check`, `sampler_check` all OK; compile clean; boot clean.

---

## 24. Round nineteen (2026-08-04) - the staircase, recovered from git

User: nothing sounds like singing, the text is never slow enough, a cadence has
speed-ups and slow-downs, and the earliest versions could do it - go look.

### What the archaeology found

`git show 11d3c2a8:axis/ghost/scripts/voice.gd` (the first commit, 2026-07-18,
386 lines). Its entire melody was one line:

    "semitones": decl + accent,

Every phone got the sentence's declination; only the stressed vowel got the
accent bump; silences reset to 0.0. No wander, no attractor shelf, no walk, no
continuity pass. The contour was a STAIRCASE - discrete pitch levels, held flat
across each segment, with instant transitions and a hard reset at every pause.
That is what sounded like singing, and it was an accident of the code being
unfinished.

Every naturalness fix since has sanded it off, and the biggest was the f0
continuity pass in fcb4f58f, which lerps every consonant onto the line between
its neighbouring vowels. That turned the steps into a continuous glide - a
correct fix for the "picket fence" complaint it was written for, and the thing
that made singing unreachable.

### `song`: the ninth trait axis

Not a style layer over speech - it swaps four behaviours at once, because that
is what the difference between speaking and singing actually is:

- **Notes.** A sung vowel takes a whole number of BEATS rather than its natural
  length, and a prominent syllable takes more beats than a weak one
  (`1 + round(prom * 2)`). That is where the cadence comes from: uniform
  stretching gives a drone, integer beats give long-short patterns that speed
  up and slow down against a pulse. The beat follows the speaker's own tempo
  and drawl, so it needs no separate roll.
- **Steps.** The continuity glide is gated by `1 - song`, so at the top of the
  axis a consonant HOLDS the note it is inside instead of gliding toward the
  next one. This is the original staircase, restored as a capability rather
  than an accident.
- **Scale.** The anchor-shelf pull goes to 0.95, engaging fully by song 0.5.
  Quantization is the defining feature, so it does not fade in; what fades in
  along the axis is note LENGTH.
- **Vibrato.** 5.5 Hz, 0.38 st, ramped in over 180 ms so short notes do not
  wobble. Without it a held quantized note reads as a robotic tone, not a sung
  one. The ProsodyField wander is also scaled by `1 - song`: a held note has to
  be still for the vibrato to read as vibrato rather than drift.

Only the upper half of the trait axis sings, so the curated default and half of
every roll stay pure speech and a singing voice is something you FIND.

Separately, and for speech too: `rate` widened from `2^(0.35*pace)` to
`2^(0.55*pace)`. The slow end could only reach 1.28x slower than neutral, which
is not slow enough for a deliberate reading.

### Measured (`tests/song_check.gd`, new)

    song   median note   p90     on-note   |df0/dt|
    0.0        71 ms    143 ms    35.2%    382 cents/s
    0.5       456 ms   1161 ms    54.1%    103 cents/s
    1.0       933 ms   1911 ms    61.2%     85 cents/s

Speech vowels were measured at a median of 104 ms with only 0.17% reaching
400 ms, which is why no seed could ever sing however it was rolled. 61.2% of
voiced time on-note matches the 59.7% the earlier counterfactual predicted for
a forced pull, so the quantizer is now doing what the documentation always
claimed it did.

### Three bugs the measurement caught

- **The metric measured the vibrato and called it bad tuning.** A +-38 cent
  waver spends most of its cycle outside a 25 cent window, so a perfectly
  quantized note scored ~40%. The tuning check now smooths over one vibrato
  period first.
- **The drone bank was detuned from the melody**, as suspected in §21 but not
  then fixed. Strings were tuned to the raw anchor while the melody realizes it
  at `anchor * inflect`, so any voice with `inflect != 1` had its drone a
  growing interval out of tune with the note it was answering - and the
  proximity test compared inflect-scaled units against raw ones. Both now use
  the realized pitch. This matters far more for a singing voice, since the
  drone is tuned to the same shelf the melody now sits on.
- The synth realizes `... * 1.06`, a fixed +1.01 st offset that has to come off
  a tracked f0 before it means anything against the shelf. Without it the
  tuning number was meaningless.

### Honest limits

- Intelligibility is unchanged by any of this. `song` makes a voice sing; it
  does not make a spoken reading easier to follow, and the standing complaint
  (~50% without subtitles) is NOT addressed here.
- At song 1.0 the notes are long (median 933 ms, max 2.16 s). That is the top
  of the axis and meant to be extreme, but a chapter read at 1.0 would take
  hours.
- Vibrato is a fixed rate and depth. Real vibrato varies with pitch, loudness
  and effort, and accelerates slightly into a held note.
- The beat grid quantizes DURATION but nothing aligns notes to a shared bar
  line, so there is no metre - phrases do not start on a downbeat.

Gates: `g2p_check`, `pure_say`, `voice_check`, `sampler_check` all OK; compile
clean; boot clean. Listen: /tmp/ghost_scratch/song_0.wav, song_5.wav,
song_10.wav - the same sentence at three points along the axis.

---

## 25. Round twenty (2026-08-04) - singing gets a cadence, and stops being the default

Two reports after §24 shipped `song`: ~90% of fished seeds were singers, and the
singing voices held a near-constant cadence, turning a 7 minute read into 15.

### Why almost everything sang

`Spec.sample` gave `song` the same `randfn(0, 0.55)` as the timbre axes, so half
of all WILD rolls were above zero before the belt touched them. Then the belt
compounds it: `_pick_parent` is acceptance-weighted, children inherit with
generation-decaying jitter, and `_temper_traits` pulls every draw toward the
party's centre. One kept singer drags its whole line across the line, and the
trust region then holds it there.

`song` is a MODE, not a shade, so it is now drawn with an explicit incidence
(`SONG_INCIDENCE` 0.28) and the negative draw is pushed clear of zero
(`[-1.0, -0.25]`) so a non-singer's children do not cross by jitter alone.
Measured over 4000 wild rolls: **29% sing at all, 17% sing strongly.** The roll
has to sit below the rate you want to meet in play, because the belt only ever
amplifies what it is given.

### Why every sung word was slow

The beat grid stretched EVERY vowel onto 1-3 beats. That is a drone, not a
cadence - and it is the same mistake as reading every content word with a full
accent. Music alternates: notes are HELD at the joints of a phrase and RUN
through in between, and the contrast IS the music. The user's framing, which is
the right one: "you could be rapping one minute, and singing the next."

**The sustain gate.** A syllable is held when a slow seeded cycle over the
syllable count, its prominence, and whether it ends a phrase add up past
`SUSTAIN_BAR`. Same thresholded-drive idiom as the activation channels, so it is
sparse and self-spacing by construction rather than by a rate constant. The
cycle's period and phase are derived from `drawl`, `pace`, `lilt` and `grit` -
traits the speaker already has - so a drawling singer holds notes further apart
than a brisk one and no new roll is needed. A syllable that is NOT held runs at
`SONG_RUN` 0.72 of its spoken length, floored at `SONG_RUN_MIN` 75 ms.

Two things fell out of that and both had to be fixed:

- **Runs were too fast to have a pitch.** At 49 ms the f0 EMA (35 ms) never
  reached target, so the fast passages smeared off-key and the tuning number
  fell straight back to speech levels. A singer ARRIVES on a pitch where a
  speaker glides onto it, so the f0 time constant is now `lerp(35 ms, 12 ms,
  song)` - which is also the original staircase's character.
- **The terminal contour was knocking the held notes off the scale.** The
  sentence-ending fall and the comma rise are applied AFTER the anchor pull, and
  they land on the longest vowels in the reading - which, when singing, are
  exactly the syllables the sustain gate chose, because a phrase ending is one
  of the things that makes a syllable worth holding. So the one note a listener
  had the best chance of hearing as a pitch was guaranteed to sit 1 to 6.5
  semitones off the shelf. `_resnap` re-quantizes after the contour, which turns
  a fall into a fall TO A NOTE - which is what a cadence is. This was logged as
  a MINOR finding back in the §21 investigation and was not minor at all once
  the sustain gate started choosing the same syllables.

### Measured

    song  median  p90     on-note  held-note  |df0/dt|  total  spread  sustained
    0.0     71 ms  143 ms   35.2%    38.5%    382 c/s    5.8s  CV 0.49    0%
    0.5     74 ms  691 ms   57.8%    80.0%    227 c/s    7.1s  CV 1.35   10%
    1.0     72 ms 1175 ms   74.2%    91.6%     90 c/s    8.6s  CV 1.75   15%

"held-note" is tuning restricted to frames that are actually still (under
50 cents/s). Once most syllables became short runs, the overall figure was
dominated by transitions, which is not where a tune lives - and it was the
metric that exposed the terminal-contour bug: 34.4% before the re-snap, 91.6%
after.

Total length is now 1.22x speech at song 0.5 and 1.48x at 1.0, against roughly
2x before, and the syllable spread went from CV 0.49 (speech) to 1.75. Long
notes at the joints, runs in between.

### Honest limits

- The sustained fraction is 10-15%, chosen by the drive threshold rather than
  targeted. It is an allowance, as asked, not a rule - but nothing guarantees a
  particular density, and a text whose prominences fall out of phase with the
  cycle will hold fewer notes.
- Still no METRE: the beat grid quantizes duration, but nothing aligns notes to
  a shared bar line, so phrases do not start on a downbeat and two voices
  reading together would not agree on a pulse.
- Intelligibility is untouched by any of this.

---

## 26. Round twenty-one (2026-08-04) - the fishing field: audit, and two fixes

User's report: seeds stop changing as you drift further out, and the belt's
"interference" does not behave like interference. Audited by three independent
agents plus skeptics. Their findings, after verification:

1. **Distance is a clock, not a position.** `_drift_dist` is a scalar odometer
   integrated from time and velocity; the outward leg reads nothing about
   traits, genome or belt, and has no direction. `_cast_vector` DOES move the
   candidate through the real 25-D space as the odometer grows, but its only
   consumer is the moon-phase drawing. So "distance" and "interference" are not
   connected today.
2. **Every gameplay consumer of drift saturated at odometer 1.0**, which the
   warp crosses in ~16 s. Past that only the caption changed. This alone is the
   reported symptom, and it was a clamp rather than a physics.
3. **The anneal cannot express interference in principle.** `_adreno_step` is a
   sum of linear springs, which reduces exactly to `W * (tbar - x)` - one
   attractor and one stiffness however many seeds there are. It also collapses
   harder as the belt fills: caught-trait spread falls 0.152 -> 0.012 per axis.
4. **Its force is distance-INDEPENDENT** - a `(m - x)` term means a member
   sitting FAR from the candidate pulls HARDER, the exact inverse of the intent.
5. **An interference field already exists** (`_field_reception`): per-source
   exponential falloff, amplitude-weighted centre frequency, pairwise beats,
   with per-seed strength from acceptance and per-seed frequency from pitch.
   But it drives the audio reception filter only, never traits or genome; its
   source positions are lineage HASHES rather than the seeds' coordinates; it
   has one shared falloff length so no seed can be long-range and another
   short-range; and its phase is wall-clock times array indices.
6. **The trust region widens with drift but its centre never moves**, so
   distance only inflates a cage still bolted to the belt's centroid.
7. **The metric was broken before any field could sit on it.** `_seed_vector`
   normalized each genome delta by `|prior|`, so a gene's weight was inversely
   proportional to how big its numbers happen to be: `hesit_bias` (prior 0.25)
   counted 38x more per unit than `breath_span` (prior 9.5) and carried 31% of
   all variance alone, with effective dimensionality ~7.6 of 25.

### Shipped now (the two cheapest, and #7 is a prerequisite for everything else)

- **`_seed_vector` normalizes by each gene's own G_BOUNDS spread.** Every
  distance, bearing, acceptance-weighted centre and the catch card's kinship
  colour were mostly reading one arbitrary gene. G_BOUNDS is the range the gene
  is actually sampled and clamped over, so it is the honest unit.
- **`_drift_free()`**: `log1p(_drift_reach())`, unbounded but soft-compressed,
  now feeding the three quantities that should keep growing with distance - the
  wild-throw chance, the child jitter, and (through them) how far a throw lands
  from the belt. Near field is unchanged in feel (log1p is ~x for small x); a
  long haul keeps paying with diminishing returns instead of hitting a wall.
  Reward, the reception band and anything laid out against a finite on-screen
  line deliberately still read the clamped `_drift_norm`.

### The field itself - designed, NOT built

Full design note in the workflow output. In brief: run the field in the existing
3-axis frame (brightness / damage / drive) extended with `song` and the genome
folded through a baked hyperplane bank - justified by measurement, not taste
(a random cast's projection onto a source's wave vector has sd 0.58 in 3-D and
0.20 in 25-D, where 90% of casts land inside |cos| < 0.33 and angle stops
discriminating). Per seed, all DERIVED rather than rolled: position from
`_seed_vector`, strength from acceptance, range from `ring * (1 - damp)`
normalized against the belt's median pairwise distance (~40x span, so
long-range beacons and point-blank seeds fall out of genes that already mean
reach and absorption), wave vector and phase from the seed's strongest
`_lineage_mods` entry. Field:

    a_i  = A_i * exp(-d_i / L_i)
    th_i = dot(k_i, p - x_i) + phi_i
    M    = sum_i a_i * cos(th_i) * unit(x_i - p)

`k_i` is a vector, so phase depends on distance AND angle and sources reinforce
or cancel by geometry. `cos` has zero spatial mean, so `M` perturbs without
biasing toward the centroid - the structural guard against the collapse failure
mode. Migration is seven steps, each leaving a working game, starting with a
standalone `scripts/voice_field.gd` plus `tests/field_check.gd` and no call
sites. Do NOT ship the later steps before the metric fix, or the field is a
field over `hesit_bias`.

Noted as possibly the larger share of the complaint, and a separate mechanism:
`Spec.influences` - the pooled-oscillator blend that IS the belt's audible
alchemy - is assigned in exactly one place (`export_take`) and never on a live
path, and an accepted catch's `override` genome replaces the blend outright. The
alchemy layer switches itself off the moment you catch anything.

Gates: `voice_check`, `g2p_check` OK; compile clean; synthesis-mode boot clean.
