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
