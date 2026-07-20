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
