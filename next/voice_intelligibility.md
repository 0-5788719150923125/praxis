# Voice: the intelligibility program

Written 2026-08-08 from a six-audit investigation of the current engine. Every number below was
measured against the code as it stands, not estimated. File references are `file:line` into
`axis/ghost/`.

The complaint that started this: "the voices are essentially uninterpretable unless you're reading
along with the subtitles; I would be lucky to understand even 50%." The goal: narrate 35 chapters
into video, with the reading voice bred in the fishing game.

---

## 1. Diagnosis

The engine's *phonetic knowledge is sound*. `phonemes.gd`'s TABLE carries textbook Klatt values:
sibilants peaked high (/s/ at 5300 and 7400 Hz, /sh/ at 2400 and 3600), /f/ and /th/ flat and
diffuse in 4-10 kHz, per-place burst spectra for the stops, per-place nasal zeros. The comment at
`phonemes.gd:80-82` states the correct design principle: for the weak fricatives "the cue is
spectral CONTRAST against the neighbour, not level."

Almost nothing downstream honours that. The damage is in the realization, the effects, and the
front end. Ranked by share of the intelligibility loss:

### Cause 1 - non-phonemic hiss masks the entire consonant band (~40% of the problem)

`spec.breath` (aspiration, default 0.05) and the `air` band (default 0.02, highpassed at
`air_cut` 3000 Hz) are injected into the source BEFORE the cascade, so the formants shape them
exactly like the voice and they land on top of it.

| band | share of vowel power supplied by hiss |
|---|---|
| 1.5-2.5 kHz | 70.8% |
| 2.5-4 kHz | 90.1% |
| 4-6 kHz | 84.9% |

Consequence, measured directly by rendering with and without:

| contrast, 2.5-5 kHz | with hiss | without | delta |
|---|---|---|---|
| fricative vs vowel | -4.05 dB | +5.96 dB | +10.02 |
| stop burst vs vowel | -21.18 dB | | ~+10 |

A fricative is currently **quieter than the vowel beside it, in its own cue band**. That single fact
explains the reported symptom better than anything else found.

Category: wrong constants plus a routing choice. Cheap.

### Cause 2 - the consonant level column does not control level (~25%)

`namp` in the TABLE scales resonators that are **peak-normalised**, while delivered power goes as
`sum(a_k^2 * BW_k)`. So the column the author used to balance the inventory does not do that:

- 4.4 dB of `namp` span across the six stops produces **16.5 dB** of delivered burst span.
- The four voiceless fricatives land within **3.2 dB** of each other. Natural spread is 10-18 dB.
- /sh/ (`namp` 1.8, deliberately the highest) delivers **3.2 dB quieter than /s/**. Natural /sh/ is
  3-6 dB *louder*.
- Stop bursts sit **24-37 dB** under the adjacent vowel. Natural is 8-15 dB.

Then the whole parallel branch passes through one global `FRIC_LEVEL := 0.05` (`voice.gd:188`,
applied once at `voice.gd:1924`), which is -26 dB on every fricative and every burst in the
language.

Category: wrong scaling law. Cheap to correct, needs a small derivation at table-load time.

### Cause 3 - place of articulation is not transmitted (~20%)

- **All six stops share one formant onset.** `voice.gd:1328` retargets the cascade to the FOLLOWING
  vowel before the release run, so the locus excursion is spent before the burst occurs. Measured
  onset-F2 spread across the three voiceless places: **35 Hz** in /AA/, 22 Hz in /UW/. Natural is
  600-800 Hz. /b/=/d/=/g/ and /p/=/t/=/k/ acoustically.
- **/b d g/ have no release event at all.** `voice.gd:1331` gates the VOT run on voiceless only, so
  a voiced stop is 45-48 ms of closure, a 9 ms burst at -42 dB, then the vowel. 9.3% of all phones.
- **VOT is 24-28 ms for /p t k/**, sitting on the 25-35 ms English category boundary; aspirated
  stops should run 50-90 ms. Voicing contrast is at chance.
- **Voiced fricatives are inverted**: voice bar exceeds frication by 9.4 to 13.7 dB, where the
  voiceless twins run -30.9 to -37.7. A 40-50 dB swing between /s/ and /z/.

Category: missing acoustic mechanism. This is the real engineering, but it is bounded and local.

### Cause 4 - vowel space collapse (~10%)

`reduce = 0.7` (`voice.gd:1101`) lerps **all three** formants 70% toward a single fixed schwa
`[500, 1500, 2500]` for every CMUdict stress-0 vowel. Measured vowel space hull:

| | Bark^2 |
|---|---|
| TABLE targets | 18.74 |
| after reduce=0.7 | 1.59 (8.5%) |
| LPC-measured, unstressed | 1.69 vs 11.49 stressed (85% collapse) |

It also drags F3, raising /ER/'s F3 from 1690 to 2257 Hz. A low F3 *is* American /r/, so unstressed
"-er" is destroyed - 7 of 7 tokens in the test take.

Compounding it: the formant trajectory is a first-order EMA and `LOCUS_SHARE = 0.35` starts pulling
toward the next target before the current one is reached, so **no segment ever arrives and holds**.
Measured median attainment 0.77, falling to 0.61 at fast pace. Glides and nasals get the *slowest*
time constant (32 ms, `voice.gd:1677-1678`) on the *shortest* segments (60-75 ms) - backwards, since
a glide is a ballistic gesture.

### Cause 5 - effects erase what survives (~5%, but free to fix)

- **The drone bank holds a -16 dB floor under every pause.** `DRONE` defaults true (`voice.gd:140`)
  and sums unconditionally into every sample (`voice.gd:1955`). Boundary trough depth at a 40 ms
  gap: **-15.5 dB with drone, -31.8 dB without**. Silence never arrives, so word boundaries never
  appear. This, not the 4 ms inter-word gap, is why words run together.
- **The leveler never compresses.** Its detector is block-mean |x| (median 0.023, p95 0.128) but
  `COMP_TARGET` is 0.42, a near-peak figure. 99.4% of blocks sit below the clamp point, so gain is
  pinned at `COMP_MAX` 1.5 permanently. The mastering stage is a constant gain wearing a
  compressor's name.
- **The location bandpass is baked into exports.** A Q=2.5 state-variable bandpass, 55% wet, 1.8x
  makeup, centred in the F1/F2 region, plus a carrier tone. Unlike presence it goes through
  `bake_location` into the rendered take. Measured at prox 1.0 / 400 Hz: +5.6 dB at 300-800 Hz and
  -6.2 to -6.4 dB across everything above 2.5 kHz. It should never touch a chapter render.
- **There is no switch anywhere that produces a clean, dry, full-bandwidth voice.** Five effect
  families, five unrelated gating mechanisms, and the echo has no override at all (it is a
  per-segment planner value).

### Cause 6 - the front end, which is the worst of it for a book

On typographic manuscript text, **10.4% of tokens are damaged before synthesis begins**:

| defect | file | measured |
|---|---|---|
| numerals silently deleted (no digit in any char table) | `phonemes.gd:119-137, 549-571` | 2.2% of tokens dropped; 0 of 11 numeric forms survived |
| curly quotes, apostrophes, em dash, ellipsis in no char set | `phonemes.gd:216, 223` | coverage 88.3% typographic vs 92.7% ASCII |
| punctuation stripped before quotes, so dialogue-final words miss the dictionary | `phonemes.gd:216-223` | fixing order alone: 88.3% -> 95.7% |
| hyphenated compounds fused, parts never looked up | `phonemes.gd:549-571` | 7/7 wrong |
| newline treated as a sentence stop | `phonemes.gd:270-273` | 18 spurious full stops per chapter sample |
| abbreviation periods treated as stops | | 7 of 42 stops false |
| suffix/stem machinery unreachable (letter rules always return non-empty, so the raw base always wins) | `phonemes.gd:631-642` | 12/12 restore_e and 12/12 undouble cases dead |
| letter-to-sound quality | `phonemes.gd:119-137` | **19 of 27 ordinary out-of-dictionary words plainly wrong (70%)** |

### Cause 7 - the game is selecting *for* unintelligibility

This one deserves its own heading because it is a structural trap, not a bug.

The toll (`synth_editor.gd:1314-1317`) frays exactly `[grit, air, breath]` toward the rail as a
foreign seed is reeled in: `x += TOLL_FRAY * toll^2 * (1 - x)`, saturating, with no opposing term.
At difficulty 1.0 the medians land at grit 0.89 / air 0.90 / breath 0.92, giving shimmer 0.081,
`air_cut` down to 1938 Hz, `air_gain` 0.047, breath 0.116.

Those are the same three parameters identified in Cause 1 as the primary masking source, at roughly
2.4x the default. **The reward mechanic drives the voice toward maximum masking.** So the seeds you
work hardest to catch are the least intelligible ones you own, which matches the reported
experience.

Related: the annealed genome bypasses `G_BOUNDS` entirely (`voice.gd:585-589` takes the override
branch; bounds are applied only inside `_lineage_genome` at `voice.gd:730-732`). At d=1.0 that
yields `act_thr` p05 = **-0.898** and `breath_span` p05 = **-3.771**, both negative, both below a
floor that exists specifically to "keep an elaborate lineage a VOICE".

And fishing cannot breed the problems away even in principle: the genome is an unweighted mean of
the PRIOR plus each lineage (`voice.gd:591-599`), so with one lineage every gene is a 50/50 average
and **can move at most halfway to its bound**. Best achievable hesitation rate by breeding is ~4.8%
against a 1.6% default. The only escape from the prior is the toll, which is the thing making it
worse.

### Cause 8 - inflection

Seven melodic terms sum with no budget: declination, motif tilt, ring (to ±3.15 st), accent
(`spec.f0_accent` hardcoded 4.0 at `voice.gd:336`, reaching +7.5 st), microprosody, terminal
contour, and the pitch attractors. Measured per-sentence f0 span: **p50 8.95 st, p95 13.89, max
16.04** at `lilt = 0`, and 17.91 st median at `lilt = +1`. Adjacent-vowel jumps reach 7.13 st at
p95. That is the "all over the place" complaint, quantified.

Also unasked-for: breath pauses fire on **7.5% of words** with 41% landing mid-phrase
(`voice.gd:881-882`, a syllable counter with no reference to syntax), and disfluencies at **16 per
1000 words**, 40% of them a spoken "uh". Extrapolated over a 140k-word book: roughly **900 audible
"uh"s**.

---

## 2. Fastest path to an audible difference

Ordered by impact over effort. Everything in this section is constants or a flag, no new
mechanisms, and all of it is reversible.

1. **`NOISE_FX := false`** (`voice.gd:111`) - the flag already exists. Recovers ~10 dB of
   consonant-to-vowel contrast immediately. This is the single biggest win and it costs one
   character. Listen first; then decide whether the correct end state is "off" or "tied to glottal
   open quotient and pushed 10-12 dB down", which is the principled version.
2. **`DRONE := false`** (`voice.gd:140`) - restores 16 dB of boundary trough depth. Word boundaries
   reappear. Keep the drone as an opt-in atmosphere for the game, not a default of the voice.
3. **Zero the echo send and the location bake for renders** - the echo has no override today, so
   this needs a small addition rather than a flag flip, but it is a few lines. `bake_location`
   should not be in the export path.
4. **Cap `reduce` at 0.25-0.35 and exempt F3** (`voice.gd:1101`, `voice.gd:1570`) - recovers most
   of the vowel space and all of unstressed /r/. Undershoot already supplies the centralization a
   short vowel should have; the explicit lerp is doubling it.
5. **Raise `FRIC_LEVEL`, and give bursts their own scaler** - a burst is a transient, a fricative is
   a steady state; sharing one seat is why neither can be set correctly. Do this after 1, since the
   right level is unknowable while the hiss is masking the measurement.
6. **Fix the front-end normalization order** (strip wrappers, then terminal punctuation, then
   wrappers again) and add the Unicode punctuation block. Coverage 88.3% -> 95.7% for one
   reordering.
7. **Apply `G_BOUNDS` in `ProsodyWalk._init` to both branches** - a correctness fix, not a design
   change. The constant already exists and states its own intent.

Expect items 1, 2 and 4 together to be the difference between "unusable" and "usable but rough".

---

## 3. The measurement gate

This project's history (`next/voice_rca.md` sections 7-16) is a dozen rounds of chasing artifacts
by ear, and at least three of those rounds traded intelligibility for pleasantness - the frication
level was repeatedly reduced because it "offended", and the diagnosis above shows the offence was
masking, not level. **Judging by ear alone is what produced this state.** A gate has to land
alongside the cheap fixes, not after them.

Three tiers, in the existing `tests/` convention:

- **Cheap always-on proxy, numpy only, no new dependencies.** Three numbers, all of which the audit
  agents already computed successfully, so the code path is proven: (a) per-phoneme-class delivered
  level relative to mean vowel, checked against a target table; (b) vowel space hull area in Bark²,
  stressed and unstressed separately; (c) word-boundary trough depth distribution. Wire into
  `tests/voice_check.gd`, which is already the gate. This catches every regression class in this
  document.
- **Primary metric: ASR word error rate.** Synthesize known text, transcribe, compare - the
  standard objective intelligibility proxy, and unusually clean here because alignment is known by
  construction. `torch` and `transformers` are installed; no ASR weights are cached, so this needs
  a one-time download (whisper-small is ~500 MB). **Your call, and worth stating plainly: an ASR
  model used as a test oracle ships nothing and never touches the render path, so it does not
  violate the no-generative-AI constraint - but it is a dependency and a download, so it is your
  decision, not mine.** Target: get WER under 10% on clean narration. Today it would plausibly be
  40-60%.
- **Human protocol, minutes not hours.** A modified rhyme test over minimal pairs - bat/pat/mat/cat,
  sit/sip/sick, thin/fin/sin - which isolates *which contrast* is failing instead of yielding a
  vague verdict. Given Cause 3, expect stop place and /f/-/th/ to be at chance today. This is the
  test that would have caught the collapsed stop inventory years earlier than an artifact hunt.

---

## 4. The acoustic work

Only after the gate exists. Each item is bounded and local.

1. **Make the amplitude column mean what it says.** Change `namp` to a target delivered level in dB
   relative to a reference vowel, and derive per-pole gains at load time from
   `sum(a_k^2 * BW_k * pi/2)`. This makes the whole inventory tunable by a phonetician's numbers
   instead of by trial. Prerequisite for balancing anything.
2. **Hold the stop's own locus through the release**, and let the following vowel carry the glide.
   Voiceless fricatives already get exactly this treatment at `voice.gd:1355` (`nxt if voiced else
   []`), so the pattern is in the codebase; the stop branch simply does not use it. This is the fix
   that makes /b/, /d/ and /g/ different sounds.
3. **Per-phoneme VOT as a table field**, not `dur * 0.3`: 55-75 ms for aspirated /p t k/, 5-15 ms
   for /b d g/, near zero in /s/-clusters. Add a short voiced-VOT run so /b d g/ have a release
   event at all.
4. **Rebalance voiced obstruents** - decouple the voicing amplitude from the frication level and
   honour `seg.amp` so stress reaches them.
5. **Replace the fixed-tau EMA with a time-normalized trajectory**: a fixed 40-60 ms transition at
   segment onset, then an explicit HOLD at target, then the locus glide. This is the structural fix
   for undershoot, and it also fixes the glide/nasal inversion for free.
6. **Give the melody a budget**, the way `MOD_BUDGET` (`voice.gd:542`) already does for the
   modulator set. Split structural terms (declination, accent scaled by prominence, terminal
   contour) from decorative ones (motif tilt, ring, attractors) and let the decorative set compete
   for a fixed share. Target a p95 sentence span near 6-8 st rather than 13.9.
7. **Move the toll off `[grit, air, breath]`** onto axes a listener tolerates: pitch register, tract
   length, tempo, drawl, drone character. The intent - a foreign catch should cost something
   audible - is good. The channel choice is what makes caught seeds unintelligible.
8. **Breath placement as a search over legal sites**, not a syllable counter. `Phrasing` already
   knows content vs function words and phrase ends; let the counter propose and let phrasing choose.
9. **Hesitation behind a register switch**, not on the genome. It is a property of spontaneous
   speech, not of a person. Keep it for the fishing instrument where it makes the wait alive; it has
   no place in a chapter.

---

## 5. Architecture

Your read is right, but not for the reason you gave. `voice.gd` is 1198 lines of code under 798
lines of comment, so the problem is not volume - it is **concentration and typing**.

The measured facts:

- **Not one of the six stage boundaries carries a typed record.** Text to words, words to phrasing,
  words to prosody, prosody to segments, segments to acoustics, acoustics to delivery: all pass bare
  `Dictionary`/`Array`, and every consumer reads through `.get(key, default)`. A field a producer
  forgets, misspells, or stops writing resolves to a default forever instead of failing at parse
  time. This codebase already names that as a known bug class in `mask_session.gd:110-118`.
- **312 segments from a 72-word paragraph arrive in two different key shapes** (75 with 11 keys, 237
  with 12) from three constructors.
- **`plan()` writes `amp` on every segment; `synth()` reads it only for vowels, glides and nasals.**
  Fricatives, stops and aspiration - **33.4% of the take by time, 101 of 312 segments** - silently
  discard their planned dynamics, carrying a mean |amp - 1| of 0.134 that nobody hears. Emphasis,
  the nuclear accent, function-word reduction: all computed, all thrown away, for a third of the
  audio. This is the single clearest argument for typed records, and it is also a direct
  intelligibility bug, since stress cannot reach the consonants.
- `synth_state()` is a flat **61-key untyped Dictionary** tangling filter state, source state, EMA
  smoothers, prosody realization, effect state, mastering state and the output timing map in one
  namespace.
- `_run_frames` is **407 lines with 12 positional parameters**, nesting depth 8, spanning eight
  unrelated concerns.

Where you are *not* right: a per-sample DSP kernel being long and imperative is not automatically a
design flaw. At 44.1 kHz, one virtual call per stage per sample is ~220k dynamic dispatches per
second of audio on a thread that must hold a 2.5 s lead, and GDScript cannot afford that. **Keep the
inner loop flat.** The fix is to lift everything that is *not* the inner loop out of it: the ~86
lines of preamble, ~85 of per-frame scheduling, ~62 of drone, ~48 of master bus, ~16 of write-back.
That leaves roughly 110 lines of actual per-sample kernel, which is a reasonable size for what it is.

### The target shape

Eight stages, each with one typed record crossing its boundary. GDScript 4 inner classes
(`class X extends RefCounted` with typed `var`s) give real parse-time member checking, which
`.get()` throws away:

```
SourceText  -> normalize   -> NormalizedText   (numerals, unicode, abbreviations expanded)
            -> tokenize    -> Token[]          (surface, wrappers, terminal punct, pause class)
            -> g2p         -> LexicalWord[]    (phones, stress, syllables, provenance: dict|rules|inline)
            -> phrase      -> PhrasedWord[]    (prominence, phrase position, given/new)
            -> plan        -> Segment[]        (phone, dur, f0, amp, reduce, echo, locus)
            -> realize     -> AudioBlock[]     (the kernel; state as typed sub-objects)
            -> master      -> Take             (pcm + word/phone timing map)
            -> deliver     -> stream or file
```

Three things become data rather than code:

- **A phoneme is a definition record**, not a table entry plus special cases scattered through
  `_run_frames`. Fields: type, formant targets, bandwidths, parallel poles, target delivered level
  in dB, VOT, closure fraction, locus behaviour. Then item 1 in section 4 becomes a load-time
  derivation over that record rather than a hand-tuned column.
- **A voice configuration is one object** whether it came from a preset, a bred seed, or a
  diagnostic A/B. Today `RAW_MODE`, `NOISE_FX`, `DRONE`, `FRIC_LEVEL` are static vars and module
  constants reachable only by editing the file - which is why every past investigation had to patch
  source to run an experiment, and why `tests/pure_say.gd` has to save and restore a static var.
- **The effect chain is a declared ordered list**, inspectable and per-context, so "clean narration"
  and "fishing atmosphere" are two configurations rather than two code paths. This is the same move
  `mask_session.gd`'s `MASK_EFFECTS`/`EFFECT_CONTROLS` already makes for the mask editor, so it
  matches house style.

### Ordering, which is the real question

**Do not restructure first.** It would delay every audible improvement behind a refactor with no
gate to prove it did not regress anything, which is exactly the failure pattern this project already
has.

**Do not restructure last** either, because items 1-5 in section 4 each touch a boundary that wants
a type.

Interleave: cheap constants (section 2) land immediately with no structural change. Then the gate.
Then introduce the typed record *for the one stage each acoustic fix touches*, as part of that fix.
`Segment` comes first because item 1 and the `amp` bug both live there. `PhonemeDef` comes with the
amplitude-column work. `SynthState`'s split into six typed sub-objects comes with the trajectory
rewrite. By the end of section 4 the pipeline is typed, and no day was spent on typing alone.

Determinism: seeds must reproduce forever, so every restructure step needs a byte-identical check on
a fixed lineage before and after. `tests/voice_check.gd` should assert that. Note that the acoustic
fixes themselves *will* change output - that is the point - so bump a version marker on the voice
config and treat pre-fix seeds as a different generation rather than trying to preserve them.

---

## 6. The book pipeline

Beyond intelligibility, feeding 35 chapters needs:

- **A real text normalization stage** (section 4 has none today). Cardinals, ordinals, decimals,
  times, currency, percentages, years, roman numerals, abbreviations, acronyms. Without it numbers
  are deleted silently.
- **Pronunciation overrides for invented proper nouns.** The inline `[K AE T]` mechanism already
  exists and is the right primitive; what is missing is a per-project lexicon file so a name is
  spelled phonetically once, not at every occurrence. Given 70% letter-to-sound failure on
  out-of-dictionary words, this is not optional for fiction.
- **A pronunciation audit pass**: run a chapter, report every token that fell through to letter
  rules with its phone string, so you can triage the ones that matter before rendering. Cheap, and
  it turns an invisible failure into a worklist.
- **Chapter-scale batching** with a stable seed: same bred lineage, same voice config version,
  across all 35 renders. Voice drift between chapters would be worse than a mediocre voice.
- **A clean-narration effect configuration** (section 2, item 3) that the export path selects, so
  the chapter render never inherits the fishing atmosphere.

---

## 7. Sequencing

```
A. Cheap constants (hours)          NOISE_FX, DRONE, reduce cap + F3 exempt, G_BOUNDS fix
                                    -> listen. Independent of everything else.
B. Front-end normalization order    coverage 88.3 -> 95.7%. Independent of A.
   + unicode punctuation (hours)
C. Measurement gate (small)         numpy proxies into voice_check.gd. Needs A to have a baseline.
   + ASR WER (decision needed)
D. Amplitude column as dB target    needs C to verify. Unblocks all consonant balancing.
   (small-medium)
E. Stop locus + VOT + voiced        needs D. The largest single intelligibility gain after A.
   release (medium)
F. Trajectory rewrite (medium)      needs C. Independent of D/E.
G. Melody budget + breath placement needs C. Independent of D/E/F.
   + hesitation register (medium)
H. Toll channel change (small)      independent, but pointless before A.
I. Text normalization stage +       needs B. Required before any real chapter render.
   project lexicon (medium)
J. Clean-narration config (small)   needs A. Required before any real chapter render.
```

A and B are independent and can land today. C gates everything after. E is the big one.

---

## 8. Open questions

1. **ASR oracle: yes or no?** ~500 MB one-time download, test-only, ships nothing. Without it the
   gate is acoustic proxies only, which catch regressions but do not measure intelligibility
   directly.
2. **What should the default voice be optimized for?** The instrument and the narrator want
   different things - the fishing loop wants a voice that is alive and strange, the chapter wants
   one that is clear. Section 4 item 9 assumes these become two registers over one engine. Confirm.
3. **Is `reduce` doing anything you want?** The recommendation caps it hard. If you like the
   heavily-reduced quality on some seeds, it should become a genome axis rather than a constant.
4. **Seed generation break.** The acoustic fixes change output for every existing lineage. Confirmed
   acceptable (no belt to preserve), but worth a version marker so old takes are explicable.
5. Unmeasured: cross-platform float determinism of the synthesizer. Not urgent while everything runs
   on one machine, but the game's guarantees depend on it.

---

## 9. Status, 2026-08-08

**Landed.** Phase A in full (`NOISE_FX` off, `DRONE` off, `reduce` 0.7 -> 0.3, F3 exempted from
reduction, `G_BOUNDS` applied at point of use). Phase C, the measurement gate:
`axis/ghost/tests/render_fixtures.gd` renders three fixed fixtures with ground-truth alignment, and
`axis/ghost/measure_voice.py` measures them with numpy (per-class delivered level against
literature bands, vowel-space hull area in Bark², word-boundary trough depth, octave-band energy)
plus optional ASR word error rate. Artifacts land in `axis/ghost/build/voice/`, gitignored. Phase D
in full: the parallel branch is now power-normalized in `_tune_parallel`, so `namp` controls
delivered level for the first time, and the column was re-seated from measurement. Phase E items 2
and 3: stops hold their own locus through the release and hand the glide forward, and /b d g/ have
a voiced release event.

**Measured effect.**

| metric | before | after |
|---|---|---|
| sibilant level re vowel | -8.1 dB | -3.8 dB (target -8 to +2) |
| weak fricative re vowel | -6.4 dB (at parity with sibilants) | -12.4 to -17.1 dB (target -25 to -10) |
| voiceless fricative spread | 3.2 dB | 8.5 dB (natural 10-18) |
| voiced stop level re vowel | -23.0 dB | -18.2 dB (target -20 to -6) |
| word-boundary trough, >=40 ms gaps | -15.5 dB | digital silence |
| ASR WER, narration fixture | not measured | **4.5%** |
| ASR WER, consonant-dense fixture | not measured | **36.7%** |

`tests/voice_check.gd` still reports ALL OK, so trait determinism, lineage reproducibility and
ornament aging survived every change.

**Read the WER numbers carefully.** 4.5% on connected narrative prose is a genuinely good result,
but whisper carries a language model that repairs phonetic damage from context, so it flatters
context-rich text. The consonant-dense fixture at 36.7% is the more honest signal, and its error
pattern is diagnostic: `Take the cake, bake the dough, and go` came back as `Tape the cape, tape the
toe, a bone`. Every one of those is a stop PLACE or VOICING confusion. The locus fix helped and did
not finish the job.

**Not yet done**, in the order the plan gives them: Phase E item 1 (per-phoneme VOT as a table
field - the remaining stop confusions point straight at it), Phase F (the trajectory rewrite; the
consonant-dense fixture still shows a 0.216 vowel-space ratio against a 0.40 gate, so undershoot is
unaddressed), Phase G (melody budget, breath placement, hesitation register), Phase B and I (the
whole text front end, which is untouched and is the largest single risk to the 35-chapter run),
Phase H (toll channel), Phase J (clean-narration configuration).

---

## 10. Second round, 2026-08-08: the Klatt reference comparison

Triggered by the user's report that a past RAW_MODE bypass sounded equally bad, which rules out the
whole modulation stack and points at the base source-filter path.

**Platform verdict: not the constraint, with numbers.** A kernel carrying Klatt's entire Fig. 6
inventory (17 second-order sections, retuned every 1.45 ms, pitch-synchronous noise) benchmarks at
**9.0x real time, 11.1% of one core** on this machine - 1.9x faster than ghost's current six-pole
kernel, because 67% of ghost's per-sample cost is `Reso.step()` dispatch rather than arithmetic.
Klatt 1980 ran 6x real time on a PDP-11/45; DECtalk scored 97% on the Modified Rhyme Test against
97-99% for natural speech. GDScript floats are IEEE doubles, `Reso.tune` hits its bandwidth within
1 Hz, and the audio path is bit-transparent.

**The measurement that explains why nothing converged.** Whole-take spectral tilt averages voice
with consonant noise, and ghost's two errors are opposite, so they cancel. Vowels-only tilt was
-11.5 dB/oct (too dark) while /S/ measured **+10.7 dB/oct** (too bright). Muffled and hissing were
one metric apart the whole time. `measure_voice.py` now measures tilt per class and never gates on
the mixed number.

**Four named causes, each with a primary source, all fixed this round.**

1. *The hiss.* Klatt 1980 p.977 low-passes the noise source so lip radiation "exactly cancels out
   the effects of LPF", giving flat frication. Ghost had no such filter, so the parallel branch ate
   the full +6 dB/oct differentiator. Frication now sums after radiation. **Obstruent tilt +5.2 to
   +10.7 -> -1.2 to -2.6 dB/oct.** This is why every FRIC_LEVEL ladder in voice_rca.md failed: the
   level was never the problem.
2. *The muffle.* Klatt Table I clusters F4/F5 at 3300/3750 to produce "an energy concentration
   around 3 to 3.5 kHz and a rapid falloff above about 4 kHz". Ghost had 3400/4700, 1300 Hz apart.
   F5 -> 3800, B5 -> 200. **2500-4000 Hz band energy -20.79 -> -7.54 dB; 1500-2500 -16.47 ->
   -11.33.** Measured vowel spectra had matched the analytic prediction of ghost's own equations
   within 1.5 dB, so nothing downstream was causing it and nothing downstream could have fixed it.
3. *The clicks.* KLSYN88 ch. 3 (after Fujisaki and Azemi 1971) rescales `y1`/`y2` by `sqrt(A'/A)`
   every time F1-F3 are set, naming the symptom "clicks and burps". Ghost retuned six poles 690
   times per second per pole with the history untouched. Now rescaled in `Reso.tune`. Note the
   earlier session zeroed the PARALLEL bank for this reason, which did nothing - that bank is
   already silent between consonants; the cascade is the one that matters and can never be zeroed.
4. *Target attainment.* Only 25.5% of vowels reached within 5% of their F1 target. The locus glide
   is NOT implicated (removing it entirely moves attainment by 0.00) - it is the EMA time constant,
   24 ms giving 3 tau = 72 ms against a median 95 ms vowel. Time constants halved.

**Honest result: the spectra improved and the ASR proxy got worse.**

| | before round 2 | after |
|---|---|---|
| obstruent tilt | +5.2 to +10.7 dB/oct | -1.2 to -2.6 (flat target) |
| 2500-4000 Hz energy | -20.79 dB | -7.54 dB |
| 6000-14000 Hz energy | -20.6 dB | -24.7 dB |
| WER, narration | 4.5% | **10.4%** |
| WER, consonant-dense | 36.7% | **40.8%** |

Individual errors moved both ways ("torn" -> "cold" and "cycle ship" -> "sunken ship" fixed; "for"
-> "from" and "Six thin" -> "Sixteen" newly broken). At 67 words a 3-to-7 error change is close to
the resolution limit of the fixture, so this is **unresolved, not a clean regression** - but it is
not a win either, and it must not be reported as one. The fixture set needs to be several times
larger before WER can adjudicate changes this size.

**Open, in priority order.** Restage `BURST_GAIN` (voiceless stops now measure -16.5 to -18.0 dB re
vowel against a -15 to -5 target). Vowel-space ratio is 0.24-0.37 against a 0.40 gate. Peak is
pinned at -1.54 dBFS on every fixture even after `OUT_GAIN` 4.5 -> 3.2, so the lookahead limiter is
still working on transients and the crest factor wants investigating. Then the items the reference
read ranked next: bounded noise envelope replacing the unbounded `nampsm` EMA (74 ms decay against
Klatt's bounded 5 ms), pitch-synchronous noise modulation, period-latched voicing amplitude, and
the parallel branch derived from the cascade's own formants - which Klatt p.981 names explicitly as
what prevents frication "dissociating from the rest of the speech signal", and which is the real
answer to the RCA's second-source finding.

### Round 2b: the clicks are still unsolved, and here is everything they are not

Seven interventions, each with a plausible mechanism, each measured. **None moved the click rate**,
which has sat at 4-5/s (coarse detector) or 13-14/s (6-sigma block-local detector) since before any
of this work began:

| intervention | rationale | result |
|---|---|---|
| lower `FRIC_LEVEL` | peaks hitting saturation | 0.006% above the knee; not saturation |
| drop the burst forward-glide | tract traversing its excursion in 8 ms | unchanged |
| `NOISE_FX` on | noise floor was masking them | 4.87 vs 5.15/s |
| zero parallel-bank state on retune | filter transient at coefficient change | identical to 4 decimals |
| `sqrt(A'/A)` cascade history rescale | Klatt KLSYN88 "clicks and burps" | **made it worse** - 689 Hz AM, heard as "electrical" |
| `FRAME` 64 -> 16 | zipper noise from parameter stepping | 12.87 -> 11.81/s |
| `RAW_MODE` on (all modulation bypassed) | user hypothesis: modulation stepping | **14.03/s, higher than with modulation on** |
| unipolar pulse table + DC blocker + 512 points | per-period gain stepping a -0.30 pedestal | 12.87 -> 14.15/s |

What this rules out: saturation, the noise route, the masking floor, filter state carry-over, the
parameter update rate, the entire modulation stack, and the glottal pedestal. The events sit on
voiced phones (AE, AH, T, D, K, M) and at stop bursts.

**The most likely remaining explanation is that the detector is partly flagging glottal closures.**
A closure is the sharpest legitimate event in voiced speech, and raising the wavetable to 512
points made closures sharper - which is exactly when the count went up. The reference read's own
detector reported 3.19 events/s confined to stop phones; this reimplementation reports 13-14/s
including vowels, so the two disagree and neither has been validated against a human reference
recording passed through the same code. **Do not tune against this metric until it is calibrated
on real speech.** Chasing it has now cost more than it has returned.

**Nothing has ever measured what the user actually hears.** `render_fixtures.gd` calls
`Voice.render`; the live path is `VoiceStream._push_available`, which applies a presence gain and a
one-pole low-pass at `900 * 2^(4*presence)` Hz - 4135 Hz at presence 0.55. A capture path through
`VoiceStream` at presence 1.0 is the single highest-value missing measurement.
