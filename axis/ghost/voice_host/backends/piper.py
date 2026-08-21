"""Piper voices, run as raw ONNX. No GPL code is imported, linked or shipped.

THE LICENSING SITUATION, because it drives every design choice here
-------------------------------------------------------------------
`rhasspy/piper` was archived 2025-10-06. Development moved to
`OHF-Voice/piper1-gpl`, which relicensed to **GPL-3.0-or-later** in v1.3.0
because the wheel vendors eSpeak-NG. So the obvious route - `pip install
piper-tts` - would make a shipped game GPL.

It is also unnecessary. A Piper voice is two files: an `.onnx` graph and an
`.onnx.json` config, distributed from an MIT-tagged Hugging Face repository as
DATA. The config carries a `phoneme_id_map`. ghost already has its own G2P, so
we can map our phonemes to ids ourselves and call onnxruntime (MIT) directly.
Nothing GPL is involved at inference or distribution time.

THE PER-VOICE TRAP, which is worse than it looks
------------------------------------------------
Voice checkpoints are licensed individually AND the license is transitive
through fine-tuning, which is not obvious and is not flagged anywhere machine
readable. Piper's own default demo voice, `lessac`, is Blizzard 2013, which
restricts use to research and explicitly excludes "the development, marketing,
commercialisation, sale or licencing of voice synthesis products". Voices
fine-tuned FROM lessac inherit that: amy, joe, hfc_female, hfc_male, libritts_r.
Separately ryan and the hfc pair are CC BY-NC-SA, and kathleen is CC0 data
fine-tuned from ryan.

So VOICES below is an allowlist, not a catalogue, and every entry records its
dataset AND its derivation chain. Adding a voice means reading its MODEL_CARD.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from . import Backend, BackendError, register

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# VITS hop length: one duration-predictor frame is this many samples.
HOP_LENGTH = 256

# Sentence-final marks. ghost's front end already emits these as phones, so the
# split costs nothing and needs no text.
SENTENCE_END = {".", "!", "?"}

# PAUSE AFTER PUNCTUATION - EXTRA seconds of real silence at `pause_scale` = 1.0, on top of
# whatever the model already does rather than instead of it.
#
# WHY IT EXISTS. A VITS model gets punctuation as a phoneme id and decides for itself how long to
# dwell. The report was "the model runs through commas and periods, and especially colons, too
# quickly". The model still SEES the mark, so its intonation is untouched; this only lengthens the
# rest the mark is asking for.
#
# WHY THESE NUMBERS - measured, not guessed. Span of the token owning the mark,
# en_US-ljspeech-medium, same sentence throughout:
#   no mark 0.279 s | "," 0.430 (+0.151) | ";" 0.465 (+0.186) | ":" 0.580 (+0.301)
# Piper is NOT skipping these, and the colon already rests twice as long as a comma. So the
# reported "colons run through" was never the model: ghost was sending every colon to it AS A
# COMMA (generative_editor.gd collapsed the mark before it left Godot). Fixing that is most of the
# cure. This table is the seasoning, and it stays small deliberately - an earlier draft added
# 0.26 s to a colon, making it a ~0.58 s rest, longer than a full stop, which inverts the
# punctuation hierarchy. The ordering : > ; > , is preserved so the marks stay distinguishable.
#
# The three sentence-final marks are 0.32 because that is EXACTLY the `sentence_gap` default this
# file has always shipped: the table subsumes that parameter rather than stacking on it, so
# `pause_scale` = 1.0 with no explicit `sentence_gap` reproduces the old behaviour sample for
# sample. A caller still sending `sentence_gap` still wins for . ! ? - it is the same knob.
#
# Anyone wanting more reaches for the Pause slider, which runs to 10x (see
# generative_editor.MAX_PAUSE_SCALE); the base stays calibrated to what the model actually does.
PAUSE_AFTER: dict[str, float] = {
    ",": 0.10,
    ";": 0.13,
    ":": 0.16,
    # Sentence-final marks ARE the sentence gap - unified, never added to it (see _gap_for).
    ".": 0.32,
    "!": 0.32,
    "?": 0.32,
}

# Longest silence to leave BETWEEN two sentences, however far pause_scale is pushed - it is
# the one gap that compounds over a whole chapter. Mirrors generative_editor.SEAM_CEILING.
SEAM_CEILING = 1.2

# Click avoidance for spliced silence - see _splice_pauses.
SPLICE_SEARCH_MS = 3.0  # how far the cut may move to find a quieter sample
SPLICE_FADE_MS = 2.0  # raised-cosine ramp into and out of the silence
SPLICE_ENV_MS = 4.0  # moving-average window used to find the quiet PLACE
SPLICE_REACH_MS = 70.0  # how far forward the cut may look for the real gap
SPLICE_BACK_MS = 0.0  # ...and how far back: none, see _quiet_point


def _split_sentences(phones: list) -> list[list]:
    """Break a phone stream at sentence-final punctuation, keeping the mark."""
    out, cur = [], []
    for p in phones:
        cur.append(p)
        if str(p) in SENTENCE_END:
            out.append(cur)
            cur = []
    if cur and any(str(x).strip() for x in cur):
        out.append(cur)
    return out or [list(phones)]


def _pause_scale(params: dict) -> float:
    """The user's pause multiplier. 1.0 = the table, 0.0 = none, 2.0 = double."""
    try:
        return max(0.0, float(params.get("pause_scale", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _pause_for(mark: str, params: dict) -> float:
    """Extra silence after a MID-SENTENCE mark, in seconds.

    CAPPED IN PROPORTION TO THE SENTENCE GAP, and that is the fix for a reported bug rather
    than a precaution. `_gap_for` bounds a sentence's own pause at SEAM_CEILING because it
    compounds over a chapter; this one used to be uncapped, and the two together invert the
    punctuation hierarchy as soon as the Pause slider leaves 1. At the slider's top (10x, see
    generative_editor.MAX_PAUSE_SCALE) a comma took 1.0 s of spliced silence ON TOP of the
    ~0.3 s the model already dwells there, while a full stop stayed pinned at 1.2 s total - so
    the commas became the longest rests in the paragraph and the full stops sounded skipped.
    Reported exactly that way: "a large breath was being taken after every single comma... NO
    breaths were being taken after punctuation: it was only happening on commas."

    The ceiling is each mark's OWN SHARE of the sentence ceiling, so `,` < `;` < `:` < `.`
    holds at every scale instead of only at 1. Below about 3.7x nothing changes at all - the
    default render is sample for sample what it was.
    """
    m = str(mark)
    base = PAUSE_AFTER.get(m, 0.0)
    if base <= 0.0:
        return 0.0
    ceiling = SEAM_CEILING * (base / PAUSE_AFTER["."])
    return min(base * _pause_scale(params), ceiling)


def _discourse_plan(groups: list, params: dict) -> list[dict]:
    """Per-sentence rate and pitch from DISCOURSE STRUCTURE, not from a clock.

    Piper is a sentence-level model. It declines pitch across a phrase, lengthens
    finally, and handles a question - all of it well. What it cannot know is that
    this is the fourth sentence of a paragraph, because it never sees the
    paragraph. So every sentence starts from the same register and runs at the
    same rate, and a chapter is several hundred identical arcs laid end to end.
    That is the whole of "the delivery remains more or less constant": before
    this, `length_scale` / `noise_w` were set once per take and every sentence in
    a forty-minute chapter got the same three numbers.

    The rules here are the documented ones, not invented shape:

      PARATONE (the intonational paragraph). Speakers reset F0 upward at the
      start of a discourse unit and let it decline across it, and the size of the
      reset scales with the depth of the boundary. Lehiste (1975) named the
      intonational paragraph; 't Hart/Collier/Cohen (1990) is the standard
      treatment of declination; Sluijter & Terken (1993) and Nakajima & Allen
      (1993) tie reset size to boundary depth. This is why a slow undulation is
      the right instinct and a slow SINE is not: the wave is real, but its period
      is the paragraph, so it has to be phase-locked to the text. A fixed-period
      oscillator drifts against the prose and lands its peaks on whatever happens
      to be there, which is the one thing a real speaker never does.

      FINAL LENGTHENING. Material before a prosodic boundary is lengthened, and
      the amount indexes the strength of that boundary (Klatt 1975; Wightman,
      Shattuck-Hufnagel, Ostendorf & Price 1992). So the slowing is progressive
      into the end of the unit rather than a flat rate for every sentence.

      LENGTH-CONDITIONED RATE. The longer the utterance, the shorter its
      segments - speakers compress long stretches and dwell on short ones
      (Lindblom's anticipatory shortening). A one-clause sentence after a long
      one genuinely lands harder, which is free emphasis from structure alone.

      VOCAL EFFORT, which is not volume. A louder voice has a FLATTER source
      spectrum - more high-frequency energy, because the glottal pulse is sharper
      - and a quieter one a steeper tilt. That is why turning a level down reads
      as "further away" rather than "speaking softly": distance is a filter, and
      effort is a different filter. Coupling the tilt to the level is what makes
      the difference read as the speaker easing off rather than the fader moving,
      and effort declines across a paragraph for the same reason pitch does (both
      follow subglottal pressure), so it rides the same contour as the arc.

    `dynamics` (0..1) scales the whole timing half, `prosody_arc` is the paragraph
    pitch arc in SEMITONES peak-to-peak, and `effort` scales the tilt/level
    contour. All default to 0, so an untouched session synthesizes exactly what it
    did before.
    """
    depth = max(0.0, float(params.get("dynamics", 0.0)))
    arc = max(0.0, float(params.get("prosody_arc", 0.0)))
    effort = max(0.0, float(params.get("effort", 0.0)))
    n = len(groups)
    lens = [max(1, len(g)) for g in groups]
    typical = sum(lens) / float(n) if n else 1.0
    # THE CALLER'S PLACE IN THE READING, when it knows it. A request is one sentence
    # (generative_editor.CHUNK_SENTENCES = 1), so deriving position from `groups` makes
    # every sentence the first and last of its own unit at once - u = 0 forever, no final
    # lengthening ever, and the arc flattened to a constant offset. Only the caller has
    # the paragraph structure, so it sends the position and this shapes it.
    u_in = params.get("plan_u")
    v_in = float(params.get("plan_v", 0.0) or 0.0)
    plan = []
    for i, g in enumerate(groups):
        # 0 at the top of the unit, 1 at its last sentence.
        if u_in is not None:
            u = min(max(float(u_in), 0.0), 1.0)
        else:
            u = 0.0 if n <= 1 else i / (n - 1.0)
        # Reset high, decline across the unit. Squared so most of the fall happens
        # late, which is the shape declination actually has.
        # NESTED ARCS. The paragraph's own decline, plus the slower one its SECTION is
        # making - so a run of one-sentence paragraphs, where the fast arc has no room to
        # move at all, still travels. Weighted toward the local shape where there is one:
        # the section is a swell under the writing, not the writing.
        semis = (
            arc * (0.5 - u * u * 0.5 - u * 0.5) * 0.62
            + arc * (0.5 - v_in * v_in * 0.5 - v_in * 0.5) * 0.38
        )
        # Progressive final lengthening into the unit's boundary, strongest at the
        # end. 18% at full depth: a real pre-boundary rime lengthens far more than
        # that, but this is applied to the WHOLE sentence, not just its last rime.
        rate = 1.0 - depth * 0.18 * (u * u) - depth * 0.07 * (v_in * v_in)
        # Long sentences run a little faster, short ones a little slower.
        rel = lens[i] / typical if typical > 0 else 1.0
        rate *= 1.0 + depth * 0.10 * (1.0 - min(max(rel, 0.5), 2.0)) * 0.5
        mark = ""
        for t in reversed(g):
            mark = str(t.get("punct", ""))
            if mark:
                break
        if mark == "?":
            rate *= 1.0 + depth * 0.04  # questions run slightly quicker...
            semis += arc * 0.20  # ...and end higher
        elif mark == "!":
            rate *= 1.0 + depth * 0.06
            semis += arc * 0.12
        # Rhythmic variety: the duration predictor's own noise, varied per sentence
        # instead of pinned for the take. Alternating rather than random so a
        # re-render is identical - an export must not be a different performance.
        nw = 1.0 + depth * 0.25 * (1.0 if i % 2 == 0 else -1.0)
        # +1 at the top of the unit, -1 at its end: the effort contour, shared by
        # the tilt and the level so they can never disagree about which way is
        # louder. Squared-ish like the pitch decline, for the same reason.
        e = (
            1.0
            - 2.0
            * ((u * u * 0.5 + u * 0.5) * 0.62 + (v_in * v_in * 0.5 + v_in * 0.5) * 0.38)
        ) * effort
        plan.append(
            {
                "rate": max(0.55, rate),
                "semis": semis,
                "noise_w_mul": min(1.6, max(0.5, nw)),
                "tilt": 0.45 * e,
                "gain_db": 2.6 * e,
            }
        )
    return plan


def _effort(a, sr: int, tilt: float, gain_db: float):
    """Vocal effort: spectral TILT first, level second.

    The high band is what changes with effort, so it is what gets moved - a boxcar
    moving average is the low band and the residual is the high one, which is a
    crude filter and exactly the right amount of crude for a gentle shelf. It is
    also O(n) via a cumulative sum, which matters: this runs over every sentence
    and a per-sample IIR loop in Python would cost more than the model does.
    """
    import numpy as np

    if abs(tilt) < 1e-3 and abs(gain_db) < 1e-3:
        return a
    if abs(tilt) >= 1e-3 and a.size > 8:
        # ~1.2 kHz corner: below the speech formants that carry effort, above the
        # fundamental, so the tilt moves brightness and not the voice's weight.
        n = max(2, int(round(sr / 1200.0)))
        pad = np.concatenate(
            [np.full(n, a[0], dtype=np.float32), a, np.full(n, a[-1], dtype=np.float32)]
        )
        c = np.cumsum(pad, dtype=np.float64)
        lo = ((c[n:] - c[:-n]) / n).astype(np.float32)[: a.size]
        # A TILT, so the low band is cut by as much as the high band is lifted -
        # `a + tilt*(a - lo)` only ADDS treble, which is a level change wearing a
        # filter's clothes (measured +5.7 dB where +2.6 was asked for, and the two
        # controls stop being independent). Then renormalise to the level the
        # sentence came in at, so `gain_db` below is the ONLY thing that sets level
        # and the number in the tooltip is the number you get.
        rms0 = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        a = a + tilt * (a - 2.0 * lo)
        rms1 = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        if rms1 > 1e-9 and rms0 > 1e-9:
            a = (a * (rms0 / rms1)).astype(np.float32)
    if abs(gain_db) >= 1e-3:
        a = a * float(10.0 ** (gain_db / 20.0))
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def _resample(a, ratio: float):
    """Play `a` back `ratio` times faster - the same trick the Tone pitch shift uses.

    Linear is enough: the arc is a couple of semitones, well inside where a
    higher-order kernel would be audible (the editor's own resampler says the
    same thing for the same reason).
    """
    import numpy as np

    if abs(ratio - 1.0) < 1e-4 or a.size < 2:
        return a
    n = max(1, int(round(a.size / ratio)))
    idx = np.linspace(0.0, a.size - 1.0, n)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, a.size - 1)
    frac = (idx - lo).astype(np.float32)
    return (a[lo] * (1.0 - frac) + a[hi] * frac).astype(np.float32)


def _gap_for(mark: str, params: dict) -> float:
    """Seconds between two sentences, for a sentence ending in `mark`.

    Unified with PAUSE_AFTER rather than added to it: a full stop's pause is
    this gap and nothing else, so `pause_scale` cannot double-count it.
    """
    raw = params.get("sentence_gap")
    if raw is None:
        base = PAUSE_AFTER.get(str(mark), PAUSE_AFTER["."])
    else:
        try:
            base = float(raw)
        except (TypeError, ValueError):
            base = PAUSE_AFTER["."]
    # Capped to match the editor's own seam ceiling (generative_editor.SEAM_CEILING): the
    # gap between sentences compounds over a whole chapter, so it is the one place the
    # slider is deliberately not linear. Mid-sentence marks are capped too, in proportion -
    # see _pause_for for what leaving them uncapped did to the punctuation hierarchy.
    return min(max(0.0, base * _pause_scale(params)), SEAM_CEILING)


def _quiet_point(
    audio, want: int, search: int, lo: int, hi: int, env: int, reach: int, back: int
) -> int:
    """The best sample to cut at: the quietest PLACE near `want`, not merely the
    quietest sample.

    The first version searched a couple of milliseconds for the minimum of |x|,
    which on voiced material is a zero crossing - that stops the cut clicking,
    but every period of a vowel has a zero crossing, so it would happily cut
    straight through the middle of one. "We, we, unbearably we." exposed it: the
    model's token boundary for a short word ending in a vowel sits at 27% of that
    word's own peak, so the silence went in mid-vowel and the vowel resumed after
    it. Heard as the word being truncated and strange, which is exactly what it
    was - a hole punched in a vowel.

    So the search is two-stage. First find the quietest ENVELOPE (a short moving
    average of |x|) over a much wider window - that lands in the real gap between
    the words rather than on an arbitrary zero of the carrier. Then refine to the
    nearest zero crossing inside it, which is what keeps the edges click-free.

    `hi` bounds the forward reach at the next token's start, so a pause can never
    migrate into the following word.
    """
    import numpy as np

    n = int(audio.size)
    want = min(max(int(want), 0), n)
    lo = max(0, min(int(lo), n))
    top = min(n, max(int(hi), 0)) if hi else n
    # NEVER EARLIER THAN THE NOMINAL POINT. A cut before the mark's own token ends would put
    # the silence inside that token's own word, and would shift the token by its own pause.
    # The gap we are looking for is always at or after the boundary, so `back` is only ever a
    # refinement allowance, never a licence to precede it.
    # THE CUT LIVES IN [token end, next token start], and nowhere else.
    #   - never EARLIER than the nominal end, or the silence lands inside the mark's own word
    #     and that word shifts by its own pause;
    #   - never LATER than the next token's start, or the silence lands inside the FOLLOWING
    #     word, and the shift rule (a span's start is inclusive, its end is not) stops lining up.
    # With no bound from the caller there is no safe room to move at all, so the cut stays put -
    # every production call site passes the bound.
    if not hi:
        return max(want, lo)
    low = max(lo, want - max(0, back))
    high = min(top, want + max(1, reach))
    if low >= high:
        return max(want, lo)
    # Stage 1: envelope trough. Cumulative sum gives the moving average in one pass.
    w = max(1, int(env))
    mag = np.abs(audio[low:high]).astype(np.float64)
    if mag.size > w:
        c = np.concatenate(([0.0], np.cumsum(mag)))
        smooth = (c[w:] - c[:-w]) / float(w)
        centre = low + w // 2 + int(np.argmin(smooth))
    else:
        centre = low + int(np.argmin(mag))
    # Stage 2: nearest zero crossing to that trough, so neither new edge steps.
    zl = max(lo, want, centre - search)
    zh = min(top, centre + search + 1)
    if zl >= zh:
        return max(centre, lo)
    win = np.abs(audio[zl:zh])
    best = float(win.min())
    cand = np.nonzero(win <= best + 1e-6)[0] + zl
    return int(cand[np.argmin(np.abs(cand - centre))])


def _ramp(seg, n: int, fade_in: bool, fade_out: bool) -> None:
    """Raised-cosine fade at the edges of `seg`, in place."""
    import numpy as np

    k = min(int(n), int(seg.size))
    if k <= 0:
        return
    w = (0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)).astype(seg.dtype)
    if fade_in:
        seg[:k] *= w
    if fade_out:
        seg[seg.size - k :] *= w[::-1]


def _splice_pauses(audio, points, sr: int):
    """Insert real silence INTO an utterance without re-synthesizing it.

    `points` is [(time_seconds, pause_seconds)], ascending; each time is the
    NOMINAL end of the token or phone carrying the mark. Returns
    `(audio, inserted)` where `inserted` is [(nominal_time, actual_seconds)] and
    the actual seconds are sample-exact, so a timing shifted by them can never
    drift away from the waveform. With nothing to insert the input array is
    returned untouched - byte-identical, not merely equal.

    NOT a sentence split. Splitting at every comma would make each clause its own
    utterance and hand it sentence-final intonation, which is a worse defect than
    the one being fixed. The sentence is synthesized whole, prosody intact, and
    the silence is spliced in afterwards.

    CLICKS. Cutting a waveform at an arbitrary sample and butting digital zero
    against it is a step discontinuity, which is a click - this project has been
    bitten by exactly that before. Two cheap defences, both applied:
      1. the cut moves up to SPLICE_SEARCH_MS to the quietest sample in reach,
         i.e. the nearest zero crossing on voiced material, so the two new edges
         are already near zero;
      2. a SPLICE_FADE_MS raised-cosine ramp out of the speech and back into it,
         which removes the residual slope discontinuity the zero crossing leaves.
    The ramp is only ever applied to samples the search has already established
    are near-silent, so it costs no audible speech; alone it would be enough, but
    2 ms of ramp on a loud sample is itself faintly audible, hence both.
    """
    import numpy as np

    # A point may carry a third field: the time of the next token, which bounds how far
    # the cut may slide forward. Two-field points (the older form, and the tests) mean
    # 'no bound'.
    pts = [
        (float(p[0]), float(p[1]), (float(p[2]) if len(p) > 2 else 0.0))
        for p in points
        if float(p[1]) > 0.0
    ]
    if not pts:
        return audio, []
    pts.sort(key=lambda p: p[0])
    search = max(1, int(round(SPLICE_SEARCH_MS * sr / 1000.0)))
    env = max(1, int(round(SPLICE_ENV_MS * sr / 1000.0)))
    reach = max(1, int(round(SPLICE_REACH_MS * sr / 1000.0)))
    back = max(0, int(round(SPLICE_BACK_MS * sr / 1000.0)))
    fade = max(1, int(round(SPLICE_FADE_MS * sr / 1000.0)))

    pieces: list = []
    inserted: list = []
    prev = 0
    need_in = False
    for t, secs, limit in pts:
        pad = int(round(secs * sr))
        if pad <= 0:
            continue
        hi = int(round(limit * sr)) if limit > 0.0 else 0
        cut = max(
            prev,
            _quiet_point(audio, int(round(t * sr)), search, prev, hi, env, reach, back),
        )
        seg = np.array(audio[prev:cut], dtype=audio.dtype, copy=True)
        _ramp(seg, fade, need_in, True)
        pieces.append(seg)
        pieces.append(np.zeros(pad, dtype=audio.dtype))
        # The shift is keyed to where the silence ACTUALLY went, not to the nominal
        # mark: the cut may now slide tens of milliseconds to reach the gap between
        # the words, and a timing shifted from the nominal point would drift off the
        # waveform by that much. `hi` keeps the cut at or before the next token's
        # start, so the mark's own token still ends before it and only what follows
        # moves.
        # max(): the cut is at or after the mark in SAMPLES, but a sample index divided by
        # the rate can land a hair BELOW the nominal seconds (round(0.25*22050)/22050 =
        # 0.24997), and that hair flips the end-exclusive comparison in _shift - the mark's
        # own token would then be shifted by its own pause. Never report earlier than the
        # mark; report the real position whenever the cut genuinely moved.
        # Clamped into [mark, next token start] for the same rounding reason at BOTH ends:
        # a sample index converted back to seconds can land a hair either side of the
        # boundary it was derived from, and either hair flips a comparison in _shift -
        # below the mark shifts the mark's own token, above the next token's start stops
        # that token shifting at all.
        at = max(t, cut / float(sr))
        if limit > 0.0:
            at = min(at, limit)
        inserted.append((at, pad / float(sr)))
        prev, need_in = cut, True
    if not inserted:
        return audio, []
    tail = np.array(audio[prev:], dtype=audio.dtype, copy=True)
    _ramp(tail, fade, need_in, False)
    pieces.append(tail)
    return np.concatenate(pieces), inserted


_warned_unaligned = False


def _warn_unaligned(marks) -> None:
    """Say so, once, when a mid-sentence pause was asked for and cannot be placed.

    Splicing needs the duration predictor's spans to know WHERE the mark ends.
    An unpatched voice has none - the same condition that costs the subtitles -
    and the pause would otherwise just quietly not happen, which is exactly the
    complaint this feature exists to answer.
    """
    global _warned_unaligned
    if _warned_unaligned or not marks:
        return
    _warned_unaligned = True
    print(
        "ghost/voice: no alignment output from this voice, so pauses after "
        + " ".join(sorted(set(marks)))
        + " cannot be placed (sentence pauses are unaffected)",
        file=sys.stderr,
    )


def _shift(t: float, inserted, inclusive: bool) -> float:
    """`t` moved later by every pause inserted before it.

    A span's START is inclusive (a pause landing exactly on it happened first,
    so the span begins after it) and its END is not (a pause landing exactly on
    the end belongs after the span, which is what puts the mark's own token in
    front of its own silence instead of underneath it).
    """
    add = 0.0
    for at, dur in inserted:
        if at < t or (inclusive and at <= t):
            add += dur
    return t + add


def _espeak_word(text: str) -> str:
    """A token as eSpeak should SEE it: an internal hyphen is a word boundary.

    A HYPHEN INSIDE A WORD IS NOT PUNCTUATION AND IT IS NOT SILENT EITHER - it is
    the boundary between two words that are spelled as one, and a reader says
    "ten-forty" exactly the way they say "ten forty". ghost keeps the hyphen all
    the way here on purpose (the karaoke line shows the source spelling, so
    "twenty-five" must not become "twenty five" on screen - see
    TextNorm._expand_core), which left this the one place that has to turn the
    spelling back into a boundary.

    Handing the hyphenated spelling straight to the phonemizer did not, because
    eSpeak returns the SAME PHONES either way and only the word space differs:

        ten-forty     -> tˈɛnfˈɔːɹɾi        ten forty     -> tˈɛn fˈɔːɹɾi
        forty-second  -> fˈɔːɹɾisˈɛkənd     forty second  -> fˈɔːɹɾi sˈɛkənd
        self-report   -> sˈɛlfɹᵻpˈɔːɹt      self report   -> sˈɛlf ɹᵻpˈɔːɹt

    so this changes no pronunciation anywhere - it restores a boundary that was
    being dropped. And the model does not ignore that boundary. Two primary
    stresses welded together with no space between them is a shape the training
    data does not contain, and en_US-libritts-high answers it by opening a hole
    in the middle of the word. Measured, longest near-silence INSIDE the token,
    averaged over three renders, glued vs spaced:

        forty-second  0.40 s -> 0.07 s      x-ray        0.30 s -> 0.06 s
        twenty-five   0.25 s -> 0.07 s      night-light  0.23 s -> 0.03 s

    which is the reported "the hyphen forces a pause between each word". The
    other four installed voices never opened the hole, so this had been sitting
    under whichever voice was selected.

    A compound eSpeak already reads with ONE primary ("re-enter" -> ɹˌiːˈɛntɚ)
    was never affected and is unaffected by this: it gets the same phones and
    one more space.

    Only the ASCII hyphen, because that is the only one that survives TextNorm -
    it folds the typographic hyphens to it and an em dash to a comma. A token
    that is nothing BUT dashes never arrives (phonemes.gd turns a spaced dash
    into punctuation) but is returned untouched if one ever does, since handing
    the phonemizer an empty string drops the item and takes the alignment of
    every later word with it.
    """
    spoken = " ".join(text.replace("-", " ").split())
    return spoken or text.strip()


# Phoneme id stream shape, from piper1-gpl docs/ALIGNMENTS.md:
#   [BOS, PAD, id, PAD, id, PAD, ..., EOS]
BOS, EOS, PAD = 1, 2, 0

# Word-spaces prepended to every utterance so the model's onset ramp does not land on
# the first real word. Two, by measurement - see the table in _symbols.
LEAD_IN_SPACES = 2

# U+0329 COMBINING VERTICAL LINE BELOW - the IPA "syllabic" mark. eSpeak puts it on a
# consonant that is carrying a syllable on its own, which in American English is mostly
# the glottalized -ten/-tain family: "written" comes back as /ɹˈɪʔn̩/, "certain" as
# /sˈɜːʔn̩/, "gotten" as /ɡˈɑːʔn̩/.
#
# Not every voice can spell it. Measured across the five installed here, four carry
# 157-symbol maps that include it and en_US-libritts-high carries a 130-symbol map that
# does not - and on that voice the mark was simply dropped, leaving /ɹˈɪʔn/: a glottal
# stop and a bare consonant with nothing to stand as the second syllable. Across
# north-star that is 146 tokens of 19 word types, and they are not rare words - "written"
# 51 times, "certain" 34, "gotten" 15.
#
# So substitute rather than drop. A syllabic consonant IS a schwa plus that consonant -
# it is the same sound written more tightly, and eSpeak itself spells the un-glottalized
# members of the family that way already ("prison" -> /pɹˈɪzən/, "sudden" -> /sˈʌdən/).
# Inserting the schwa BEFORE the base gives /ɹˈɪʔən/, which every voice can say.
SYLLABIC = "\u0329"
SCHWA = "\u0259"

# Commercially clean English voices only. Verified MODEL_CARD by MODEL_CARD,
# 2026-08-09. Anything not on this list needs its card read before it is added.
VOICES: dict[str, dict] = {
    "en_US-ljspeech-medium": {
        "path": "en/en_US/ljspeech/medium/en_US-ljspeech-medium",
        "license": "Public domain (LJ Speech)",
        "chain": "trained from scratch",
        "speakers": 1,
        "notes": "Single speaker, small, the quickest thing to prove the pipeline.",
    },
    "en_US-libritts-high": {
        "path": "en/en_US/libritts/high/en_US-libritts-high",
        "license": "CC BY 4.0 (LibriTTS, openslr.org/60) - attribution required",
        "chain": "trained from scratch",
        "speakers": 904,
        "notes": "904 speakers under one checkpoint. The natural analogue of the "
        "fishing game's genome: identity becomes a speaker id you can "
        "throw, catch and interpolate.",
    },
    "en_US-kristin-medium": {
        "path": "en/en_US/kristin/medium/en_US-kristin-medium",
        "license": "Public domain (LibriVox, ~11.5h)",
        "chain": "trained from scratch",
        "speakers": 1,
    },
    "en_US-norman-medium": {
        "path": "en/en_US/norman/medium/en_US-norman-medium",
        "license": "Public domain (LibriVox, ~15.5h)",
        "chain": "trained from scratch",
        "speakers": 1,
    },
    "en_US-john-medium": {
        "path": "en/en_US/john/medium/en_US-john-medium",
        "license": "Public domain (LibriVox, ~12.5h)",
        "chain": "fine-tuned from kristin (clean chain)",
        "speakers": 1,
    },
}


@register
class PiperBackend(Backend):
    name = "piper"

    @classmethod
    def describe(cls) -> dict:
        d = super().describe()
        d.update(
            {
                "streaming": False,  # per-sentence today; sub-utterance is possible
                "phoneme_input": True,  # the point of this backend
                "duration_control": True,  # length_scale
                "pitch_control": False,  # VITS has no direct f0 handle
                "timings": True,  # exact, from the duration predictor
                "reference_audio": False,
                "singing": False,
            }
        )
        return d

    @classmethod
    def owns(cls, voice: str) -> bool:
        return voice in VOICES

    def __init__(self) -> None:
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise BackendError(
                "onnxruntime/numpy are not installed in the voice environment"
            ) from exc
        import homographs  # ghost's own; HERE is on sys.path - see host.py
        import vowel_probe

        self._sessions: dict[str, Any] = {}
        self._configs: dict[str, dict] = {}
        # Homograph resolution phonemizes through this backend's OWN word-level
        # call, so a carrier reading and a bare reading are always comparable.
        # Constructed here rather than per request because its probe cache is the
        # thing that makes it free after the first few sentences.
        self._homographs = homographs.Homographs(self._espeak)
        # Whether a checkpoint says the vowel its phoneme string names. Constructed
        # here for the same reason: the verdict is per voice and the probe RENDERS, so
        # it is asked once and then remembered - in memory here,
        # and on disk beside the weights so a restart does not pay for it again.
        self._vowels = vowel_probe.VowelProbe(
            self._espeak, self._render_symbols, self._root()
        )

    # -- storage -----------------------------------------------------------

    @staticmethod
    def _root() -> Path:
        # Weights never live in the repo. This mirrors user:// on the Godot
        # side and is created on demand.
        root = Path.home() / ".local" / "share" / "ghost" / "voices" / "piper"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _files(self, voice: str) -> tuple[Path, Path]:
        root = self._root()
        return root / f"{voice}.onnx", root / f"{voice}.onnx.json"

    def voices(self) -> list[dict]:
        out = []
        for vid, meta in VOICES.items():
            onnx, cfg = self._files(vid)
            out.append(
                {
                    "id": vid,
                    "name": vid.replace("en_US-", "").replace("-", " "),
                    "backend": self.name,
                    "license": meta["license"],
                    "derivation": meta["chain"],
                    "speakers": meta["speakers"],
                    "installed": onnx.exists() and cfg.exists(),
                    "notes": meta.get("notes", ""),
                }
            )
        return out

    def ensure(self, voice: str) -> dict:
        if voice not in VOICES:
            raise BackendError(f"unknown or non-allowlisted voice '{voice}'")
        onnx, cfg = self._files(voice)
        if onnx.exists() and cfg.exists():
            return {"voice": voice, "installed": True, "downloaded": False}
        base = f"{HF_BASE}/{VOICES[voice]['path']}"
        for url, dest in ((f"{base}.onnx", onnx), (f"{base}.onnx.json", cfg)):
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, tmp)
            except Exception as exc:  # noqa: BLE001
                tmp.unlink(missing_ok=True)
                raise BackendError(f"could not fetch {url}: {exc}") from exc
            tmp.replace(dest)  # atomic: a reader sees whole file or none
        return {
            "voice": voice,
            "installed": True,
            "downloaded": True,
            "bytes": onnx.stat().st_size + cfg.stat().st_size,
        }

    # -- model -------------------------------------------------------------

    def _load(self, voice: str):
        if voice in self._sessions:
            return self._sessions[voice], self._configs[voice]
        self.ensure(voice)
        import onnxruntime as ort

        onnx, cfgp = self._files(voice)
        self._ensure_aligned(onnx)
        cfg = json.loads(cfgp.read_text())
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # the protocol owns stdout
        sess = ort.InferenceSession(
            str(onnx), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sessions[voice] = sess
        self._configs[voice] = cfg
        # ASK THIS CHECKPOINT WHETHER IT SAYS THE VOWEL ITS PHONEME STRING NAMES.
        # en_US-libritts-high renders a word-initial `ɹˈiːd` with the vowel of `rid`,
        # which is its training transcripts carrying the same homograph bug one level
        # down - see vowel_probe.py. Here rather than in `_run` for two reasons: it is
        # a property of the weights, so once per load is the right cadence and a check
        # on every sentence is not; and it RENDERS, so it must not happen inside one.
        # Being on the cold path only, it also never touches a session a caller
        # supplied itself - the alignment tests seed `_sessions` with a stand-in whose
        # phoneme map allocates ids on lookup, and probing that would silently move
        # every id in the audio under test.
        self._vowels.measure(
            voice, cfg, sess, str(cfg.get("espeak", {}).get("voice", "en-us"))
        )
        return sess, cfg

    @staticmethod
    def _ensure_aligned(onnx: Path) -> None:
        """Expose the duration predictor, once, on first load of a voice.

        Stock Piper graphs return audio only. Without the duration tensor there
        are no phoneme times, no word times, and therefore NO SUBTITLES - which
        is how a voice that had been hand-patched during development worked
        while every other voice silently had no overlay at all. Patching belongs
        here, where a voice is first used, not in a script someone has to
        remember to run.

        Idempotent and cheap: the check is one session open, and a patched graph
        is left alone.
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        probe = ort.InferenceSession(
            str(onnx), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        if len(probe.get_outputs()) > 1:
            return
        del probe
        try:
            import onnx as onnx_mod
        except ImportError:
            # no `onnx` package: synthesis still works, subtitles do not
            print(
                f"ghost/voice: {onnx.name} has no alignment output and the "
                "`onnx` package is missing - subtitles unavailable",
                file=sys.stderr,
            )
            return
        model = onnx_mod.load(str(onnx))
        ceil_nodes = [n for n in model.graph.node if n.op_type == "Ceil"]
        if len(ceil_nodes) != 1:
            print(
                f"ghost/voice: {onnx.name} has {len(ceil_nodes)} Ceil nodes, "
                "expected 1 - not patching",
                file=sys.stderr,
            )
            return
        tensor = ceil_nodes[0].output[0]
        model.graph.output.append(
            onnx_mod.helper.make_tensor_value_info(
                tensor, onnx_mod.TensorProto.FLOAT, None
            )
        )
        tmp = onnx.with_suffix(onnx.suffix + ".part")
        onnx_mod.save(model, str(tmp))
        tmp.replace(onnx)  # atomic: a reader sees whole graph or old graph
        print(f"ghost/voice: patched {onnx.name} for alignment", file=sys.stderr)

    # -- synthesis ---------------------------------------------------------

    # -- phonemization --------------------------------------------------

    _espeak_ready = False
    _warned_symbols: set = set()

    @classmethod
    def _espeak(cls, words: list[str], voice: str = "en-us") -> list[str]:
        """eSpeak's IPA for a list of words, in the voice's OWN eSpeak language.

        Word by word rather than whole-sentence, because ghost needs to know
        which phones belong to which word to place the karaoke subtitles, and
        eSpeak gives no word boundaries in a sentence-level transcription.

        THE LANGUAGE MUST COME FROM THE VOICE CONFIG, not from the voice's name.
        en_US-ljspeech-medium declares `espeak.voice = "en"`, which eSpeak
        resolves to BRITISH English - so the model was trained on RP symbols
        despite the American name and audio. Hardcoding "en-us" fed it æ, ʌ, ɑː
        and oʊ where it expected a, ɒ, ɒ and əʊ, and it rendered them as the
        nearest thing it knew: "lamp" came out "lump", "not" came out "nart",
        "was" came out "wars", "balanced" came out "bullenced". Every one of
        those was this single line.
        """
        if not cls._espeak_ready:
            try:
                import espeakng_loader
                from phonemizer.backend.espeak.wrapper import EspeakWrapper

                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            except ImportError as exc:
                raise BackendError(
                    'eSpeak phonemizer unavailable; set phonemizer="ghost" to '
                    "use ghost's own ARPAbet front end instead"
                ) from exc
            cls._espeak_ready = True
        from phonemizer import phonemize
        from phonemizer.separator import Separator

        # phonemizer validates against its own language list, which does not
        # include eSpeak's bare "en" - and "en" IS British in eSpeak, which is
        # the whole point of reading it from the config.
        lang = {"en": "en-gb", "en-uk": "en-gb"}.get(voice, voice)
        # EACH ITEM OF `words` IS ITS OWN UTTERANCE - a list is a batch, not a
        # sentence - and callers pass one word per item, so this is the isolated
        # reading by construction. That is deliberate (the docstring above says
        # why: alignment) and it is also what costs the homograph: eSpeak reads
        # "live" as laɪv alone and lɪv in "where they live", so ghost was getting
        # the isolated reading for every occurrence. The separator is what makes
        # a MULTI-word item splittable, which is how homographs.py asks the same
        # question again with the syntax attached - it is not sentence context
        # here.
        out = phonemize(
            words,
            language=lang,
            backend="espeak",
            strip=True,
            with_stress=True,
            njobs=1,
            separator=Separator(word=" ", phone=""),
        )
        return [o.strip() for o in out]

    def _symbols(
        self,
        tokens: list,
        phonemizer: str,
        espeak_voice: str = "en-us",
        voice: str = "",
    ) -> list:
        """(codepoint, token index) pairs for a chunk.

        A token carries its source text AND, optionally, ghost's own ARPAbet.
        Inline [K AE T] overrides arrive with `arpa` set and ALWAYS win - that
        is the whole point of the override, and it is how invented proper nouns
        stay pronounceable whichever phonemizer is in use.

        A token may also carry `ipa`, which is eSpeak's OWN reading of that word
        under the part of speech it was tagged with (homographs.py). It sits
        below `arpa` - the author still outranks it - and above the word-by-word
        transcription it exists to correct.

        Whichever of the three a reading comes from, it is written out in the spelling
        THIS checkpoint was measured to render correctly before it leaves here (see
        vowel_probe.py). That applies to the authored override too, and deliberately:
        the repair does not change which vowel is named, only how it is spelled for a
        model that mis-renders the short spelling, so an author who writes [R IY1 D]
        wants the same repair an eSpeak reading gets. `voice` empty - every caller
        that is not a real render - leaves every reading untouched.
        """
        import arpabet

        def written(ipa: str) -> str:
            return self._vowels.repair_for(voice, ipa) if voice else ipa

        # A token whose text is only whitespace must never reach the batch, and the
        # result must be checked rather than trusted. phonemizer DROPS an empty or
        # whitespace-only input instead of returning "" for it - measured, [" ", "the"]
        # comes back as ["ðə"], one item for two - and the zip below then pairs every
        # remaining word with its NEIGHBOUR'S phonemes and lets the last one fall
        # through to ghost's ARPAbet. Silently. The whole tail of a sentence would be
        # spoken one word out of step, and the one word that reached the fallback would
        # take CMUdict's reading, which for a homograph is not the same reading eSpeak
        # would have given.
        need = [
            i
            for i, t in enumerate(tokens)
            if not t.get("arpa") and not t.get("ipa") and str(t.get("text", "")).strip()
        ]
        espoke: dict[int, str] = {}
        if need and phonemizer == "espeak":
            words = [_espeak_word(str(tokens[i]["text"])) for i in need]
            got = self._espeak(words, espeak_voice)
            if len(got) != len(words):
                # Alignment is not recoverable from a short batch - there is no way to
                # know WHICH one was dropped - so redo it one word at a time. Slower,
                # and it happens on approximately no sentences, but a wrong-by-one
                # sentence is not something to ship for the sake of one call.
                print(
                    "ghost/voice: phonemizer returned %d results for %d words; "
                    "re-running individually to keep alignment"
                    % (len(got), len(words)),
                    file=sys.stderr,
                )
                got = [(self._espeak([w], espeak_voice) or [""])[0] for w in words]
            espoke = dict(zip(need, got))

        # LEAD-IN. The model starts its utterance at the very first phoneme and its onset
        # ramp lands ON that phoneme rather than before it, so the opening word comes out
        # short and weak - weak enough to be heard as missing ("the voice starts at
        # cartoon"; later, "The is skipped entirely"). Word-spaces ahead of the first
        # token give the onset somewhere to happen that is not a word.
        #
        # HOW MANY was measured, against a criterion that does not depend on taste: the
        # opening word should be no weaker than the SAME word later in the same sentence.
        # Comparing token 0 against token 4 of "The spoon gone past the gills." - both
        # "the" - over 8 runs each, which averages out the scatter of a stochastic
        # duration predictor:
        #
        #   lead   initial "the"   duration vs mid   amplitude vs mid
        #     0        76 ms            0.70              1.30
        #     1        97 ms            0.91              0.67    <- was here
        #     2       122 ms            1.06              0.95    <- parity on both axes
        #     3       151 ms            1.28              1.18    <- overshoots, drawls
        #
        # One space fixed the DURATION and left the amplitude at two thirds, which is why
        # the word kept being reported as dropped after the first attempt: it was the
        # right diagnosis and half the dose. Two reaches parity on both axes. Three makes
        # the opening word longer and louder than the same word mid-sentence, which reads
        # as a drawl on the first syllable of every utterance. The cost is a few tens of
        # ms of silence at the head, which the chunk seam was already providing anyway.
        out: list = [(" ", 0)] * LEAD_IN_SPACES
        for i, t in enumerate(tokens):
            if t.get("arpa"):
                got = "".join(
                    ch for ch, _ in arpabet.to_symbols([str(x) for x in t["arpa"]])
                )
                for ch in written(got):
                    out.append((ch, i))
            elif t.get("ipa"):
                for ch in written(str(t["ipa"])):
                    out.append((ch, i))
            elif i in espoke:
                for ch in written(espoke[i]):
                    out.append((ch, i))
            elif t.get("text"):
                # no eSpeak: fall back to ghost's ARPAbet if it sent any
                got = "".join(
                    ch
                    for ch, _ in arpabet.to_symbols(
                        [str(x) for x in t.get("fallback", [])]
                    )
                )
                for ch in written(got):
                    out.append((ch, i))
            for ch in str(t.get("punct", "")):
                out.append((ch, i))
            out.append((" ", i))
        return out

    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: str,
        params: dict[str, Any],
        phonemes: Any = None,
    ) -> dict:
        """Synthesize SENTENCE BY SENTENCE, joined with real silence.

        A whole paragraph handed over as one utterance comes back rushed: the
        model has no reason to breathe at a full stop it is in the middle of,
        and the reported symptom was exactly that - "the sentences feel rushed,
        like the model should have hesitated before starting a new one".
        Splitting also keeps each inference short, which is what makes a
        streaming path possible later.
        """
        import numpy as np

        sess, cfg = self._load(voice)
        pmap: dict = cfg["phoneme_id_map"]
        sample_rate = int(cfg["audio"]["sample_rate"])

        tokens = params.get("tokens")
        if tokens:
            return self._synth_tokens(list(tokens), voice, out_path, params, cfg, sess)
        sentences = _split_sentences(list(phonemes))
        if len(sentences) > 1:
            chunks, phones_all, cursor = [], [], 0.0
            for si, sent in enumerate(sentences):
                sub = self.synthesize(
                    text,
                    voice,
                    out_path + f".s{si}",
                    {**params, "_no_split": True},
                    sent,
                )
                import wave as _w

                with _w.open(sub["wav"]) as fh:
                    import numpy as _np

                    a = (
                        _np.frombuffer(fh.readframes(fh.getnframes()), "<i2").astype(
                            _np.float32
                        )
                        / 32768.0
                    )
                Path(sub["wav"]).unlink(missing_ok=True)
                for ph in sub.get("phones", []):
                    phones_all.append(
                        {
                            "p": ph["p"],
                            "t0": round(ph["t0"] + cursor, 4),
                            "t1": round(ph["t1"] + cursor, 4),
                        }
                    )
                chunks.append(a)
                cursor += len(a) / sub["sample_rate"]
                if si < len(sentences) - 1:
                    import numpy as _np

                    # the mark this sentence ended on decides its own pause
                    gap = _gap_for(str(sent[-1]).strip() if sent else "", params)
                    pad = int(round(gap * sub["sample_rate"]))
                    if pad > 0:
                        chunks.append(_np.zeros(pad, dtype=_np.float32))
                        cursor += pad / sub["sample_rate"]
            import numpy as _np

            joined = (
                _np.concatenate(chunks) if chunks else _np.zeros(1, dtype=_np.float32)
            )
            sr = int(cfg["audio"]["sample_rate"])
            self._write_wav(out_path, joined, sr)
            return {
                "wav": out_path,
                "sample_rate": sr,
                "duration": float(joined.size) / sr,
                "phones": phones_all,
                "aligned": bool(phones_all),
                "sentences": len(sentences),
            }

        if not phonemes:
            raise BackendError(
                "this backend takes phonemes, not text: ghost's own G2P supplies "
                "them, which is what keeps eSpeak-NG (GPL) out of the build"
            )

        # [BOS, PAD, id, PAD, ..., EOS]. Track which ids came from which input
        # phoneme so the durations can be folded back into words later.
        # ghost sends its own ARPAbet. Translating here rather than in Godot
        # keeps the [K AE T] override working on this path and keeps eSpeak-NG
        # out of the build entirely - see arpabet.py.
        import arpabet

        symbols = arpabet.to_symbols(list(phonemes))

        # [BOS, PAD, id, PAD, id, PAD, ..., EOS] - the pad comes BEFORE each id
        # and after the last one, per piper1-gpl docs/ALIGNMENTS.md. Emitting it
        # AFTER instead drops the pad that follows BOS, which shifts the whole
        # stream by one relative to what the model was trained on.
        # (ghost's own ARPAbet carries no syllabic mark, so this is a no-op here -
        # it runs for symmetry with the eSpeak path rather than because it fires.)
        symbols, _folded = self._fold_syllabic(symbols, pmap)
        ids: list[int] = [BOS]
        owner: list[int] = [-1]
        missing: list[str] = []
        for sym, src in symbols:
            mapped = pmap.get(sym)
            if mapped is None:
                missing.append(sym)
                continue
            ids.append(PAD)
            owner.append(src)
            for m in mapped:
                ids.append(int(m))
                owner.append(src)
        ids.append(PAD)
        owner.append(-1)
        ids.append(EOS)
        owner.append(-1)

        if missing:
            # Loud, not silent. A symbol absent from phoneme_id_map is dropped
            # by the encoder without complaint, and that exact failure mode
            # deleted every numeral from ghost's own front end for months.
            raise BackendError(
                "phonemes not in this voice's phoneme_id_map: "
                + " ".join(sorted(set(missing))[:12])
                + "  (regenerate the ARPAbet mapping against this voice)"
            )

        inf = cfg.get("inference", {})
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array(
                [
                    float(params.get("noise_scale", inf.get("noise_scale", 0.667))),
                    float(params.get("length_scale", inf.get("length_scale", 1.0))),
                    float(params.get("noise_w", inf.get("noise_w", 0.333))),
                ],
                dtype=np.float32,
            ),
        }
        if int(cfg.get("num_speakers", 1)) > 1:
            feeds["sid"] = np.array([int(params.get("speaker", 0))], dtype=np.int64)

        outputs = sess.run(None, feeds)
        audio = np.asarray(outputs[0]).squeeze().astype(np.float32)

        phone_times = []
        if len(outputs) > 1:
            phone_times = self._durations(outputs[1], owner, phonemes, sample_rate)

        # Mid-sentence marks arrive on this path as phones of their own (see
        # arpabet.PUNCT_PASSTHROUGH), so each one already has a span to splice
        # after. Needs the durations: without them there is no place to cut, and
        # that is the same condition under which there are no subtitles either.
        points, unplaceable = [], []
        for i, sym in enumerate(phonemes):
            mark = str(sym).strip()
            if mark in SENTENCE_END or _pause_for(mark, params) <= 0.0:
                continue  # . ! ? are the gap between sentences, above
            if i < len(phone_times):
                nxt = phone_times[i + 1]["t0"] if i + 1 < len(phone_times) else 0.0
                points.append(
                    (float(phone_times[i]["t1"]), _pause_for(mark, params), float(nxt))
                )
            else:
                unplaceable.append(mark)
        _warn_unaligned(unplaceable)
        if points:
            audio, inserted = _splice_pauses(audio, points, sample_rate)
            for ph in phone_times:
                ph["t0"] = round(_shift(ph["t0"], inserted, True), 4)
                ph["t1"] = round(_shift(ph["t1"], inserted, False), 4)

        self._write_wav(out_path, audio, sample_rate)
        return {
            "wav": out_path,
            "sample_rate": sample_rate,
            "duration": float(audio.size) / sample_rate,
            "phones": phone_times,
            "aligned": bool(phone_times),
        }

    def _synth_tokens(
        self, tokens: list, voice: str, out_path: str, params: dict, cfg: dict, sess
    ) -> dict:
        """Token path: sentence-split, phonemize, synthesize, join.

        Tokens carry their own text, so the sentence split happens HERE on the
        punctuation ghost already parsed, and the phonemizer sees whole words
        with their boundaries intact.

        TWO KINDS OF PAUSE, one table. A sentence-final mark ends a group and its
        pause is the silence BETWEEN groups, exactly as before. Every other mark
        (, ; :) stays inside its group - the sentence is synthesized whole and
        the silence is spliced into the audio afterwards, so a comma never gets
        the falling intonation of a sentence end. Both are `PAUSE_AFTER` seconds
        times `pause_scale`.
        """
        import numpy as np

        phonemizer = str(params.get("phonemizer", "espeak"))
        groups: list[list] = []
        cur: list = []
        for t in tokens:
            cur.append(t)
            if str(t.get("punct", "")) in SENTENCE_END:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)

        sr = int(cfg["audio"]["sample_rate"])
        chunks: list = []
        times: list = []
        cursor = 0.0
        base = 0
        plan = _discourse_plan(groups, params)
        for gi, group in enumerate(groups):
            # THE SENTENCE'S OWN PROSODY. A pitch move is bought exactly the way the
            # Tone shift buys one: render `pr` times slower, then play back `pr`
            # times faster - so the sentence lands at its nominal duration and a
            # higher register, and the two halves of the plan stay independent.
            step = plan[gi]
            pr = 2.0 ** (float(step["semis"]) / 12.0)
            gp = dict(params)
            gp["length_scale"] = (
                float(
                    params.get(
                        "length_scale",
                        cfg.get("inference", {}).get("length_scale", 1.0),
                    )
                )
                * pr
                / float(step["rate"])
            )
            gp["noise_w"] = float(
                params.get("noise_w", cfg.get("inference", {}).get("noise_w", 0.333))
            ) * float(step["noise_w_mul"])
            audio, per_token = self._run(group, voice, gp, cfg, sess, phonemizer)
            # SAY WHICH WORDS CAME BACK WITHOUT A SPAN. ghost keeps them in the karaoke line
            # either way now (it interpolates their timing - see
            # GenerativeEditor._bridge_words), but a token the aligner cannot place is a real
            # fault: it means the phoneme stream carries nothing for that word, so it is very
            # likely not being SPOKEN either. That is how a year went missing from a chapter
            # render with nothing anywhere to show it.
            unplaced = [
                str(t.get("text", ""))
                for ti, t in enumerate(group)
                if ti not in per_token and str(t.get("text", "")).strip()
            ]
            if unplaced:
                print(
                    "ghost/voice: no alignment for %d of %d tokens in this sentence: %s"
                    % (len(unplaced), len(group), " ".join(unplaced[:8])),
                    file=sys.stderr,
                )
            audio = _effort(audio, sr, float(step["tilt"]), float(step["gain_db"]))
            if abs(pr - 1.0) > 1e-4:
                audio = _resample(audio, pr)
                # the alignment was measured before the resample, so it moves with it
                per_token = {k: (v[0] / pr, v[1] / pr) for k, v in per_token.items()}
            # mid-sentence marks: splice their silence into this group's audio
            points, unplaceable = [], []
            for ti, tok in enumerate(group):
                mark = str(tok.get("punct", ""))
                if mark in SENTENCE_END or _pause_for(mark, params) <= 0.0:
                    continue  # . ! ? are the gap between groups, below
                if ti in per_token:
                    # The third field bounds how far the cut may slide forward: the next
                    # token's start, so a pause can never end up inside the word after it.
                    nxt = per_token.get(ti + 1)
                    points.append(
                        (
                            float(per_token[ti][1]),
                            _pause_for(mark, params),
                            float(nxt[0]) if nxt is not None else 0.0,
                        )
                    )
                else:
                    unplaceable.append(mark)
            _warn_unaligned(unplaceable)
            audio, inserted = _splice_pauses(audio, points, sr)
            for ti, span in per_token.items():
                # float(): numpy scalars are not JSON-serializable and the
                # protocol is JSON
                times.append(
                    {
                        "index": base + ti,
                        "t0": round(_shift(float(span[0]), inserted, True) + cursor, 4),
                        "t1": round(
                            _shift(float(span[1]), inserted, False) + cursor, 4
                        ),
                    }
                )
            chunks.append(audio)
            cursor += len(audio) / sr
            base += len(group)
            if gi < len(groups) - 1:
                gap = _gap_for(str(group[-1].get("punct", "")) if group else "", params)
                pad = int(round(gap * sr))
                if pad > 0:
                    chunks.append(np.zeros(pad, dtype=np.float32))
                    cursor += pad / sr
        joined = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        self._write_wav(out_path, joined, sr)
        return {
            "wav": out_path,
            "sample_rate": sr,
            "duration": float(joined.size) / sr,
            "tokens": times,
            "aligned": bool(times),
            "sentences": len(groups),
        }

    @staticmethod
    def _fold_syllabic(symbols: list, pmap: dict) -> tuple[list, bool]:
        """Rewrite syllabic consonants this voice cannot spell as schwa + consonant.

        The mark FOLLOWS its base in the stream, so the schwa is inserted one place
        back - `[.., ("n", i), ("\u0329", i)]` becomes `[.., ("\u0259", i), ("n", i)]`.
        The inserted symbol keeps the base's source index, so per-token alignment (and
        the karaoke timing built on it) is unaffected.

        A no-op on the four voices whose maps do carry U+0329, and on any voice that
        somehow lacks a schwa there is nothing better to do than fall through to the
        existing drop.
        """
        # LOOK AT THE STREAM BEFORE LOOKING AT THE MAP. Two reasons, and the second is
        # the load-bearing one: most utterances contain no syllabic mark at all, so
        # probing is wasted work - and a phoneme_id_map is not always an inert dict.
        # The test harness's stand-in ALLOCATES AN ID on first lookup, so probing it for
        # two symbols that never appear silently shifted every subsequent id by two and
        # changed the synthesized audio. Cheap early-out, no side effects.
        if not any(sym == SYLLABIC for sym, _ in symbols):
            return symbols, False
        # .get() rather than `in`: that same stand-in supports lookup but not membership.
        if pmap.get(SYLLABIC) is not None or pmap.get(SCHWA) is None:
            return symbols, False
        out: list = []
        folded = False
        for sym, src in symbols:
            if sym == SYLLABIC:
                if out:
                    out.insert(len(out) - 1, (SCHWA, out[-1][1]))
                    folded = True
                # A mark with nothing in front of it has no base to be syllabic ON, so
                # it is discarded rather than turned into a stray leading schwa - which
                # would be an invented vowel at the head of the utterance.
                continue
            out.append((sym, src))
        return out, folded

    def _run(
        self, group: list, voice: str, params: dict, cfg: dict, sess, phonemizer: str
    ):
        """One sentence: symbols -> ids -> audio, plus per-token time spans."""
        espeak_voice = str(cfg.get("espeak", {}).get("voice", "en-us"))
        # Word by word, a homograph has no syntax to be read by, so eSpeak gives
        # its default reading every time - "he read the book" came out present
        # tense. Ask it again in a frame that forces the part of speech, and only
        # where the answer actually changes. See homographs.py; a no-op if nltk
        # is not installed. English only, because the carrier frames and the
        # tagger both are.
        if phonemizer == "espeak" and espeak_voice.split("-")[0] in ("en", ""):
            self._homographs.annotate(group, espeak_voice)
        # The reading is then written in the spelling this checkpoint was measured
        # to render correctly - see `_load`, which is where that measurement happens,
        # and vowel_probe.py for what it is measuring.
        symbols = self._symbols(group, phonemizer, espeak_voice, voice)
        return self._render_symbols(symbols, cfg, sess, params)

    def _render_symbols(self, symbols: list, cfg: dict, sess, params: dict):
        """(symbol, source) pairs -> audio, plus a time span per source.

        Split out of `_run` so the vowel probe can render a phoneme string it built
        itself and get spans back keyed to single SYMBOLS rather than to words - a
        vowel is a symbol, and there is no other way to measure one. Nothing here
        interprets the source index; it is carried through and handed back.
        """
        import numpy as np

        pmap: dict = cfg["phoneme_id_map"]
        # eSpeak is where U+0329 comes from, so this is the call that matters.
        symbols, folded = self._fold_syllabic(symbols, pmap)
        ids: list[int] = [BOS]
        owner: list[int] = [-1]
        missing: list[str] = []
        for sym, src in symbols:
            mapped = pmap.get(sym)
            if mapped is None:
                missing.append(sym)
                continue
            ids.append(PAD)
            owner.append(src)
            for m in mapped:
                ids.append(int(m))
                owner.append(src)
        ids.append(PAD)
        owner.append(-1)
        ids.append(EOS)
        owner.append(-1)
        if folded and SYLLABIC not in self._warned_symbols:
            self._warned_symbols.add(SYLLABIC)
            print(
                "ghost/voice: this voice has no U+0329 (syllabic); writing it as a "
                "schwa before the consonant instead",
                file=sys.stderr,
            )
        if missing:
            # DROP, do not fail. eSpeak emits marks that a given voice's map may
            # not carry, and raising costs the whole request - which is how a
            # cosmetic gap once became "request failed" and stopped playback dead.
            #
            # The one that actually mattered, U+0329, is no longer dropped at all;
            # it is rewritten above (see the SYLLABIC constant). Anything still
            # landing here is genuinely unhandled, so say so plainly rather than
            # describing it as a nuance.
            #
            # Reported once per symbol per process, to stderr, which ghost now
            # echoes to the terminal - visible without being fatal.
            for sym in sorted(set(missing)):
                if sym in self._warned_symbols:
                    continue
                self._warned_symbols.add(sym)
                print(
                    "ghost/voice: dropping U+%04X (%r), absent from this "
                    "voice's phoneme_id_map" % (ord(sym), sym),
                    file=sys.stderr,
                )

        inf = cfg.get("inference", {})
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array(
                [
                    float(params.get("noise_scale", inf.get("noise_scale", 0.667))),
                    float(params.get("length_scale", inf.get("length_scale", 1.0))),
                    float(params.get("noise_w", inf.get("noise_w", 0.333))),
                ],
                dtype=np.float32,
            ),
        }
        if int(cfg.get("num_speakers", 1)) > 1:
            feeds["sid"] = np.array([int(params.get("speaker", 0))], dtype=np.int64)
        out = sess.run(None, feeds)
        audio = np.asarray(out[0]).squeeze().astype(np.float32)

        spans: dict = {}
        if len(out) > 1:
            frames = np.asarray(out[1]).squeeze().astype(np.float64)
            if frames.ndim == 1 and frames.size == len(owner):
                t = 0.0
                for dur, who in zip(
                    frames * HOP_LENGTH / float(cfg["audio"]["sample_rate"]), owner
                ):
                    if who >= 0:
                        if who not in spans:
                            spans[who] = [t, t + dur]
                        else:
                            spans[who][1] = t + dur
                    t += dur
        return audio, spans

    @staticmethod
    def _durations(
        w_ceil, owner: list[int], phonemes: list, sample_rate: int
    ) -> list[dict]:
        """Per-phoneme boundaries from the VITS duration predictor.

        `w_ceil` is the ceiling of the stochastic duration predictor, one entry
        per phoneme id including pads and BOS/EOS, in frames. Multiply by the
        hop length for samples. This is the plan the synthesizer actually used,
        so it is exact - strictly better than running a second forced-alignment
        model, and free.

        Only present when the voice's ONNX has been patched to expose it (see
        patch_alignment.py). Absent is fine; timings are then unavailable.
        """
        import numpy as np

        frames = np.asarray(w_ceil).squeeze().astype(np.float64)
        if frames.ndim != 1 or frames.size != len(owner):
            return []
        samples = frames * HOP_LENGTH
        # fold every id's duration back onto the input phoneme that produced it,
        # pads included - a pad's time belongs to the phoneme it followed
        spans: dict[int, float] = {}
        for dur, who in zip(samples, owner):
            if who >= 0:
                spans[who] = spans.get(who, 0.0) + float(dur)
        out, cursor = [], 0.0
        for i, sym in enumerate(phonemes):
            dur = spans.get(i, 0.0) / sample_rate
            out.append(
                {"p": str(sym), "t0": round(cursor, 4), "t1": round(cursor + dur, 4)}
            )
            cursor += dur
        return out

    @staticmethod
    def _write_wav(path: str, audio, sample_rate: int) -> None:
        import wave
        import numpy as np

        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        tmp = path + ".part"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        Path(tmp).replace(path)  # atomic, same discipline as Voice.write_wav
