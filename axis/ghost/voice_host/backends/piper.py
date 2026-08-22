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
import math
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

# WHAT THE MODEL ALREADY RESTS at each mark, in seconds, measured with its own noise
# switched off so the number is the model's and not one sample of it. The comma, semicolon
# and colon figures are the token-span increments recorded above; the sentence-final one is
# the trailing silence of a rendered sentence (0.138-0.145 s here) plus the leading silence
# of the one after it (0.044), because with one sentence per request those two are what sit
# either side of a seam.
#
# THIS TABLE IS WHY THE PAUSE SLIDER GOT ITS PROPORTIONS BACK. What a reader hears at a mark
# is the model's rest plus ours, and only the second half was ever being scaled - so turning
# the dial up did not lengthen the rests, it lengthened the DIFFERENCES between them. At 1.0
# a full stop rests twice as long as a comma; under a linear scale that was 2.3x by 2.0,
# 2.8x by 6.0 and 2.9x at the top. Reported exactly that way: "the pause after a comma and
# the pause after a sentence are very different: the pause after a sentence seems to be much
# longer... at 6.0 the comma-pauses feel about right, while the period-pauses feel far too
# slow." Both halves of that sentence are one bug.
#
# THESE FIGURES ARE AN ESTIMATE AND ONLY THE SEAM USES THEM. The dwell is a property of the
# checkpoint, the pace and the sentence, and a table cannot be right about all three: the
# comma entry here says 0.15 s and en_US-john-medium actually rests 0.41 at the comma in
# "He waited, and the tide came in." Get it wrong and the dial scales the ERROR instead of
# the rest - the same bug one level down - so `_splice_pauses` MEASURES it in the waveform
# and this table is used only where there is no waveform to measure: the seam between two
# separately rendered sentences, which the editor inserts, and the accounting.
#
# The seam is the one place an estimate is safe, because it is the one dwell that does not
# vary much - the trailing silence of a render plus the leading silence of the next, both
# measured repeatedly at 0.138-0.145 and 0.044.
DWELL: dict[str, float] = {
    ",": 0.15,
    ";": 0.19,
    ":": 0.30,
    ".": 0.18,
    "!": 0.18,
    "?": 0.18,
}

# THE PAUSE CURVE. The dial multiplies the WHOLE rest - the model's own plus ours - so every
# mark keeps its share of the reading at every setting.
#
# A POWER LAW, not the saturating exponential this was first written as, and the difference
# is the whole of a report: "even completely maxxed-out at 10x, the pause effect barely seems
# to work with the Urgent tone... perhaps we need to allow for 100x pause". The instinct was
# right and the remedy would not have worked - an exponential that reaches 3.2 at the top of
# the dial reaches 3.28 at 20 and 3.29 at 100, so a bigger number buys nothing whatever. It
# was the CURVE that topped out, not the dial.
#
# `d ** PAUSE_GAIN` is exactly 1.0 at 1.0 by construction (no normalising constant to round
# and no default to drift, which the fitted pair before it managed to do), tracks the old
# curve within 3% up to 3.0 - the part nobody complained about - and then keeps climbing
# instead of flattening: 5.0x at the top of the dial, where a full stop rests two and a half
# seconds. The exponent is the reach, said once: log(5)/log(10).
#
# It is still concave, which is the other half of "the toggle feels very finicky": every unit
# of dial adds less than the one before it. Reach costs step size and that is the trade being
# made here - 5.0 to 6.0 moves a rest by 13% where the flatter curve moved it by 6%.
PAUSE_GAIN = 0.69897


def _pause_multiplier(params: dict) -> float:
    """How much longer than natural every rest in this reading is. 1.0 at Pause 1.0."""
    return _pause_scale(params) ** PAUSE_GAIN


def _rest_from(
    dwell: float, top_up: float, mult: float, have: float | None = None
) -> float:
    """How much silence to ADD at a mark, given `have` seconds of it already there.

    The whole pause rule, in two halves that must not be confused:

      THE TARGET is `(dwell + top_up) * mult` - the rest a reader should hear at this
      mark, which is the mark's natural rest at Pause 1.0 with the dial applied to the
      whole of it. It comes from the TABLE, so it is the same at every comma in a chapter
      and it keeps the same ratio to a full stop at every dial setting.

      WHAT IS ALREADY THERE is `have`, measured off the waveform by `_splice_pauses`, and
      it is subtracted, because we can only add.

    THOSE TWO USED TO BE THE SAME NUMBER AND THAT WAS THE BUG. The target was
    `(have + top_up) * mult`: the dial multiplied the dwell this particular comma happened
    to get. But the dwell is not a property of the checkpoint, it is a draw from its
    stochastic duration predictor - measured over four sentences of the reporter's chapter
    at his own settings, the same mark in the same voice came back 0.000, 0.000, 0.023,
    0.077, 0.320 and 0.401 seconds - so multiplying it by 3.62 turned a 0.4 s scatter into
    a 1.45 s one. The commas in one reading came out 0.36, 0.44, 1.52 and 1.81 seconds,
    against 1.77 for a full stop. Reported as "the pause after a comma feels twice as long
    as a pause after a sentence... the comma-pauses feel far too slow", and both halves of
    that are true of the same reading at different commas.

    The measurement stays where it belongs. `have` is a fact about this render and it is
    the only honest way to know how much of the target is already paid for - the table
    cannot answer that, which is what `_splice_pauses` learned the hard way. What it may
    not do is decide how long the rest should BE.

    `have` defaults to `dwell` for the callers that have no waveform to measure - the
    sentence seam and the accounting - where the two genuinely are the same number.

    Floored at zero: below about 0.6 on the dial the model is resting longer than the dial
    is asking for and there is nothing to take away. That is a real limit of splicing rather
    than a choice - shortening a rest would mean re-synthesizing the sentence.
    """
    # At 1.0, WHERE THERE IS NOTHING TO MEASURE, the answer IS the top-up, said exactly
    # rather than computed round the houses: `(0.18 + 0.5) * 1.0 - 0.18` is
    # 0.49999999999999994 in binary floating point, and the seam is the one figure here
    # that has to be reproducible to the sample.
    #
    # IT IS NOT A SHORTCUT WHERE THERE IS. `have` is what this render actually rests, so
    # `(dwell + top_up) * 1.0 - have` is a different number from `top_up` and returning
    # the second would put a step in the dial AT ITS OWN DEFAULT - a comma of 0.10 s at
    # 1.0 and 0.29 s at 1.2. The exactness argument only applies when the two are the same
    # number, so that is exactly when it is taken.
    if mult == 1.0 and (have is None or have == dwell):
        return max(0.0, top_up)
    return max(0.0, (dwell + top_up) * mult - (dwell if have is None else have))


def _top_up(mark: str, params: dict, base: float | None = None) -> float:
    """Our own share of the rest at Pause 1.0: the table, or the caller's sentence_gap."""
    return PAUSE_AFTER.get(str(mark), 0.0) if base is None else base


def _rest_for(mark: str, params: dict, base: float | None = None) -> float:
    """`_rest_from` with the dwell taken from DWELL rather than from the audio.

    For the two places that have no waveform to measure: the sentence seam, which the editor
    inserts between two separate renders, and the accounting. `_splice_pauses` measures the
    real thing - see the note there about why a table cannot do this job on its own.
    """
    m = str(mark)
    return _rest_from(
        _dwell_for(m, base), _top_up(m, params, base), _pause_multiplier(params)
    )


def _dwell_for(mark: str, base: float | None = None) -> float:
    """The mark's own natural rest at Pause 1.0 - what the target is built on.

    `base` is the caller's `sentence_gap` overriding the top-up, and a caller who names
    its own top-up for a mark the table does not carry still needs a dwell to go with it;
    a full stop's is the honest default, since the only caller that does this is the
    sentence seam.
    """
    m = str(mark)
    if m in DWELL:
        return DWELL[m]
    return DWELL["."] if base is not None else 0.0


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

    ONE MULTIPLIER ON THE WHOLE REST, for every mark - see DWELL and `_rest_for`. That is
    what keeps the punctuation hierarchy upright, and it has now been got wrong twice in
    opposite directions, which is worth recording because both were reported by ear and both
    were real.

    FIRST, as an inversion: this was uncapped while the sentence gap was pinned at an
    absolute 1.2 s, so past 1.0 the commas overtook the full stops - "a large breath was
    being taken after every single comma... NO breaths were being taken after punctuation".
    Patched by giving each mark its own share of that same absolute ceiling.

    SECOND, as a dead dial: shares of an absolute ceiling are still absolute, so every mark
    hit its cap at the same 3.75x and the top 62% of a 10x slider did nothing whatever - "the
    Pause option has no effect at all". Patched by scaling every mark linearly instead.

    THIRD, and this is the one the two before it were hiding: scaling OUR silence linearly
    does not scale the REST linearly, because the model is resting too and its share does not
    move. So the dial stretched the gaps between the marks rather than the marks - the full
    stop ran away from the comma, and no single setting could suit both. The multiplier now
    applies to the total, so the ratios at 10x are the ratios at 1x, and the curve saturates
    so the far end of the dial is long rather than broken.
    """
    if PAUSE_AFTER.get(str(mark), 0.0) <= 0.0:
        return 0.0
    return _rest_for(mark, params)


# The largest `dynamics` this file can be handed - generative_editor._open_up's value at
# the top of the slider. Every timing coefficient below is written as the figure it should
# deliver THERE, divided by this, so the numbers in the comments are the numbers a reader
# gets at full travel. Kept here rather than imported because the editor is Godot and this
# is Python; if that curve changes, this changes with it.
DEPTH_TOP = 2.5


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
        # end. 18% AT THE TOP OF THE DIAL: a real pre-boundary rime lengthens far more
        # than that, but this is applied to the WHOLE sentence, not just its last rime.
        #
        # THE CEILING HAD TO BE DIVIDED BY `_open_up`'s AND THAT IS THE BUG THIS FIXES.
        # These coefficients were written against a `depth` of 0..1 and read as their own
        # ceilings - the comment above has said "18%" since the day it was written.
        # generative_editor._open_up later opened the delivery dials to 2.5x at the top of
        # their travel, which multiplied every number here by two and a half without
        # touching this file: the paragraph term reached 45%, the section term 17%, and a
        # sentence at the end of both ran at rate 0.375, i.e. two and a half times its own
        # length. Reported as "the cadence/pace/speed of the voice becomes slower and
        # slower and slower... even at very small Arc values", and the Arc was innocent -
        # at the reported settings (arc 0.06, dynamics 0.62) the arc contributes 0.0% of
        # it and Dynamics contributes 15%, measured on the reporter's own voice and text.
        #
        # Scaled by DEPTH_TOP so the documented figures are what the top of the dial
        # actually delivers, which is what everything in this function was calibrated as.
        rate = (
            1.0
            - depth * (0.18 / DEPTH_TOP) * (u * u)
            - depth * (0.07 / DEPTH_TOP) * (v_in * v_in)
        )
        # Long sentences run a little faster, short ones a little slower.
        rel = lens[i] / typical if typical > 0 else 1.0
        rate *= 1.0 + depth * (0.10 / DEPTH_TOP) * (1.0 - min(max(rel, 0.5), 2.0)) * 0.5
        mark = ""
        for t in reversed(g):
            mark = str(t.get("punct", ""))
            if mark:
                break
        if mark == "?":
            rate *= 1.0 + depth * (
                0.04 / DEPTH_TOP
            )  # questions run slightly quicker...
            semis += arc * 0.20  # ...and end higher
        elif mark == "!":
            rate *= 1.0 + depth * (0.06 / DEPTH_TOP)
            semis += arc * 0.12
        # Rhythmic variety: the duration predictor's own noise, varied per sentence
        # instead of pinned for the take. Alternating rather than random so a
        # re-render is identical - an export must not be a different performance.
        nw = 1.0 + depth * (0.25 / DEPTH_TOP) * (1.0 if i % 2 == 0 else -1.0)
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


# --- THE SOURCE AND THE FILTER, SEPARATED -----------------------------------
#
# Two things this backend is asked for turn out to be one operation seen from two
# sides. Speech is a SOURCE - the glottal buzz, which carries pitch - driven
# through a FILTER, the vocal tract, whose resonances carry the words and the
# identity of whoever is speaking. A whisper is that filter with the buzz taken
# out and noise put in its place. A pitch move that keeps the speaker is the buzz
# moved with the filter held still. Both need the same thing first: an estimate of
# the filter, frame by frame, separated from whatever is driving it.
#
# LINEAR PREDICTION is that estimate, and it is the right one here for a reason
# beyond convenience: measure_voice.py already tracks this voice's formants with
# the same autocorrelation-plus-Levinson method, so the gate and the effect agree
# by construction about where a formant is. An LPC fit of order p is a smooth
# spectral envelope through the harmonics rather than a curve over them - which is
# exactly the split being asked for, because the harmonics ARE the source.
#
# NUMPY ONLY. voice_host/requirements.txt is deliberately tiny and scipy is not in
# it, which rules out lfilter and every IIR path. Nothing below needs one: both
# effects are magnitude edits on a windowed FFT, overlap-added back, which numpy
# does natively. The Levinson recursion is vectorised ACROSS frames - the loop
# runs `order` times over a few hundred frames at once rather than a few hundred
# times over one - so a sentence costs milliseconds rather than seconds.

_LPC_FRAME = 0.032  # s: long enough for two pitch periods of a low male voice
_LPC_HOP = 0.008  # s: 75% overlap, where a Hann window's square sums flat
# The noise a whisper is made of. FIXED, because a re-render must be the same
# performance - the same reason the duration-predictor jitter alternates by
# sentence index instead of sampling (see `_discourse_plan`).
_WHISPER_SEED = 0x5EED
# How much smoother than the data the whisper's envelope is asked to be. See
# `_lpc_envelopes`; 0.98 is where the fundamental leaves without the vowel going
# with it. The formant lock does NOT use this - it wants the sharpest envelope it
# can get, and it is putting one real envelope back rather than inventing one.
_WHISPER_GAMMA = 0.98


def _lpc_windows(sr: int) -> tuple:
    """(window length, hop, FFT size, Hann window) for this sample rate."""
    import numpy as np

    win = max(64, int(round(_LPC_FRAME * sr)) // 2 * 2)
    hop = max(16, int(round(_LPC_HOP * sr)))
    nfft = 1
    while nfft < win:
        nfft *= 2
    return win, hop, nfft, np.hanning(win).astype(np.float32)


def _framed(x, win: int, hop: int):
    """`x` cut into overlapping frames, one row each."""
    import numpy as np

    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx].astype(np.float64)


def _lpc_envelopes(frames, order: int, nfft: int, gamma: float = 1.0):
    """The spectral envelope of every frame at once.

    Levinson-Durbin, vectorised across frames: the recursion is over the ORDER,
    and every quantity in it is a scalar per frame, so each of the `order` steps
    is one vector operation over all of them. The autocorrelation comes from an
    FFT for the same reason - one batched transform instead of a loop.

    `gamma` under 1 is BANDWIDTH EXPANSION, and the whisper needs it. An envelope
    is supposed to pass over the harmonics rather than through them, but a fit of
    this order has enough poles to start tracking individual ones at a low pitch,
    and any harmonic ripple left in the envelope is printed straight back onto the
    noise - a whisper with the voice faintly still in it. Scaling the coefficients
    by gamma^k pulls every pole in off the unit circle, which widens all of them
    at once and is the standard speech-coding way to say "be smoother than the
    data". Measured on a synthetic 118 Hz vowel: harmonicity 0.24 at gamma 1,
    0.06 at 0.98, and at 210 Hz 0.28 against 0.09. It costs a little formant
    accuracy - F1 reads about 12% high at 0.98, F2 and F3 within 3% - and a
    raised F1 is what whispered vowels do anyway.
    """
    import numpy as np

    n = frames.shape[1]
    size = 1
    while size < 2 * n:
        size *= 2
    spec = np.fft.rfft(frames, size, axis=1)
    r = np.fft.irfft(spec * np.conj(spec), size, axis=1)[:, : order + 1]
    # A ridge on the zero lag. Silence and pure tones are both singular for a
    # plain fit, and a sentence has silence at both ends of it by construction.
    r[:, 0] = r[:, 0] * 1.0001 + 1e-9
    f = frames.shape[0]
    a = np.zeros((f, order + 1))
    a[:, 0] = 1.0
    err = r[:, 0].copy()
    for i in range(1, order + 1):
        acc = r[:, i].copy()
        if i > 1:
            acc += np.sum(a[:, 1:i] * r[:, i - 1 : 0 : -1], axis=1)
        k = -acc / np.maximum(err, 1e-12)
        if i > 1:
            prev = a[:, 1:i].copy()
            a[:, 1:i] = prev + k[:, None] * prev[:, ::-1]
        a[:, i] = k
        err = np.maximum(err * (1.0 - k * k), 1e-12)
    if gamma < 1.0:
        a = a * (gamma ** np.arange(order + 1))[None, :]
    # |H(w)| = sqrt(gain) / |A(w)|
    aw = np.fft.rfft(a, nfft, axis=1)
    return np.sqrt(err)[:, None] / np.maximum(np.abs(aw), 1e-9)


def _overlap_add(frames, hop: int, length: int, window, incoherent: bool = False):
    """Frames back to a signal, windowed again and normalised by what overlapped.

    The second window is not decoration: a frame whose magnitude spectrum has been
    edited is no longer confined to the frame it came from, and adding the raw
    result back seams audibly. Dividing by the accumulated square of the window
    makes the reconstruction exact where nothing was edited.

    `incoherent` is for frames that are NOISE. Overlapping copies of a signal add
    up; overlapping independent noise adds up in POWER, so the same division
    leaves it 2.7 dB quiet - measured, and heard as the whisper being too far
    away. The square root is the right normaliser there. (Making the noise
    continuous instead, so that it adds coherently, is the obvious alternative and
    is worse: the frames then agree with each other once per hop, which puts a
    125 Hz buzz right in the middle of the pitch range a whisper is supposed to
    have vacated. Measured at harmonicity 0.37 against 0.24.)
    """
    import numpy as np

    win = frames.shape[1]
    out = np.zeros(length + win, dtype=np.float64)
    norm = np.zeros_like(out)
    w2 = (window * window).astype(np.float64)
    for i in range(frames.shape[0]):
        s = i * hop
        out[s : s + win] += frames[i] * window
        norm[s : s + win] += w2
    den = np.maximum(norm[:length], 1e-6)
    return (out[:length] / (np.sqrt(den) if incoherent else den)).astype(np.float32)


def _rms_match(frames, target_frames):
    """Per-frame level, restored. Both effects rebuild a frame from scratch."""
    import numpy as np

    have = np.sqrt(np.mean(frames * frames, axis=1))
    want = np.sqrt(np.mean(target_frames * target_frames, axis=1))
    return frames * (want / np.maximum(have, 1e-9))[:, None]


def _whisper(a, sr: int, amount: float):
    """Keep the vocal tract, replace the voice in it with breath.

    A whisper is not a quiet voice and it is not a breathy one: the vocal folds
    are not vibrating at all, so there is no fundamental and no harmonics, and the
    words survive entirely in the resonances. That is why no inference parameter
    can produce one - a VITS checkpoint trained on modal speech has no whispered
    speech to sample - and why a filter can: rebuild each frame as NOISE shaped by
    that frame's own spectral envelope and the tract is untouched while the source
    is gone.

    `amount` blends against the original, which is a real setting rather than a
    fader: half way is a stage whisper, a voice that has not quite left.
    """
    import numpy as np

    amount = min(1.0, max(0.0, float(amount)))
    if amount <= 1e-3 or a.size < 4:
        return a
    win, hop, nfft, window = _lpc_windows(sr)
    frames = _framed(a, win, hop) * window
    env = _lpc_envelopes(frames, _lpc_order(sr), nfft, _WHISPER_GAMMA)
    # A FRESH DRAW PER FRAME, not one continuous noise - see `_overlap_add`.
    rng = np.random.default_rng(_WHISPER_SEED)
    noise = rng.standard_normal(frames.shape)
    spec = np.fft.rfft(noise * window, nfft, axis=1) * env
    built = np.fft.irfft(spec, nfft, axis=1)[:, :win]
    built = _rms_match(built, frames)
    wet = _overlap_add(built, hop, a.size, window, incoherent=True)
    # ...and the level trimmed once at the end. The per-frame match and the
    # incoherent normaliser between them still land about a decibel low, and a
    # whisper that arrives quieter than it was asked to be is indistinguishable
    # from the Presence dial having moved on its own.
    dry = a.astype(np.float64)
    have = float(np.sqrt(np.mean(wet.astype(np.float64) ** 2)))
    want = float(np.sqrt(np.mean(dry * dry)))
    if have > 1e-9:
        wet = wet * (want / have)
    # POWER-PRESERVING, because the two halves of this blend are UNCORRELATED. A whisper is
    # noise through the vocal tract and the voice under it is periodic, so they do not add
    # the way two takes of the same signal would - they add in POWER, and a straight
    # `a*wet + (1-a)*dry` therefore lands at sqrt(a^2 + (1-a)^2) of the level. That is -2.95 dB
    # at Hushed's 0.45 and nothing at all at either end, which is exactly how it was reported:
    # "with hushed specifically, it is too quiet", while Whispered at full strength was fine.
    # Measured against the prediction to 0.02 dB before it was touched.
    norm = math.sqrt(amount * amount + (1.0 - amount) ** 2)
    out = (amount * wet + (1.0 - amount) * dry) / max(norm, 1e-9)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _muffle(a, sr: int, amount: float):
    """A voice coming through something: the high end taken off, and a little level with it.

    Not the same thing as distance, which is what the panel's Presence dial does, and that
    is the whole reason this exists rather than reusing it. Distance is a property of the
    ROOM and belongs to whoever is directing the reading; a muffle is a property of the
    VOICE - Gruff speaks through a mask - and belongs to the tone that asked for it. Wiring
    a tone into the Presence dial made those two the same knob and cost the reader control
    of both: "there is no way to correct that."

    A one-pole low-pass, which is the right amount of crude. Something over the mouth is a
    gentle, wide-band roll-off rather than a filter with a corner you can hear, and the
    single pole is also the cheapest thing that cannot ring. The level trim goes with it
    because a covered voice really is quieter - not by much, or it stops being a reading.
    """
    import numpy as np

    amount = min(1.0, max(0.0, float(amount)))
    if amount <= 1e-3 or a.size < 2:
        return a
    # 9 kHz wide open down to about 1.4 kHz at full - a mask, not a telephone.
    cut = 9000.0 * math.pow(0.155, amount)
    x = math.exp(-2.0 * math.pi * cut / float(sr))
    # The pole written out as its impulse response rather than run as a recursion: a
    # per-sample Python loop over a sentence is 200k iterations for a filter whose kernel
    # is spent inside 40 taps (x^40 is 1e-7 at the widest setting here).
    taps = np.arange(40)
    kernel = (1.0 - x) * np.power(x, taps)
    out = np.convolve(a.astype(np.float64), kernel)[: a.size]
    # ...and put back what the filter took off the level, less the trim the muffle is worth.
    have = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    want = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) * (1.0 - 0.18 * amount)
    if have > 1e-9:
        out = out * (want / have)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _restore_formants(shifted, source, ratio: float, sr: int):
    """Put `source`'s formants back onto `shifted` after a resample.

    THE BUG THIS ANSWERS. A resample moves the whole spectrum, so buying a pitch
    move with one also moves the formants - and formants are how big a speaker is.
    The paragraph arc is bought exactly that way, and at any real depth it did not
    read as a reader settling across a paragraph, it read as a different, larger
    person finishing it: "the voice becomes lower and lower and lower... it
    completely transforms the voice into another voice by the end of the arc".

    The fix is the same split as the whisper, used the other way round. The
    resampled frame already has the pitch that was wanted. Divide its spectrum by
    its OWN envelope and multiply by the envelope of the frame it came from, and
    the harmonics stay where the resample put them while the resonances go back
    where the speaker had them. Frame i of the output stands at `i * hop * ratio`
    in the source, because that is what the resample did to the time axis.
    """
    import numpy as np

    if abs(ratio - 1.0) < 1e-4 or shifted.size < 4 or source.size < 4:
        return shifted
    win, hop, nfft, window = _lpc_windows(sr)
    out_frames = _framed(shifted, win, hop) * window
    n = out_frames.shape[0]
    # the same frames, located in the pre-resample audio
    src = source
    if src.size < win:
        src = np.pad(src, (0, win - src.size))
    starts = np.minimum(
        np.round(np.arange(n) * hop * ratio).astype(np.int64), src.size - win
    )
    starts = np.maximum(starts, 0)
    src_frames = src[starts[:, None] + np.arange(win)[None, :]].astype(np.float64)
    src_frames = src_frames * window
    order = _lpc_order(sr)
    nfft_env = nfft
    want = _lpc_envelopes(src_frames, order, nfft_env)
    have = _lpc_envelopes(out_frames, order, nfft_env)
    # Bounded, because a frame of near-silence has an envelope that means nothing
    # and dividing by it is how a click gets made.
    corr = np.clip(want / np.maximum(have, 1e-9), 0.05, 20.0)
    spec = np.fft.rfft(out_frames, nfft, axis=1) * corr
    built = np.fft.irfft(spec, nfft, axis=1)[:, :win]
    built = _rms_match(built, out_frames)
    out = _overlap_add(built, hop, shifted.size, window)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _lpc_order(sr: int) -> int:
    """One pole pair per kHz, plus two for the glottal tilt - the usual rule."""
    return int(min(36, 2 + sr // 1000))


def _nominal_seconds(frames, ratio: float, sr: int) -> float:
    """How long this sentence WOULD have run at `1/ratio` of the length scale it was given.

    THE BUG THIS ANSWERS. A per-sentence pitch move is bought by rendering `pr` times slower
    and playing back `pr` times faster, and the two are assumed to cancel in duration. They
    do not. The duration predictor's output is CEILED to whole frames per phoneme id - pads
    and BOS/EOS included, which is most of the ids - so a token already at its one-frame
    floor cannot get shorter, and a sizeable part of every sentence does not respond to
    length_scale at all. Measured across three voices, that fixed part is 13% to 46% of a
    sentence, varying with the voice and the sentence both, which is why no constant can
    stand in for it.

    What it sounds like: the paragraph arc opens a unit high (pr > 1, renders long, plays
    back short) and closes it low (pr < 1, renders short, plays back long), so the reading
    accelerates into every paragraph and drags out of it - by 5.1% and 2.0% at the top of
    the Arc dial, on a 3-second sentence. Reported as "the pace grows slower and slower over
    time, decreasing in speed with the increase in the Arc value", and the report is exact:
    the tilt is proportional to the arc.

    The frames are the plan the synthesizer actually used, so the answer is computable rather
    than remembered: `frames_i = ceil(d_i * L)` bounds each `d_i` to within one frame, its
    midpoint is the unbiased choice inside that bound, and rescaling that midpoint gives what
    the other scale would have asked for.

    THE EXPECTATION IS EXACT, not a half-frame rule of thumb. `frames_i = m` says only that
    `d_i * L` fell in `(m-1, m]`, so at the other scale it falls in `((m-1)/r, m/r]` - an
    interval whose width is one frame divided by the ratio. Averaging the ceiling over it is
    the integral of a staircase, which is closed form - and it has to be general in the width,
    because a PULL-DOWN (ratio under 1) widens the interval past a whole frame and a two-term
    version that assumed one straddle read 6% long on exactly those, which is the end of every
    paragraph. Two cruder versions were
    measured first and both left a systematic tilt behind: ceiling the midpoint counts the
    rounding twice (+3.1% of a sentence), and adding half a frame back counts it slightly
    short (-1.1% on one voice, -4.4% on another, because how much a voice floors at one
    frame varies).

    Exact at ratio 1 by construction, which is the property worth having: with no arc there
    is no correction and the render is what it always was, byte for byte.
    """
    import numpy as np

    if frames is None or ratio <= 0.0:
        return 0.0

    def area(x):
        """The integral of ceil() from 0 to x - a staircase, so it is closed form."""
        k = np.floor(x)
        return k * (k + 1.0) * 0.5 + (k + 1.0) * (x - k)

    m = np.asarray(frames, dtype=np.float64)
    lo = np.maximum(m - 1.0, 0.0) / ratio
    hi = m / ratio
    width = np.maximum(hi - lo, 1e-12)
    exp = (area(hi) - area(lo)) / width
    # AN ID AT THE FLOOR STAYS AT THE FLOOR, and this is the whole fixed fraction. A one-frame
    # id is one frame because its predicted duration was UNDER a frame, and most of them are
    # the pads between phonemes, whose duration is near zero rather than anywhere near uniform
    # in the frame they were rounded up into. Averaging the ceiling over that frame credits
    # them with half of one they never had, and then a pull-down (which widens the interval
    # past a whole frame) turns that into a whole extra frame each: measured +6% of a
    # sentence on all three voices at the bottom of the range, which is every paragraph end.
    exp = np.where(m <= 1.0, 1.0, exp)
    return float(np.maximum(exp, 1.0).sum()) * HOP_LENGTH / float(sr)


# A first guess at that fixed fraction, used to ASK for a length scale that will land near
# the pitch move requested. It only has to be in the right neighbourhood: whatever it gets
# wrong, `_nominal_seconds` measures afterwards and the resample corrects, so the timing is
# exact either way and only the delivered semitones drift a little with it.
PACE_FIXED_GUESS = 0.30


def _pitch_length(pr: float) -> float:
    """The length-scale factor to REQUEST for a pitch move of `pr`.

    Solving `A*L + b = pr * (A*ls + b)` for L, with `b/(A*ls) = f/(1-f)`: asking for exactly
    `pr` lands short of it, because the fixed part comes back unscaled and then gets divided
    by `pr` along with everything else.
    """
    f = PACE_FIXED_GUESS / (1.0 - PACE_FIXED_GUESS)
    return pr + f * (pr - 1.0)


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
    # The same rule as every other mark, through the same function: the dial scales the whole
    # rest and the model's own share is subtracted back off. A caller's `sentence_gap` still
    # wins over the table - it is the same knob - but it is now a TOP-UP at 1.0 rather than
    # the answer, exactly like the table entry it replaces.
    return _rest_for(str(mark), params, base)


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


def _splice_pauses(audio, points, sr: int, mult: float = 1.0):
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

    # A point is (nominal time, top-up, next token's start, the mark's own dwell). The
    # third field bounds how far the cut may slide forward; the fourth is the mark's
    # natural rest from DWELL, which is what the TARGET is built on - see `_rest_from`.
    # Shorter points are the older form and the tests: no bound, and no table entry, in
    # which case the measured dwell stands in for it and the rule is what it was before
    # the target and the measurement were separated.
    pts = [
        (
            float(p[0]),
            float(p[1]),
            (float(p[2]) if len(p) > 2 else 0.0),
            (float(p[3]) if len(p) > 3 else None),
        )
        for p in points
        if float(p[1]) > 0.0
    ]
    # ONE RULE AT EVERY SETTING OF THE DIAL, INCLUDING ITS DEFAULT. `mult` = 1.0 used to
    # short-circuit to "insert exactly what you were asked for", which is what every caller
    # did before the dial learned to scale a whole rest. That left a step at the dial's own
    # default - a comma of 0.10 s at 1.0 and 0.29 s at 1.2 - so it is gone, and the rule
    # below runs at 1.0 like everywhere else.
    #
    # WHY THE SILENCE IS MEASURED HERE. Because it is the only place that can see it, and
    # because the table's guess at it is wrong: it said this voice dwells 0.15 s at a comma
    # and the measured figures run from 0.000 to 0.401. What is measured is used for what
    # it can answer - HOW MUCH OF THE TARGET IS ALREADY PAID FOR - and nothing else. It
    # does not set the target: multiplying a per-instance draw by the dial is what made one
    # reading's commas come out 0.36, 0.44, 1.52 and 1.81 seconds. See `_rest_from`.
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
    quiet = float(np.max(np.abs(audio))) * 0.02 if audio.size else 0.0
    for t, secs, limit, dwell in pts:
        hi = int(round(limit * sr)) if limit > 0.0 else 0
        cut = max(
            prev,
            _quiet_point(audio, int(round(t * sr)), search, prev, hi, env, reach, back),
        )
        have = _silence_around(audio, cut, quiet, sr)
        pad = int(
            round(
                _rest_from(dwell if dwell is not None else have, secs, mult, have) * sr
            )
        )
        if pad <= 0:
            continue
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


# THE CEILING ON A REST NOBODY ASKED FOR, as a multiple of the median rest between two
# words in the SAME sentence, floored so a brisk sentence still has room for a real one.
#
# Relative rather than absolute because every one of pace, length_scale, the tone preset,
# the Arc resample and the checkpoint itself moves the whole distribution together; a
# figure in seconds would fire constantly at a slow pace and never at a quick one. The
# sentence's own median is the only reference that tracks all of them at once.
#
# 3.0 IS WHERE THE HISTOGRAM IS EMPTY, and that is the whole argument for it. Every
# unmarked boundary rest over its own sentence's median, 3036 of them across 16 sentences
# of the reporter's chapter, four speakers and three paces:
#
#     x1.50-1.75  205 |  x2.25-2.50  22  |  x3.00-3.25   0  |  x4.00-5.00   0
#     x1.75-2.00   76 |  x2.50-2.75  13  |  x3.25-3.50   2  |  x5.00-5.25   1
#     x2.00-2.25   27 |  x2.75-3.00   5  |  x3.50-4.00   2  |  x5.25-5.50   1
#
# One continuous mass thinning out to nothing at 3.0, an empty bin, and then a handful
# scattered out to 5.4x with nothing between. That is two distributions, not one tail, and
# 0.16% of boundaries are in the second one. The cut brings those back to 3.0 rather than
# to the median: the point is to make the rest indistinguishable from one the model meant,
# not to flatten it.
#
# The floor is a safety net for a short sentence, where a median over four or five
# boundaries is not a reliable estimate of anything. It binds at a brisk pace and nowhere
# else - at the reporter's settings the median rest is 0.046 s, so the ratio decides.
REST_RATIO = 3.0
REST_FLOOR = 0.12


def _trim_rests(audio, rests, sr: int):
    """Cut the unplanned pause out of the middle of a clause.

    THE DEFECT. Piper's duration predictor is stochastic - that is what `noise_w` is the
    noise of - and the thing it predicts a duration FOR includes the blanks between the
    phonemes. A blank is also where the model puts a pause, so a blank's duration is not
    scattered around one value, it is bimodal: a few tens of milliseconds nearly always,
    and a quarter of a second when the sample lands in the other mode. Draw enough of them
    and every so often one lands there in the middle of a clause, with no punctuation
    anywhere near it and nothing in the text asking for it.

    Reported as "weird stuttering in their cadence, around specific words" - the
    reporter's own opening sentence has one, a 0.29 s hole between `house` and `in it` at
    -34 dB, in three renders out of eight. Reading in the sentence rather than word by
    word (see WORD_BREAK) shortened the tail a long way - the 99th percentile of a
    boundary rest went 0.120 -> 0.104 - but it cannot remove the mode, and on the
    reporter's own speaker it did not: 0.383 s after `country`, 0.302 s after `it`.

    It is not fixable at the source. The duration plan and the waveform come out of one
    ONNX call, so there is nothing to clamp before the audio exists, and the only dial
    that reaches it is `noise_w` - which is the rhythmic variety of the whole reading, and
    turning it down to stop one hole in six hundred boundaries buys a metronome.

    THE PLAN SAYS WHERE, THE WAVEFORM SAYS HOW LONG. `rests` is [(t0, t1)] from the frame
    plan, for word boundaries carrying NO punctuation - a marked one is `_splice_pauses`'s
    to own, and shortening a full stop because the model was generous with it would be
    this function arguing with the Pause dial. But the plan is not the hole: a word tapers
    into a rest and starts up out of it, and a phone beside the blank can render as
    silence itself, so the audible gap runs past the plan's idea of it at both ends.
    Measured - bringing the PLAN back to the ceiling left one boundary with 0.264 s of
    silence still in it. So the plan is used only to find the boundary, the real silence
    around it is measured, and the ceiling applies to that. Which is the lesson
    `_splice_pauses` had to learn one dial over, in the other direction.

    The excess comes out of the MIDDLE of the silence, so the ramp that keeps the join
    from clicking has silence to land on at both ends rather than the last few
    milliseconds of a word. Returns `(audio, removed)` with `removed` in the form `_shift`
    takes, the seconds negative. With nothing to cut the input array is returned
    untouched - byte-identical, not merely equal.
    """
    import numpy as np

    spans = [(float(a), float(b)) for a, b in rests if float(b) > float(a)]
    if not spans or audio.size == 0:
        return audio, []
    lens = sorted(b - a for a, b in spans)
    cap = max(REST_FLOOR, REST_RATIO * lens[len(lens) // 2])
    fade = max(1, int(round(SPLICE_FADE_MS * sr / 1000.0)))
    quiet = float(np.max(np.abs(audio))) * 0.02
    pieces: list = []
    removed: list = []
    prev = 0
    need_in = False
    for a, b in spans:
        mid = int(round(0.5 * (a + b) * sr))
        lo, hi = _silent_span(audio, mid, quiet, sr)
        lo = max(lo, prev)
        n = int(round(((hi - lo) / float(sr) - cap) * sr))
        if n <= 0 or hi - lo <= n + 2 * fade:
            continue
        at = lo + (hi - lo - n) // 2
        seg = np.array(audio[prev:at], dtype=audio.dtype, copy=True)
        _ramp(seg, fade, need_in, True)
        pieces.append(seg)
        removed.append((at / float(sr), -n / float(sr)))
        prev, need_in = at + n, True
    if not removed:
        return audio, []
    tail = np.array(audio[prev:], dtype=audio.dtype, copy=True)
    _ramp(tail, fade, need_in, False)
    pieces.append(tail)
    return np.concatenate(pieces), removed


def _silent_span(audio, at: int, thresh: float, sr: int) -> tuple:
    """(first, last+1) of the contiguous near-silence containing sample `at`.

    Walked outward from the point rather than measured between the token spans, because a
    span is the aligner's opinion about where a word ends and this is a question about the
    waveform. Bounded at half a second in each direction: past that it is not a dwell, it is
    the end of the utterance, and topping THAT up would put the sentence's own tail inside
    the multiplier. `(at, at)` - an empty span - if `at` is not in silence at all.
    """
    import numpy as np

    if audio.size == 0 or thresh <= 0.0:
        return at, at
    span = int(0.5 * sr)
    lo = max(0, at - span)
    hi = min(audio.size, at + span)
    a = np.abs(audio[lo:hi])
    if a.size == 0:
        return at, at
    i = min(max(at - lo, 0), a.size - 1)
    if a[i] > thresh:
        return at, at
    loud = np.where(a > thresh)[0]
    left = loud[loud < i]
    right = loud[loud > i]
    start = int(left[-1]) + 1 if left.size else 0
    end = int(right[0]) if right.size else a.size
    return lo + start, lo + end


def _silence_around(audio, at: int, thresh: float, sr: int) -> float:
    """Seconds of contiguous near-silence containing `at` - what the model rests here."""
    lo, hi = _silent_span(audio, at, thresh, sr)
    return float(hi - lo) / sr


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

# THE WORD BREAK THAT COSTS NOTHING - U+200B ZERO WIDTH SPACE, written between the words
# of a sentence before it is handed to eSpeak.
#
# eSpeak reads a SENTENCE, and reading a sentence is how it decides which words are
# stressed. Handed one word at a time it can only give the citation form, and the
# citation form of a function word carries a primary stress no reader would ever put
# there. Measured over this chapter's 4148 words, with `_espeak_word` spellings:
#
#     a      ˈeɪ  alone      ɐ    in the sentence     46 times
#     I      ˈaɪ             aɪ                       37
#     in     ˈɪn             ɪn                       29
#     to     tuː             tə                       24
#     that   ðˈæt            ðæt                      22
#     it     ɪt              ɪɾ                       12
#
# `a` is the one to look at twice: ˈeɪ is not a stressed schwa, it is the LETTER A, and
# ghost was saying it that way 46 times in one chapter. The rest are a rhythm defect
# rather than a wrong word - a reading in which every preposition, article and modal is
# stressed is by definition an even one, and an even one is what a listener calls jerky.
#
# Reported as "weird stuttering in their cadence, around specific words... `in` and
# `in it`", with the delivery generally "jerky/uneven".
#
# So why not simply phonemize the sentence? Because eSpeak WELDS across word boundaries
# when it does - "not a" comes back nˌɑːɾə, "in the" as ɪnðə, "out of" as ˌaʊɾəv - and
# ghost cannot use a reading it cannot cut into words: the karaoke line, the per-token
# timings and every pause placement are keyed to knowing which phones belong to which
# word. Measured on the same chapter, a plain join loses that on 103 of 230 sentences.
#
# A zero-width space between the words is the whole fix. It is a word boundary to
# eSpeak, so nothing welds across it; it has no phones of its own, so nothing leaks into
# the transcription; and it does not stop eSpeak analysing the sentence, so the stresses
# stay the sentence's own. Same chapter: 6 sentences of 230 still come back with a
# different number of pieces than words (2.6%, and those fall back to the old path), and
# of the 127 sentences a plain join DID align, the zero-width version agrees with it on
# 124. A pipe and a double bar were measured too; the pipe behaves identically and the
# double bar disturbs the stresses, so the invisible one wins on nothing but taste.
WORD_BREAK = "\u200b"

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
        # The frame plan of the most recent render - see `_render_symbols`.
        self._last_frames = None
        self._last_rests: list = []

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

        Each item of the list is its own utterance, so what an item contains is
        what eSpeak gets to read. `_in_sentence` hands it a whole sentence with
        zero-width breaks between the words, which is the normal path; the
        word-by-word batch below it is the fallback for a sentence eSpeak runs
        together anyway, and `homographs.py` hands it carrier phrases.

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
        # sentence. One word per item is therefore the ISOLATED reading by
        # construction, and that reading costs both the homograph and the stress:
        # eSpeak reads "live" as laɪv alone and lɪv in "where they live", and it
        # stresses every function word it is shown on its own. The separator is
        # what makes a MULTI-word item splittable again, which is how both
        # `_in_sentence` and homographs.py get a reading back per word after
        # asking the question with the syntax attached.
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

    def _in_sentence(
        self, tokens: list, need: list, espeak_voice: str
    ) -> dict[int, str]:
        """eSpeak's reading of each word AS IT SITS IN THIS SENTENCE.

        Returns {token index: IPA} for every index in `need`, or {} if the reading
        could not be cut back into words - in which case the caller falls back to the
        word-by-word batch, which is what this whole file did before. See WORD_BREAK
        for what the sentence buys and why the words are separated the way they are.

        EVERY token contributes its spelling to the sentence, including the ones that
        are not in `need`: a word carrying an authored override or a homograph reading
        still conditions the stress of its neighbours, and dropping it from the string
        would be asking eSpeak about a sentence nobody wrote. Only the pieces for `need`
        are read back out.

        A token may be more than one eSpeak word - `_espeak_word` turns an internal
        hyphen into a boundary, which is the whole point of that function - so the
        pieces are taken by COUNT rather than one apiece, and rejoined with the space
        that keeps the boundary the hyphen asked for.
        """
        spoken: list[str] = []
        counts: dict[int, tuple[int, int]] = {}
        for i, t in enumerate(tokens):
            text = str(t.get("text", ""))
            if not text.strip():
                continue
            w = _espeak_word(text)
            parts = w.split()
            if not parts:
                continue
            counts[i] = (len(spoken), len(parts))
            spoken.extend(parts)
        if not spoken:
            return {}
        joined = (" %s " % WORD_BREAK).join(spoken)
        try:
            got = (self._espeak([joined], espeak_voice) or [""])[0].split()
        except BackendError:
            raise
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a phonemizer fault must not lose audio
            print(
                "ghost/voice: sentence-level phonemization failed (%s); "
                "falling back to word by word" % exc,
                file=sys.stderr,
            )
            return {}
        if len(got) != len(spoken):
            # eSpeak still ran the words together somewhere, so there is no honest way
            # to say which phones belong to which word. The isolated batch is wrong
            # about stress but right about alignment, and alignment is the one this
            # file cannot trade away - a mis-cut reading puts the karaoke line and
            # every pause on the wrong word for the rest of the sentence.
            self._warn_welded()
            return {}
        out: dict[int, str] = {}
        for i in need:
            at = counts.get(i)
            if at is None:
                return {}
            start, n = at
            out[i] = " ".join(got[start : start + n])
        return out

    def _warn_welded(self) -> None:
        """Once per process: the sentence reading could not be cut into words."""
        if "\u200b" in self._warned_symbols:
            return
        self._warned_symbols.add("\u200b")
        print(
            "ghost/voice: eSpeak ran words together in a sentence despite the word "
            "break; those sentences are read word by word instead (about 3% of them)",
            file=sys.stderr,
        )

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
            espoke = self._in_sentence(tokens, need, espeak_voice)
        if need and phonemizer == "espeak" and not espoke:
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
                # the top-up and the table dwell, like the token path - the splicer
                # measures what is already there and tops it up to the target
                points.append(
                    (
                        float(phone_times[i]["t1"]),
                        _top_up(mark, params),
                        float(nxt),
                        _dwell_for(mark),
                    )
                )
            else:
                unplaceable.append(mark)
        _warn_unaligned(unplaceable)
        if points:
            audio, inserted = _splice_pauses(
                audio, points, sample_rate, _pause_multiplier(params)
            )
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
            base_ls = float(
                params.get(
                    "length_scale", cfg.get("inference", {}).get("length_scale", 1.0)
                )
            )
            # ...and ASK for more than `pr`, because part of a sentence does not scale and
            # the playback divides all of it. See `_pitch_length` / `_nominal_seconds`.
            gp["length_scale"] = base_ls * _pitch_length(pr) / float(step["rate"])
            gp["noise_w"] = float(
                params.get("noise_w", cfg.get("inference", {}).get("noise_w", 0.333))
            ) * float(step["noise_w_mul"])
            audio, per_token = self._run(group, voice, gp, cfg, sess, phonemizer)
            # The frame plan's word-boundary rests, before anything resamples them.
            group_rests = list(self._last_rests)
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
            # THE SOURCE, if this reading wants a different one. Before the pitch
            # move rather than after, because a whisper has no pitch to move and
            # doing it the other way round would spend the work twice.
            audio = _whisper(audio, sr, float(params.get("whisper", 0.0)))
            audio = _muffle(audio, sr, float(params.get("muffle", 0.0)))
            if abs(pr - 1.0) > 1e-4:
                # THE RATIO IS MEASURED, NOT ASSUMED. Playing back by exactly `pr` is what
                # tilted the pace across every paragraph; playing back by the ratio between
                # what this sentence actually rendered to and what it would have rendered to
                # unshifted lands the timing exactly, whatever the voice did with the length
                # scale. The pitch then moves by that same measured ratio rather than by the
                # nominal one, which is a few percent either way of what was asked for and
                # is the half of this trade nobody can hear.
                nominal = _nominal_seconds(self._last_frames, _pitch_length(pr), sr)
                played = float(audio.size) / float(sr)
                ratio = pr
                if nominal > 0.05 and played > 0.05:
                    ratio = played / nominal
                    # A guard, not a tuning: a mis-shaped frame plan must not be able to
                    # halve or double a sentence. Outside this the nominal figure is not
                    # believable and the requested move is the better answer.
                    if ratio < pr * 0.6 or ratio > pr * 1.6:
                        ratio = pr
                pr = ratio
                shifted = _resample(audio, pr)
                # ...AND THE FILTER HELD STILL WHILE THE SOURCE MOVES. Default on:
                # the arc is a change of REGISTER, and a register change that also
                # changes how big the speaker is was the bug. `formant_lock` = 0
                # restores the old behaviour, which is what the gate A/Bs against.
                if float(params.get("formant_lock", 1.0)) > 0.5:
                    shifted = _restore_formants(shifted, audio, pr, sr)
                audio = shifted
                # the alignment was measured before the resample, so it moves with it
                per_token = {k: (v[0] / pr, v[1] / pr) for k, v in per_token.items()}
                group_rests = [(a / pr, b / pr, w) for a, b, w in group_rests]
            # A REST NOBODY ASKED FOR, shortened before the ones that were asked for are
            # put in. Unmarked boundaries only: a mark's rest belongs to `_splice_pauses`
            # and to the Pause dial, and this must never be caught arguing with either.
            audio, cut = _trim_rests(
                audio,
                [
                    (a, b)
                    for a, b, w in group_rests
                    if w < len(group) and not str(group[w].get("punct", ""))
                ],
                sr,
            )
            if cut:
                per_token = {
                    k: (
                        max(0.0, _shift(v[0], cut, True)),
                        max(0.0, _shift(v[1], cut, False)),
                    )
                    for k, v in per_token.items()
                }
                per_token = {k: (a, max(a, b)) for k, (a, b) in per_token.items()}
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
                            # the TOP-UP, not the finished figure: what to add here also
                            # depends on what the model is already resting, and only the
                            # splicer can see that (it has the audio). See _splice_pauses.
                            _top_up(mark, params),
                            float(nxt[0]) if nxt is not None else 0.0,
                            # ...and the mark's own natural rest, which is what the
                            # target is built on. The measured dwell says how much of
                            # that target is already paid for; it does not get to say
                            # how long the rest should be. See `_rest_from`.
                            _dwell_for(mark),
                        )
                    )
                else:
                    unplaceable.append(mark)
            _warn_unaligned(unplaceable)
            audio, inserted = _splice_pauses(
                audio, points, sr, _pause_multiplier(params)
            )
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
        # Which ids are the REST between two words - the blanks and the word-space
        # itself. `_trim_rests` needs to know where a boundary rest starts and stops,
        # and the id stream is the only place that is written down: a token's span
        # runs to the end of its own trailing space, so the rest is split between two
        # spans and cannot be recovered from them afterwards. -1 = not a rest, else the
        # index of the token whose word-space this run carries.
        rest_of: list[int] = [-2]
        for sym, src in symbols:
            mapped = pmap.get(sym)
            if mapped is None:
                missing.append(sym)
                continue
            ids.append(PAD)
            owner.append(src)
            rest_of.append(-1)
            for m in mapped:
                ids.append(int(m))
                owner.append(src)
                rest_of.append(src if sym == " " else -2)
        ids.append(PAD)
        owner.append(-1)
        rest_of.append(-1)
        ids.append(EOS)
        owner.append(-1)
        rest_of.append(-2)
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
        # THE DURATION PLAN, kept: `_pace_keep` needs it to undo what a per-sentence
        # length_scale does to the timing. Stashed rather than returned because the vowel
        # probe renders through this same function and unpacks a pair.
        self._last_frames = None
        # [(t0, t1, token index)] for every run of blanks carrying a word-space, in the
        # same seconds `spans` is in. Stashed rather than returned for the reason
        # `_last_frames` is: the vowel probe renders through here and unpacks a pair.
        self._last_rests: list = []
        if len(out) > 1:
            frames = np.asarray(out[1]).squeeze().astype(np.float64)
            if frames.ndim == 1 and frames.size == len(owner):
                self._last_frames = frames
                t = 0.0
                run_at = 0.0
                run_of = -1
                spoke = False
                for dur, who, rest in zip(
                    frames * HOP_LENGTH / float(cfg["audio"]["sample_rate"]),
                    owner,
                    rest_of,
                ):
                    if rest == -2:
                        # A rest is only a rest when speech stands on BOTH sides of it.
                        # The lead-in (see LEAD_IN_SPACES) and the utterance's own tail
                        # are neither, and shortening either would be undoing something
                        # this file asked for on purpose.
                        if run_of >= 0 and spoke and who >= 0:
                            self._last_rests.append((run_at, t, run_of))
                        spoke = spoke or who >= 0
                        run_of = -1
                        run_at = t + dur
                    elif rest >= 0:
                        run_of = rest
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
