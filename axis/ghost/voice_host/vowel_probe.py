#!/usr/bin/env python3
"""Whether a checkpoint actually says the vowel its phoneme string names - measured, not assumed.

THE BUG THIS EXISTS FOR
-----------------------
"the house of healing that can read them" was read with the vowel of `rid`. Every
stage upstream was correct: the tagger called it VB, homographs.py decided `bare
form after MD 'can'` and left the word alone, and eSpeak's own reading - the one
handed to the model - was `ɹˈiːd`. The phonemes said "reed" and the audio said
"rid", so the defect is in the acoustic model and nowhere else.

Measured on en_US-libritts-high, speaker 32, one frame, one word swapped, five
renders of each. F1 is the discriminator: this reader's /iː/ sits at F1 343 and its
/ɪ/ at F1 444.

    that can feed them    fˈiːd     F1 283      that can read them   ɹˈiːd  F1 463
    that can seed them    sˈiːd     F1 298      that can reed them   ɹˈiːd  F1 447
    that can beat them    bˈiːt     F1 284      that can breed them  bɹˈiːd F1 333
    that can reach them   ɹˈiːtʃ    F1 322      that can freed them  fɹˈiːd F1 330

`reed` fails exactly as `read` does, and the model never sees spelling, so this is
the SYMBOL STRING and not the word. Put any consonant in front of the ɹ and it
recovers; change the coda and it recovers. The one broken string is a word-initial
`ɹˈiːd`.

That is the signature of a contaminated training set, and the contamination is this
same homograph bug one level down: eSpeak phonemized the LibriTTS transcripts the
way ghost does, so every past-tense "read" in that corpus was labelled `ɹˈiːd` over
audio of a reader saying `ɹˈɛd`. The checkpoint learned the string is ambiguous and
now hedges to a mid vowel. It is not confined to one word either - `healing` laxes
the same way in connected speech in that voice (F1 413 against a 343 target).

THE REPAIR IS PHONETICALLY NULL
-------------------------------
English /iː/ in a stressed syllable before a consonant already has an offglide;
writing it is a spelling choice, not a different vowel. Written, the string leaves
the region the training data poisoned:

    ɹˈiːd -> ɹˈiːjd    speaker 32  F1 478 -> 298      speaker 29  F1 511 -> 377

and on the words that were never broken it is a no-op within measurement noise -
feed, seed, beat, keep, reach, meet and breed all move less than the scatter of the
model's own stochastic duration predictor.

WHY IT IS ASKED AND NOT LISTED
------------------------------
Because it is NOT true of every checkpoint, and a table would say it was. Measured
on the other four installed voices, `read` is already correct, and on
en_US-kristin-medium the offglide actively HURTS (F2 2730 -> 2070). A voice added
next year is a coin toss. So the same policy homographs.py has: the question is put
to the model that has to answer it, in a form it can answer, and the answer is
cached rather than written down. Three probe words in one frame, an /iː/ the model
gets right, an /ɪ/ for the other end of the scale, and the string under suspicion
between them. If the suspect lands nearer the /ɪ/, the checkpoint needs the repair;
if writing the offglide does not actually move it back, the repair is declined.

The probe words are a PROBE and not a lexicon - the same thing homographs.py's four
carrier frames are. Nothing here is keyed to a word: the repair, once adopted,
applies to every stressed /iː/ before a consonant, which is why it has to be shown
harmless on words that were never broken before it is allowed to run at all.

Costs three short inferences for a healthy voice and four for one that needs the
repair, once per
voice per install, cached to disk beside the weights and taken on the cold path of
`_load` so it never happens inside a reading. If numpy is missing, the ONNX has no
duration output to locate a vowel with, or anything else goes wrong, the answer is
"no repair" and the reading is exactly what it was before: this can decline, but it
cannot fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# A stressed /iː/ with a consonant after it. The stress mark has to be in the SAME
# word (no space and no second mark between), the lookahead has to find something
# that is not a vowel, not a length mark and not the offglide already - and being a
# lookahead, it also fails at the end of a word, which is what keeps `fɹˈiː` alone.
OFFGLIDE = re.compile(r"(ˈ[^ˈˌ ]*?iː)(?=[^aeiouɐəɚɛɪʊʌɔɑːjˈˌ ])")

# ONE FRAME AND THREE WORDS.
#
# THE FRAME IS LONG ON PURPOSE, and a short one is why this file took three passes to
# settle. The defect only appears in CONNECTED SPEECH - a word said alone comes out
# right on every voice measured - so a two-word carrier is a weak instrument for it.
# Measured at speaker 0 of the broken checkpoint, 24 renders per cell, F1 medians:
#
#     "that can {} them"                     tense 409  lax 523  suspect 609  fixed 522
#     "the house of healing that can {} them" tense 397  lax 535  suspect 625  fixed 432
#
# In the short frame the repaired string lands ON the lax reference and the second gate
# below is deciding a 87 Hz difference between two measurements that scatter by 40 - a
# coin toss about one cold start in ten, which is exactly what it did. In the long one
# the repair carries the vowel back to the tense reference and the same gate is reading
# 193 Hz. Same code, same weights; the frame was the whole difference. It is the shape
# the defect was reported in, which is not a coincidence: a probe should ask the question
# in the position the answer matters.
#
# The words are onset-matched: all three are ɹ + vowel + stop, so the span the formants
# are taken from lines up across the renders without any alignment work. `reap` is an
# /iː/ every checkpoint measured gets right, `rip` is the other end of the scale, and
# `reed` is the string under suspicion - which is `read`'s string, in a spelling that
# cannot be confused for a homograph question.
FRAME = "the house of healing that can {} them"
SUSPECT = "reed"
TENSE_REF = "reap"
LAX_REF = "rip"

# HOW THE PROBE RENDERS, and this is the part that took three passes to get right.
#
# A VITS voice is stochastic by design: the duration predictor draws from noise and so
# does the flow. That is what keeps a reading from sounding mechanical, and it is
# exactly wrong for a measurement. Probing at speaking settings, the F1 of a 50 ms vowel
# scattered by 40-50 Hz per render with a heavy tail - LPC occasionally picks the wrong
# pole - against margins of the same order, so the verdict flipped about one cold start
# in ten. Averaging did not fix it, a median of three did not fix it, and neither would
# have: the instrument was noisy, and the answer to a noisy instrument is to turn the
# noise off rather than to sample it harder.
#
# So the probe renders with both noise terms at zero and the clock slowed down. The
# vowel gets three times as long for LPC to fit, and the whole measurement becomes
# EXACTLY repeatable - five consecutive probes of five voices returned identical ratios
# to four decimal places. One render per word is then not a sample, it is the answer,
# which is also why this costs less than the version that could not make up its mind.
#
# Nothing here reaches a reading. These settings exist inside the probe and nowhere else.
PROBE_PARAMS = {"noise_scale": 0.0, "noise_w": 0.0, "length_scale": 3.0}

# How far from the /iː/ reference the suspect has to land before the checkpoint is
# called broken, as a multiple of that reader's OWN /iː/-to-/ɪ/ distance. Measured on
# the five installed voices, deterministically:
#
#     en_US-libritts-high   1.46      <- past its own /ɪ/, and the only one that is broken
#     en_US-john-medium     0.51
#     en_US-ljspeech-medium 0.23
#     en_US-kristin-medium  0.10
#     en_US-norman-medium   0.02
#
# 1.0 sits between the two nearest neighbours with about a threefold margin on each
# side, and it is a meaningful place to put a line rather than a gap in the data: at 1.0
# the model is rendering the string further from the vowel it names than its own lax
# vowel is. Below that is a reader with an unremarkable spread; john is the one to watch
# if this ever needs revisiting, and it is not close.
BROKEN_AT = 1.0

# How much WORSE the repair may make the vowel before it is refused, on the same scale.
#
# THE TEST IS NON-REGRESSION AND NOT IMPROVEMENT, and the asymmetry is deliberate: a
# repair that does nothing IS nothing, so the only outcome worth refusing is one that
# drags the vowel further from where it belongs. Whether the rewrite helps is already
# settled above. This gate exists for a checkpoint that does not yet exist - one broken
# in a way the offglide reaches the wrong way - and it should not be doing any other job.
MAX_LOSS = 0.10

# Where in the vowel's own span to measure. Skipping the first third steps past the
# transition out of the ɹ, which is formant-heavy and would drag F1 in the same
# direction for every word.
NUCLEUS = (0.30, 0.90)


def repair(ipa: str) -> str:
    """Write the offglide English already has. A no-op unless the string has a target."""
    return OFFGLIDE.sub(r"\1j", ipa)


def _formants(seg, sr: int) -> list:
    """F1..Fn of one vowel span, by LPC. Empty when the span is too short to fit."""
    import numpy as np

    if len(seg) < 256:
        return []
    order = int(2 + sr / 1000)
    x = np.asarray(seg, dtype=np.float64) * np.hamming(len(seg))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pre-emphasis
    r = np.correlate(x, x, "full")[len(x) - 1 : len(x) + order]
    if len(r) <= order or r[0] <= 0:
        return []
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        k = -(a[:i] @ r[i:0:-1]) / e
        a[1 : i + 1] = a[1 : i + 1] + k * a[i - 1 :: -1][:i]
        e *= 1 - k * k
        if e <= 0:
            return []
    out = []
    for z in np.roots(a):
        if np.imag(z) <= 0:
            continue
        f = np.arctan2(np.imag(z), np.real(z)) * sr / (2 * np.pi)
        bw = -0.5 * (sr / (2 * np.pi)) * np.log(abs(z))
        # A formant is a narrow peak. Wide poles are the glottal source and the
        # spectral tilt, and they sit wherever they like.
        if 90 < f < sr / 2 - 200 and bw < 400:
            out.append(f)
    return sorted(out)


class VowelProbe:
    """One per backend instance; the verdict is per voice and survives the process.

    `speak(words, lang)` is the backend's own word-level phonemizer and `render(symbols,
    params)` its symbols-to-audio step - the same two calls the real path makes, so a
    probe render and a reading render cannot drift apart.
    """

    def __init__(self, speak, render, cache_dir) -> None:
        self._speak = speak
        self._render = render
        self._dir = Path(cache_dir)
        self._verdict: dict[str, bool] = {}

    # -- public ------------------------------------------------------------

    def repair_for(self, voice: str, ipa: str) -> str:
        """`ipa`, with the offglide written if THIS voice was measured to need it."""
        return repair(ipa) if self._verdict.get(voice) else ipa

    def measure(self, voice: str, cfg: dict, sess, lang: str = "en-us") -> bool:
        """Decide once whether `voice` renders a bare stressed /iː/ as the wrong vowel."""
        if voice in self._verdict:
            return self._verdict[voice]
        cached = self._load(voice)
        if cached is None:
            try:
                cached = self._run(voice, cfg, sess, lang)
            except Exception as exc:  # noqa: BLE001 - a probe fault must not lose audio
                print(f"ghost/voice: vowel probe failed ({exc})", file=sys.stderr)
                cached = False
            self._save(voice, cached)
        self._verdict[voice] = cached
        return cached

    # -- internals ---------------------------------------------------------

    def _path(self, voice: str) -> Path:
        return self._dir / f"{voice}.probe.json"

    def _load(self, voice: str):
        try:
            got = json.loads(self._path(voice).read_text())
        except Exception:  # noqa: BLE001 - absent, unreadable or half-written
            return None
        return bool(got["offglide"]) if isinstance(got, dict) and "offglide" in got else None

    def _save(self, voice: str, offglide: bool) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(voice).with_suffix(".json.part")
            tmp.write_text(json.dumps({"offglide": offglide, "probe": SUSPECT}))
            tmp.replace(self._path(voice))  # atomic: a reader sees whole file or none
        except Exception:  # noqa: BLE001 - a read-only voices dir just re-probes
            pass

    def _f1(self, voice: str, cfg: dict, sess, lang: str, word_ipa: str) -> float:
        """F1 of the vowel in `word_ipa`, said inside the frame. One render, exactly."""
        import numpy as np

        sr = int(cfg["audio"]["sample_rate"])
        before, after = FRAME.split("{}")
        heads = [w for w in before.split() if w]
        tails = [w for w in after.split() if w]
        spoken = self._speak(heads + tails, lang)
        if len(spoken) != len(heads) + len(tails):
            raise RuntimeError("frame phonemization came back the wrong length")
        words = spoken[: len(heads)] + [word_ipa] + spoken[len(heads) :]

        # EVERY CHARACTER ITS OWN SOURCE INDEX. The render step keys its spans by
        # whatever it is handed, so numbering the characters is what turns a per-token
        # span into a per-symbol one - and a vowel is a symbol, not a word.
        symbols: list = []
        for wi, w in enumerate(words):
            if wi:
                symbols.append((" ", len(symbols)))
            for ch in w:
                symbols.append((ch, len(symbols)))
        # The vowel of the probe word: its `i`/`ɪ` and any length mark after it.
        at = sum(len(w) for w in words[: len(heads)]) + len(heads)
        want = [
            k
            for k in range(at, at + len(word_ipa))
            if symbols[k][0] in "iɪ" or (symbols[k][0] == "ː" and symbols[k - 1][0] in "iɪ")
        ]
        if not want:
            raise RuntimeError(f"no vowel to measure in {word_ipa!r}")

        audio, spans = self._render(symbols, cfg, sess, dict(PROBE_PARAMS))
        got = [spans[k] for k in want if k in spans]
        if not got:
            raise RuntimeError("this voice reports no phoneme timings")
        t0 = min(s[0] for s in got)
        t1 = max(s[1] for s in got)
        lo = int((t0 + (t1 - t0) * NUCLEUS[0]) * sr)
        hi = int((t0 + (t1 - t0) * NUCLEUS[1]) * sr)
        f = _formants(np.asarray(audio)[lo:hi], sr)
        if not f:
            raise RuntimeError("no measurable vowel in the probe render")
        return float(f[0])

    def _run(self, voice: str, cfg: dict, sess, lang: str) -> bool:
        spoken = self._speak([SUSPECT, TENSE_REF, LAX_REF], lang)
        if len(spoken) != 3:
            raise RuntimeError("probe phonemization came back the wrong length")
        suspect, tense, lax = spoken
        fixed = repair(suspect)
        if fixed == suspect:
            # Nothing to write - this language or voice does not spell the vowel the
            # way the repair looks for, so there is no repair to offer.
            return False

        f_tense = self._f1(voice, cfg, sess, lang, tense)
        f_lax = self._f1(voice, cfg, sess, lang, lax)
        f_suspect = self._f1(voice, cfg, sess, lang, suspect)
        # A reader whose two reference vowels come back on top of each other has told us
        # the measurement is not working on this voice, not that it is healthy. It is
        # also the denominator of everything below.
        scale = abs(f_lax - f_tense)
        if scale < 60.0:
            print(
                "ghost/voice: %s does not separate /iː/ from /ɪ/ in the probe "
                "(F1 %.0f vs %.0f); leaving its readings alone"
                % (voice, f_tense, f_lax),
                file=sys.stderr,
            )
            return False
        if abs(f_suspect - f_tense) <= BROKEN_AT * scale:
            return False  # says the vowel it names

        f_fixed = self._f1(voice, cfg, sess, lang, fixed)
        loss = abs(f_fixed - f_tense) - abs(f_suspect - f_tense)
        better = loss < MAX_LOSS * scale
        print(
            "ghost/voice: %s renders %s at F1 %.0f, %.2f of its own /iː/-to-/ɪ/ "
            "distance away from the /iː/ it names (%.0f, with /ɪ/ at %.0f); %s"
            % (
                voice,
                suspect,
                f_suspect,
                abs(f_suspect - f_tense) / scale,
                f_tense,
                f_lax,
                "writing the offglide (F1 %.0f)" % f_fixed
                if better
                else "the offglide makes it worse (F1 %.0f), leaving it" % f_fixed,
            ),
            file=sys.stderr,
        )
        return better
