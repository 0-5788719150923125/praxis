#!/usr/bin/env python3
"""Homograph readings chosen by part of speech - and elicited from eSpeak itself.

THE BUG THIS EXISTS FOR
-----------------------
"He read the book" and "I will read the book" are not the same word, and the
generative reading was saying the present-tense one both times - ten times in a
single export. It is not a lexicon gap: eSpeak-NG knows both readings and picks
between them from surrounding syntax. It never gets the chance here, because
`PiperBackend._espeak` phonemizes WORD BY WORD (it has to: ghost needs to know
which phones belong to which word to place the karaoke line and the per-token
timings, and a sentence-level transcription gives no word boundaries - eSpeak
happily welds "of the" into one ʌvðə). A word alone on the line has no syntax,
so eSpeak falls back to its default reading, forever.

WHY THERE IS NO PRONUNCIATION TABLE IN THIS FILE
------------------------------------------------
The obvious fix is a homograph dictionary - `g2p_en` ships one (370 headwords,
CMUdict readings, one Penn tag per entry) and it was the first thing tried. Two
measurements killed it:

  1. Its READ entry is INVERTED (`READ|R IY1 D|R EH1 D|VBD` reads "use R IY1 D
     when the tag is VBD"), so it leaves the exact reported bug in place and
     breaks the present tense as well.
  2. More fundamentally, its readings are CMUdict's and this model wants
     eSpeak's. Translating all 371 entries through `arpabet.to_symbols` and
     comparing against eSpeak's own transcription of the same word: 223 of them
     disagreed. Not lexically - ɐ vs ə, ᵻ vs ɪ, secondary stress on a different
     syllable - but that is precisely the class of difference that made "was"
     come out as "wars" (see requirements.txt). Substituting a dictionary
     reading would have changed how those words sound in EVERY context, to buy
     a fix in one.

So nothing is substituted from a table. eSpeak is asked the question it already
knows how to answer, in a form it can answer: the word is phonemized a second
time inside a CARRIER PHRASE that forces the part of speech the tagger found.

    they will {} them     verb, base/present      (VB VBP VBZ VBG)
    they have {} them     verb, past/participle   (VBD VBN)
    the {} was            noun                    (NN NNS NNP NNPS)
    the {} thing          adjective               (JJ JJR JJS)

Measured, that is enough to move eSpeak off its default:

    read     alone ɹˈiːd      "they have read them"    ɹˈɛd
    record   alone ɹˈɛkɚd     "they will record them"  ɹᵻkˈɔːɹd
    live     alone lˈaɪv      "they will live them"    lˈɪv
    wound    alone wˈuːnd     "they have wound them"   wˈaʊnd
    refuse   alone ɹᵻfjˈuːz   "the refuse was"         ɹˈɛfjuːs
    tear     alone tˈɪɹ       "they will tear them"    tˈɛɹ

and the reading that comes back is eSpeak's own, in eSpeak's own symbols, for a
model trained on eSpeak - so the substitution cannot introduce a convention
mismatch the way a dictionary one does.

THE CARRIERS END IN A CONSONANT ON PURPOSE. The first pair were "they will {}
it" and "the {} is", and over a 3000-word sample they reported 257 words as
having a POS-conditioned reading. Nearly all of them were LINKING R: `miller`
is mˈɪlɚ alone and mˈɪlɚɹ before a vowel, which is cross-word phonology and has
nothing to do with part of speech. Following the slot with `them`/`was`/`thing`
drops that to 24 words in 3000, and the residue is genuine (update, deliberate,
degenerate, abuses...) plus a handful that differ only in stress LEVEL, which
the comparison below normalizes away.

WHAT CAN AND CANNOT CHANGE
--------------------------
The same safety property `Phonemes.SPEAK_AS` has, for the same reason: a word is
only ever rewritten when the carrier reading DIFFERS from the reading the
word-by-word path already produces. A word eSpeak reads one way in every frame -
`lead`, `minute`, `sow`, `row`, which eSpeak simply does not distinguish - is
untouched, and so is every word that is not a homograph at all. An authored
`[R EH1 D]` override (or a `Phonemes.SPEAK_AS` / `names:` entry, which reach
here the same way) arrives with `arpa` set and is skipped outright: the author
still outranks everything.

Tagging is nltk's averaged perceptron, over the sentence ghost already split.
If nltk or its tagger data is missing the whole pass turns itself off and the
reading is exactly what it was before - this can degrade, but it cannot fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Penn Treebank tag -> the frame that forces that reading, and which word of the
# phonemized result is ours. Anything not listed (adverbs, determiners, numbers,
# foreign words) is left alone: eSpeak has no reading keyed to those, so a probe
# would cost a call and can only introduce noise.
CARRIERS: dict[str, tuple[str, int, int]] = {
    #        frame                    slot  expected words
    "VB": ("they will {} them", 2, 4),
    "VBD": ("they have {} them", 2, 4),
    "NN": ("the {} was", 1, 3),
    "JJ": ("the {} thing", 1, 3),
}

FAMILY: dict[str, str] = {
    "VB": "VB",
    "VBP": "VB",
    "VBZ": "VB",
    "VBG": "VB",
    "VBD": "VBD",
    "VBN": "VBD",
    "NN": "NN",
    "NNS": "NN",
    "NNP": "NN",
    "NNPS": "NN",
    "JJ": "JJ",
    "JJR": "JJ",
    "JJS": "JJ",
}

# Forms of BE and HAVE. A word tagged as anything at all directly after one of
# these is a PARTICIPLE - "the letter was read aloud" is tagged JJ by the
# perceptron, and read-as-an-adjective is not a reading anyone wants. This is
# grammar, not a word list: it never mentions a homograph.
AUXILIARIES = frozenset("is are was were be been being am has have had having".split())

# Stress LEVEL is not a reading. `above` is əbˈʌv alone and əbˌʌv in "the above
# thing"; treating that as a homograph would rewrite the word for nothing. Stress
# PLACEMENT is very much a reading (ˈɑːbdʒɛkt vs ɑːbdʒˈɛkt is the whole
# noun/verb distinction for `object`), so the marks are levelled, never dropped.
_LEVEL = str.maketrans({"ˌ": "ˈ"})


def _same_reading(a: str, b: str) -> bool:
    return a.translate(_LEVEL) == b.translate(_LEVEL)


class _Tagger:
    """nltk's perceptron tagger, loaded on demand and never twice.

    The model is DATA, not code, and nltk does not ship it in the wheel - it is
    a ~6 MB download on first use. ghost already downloads voice models at
    runtime, so that is a pattern it has rather than a new one; the difference
    is that this one is allowed to fail. `state` ends up as the tagging callable
    or as False, and False means every caller below quietly does nothing.
    """

    state: object = None  # None = untried, False = unavailable, else callable

    @classmethod
    def get(cls):
        if cls.state is not None:
            return cls.state or None
        cls.state = False
        try:
            import nltk
        except ImportError:
            print(
                "ghost/voice: nltk is not installed; homograph readings will "
                "use eSpeak's default for every part of speech",
                file=sys.stderr,
            )
            return None
        # Keep the corpus inside the venv ghost owns rather than in the user's
        # home, so removing user://voice_venv removes all of it.
        store = Path(sys.prefix) / "nltk_data"
        if str(store) not in nltk.data.path:
            nltk.data.path.insert(0, str(store))
        for attempt in (0, 1):
            try:
                nltk.pos_tag(["read"])
                cls.state = nltk.pos_tag
                return cls.state
            except LookupError:
                if attempt:
                    break
                print(
                    "ghost/voice: downloading the nltk part-of-speech tagger "
                    "(once) so homographs can be read in context…",
                    file=sys.stderr,
                )
                store.mkdir(parents=True, exist_ok=True)
                for name in (
                    "averaged_perceptron_tagger_eng",
                    "averaged_perceptron_tagger",
                ):
                    try:
                        nltk.download(name, download_dir=str(store), quiet=True)
                    except Exception:  # noqa: BLE001 - offline is not fatal here
                        pass
            except Exception as exc:  # noqa: BLE001
                print(f"ghost/voice: nltk tagger unavailable ({exc})", file=sys.stderr)
                break
        print(
            "ghost/voice: no nltk tagger data; homograph readings will use "
            "eSpeak's default for every part of speech",
            file=sys.stderr,
        )
        return None


class Homographs:
    """One per backend instance; the probe cache is what makes it worth keeping.

    `speak(utterances, lang)` is the backend's own word-level phonemizer - the
    SAME call the main path uses, so a carrier reading and a bare reading are
    always comparable.
    """

    def __init__(self, speak) -> None:
        self._speak = speak
        # (lang, text) -> {family or "" : ipa}. A chapter re-uses its vocabulary
        # heavily, so this settles within the first few sentences.
        self._cache: dict[tuple[str, str], dict[str, str]] = {}

    # -- public ------------------------------------------------------------

    def annotate(self, tokens: list, lang: str) -> int:
        """Set `ipa` on every token whose part of speech changes its reading.

        Mutates `tokens` in place and returns how many were rewritten. Tokens are
        one sentence - the unit the tagger needs and the unit `_synth_tokens`
        already groups by.
        """
        tag = _Tagger.get()
        if tag is None:
            return 0
        words, owner = self._tagger_input(tokens)
        if not words:
            return 0
        try:
            tagged = tag(words)
        except Exception as exc:  # noqa: BLE001 - a tagger fault must not lose audio
            print(f"ghost/voice: tagging failed ({exc})", file=sys.stderr)
            _Tagger.state = False
            return 0

        wants: list[tuple[int, str]] = []
        for wi, (_word, penn) in enumerate(tagged):
            ti = owner[wi]
            if ti < 0:
                continue
            fam = FAMILY.get(penn)
            # A PARTICIPLE THE TAGGER CALLED SOMETHING ELSE. "were read aloud"
            # comes back JJ and "was wound tight" comes back NN; both are the
            # past reading, and the auxiliary in front says so outright.
            if (
                wi
                and words[wi - 1].lower() in AUXILIARIES
                and fam in (None, "JJ", "NN")
            ):
                fam = "VBD"
            if fam is not None:
                wants.append((ti, fam))
        if not wants:
            return 0

        self._fill(lang, [(str(tokens[ti].get("text", "")), fam) for ti, fam in wants])

        changed = 0
        for ti, fam in wants:
            tok = tokens[ti]
            got = self._reading(lang, str(tok.get("text", "")), fam)
            if got:
                tok["ipa"] = got
                changed += 1
        return changed

    # -- internals ---------------------------------------------------------

    def _tagger_input(self, tokens: list) -> tuple[list[str], list[int]]:
        """The sentence as the tagger wants it, and which token each word is.

        Punctuation goes in as its own item (it is a strong feature - the tagger
        reads "yesterday ." differently from "yesterday") and maps to token -1,
        meaning "context only, never rewritten".
        """
        words: list[str] = []
        owner: list[int] = []
        for ti, tok in enumerate(tokens):
            text = str(tok.get("text", "")).strip()
            if text:
                words.append(text)
                # An authored [R EH1 D] arrives with `arpa` and outranks this
                # whole file; a hyphenated token is two words to eSpeak and so
                # has no single slot in a carrier. Both stay as context.
                owner.append(ti if (not tok.get("arpa") and text.isalpha()) else -1)
            punct = str(tok.get("punct", "")).strip()
            if punct:
                words.append(punct)
                owner.append(-1)
        return words, owner

    def _fill(self, lang: str, wanted: list[tuple[str, str]]) -> None:
        """Phonemize every probe not already cached, in ONE batch.

        eSpeak is fast enough that this is not the reason for batching - 400
        carrier utterances measured at 0.03 s. One call per sentence rather than
        per word is simply less to go wrong in the alignment checks below.
        """
        utts: list[str] = []
        slots: list[tuple[str, str, int, int]] = []  # text, family, slot, expected
        for text, fam in wanted:
            entry = self._cache.setdefault((lang, text), {})
            if "" not in entry:
                entry[""] = ""  # claimed, so a repeated word is probed once
                utts.append(text)
                slots.append((text, "", 0, 1))
            if fam not in entry:
                entry[fam] = ""
                frame, at, count = CARRIERS[fam]
                utts.append(frame.format(text))
                slots.append((text, fam, at, count))
            # A past-tagged NOUN/VERB pair still wants the verb reading:
            # "the ledgers record no discrepancy" is ɹᵻkˈɔːɹd whether the tagger
            # called it VBD or VBP. The base frame is fetched alongside so the
            # fallback in `_reading` costs no second round trip.
            if fam == "VBD" and "VB" not in entry:
                entry["VB"] = ""
                frame, at, count = CARRIERS["VB"]
                utts.append(frame.format(text))
                slots.append((text, "VB", at, count))
        if not utts:
            return
        try:
            spoken = self._speak(utts, lang)
        except Exception as exc:  # noqa: BLE001
            print(f"ghost/voice: homograph probe failed ({exc})", file=sys.stderr)
            return
        if len(spoken) != len(utts):
            # phonemizer drops inputs it does not like, and a short batch cannot
            # be realigned - the same failure `_symbols` guards against.
            print(
                "ghost/voice: homograph probe returned %d of %d; skipping this "
                "sentence" % (len(spoken), len(utts)),
                file=sys.stderr,
            )
            return
        for (text, fam, at, count), out in zip(slots, spoken):
            parts = str(out).strip().split()
            # eSpeak welds function words together when it feels like it. If the
            # frame did not come back with the shape we sent, the slot is not
            # where we think it is, so nothing is recorded and the word keeps the
            # reading it already had.
            if len(parts) == count:
                self._cache[(lang, text)][fam] = parts[at]

    def _reading(self, lang: str, text: str, fam: str) -> str:
        entry = self._cache.get((lang, text), {})
        bare = entry.get("", "")
        got = entry.get(fam, "")
        if fam == "VBD" and (not got or _same_reading(got, bare)):
            got = entry.get("VB", "")
        if not got or not bare or _same_reading(got, bare):
            return ""
        return got
