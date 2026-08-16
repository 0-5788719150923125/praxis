#!/usr/bin/env python3
"""Tests for part-of-speech homograph readings (homographs.py).

Run it directly with the voice venv's python - no pytest needed:

    ~/.local/share/godot/app_userdata/ghost/voice_venv/bin/python \
        axis/ghost/voice_host/test_homographs.py

Two halves, and the second is the one that matters.

  SWITCHES - constructions the pass exists for. "He read the book yesterday"
  must come back with the PAST reading, and it must be eSpeak's own past
  reading, not a dictionary's.

  HOLDS - constructions it must NOT touch, which should always outnumber the
  switches. Every override replaces a reading eSpeak usually gets right, so the
  safety property is the point: an authored [R EH1 D] still wins, a word eSpeak
  reads one way in every frame is left alone, and an ordinary sentence of
  ordinary words comes out byte for byte what it was before.

The tagger and eSpeak are both real here - a fake would be answering a different
question. That means the first run downloads the nltk tagger data; if neither
that nor eSpeak is available the suite says so and exits non-zero rather than
passing vacuously.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import homographs  # noqa: E402

LANG = "en-us"


def _speak(utts: list[str], voice: str = LANG) -> list[str]:
    """eSpeak, word-separated - the same call the backend makes."""
    import espeakng_loader
    from phonemizer import phonemize
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.separator import Separator

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return [
        o.strip()
        for o in phonemize(
            utts,
            language={"en": "en-gb", "en-uk": "en-gb"}.get(voice, voice),
            backend="espeak",
            strip=True,
            with_stress=True,
            njobs=1,
            separator=Separator(word=" ", phone=""),
        )
    ]


def _tokens(sentence: str) -> list:
    """A sentence as ghost sends it: word tokens with terminal punctuation."""
    out = []
    for raw in sentence.split():
        punct = ""
        while raw and raw[-1] in ".,;:!?":
            punct = raw[-1] + punct
            raw = raw[:-1]
        out.append({"text": raw, "punct": punct[:1], "fallback": []})
    return out


def _read(sentence: str, word: str) -> str:
    """The reading `word` ends up with in `sentence` - "" meaning unchanged."""
    toks = _tokens(sentence)
    homographs.Homographs(_speak).annotate(toks, LANG)
    for t in toks:
        if t["text"].lower() == word:
            return str(t.get("ipa", ""))
    raise AssertionError(f"{word!r} is not in {sentence!r}")


def _bare(word: str) -> str:
    """What the word-by-word path produces today, and the thing to beat."""
    return _speak([word])[0]


# -- SWITCHES: it must fire here -------------------------------------------

SWITCHES = [
    # the reported bug, in the two shapes a chapter actually contains
    ("He read the book yesterday.", "read", "past"),
    ("She read to him until the lamp went out.", "read", "past"),
    ("Nobody read the sign.", "read", "past"),
    ("She has read every page.", "read", "past"),
    ("He had read the whole ledger.", "read", "past"),
    # a participle the perceptron calls something else; the auxiliary decides
    ("The letter was read aloud.", "read", "past"),
    ("The names were read out in order.", "read", "past"),
    # other homographs the same machinery covers, for free
    ("Historians record the year.", "record", "verb"),
    ("They will present the case.", "present", "verb"),
    ("The refuse was piled by the door.", "refuse", "noun"),
    ("He had wound the clock too tight.", "wound", "wound"),
]

# What each expected reading IS, asked of eSpeak rather than written down here -
# a literal ɹˈɛd in this file would be a second source of truth for the same
# question, and the wrong one the day a voice speaks British English.
EXPECT = {
    "past": ("they have {} them", 2),
    "verb": ("they will {} them", 2),
    "noun": ("the {} was", 1),
    "wound": ("they have {} them", 2),
}


def check_switches() -> None:
    for sentence, word, kind in SWITCHES:
        frame, slot = EXPECT[kind]
        want = _speak([frame.format(word)])[0].split()[slot]
        got = _read(sentence, word)
        assert got, f"{word!r} unchanged in {sentence!r} (still {_bare(word)})"
        assert got == want, f"{sentence!r}: {word} -> {got}, wanted {want}"
        print(f"    {sentence:46s} {word} {_bare(word)} -> {got}")


# -- HOLDS: it must not fire here ------------------------------------------

HOLDS = [
    # present and future: eSpeak's default is already right
    ("I will read the book tomorrow.", "read"),
    ("Can you read this?", "read"),
    ("Read it aloud.", "read"),
    ("They read quietly each night.", "read"),
    # words eSpeak does not distinguish at all - there is nothing to switch to
    ("He led the way with a lead pipe.", "lead"),
    ("Wait one minute.", "minute"),
    ("The row of chairs.", "row"),
    # the noun reading of a pair whose default IS the noun
    ("The record shows nothing.", "record"),
    ("A record of the year.", "record"),
    ("The present was wrapped.", "present"),
    # ordinary words, ordinary sentences: nothing here is a homograph
    ("The lamp went out and the room was quiet.", "lamp"),
    ("She walked to the window and looked down.", "walked"),
    ("The miller was quiet.", "miller"),
    ("The heater was warm.", "heater"),
    ("It was above the door.", "above"),
    ("They get it.", "get"),
]


def check_holds() -> None:
    for sentence, word in HOLDS:
        got = _read(sentence, word)
        assert not got, f"{sentence!r}: {word} was rewritten to {got} for nothing"
        print(f"    {sentence:46s} {word} held at {_bare(word)}")


def check_author_wins() -> None:
    """An authored [R EH1 D] outranks the tagger, exactly as it outranks eSpeak."""
    toks = _tokens("He read the book yesterday.")
    toks[1]["arpa"] = ["R", "IY1", "D"]
    homographs.Homographs(_speak).annotate(toks, LANG)
    assert "ipa" not in toks[1], f"author's reading was overwritten by {toks[1]['ipa']}"
    print("    [R IY1 D] survived a VBD context")


def check_whole_sentence_untouched() -> None:
    """A sentence with no homograph in it must come back with NOTHING set.

    The blunt version of the safety property: not "the words I thought about are
    unchanged" but "the pass touched nothing at all".
    """
    lines = [
        "The spoon went past the gills and the water closed over it.",
        "Snow came down through the lamplight, slow and without any wind.",
        "He stood at the door for a while and then he went inside.",
    ]
    for line in lines:
        toks = _tokens(line)
        n = homographs.Homographs(_speak).annotate(toks, LANG)
        assert n == 0, f"{line!r}: {n} word(s) rewritten - " + str(
            [t["text"] for t in toks if t.get("ipa")]
        )
        print(f"    {line}")


def check_degrades_without_tagger() -> None:
    """No nltk, no change - never an exception, never a lost sentence."""
    saved = homographs._Tagger.state
    try:
        homographs._Tagger.state = False
        toks = _tokens("He read the book yesterday.")
        assert homographs.Homographs(_speak).annotate(toks, LANG) == 0
        assert not any(t.get("ipa") for t in toks)
        print("    tagger off -> zero rewrites, no exception")
    finally:
        homographs._Tagger.state = saved


def check_probe_is_cached() -> None:
    """The second sentence of a chapter must not re-probe the first one's words."""
    calls = []

    def counting(utts: list[str], voice: str = LANG) -> list[str]:
        calls.append(len(utts))
        return _speak(utts, voice)

    hg = homographs.Homographs(counting)
    hg.annotate(_tokens("He read the book yesterday."), LANG)
    first = sum(calls)
    calls.clear()
    hg.annotate(_tokens("He read the book yesterday."), LANG)
    assert first > 0, "the first pass probed nothing at all"
    assert not calls, f"the repeat pass re-probed {sum(calls)} utterance(s)"
    print(f"    {first} probes for the first pass, 0 for the repeat")


CHECKS = [
    check_switches,
    check_holds,
    check_author_wins,
    check_whole_sentence_untouched,
    check_degrades_without_tagger,
    check_probe_is_cached,
]


def main() -> int:
    try:
        _speak(["read"])
    except Exception as exc:  # noqa: BLE001
        print(f"eSpeak is not available in this environment: {exc}")
        return 2
    if homographs._Tagger.get() is None:
        print("the nltk tagger is not available in this environment")
        return 2
    failed = 0
    for fn in CHECKS:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"    FAIL {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
