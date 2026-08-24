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


def _read(sentence: str, word: str, context: tuple = ()) -> str:
    """The reading `word` ends up with in `sentence` - "" meaning unchanged.

    `context` is any sentences that come BEFORE it, fed in reading order through
    the same instance. That is not decoration: the narrative prior is built from
    them, and it is the only thing that can settle a sentence like "I read your
    book in two nights", where the clause itself holds no tense at all.
    """
    hg = homographs.Homographs(_speak)
    for prior in context:
        hg.annotate(_tokens(prior), LANG)
    toks = _tokens(sentence)
    hg.annotate(toks, LANG)
    for t in toks:
        if t["text"].lower() == word:
            return str(t.get("ipa", ""))
    raise AssertionError(f"{word!r} is not in {sentence!r}")


# A few sentences of ordinary past-tense narrative, used as `context` wherever the
# claim under test is about the DISCOURSE rather than the clause. Deliberately dull
# and homograph-free, so all they contribute is a tense.
PAST_NARRATIVE = (
    "He came down the stairs and opened the door.",
    "The rain had stopped and the yard smelled of it.",
    "She waited by the gate until he found her.",
)


def _bare(word: str) -> str:
    """What the word-by-word path produces today, and the thing to beat."""
    return _speak([word])[0]


# -- SWITCHES: it must fire here -------------------------------------------

SWITCHES = [
    # the reported bug, in the two shapes a chapter actually contains
    ("He read the book yesterday.", "read", "past", ()),
    ("She read to him until the lamp went out.", "read", "past", ()),
    ("Nobody read the sign.", "read", "past", ()),
    ("She has read every page.", "read", "past", ()),
    ("He had read the whole ledger.", "read", "past", ()),
    # a participle the perceptron calls something else; the auxiliary decides
    ("The letter was read aloud.", "read", "past", ()),
    ("The names were read out in order.", "read", "past", ()),
    # AN ADVERB DOES NOT BREAK THE AUXILIARY'S GRIP. All three of these missed the
    # auxiliary rule while it only looked at the immediate neighbour, and a whole-book
    # measurement is what turned them up.
    ("A book that is not read has to live somewhere else.", "read", "past", ()),
    ("You have never read the first thing.", "read", "past", ()),
    ("They had only ever read about it.", "read", "past", ()),
    # ...AND NEITHER DOES A CONTRACTION. ghost keeps `I've` as one token so the karaoke
    # line shows the source spelling, and the tagger calls it a proper noun - so nothing
    # in the cascade saw the auxiliary until the enclitic test existed.
    ("I've read it more times than you would believe.", "read", "past", ()),
    ("If you've read this book from the front it is obvious.", "read", "past", ()),
    # THE SECOND REPORT, verbatim from north-star ch12 - three sites in one
    # paragraph, all past, all previously spoken in the present. The tagger calls
    # the first two VBP and the third VB: no morphological evidence at all, so
    # every one of these is decided by the clause around it rather than the word.
    (
        "I stood at the mailbox and read that seed catalog cover to cover.",
        "read",
        "past",
        (),
    ),  # conjunct of `stood`
    (
        "You told me I'd forgotten I read it, and you said it flat.",
        "read",
        "past",
        (),
    ),  # complement of `told`
    (
        "I read your book in two nights, the year it came.",
        "read",
        "past",
        (),
    ),  # no tense in the clause at all - the narrative decides
    # the same shapes, generalised
    ("She sat down and read for an hour.", "read", "past", ()),
    ("We read it and left.", "read", "past", ()),  # first conjunct, tense on the right
    ("I read it last night.", "read", "past", PAST_NARRATIVE),
    # other homographs the same machinery covers, for free
    ("Historians record the year.", "record", "verb", ()),
    ("They will present the case.", "present", "verb", ()),
    ("The refuse was piled by the door.", "refuse", "noun", ()),
    ("He had wound the clock too tight.", "wound", "wound", ()),
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
    for sentence, word, kind, context in SWITCHES:
        frame, slot = EXPECT[kind]
        want = _speak([frame.format(word)])[0].split()[slot]
        got = _read(sentence, word, context)
        assert got, f"{word!r} unchanged in {sentence!r} (still {_bare(word)})"
        assert got == want, f"{sentence!r}: {word} -> {got}, wanted {want}"
        print(f"    {sentence[:58]:60s} {word} {_bare(word)} -> {got}")


# -- HOLDS: it must not fire here ------------------------------------------

# COORDINATION AGREES IN TENSE, and a VBD tag on an invariant verb is a guess the clause can
# overrule. "They are issued a face and they read everything through it" is present in both
# halves; the tagger says VBD and the sentence says otherwise. The three HOLDS below are the
# constructions that must NOT be flipped by that rule, and each one broke a different clause of
# its guard while it was being written: a complement clause ("I know he read it" - no coordinator
# crossed), a past coordinate ("stood up and read" - a past verb intervenes), and a reduced
# passive (", read out of him" - a participial phrase has no subject before the verb).
PRESENT_SWITCHES = [
    ("They are issued a face and they read everything through it.", "read"),
    ("The clerks are patient and they read every line.", "read"),
]

HOLDS = [
    # present and future: eSpeak's default is already right. Each of these carries
    # the PAST_NARRATIVE context, because holding without a prior proves nothing -
    # the question is whether a licenser still beats a past-tense chapter.
    ("I will read the book tomorrow.", "read", PAST_NARRATIVE),
    ("Can you read this?", "read", PAST_NARRATIVE),
    ("He cannot read it.", "read", PAST_NARRATIVE),
    ("He did read it.", "read", PAST_NARRATIVE),
    ("She wanted to read it.", "read", PAST_NARRATIVE),
    # IMPERATIVES have no subject, and that outranks both the tag and the prior.
    # The tagger calls the first VB, the second VB and the third VBN.
    ("Read it aloud.", "read", PAST_NARRATIVE),
    ("Now read the list again.", "read", PAST_NARRATIVE),
    ("Read from the near face and stop.", "read", PAST_NARRATIVE),
    # words eSpeak does not distinguish at all - there is nothing to switch to
    ("He led the way with a lead pipe.", "lead", ()),
    ("Wait one minute.", "minute", ()),
    ("The row of chairs.", "row", ()),
    # the noun reading of a pair whose default IS the noun
    ("The record shows nothing.", "record", ()),
    ("A record of the year.", "record", ()),
    ("The present was wrapped.", "present", ()),
    # A NOUN SLOT beats a verb tag: nothing but a noun can follow "the most
    # legible", and the tagger called this one a verb.
    ("The most legible object a person makes.", "object", ()),
    # ...and the noun/verb fallback must not reach a word that was never tagged a
    # verb. "was record heat" is not the verb `record`.
    ("It was record heat that year.", "record", ()),
    # A REGULAR -ed FORM is already past and must not be handed the past frame:
    # eSpeak reads "they have resented them" as re-sented, the perfume.
    ("They flattered inward and resented outward.", "resented", PAST_NARRATIVE),
    # CAPITALS ARE NOT A PART OF SPEECH - both of these come back NNP.
    ("THAT DO NOT CONVENE.", "that", ()),
    ("Refuse, not cannot.", "refuse", ()),
    # FUNCTION WORDS, which are tagged as verbs and get reduced inside any frame.
    # This is the case the first version shipped without: every one of these sits
    # after an auxiliary, which is exactly where the participle rule fires.
    ("It was a refund.", "a", ()),
    ("I was in the room when you said it.", "in", ()),
    ("Well, so was I.", "i", ()),
    ("The mail has always run five years late.", "has", ()),
    ("I had to sit down on the porch.", "to", ()),
    ("It will be one word, maybe two.", "be", ()),
    # ordinary words, ordinary sentences: nothing here is a homograph
    ("The lamp went out and the room was quiet.", "lamp", ()),
    ("She walked to the window and looked down.", "walked", ()),
    ("The miller was quiet.", "miller", ()),
    ("The heater was warm.", "heater", ()),
    ("It was above the door.", "above", ()),
    ("They get it.", "get", ()),
]


def check_present_coordination() -> None:
    """A coordinated present clause overrules a past TAG on an invariant verb."""
    # The present reading of an invariant verb IS its bare-verb reading, which is the frame the
    # switches already use for the verb direction.
    frame, slot = EXPECT["verb"]
    for sentence, word in PRESENT_SWITCHES:
        want = _speak([frame.format(word)])[0].split()[slot]
        got = _read(sentence, word, ())
        # Either the pass rewrote it to the present reading, or it left the base reading -
        # which for `read` IS the present one. Both are correct; being rewritten to the PAST
        # reading is not.
        assert got in (
            "",
            want,
        ), f"{sentence!r}: {word} -> {got}, wanted {want} or no change"
        print(f"    {sentence[:58]:60s} {word} -> {got or 'left alone'}")


def check_holds() -> None:
    for sentence, word, context in HOLDS:
        got = _read(sentence, word, context)
        assert not got, f"{sentence!r}: {word} was rewritten to {got} for nothing"
        print(f"    {sentence[:58]:60s} {word} held at {_bare(word)}")


# The habitual present inside a past-tense chapter. There is no rule that settles
# these - "sometimes I read the paper" and "sometimes I read the paper that year"
# differ by nothing the sentence contains - so the pass gets them wrong, and this
# check pins WHERE the wrongness lives rather than pretending it does not.
#
# An earlier draft vetoed them with a hand-typed list of habitual adverbs, which is
# the kind of list that never finishes. What is asserted instead: every one of these
# is decided by the narrative prior and by nothing else. That keeps the limitation
# confined to the single weakest rule - the one the audit prints - and it fails the
# day some stronger rule starts reaching them, which would be a real regression.
KNOWN_SOFT = [
    "Sometimes I read the paper.",
    "I usually read at night.",
    "We often read together.",
]


def check_soft_cases_stay_soft() -> None:
    for sentence in KNOWN_SOFT:
        hg = homographs.Homographs(_speak)
        for prior in PAST_NARRATIVE:
            hg.annotate(_tokens(prior), LANG)
        hg.annotate(_tokens(sentence), LANG)
        why = [r for w, _f, r in hg.last_reasons if w.lower() == "read"]
        assert why, f"{sentence!r}: `read` was not considered at all"
        assert why[0].startswith("past-tense narrative"), (
            f"{sentence!r}: decided by {why[0]!r}, not by the prior - a stronger "
            "rule is now reaching a case nothing in the sentence can settle"
        )
        print(f"    {sentence:36s} soft, by {why[0]}")


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
        # WITH AUXILIARIES IN THEM. The first version of this check had none, and
        # that is precisely why it passed while the participle rule was quietly
        # rewriting `a`, `to`, `in` and `I` wherever one of these stood in front
        # of them.
        "It was a fine day, and I was in the room when you said so.",
        "The mail has always run late, and it will be late again.",
        "He had to sit down, because the porch was where he had been.",
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
    saved = homographs._Corpus.tag
    try:
        homographs._Corpus.tag = None
        toks = _tokens("He read the book yesterday.")
        assert homographs.Homographs(_speak).annotate(toks, LANG) == 0
        assert not any(t.get("ipa") for t in toks)
        print("    tagger off -> zero rewrites, no exception")
    finally:
        homographs._Corpus.tag = saved


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
    check_present_coordination,
    check_holds,
    check_soft_cases_stay_soft,
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
    if not homographs._Corpus.ready():
        print("the nltk tagger/stopword data is not available in this environment")
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
