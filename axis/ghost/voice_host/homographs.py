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

Tagging is nltk's averaged perceptron, over the sentence ghost already split, and
the tag alone turned out not to be enough - see the tense pass below. If nltk or
its data is missing the whole pass turns itself off and the reading is exactly what
it was before: this can degrade, but it cannot fail.

AND IT KEEPS NO WORD LISTS. Not the homographs (eSpeak knows them), not the
function words (nltk's stopword corpus knows them), not the modals, coordinators,
determiners or adverbs (the Penn tags name them). One twelve-word constant survives,
the inflection of BE and HAVE, and it is closed by construction. This is a rule, not
a preference: a list assembled from the mistakes in one chapter would need extending
after the next one, forever, and two separate drafts of this file went that way
before the policy was written down. Where a category cannot be sourced, the
limitation is documented and left in place instead - see rule 8.
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

# Everything below is the TENSE pass, and it exists because the tagger alone was
# not enough. Reported after the first attempt shipped, from one chapter:
#
#   "I stood at the mailbox and read that seed catalog"    tagged VB
#   "I read your book in two nights"                       tagged VBP
#   "you told me I'd forgotten I read it"                  tagged VBP
#
# All three are past and all three came out present. This is not a bad tagger -
# `read` is spelled the same in every tense, so there is NO morphological evidence
# in the word itself, and a per-token classifier has nothing local to go on. VB and
# VBP from this tagger therefore mean "no evidence found", not "present tense".
#
# The evidence is in the clause, and English puts it there by rule: conjuncts share
# a tense ("I stood ... and read"), a complement inherits its matrix clause's tense
# ("I'd forgotten I read it"), and a narrative holds one tense across sentences.
# So when the tag carries no evidence, the tense is taken from the nearest preceding
# FINITE verb, and failing that from the tense the surrounding prose has been
# written in.
#
# NOTHING BELOW IS A LIST OF WORDS THAT WENT WRONG. That distinction is the whole
# design of this section, and it was arrived at the hard way: the first attempt at
# it grew a 150-word hand-typed table of "function words", which is a list with no
# end - every chapter would have added to it. What replaced that table:
#
#   * the CLOSED CLASS comes from nltk's own stopword corpus, which is maintained
#     by other people and already ships with the dependency the tagger needs;
#   * MODALS, COORDINATORS, ADVERBS, DETERMINERS and every other category are
#     PENN TREEBANK TAGS, an published inventory the tagger already assigns. That
#     is what killed the modal list: `cannot`, `can't`, `won't` and `wouldn't` all
#     come back VBZ rather than MD, so enumerating them was never going to finish;
#   * the one category that is neither - a HABITUAL adverb, which would have let
#     "sometimes I read the paper" escape a past-tense chapter - was DROPPED rather
#     than guessed at, and is written up as a known limitation instead.
#
# What is left is one twelve-word constant, and it is closed by construction rather
# than by hope: the complete inflection of BE and HAVE.

# Every verb tag - "the tagger thinks this is a verb at all", which is a different
# question from what tense it is, and the one that decides whether the noun/verb
# fallback in `_reading` is allowed to run.
VERB_TAGS = frozenset(("VB", "VBP", "VBZ", "VBG", "VBD", "VBN"))
# Tags that ARE evidence of tense, and which way they point.
PAST_TAGS = frozenset(("VBD", "VBN"))
PRESENT_TAGS = frozenset(("VBZ",))
# ...and the two that are not evidence of anything: the bare form and the
# non-third-person present, which is spelled identically to the past for `read`,
# `cut`, `set`, `put` and every other invariant verb.
UNMARKED_TAGS = frozenset(("VB", "VBP"))

# PRESENT EVIDENCE FROM AN AUXILIARY, which is the one place a VBP tag can be trusted.
#
# VBP is unmarked on the word being decided (`read` spells its present and its past alike, so a
# VBP on it means the tagger found nothing), and it is not much better on an arbitrary other
# word: measured, this tagger returns VBP for `forgotten`, which is morphologically a participle
# and can only be VBN. Trusting VBP wholesale therefore traded one error for another - it fixed
# "they are issued a face and they read everything through it" and broke "I'd forgotten I read
# it", where the scan stopped at the mis-tagged participle instead of walking on to `told`.
#
# What IS reliable is an inflected form of BE or HAVE. `am`, `is`, `are`, `have`, `has` are the
# commonest present cue in English prose, they are the words that carry the tense of every
# periphrastic verb, and they are already enumerated here - closed by construction, not by hope.
# So: VBZ anywhere, VBP only on those.

# THE ONE WORD LIST IN THIS FILE, and the only one that can be complete: every
# inflected form of BE and HAVE. English has three auxiliary verbs and these are
# two of them - the two whose complement is a PARTICIPLE ("was read aloud", "had
# wound the clock"). The third, DO, takes a bare infinitive instead ("did read"),
# and it does not need naming here: it falls out as "a stopword tagged as a verb
# that is not one of these", which is also how `cannot`, `won't` and `doesn't` are
# caught without any of them being written down.
BE_HAVE = frozenset("be been being am is are was were have has had having".split())

# What can stand between the start of a sentence and an IMPERATIVE: adverbs, an
# interjection, a leading conjunction. What cannot is a SUBJECT, and that is the
# actual definition - an imperative has none.
PRE_IMPERATIVE = frozenset(("RB", "RBR", "RBS", "UH", "CC"))

# A word directly after one of these is a NOUN or an adjective; nothing else can
# follow "the most legible". The tagger called `object` a verb in "the most legible
# object a person ever makes" and the sentence's `is` then made it a present-tense
# one, so the noun came out ɑːbdʒˈɛkt. Position beats the tag here.
NOUN_POSITION = frozenset(("DT", "JJ", "JJR", "JJS", "PRP$", "POS", "PDT"))

# Where a leftward scan stops: a coordination or any punctuation. Past one of these
# the words belong to another clause with its own licensers - "I will go and read"
# must not find `will`, because the conjunct it governs is `go`. Penn tags its
# punctuation as itself, so this needs no characters spelled out either.
CLAUSE_BREAK = frozenset(("CC", ",", ":", ".", "``", "''"))

# How far back a licenser may sit. "Can you read this?" puts a subject between the
# modal and the verb; three words covers that and "would he not read" without
# reaching into an unrelated clause.
LICENSER_REACH = 3

# The narrative prior decays per sentence, so a chapter that changes tense (a
# present-tense passage inside a past-tense book) follows within a few sentences
# rather than being outvoted by everything before it. ~6 sentences of memory.
NARRATIVE_DECAY = 0.85
# How lopsided the count has to be before the prior is allowed to decide. It is the
# weakest evidence in the file and it only ever speaks when nothing else can.
NARRATIVE_MARGIN = 1.5

# Stress is not a reading, but where the stress SITS is.
#
#   `above`  əbˈʌv alone, əbˌʌv in "the above thing"   - level, not a homograph
#   `I`      ˈaɪ alone,   aɪ mid-frame                 - a mark appearing, not a reading
#   `object` ˈɑːbdʒɛkt vs ɑːbdʒˈɛkt                    - THE noun/verb distinction
#
# So: level the marks, compare the bare segments, and compare where the marks are -
# but treat "no mark at all" as matching any position, because an unstressed
# rendering of the same segments is the same word said in a sentence.
_LEVEL = str.maketrans({"ˌ": "ˈ"})
_MARK = "ˈ"


def _enclitic_aux(word: str) -> bool:
    """`I've`, `they'd`, `we're`, `I'm` - a subject with BE or HAVE fused onto it.

    ghost keeps the contraction as ONE token on purpose (the karaoke line shows the source
    spelling), and this tagger reads `I've` as a proper noun - so nothing in the cascade saw the
    auxiliary that is plainly there, and "I've read it more times than you would believe" had no
    participle evidence at all. A suffix test rather than a list, because these four enclitics ARE
    the contracted forms of be and have, and there are only four.

    `'s` is deliberately absent: it is also the possessive, and "the man's read on the situation"
    is a noun. Treating it as an auxiliary would make that a participle.
    """
    return word.endswith(("'ve", "'d", "'re", "'m"))


def _tense_at(tags: list, lower: list, j: int) -> str:
    """"past", "present" or "" for the verb at `j` - a participle's tense read off its AUXILIARY.

    "are issued" is present and "was issued" is past, and the participle is VBN in both, so a
    scan that took VBN at face value read the passive of a present clause as evidence FOR the
    past. The tense of a periphrastic verb sits on its auxiliary; this is the one place that is
    written down, and rules 6 and 7 both consult it rather than reading tags directly.

    A bare VBN with no auxiliary in front of it - "the letter read aloud in court" - is a reduced
    relative, and past.
    """
    tag = tags[j]
    if tag in ("VBN", "VBG") and j and lower[j - 1] in BE_HAVE:
        return _tense_at(tags, lower, j - 1)      # the auxiliary carries the tense
    if tag in PAST_TAGS:
        return "past"
    if tag == "VBZ" or (tag == "VBP" and lower[j] in BE_HAVE):
        return "present"
    return ""


def _shape(s: str) -> tuple:
    """(segments, stress positions) for a reading, with levels collapsed."""
    segs: list = []
    at: list = []
    for ch in s.translate(_LEVEL):
        if ch == _MARK:
            at.append(len(segs))
        else:
            segs.append(ch)
    return "".join(segs), tuple(at)


def _same_reading(a: str, b: str) -> bool:
    sa, pa = _shape(a)
    sb, pb = _shape(b)
    if sa != sb:
        return False
    return not pa or not pb or pa == pb


class _Corpus:
    """nltk's tagger and stopword list, loaded on demand and never twice.

    Both are DATA, not code, and nltk does not ship either in the wheel - together
    they are a ~6 MB download on first use. ghost already downloads voice models at
    runtime, so that is a pattern it has rather than a new one; the difference is
    that this one is allowed to fail. `tag` and `stop` end up filled in or the whole
    pass is off, and off means the reading is exactly what it was before.

    They are fetched together and fail together on purpose. The stopword list is not
    an optimisation - without it the participle rule rewrites `a`, `to`, `in` and
    `I` wherever one of them follows a `was` - so a half-loaded state would be worse
    than none at all.
    """

    tag = None  # callable(list[str]) -> [(word, penn)]
    stop: frozenset = frozenset()
    _tried = False

    # nltk renamed its English tagger between 3.8 and 3.9; ask for both and take
    # whichever the installed version can find.
    RESOURCES = (
        "averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger",
        "stopwords",
    )

    @classmethod
    def ready(cls) -> bool:
        if cls._tried:
            return cls.tag is not None
        cls._tried = True
        try:
            import nltk
            from nltk.corpus import stopwords
        except ImportError:
            print(
                "ghost/voice: nltk is not installed; homograph readings will "
                "use eSpeak's default for every part of speech",
                file=sys.stderr,
            )
            return False
        # Keep the corpora inside the venv ghost owns rather than in the user's
        # home, so removing user://voice_venv removes all of it.
        store = Path(sys.prefix) / "nltk_data"
        if str(store) not in nltk.data.path:
            nltk.data.path.insert(0, str(store))
        for attempt in (0, 1):
            try:
                nltk.pos_tag(["read"])
                words = frozenset(stopwords.words("english"))
                if not words:
                    raise LookupError("empty stopword list")
                cls.tag, cls.stop = nltk.pos_tag, words
                return True
            except LookupError:
                if attempt:
                    break
                print(
                    "ghost/voice: downloading the nltk tagger and stopword list "
                    "(once) so homographs can be read in context…",
                    file=sys.stderr,
                )
                store.mkdir(parents=True, exist_ok=True)
                for name in cls.RESOURCES:
                    try:
                        nltk.download(name, download_dir=str(store), quiet=True)
                    except Exception:  # noqa: BLE001 - offline is not fatal here
                        pass
            except Exception as exc:  # noqa: BLE001
                print(f"ghost/voice: nltk unavailable ({exc})", file=sys.stderr)
                break
        print(
            "ghost/voice: no nltk data; homograph readings will use eSpeak's "
            "default for every part of speech",
            file=sys.stderr,
        )
        return False

    @classmethod
    def closed_class(cls, word: str) -> bool:
        """Is this one of English's function words?

        nltk's stopword corpus IS that list - determiners, pronouns, prepositions,
        conjunctions, particles, the auxiliaries and their negated contractions -
        maintained by other people and covering every word the hand-written table
        this replaced had accumulated, with none of the homographs in it.
        """
        return word in cls.stop


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
        # Decayed counts of tensed verbs seen so far - the narrative prior. One
        # instance lives for the whole session and sentences arrive in reading
        # order, so by the second sentence of a past-tense chapter this is already
        # saying "past" and it keeps saying it.
        self._past = 0.0
        self._present = 0.0
        # Set by `_family` for the audit output; not used by the reading itself.
        self.last_reasons: list[tuple[str, str, str]] = []

    # -- public ------------------------------------------------------------

    def annotate(self, tokens: list, lang: str) -> int:
        """Set `ipa` on every token whose part of speech changes its reading.

        Mutates `tokens` in place and returns how many were rewritten. Tokens are
        one sentence - the unit the tagger needs and the unit `_synth_tokens`
        already groups by.
        """
        if not _Corpus.ready():
            return 0
        words, owner = self._tagger_input(tokens)
        if not words:
            return 0
        try:
            tagged = _Corpus.tag(words)
        except Exception as exc:  # noqa: BLE001 - a tagger fault must not lose audio
            print(f"ghost/voice: tagging failed ({exc})", file=sys.stderr)
            _Corpus.tag = None
            return 0

        tags = [p for _w, p in tagged]
        lower = [w.lower() for w in words]
        # BEFORE resolving anything: this sentence's own tensed verbs are part of
        # the narrative the next sentence will be read against, and part of the one
        # THIS sentence is read against too ("I stood ... and read" is settled by
        # its own `stood`).
        self._observe(tags)

        self.last_reasons = []
        wants: list[tuple[int, str, bool]] = []
        for wi in range(len(tagged)):
            ti = owner[wi]
            if ti < 0:
                continue
            fam, why = self._family(lower, tags, wi, words[wi])
            if fam is not None:
                wants.append((ti, fam, tags[wi] in VERB_TAGS))
                self.last_reasons.append((words[wi], fam, why))
        if not wants:
            return 0

        self._fill(
            lang, [(str(tokens[ti].get("text", "")), fam) for ti, fam, _v in wants]
        )

        changed = 0
        for ti, fam, verbal in wants:
            tok = tokens[ti]
            got = self._reading(lang, str(tok.get("text", "")), fam, verbal)
            if got:
                tok["ipa"] = got
                changed += 1
        return changed

    # -- the tense pass ----------------------------------------------------

    def _observe(self, tags: list) -> None:
        """Fold one sentence's tensed verbs into the narrative prior."""
        self._past *= NARRATIVE_DECAY
        self._present *= NARRATIVE_DECAY
        for t in tags:
            if t in PAST_TAGS:
                self._past += 1.0
            elif t in PRESENT_TAGS:
                self._present += 1.0

    def _narrative(self) -> str:
        """ "past", "present" or "" - what tense the prose has been written in."""
        if self._past > self._present * NARRATIVE_MARGIN + 0.5:
            return "past"
        if self._present > self._past * NARRATIVE_MARGIN + 0.5:
            return "present"
        return ""

    def _family(self, lower: list, tags: list, wi: int, word: str = "") -> tuple:
        """Which carrier this word wants, and the rule that decided it."""
        fam, why = self._decide(lower, tags, wi, word or lower[wi])
        # A WORD ALREADY SPELLED IN THE PAST HAS NO USE FOR THE PAST FRAME, and
        # asking anyway is not free: eSpeak reads "they have resented them" as
        # ɹiːsˈɛntᵻd - re-sented, the perfume - where the word alone is correctly
        # ɹᵻzˈɛntᵻd. The frame is there for verbs whose past is spelled like their
        # present (`read`, `wound`, `cut`, `set`); a regular -ed form is already
        # unambiguous, so it keeps the base frame and the artifact never arises.
        if fam == "VBD" and lower[wi].endswith("ed"):
            return "VB", why + ", already -ed"
        return fam, why

    def _decide(self, lower: list, tags: list, wi: int, word: str) -> tuple:
        """The rule that picks a family, and its name.

        Returns (family or None, reason). The order below IS the priority order,
        strongest evidence first, and every rule is about grammar rather than about
        any particular word.
        """
        penn = tags[wi]
        fam = FAMILY.get(penn)
        if _Corpus.closed_class(lower[wi]):
            return None, "closed class"

        # CAPITALS ARE NOT A PART OF SPEECH. The perceptron reads a capital as a
        # proper noun, so the first word of any sentence and anything shouted in
        # full caps comes back NNP - "Refuse not, cannot" and "FOR THE COURTS THAT
        # DO NOT CONVENE" both did. Neither is a proper noun and neither wants a
        # noun frame, and the capital carries no information here because the
        # position or the shouting explains it. Left alone rather than guessed at.
        if penn in ("NNP", "NNPS") and (wi == 0 or word.isupper()):
            return None, "capitalised by position, not a proper noun"

        # 1. A PARTICIPLE THE TAGGER CALLED SOMETHING ELSE. "were read aloud" comes
        #    back JJ and "was wound tight" comes back NN; the auxiliary in front says
        #    outright that both are participles.
        #
        #    Deliberately NOT guarded by the tag. The tagger calls `wound` a
        #    preposition in "he had wound the clock", so a guard that required a
        #    verb-ish tag would drop the one case this rule exists for; what keeps
        #    "it was a refund" and "I was in the room" out is the closed-class check
        #    a few lines up, which is nltk's stopword corpus and not a guess.
        #    ADVERBS DO NOT BREAK THE AUXILIARY'S GRIP, and the immediate-neighbour test used to
        #    let them: "is not read", "have never read", "had already read" are all participles
        #    and all three missed this rule, which is how a whole-book measurement caught them
        #    (the veto in 3b then flipped them to the present reading, which is how they became
        #    visible at all). Only adverbs and negation may be skipped - a subject or an object
        #    in between means the auxiliary governs something else.
        for back in range(1, LICENSER_REACH + 1):
            j = wi - back
            if j < 0:
                break
            if lower[j] in BE_HAVE or _enclitic_aux(lower[j]):
                return "VBD", "after auxiliary %r" % lower[j]
            if tags[j] not in ("RB", "RBR", "RBS"):
                break

        # 2. POSITION BEATS A VERB TAG for a word sitting where no verb can sit.
        #    Only a verb tag: a determiner after a determiner ("half a breath",
        #    "all that winter") is still a determiner, and pulling `a` and `that`
        #    into a noun frame is how this rule first went wrong.
        if wi and tags[wi - 1] in NOUN_POSITION and penn in VERB_TAGS:
            return "NN", "after %s %r, a noun slot" % (tags[wi - 1], lower[wi - 1])

        # 3. AN IMPERATIVE HAS NO SUBJECT, and that outranks the tag - the tagger
        #    calls a sentence-opening "Read from the near face" a participle
        #    because it has seen far more participles in that position. Verbs only,
        #    for the same reason as above: a sentence-initial noun is just a noun.
        if penn in VERB_TAGS and all(tags[j] in PRE_IMPERATIVE for j in range(wi)):
            return "VB", "no subject before it, imperative"

        # 3b. COORDINATION AGREES IN TENSE, and that outranks a past tag on a verb whose past is
        #     spelled like its present. Rule 4 below believes a marked tag, which is right when
        #     the SPELLING marks it - but `read`, `cut`, `set` and `wound` are spelled the same
        #     either way, so VBD on one of them is the tagger guessing from context, and a
        #     coordinated present clause is that same context saying otherwise.
        #
        #     Every clause of the guard is doing work. Across exactly ONE coordinator, which is
        #     rule 7's own reach and the reason a complement clause cannot qualify - without it
        #     "I know he read it" would flip, since `know` is present and the reading is past.
        #     Nothing past in between, or "he stood up and read the letter" would flip too. And
        #     never on an -ed form: that is marked by its spelling, so the tag is not a guess.
        #     A FINITE CLAUSE HAS A SUBJECT, and requiring one immediately before the verb is
        #     what keeps this off reduced passives - ", read out of him by the only unit" and
        #     ", one woman, read from two sides," are participial phrases, not coordinated
        #     clauses, and both were flipped to the present reading before this guard existed.
        #     A comma is not a subject; a pronoun or a noun is. (VBN is excluded for the same
        #     reason: a participle tag names a FORM, not a tense the tagger guessed at, and
        #     `read` as a participle is ɹˈɛd whatever tense the clause around it carries.)
        subject_before = wi > 0 and tags[wi - 1] in (
            "PRP", "NN", "NNS", "NNP", "NNPS", "EX", "WDT", "WP"
        )
        if penn == "VBD" and subject_before and not lower[wi].endswith("ed"):
            crossed = False
            for j in range(wi - 1, -1, -1):
                if tags[j] == "CC":
                    if crossed:
                        break
                    crossed = True
                    continue
                if tags[j] in (".", ":", "``", "''"):
                    break
                if not crossed:
                    continue
                t = _tense_at(tags, lower, j)
                if t == "past":
                    break
                if t == "present":
                    return FAMILY["VBP"], "coordinated with present %r" % lower[j]

        # 4. The tag is real evidence - a past form, a participle, a third-person
        #    present, a noun, an adjective. Believe it.
        if penn not in UNMARKED_TAGS:
            return fam, "tagged %s" % penn

        # From here the tag is VB or VBP, which for an invariant verb means the
        # tagger found NOTHING, not that the word is present tense.

        # 5. A modal, `do` or infinitival `to` forces the bare form outright.
        for back in range(1, LICENSER_REACH + 1):
            j = wi - back
            if j < 0 or tags[j] in CLAUSE_BREAK:
                break
            if tags[j] in ("MD", "TO"):
                return "VB", "bare form after %s %r" % (tags[j], lower[j])
            # ...and DO, plus every negated modal the tagger hands back as a plain
            # verb rather than as MD - `cannot`, `can't`, `won't`, `doesn't` all
            # come back VBZ. A closed-class word tagged as a verb is an auxiliary,
            # and the two whose complement is a participle are named above; every
            # other one of them takes the bare form. No modal is listed anywhere.
            if _Corpus.closed_class(lower[j]) and tags[j] in VERB_TAGS:
                return "VB", "bare form after %s %r" % (tags[j], lower[j])

        # 6. THE NEAREST PRECEDING FINITE VERB. Conjuncts share a tense ("I stood
        #    ... and read"), and a complement clause inherits its matrix clause's
        #    ("I'd forgotten I read it"). Both are the same scan, so both are one
        #    rule rather than two special cases.
        for j in range(wi - 1, -1, -1):
            t = _tense_at(tags, lower, j)
            if t == "past":
                return "VBD", "follows %s %r" % (tags[j], lower[j])
            if t == "present":
                return fam, "follows %s %r" % (tags[j], lower[j])

        # 7. ...or the first conjunct's tense, taken from the second: "We read it
        #    and left." Only across a coordinator, so an unrelated later clause
        #    cannot reach back.
        crossed = False
        for j in range(wi + 1, len(tags)):
            if tags[j] in CLAUSE_BREAK:
                crossed = True
                continue
            if not crossed:
                continue
            t = _tense_at(tags, lower, j)
            if t == "past":
                return "VBD", "conjoined with %s %r" % (tags[j], lower[j])
            if t == "present":
                return fam, "conjoined with %s %r" % (tags[j], lower[j])

        # 8. Nothing in the clause at all. Fall back to the tense the prose is
        #    written in - much the weakest evidence here, and the only rule that can
        #    be wrong in a way no amount of grammar fixes: a habitual present inside
        #    a past-tense chapter ("sometimes I read the paper") comes out past. An
        #    earlier draft vetoed that with a list of habitual adverbs, which is
        #    exactly the kind of list this file does not keep; the audit below prints
        #    every decision that rested on this rule instead, and over a whole book
        #    there were six. A limitation someone can see beats a list nobody can
        #    finish.
        if self._narrative() == "past":
            return "VBD", "past-tense narrative (%.1f past / %.1f present)" % (
                self._past,
                self._present,
            )
        return fam, "no evidence, left as %s" % penn

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

    def _reading(self, lang: str, text: str, fam: str, verbal: bool) -> str:
        entry = self._cache.get((lang, text), {})
        bare = entry.get("", "")
        got = entry.get(fam, "")
        # A NOUN/VERB PAIR STILL WANTS THE VERB READING WHEN IT IS TAGGED PAST.
        # "the ledgers record no discrepancy" is ɹᵻkˈɔːɹd whichever past-or-present
        # tag it drew, and eSpeak's past frame answers with the noun for a word
        # whose past tense is not its bare form. So a VBD that changes nothing
        # falls through to the base verb frame.
        #
        # `verbal` gates it, and that gate is the difference between a fix and a
        # regression: the auxiliary rule promotes NOUNS to VBD too ("was wound
        # tight"), and without the gate "it was record heat" would fall through to
        # the verb frame and be spoken ɹᵻkˈɔːɹd. A word the tagger never called a
        # verb takes the past frame's answer or nothing.
        if verbal and fam == "VBD" and (not got or _same_reading(got, bare)):
            got = entry.get("VB", "")
        if not got or not bare or _same_reading(got, bare):
            return ""
        return got


# -- audit -----------------------------------------------------------------


def _audit(paths: list) -> int:
    """Print every reading this pass would change in a script, and why.

    `tests/pronounce_audit.gd` answers "which words in this chapter are uncertain";
    this answers the follow-up, "and what did the machine actually decide about
    them" - which is the question that comes up after a chapter has been heard.
    Sentences are fed in reading order so the narrative prior builds exactly as it
    does during a real reading; auditing one sentence out of context will not
    reproduce what the chapter does, and that is the point rather than a flaw.

        <venv>/bin/python voice_host/homographs.py chapters/12-*.md
    """
    import re

    from backends.piper import PiperBackend  # noqa: F401 - registers/loads espeak

    speak = PiperBackend._espeak
    hg = Homographs(speak)
    total = 0
    for path in paths:
        text = Path(path).read_text(encoding="utf8", errors="replace")
        print(f"\n=== {path}")
        for raw in re.split(r"(?<=[.!?])\s+", text):
            line = " ".join(raw.split())
            if not line:
                continue
            toks = []
            for word in line.split():
                punct = ""
                while word and word[-1] in ".,;:!?\"'":
                    punct = word[-1] + punct
                    word = word[:-1]
                word = word.lstrip("\"'([")
                if word:
                    toks.append({"text": word, "punct": punct[:1]})
            if not toks:
                continue
            hg.annotate(toks, "en-us")
            for i, tok in enumerate(toks):
                if not tok.get("ipa"):
                    continue
                total += 1
                why = ""
                for word, _fam, reason in hg.last_reasons:
                    if word == tok["text"]:
                        why = reason
                        break
                print(f"  {tok['text']:<12} -> {tok['ipa']:<14} {why}")
                print(
                    f"      {' '.join(t['text'] for t in toks[max(0, i - 7): i + 6])}"
                )
    print(f"\n{total} reading(s) changed")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(_audit(sys.argv[1:]))
