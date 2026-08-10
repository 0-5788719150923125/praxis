extends SceneTree

## Gate for [Phonemes.SPEAK_AS] - the words ghost pronounces itself because eSpeak
## reads them wrong.
##
## Separate from `g2p_check.gd`, which grades `word_to_phones` one word at a time
## against `data/reference.yml`. These entries cannot be tested that way: the whole
## point of a homograph is that the reading depends on the NEIGHBOURS, so a case is
## a sentence, not a word.
##
## Two halves, and the second is the one that matters:
##   SWITCHES - the constructions the rule exists for. eSpeak is measurably wrong on
##              each of these, so a regression here is the original bug returning.
##   HOLDS    - the constructions the rule must NOT touch. Every entry in SPEAK_AS
##              overrides a word that was previously handled by eSpeak, and eSpeak is
##              right far more often than it is wrong; a rule that fires too eagerly
##              breaks readings that already worked. This half is the safety net, and
##              it should always outnumber the first.
##
## `lives` regressed once in the field with no test standing behind it. That is why
## it is here alongside the entry that prompted this file.
##
## Run: godot --headless --path axis/ghost --script tests/homograph_check.gd

# sentence, the word to inspect, the ARPAbet expected (stress digits optional)
const SWITCHES := [
	# record: eSpeak reads the VERB as the NOUN after a plural-noun subject, because it
	# parses "<noun> record" as a compound. Pronoun subjects it already gets right, and
	# they are here so the rule is proven not to have BROKEN them on its way past.
	["The ledgers record no discrepancy that spring.", "record", "R IH K AO R D"],
	["The ledger records no discrepancy.", "records", "R IH K AO R D Z"],
	["Historians record the year.", "record", "R IH K AO R D"],
	["The books record a loss.", "record", "R IH K AO R D"],
	["The clerks record every entry.", "record", "R IH K AO R D"],
	["They record no discrepancy.", "record", "R IH K AO R D"],
	["He records everything.", "records", "R IH K AO R D Z"],
	["We record the sound.", "record", "R IH K AO R D"],
	["He will record the meeting.", "record", "R IH K AO R D"],
	["I want to record this.", "record", "R IH K AO R D"],
	# lives: the same failure shape - "prisoner lives" parsed as a compound noun.
	["The way a prisoner lives in his cell.", "lives", "L IH V Z"],
	["He lives here.", "lives", "L IH V Z"],
	["Their lives changed.", "lives", "L AY V Z"],
	["Many lives were lost.", "lives", "L AY V Z"],
	# hums: eSpeak spells these out ("hmm" -> həm, heard as "hem").
	["Hmm, you said, the remote still raised.", "hmm", "M M"],
	["Hm, he said.", "hm", "M M"],
]

# The half that guards against over-firing. eSpeak is already correct on all of these.
const HOLDS := [
	["The record shows nothing.", "record", "R EH K ER D"],
	["A record of the year.", "record", "R EH K ER D"],
	["She broke the record.", "record", "R EH K ER D"],
	["He put on a record player.", "record", "R EH K ER D"],
	["For the record, he lied.", "record", "R EH K ER D"],
	["It was a record year.", "record", "R EH K ER D"],
	["The records show nothing.", "records", "R EH K ER D Z"],
	["The records of the parish.", "records", "R EH K ER D Z"],
	# the veto: an object-like word follows, but a determiner precedes, so it is the noun
	["The record a clerk kept was lost.", "record", "R EH K ER D"],
	["Her records a stranger found.", "records", "R EH K ER D Z"],
]


# Invented words and proper nouns from data/english.yml's `names:` block. These must beat
# eSpeak, which reads them as `bˌaɪoʊˈʌks` and - for the second - spells the leading N out
# loud as the letter "en".
const NAMES := [
	["She calls them the Biouks.", "biouks", "B IY UW K S"],
	["The Biouks are still singing.", "biouks", "B IY UW K S"],
	["Njandejara, she calls him.", "njandejara", "N Y AA N D EH D ZH AA R AH"],
]

# Markdown is typography and must never be spoken. eSpeak turns a bare asterisk into the
# WORD "asterisk", so an emphasised line came out with an extra spoken word at each end.
const MARKUP := [
	["*I will never hurt you*", ["i", "will", "never", "hurt", "you"]],
	["**bold** word", ["bold", "word"]],
	["a `code` span", ["a", "code", "span"]],
	["# Heading text", ["heading", "text"]],
	# A full stop UNDER a closing quote, at a line break. The strip used to run one pass
	# of each, so the newline shielded the quote, and once the quote was peeled the
	# period was welded to the word: `know.` reached the dictionary as a miss. Any word
	# in that position would also have missed the homograph table, so the last word of a
	# line of dialogue could never be resolved.
	["he said, \"I know.\"\nThen left.", ["he", "said", "i", "know", "then", "left"]],
	["(aside),\nnext", ["aside", "next"]],
]

# The same trap, checked for the thing that actually breaks: a homograph in the position
# where the tokenizer used to fail must still resolve.
const SWITCHES_TRAILING := [
	["The ledgers record no discrepancy.\"\nHe signed it.", "record", "R IH K AO R D"],
]


func _init() -> void:
	var fails := 0
	fails += _run("switches", SWITCHES)
	fails += _run("holds", HOLDS)
	fails += _run("names", NAMES)
	fails += _run("trailing", SWITCHES_TRAILING)
	fails += _markup()
	if fails > 0:
		print("homograph_check: %d FAILURE(S)" % fails)
		quit(1)
		return
	print("homograph_check: ALL OK (%d cases)"
		% (SWITCHES.size() + HOLDS.size() + NAMES.size() + MARKUP.size()
			+ SWITCHES_TRAILING.size()))
	quit()


## The whole token stream, so a SPURIOUS word is caught as well as a missing one - which
## is the actual failure here. Checking only that the real words survive would pass
## happily while "asterisk" was still being spoken at both ends of the line.
func _markup() -> int:
	var bad := 0
	for c in MARKUP:
		var got: Array = []
		for s in Phonemes.parse(TextNorm.normalize(String(c[0]))):
			for w in s:
				var t := String(w.get("text", "")).to_lower().strip_edges()
				if not t.is_empty():
					got.append(t)
		var want: Array = c[1]
		if got != want:
			print("  %-9s %-46s want: %-30s got: %s"
				% ["markup", c[0], str(want), str(got)])
			bad += 1
	print("homograph_check: %-9s %d/%d" % ["markup", MARKUP.size() - bad, MARKUP.size()])
	return bad


func _run(label: String, cases: Array) -> int:
	var bad := 0
	for c in cases:
		var got := _phones_for(String(c[0]), String(c[1]))
		var want := _bare(String(c[2]))
		if got != want:
			print("  %-9s %-46s want: %-16s got: %s" % [label, c[0], want, got])
			bad += 1
	print("homograph_check: %-9s %d/%d" % [label, cases.size() - bad, cases.size()])
	return bad


## Strip stress digits, so a case can be written either way.
func _bare(spec: String) -> String:
	var out: Array = []
	for p in spec.split(" ", false):
		var t := String(p)
		while t.length() > 0 and t[t.length() - 1] >= "0" and t[t.length() - 1] <= "9":
			t = t.substr(0, t.length() - 1)
		out.append(t)
	return " ".join(out)


## The phones the front end assigns to `target` in `text`, stress digits dropped.
## Runs the REAL path - TextNorm then Phonemes.parse - so folding and tokenization
## are exercised too (a quoted "Hmm," reaching the table at all was itself a bug).
func _phones_for(text: String, target: String) -> String:
	for s in Phonemes.parse(TextNorm.normalize(text)):
		for w in s:
			if String(w.get("text", "")).to_lower().strip_edges() == target:
				var out: Array = []
				for p in w.get("phones", []):
					out.append(String(p))
				return " ".join(out)
	return "<not found>"
