extends RefCounted
class_name TextNorm

## TextNorm - turn written English into speakable words, before anything else.
##
## This stage did not exist, and its absence was silent rather than noisy, which
## is why it survived so long. Two independent defects, measured on manuscript
## prose (see next/voice_intelligibility.md):
##
##   NUMERALS VANISHED. [Phonemes.word_to_phones] walks a word character by
##   character looking each up in DIGRAPHS/SINGLES, which contain only letters,
##   and the loop has no else branch - an unmatched character is skipped. A
##   purely numeric token therefore returns an EMPTY phone array and the word is
##   dropped from the utterance entirely. Not mispronounced: gone, with nothing
##   in the audio or the subtitles to show it. 2.2% of tokens on ordinary prose.
##
##   TYPOGRAPHIC PUNCTUATION BROKE LOOKUP. The punctuation set was the ASCII
##   literal ".,!?;:" and the wrapper sets were ASCII, so U+2018/2019 (curly
##   quotes), U+201C/201D, U+2013/2014 (dashes) and U+2026 (ellipsis) matched
##   nothing, survived into the dictionary key, and turned every quoted or
##   contracted word into a miss that fell through to letter-to-sound rules.
##   Dictionary coverage measured 88.3% on typographic text against 92.7% for
##   the same text in ASCII, with 10.4% of tokens damaged in total.
##
## The fix is ordered: fold characters first, then expand numbers, then hand
## clean ASCII to the tokenizer. Everything here is deterministic and has no
## opinion about voice - it is a text transform, and it is equally required by
## a procedural synthesizer and by any neural backend (see VOICE_PLAN.md P0).

# --- character folding ------------------------------------------------------
# Typographic punctuation folded to the ASCII the rest of the front end already
# understands. The dashes become a comma because that is what an em dash does
# prosodically: it is a break, not a word.
const FOLD := {
	"‘": "'", "’": "'", "‚": ",", "‛": "'",   # single quotes
	"“": "\"", "”": "\"", "„": "\"", "‟": "\"", # double quotes
	"′": "'", "″": "\"",                                 # primes
	"‐": "-", "‑": "-", "‒": "-", "–": "-",     # hyphens/en dash
	"—": ",", "―": ",",                                  # em dash = a break
	"…": ",",                                                 # ellipsis = a break
	" ": " ", " ": " ", " ": " ", " ": " ",     # exotic spaces
	"­": "",                                                  # soft hyphen
	"×": " times ", "÷": " divided by ",
	"½": " one half ", "¼": " one quarter ", "¾": " three quarters ",
	"°": " degrees ", "™": "", "®": "", "©": "",
}

# --- number words -----------------------------------------------------------
const ONES := ["zero", "one", "two", "three", "four", "five", "six", "seven",
	"eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
	"fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
const TENS := ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
	"eighty", "ninety"]
const SCALES := [[1000000000, "billion"], [1000000, "million"], [1000, "thousand"]]
# Ordinals that are not just "<cardinal>th".
const ORDINAL := {
	"one": "first", "two": "second", "three": "third", "five": "fifth",
	"eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}

# Abbreviations whose trailing period is NOT a sentence boundary. Expanded to
# their spoken form, which also removes the period and the false full stop with
# it - a chapter of dialogue was measuring 7 false sentence stops in 42.
const ABBREV := {
	"mr": "mister", "mrs": "missus", "ms": "miz", "dr": "doctor",
	"st": "saint", "mt": "mount", "prof": "professor", "rev": "reverend",
	"jr": "junior", "sr": "senior", "vs": "versus", "etc": "et cetera",
	"approx": "approximately", "dept": "department", "est": "established",
	"fig": "figure", "no": "number", "vol": "volume", "ch": "chapter",
}

## LETTER NAMES, for the letter runs inside an alphanumeric token - "H8" is a board square and
## reads "aitch eight", "v2" reads "vee two". This is the alphabet, not a word list: it is closed,
## it does not change, and there is no corpus to source it from - the same standing the digit
## vocabulary above has, which is why it lives beside it.
##
## IT IS NEVER APPLIED TO A WORD ON ITS OWN. `a`, `I` and `O` are real English words, and putting
## letter names in the lexicon would have read every article as "ay". Only a single-letter run
## INSIDE a token that also contains a digit is a letter being named.
const LETTER_NAMES := {
	"a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef", "g": "gee",
	"h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el", "m": "em", "n": "en",
	"o": "oh", "p": "pee", "q": "cue", "r": "ar", "s": "ess", "t": "tee", "u": "you",
	"v": "vee", "w": "double you", "x": "ex", "y": "why", "z": "zee",
}


## Abbreviations that are ALSO ordinary English words, and so need evidence before being
## treated as abbreviations at all. See [method _abbrev_fits].
##   NEEDS_NUMBER - a reference mark, followed by the number it refers to
##   NEEDS_NAME   - a title, followed by the name it titles
const NEEDS_NUMBER := ["no", "fig", "vol", "ch"]
const NEEDS_NAME := ["mr", "mrs", "ms", "dr", "st", "mt", "prof", "rev", "sr", "jr"]

const CURRENCY := {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}

## TEMPLATE MACROS, and their DEFAULTS.
##
##     ${CHAPTERS_BEFORE_IN_WORDS:25}   ->  25
##     ${TITLE:Charlotte's Web of Lies} ->  Charlotte's Web of Lies
##     ${WORD_COUNT_IN_WORDS}           ->  (nothing, and a warning)
##
## A manuscript can carry facts about itself that only its BUILD knows - how many
## chapters precede this one in this edition, how many words the finished book
## runs to. Prose that states such a number goes stale the moment a chapter is
## inserted ahead of it, so the source writes the placeholder instead and the
## build resolves it (rift's `build.py`, where the word-count ones are a
## fixed point: the rendered number changes the count it describes).
##
## Ghost is not the build. It is handed one chapter in a text box with no book
## around it, so it cannot compute any of those - which is what the DEFAULT is
## for: the value to read when the resolver is not present. Nothing here ever
## tries to work the real figure out; it reads what the source said to read.
##
## THE MACRO TEXT IS NEVER SPOKEN. That is the whole requirement, and the reason
## this sits at the very top of [method normalize] rather than anywhere later:
## `${CHAPTERS_BEFORE_IN_WORDS:25}` is one whitespace-delimited token, so every
## pass below would treat it as a word - and it was read aloud, dollar sign,
## underscores and all.
##
## A DEFAULT MAY BE ANY LENGTH: a word, a phrase, several sentences, whole
## paragraphs with blank lines in them. That is why this is a scanner rather
## than the regex it started as - a pattern has to say what a value may contain,
## and `[^{}]*` said "no braces", which silently refused to match
## `${Q:the set {a, b}}` and left the entire macro to be read aloud. Depth
## counting takes the value to its MATCHING brace instead, so the value's
## content is not the parser's business.
const NAME_HEAD := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
const NAME_TAIL := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789."


## The entry point. Strip markup, fold, expand, and hand back plain ASCII words.
##
## ABBREVIATIONS ARE RESOLVED BEFORE NUMERALS, and the order is load-bearing rather than
## incidental. The abbreviations that are also ordinary words - `no` above all - can only
## be told apart by what FOLLOWS them: "No. 5" is a number, "the answer was no." is not.
## Run the numeral expansion first and that evidence is gone, because by then the 5 has
## become the word "five" and nothing distinguishes the two cases at all.
static func normalize(text: String) -> String:
	return String(normalize_marked(text)["text"])


## The same normalization, plus WHAT EACH REWRITTEN RUN WAS WRITTEN AS.
##
## Returns {"text": String, "spans": [{at, len, src}]}, where each span covers a
## run of the returned text that does NOT read the way the source spelled it -
## `two thousand nine` for a source of `2009`, `five dollars` for `$5`, `doctor`
## for `Dr.`. Runs that were not rewritten get no span, because for those the
## text already is the source.
##
## WHY THIS EXISTS. Normalization is for the phoneme lookup and never for the
## reader's eyes - [Phonemes.parse] has said exactly that beside `display` since
## it was written - but `display` was taken from the text AFTER this stage, by
## which point the source spelling no longer existed anywhere. So the subtitle
## under a sentence ending "...who left the building in 2009." read "...in two
## thousand nine.", which is not what the page says. The voice must say the
## number and the screen must show the numeral, and that needs both spellings to
## survive the pipeline together rather than one replacing the other.
##
## [method normalize] is this function with the spans thrown away, rather than a
## second copy of the chain, because two paths through the same five passes is
## how the audio and the subtitles start disagreeing about what was said.
static func normalize_marked(text: String) -> Dictionary:
	# MACROS FIRST, before markup or folding. A macro is a token to the passes
	# below, and its default has to travel the whole pipeline afterwards - a
	# default of `25` is only heard as "twenty five" because _expand_numbers
	# still gets to see it. It is also why no span is recorded for one: the
	# default IS the source as far as a reader is concerned, and showing
	# `${CHAPTERS_BEFORE_IN_WORDS:25}` in a subtitle would be the original bug
	# with a longer token.
	var s := _expand_macros(text)
	s = _strip_markdown(s)
	s = _fold(s)
	var abbrev: Array = []
	s = _expand_abbrev(s, abbrev)
	var spans: Array = []
	s = _expand_numbers(s, abbrev, spans)
	return {"text": s, "spans": spans}


## Remove Markdown's own punctuation, which is TYPOGRAPHY and must never be spoken.
##
## Found by auditing a real manuscript: scripts arrive as `.md` files, and an emphasis
## marker is not silent to a phonemizer - it is a WORD. Measured, eSpeak returns
##   "*I will never hurt you*"  ->  "ASTERISK I will never hurt you ASTERISK"
##   "**bold** word"            ->  "asteriskasterisk bold asteriskasterisk word"
## which is not a mispronunciation but a whole extra spoken word at each end of every
## emphasised phrase, and there were three chapters full of them. Nothing anywhere
## warned about it; it would simply have been read aloud that way.
##
## UNDERSCORE EMPHASIS IS A NARROWER CASE and used to be left alone entirely, on the
## grounds that eSpeak already drops `_like this_` and that stripping underscores would
## damage identifiers a technical book legitimately quotes. The first half is true - it
## drops them even word by word, which is how this phonemizes (measured: `_the` -> ðə) -
## but the second half missed that the page sees them too. Nothing was spoken, and the
## SUBTITLE read `_no one_` with the markers on it, because `display` is taken from this
## text and there is no earlier copy of it to fall back to.
##
## So the emphasis pair is dropped and everything else is left exactly as it was, by the
## flanking rule markdown itself uses: an opener is preceded by whitespace or an opening
## bracket and followed by a non-space, a closer is preceded by a non-space and followed
## by whitespace or punctuation, and BOTH must be present on the same line before either
## is touched. `snake_case` is untouched because neither underscore flanks anything;
## `_private` alone is untouched because nothing closes it; and a RUN of underscores is
## never emphasis here, which keeps `__init__` whole. That last exclusion costs `__bold__`
## and is the right trade: prose writes bold as `**bold**`, and a dunder is unambiguous.
static func _strip_markdown(text: String) -> String:
	var drop := _emphasis_underscores(text)
	var out := ""
	var at_line_start := true
	var i := 0
	while i < text.length():
		var c := text[i]
		if c == "_" and drop.has(i):
			i += 1
			continue
		if c == "\n":
			at_line_start = true
			out += c
			i += 1
			continue
		# Heading hashes and blockquote arrows, but only where they are structural -
		# at the head of a line. A `#` inside a sentence usually means "number", and a
		# `>` usually means "greater than"; neither should be touched there.
		if at_line_start and (c == "#" or c == ">"):
			while i < text.length() and (text[i] == "#" or text[i] == ">" or text[i] == " "):
				i += 1
			continue
		if c != " " and c != "\t":
			at_line_start = false
		# Emphasis, strikethrough and inline code. Dropped outright rather than folded to
		# a break: they mark a span, they do not mark a pause, and turning them into
		# commas would put rests where the writing has none.
		if c == "*" or c == "`" or c == "~":
			i += 1
			continue
		out += c
		i += 1
	return out


## Which underscores in `text` are emphasis markers, as a set of indices.
##
## Separate from the strip itself because it needs to look FORWARD: an underscore is
## only a marker if its partner exists, and that cannot be decided by a scanner that
## has read one character. See [method _strip_markdown] for the rule and what it
## deliberately leaves alone.
static func _emphasis_underscores(text: String) -> Dictionary:
	const OPEN_BEFORE := " \t\n([{\"'\u201c\u2018"
	const SPACE := " \t\n"
	const CLOSE_AFTER := " \t\n.,;:!?)]}\"'\u201d\u2019"
	var drop := {}
	var n := text.length()
	var i := 0
	while i < n:
		if text[i] != "_":
			i += 1
			continue
		var run := i
		while run < n and text[run] == "_":
			run += 1
		if run - i != 1:
			i = run                     # `__` and longer are never emphasis here
			continue
		var before := text[i - 1] if i > 0 else " "
		var after := text[i + 1] if i + 1 < n else " "
		if not OPEN_BEFORE.contains(before) or SPACE.contains(after) or after == "_":
			i += 1
			continue
		# ...and its partner, on this line only: emphasis does not span a paragraph,
		# and an unmatched opener that reached the end of a chapter would take every
		# underscore after it with it.
		var k := i + 1
		var close := -1
		while k < n and text[k] != "\n":
			if text[k] != "_":
				k += 1
				continue
			var kr := k
			while kr < n and text[kr] == "_":
				kr += 1
			if kr - k == 1:
				var pb := text[k - 1]
				var pa := text[k + 1] if k + 1 < n else " "
				if not SPACE.contains(pb) and pb != "_" and CLOSE_AFTER.contains(pa):
					close = k
					break
			k = kr
		if close < 0:
			i += 1
			continue
		drop[i] = true
		drop[close] = true
		i = close + 1
	return drop


## Every `${...}` in `text`, as [{at, end, name, has_default, value, broken}].
##
## ONE definition, because [method _expand_macros] and [method unresolved_macros]
## must agree exactly about what a macro is - the first decides what is read and
## the second decides what is reported, and a panel that reports a different set
## from the one that was dropped is worse than a panel that reports nothing.
##
## `broken` marks a `${` that never closed, or one whose name is not a name.
## Those are left in the text VERBATIM rather than guessed at, so they are read
## aloud - which is the loudest possible signal and is reported as such. The
## alternative, swallowing to end of file, would delete the rest of a chapter.
static func _scan_macros(text: String) -> Array:
	var out: Array = []
	var i := text.find("${")
	while i >= 0:
		var j := i + 2
		while j < text.length() and (text[j] == " " or text[j] == "\t"):
			j += 1
		var n0 := j
		while j < text.length() and (NAME_TAIL if j > n0 else NAME_HEAD).contains(text[j]):
			j += 1
		var name := text.substr(n0, j - n0)
		var k := j
		while k < text.length() and (text[k] == " " or text[k] == "\t"):
			k += 1
		var row := {"at": i, "end": i + 2, "name": name, "has_default": false,
			"value": "", "broken": true}
		if not name.is_empty() and k < text.length() and text[k] == "}":
			row = {"at": i, "end": k + 1, "name": name, "has_default": false,
				"value": "", "broken": false}
		elif not name.is_empty() and k < text.length() and text[k] == ":":
			# TO THE MATCHING BRACE. The value is taken verbatim from here, so it
			# may be a word, a paragraph, or several with blank lines between
			# them - and a brace inside it nests rather than ending it.
			var depth := 1
			var v0 := k + 1
			var e := v0
			while e < text.length():
				if text[e] == "{":
					depth += 1
				elif text[e] == "}":
					depth -= 1
					if depth == 0:
						break
				e += 1
			if depth == 0:
				row = {"at": i, "end": e + 1, "name": name, "has_default": true,
					"value": text.substr(v0, e - v0), "broken": false}
		out.append(row)
		i = text.find("${", int(row["end"]))
	return out


## How one unusable macro is named to a human. One definition, so the log and the
## panel say the same thing about the same macro.
static func _macro_label(m: Dictionary) -> String:
	if not bool(m["broken"]):
		return String(m["name"])
	if String(m["name"]).is_empty():
		return "${…} (not a macro name)"
	return "${%s (never closed)" % String(m["name"])


## Replace every `${NAME:default}` with its default.
##
## A macro with NO default expands to nothing, and warns with its name. Both
## halves matter: reading the placeholder aloud is the failure being fixed, and
## deleting it in silence is the failure this project keeps having to un-fix -
## so it goes, and it says so, exactly like an unaligned word in a take.
## `${NAME:}` is an explicitly empty default and is silent, because writing it is
## a deliberate way to delete a placeholder from the reading.
static func _expand_macros(text: String) -> String:
	if not text.contains("${"):
		return text                 # the overwhelmingly common case, at no cost
	var out := ""
	var at := 0
	var bad: PackedStringArray = PackedStringArray()
	for row in _scan_macros(text):
		var m: Dictionary = row
		out += text.substr(at, int(m["at"]) - at)
		if bool(m["broken"]):
			out += text.substr(int(m["at"]), int(m["end"]) - int(m["at"]))
			bad.append(_macro_label(m))
		elif bool(m["has_default"]):
			out += String(m["value"])
		else:
			bad.append(_macro_label(m))
		at = int(m["end"])
	out += text.substr(at)
	if not bad.is_empty():
		push_warning("ghost/text: %d template macro(s) will not read as intended: %s - write "
			% [bad.size(), ", ".join(bad)] + "${NAME:value} to say what to read instead")
	return out


## The macros in `text` that will not be read as intended - no default, or
## malformed. Same scan and same rule as [method _expand_macros]; this is only
## the question asked without the answer, so a panel can say so before forty
## minutes of narration is made.
static func unresolved_macros(text: String) -> PackedStringArray:
	var out: PackedStringArray = PackedStringArray()
	if not text.contains("${"):
		return out
	for row in _scan_macros(text):
		var m: Dictionary = row
		if bool(m["broken"]) or not bool(m["has_default"]):
			out.append(_macro_label(m))
	return out


static func _fold(text: String) -> String:
	var out := ""
	for i in text.length():
		var c := text[i]
		out += String(FOLD[c]) if FOLD.has(c) else c
	return out


## Cardinals under a thousand. The recursive core everything else calls.
static func _under_thousand(n: int) -> String:
	if n < 20:
		return String(ONES[n])
	if n < 100:
		var t: String = TENS[n / 10]
		return t if n % 10 == 0 else t + " " + String(ONES[n % 10])
	var h: String = String(ONES[n / 100]) + " hundred"
	return h if n % 100 == 0 else h + " " + _under_thousand(n % 100)


## Any non-negative integer, spoken the way a reader would say it.
static func cardinal(n: int) -> String:
	if n == 0:
		return "zero"
	var parts: Array = []
	var rest := n
	for pair in SCALES:
		var value: int = pair[0]
		if rest >= value:
			parts.append(_under_thousand(rest / value) + " " + String(pair[1]))
			rest = rest % value
	if rest > 0:
		parts.append(_under_thousand(rest))
	return " ".join(parts)


static func ordinal(n: int) -> String:
	var words := cardinal(n)
	var idx := words.rfind(" ")
	var head := words.substr(0, idx + 1) if idx >= 0 else ""
	var tail := words.substr(idx + 1)
	if ORDINAL.has(tail):
		return head + String(ORDINAL[tail])
	if tail.ends_with("y"):                      # twenty -> twentieth
		return head + tail.substr(0, tail.length() - 1) + "ieth"
	return head + tail + "th"


## A four-digit year reads as two pairs - "nineteen eighty five", not "one
## thousand nine hundred and eighty five". 2000-2009 are the exception that
## every rule-based normalizer gets wrong; they read as full cardinals.
static func year(n: int) -> String:
	if n < 1100 or n > 2999:
		return cardinal(n)
	var hi := n / 100
	var lo := n % 100
	# THE 2000s FIRST. `lo == 0` used to win, so 2000 read as "twenty hundred" - a reading no
	# one uses, and the one four-digit year most likely to appear in a manuscript after 1999.
	if n >= 2000 and n <= 2009:
		return cardinal(n)
	if lo == 0:
		return cardinal(hi) + " hundred"
	if lo < 10:
		return cardinal(hi) + " oh " + cardinal(lo)
	return cardinal(hi) + " " + cardinal(lo)


static func _is_digits(s: String) -> bool:
	if s.is_empty():
		return false
	for i in s.length():
		if s[i] < "0" or s[i] > "9":
			return false
	return true


## Runs on FOLDED text, so it only ever sees ASCII.
## Walk the text token by token, replacing anything numeric with words.
##
## SPLIT ON ALL WHITESPACE, AND PUT IT BACK EXACTLY. This ran on `text.split(" ", false)`,
## which is a split on the SPACE CHARACTER - so a token at the end of a line arrived as
## `"2009.\n"`, with the newline welded on. `_expand_token` peels trailing punctuation from a
## set that does not contain `\n`, so it stopped at the first character, the digits never
## reached `_expand_core`, and the numeral survived the pass untouched. Downstream,
## [method Phonemes.word_to_phones] found no letters in it, returned an empty phone array, and
## the word was dropped from the utterance - which is the exact defect this whole file was
## written to fix, still live for any number that happened to end a paragraph.
##
## Reported from a chapter render: `2009` gone from "an opponent who left the building in
## 2009." - absent from the audio AND from the subtitles, with the sentence simply ending at
## "in". Measured in the take's own sidecar: the gap after "in" is 1.28 s, the ordinary
## inter-sentence gap, so nothing was spoken there either.
##
## The whitespace is emitted verbatim rather than rejoined with a single space, because it is
## STRUCTURE: paragraph breaks reach the sentence splitter, and `\n` is a full stop to it.
## `marks_in` are the spans [method _expand_abbrev] recorded over THIS text, and
## `marks_out` collects the spans over the RETURNED text - both what this pass
## rewrites and whatever came in, moved to where it now sits. The two are merged
## here rather than composed afterwards because this is the pass that knows both
## offsets: it walks the input and builds the output in one go, so at every token
## boundary it holds the input cursor and the output length at the same moment.
##
## The two rewrites cannot overlap: nothing in ABBREV expands to digits, so a
## token is at most one of them.
static func _expand_numbers(text: String, marks_in: Array = [],
		marks_out: Array = []) -> String:
	var out := ""
	var mi := 0                         # next unconsumed incoming mark
	var starts := true                  # is the next token the start of a sentence?
	var i := 0
	var n := text.length()
	while i < n:
		var c := text[i]
		if c == " " or c == "\t" or c == "\n" or c == "\r":
			out += c
			# A LINE BREAK STARTS A SENTENCE, so a numeral after one keeps its capital. The
			# tokenizer already treats `\n` as a full stop (Phonemes.parse reads it as a "stop"
			# pause), and the capital is not cosmetic: the sentence splitter reads a lower-case
			# word after a stop as a continuation, so "2009 opened it." at the head of a
			# paragraph would have been welded onto the end of the paragraph before it.
			if c == "\n":
				starts = true
			i += 1
			continue
		var j := i
		while j < n and not (text[j] in " \t\n\r"):
			j += 1
		var tok := text.substr(i, j - i)
		var was := i                    # this token's offset in the INPUT
		i = j
		var done := _expand_token(tok)
		# A NUMERAL OPENING A SENTENCE KEEPS ITS CAPITAL. "5 was missing." expands to
		# "five was missing", which is wrong on its own terms - a sentence starts with a
		# capital - and it also destroys the one signal the sentence splitter has: a
		# lower-case word after a full stop is read as a continuation (a dialogue tag),
		# so the expansion silently welded two sentences into one subtitle card.
		if starts and done != tok and done.length() > 0:
			done = done.substr(0, 1).to_upper() + done.substr(1)
		# What a reader should see here: whatever the SOURCE said. An incoming
		# mark means an earlier pass already replaced this token, so its `src` is
		# the original and beats the token in hand, which is that pass's output.
		var src := tok
		var span := done.length()
		var rewritten := done != tok
		while mi < marks_in.size() and int((marks_in[mi] as Dictionary)["at"]) < was:
			mi += 1
		if mi < marks_in.size() and int((marks_in[mi] as Dictionary)["at"]) == was:
			var m: Dictionary = marks_in[mi]
			src = String(m["src"])
			# ITS length, not this token's. An abbreviation can expand to more
			# than one word - `etc.` becomes `et cetera` - and this pass sees
			# only the first of them, so measuring the span here would cover
			# `et` and leave `cetera` reading as itself. Nothing in ABBREV
			# expands to digits, so the run cannot also have been rewritten
			# here and its length is still the one the earlier pass recorded.
			span = int(m["len"])
			rewritten = true
			mi += 1
		if rewritten:
			marks_out.append({"at": out.length(), "len": span, "src": src})
		out += done
		# The next token starts a sentence if this one ended one. Closing wrappers are
		# stripped first so `it."` counts.
		var tail := tok.rstrip("\"')]")
		starts = tail.length() > 0 and tail[tail.length() - 1] in ".!?"
	return out


static func _expand_token(tok: String) -> String:
	# hold trailing punctuation aside so it still reaches the tokenizer
	var tail := ""
	while tok.length() > 0 and tok[tok.length() - 1] in ".,!?;:\"')":
		tail = tok[tok.length() - 1] + tail
		tok = tok.substr(0, tok.length() - 1)
	var head := ""
	while tok.length() > 0 and tok[0] in "\"'(":
		head += tok[0]
		tok = tok.substr(1)
	if tok.is_empty():
		return head + tail

	var body := _expand_core(tok)
	return head + body + tail


static func _expand_core(tok: String) -> String:
	# currency: $5, £1.50
	var first := tok.substr(0, 1)
	if CURRENCY.has(first) and tok.length() > 1:
		var amount := _expand_core(tok.substr(1))
		return amount + " " + String(CURRENCY[first])
	# percentages
	if tok.ends_with("%"):
		return _expand_core(tok.substr(0, tok.length() - 1)) + " percent"
	# ordinals written with a suffix: 1st, 22nd, 3rd, 14th
	for suf in ["st", "nd", "rd", "th"]:
		if tok.length() > suf.length() and tok.to_lower().ends_with(suf):
			var digits := tok.substr(0, tok.length() - suf.length())
			if _is_digits(digits):
				return ordinal(int(digits))
	# times: 9:30
	if tok.contains(":"):
		var bits := tok.split(":")
		if bits.size() == 2 and _is_digits(bits[0]) and _is_digits(bits[1]):
			var m := int(bits[1])
			if m == 0:
				return cardinal(int(bits[0])) + " o'clock"
			return cardinal(int(bits[0])) + (" oh " if m < 10 else " ") + cardinal(m)
	# decimals: 3.14 reads digit by digit after the point
	if tok.contains("."):
		var bits2 := tok.split(".")
		if bits2.size() == 2 and _is_digits(bits2[0]) and _is_digits(bits2[1]):
			var frac := ""
			for i in String(bits2[1]).length():
				frac += " " + String(ONES[int(String(bits2[1])[i])])
			return cardinal(int(bits2[0])) + " point" + frac
	# thousands separators: 1,200
	if tok.contains(",") and _is_digits(tok.replace(",", "")):
		return cardinal(int(tok.replace(",", "")))
	# NUMERIC ranges only: 1939-1945, 3-4. The split used to run on ANY hyphenated token,
	# which quietly deleted the hyphen from every compound in the text - "twenty-five"
	# reached the subtitles as "twenty five", and so did "self-report", "hundred-year"
	# and "off-by-one". A hyphen is part of how the word is spelled, and the karaoke line
	# shows the source spelling, so removing it is a visible error even where the sound is
	# unaffected. Split only where a number is genuinely involved; everything else is a
	# word and stays whole, which the tokenizer and both phonemizers already handle.
	if tok.contains("-") and tok.length() > 1:
		var pieces := tok.split("-", false)
		var numeric := false
		for piece in pieces:
			if _is_digits(String(piece)):
				numeric = true
				break
		if numeric:
			var parts: Array = []
			for piece in pieces:
				parts.append(_expand_core(String(piece)))
			return " ".join(parts)
	# bare digits: a plausible year reads as pairs, everything else as a cardinal
	if _is_digits(tok):
		var n := int(tok)
		if tok.length() == 4 and n >= 1100 and n <= 2999:
			return year(n)
		return cardinal(n)
	# LETTERS AND DIGITS WELDED TOGETHER - "H8", "v2", "3D", "A4". Everything above has had its
	# chance, so what is left is an identifier, and it is read as its parts. This is the last of
	# the silent-loss family: nothing dropped the token, but [method Phonemes.word_to_phones]
	# walks LETTERS, so "H8" came back as a single phone and "H88" the same - audible as a
	# mumble, and measured on the manuscript at fifteen tokens (a chess chapter's board squares
	# and three version numbers). A digit run reads as a number, a lone letter as its name, and a
	# real word inside the token is left alone.
	var has_alpha := false
	var has_digit := false
	for i in tok.length():
		var c := tok[i]
		if c >= "0" and c <= "9":
			has_digit = true
		elif (c >= "a" and c <= "z") or (c >= "A" and c <= "Z"):
			has_alpha = true
	if has_alpha and has_digit:
		var parts: Array = []
		var run := ""
		var run_digit := false
		for i in tok.length():
			var c2 := tok[i]
			var is_d := c2 >= "0" and c2 <= "9"
			if run.is_empty() or is_d == run_digit:
				run += c2
				run_digit = is_d
				continue
			parts.append(_run_words(run, run_digit))
			run = c2
			run_digit = is_d
		if not run.is_empty():
			parts.append(_run_words(run, run_digit))
		return " ".join(parts)
	return tok


## One run of an alphanumeric token: digits as a number, a single letter as its name, anything
## longer as the word it already is.
static func _run_words(run: String, digits: bool) -> String:
	if digits:
		return cardinal(int(run))
	if run.length() == 1 and LETTER_NAMES.has(run.to_lower()):
		return String(LETTER_NAMES[run.to_lower()])
	return run


## Dotted abbreviations, expanded so their period stops being read as the end of
## a sentence. Case-insensitive on the abbreviation, and only when the period is
## actually present - "no" the word must not become "number".
## `marks` collects {at, len, src} over the RETURNED text for every token this
## rewrites, so [method normalize_marked] can tell a reader that `doctor` was
## written `Dr.`. Left empty by every other caller; the pass is unchanged.
static func _expand_abbrev(text: String, marks: Array = []) -> String:
	var out: Array = []
	var at := 0                       # output offset of the token about to be written
	var words := text.split(" ", false)
	for i in words.size():
		var raw := words[i]
		var tok := String(raw)
		# A WRAPPER MUST NOT HIDE THE ABBREVIATION. The match was against the whole token, so
		# `"Mr.` never equalled `mr.`: the period survived as a period and the sentence broke
		# inside the name. That was invisible while punctuation only shaped intonation, but it
		# is a real silence now that a full stop carries a measured pause - a rest dropped in
		# the middle of `"Mr. Smith,"`. Strip the wrappers, match, put them back.
		var head := ""
		var body := tok
		while body.length() > 0 and body[0] in "\"'(":
			head += body[0]
			body = body.substr(1)
		var tail := ""
		while body.length() > 0 and body[body.length() - 1] in "\"')":
			tail = body[body.length() - 1] + tail
			body = body.substr(0, body.length() - 1)
		var lower := body.to_lower()
		var matched := false
		for key in ABBREV:
			if lower == key + "." and _abbrev_fits(key, words, i):
				var said := head + String(ABBREV[key]) + tail
				marks.append({"at": at, "len": said.length(), "src": tok})
				out.append(said)
				at += said.length() + 1        # the joining space
				matched = true
				break
		if not matched:
			out.append(tok)
			at += tok.length() + 1
	return " ".join(out)


## Is this really the abbreviation, or the ordinary word spelled the same way that
## happens to end a sentence?
##
## The distinction the table could not make. `no` -> `number` fired on any token ending
## in a period, and a sentence-final "no" ALWAYS ends in a period - so "The answer was
## no." was spoken, and subtitled, as "The answer was number". The guard the original
## comment claimed ("only when the period is actually present") is no guard at all,
## because the period is exactly what a sentence-final word has.
##
## What separates them is what comes NEXT:
##   a reference abbreviation is followed by its number  - "No. 5", "fig. 3", "ch. 12"
##   a title is followed by the name it titles           - "Mr. Smith", "St. Paul"
## Everything else in the table ("etc.", "vs.", "approx.") is never an ordinary English
## word, so it needs no evidence and keeps expanding unconditionally.
static func _abbrev_fits(key: String, words: PackedStringArray, at: int) -> bool:
	var needs_number := NEEDS_NUMBER.has(key)
	var needs_name := NEEDS_NAME.has(key)
	if not needs_number and not needs_name:
		return true
	if at + 1 >= words.size():
		return false                     # nothing follows: it ended a sentence, so it is the word
	var nxt := String(words[at + 1]).lstrip("\"'(")
	if nxt.is_empty():
		return false
	if needs_number:
		return nxt[0] >= "0" and nxt[0] <= "9"
	# a title needs a capitalised word after it, which is what a name looks like
	return nxt[0] >= "A" and nxt[0] <= "Z"

