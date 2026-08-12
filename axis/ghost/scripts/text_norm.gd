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

## Abbreviations that are ALSO ordinary English words, and so need evidence before being
## treated as abbreviations at all. See [method _abbrev_fits].
##   NEEDS_NUMBER - a reference mark, followed by the number it refers to
##   NEEDS_NAME   - a title, followed by the name it titles
const NEEDS_NUMBER := ["no", "fig", "vol", "ch"]
const NEEDS_NAME := ["mr", "mrs", "ms", "dr", "st", "mt", "prof", "rev", "sr", "jr"]

const CURRENCY := {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}


## The entry point. Strip markup, fold, expand, and hand back plain ASCII words.
##
## ABBREVIATIONS ARE RESOLVED BEFORE NUMERALS, and the order is load-bearing rather than
## incidental. The abbreviations that are also ordinary words - `no` above all - can only
## be told apart by what FOLLOWS them: "No. 5" is a number, "the answer was no." is not.
## Run the numeral expansion first and that evidence is gone, because by then the 5 has
## become the word "five" and nothing distinguishes the two cases at all.
static func normalize(text: String) -> String:
	var s := _strip_markdown(text)
	s = _fold(s)
	s = _expand_abbrev(s)
	s = _expand_numbers(s)
	return s


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
## Underscore emphasis is left alone deliberately - eSpeak already drops `_like this_`
## correctly, and stripping underscores here would damage identifiers a technical book
## legitimately quotes.
static func _strip_markdown(text: String) -> String:
	var out := ""
	var at_line_start := true
	var i := 0
	while i < text.length():
		var c := text[i]
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
	if lo == 0:
		return cardinal(hi) + " hundred"
	if n >= 2000 and n <= 2009:
		return cardinal(n)
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


## Walk the text token by token, replacing anything numeric with words. Runs on
## FOLDED text, so it only ever sees ASCII.
static func _expand_numbers(text: String) -> String:
	var out: Array = []
	var starts := true                  # is the next token the start of a sentence?
	for raw in text.split(" ", false):
		var tok := String(raw)
		var done := _expand_token(tok)
		# A NUMERAL OPENING A SENTENCE KEEPS ITS CAPITAL. "5 was missing." expands to
		# "five was missing", which is wrong on its own terms - a sentence starts with a
		# capital - and it also destroys the one signal the sentence splitter has: a
		# lower-case word after a full stop is read as a continuation (a dialogue tag),
		# so the expansion silently welded two sentences into one subtitle card.
		if starts and done != tok and done.length() > 0:
			done = done.substr(0, 1).to_upper() + done.substr(1)
		out.append(done)
		# The next token starts a sentence if this one ended one. Closing wrappers are
		# stripped first so `it."` counts.
		var tail := tok.rstrip("\"')]")
		starts = tail.length() > 0 and tail[tail.length() - 1] in ".!?"
	return " ".join(out)


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
	return tok


## Dotted abbreviations, expanded so their period stops being read as the end of
## a sentence. Case-insensitive on the abbreviation, and only when the period is
## actually present - "no" the word must not become "number".
static func _expand_abbrev(text: String) -> String:
	var out: Array = []
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
				out.append(head + String(ABBREV[key]) + tail)
				matched = true
				break
		if not matched:
			out.append(tok)
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

