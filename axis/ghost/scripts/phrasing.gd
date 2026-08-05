extends RefCounted
class_name Phrasing

## Phrasing - sentence-level prominence, derived from the TEXT.
##
## [Phonemes] gives every word its LEXICAL stress: which syllable of `cushion`
## is strong. That is a property of the word, identical every time it is said.
## What a reading also needs is SENTENCE stress: which words in this clause get
## leaned on at all, and which are thrown away. That is a property of the
## position, and it is the difference between a machine reciting a word list and
## a person saying a sentence.
##
## Before this stage, the accent was the walk's coin flip: `randf() < 0.22 *
## appetite` decided whether a word got emphasis, so the same sentence stressed
## different words on different seeds and no seed stressed them where English
## does. The walk was GENERATING prosody it should only have been COLOURING.
##
## So: this stage sets the baseline from four well-established, parser-free
## rules, and [Voice.ProsodyWalk] modulates around it. Nothing here is seeded -
## the same text always gets the same phrasing, and the voice's temperament
## decides how hard it leans, not where.
##
##   1. CONTENT vs FUNCTION - function words carry no accent in English. This
##      is the oldest and strongest of the rules.
##   2. THE NUCLEAR ACCENT - the last content word before a boundary carries the
##      main accent of its phrase (the nuclear stress rule). It is the single
##      most audible prosodic event in a clause, and it was entirely absent.
##   3. DE-ACCENTING GIVEN INFORMATION - a content word already said in the last
##      few seconds is old news and gets demoted. This is why "I bought a CAR.
##      The car was RED" does not stress `car` twice.
##   4. STRESS CLASH - two full accents in adjacent syllables is not English
##      rhythm; the earlier of the pair steps back.
##
## Sentence-level FOCUS (contrastive stress: "I said the RED one") genuinely
## needs syntax or an author's mark and is not attempted here.

## How far back a repeated content word still counts as "given", in words.
const GIVEN_WINDOW := 14
## Prominence levels, before the walk touches them.
const P_FUNCTION := 0.12
const P_CONTENT := 0.55
const P_ONSET := 0.68              # first content word of a sentence
const P_NUCLEAR := 1.0             # last content word before a boundary
const P_GIVEN := 0.45              # multiplier for information already given
const P_CLASH := 0.75              # multiplier for the earlier of two accents


## Annotate every word of every sentence in place with `prominence` (0..1) and
## `nuclear` (bool). Takes the sentence array exactly as [Phonemes.parse]
## returns it.
static func annotate(sentences: Array) -> void:
	var recent: Array = []                 # content words seen lately, newest last
	for words in sentences:
		# 1 + 2: base level, then the nuclear accent of each phrase. A phrase
		# ends at a comma, at the sentence end, or at the last word.
		var phrase_start := 0
		for i in (words as Array).size():
			var w: Dictionary = words[i]
			var content: bool = bool(w.get("stressed", true)) and not bool(w.get("hesit", false))
			w["prominence"] = P_CONTENT if content else P_FUNCTION
			w["nuclear"] = false
			var ends: bool = String(w.get("pause_after", "none")) != "none" or i == (words as Array).size() - 1
			if ends:
				_mark_phrase(words, phrase_start, i)
				phrase_start = i + 1
		# 3: demote what was already said
		for w in words:
			if float(w.get("prominence", 0.0)) <= P_FUNCTION:
				continue
			var key := String(w.get("text", ""))
			if key.is_empty():
				continue
			if recent.has(key) and not bool(w.get("nuclear", false)):
				# the nucleus keeps its accent even when repeated: it is
				# carrying the phrase, not introducing a new referent
				w["prominence"] = float(w["prominence"]) * P_GIVEN
			recent.append(key)
			if recent.size() > GIVEN_WINDOW:
				recent.remove_at(0)
		# 4: two full accents back to back - the first steps back
		for i in range((words as Array).size() - 1):
			var a: Dictionary = words[i]
			var b: Dictionary = words[i + 1]
			if float(a.get("prominence", 0.0)) >= 0.5 and float(b.get("prominence", 0.0)) >= 0.5 \
					and not bool(a.get("nuclear", false)):
				a["prominence"] = float(a["prominence"]) * P_CLASH


## The nuclear accent of one phrase: the LAST content word in it. Falls back to
## the last word of the phrase when the phrase is all function words, so a
## phrase is never left with nothing to land on.
static func _mark_phrase(words: Array, from_i: int, to_i: int) -> void:
	if to_i < from_i:
		return
	var pick := -1
	for i in range(to_i, from_i - 1, -1):
		if float((words[i] as Dictionary).get("prominence", 0.0)) > P_FUNCTION:
			pick = i
			break
	if pick < 0:
		pick = to_i
	var w: Dictionary = words[pick]
	w["prominence"] = P_NUCLEAR
	w["nuclear"] = true
	# the first content word of the phrase gets an onset accent too, unless it
	# IS the nucleus - English phrases tend to be prominent at both ends
	for i in range(from_i, to_i + 1):
		var f: Dictionary = words[i]
		if float(f.get("prominence", 0.0)) > P_FUNCTION:
			if i != pick:
				f["prominence"] = maxf(float(f["prominence"]), P_ONSET)
			break
