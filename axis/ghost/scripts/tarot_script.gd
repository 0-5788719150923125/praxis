extends RefCounted
class_name TarotScript

## TarotScript - a tarot reading as the voice reads it and the table performs it.
##
## The reading is prose, and a handful of own-line marks for what happens at the table between
## the passages:
##
##     <!-- tarot: shuffle -->    the deck shuffles under the words that follow
##     <!-- tarot: draw 3 -->     the card on the table goes down into the spread (if one is up),
##                                then the third card is drawn, turned and held up beside its
##                                booklet entry while the words that follow are read
##     <!-- tarot: jumper 1 -->   the first card is not drawn: it flies out of the shuffle on its
##                                own, lands, and is picked up and held up as a drawn card is
##     <!-- tarot: spread -->     the last card goes down; the whole spread lies on the table
##
## THE READING IS WRITTEN FOR THE TABLE, by [TarotProducer], one passage per mark - the intro
## after `shuffle`, one passage per card after its `draw`, the close after `spread` - and a
## hand-written one works the same way.
##
## ONE WALK, TWO READERS, as in [TabletScript]: [method parse] produces both the spoken words the
## table follows the voice by and the text the voice reads ([member speakable]), in which every
## mark is a rest long enough to perform it (`<!-- action-hold: S -->`, spliced by the voice like a
## hesitation). The rest and the table's schedule come from the same [method rest_of], so the
## voice waits exactly as long as the cards move. The shuffle is the exception that proves it:
## readers talk while they shuffle, so its rest is only the moment it takes to begin.

## An own-line mark. Group 1 is the verb, 2 the card's place in the reading (1-based).
const MARK := "^\\s*<!--\\s*tarot\\s*:\\s*(shuffle|draw|jumper|spread)\\s*(\\d+)?\\s*-->\\s*$"
## What makes a document a tarot reading.
const IS_TAROT := "<!--\\s*tarot\\s*:\\s*(?:shuffle|draw|jumper|spread)"

## THE TIMING OF THE TABLE, in seconds - the rests the voice takes for each action, and the
## length the table performs it in. The phase constants inside each are the table's own (see
## [TarotMedium]); these are their sums.
##
## The shuffle starts a beat before the first word, so the hands are already busy when the
## voice comes in.
const SHUFFLE_LEAD := 1.2
## The card on display goes down into the spread: the booklet closes and the card travels to
## its place and settles.
const LAY := 1.7
## A card is drawn: the deck squares, the top card slides off, lifts, turns over and comes up to
## be shown, and its booklet entry opens beside it.
const DRAW := 3.4
## Before the FIRST card: the deck, shuffled in the middle of the table, is squared and pushed to
## the side it is drawn from, clearing the middle for the spread.
const PUSH := 1.3
## A jumper: it flies out of the shuffle and lands, the reader looks at it a moment, then it is
## picked up and shown as a drawn card is.
const JUMP := 4.2
## The spread, once the last card is down: a moment to take it in before the close.
const SETTLE := 1.8


static func _rx(pattern: String) -> RegEx:
	var r := RegEx.new()
	r.compile(pattern)
	return r


static func is_tarot(body: String) -> bool:
	return _rx(IS_TAROT).search(body) != null


## The rest the voice takes for one action. [param showing] is whether a card is up on display
## when it begins (every draw but the first, and the spread), which first has to be laid down.
static func rest_of(kind: String, showing: bool) -> float:
	var lay := LAY if showing else 0.0
	match kind:
		"shuffle":
			return SHUFFLE_LEAD
		"draw":
			# nothing showing means nothing drawn yet: this is the first card, and the deck moves first
			return lay + DRAW if showing else PUSH + DRAW
		"jumper":
			return lay + JUMP
		"spread":
			return lay + SETTLE
	return 0.0


## [param body] as the voice should read it - see the class note. Any other text is returned
## untouched.
static func speakable(body: String) -> String:
	if not is_tarot(body):
		return body
	return String(parse(body)["speakable"])


## THE WALK. Returns:
##   passages  - [{kind, card, text}]: each mark and the words read after it, in order
##   actions   - [{kind, card, after, dur}]: `after` = how many spoken words precede the mark
##   spoken    - PackedStringArray: every spoken word, normalized ([method TabletScript.norm])
##   speakable - the text the voice reads
##   cards     - how many cards the reading draws
static func parse(body: String) -> Dictionary:
	var src := Manuscript.strip_frontmatter(body)
	var mark := _rx(MARK)
	var passages: Array = []
	# an Array, not a PackedStringArray: appending through `(d[k] as PackedStringArray)` appends
	# to a copy and every line is silently lost
	var cur := {"kind": "", "card": 0, "lines": []}
	for line in src.split("\n"):
		var m := mark.search(line)
		if m == null:
			(cur["lines"] as Array).append(line)
			continue
		passages.append(cur)
		var num := m.get_string(2)
		cur = {"kind": m.get_string(1), "card": int(num) if not num.is_empty() else 0, "lines": []}
	passages.append(cur)
	var out_passages: Array = []
	var actions: Array = []
	var spoken := PackedStringArray()
	var speak := PackedStringArray()
	var showing := false
	var cards := 0
	for p in passages:
		var kind := String((p as Dictionary)["kind"])
		var text := "\n".join(PackedStringArray((p as Dictionary)["lines"] as Array)).strip_edges()
		# THE READER'S DELIVERY MARKS go on to the voice - a delivery, a hesitation - but no comment is
		# ever a word the table follows, and any other note is not read at all
		var voiced := _voice_marks(text)
		text = Manuscript._rx(Manuscript.COMMENT).sub(text, "", true).strip_edges()
		if not kind.is_empty():
			var dur := rest_of(kind, showing)
			actions.append({"kind": kind, "card": int((p as Dictionary)["card"]),
				"after": spoken.size(), "dur": dur})
			speak.append("<!-- action-hold: %s -->" % String.num(dur, 2))
			if kind == "draw" or kind == "jumper":
				showing = true
				cards = maxi(cards, int((p as Dictionary)["card"]))
			elif kind == "spread":
				showing = false
		if kind.is_empty() and text.is_empty():
			continue
		out_passages.append({"kind": kind, "card": int((p as Dictionary)["card"]), "text": text})
		for w in text.split(" ", false):
			for piece in String(w).split("\n", false):
				var n := TabletScript.norm(piece)
				if not n.is_empty():
					spoken.append(n)
		if not text.is_empty():
			speak.append(voiced)
	return {"passages": out_passages, "actions": actions, "spoken": spoken,
		"speakable": "\n\n".join(speak), "cards": cards}


## [param text] with every comment taken out but the ones the voice acts on: a delivery, a hesitation.
static func _voice_marks(text: String) -> String:
	var lean := _rx(Manuscript.DELIVERY)
	var hes := _rx(Manuscript.HESITATION)
	var out := ""
	var at := 0
	for m in _rx(Manuscript.COMMENT).search_all(text):
		out += text.substr(at, m.get_start() - at)
		at = m.get_end()
		if lean.search(m.get_string()) != null or hes.search(m.get_string()) != null:
			out += m.get_string()
	return (out + text.substr(at)).strip_edges()


## THE VIDEO'S CHAPTERS, from a rendered take: when the intro, each card and the spread begin,
## found by following the take's own word timings ([param words], a sidecar's) through the reading
## exactly as the table does - so a chapter starts where the cards move. `[{t, kind, card}]`,
## the first at 0.
static func chapters(body: String, words: Array) -> Array:
	var p := parse(body)
	var f := ReadingFollower.new()
	f.reset(p["spoken"])
	f.extend(words)
	var out: Array = [{"t": 0.0, "kind": "intro", "card": 0}]
	for e in f.place(p["actions"], 0.0, 0.25, 0.2):
		var a: Dictionary = (e as Dictionary)["a"]
		if String(a["kind"]) == "shuffle":
			continue
		out.append({"t": maxf(0.0, float((e as Dictionary)["t0"])), "kind": String(a["kind"]),
			"card": int(a["card"])})
	return out


## A reading, written out: [param passages] are `{kind, card, text}` in order (the shape
## [method parse] returns), each its mark and then its words.
static func compose(passages: Array) -> String:
	var parts := PackedStringArray()
	for p in passages:
		var kind := String((p as Dictionary).get("kind", ""))
		var card := int((p as Dictionary).get("card", 0))
		if not kind.is_empty():
			parts.append("<!-- tarot: %s -->" % kind if kind in ["shuffle", "spread"]
				else "<!-- tarot: %s %d -->" % [kind, card])
		var text := String((p as Dictionary).get("text", "")).strip_edges()
		if not text.is_empty():
			parts.append(text)
	return "\n\n".join(parts) + "\n"
