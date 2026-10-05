extends RefCounted
class_name TarotDeck

## TarotDeck - the cards a show reads with, and how an episode's deck is shuffled.
##
## THE DECK IS THE SHOW'S, and it is DATA: a `## Cards` section of the show's brief, one card per
## list item, each with an optional numeral and its meaning -
##
##     ## Cards
##
##     ### Major Arcana
##     - XVI. The Tower: sudden upheaval, a structure that was never sound. Reversed: ...
##     - 0. The Fool: ...
##
## - so a show can read the standard tarot or a deck of its own, the way many readers now use
## "alternative" decks of their own invention: an oracle of forty-four cards, a deck of Mondays.
## A `###` heading inside the section names the group the cards under it belong to (a suit, an
## arcana). Prose in the section is kept as instructions; only the list is the deck.
##
## A show that defines no cards reads the STANDARD deck: the twenty-two Major Arcana in their
## Rider-Waite-Smith order (Strength 8, Justice 11) and four suits of Ace to King, generated here
## because the structure is closed, with meanings from the CC0 corpus ([constant CORPUS]).
## [method standard_section] writes it out as a Cards section, to start a show's own from.
##
## THE DRAW IS NEVER AN AGENT'S. The deck is defined before anyone writes a word, and an episode's
## cards are a seeded shuffle of it ([method shuffled]) - the seed drawn from the operating
## system's cryptographic randomness when the episode is made, and kept, so the episode can be
## made again exactly. No agent picks a card, and none is shown the deck's order.

const MAJORS := ["The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
	"The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit", "Wheel of Fortune",
	"Justice", "The Hanged Man", "Death", "Temperance", "The Devil", "The Tower", "The Star",
	"The Moon", "The Sun", "Judgement", "The World"]
const SUITS := ["Wands", "Cups", "Swords", "Pentacles"]
const ELEMENTS := {"Wands": "fire", "Cups": "water", "Swords": "air", "Pentacles": "earth"}
const RANKS := ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
	"Page", "Knight", "Queen", "King"]

## Where the standard deck's meanings come from.
const CORPUS := "res://data/tarot/meanings.json"
## The heading that opens a show's deck: `## Cards` or `## The Cards`. Not "Deck": a brief
## describing its deck's LOOK under that heading, in bullets, would become a deck of its bullets.
const HEADING := "(?i)^(#{1,6})\\s*(?:the\\s+)?cards\\s*:?\\s*$"
const ITEM := "^\\s*(?:[-*+]|\\d+[.)])\\s+(.+)$"
## A card's own numeral, ahead of its name: `0.`, `XVI.`, `12)`.
const NUMERAL := "^(0|[IVXLCDM]+|\\d+)[.)]\\s+(.+)$"

static var _standard: Array = []
static var _corpus: Dictionary = {}
static var _corpus_loaded := false


static func _rx(p: String) -> RegEx:
	var r := RegEx.new()
	r.compile(p)
	return r


# --- the standard deck ----------------------------------------------------------------------

## The standard 78, in deck order: `{key, name, numeral, group, meaning}`. The numerals are the
## ones a Rider-Waite-Smith card prints: the major's own (the Fool's is 0), a pip's from Two to
## Ten; an Ace and a court card carry none.
static func standard() -> Array:
	if not _standard.is_empty():
		return _standard
	for i in MAJORS.size():
		_standard.append(_std_card("major_%02d" % i, MAJORS[i], "0" if i == 0 else roman(i), "Major Arcana"))
	for s in SUITS:
		for r in RANKS.size():
			var n := r + 1
			_standard.append(_std_card("%s_%02d" % [String(s).to_lower(), n], "%s of %s" % [RANKS[r], s],
				roman(n) if n >= 2 and n <= 10 else "", String(s)))
	return _standard


static func _std_card(key: String, name: String, numeral: String, group: String) -> Dictionary:
	return {"key": key, "name": name, "numeral": numeral, "group": group,
		"meaning": _corpus_line(key)}


## A standard card's meaning from the corpus, as one line: its keywords, then reversed.
static func _corpus_line(key: String) -> String:
	if not _corpus_loaded:
		_corpus_loaded = true
		var j := JSON.new()
		var raw := FileAccess.get_file_as_string(CORPUS)
		if not raw.is_empty() and j.parse(raw) == OK and j.data is Dictionary:
			_corpus = (j.data as Dictionary).get("cards", {})
		else:
			push_warning("ghost: tarot meanings missing or unreadable at %s" % CORPUS)
	var m: Dictionary = _corpus.get(key, {}) if _corpus.get(key) is Dictionary else {}
	if m.is_empty():
		return ""
	var up := ", ".join(PackedStringArray(m.get("keywords", [])))
	var rev := PackedStringArray()
	for p in (m.get("reversed", []) as Array).slice(0, 2):
		rev.append(String(p).substr(0, 1).to_lower() + String(p).substr(1))
	if rev.is_empty():
		return up
	var tail := "; ".join(rev)
	var last := tail.rstrip("\"')")
	return "%s. Reversed: %s%s" % [up, tail, "" if last.ends_with(".") or last.ends_with("!") or last.ends_with("?") else "."]


static func roman(n: int) -> String:
	var vals := [10, 9, 5, 4, 1]
	var syms := ["X", "IX", "V", "IV", "I"]
	var out := ""
	for i in vals.size():
		while n >= int(vals[i]):
			out += String(syms[i])
			n -= int(vals[i])
	return out


## The standard deck written out as a show's `## Cards` section - to start a deck of one's own
## from, or to make a show's deck explicit and editable.
static func standard_section() -> String:
	var out := PackedStringArray(["## Cards", ""])
	var group := ""
	for c in standard():
		var d: Dictionary = c
		if String(d["group"]) != group:
			group = String(d["group"])
			if out.size() > 2:
				out.append("")
			out.append("### " + group)
			out.append("")
		var num := String(d["numeral"])
		out.append("- %s%s: %s" % [(num + ". ") if not num.is_empty() else "", d["name"], d["meaning"]])
	return "\n".join(out) + "\n"


# --- a show's own deck ------------------------------------------------------------------------

## The cards [param body] defines in its Cards section, in order; empty when it defines none.
static func parse(body: String) -> Array:
	var span := _section(body)
	if span.is_empty():
		return []
	var lines: PackedStringArray = span["lines"]
	var item := _rx(ITEM)
	var numeral := _rx(NUMERAL)
	var sub := _rx("^(#{1,6})\\s+(.+?)\\s*$")
	var out: Array = []
	var keys := {}
	var group := ""
	for line in lines:
		var l := String(line)
		var h := sub.search(l)
		if h != null:
			group = h.get_string(2).strip_edges()
			continue
		var m := item.search(l)
		if m == null or _indented(l):
			# a continuation of the card above it, indented under its item - a nested bullet
			# (`  - Reversed: ...`) included: it says more about that card, it is not one
			if not out.is_empty() and _indented(l) and not l.strip_edges().is_empty():
				var last: Dictionary = out[-1]
				var more := m.get_string(1).strip_edges() if m != null else l.strip_edges()
				last["meaning"] = (String(last["meaning"]) + " " + more).strip_edges()
			continue
		var text := m.get_string(1).replace("**", "").replace("__", "").strip_edges()
		var num := ""
		var nm := numeral.search(text)
		if nm != null:
			num = nm.get_string(1)
			text = nm.get_string(2)
		# the name ends at the FIRST separator, whichever it is (a colon, or a spaced dash of any length)
		var name := text
		var meaning := ""
		var cut := -1
		var cut_len := 0
		for sep in [": ", " - ", " \u2014 ", " \u2013 "]:
			var at := text.find(sep)
			if at > 0 and (cut < 0 or at < cut):
				cut = at
				cut_len = String(sep).length()
		if cut > 0:
			name = text.substr(0, cut)
			meaning = text.substr(cut + cut_len)
		name = name.strip_edges().trim_suffix(":")
		if name.is_empty():
			continue
		var key := TarotEpisode.slug(name)
		var k := key
		var n := 2
		while keys.has(k):
			k = "%s-%d" % [key, n]
			n += 1
		keys[k] = true
		out.append({"key": k, "name": name, "numeral": num, "group": group, "meaning": meaning.strip_edges()})
	return out


## The deck [param body]'s show reads with: its own cards, or the standard 78.
static func of(body: String) -> Array:
	var own := parse(body)
	return own if not own.is_empty() else standard()


## [param body] as the agents are handed it: the Cards section's LIST taken out (each card's
## meaning reaches a writer only when that card is drawn), its prose kept, and a line saying how
## many cards the deck has.
static func strip(body: String) -> String:
	var span := _section(body)
	if span.is_empty():
		return body
	var item := _rx(ITEM)
	var kept := PackedStringArray()
	var listed := false
	for line in span["lines"] as PackedStringArray:
		var l := String(line)
		if item.search(l) != null and not _indented(l):
			listed = true
			continue
		if _indented(l) and listed:
			continue
		kept.append(l)
	var n := parse(body).size()
	var head := String(span["head"])
	var note := "(This deck has %d cards. Each card's meaning is given when it is drawn.)" % n
	var mid := "\n".join(kept).strip_edges()
	var section := head + "\n\n" + ((mid + "\n\n") if not mid.is_empty() else "") + note + "\n"
	return String(span["before"]) + section + String(span["after"])


## A line indented under the item above it.
static func _indented(l: String) -> bool:
	return l.begins_with("  ") or l.begins_with("\t")


## The Cards section of [param body]: `{before, head, lines, after}`, or empty.
static func _section(body: String) -> Dictionary:
	var src := Manuscript.strip_frontmatter(body)
	var lines := src.split("\n")
	var head := _rx(HEADING)
	var start := -1
	var level := 0
	for i in lines.size():
		var m := head.search(String(lines[i]))
		if m != null:
			start = i
			level = m.get_string(1).length()
			break
	if start < 0:
		return {}
	var end := lines.size()
	var any := _rx("^(#{1,6})\\s+")
	for i in range(start + 1, lines.size()):
		var m := any.search(String(lines[i]))
		if m != null and m.get_string(1).length() <= level:
			end = i
			break
	return {"before": "\n".join(lines.slice(0, start)) + ("\n" if start > 0 else ""),
		"head": String(lines[start]), "lines": lines.slice(start + 1, end),
		"after": ("\n" + "\n".join(lines.slice(end))) if end < lines.size() else ""}


# --- the draw ---------------------------------------------------------------------------------

## THE SHUFFLE: [param deck] in the order this [param seed] cuts it, each card with whether it
## comes up reversed (only when [param reversals]; about a third do, as readers who use them
## find). Returns copies of the cards, each with `reversed` added.
static func shuffled(deck: Array, seed: int, reversals: bool) -> Array:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "tarot-shuffle"])
	var order: Array = deck.duplicate()
	# Fisher-Yates, on the seeded stream - Array.shuffle() draws from the global one
	for i in range(order.size() - 1, 0, -1):
		var j := rng.randi_range(0, i)
		var t: Variant = order[i]
		order[i] = order[j]
		order[j] = t
	var out: Array = []
	for c in order:
		var card := (c as Dictionary).duplicate()
		card["reversed"] = reversals and rng.randf() < 0.32
		out.append(card)
	return out


## A fresh episode seed, from the operating system's cryptographic randomness - unpredictable,
## and kept, so the episode it makes can be made again.
static func true_seed() -> int:
	var b := Crypto.new().generate_random_bytes(4)
	return int(b.decode_u32(0) % 999999) + 1
