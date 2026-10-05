extends RefCounted
class_name TarotPrompts

## TarotPrompts - what each agent behind a tarot episode is told. Pure: strings in, strings out,
## so a gate can hold every prompt to the rules (see tests/tarot_check.gd).
##
## THE FRAMEWORK IS GENERIC; THE SHOW IS THE BRIEF. Everything here is true of any tarot channel:
## how a reading video is built, who knows what when, what a spoken line may contain, what the
## painter must leave off a card. What THIS show is - its premise, its humor, its rules, its
## catchphrases - is the brief, the body of the show's document, handed to every agent verbatim.
##
## FIVE ROLES, and each is told only what its job needs:
##
##   producer - plans the episode BEFORE the shuffle: its title and angle, the spread, and a
##              whole deck's look. It knows no card, because no card has been drawn.
##   designer - the deck's creator, once per drawn card: that card's illustration and its entry in
##              the deck's little booklet. One card per run - it never learns what else was drawn.
##   reader   - the voice of the video, ONE PASSAGE AT A TIME, in drawing order: each run is shown
##              the reading so far and the cards turned over so far, and nothing else. The next
##              card does not exist anywhere in its input until it has been drawn. That is the
##              whole of "no cheating", and [method reader] is where it is kept: it is handed the
##              cards drawn so far, and has no other way to learn one.
##   painter  - the deck's pictures; never shown the reading.
##   set dresser - the reader's table: what stands on it, described to be built; knows no card.

## How many words each passage runs to - the length of a tarot reading video comes from these.
const WORDS := {"intro": [130, 190], "card": [110, 170], "close": [90, 140]}

## The shape every reader passage must keep, said once for every reader prompt.
## WHEN THE CARDS MOVE, for every reader prompt. A passage is spoken whole and the table acts only
## after it, in the silence before the next - so a passage that ended "And there. That's the first
## one." announced a card the viewer had not seen yet (2026-10-05, "she's signaling the draw before
## it even happened").
const MOVES := """THE CARDS MOVE ONLY BETWEEN PASSAGES. Your passage is spoken whole, and nothing on the table moves until your last word. Then, in silence: after the intro, the first card is drawn and turned over; after a card's passage, that card is laid down and the next is drawn and turned over; after the last card's passage, it is laid down with the others. The next passage begins with that done, in front of the viewer. So whatever you say happens BEFORE the next move: lead into it ("let's see what comes", "let me set this one down"), never report it as done ("there", "there it is", "here it is", "that's the first one", "beside the others") - those words belong to the passage after it, when the viewer can see it. Do not try to time a move inside your passage with a pause: nothing moves until you finish."""

const SPOKEN_RULES := """FORMAT - everything you write is spoken aloud by a voice over the video, word for word:
- Spoken words only. No stage directions, no headings, no lists, no emoji, no markdown, no quotation marks around the passage, no sound effects in brackets.
- You may put single asterisks around ONE word you lean on - *this* - and only now and then.
- Write numbers, symbols and abbreviations the way they are said ("eleven eleven", "twenty percent", "okay").
- Plain paragraphs. Short sentences land; vary the rhythm.
- The viewer sees the table, not you. Never describe your own hands or face.

HOW YOU SAY IT - the voice reading your words has a steady delivery of its own, and you can lean it a little, the way a real reader speeds up as they get going, goes graver for a hard card, or barely pauses when they are on a roll. Put a mark on a line of its own before the stretch: <!-- delivery: quicker, brighter -->. The words: quicker or slower; brighter or graver (a touch higher and livelier, or lower and flatter); tighter or looser (shorter pauses, as when you are on a roll, or longer ones, letting a moment sit); louder or softer (projecting, or leaning in); or one of excited, serious, hushed, urgent, playful, tender, dry. It lasts to the end of its paragraph and eases in and out over a few sentences. Where you would really stop - a beat before a reveal - put <!-- hesitation --> at that exact spot (<!-- hesitation: 2 --> for two seconds). Sparingly: most paragraphs carry no mark at all, and a passage seldom more than one or two. These marks are never read aloud, so never also describe the delivery in words."""


## The set dresser's reply, as a shape to copy - one thing no table would hold.
const SET_EXAMPLE := """{
  "idea": "two or three sentences: this table, and the reader who set it",
  "things": [
    {
      "name": "a boxwood chess pawn",
      "why": "the reader's reason for it, in a few words",
      "place": "back left",
      "group": "a",
      "turn": 0,
      "parts": [
        {"shape": "lathe", "smooth": true, "material": "boxwood",
         "profile": [[0,0],[1.6,0],[1.6,0.4],[1.1,0.8],[0.6,2.6],[1.0,2.9],[0.6,3.2],[0.9,4.0],[0,4.5]],
         "ornament": {"kind": "bands", "count": 2, "depth": 0.5, "from": 0.0, "to": 0.15}}
      ]
    }
  ],
  "materials": {
    "boxwood": {"kind": "wood", "color": "#d9b77a", "color2": "#a87d45", "polish": 0.5, "pattern": 0.3}
  }
}"""


## What the producer is told about the deck: the standard tarot, or the show's own cards by name
## (their meanings reach a writer only as each is drawn).
static func deck_line(deck: Array) -> String:
	if deck.is_empty() or (deck.size() == 78 and String((deck[0] as Dictionary).get("key", "")) == "major_00"):
		return "THE DECK is the standard tarot: the 78 cards of the Rider-Waite-Smith tradition."
	var names := PackedStringArray()
	for c in deck:
		names.append(String((c as Dictionary).get("name", "")))
	return "THE DECK is this show's own, %d cards: %s." % [deck.size(), ", ".join(names)]


## The context every agent shares: which show, and its brief.
static func show_context(title: String, brief: String) -> String:
	return ("You are part of the team behind a YouTube tarot channel called \"%s\". Each episode is "
		+ "one tarot reading video, filmed from the reader's chair: the deck is shuffled on a cloth, "
		+ "cards are drawn one at a time and held up to the camera beside the deck's little booklet, "
		+ "then laid down into a spread while the reader talks.\n\n"
		+ "THE SHOW'S BRIEF, from its creator. It governs everything you write:\n<brief>\n%s\n</brief>") \
		% [title, brief.strip_edges()]


## THE DICE: numbers drawn from the seed that push an episode somewhere the show has not been.
## Numbers, not word lists - a place, a year, a hue, an hour - so the space they draw from is
## the whole world rather than a list someone typed.
static func dice(seed: int) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "tarot-dice"])
	var lat := snappedf(rng.randf_range(-55.0, 70.0), 0.1)
	var lon := snappedf(rng.randf_range(-180.0, 180.0), 0.1)
	# THE CARD STOCK'S LIGHTNESS, the one die the producer is held to: asked only for "card
	# stock", it printed every deck on cream. Its own rng, so the dice above keep their values.
	var stock := RandomNumberGenerator.new()
	stock.seed = hash([seed, "tarot-stock"])
	return {
		"place": "%.1f°%s %.1f°%s" % [absf(lat), "N" if lat >= 0.0 else "S", absf(lon),
			"E" if lon >= 0.0 else "W"],
		"year": rng.randi_range(-2400, 2150),
		"hue": rng.randi_range(0, 359),
		"hour": rng.randi_range(0, 23),
		"direction": rng.randi_range(1, 12),
		"stock": stock.randi_range(5, 95),
	}


## The registry keys a look may name, for the producer to choose from: the faces the card titles
## are set in and the frames round them. Passed in by the caller so this file names no asset.
static func producer(title: String, brief: String, seed: int, cards: int, reversals: bool,
		faces: Dictionary, frames: Dictionary, history: Array, deck: Array = []) -> Dictionary:
	var d := dice(seed)
	var past := PackedStringArray()
	for h in history.slice(0, 12):
		var p: Dictionary = (h as Dictionary).get("plan", {})
		var look: Dictionary = p.get("look", {})
		past.append("- \"%s\" (%s) - deck: %s; setting: %s" % [String(p.get("episode_title", "?")),
			String(p.get("topic", "?")), String(look.get("deck_style", "?")).substr(0, 140),
			String(look.get("setting", "?")).substr(0, 100)])
	var lines := PackedStringArray()
	lines.append("You are the PRODUCER. Plan episode #%d before the camera rolls. Nobody knows which cards will come up - the deck has not been shuffled - so plan nothing that depends on a card." % seed)
	lines.append("")
	lines.append("THE SPREAD has exactly %d positions. Name the spread and each position, and say in a short phrase what each position asks. Positions are asked in order: position 1 is the first card drawn." % cards)
	if reversals:
		lines.append("This show reads reversals: some cards will come up upside down.")
	lines.append("")
	lines.append(deck_line(deck))
	lines.append("")
	lines.append("THE LOOK is a tarot deck that has never existed, and the table it is read on. Make it specific enough that an illustrator could paint every card in one consistent hand: medium, era and influences, linework, texture, palette, how figures are drawn. Then the card back (a design that reads the same when the card is turned upside down - exactly symmetric under a half turn), how much of the deck is printed in metallic foil (`foil`, 0 for matte ink to 1 for gold leaf everywhere), the surface the cards lie on (seen from directly above), the place the table stands in (seen past the far edge of the table, out of focus), the light, and how many lit candles stand on the table (the rest of what stands on it is set separately). The card's frame, its name and its numeral are printed by the deck itself, so the illustrations carry no lettering.")
	lines.append("THE CARD STOCK this deck is printed on has a lightness of about %d out of 100 (0 is black, 100 is white). It is the card's own color, all round every picture and behind its name: choose its hue, and an ink and accent that read on it, to suit the deck." % int(d["stock"]))
	lines.append("")
	lines.append("INSPIRATION. These numbers were drawn for this episode. Let them push the episode somewhere this show has never been - a culture, a period, a material, a mood - without being literal about them: a place %s; a year %d; a hue %d degrees; the hour %d:00. Before deciding, brainstorm twelve sharply different directions for the episode (topic, angle and look together), then commit to direction number %d." % [String(d["place"]), int(d["year"]), int(d["hue"]), int(d["hour"]), int(d["direction"])])
	if not past.is_empty():
		lines.append("")
		lines.append("EARLIER EPISODES of this show. Do not repeat their topics, title formulas, deck styles, palettes or settings:")
		lines.append("\n".join(past))
	lines.append("")
	lines.append("Reply with ONLY a JSON object, no other text:")
	lines.append("""{
  "brainstorm": ["twelve one-line directions"],
  "episode_title": "the video's title, as it appears on YouTube",
  "description": "the video's YouTube description, in the genre's own shape (a greeting, what this reading covers, the usual disclaimers and calls to action) and told the show's way; three short paragraphs, no emoji, no links",
  "tags": ["eight to twelve search tags, the genre's own"],
  "audience": "who this collective reading says it is for",
  "topic": "what the reading is about, in a few words",
  "premise": "the episode's angle, in one or two sentences",
  "reader_mood": "how the reader is today, and any running bit for this episode - a bit the reader SAYS, never a thing done or an object shown (the viewer sees the table and the cards, never the reader), and never a way of speaking (the show's voice is fixed: no whispering, accents or singing)",
  "spread": {"name": "the spread's name", "positions": [{"name": "position name", "asks": "what it asks"}]},
  "look": {
    "deck_name": "the deck's name",
    "deck_style": "a paragraph an illustrator paints every card from",
    "palette": ["#rrggbb", "four to six colors"],
    "card_back": "the back design, symmetric under a half turn",
    "frame": {"style": "one of: %s", "stock": "#rrggbb card stock, at that lightness", "ink": "#rrggbb border and lettering", "accent": "#rrggbb"},
    "title_face": "one of: %s",
    "foil": 0.6,
    "surface": "the cloth or tabletop the cards lie on, seen from above",
    "setting": "the place beyond the table",
    "light": {"kind": "what lights the table", "color": "#rrggbb", "warmth": "warm or cool"},
    "candles": 2
  }
}""" % [", ".join(frames.keys()), ", ".join(faces.keys())])
	return {"system": show_context(title, brief), "prompt": "\n".join(lines), "dice": d}


## The deck's creator, for ONE card: its illustration and its booklet entry.
static func designer(title: String, brief: String, look: Dictionary, card: Dictionary,
		reversals: bool) -> Dictionary:
	var lines := PackedStringArray()
	lines.append("You are the DESIGNER of the tarot deck \"%s\", used on this show. Its look:" % String(look.get("deck_name", "")))
	lines.append(String(look.get("deck_style", "")))
	lines.append("Palette: %s." % ", ".join(PackedStringArray(look.get("palette", []))))
	lines.append("")
	var group := String(card.get("group", ""))
	var element := String(TarotDeck.ELEMENTS.get(group, ""))
	lines.append("Design ONE card of the deck: %s%s." % [String(card.get("name", "")),
		(" (%s%s)" % [group, (", element of " + element) if not element.is_empty() else ""]) if not group.is_empty() else ""])
	var meaning := String(card.get("meaning", "")).strip_edges()
	if not meaning.is_empty():
		lines.append("What it means in this deck - grounding, not text to copy: %s" % meaning)
	lines.append("")
	lines.append("1. THE ILLUSTRATION: what this card shows, for the illustrator who paints the whole deck - subject, composition, the traditional symbolism of this card reinterpreted in this deck's world. Its own people and creatures, described so they could not be mistaken for another card's. Tall portrait format. No words, letters or numbers anywhere in the picture: the card's frame and name are printed separately.")
	lines.append("2. ITS ENTRY IN THE DECK'S LITTLE BOOKLET, which is shown on screen beside the card. Write it the way the brief says this deck's booklet speaks (if it does not say, in the earnest, slightly old-fashioned voice decks' booklets use). It stands alone: never mention another card.")
	lines.append("")
	lines.append("Reply with ONLY a JSON object, no other text:")
	lines.append("""{
  "art": "the illustration, in two to four sentences",
  "booklet": {
    "keywords": ["three to five words or short phrases"],
    "upright": "the meaning, forty to seventy words"%s
  }
}""" % (",\n    \"reversed\": \"the meaning reversed, twenty-five to forty-five words\"" if reversals else ""))
	return {"system": show_context(title, brief), "prompt": "\n".join(lines)}


## THE SET DRESSER: sets the reader's table before the camera rolls - every thing that stands on
## it, described in parts exactly enough to be built ([Props]), and where it stands. It knows the
## episode's plan and look, and no card: none has been drawn. [param headroom] is how tall a thing
## can stand in each zone and be seen whole (centimeters, [method TarotTable.headroom]);
## [param seen] are the things earlier episodes' tables held; [param cloth] says the cloth's
## picture goes with the prompt.
static func set_dresser(title: String, brief: String, plan: Dictionary, seed: int, headroom: Dictionary,
		seen: Array, cloth: bool) -> Dictionary:
	var look: Dictionary = plan.get("look", {}) if plan.get("look") is Dictionary else {}
	var light: Dictionary = look.get("light", {}) if look.get("light") is Dictionary else {}
	var candles := clampi(int(look.get("candles", 1)), 0, TarotTable.MAX_CANDLES)
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "tarot-table-size"])
	var lo := rng.randi_range(3, 4)
	var hi := lo + 2
	var lines := PackedStringArray()
	lines.append("You are the SET DRESSER. Before the camera rolls on episode #%d you set the reader's table: you choose every thing that stands on it, describe each exactly enough for a model maker to build it, and say where it stands. The cloth, the deck and the cards are not yours - only what stands round them." % seed)
	lines.append("")
	lines.append("THIS EPISODE")
	lines.append("Title: %s" % String(plan.get("episode_title", "")))
	lines.append("For: %s" % String(plan.get("audience", "")))
	lines.append("Topic: %s" % String(plan.get("topic", "")))
	lines.append("Angle: %s" % String(plan.get("premise", "")))
	lines.append("The deck: \"%s\" - %s" % [String(look.get("deck_name", "")), String(look.get("deck_style", ""))])
	lines.append("Palette: %s." % ", ".join(PackedStringArray(look.get("palette", []))))
	lines.append("The cloth: %s%s" % [String(look.get("surface", "")), " (the picture attached, seen from above)" if cloth else ""])
	lines.append("The room past the table: %s" % String(look.get("setting", "")))
	lines.append("The light: %s." % String(light.get("kind", "candlelight")))
	lines.append("")
	lines.append("THE TABLE A READER SETS")
	lines.append("A reader sets the table on purpose, before every reading, and each thing on it has a reason in the reading's world - light to read by, protection, the four elements the suits stand for, the reading's own subject, devotion, comfort. Give each thing its reason in a few words.")
	lines.append("This table belongs to this episode: its place, its era, its materials and its palette - the things a reader of that world would own and set out. Invent them; never reach for the same few things every time.")
	lines.append("Many readers keep stones on the table: one large piece - a cluster, a geode, a sphere, a tower - or small tumbled ones of several kinds, set straight on the cloth in a loose handful, a row or an arc, or heaped in a dish or a shell; each for what it is said to hold. Give each stone its own colors, and its play of light if it has one.")
	lines.append("Exactly %d lit candle%s, in whatever holds them (a candle is any part with a wick), and %d to %d other things." % [candles, "" if candles == 1 else "s", lo, hi])
	lines.append("Compose it as a reader does: a few groups and a few things alone, never a row of things evenly spaced. Heights vary within a group, and odd numbers sit well. The middle of the cloth stays bare: the deck is shuffled there, and the cards drawn and laid.")
	lines.append("Every thing is a real object, set out in earnest. Nothing on the table carries words, letters, numbers or labels.")
	lines.append("")
	lines.append("WHERE THINGS STAND - each thing's `place`, and the tallest a thing there can be and still be seen whole:")
	for z in TarotTable.ZONES:
		lines.append("- \"%s\": %s; up to %d cm." % [z, String((TarotTable.ZONES[z] as Dictionary)["about"]), int(headroom.get(z, 12))])
	lines.append("The camera is the reader's eye, about half a meter from the middle of the cloth and looking down across it; the top of the frame passes low over the back of the cloth, which is why the back holds only short things. Candles stand at the back or the sides, never by the deck.")
	lines.append("Things that share a `group` (any word) stand together, the tallest behind; a thing with no group stands on its own. `turn` (degrees) turns a thing about its middle; at 0 its front faces the reader.")
	lines.append("")
	lines.append("HOW A THING IS DESCRIBED")
	lines.append("In centimeters. Each thing has its own origin, the middle of its base on the cloth: y is up, x is the reader's right, and +z is toward the reader - the thing's front.")
	lines.append("A thing is one or more PARTS. A part is one SHAPE of one MATERIAL, placed by `at` [x, y, z] (where the shape's own origin goes - the middle of its base) and turned by `turn` ([x, y, z] degrees, or one number for a turn about the vertical).")
	lines.append("")
	lines.append(Props.describe())
	lines.append("")
	lines.append("DETAIL: the camera sees a thing 10 cm tall at about a sixth of the picture's height, so anything under a few millimeters is lost - engraving, glaze, grain and pattern belong in the material and its ornament, not in extra parts. Most things need 1 to 5 parts; none more than %d." % Props.MAX_PARTS)
	lines.append("SIZES are the real sizes of the real things. For scale, a tarot card here is 7 x 12 cm, and the deck stands 3 cm high.")
	if not seen.is_empty():
		lines.append("")
		lines.append("EARLIER EPISODES' TABLES held these. Set none of them again: %s." % ", ".join(PackedStringArray(seen)))
	lines.append("")
	lines.append("CHECK each thing before you answer: every part rests on the cloth or on another part (nothing floats, nothing sinks through); it stands as it would really stand - a thing with a pointed or round bottom lies on its side or sits in a stand, a ring or a bowl, never balanced on its point; the lowest point is at height 0; the sizes are real and fit the place's height; a hollow vessel's profile goes up the outside and back down the inside; every candle's wax has `wick`.")
	lines.append("")
	lines.append("Reply with ONLY a JSON object, no other text. The format, shown with one thing - a chess pawn, which never belongs on this table:")
	lines.append(SET_EXAMPLE)
	return {"system": show_context(title, brief), "prompt": "\n".join(lines)}


## THE READER, one passage. [param step] is "intro", "close", or a card's place in the reading
## ("1", "2", ...). [param said] is every passage read so far, in order; [param drawn] is every
## card turned over so far - for a card step, ENDING with the card just turned. Nothing in here
## knows the deck's order: a card not in [param drawn] has not been drawn, so it is nowhere in the
## prompt. That is the property the gate checks.
##
## [param pictured]: the paintings go with this prompt (see [method TarotProducer.say_prompt]) -
## the card's own for a card, every card's for the close - so the words are told to be about
## what is painted, and the designer's plan for the painting is left out: where the painter
## went its own way, the picture on screen is the one the viewer is looking at.
static func reader(title: String, brief: String, plan: Dictionary, step: String, said: Array,
		drawn: Array, spread_size: int, pictured := false) -> Dictionary:
	var lines := PackedStringArray()
	var positions: Array = ((plan.get("spread", {}) as Dictionary).get("positions", [])) as Array
	lines.append("You are the READER: the voice of the video. You speak every word of it.")
	lines.append("")
	lines.append("THIS EPISODE")
	lines.append("Title: %s" % String(plan.get("episode_title", "")))
	lines.append("For: %s" % String(plan.get("audience", "")))
	lines.append("Topic: %s" % String(plan.get("topic", "")))
	lines.append("Angle: %s" % String(plan.get("premise", "")))
	lines.append("You today: %s" % String(plan.get("reader_mood", "")))
	lines.append("The spread: %s, %d cards:" % [String((plan.get("spread", {}) as Dictionary).get("name", "")), spread_size])
	for i in positions.size():
		var pos: Dictionary = positions[i] if positions[i] is Dictionary else {"name": str(positions[i])}
		lines.append("  %d. %s - %s" % [i + 1, String(pos.get("name", "")), String(pos.get("asks", ""))])
	var look: Dictionary = plan.get("look", {})
	lines.append("The deck: \"%s\". On the table: %s. Also on it: %s, set out before the reading. The viewer sees the table, never you." % [
		String(look.get("deck_name", "")), String(look.get("surface", "")), TarotTable.on_the_table(look)])
	lines.append("")
	if not said.is_empty():
		lines.append("WHAT YOU HAVE SAID SO FAR, in order (the viewer heard all of it):")
		for i in said.size():
			lines.append("<said>\n%s\n</said>" % String(said[i]).strip_edges())
		lines.append("")
	var earlier: Array = drawn.slice(0, drawn.size() - 1) if step.is_valid_int() else drawn
	if not earlier.is_empty():
		lines.append("CARDS ALREADY ON THE TABLE:")
		for c in earlier:
			lines.append("  - %s" % _card_line(c as Dictionary))
		lines.append("")
	var lo_hi: Array = WORDS["card"]
	if step == "intro":
		lo_hi = WORDS["intro"]
		lines.append("NOW: the video opens. You are shuffling the deck while you talk. Welcome the viewer to \"%s\", set up this episode and who it is for, and keep the patter going while the cards are mixed. You do not know any card yet - none has been drawn - so name none and predict none. End as you stop shuffling and reach for the first card: it comes out after your last word, so lead into it and never say it is out." % title)
	elif step == "close":
		lo_hi = WORDS["close"]
		lines.append("NOW: the last card has just been laid down, and all %d cards are face up on the table. Pull the reading together - the whole spread, in the light of everything you have said - and close the episode the way the brief says the show closes." % spread_size)
		if pictured:
			lines.append("The paintings above are those cards as they lie on the table, in the order they were drawn (a reversed card upside down) - the viewer sees them as you do. Draw on what is painted only if it helps you tie the reading together; you do not have to.")
	else:
		var c: Dictionary = drawn[drawn.size() - 1]
		var k := int(step)
		var pos: Dictionary = positions[k - 1] if k - 1 < positions.size() and positions[k - 1] is Dictionary else {}
		if bool(c.get("jumper", false)):
			lines.append("NOW: before you could draw, a card flew out of the deck on its own while you were shuffling - a jumper. It landed on the table and you have picked it up. Readers treat a jumper as a card that insists. React to that first.")
		else:
			lines.append("NOW: you have just drawn card %d of %d and turned it over." % [k, spread_size])
		lines.append("It is %s." % _card_line(c))
		var means := String(c.get("meaning", "")).strip_edges()
		if not means.is_empty():
			lines.append("What it means in this deck: %s" % means)
		lines.append("It sits in position %d, \"%s\" - %s." % [k, String(pos.get("name", "")), String(pos.get("asks", ""))])
		var art := String(c.get("art", "")).strip_edges()
		if pictured:
			lines.append("The viewer is looking at its picture, held up to the camera: it is the painting above%s, so anything you say about it must be what is actually painted there. Remarking on the art is OPTIONAL, and usually you will not: real readers mostly name the card and carry their train of thought on through several cards, stopping on a detail of a picture only now and then, when it serves what they are saying. If you have already remarked on a card's art in what you have said so far, do not do it again here." % (", upside down, because it came up reversed" if bool(c.get("reversed", false)) else ""))
		elif not art.is_empty():
			lines.append("The viewer is looking at its picture, held up to the camera. For you, not to recite - this deck paints it as: %s. Mention it only if it serves what you are saying." % art)
		var b: Dictionary = c.get("booklet", {}) if c.get("booklet") is Dictionary else {}
		if not b.is_empty():
			lines.append("The deck's booklet entry for it is on screen beside the card - the viewer can read it. Refer to it or argue with it if you like, but never read it out:")
			lines.append("<booklet>")
			lines.append("Keywords: %s" % ", ".join(strings(b.get("keywords", []))))
			lines.append(String(b.get("reversed" if bool(c.get("reversed", false)) and b.has("reversed") else "upright", "")))
			lines.append("</booklet>")
		lines.append("Read this card for the viewer, in this position, carrying the story on from what you have already said - the reading builds as the cards come. You know only the cards on the table; never guess at what comes next.")
		if k < spread_size:
			lines.append("End as you reach for the next card: after your last word this card goes down and the next comes up, so lead into that and never say it has happened - the next passage opens on the new card.")
		else:
			lines.append("It is the last card. End as you go to lay it down beside the others: it goes down after your last word, so never say it is down - the close opens on the whole spread.")
	lines.append("")
	lines.append(MOVES)
	lines.append("")
	lines.append("Write %d to %d words." % [int(lo_hi[0]), int(lo_hi[1])])
	lines.append("")
	lines.append(SPOKEN_RULES)
	return {"system": show_context(title, brief), "prompt": "\n".join(lines)}


## A list a writer was asked for, as strings: a JSON array, or one comma-separated string -
## writers return both.
static func strings(v: Variant) -> PackedStringArray:
	var out := PackedStringArray()
	var items: Array = v if v is Array else (Array((v as String).split(",")) if v is String else [])
	for x in items:
		var t := str(x).strip_edges()
		if not t.is_empty():
			out.append(t)
	return out


static func _card_line(c: Dictionary) -> String:
	return "%s%s" % [String(c.get("name", "")), ", reversed (upside down)" if bool(c.get("reversed", false)) else ""]


# --- the painter -----------------------------------------------------------------------------

## Every picture's prompt opens the same way: the painter is an agent, told in words to make one
## image and put it at an exact path (see [ImageGen.Codex]).
static func _paint_head(target: String) -> String:
	return ("Use your built-in image generation tool to create exactly ONE image, then save it as a "
		+ "PNG at this exact path: %s\nDo not create or modify any other file. Reply with the saved path only." % target)


static func _deck_line(look: Dictionary) -> String:
	return "STYLE (the whole deck is painted in this one hand): %s Palette: %s." % [
		String(look.get("deck_style", "")), ", ".join(PackedStringArray(look.get("palette", [])))]


## A card's face: its illustration only - the engine prints the frame and the name around it.
static func card_image(look: Dictionary, card: Dictionary, art: String, target: String,
		has_back: bool, chain: int) -> String:
	var lines := PackedStringArray([_paint_head(target), ""])
	lines.append("THE PICTURE: the illustration for %s in the tarot deck \"%s\": %s" % [
		String(card.get("name", "")), String(look.get("deck_name", "")), art.strip_edges()])
	lines.append("")
	lines.append(_deck_line(look))
	lines.append("FORMAT: PORTRAIT 2:3 (1024x1536). The card's picture only, composed for a tall card and running to every edge: no border, no frame, no panel, no title, no numbers, no letters or writing of any kind - the frame and the card's name are printed around it separately. Keep the important things away from the outer tenth on each side.")
	# The attachments arrive in this order - the back, then the earlier cards - so each group is
	# named by its place in it.
	if has_back:
		lines.append("THE FIRST ATTACHED IMAGE is the BACK of this deck: match its palette, materials and hand. Do NOT copy its pattern or its symmetry.")
	if chain > 0:
		var which := ("THE LAST %d ATTACHED IMAGE%s" if has_back else "THE %d ATTACHED IMAGE%s") \
			% [chain, "S ARE" if chain > 1 else " IS"]
		lines.append("%s %s from this same deck: match the rendering exactly - medium, palette, linework, texture, level of detail - so every card reads as one artist's. Do NOT reuse compositions or subjects; the content comes only from THE PICTURE above." % [
			which, "earlier cards" if chain > 1 else "an earlier card"])
	if chain > 0:
		lines.append("The earlier cards set the HAND, not the CAST: this card's people and creatures are its own, as the description gives them - a figure from an earlier card appears again only if the description asks for it.")
	lines.append("Every real animal named is drawn as that animal, with its true anatomy.")
	lines.append("ALWAYS: no text, no captions, no lettering, no numbers, no borders, no watermark, no signature.")
	return "\n".join(lines)


static func back_image(look: Dictionary, target: String) -> String:
	var lines := PackedStringArray([_paint_head(target), ""])
	lines.append("THE PICTURE: the BACK of the tarot deck \"%s\" - the side every card shows face down. %s" % [
		String(look.get("deck_name", "")), String(look.get("card_back", ""))])
	lines.append("It must be EXACTLY SYMMETRIC under a half turn (180 degrees), so a card lying upside down cannot be told from one the right way up.")
	lines.append("")
	lines.append(_deck_line(look))
	lines.append("FORMAT: PORTRAIT 2:3 (1024x1536). The design runs to every edge: no outer border or frame (the card's frame is printed separately), no text, no numbers, no watermark, no signature.")
	return "\n".join(lines)


static func surface_image(look: Dictionary, target: String) -> String:
	var lines := PackedStringArray([_paint_head(target), ""])
	lines.append("THE PICTURE: a photograph looking STRAIGHT DOWN at %s, laid flat on a reading table and filling the whole frame edge to edge. Even, soft light; true colors; the texture of the material sharp. Nothing on it - no cards, no objects, no hands, no text." % String(look.get("surface", "a reading cloth")))
	lines.append("Its colors belong to this palette: %s." % ", ".join(PackedStringArray(look.get("palette", []))))
	lines.append("FORMAT: SQUARE 1:1 (1024x1024). No border, no vignette, no watermark, no text.")
	return "\n".join(lines)


static func backdrop_image(look: Dictionary, target: String) -> String:
	var light: Dictionary = look.get("light", {}) if look.get("light") is Dictionary else {}
	var lines := PackedStringArray([_paint_head(target), ""])
	lines.append("THE PICTURE: the view from a chair at a tarot reader's table, looking across it at the room or the world beyond: %s. Lit by %s." % [
		String(look.get("setting", "a quiet room")), String(light.get("kind", "low lamplight"))])
	lines.append("Seated eye level; the horizon or the far wall sits in the lower third. Do NOT show the table, cards, hands or people - the table is placed in front of this picture separately, and the picture will be seen out of focus behind it, so broad shapes and light matter more than detail.")
	lines.append("Its colors belong to this palette: %s." % ", ".join(PackedStringArray(look.get("palette", []))))
	lines.append("FORMAT: LANDSCAPE 3:2 (1536x1024). No text, no watermark, no signature.")
	return "\n".join(lines)
