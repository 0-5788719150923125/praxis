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
## FOUR ROLES, and each is told only what its job needs:
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

## How many words each passage runs to - the length of a tarot reading video comes from these.
const WORDS := {"intro": [130, 190], "card": [110, 170], "close": [90, 140]}

## The shape every reader passage must keep, said once for every reader prompt.
const SPOKEN_RULES := """FORMAT - everything you write is spoken aloud by a voice over the video, word for word:
- Spoken words only. No stage directions, no headings, no lists, no emoji, no markdown, no quotation marks around the passage, no sound effects in brackets.
- You may put single asterisks around ONE word you lean on - *this* - and only now and then.
- Write numbers, symbols and abbreviations the way they are said ("eleven eleven", "twenty percent", "okay").
- Plain paragraphs. Short sentences land; vary the rhythm.
- The viewer sees the table, not you. Never describe your own hands or face."""


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
	return {
		"place": "%.1f°%s %.1f°%s" % [absf(lat), "N" if lat >= 0.0 else "S", absf(lon),
			"E" if lon >= 0.0 else "W"],
		"year": rng.randi_range(-2400, 2150),
		"hue": rng.randi_range(0, 359),
		"hour": rng.randi_range(0, 23),
		"direction": rng.randi_range(1, 12),
	}


## The registry keys a look may name, for the producer to choose from: the faces the card titles
## are set in and the things on the table. Passed in by the caller so this file names no asset.
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
	lines.append("THE LOOK is a tarot deck that has never existed, and the table it is read on. Make it specific enough that an illustrator could paint every card in one consistent hand: medium, era and influences, linework, texture, palette, how figures are drawn. Then the card back (a design that reads the same when the card is turned upside down - exactly symmetric under a half turn), how much of the deck is printed in metallic foil (`foil`, 0 for matte ink to 1 for gold leaf everywhere), the surface the cards lie on (seen from directly above), the place the table stands in (seen past the far edge of the table, out of focus), the light, how many lit candles stand on the table, and the OBJECTS on it: two to four things that belong to this setting and this episode - the reader's own clutter, a tool of the trade, something the episode is about - each described for a photographer (what it is, its material, its state). Never cards, a deck, a cloth, a person, or anything with writing on it. The card's frame, its name and its numeral are printed by the deck itself, so the illustrations carry no lettering.")
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
  "reader_mood": "how the reader is today, and any running bit for this episode - a bit the reader SAYS, never a thing done or an object shown (the viewer sees only the cloth, the deck, the cards, the candles and the objects you choose), and never a way of speaking (the show's voice is fixed: no whispering, accents or singing)",
  "spread": {"name": "the spread's name", "positions": [{"name": "position name", "asks": "what it asks"}]},
  "look": {
    "deck_name": "the deck's name",
    "deck_style": "a paragraph an illustrator paints every card from",
    "palette": ["#rrggbb", "four to six colors"],
    "card_back": "the back design, symmetric under a half turn",
    "frame": {"style": "one of: %s", "stock": "#rrggbb card stock", "ink": "#rrggbb border and lettering", "accent": "#rrggbb"},
    "title_face": "one of: %s",
    "foil": 0.6,
    "surface": "the cloth or tabletop the cards lie on, seen from above",
    "setting": "the place beyond the table",
    "light": {"kind": "what lights the table", "color": "#rrggbb", "warmth": "warm or cool"},
    "candles": 2,
    "objects": [{"what": "one object, for a photographer: what it is, its material, its state", "size": "small, medium or large"}]
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
	lines.append("The deck: \"%s\". On the table: %s. Also on it, and the only other things the viewer can see: %s." % [
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
		lines.append("NOW: the video opens. You are shuffling the deck while you talk. Welcome the viewer to \"%s\", set up this episode and who it is for, and keep the patter going while the cards are mixed. You do not know any card yet - none has been drawn - so name none and predict none. End on the moment you stop shuffling to pull the first card." % title)
	elif step == "close":
		lo_hi = WORDS["close"]
		lines.append("NOW: all %d cards are face up on the table. Pull the reading together - the whole spread, in the light of everything you have said - and close the episode the way the brief says the show closes." % spread_size)
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
			lines.append("End on the moment you reach for the next card.")
		else:
			lines.append("It is the last card. End on the moment you lay it down beside the others.")
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


## ONE OBJECT ON THE TABLE, painted alone to be cut out and stood there: seen from the reader's
## chair (the table's camera looks down at about forty degrees), lit as the table is, on nothing.
static func object_image(look: Dictionary, object: Dictionary, target: String) -> String:
	var light: Dictionary = look.get("light", {}) if look.get("light") is Dictionary else {}
	var lines := PackedStringArray([_paint_head(target), ""])
	lines.append("THE PICTURE: a photograph of ONE object, alone: %s. It stands on a tarot reader's table in %s, lit by %s." % [
		String(object.get("what", "")), String(look.get("setting", "a quiet room")), String(light.get("kind", "low lamplight"))])
	lines.append("THE VIEW: from a chair at that table, looking down at the object at about 40 degrees - the way it looks standing on a tabletop just in front of you. Its base at the bottom of the picture.")
	lines.append("NOTHING ELSE: no table, no floor, no shadow, no hands, no other objects, no text. The object is cut out of this picture and stood on the table separately.")
	lines.append("BACKGROUND: transparent (a PNG with an alpha channel) if your image tool can make one; otherwise a flat, evenly lit, pure white background (#FFFFFF) - no gradient, no texture, no vignette.")
	lines.append("FRAMING: the whole object in frame and centered, with a margin of empty background all round.")
	lines.append("Its colors sit with this palette: %s." % ", ".join(PackedStringArray(look.get("palette", []))))
	lines.append("FORMAT: SQUARE 1:1 (1024x1024). No watermark, no signature.")
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
