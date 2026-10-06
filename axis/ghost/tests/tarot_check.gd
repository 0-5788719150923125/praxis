extends SceneTree

## The tarot mode's gate: everything that can be held to a rule without a writer, a painter or a
## renderer.
##
##   godot --headless --path . --script res://tests/tarot_check.gd
##
## - THE DECK: 78 cards, the Rider-Waite-Smith order, numerals, a meaning for every card.
## - THE SHUFFLE is a pure function of the seed.
## - ONE WALK, TWO READERS: the voice's text and the table's actions come from the same parse, and
##   the rest the voice takes for each action is the one the table performs it in.
## - THE SCHEDULE puts every action in the rest the voice left for it.
## - NO CHEATING: the reader's prompt for card K names cards 1..K and NO card after it, on an
##   episode written to disk the way the producer writes one. Two-sided: the same check run on a
##   prompt handed every card fails.
## - A REDO deletes the step and exactly what was made from it.
## - The helpers agents' replies go through: lenient JSON, spoken-text cleanup, look sanitizing.
## - THE QUEUE'S LANES run one job at a time, in order, and a rerun starts from a cleared folder.
## - REPLIES OF THE WRONG SHAPE are reshaped as they land; a step that writes nothing fails.
## - A SCRUB into a reading that repeats itself finds the repeat nearest where it was asked for.
## - THE SPREAD lies clear of the deck.
## - DELETING AN EPISODE takes its whole folder and nothing else.
## - A PART ASKED FOR IS THE PART MADE: a redo makes that step (and the free steps that follow it),
##   never the whole episode - that is Generate's.
## - THE READER SEES THE PAINTING: a card's passage waits for its picture and is sent it (upside
##   down when reversed), the close is sent them all, and nothing is sent a picture not yet drawn.
## - THE TABLE is its own step, set from the plan while looking at the cloth: a new cloth keeps it,
##   a new plan takes it, no card reaches the set dresser, and its prompt names every shape,
##   material, ornament and zone the builder knows, with the headroom of each zone.
## - WHAT IS BUILT is safe whatever was written: junk is dropped or clamped, flames are capped, a
##   thing stands on y = 0 with a foot to go round, and a candle's flame is lit at its wax's top.
## - CANDLES of every form: wicks set round a top or placed by hand, a candelabra written once as a
##   GROUP copied round, groups nested only so deep, extruded outlines standing sound - and a table
##   lights no more things, and no thing more flames, than it may.
## - STONES: scattered copies never touch, heaped ones pile up without passing through each other
##   or the floor, a list of materials goes to the copies in turn, a geode keeps its crystals
##   inside it, and a play of light is kept only when the shader knows it.
## - THE CARD STOCK is the producer's choice, shown earlier decks' stocks; the card's name reads on
##   any stock, and the booklet is printed in the card's colors.
## - WHAT THE SHOW HAS ALREADY MADE reaches each agent in the part it decides (the producer names the
##   show's habits), never the episode being made, and never a card name through a reader's record.

var _fails := 0


func _init() -> void:
	# EVERY CHECK MUST REACH ITS END: a script error inside one stops it part way and returns
	# nothing, and it used to leave a gate that had checked half of something reading ALL OK
	for check in [_deck, _shuffle, _script, _schedule, _no_cheating, _redo, _helpers, _landing,
			_lanes, _rerun_clears, _scrub_near, _clear_of_deck, _trash_episode, _pictures, _only_what_was_asked, _no_objects, _moves_after_words,
			_table_step, _things_built, _stones, _card_stock, _candles, _room_prompt, _archive]:
		_ok((check as Callable).call() == true, "%s stopped part way (a script error - see above)"
			% (check as Callable).get_method())
	print("tarot_check: %s (%d failure%s)" % ["ALL OK" if _fails == 0 else "FAILED", _fails,
		"" if _fails == 1 else "s"])
	quit(1 if _fails > 0 else 0)


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails += 1
		print("  FAIL: " + what)


func _deck() -> bool:
	var cards := TarotDeck.standard()
	_ok(cards.size() == 78, "the standard deck has %d cards, not 78" % cards.size())
	var keys := {}
	var by := {}
	for c in cards:
		keys[String((c as Dictionary)["key"])] = true
		by[String((c as Dictionary)["name"])] = c
	_ok(keys.size() == 78, "card keys are not unique")
	_ok(String((cards[8] as Dictionary)["name"]) == "Strength", "major 8 is not Strength (RWS order)")
	_ok(String((cards[11] as Dictionary)["name"]) == "Justice", "major 11 is not Justice (RWS order)")
	_ok(by.has("King of Pentacles"), "there is no King of Pentacles")
	_ok(String((by["The Fool"] as Dictionary)["numeral"]) == "0", "the Fool's numeral is not 0")
	_ok(String((by["The Sun"] as Dictionary)["numeral"]) == "XIX", "the Sun's numeral is not XIX")
	_ok(String((by["Four of Cups"] as Dictionary)["numeral"]) == "IV", "a pip's numeral is not its rank")
	_ok(String((by["Queen of Swords"] as Dictionary)["numeral"]).is_empty(), "a court card carries a numeral")
	_ok(String((by["Ace of Wands"] as Dictionary)["numeral"]).is_empty(), "an Ace carries a numeral")
	var missing := 0
	for c in cards:
		if String((c as Dictionary)["meaning"]).strip_edges().is_empty():
			missing += 1
	_ok(missing == 0, "%d standard cards have no meaning from the corpus" % missing)
	# THE STANDARD DECK WRITTEN OUT reads back as itself
	var back := TarotDeck.parse(TarotDeck.standard_section())
	var same := back.size() == 78
	for i in mini(back.size(), 78):
		for k in ["name", "numeral", "group", "meaning"]:
			same = same and (back[i] as Dictionary)[k] == (cards[i] as Dictionary)[k]
	_ok(same, "the standard deck written out does not read back as itself")
	# AN ALTERNATIVE DECK, defined in a show's brief
	var body := "# A show\n\nSome brief.\n\n## The Cards\n\nRead these upside down too.\n\n### The Feed\n" \
		+ "- 1. The Algorithm: what you were shown, and why it was you\n" \
		+ "- **The Ex** - still there, still not texting\n" \
		+ "- The Pinned Comment \u2014 one in ten thousand\n" \
		+ "  (and nobody pins the other nine thousand)\n\n### The Bed\n" \
		+ "- 12) The Snooze: nine more minutes, forever\n" \
		+ "- The Ex: a second Ex, to test names\n" \
		+ "- Untitled\n\n## After\n\n- not a card\n"
	var own := TarotDeck.parse(body)
	_ok(own.size() == 6, "an alternative deck of 6 cards parsed as %d" % own.size())
	if own.size() == 6:
		_ok(own[0]["name"] == "The Algorithm" and own[0]["numeral"] == "1" and own[0]["group"] == "The Feed",
			"a numbered card in a group parsed as %s" % str(own[0]))
		_ok(own[1]["name"] == "The Ex" and String(own[1]["meaning"]).begins_with("still there"),
			"a bold name with a spaced dash parsed as %s" % str(own[1]))
		_ok(String(own[2]["meaning"]).contains("nine thousand"), "an indented line did not continue its card")
		_ok(own[3]["numeral"] == "12" and own[3]["group"] == "The Bed", "a `12)` numeral or a second group was lost")
		_ok(own[4]["key"] != own[1]["key"], "two cards with one name share a key")
		_ok(own[5]["name"] == "Untitled" and String(own[5]["meaning"]).is_empty(), "a card with no meaning was lost")
	# A NESTED BULLET SAYS MORE ABOUT ITS CARD; it is not a card
	var nested := "## Cards\n\n- The Moon: dreams\n  - Reversed: clarity at last\n- The Sun: plain day\n"
	var nd := TarotDeck.parse(nested)
	_ok(nd.size() == 2 and String(nd[0]["meaning"]).contains("Reversed: clarity at last"),
		"a nested bullet became a card: %s" % str(nd))
	_ok(TarotDeck.strip(nested).contains("This deck has 2 cards") and not TarotDeck.strip(nested).contains("clarity"),
		"stripping a deck with nested bullets miscounted it or left them in")
	# A SECTION ABOUT THE DECK'S LOOK is not a deck
	var looks := "## The Deck\n\n- Style: woodcut\n- Colors: ink and rust\n"
	_ok(TarotDeck.parse(looks).is_empty() and TarotDeck.strip(looks) == looks,
		"a '## The Deck' section of bullets was read as a deck of its bullets")
	_ok(TarotDeck.of(body).size() == 6 and TarotDeck.of("No cards here.").size() == 78,
		"a show without a Cards section does not read the standard deck")
	var stripped := TarotDeck.strip(body)
	_ok(not stripped.contains("The Algorithm") and not stripped.contains("nine thousand"),
		"the brief handed to agents still lists the cards")
	_ok(stripped.contains("Read these upside down too.") and stripped.contains("This deck has 6 cards")
		and stripped.contains("## After") and stripped.contains("- not a card"),
		"stripping the cards took the rest of the brief with it")
	print("tarot_check: deck - standard %d, %d without meanings; an alternative deck of %d" % [cards.size(), missing, own.size()])
	return true


func _shuffle() -> bool:
	var deck := TarotDeck.standard()
	var a := TarotDeck.shuffled(deck, 7, true)
	var b := TarotDeck.shuffled(deck, 7, true)
	var c := TarotDeck.shuffled(deck, 8, true)
	_ok(a == b, "the same seed shuffled differently")
	_ok(a != c, "two seeds shuffled the same")
	_ok(a.size() == 78, "a shuffle lost cards")
	var rev := 0
	for x in a:
		rev += 1 if bool((x as Dictionary)["reversed"]) else 0
	_ok(rev > 10 and rev < 40, "%d of 78 reversed - far from a third" % rev)
	var none := 0
	for x in TarotDeck.shuffled(deck, 7, false):
		none += 1 if bool((x as Dictionary)["reversed"]) else 0
	_ok(none == 0, "reversals came up with reversals off")
	_ok(not (deck[0] as Dictionary).has("reversed"), "a shuffle wrote into the deck it was given")
	var small := [{"key": "a", "name": "A"}, {"key": "b", "name": "B"}, {"key": "c", "name": "C"}]
	var names := {}
	for x in TarotDeck.shuffled(small, 3, true):
		names[String((x as Dictionary)["name"])] = true
	_ok(names.size() == 3, "an alternative deck lost cards in the shuffle")
	var seeds := {}
	for i in 20:
		var t := TarotDeck.true_seed()
		_ok(t >= 1 and t <= 999999, "a true seed out of range: %d" % t)
		seeds[t] = true
	_ok(seeds.size() >= 18, "twenty true seeds were %d distinct" % seeds.size())
	var tiny := TarotProducer.new(TarotEpisode.open("check-show", 1), {"deck": small, "cards": [5, 6]})
	_ok(tiny._spread_n() == 3, "a spread of %d was drawn from a deck of 3" % tiny._spread_n())
	_ok(TarotProducer.spread_size(3, 3, 6) == TarotProducer.spread_size(3, 3, 6), "the spread size is not a function of the seed")
	var sizes := {}
	for s in 60:
		var n := TarotProducer.spread_size(s, 3, 6)
		_ok(n >= 3 and n <= 6, "spread size %d outside 3-6" % n)
		sizes[n] = true
	_ok(sizes.size() == 4, "spread sizes do not cover the range: %s" % str(sizes.keys()))
	_ok(TarotPrompts.dice(5) == TarotPrompts.dice(5) and TarotPrompts.dice(5) != TarotPrompts.dice(6),
		"the dice are not a function of the seed")
	return true


func _script() -> bool:
	var passages := [
		{"kind": "shuffle", "card": 0, "text": "Hello, my loves. Welcome back.\n\nLet's shuffle."},
		{"kind": "jumper", "card": 1, "text": "Oh, a jumper! The Tower."},
		{"kind": "draw", "card": 2, "text": "The *Star*, reversed. Spirit says: drink water."},
		{"kind": "spread", "card": 0, "text": "Like and subscribe. It won't change your life."},
	]
	var body := TarotScript.compose(passages)
	_ok(TarotScript.is_tarot(body), "a composed reading is not recognized as one")
	_ok(not TarotScript.is_tarot("Once upon a time."), "a chapter is taken for a tarot reading")
	var p := TarotScript.parse(body)
	_ok((p["passages"] as Array).size() == 4, "the reading has %d passages, not 4" % (p["passages"] as Array).size())
	_ok(int(p["cards"]) == 2, "the reading draws %d cards, not 2" % int(p["cards"]))
	var actions: Array = p["actions"]
	_ok(actions.size() == 4, "%d actions, not 4" % actions.size())
	var kinds := []
	for a in actions:
		kinds.append(String((a as Dictionary)["kind"]))
	_ok(kinds == ["shuffle", "jumper", "draw", "spread"], "actions out of order: %s" % str(kinds))
	# THE RESTS AGREE: what the voice is told to wait, and what the table performs in
	var speak := String(p["speakable"])
	var holds := []
	for m in Manuscript._rx(TabletScript.HOLD).search_all(speak):
		holds.append(float(m.get_string(1)))
	_ok(holds.size() == 4, "the voice is given %d rests for 4 actions" % holds.size())
	var want := [TarotScript.SHUFFLE_LEAD, TarotScript.JUMP, TarotScript.LAY + TarotScript.DRAW,
		TarotScript.LAY + TarotScript.SETTLE]
	for i in mini(holds.size(), want.size()):
		_ok(absf(float(holds[i]) - float(want[i])) < 0.011, "rest %d is %.2fs, the table takes %.2fs" % [i, holds[i], want[i]])
	# EVERY WORD SPOKEN IS A WORD THE TABLE FOLLOWS, in order
	var words := PackedStringArray()
	for q in passages:
		for w in String((q as Dictionary)["text"]).split(" ", false):
			for piece in String(w).split("\n", false):
				var n := TabletScript.norm(piece)
				if not n.is_empty():
					words.append(n)
	_ok(words == p["spoken"], "the table's words are not the voice's words")
	# seven words of intro, five of the jumper's passage
	_ok(int((actions[2] as Dictionary)["after"]) == 12, "the draw is anchored after word %d, not 12" % int((actions[2] as Dictionary)["after"]))
	_ok(not speak.contains("tarot:"), "a tarot mark survives into what the voice reads")
	# THE READER'S DELIVERY MARKS reach the voice, and no comment is ever a word the table follows
	var marked := TarotScript.parse(TarotScript.compose([{"kind": "shuffle",
		"text": "Hello.\n\n<!-- delivery: excited -->\nOh wow. <!-- hesitation: 1.5 --> Look. <!-- a note --> Done."}]))
	var said := String(marked["speakable"])
	_ok(said.contains("<!-- delivery: excited -->") and said.contains("<!-- hesitation: 1.5 -->") and not said.contains("a note"),
		"the voice is not handed the reader's delivery and hesitation, or is handed a note: %s" % said)
	_ok(not (marked["spoken"] as PackedStringArray).has("delivery") and not (marked["spoken"] as PackedStringArray).has("excited")
		and not (marked["spoken"] as PackedStringArray).has("hesitation"), "a mark is counted as a word the table follows")
	# THE FIRST CARD WAITS FOR THE DECK: shuffled in the middle, it is pushed to its side first
	var plain := TarotScript.parse(TarotScript.compose([{"kind": "shuffle", "text": "one two"},
		{"kind": "draw", "card": 1, "text": "three four"}, {"kind": "draw", "card": 2, "text": "five six"}]))
	var dur: Array = []
	for a in plain["actions"]:
		dur.append(float((a as Dictionary)["dur"]))
	_ok(dur.size() == 3 and absf(float(dur[1]) - (TarotScript.PUSH + TarotScript.DRAW)) < 0.01
		and absf(float(dur[2]) - (TarotScript.LAY + TarotScript.DRAW)) < 0.01,
		"the first draw does not wait for the deck to be pushed aside, or a later one does: %s" % str(dur))
	_ok(TarotScript.speakable("Plain text.") == "Plain text.", "a non-tarot text is rewritten")
	return true


func _schedule() -> bool:
	var body := TarotScript.compose([
		{"kind": "shuffle", "text": "one two three four"},
		{"kind": "draw", "card": 1, "text": "five six seven"},
		{"kind": "spread", "text": "eight nine"}])
	var p := TarotScript.parse(body)
	var f := ReadingFollower.new()
	f.reset(p["spoken"])
	# the voice: words 0.3 s apart from t=5, with each action's rest left before its word
	var words: Array = []
	var t := 5.0
	var rest := {}
	for a in p["actions"]:
		rest[int((a as Dictionary)["after"])] = float((a as Dictionary)["dur"]) + 0.5
	for i in (p["spoken"] as PackedStringArray).size():
		t += float(rest.get(i, 0.0))
		words.append({"text": (p["spoken"] as PackedStringArray)[i], "t0": t, "t1": t + 0.25})
		t += 0.3
	f.extend(words)
	_ok(f.known_last() == words.size() - 1, "the follower matched %d of %d words" % [f.known_last() + 1, words.size()])
	var sched := f.place(p["actions"], 3.0, 0.25, 0.2)
	_ok(sched.size() == 3, "%d actions placed, not 3" % sched.size())
	for e in sched:
		var a: Dictionary = e["a"]
		var n := int(a["after"])
		var end := float(e["t0"]) + float(a["dur"]) * float(e["s"])
		if n < words.size():
			_ok(end <= float((words[n] as Dictionary)["t0"]) - 0.19, "%s ends %.2fs, after its next word at %.2fs" % [a["kind"], end, (words[n] as Dictionary)["t0"]])
		if n > 0:
			_ok(float(e["t0"]) >= float((words[n - 1] as Dictionary)["t1"]), "%s starts before the word ahead of it ends" % a["kind"])
	# A VOICE THAT SPELLS A WORD ITS OWN WAY costs that word, never the pointer
	var g := ReadingFollower.new()
	g.reset(PackedStringArray(["the", "7", "cards", "fell"]))
	g.extend([{"text": "the", "t0": 0.0, "t1": 0.2}, {"text": "seven", "t0": 0.3, "t1": 0.5},
		{"text": "cards", "t0": 0.6, "t1": 0.8}, {"text": "fell", "t0": 0.9, "t1": 1.0}])
	# float32 storage: compared approximately
	_ok(is_equal_approx(g.st0[2], 0.6) and is_equal_approx(g.st0[3], 0.9) and g.st0[1] == ReadingFollower.NO_TIME,
		"the follower lost its place over a number read out")
	return true


## A fake episode on disk, made the way the producer makes one, under a test root.
func _episode(n: int) -> TarotEpisode:
	TarotEpisode.root = "user://tarot_check"
	var ep := TarotEpisode.open("check-show", 4242)
	if DirAccess.dir_exists_absolute(ep.dir):
		for f in DirAccess.get_files_at(ep.dir):
			DirAccess.remove_absolute(ep.dir.path_join(f))
	var positions: Array = []
	for i in n:
		positions.append({"name": "Position %d" % (i + 1), "asks": "what position %d asks" % (i + 1)})
	ep.write_json("plan", {"episode_title": "A Message Meant To Find You", "audience": "everyone",
		"topic": "waiting", "premise": "the premise", "reader_mood": "calm",
		"spread": {"name": "The Spread", "positions": positions},
		"look": TarotTable.sanitize_look({"deck_name": "The Test Deck", "props": ["candle"]})})
	var cards: Array = []
	var deck := TarotDeck.shuffled(TarotDeck.standard(), ep.seed, true)
	for i in n:
		var card := (deck[i] as Dictionary).duplicate()
		card["jumper"] = false
		cards.append(card)
		ep.write_json("design:%d" % (i + 1), {"art": "art for card %d" % (i + 1), "booklet": {
			"keywords": ["kw%d" % (i + 1)], "upright": "booklet text number %d upright" % (i + 1),
			"reversed": "booklet text number %d reversed" % (i + 1)}})
	ep.write_json("draw", {"seed": ep.seed, "cards": cards})
	ep.write_text("say:intro", "Hello, my loves.")
	for i in n:
		ep.write_text("say:%d" % (i + 1), "Passage for card %d." % (i + 1))
	return ep


func _no_cheating() -> bool:
	var n := 5
	var ep := _episode(n)
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief."})
	var names: Array = []
	for k in range(1, n + 1):
		names.append(String(prod._card(k)["name"]))
	var plan: Dictionary = ep.read_json("plan")
	var leaks := 0
	for k in range(0, n + 2):
		var step := "intro" if k == 0 else ("close" if k == n + 1 else str(k))
		var upto := 0 if k == 0 else (n if k == n + 1 else k)
		var said: Array = [] if k == 0 else prod._said(mini(k - 1, n) if k <= n else n)
		var p := TarotPrompts.reader("Test Tarot", "A brief.", plan, step, said, prod._drawn(upto), n)
		var text := String(p["system"]) + "\n" + String(p["prompt"])
		for j in range(1, n + 1):
			var named := text.contains(String(names[j - 1]))
			var booklet := text.contains("booklet text number %d" % j)
			var art := text.contains("art for card %d" % j)
			var means := text.contains(String(prod._card(j)["meaning"]))
			if j > upto and (named or booklet or art or means):
				leaks += 1
				print("  card %d (%s) is in the prompt for %s" % [j, names[j - 1], step])
			if j == upto and step.is_valid_int() and not named:
				_ok(false, "the prompt for card %d does not name it" % j)
		if step == "close":
			for j in range(1, n + 1):
				_ok(text.contains(String(names[j - 1])), "the close does not know card %d" % j)
	_ok(leaks == 0, "%d later card(s) reached a reader prompt" % leaks)
	# TWO-SIDED: the same check, run on a prompt handed every card, must see them
	var cheat := TarotPrompts.reader("Test Tarot", "A brief.", plan, "1", prod._said(0), prod._drawn(n), n)
	var seen := 0
	for j in range(2, n + 1):
		seen += 1 if String(cheat["prompt"]).contains(String(names[j - 1])) else 0
	_ok(seen == n - 1, "the leak check is blind: a prompt given every card shows only %d of %d later ones" % [seen, n - 1])
	# AND THROUGH THE PRODUCER ITSELF: the prompt it would actually send, for every passage
	var prod_leaks := 0
	for k in range(0, n + 2):
		var step := "intro" if k == 0 else ("close" if k == n + 1 else str(k))
		var upto := 0 if k == 0 else (n if k == n + 1 else k)
		var p := prod.say_prompt(step)
		var text := String(p.get("system", "")) + "\n" + String(p.get("prompt", ""))
		_ok(not text.strip_edges().is_empty(), "the producer built no prompt for %s" % step)
		for j in range(upto + 1, n + 1):
			if text.contains(String(names[j - 1])) or text.contains("booklet text number %d" % j) \
					or text.contains("art for card %d" % j):
				prod_leaks += 1
				print("  card %d (%s) is in the producer's prompt for %s" % [j, names[j - 1], step])
		if step.is_valid_int():
			_ok(text.contains(String(names[upto - 1])), "the producer's prompt for card %d does not name it" % upto)
		# THE PAINTINGS SENT: card K's own for card K, all of them for the close, none before
		var sent: Array = []
		for im in p.get("images", []):
			sent.append(String((im as Dictionary)["path"]).get_file())
		var want: Array = []
		if step == "close":
			for j in range(1, n + 1):
				want.append("card_%d.png" % j)
		elif step.is_valid_int():
			want = ["card_%d.png" % upto]
		_ok(sent == want, "the reader's prompt for %s is sent the paintings %s, not %s" % [step, str(sent), str(want)])
		_ok(want.is_empty() == not (text.contains("painting above") or text.contains("paintings above")),
			"the reader's prompt for %s %s the paintings it is sent" % [step, "is not told about" if not want.is_empty() else "speaks of"])
	_ok(prod_leaks == 0, "%d later card(s) reached a prompt the producer builds" % prod_leaks)
	print("tarot_check: no cheating - %d reader prompts, %d leaks (%d through the producer)" % [n + 2, leaks, prod_leaks])
	return true


func _redo() -> bool:
	var n := 4
	var ep := _episode(n)
	for s in ["image:back", "image:surface", "image:backdrop"]:
		ep.write_text(s, "png")
	for k in range(1, n + 1):
		ep.write_text("image:card:%d" % k, "png")
	ep.write_text("say:close", "Bye.")
	ep.write_text("script", "<!-- tarot: shuffle -->")
	ep.write_json("table", {"things": []})
	_ok(ep.complete(), "a fully written episode is not complete")
	ep.invalidate("design:2")
	for s in ["design:2", "image:card:2", "say:2", "say:3", "say:4", "say:close", "script"]:
		_ok(not ep.has(s), "redoing design 2 kept %s" % s)
	for s in ["design:1", "design:3", "image:card:1", "image:card:3", "image:back", "say:1", "say:intro"]:
		_ok(ep.has(s), "redoing design 2 took %s with it" % s)
	ep.invalidate("image:back")
	_ok(not ep.has("image:back") and ep.has("image:card:3"), "redoing the back took a card's picture with it")
	ep.invalidate("say:intro")
	_ok(not ep.has("say:1") and ep.has("design:1"), "redoing the intro did not redo every passage after it")
	ep.invalidate("plan")
	_ok(not ep.has("draw") and not ep.has("design:1") and not ep.has("image:card:1"),
		"redoing the plan left the episode's cards")
	return true


func _helpers() -> bool:
	_ok(TextGen.extract_json("```json\n{\"a\": 1}\n```") is Dictionary, "a fenced JSON reply is not read")
	_ok(TextGen.extract_json("Sure! {\"a\": {\"b\": 2}} Hope that helps.") is Dictionary, "a JSON reply with words around it is not read")
	_ok(TextGen.extract_json("no json here") == null, "a reply with no JSON reads as JSON")
	var c := TarotProducer.clean_spoken("\"# Intro\n*shuffles the deck*\nHello, my loves \u2014 welcome.\n[pause]\n**Truly.**\"")
	_ok(not c.contains("#") and not c.contains("shuffles") and not c.contains("[") and not c.contains("\u2014"),
		"stage directions or markup survive the cleanup: %s" % c)
	_ok(c.contains("Hello, my loves - welcome.") and c.contains("*Truly.*"), "the cleanup lost words: %s" % c)
	var look := TarotTable.sanitize_look({"palette": ["red", "#zzzzzz"], "title_face": "Comic Sans",
		"props": ["candle", "laser", "candle"], "frame": {"style": "weird"}, "foil": 7})
	var objs := TarotTable.sanitize_look({"candles": 9, "objects": ["a brass astrolabe",
		{"what": "a cracked teacup", "size": "small"}]})
	_ok((look["palette"] as Array).size() >= 3, "a bad palette was kept")
	_ok(String(look["title_face"]) == "roman", "an unknown face was kept")
	_ok(int(look["candles"]) == 1 and not look.has("props"),
		"an older look's props did not become its candles: %s" % str(look))
	_ok(int(objs["candles"]) == TarotTable.MAX_CANDLES, "the candles were not clamped: %d" % int(objs["candles"]))
	_ok(not objs.has("objects"), "an older look's painted objects reached the table: %s" % str(objs))
	_ok(TarotTable.on_the_table(objs) == "four lit candles and the reader's own things",
		"what the reader is told is on the table: %s" % TarotTable.on_the_table(objs))
	_ok(String((look["frame"] as Dictionary)["style"]) == "line", "an unknown frame was kept")
	_ok(float(look["foil"]) == 1.0, "foil was not clamped")
	_ok(TarotEpisode.slug("Truthful Tarot!") == "truthful-tarot", "the show's slug is %s" % TarotEpisode.slug("Truthful Tarot!"))
	return true


## Lanes, on the queue itself, with no subprocess: two text jobs in one lane and one outside it.
func _lanes() -> bool:
	AgentJobs._queue = [
		{"id": "a", "kind": "text", "lane": "L"},
		{"id": "b", "kind": "text", "lane": "L"},
		{"id": "c", "kind": "text"}]
	AgentJobs._running = {}
	_ok(AgentJobs._next_startable() == 0, "the first job in a lane cannot start")
	AgentJobs._running = {"a": AgentJobs._queue.pop_at(0)}
	_ok(AgentJobs._next_startable() == 1, "a job ran beside another in its own lane")
	AgentJobs._running["c"] = AgentJobs._queue.pop_at(1)
	_ok(AgentJobs._next_startable() == -1, "a job started past the kind's limit or its lane")
	AgentJobs._running.erase("a")
	_ok(AgentJobs._next_startable() == 0, "a lane did not move on once its job ended")
	AgentJobs._queue = []
	AgentJobs._running = {}
	return true


## REPLIES OF THE WRONG SHAPE are reshaped as they land, and still build a reader's prompt; a
## local step that writes nothing is a failure on screen.
func _landing() -> bool:
	TarotEpisode.root = "user://tarot_check"
	var ep := TarotEpisode.open("check-show", 777)
	ep.invalidate("plan")
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief.", "cards": [3, 3]})
	var err := prod._land_plan(JSON.stringify({"episode_title": "T", "tags": "one, two",
		"spread": {"name": "S", "positions": ["Past", {"name": "Present", "asks": "now"}, 7]},
		"look": {"deck_name": "D"}}))
	_ok(err.is_empty(), "a plan of the wrong shape was refused: %s" % err)
	var plan: Dictionary = ep.read_json("plan") if ep.read_json("plan") is Dictionary else {}
	var pos: Array = ((plan.get("spread", {}) as Dictionary).get("positions", [])) as Array
	_ok(pos.size() == 3 and pos[0] is Dictionary and String((pos[0] as Dictionary)["name"]) == "Past"
		and String((pos[2] as Dictionary)["name"]) == "7", "positions were not reshaped: %s" % str(pos))
	_ok(plan.get("tags") is Array and plan.get("tags") == ["one", "two"], "tags as one string were not split: %s" % str(plan.get("tags")))
	var d_err := prod._land_design("design:1", JSON.stringify({"art": "a gate", "booklet": {
		"keywords": "love, union", "upright": "together"}}))
	var design: Variant = ep.read_json("design:1")
	_ok(d_err.is_empty() and design is Dictionary
		and ((design as Dictionary)["booklet"] as Dictionary)["keywords"] == ["love", "union"],
		"keywords as one string were not split: %s" % str(design))
	# the draw, then a reader's prompt from what landed - it used to fail inside the builder
	prod.spec["deck"] = TarotDeck.standard()
	prod._finish("draw", prod._make_draw())
	var p := prod.say_prompt("1")
	_ok(String(p.get("prompt", "")).contains("Past") and String(p.get("prompt", "")).contains("love, union"),
		"a reader's prompt could not be built from reshaped replies")
	# A STEP THAT WRITES NOTHING says so - two-sided: one that wrote its file does not
	var quiet := TarotProducer.new(ep, {})
	quiet._finish("script", "")
	_ok(not quiet.error_of("script").is_empty(), "a local step that wrote nothing was taken as done")
	quiet._finish("draw", "")
	_ok(quiet.error_of("draw").is_empty(), "a local step that wrote its file was taken as failed")
	# A DECK CUT BELOW THE SPREAD the plan was made for is an error, not an index past the end
	ep.invalidate("draw")
	var short := TarotProducer.new(ep, {"deck": [{"key": "a", "name": "A"}], "cards": [3, 3]})
	_ok(not short._make_draw().is_empty() and not ep.has("draw"), "a draw ran past the end of a small deck")
	ep.invalidate("plan")
	print("tarot_check: landing - reshaped plan, booklet and a reader's prompt from them")
	return true


## A RERUN STARTS FROM A CLEARED FOLDER: the last run's picture or reply would otherwise be taken
## for this run's when it ends without writing one.
func _rerun_clears() -> bool:
	var dir := ProjectSettings.globalize_path("user://tarot_check/jobs_rerun")
	DirAccess.make_dir_recursive_absolute(dir)
	for f in ["image.png", "reply.jsonl", "prompt.txt"]:
		TextGen.put(dir.path_join(f), "old")
	var err := AgentJobs.clear_outputs({"kind": "image", "dir": dir})
	_ok(err.is_empty() and not FileAccess.file_exists(AgentJobs.paint_target(dir)),
		"a painter's rerun starts beside the last run's picture")
	err = AgentJobs.clear_outputs({"kind": "text", "dir": dir})
	_ok(err.is_empty() and not FileAccess.file_exists(dir.path_join("reply.jsonl")),
		"a writer's rerun starts beside the last run's reply")
	_ok(FileAccess.file_exists(dir.path_join("prompt.txt")), "clearing a rerun took the prompt record with it")
	return true


## A SCRUB INTO A READING THAT REPEATS ITSELF: the same start words at two places, and the one
## nearest where the scrub landed wins - two-sided: with no hint, the first.
func _scrub_near() -> bool:
	var norms := PackedStringArray(["oh", "wow", "okay", "so", "the", "tower", "oh", "wow", "okay", "so", "the", "star"])
	var start := PackedStringArray(["oh", "wow", "okay", "so", "the"])
	_ok(TabletScript.find_run(norms, start) == 0, "with no hint the first match did not win")
	_ok(TabletScript.find_run(norms, start, 7) == 6, "a scrub near the repeat started at the first one")
	_ok(TabletScript.find_run(norms, start, 1) == 0, "a scrub near the first started at the repeat")
	# a better match still beats a nearer one
	var exact := PackedStringArray(["oh", "wow", "okay", "so", "the", "star"])
	_ok(TabletScript.find_run(norms, exact, 0) == 6, "a nearer, worse match beat an exact one")
	return true


## THE SPREAD LIES CLEAR OF THE DECK: two rows reaching into its corner step back, or aside, by
## the least that clears it; a spread nowhere near it is not moved.
func _clear_of_deck() -> bool:
	var card := Vector2(0.07, 0.12)
	var keep := Rect2(0.17 - 0.035 - 0.012, 0.04 - 0.06 - 0.012, 0.07 + 0.024, 0.12 + 0.024)
	var rows: Array = []
	for i in 10:
		var row := 0 if i < 5 else 1
		var k := i if row == 0 else i - 5
		rows.append({"pos": Vector3((float(k) - 2.0) * 0.085, 0.0, -0.165 + float(row) * 0.1344), "yaw": 0.02})
	var touching := 0
	for sl in rows:
		touching += 1 if TarotTable.footprint(sl["pos"], sl["yaw"], card).intersects(keep) else 0
	_ok(touching > 0, "the control does not reach the deck - the check below proves nothing")
	var off := TarotTable.clear_of(rows, card, keep)
	var still := 0
	for sl in rows:
		var moved: Vector3 = (sl["pos"] as Vector3) + Vector3(off.x, 0.0, off.y)
		still += 1 if TarotTable.footprint(moved, sl["yaw"], card).intersects(keep) else 0
	_ok(still == 0, "%d cards still lie in the deck after the spread moved by %s" % [still, str(off)])
	_ok(off.length() < 0.12, "the spread moved %.3f to clear the deck" % off.length())
	var far: Array = [{"pos": Vector3(-0.1, 0.0, -0.1), "yaw": 0.0}, {"pos": Vector3(0.0, 0.0, -0.1), "yaw": 0.0}]
	_ok(TarotTable.clear_of(far, card, keep) == Vector2.ZERO, "a spread clear of the deck was moved")
	# a card past the deck's far side rules out stepping aside (it would cross the deck)
	var past: Array = [{"pos": Vector3(0.14, 0.0, 0.03), "yaw": 0.0}, {"pos": Vector3(0.3, 0.0, 0.03), "yaw": 0.0}]
	var po := TarotTable.clear_of(past, card, keep)
	_ok(po.x == 0.0 and po.y < 0.0, "a spread was stepped aside through the deck: %s" % str(po))
	return true


## DELETING AN EPISODE takes its whole folder - jobs and all - out of the show's history, and
## leaves the show's other episodes as they were. The trash is swapped for a plain delete here,
## so the check never fills the author's trash.
func _trash_episode() -> bool:
	TarotEpisode.root = "user://tarot_check"
	var eps: Array = []
	for seed in [5, 6]:
		var ep := TarotEpisode.open("trash-show", seed)
		ep.write_json("plan", {"episode_title": "Episode %d" % seed})
		TextGen.put(ep.job_dir("plan").path_join("prompt.txt"), "a prompt")
		eps.append(ep)
	var seeds := func() -> Array:
		var out: Array = []
		for h in TarotEpisode.history("trash-show"):
			out.append(int((h as Dictionary)["seed"]))
		out.sort()
		return out
	_ok(seeds.call() == [5, 6], "the show does not list both episodes before a delete: %s" % str(seeds.call()))
	TarotEpisode.discard = _remove_tree
	var err := (eps[0] as TarotEpisode).trash()
	_ok(err.is_empty(), "deleting an episode failed: %s" % err)
	_ok(not DirAccess.dir_exists_absolute((eps[0] as TarotEpisode).dir), "a deleted episode's folder is still there")
	_ok(seeds.call() == [6], "after deleting #5 the show lists %s" % str(seeds.call()))
	_ok((eps[1] as TarotEpisode).has("plan"), "deleting one episode touched another")
	_ok(not (eps[0] as TarotEpisode).trash().is_empty(), "deleting an episode that is not there succeeded")
	(eps[1] as TarotEpisode).trash()
	TarotEpisode.discard = Callable()
	return true


func _remove_tree(path: String) -> int:
	for d in DirAccess.get_directories_at(path):
		_remove_tree(path.path_join(d))
	for f in DirAccess.get_files_at(path):
		DirAccess.remove_absolute(path.path_join(f))
	return DirAccess.remove_absolute(path)


## THE READER SEES THE PAINTING. The order: a card's passage waits for its picture, and painting a
## card again rewrites its passage and every one after it (they were written looking at it), but
## not the next card's picture. The prompt: told to talk about the painting, and given no
## designer's plan for it. The picture: no larger than the edge it is sent at, upside down for a
## reversed card - and a picture that cannot be read stops the run rather than going unseen.
func _pictures() -> bool:
	var n := 4
	var ep := _episode(n)
	for k in range(1, n + 1):
		ep.write_text("image:card:%d" % k, "png")
	ep.write_text("say:close", "Bye.")
	ep.write_text("script", "<!-- tarot: shuffle -->")
	_ok(ep.needs("say:2").has("image:card:2") and not ep.needs("say:intro").has("image:card:1"),
		"a card's passage does not wait for its painting: %s" % str(ep.needs("say:2")))
	ep.invalidate("image:card:3")
	for s in ["image:card:3", "say:3", "say:4", "say:close", "script"]:
		_ok(not ep.has(s), "painting card 3 again kept %s" % s)
	for s in ["image:card:4", "say:2", "image:card:2", "design:3"]:
		_ok(ep.has(s), "painting card 3 again took %s with it" % s)
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief."})
	var plan: Dictionary = ep.read_json("plan")
	var told := TarotPrompts.reader("T", "B", plan, "2", prod._said(1), prod._drawn(2), n, true)
	_ok(String(told["prompt"]).contains("painting above") and not String(told["prompt"]).contains("art for card 2"),
		"a pictured card's prompt still describes the designer's plan, or never mentions the painting")
	var blind := TarotPrompts.reader("T", "B", plan, "2", prod._said(1), prod._drawn(2), n, false)
	_ok(String(blind["prompt"]).contains("art for card 2"), "with no painting, the designer's plan is not given either")
	# the picture itself: a tall one, red over blue, sent at the edge, and turned over
	var img := Image.create(1200, 1800, false, Image.FORMAT_RGB8)
	img.fill_rect(Rect2i(0, 0, 1200, 900), Color(0.9, 0.1, 0.1))
	img.fill_rect(Rect2i(0, 900, 1200, 900), Color(0.1, 0.1, 0.9))
	var path := ProjectSettings.globalize_path("user://tarot_check/picture.png")
	img.save_png(path)
	for flip in [false, true]:
		var b := TextGen.image_block(path, flip)
		var raw := Marshalls.base64_to_raw(String(((b.get("source", {}) as Dictionary).get("data", ""))))
		var back := Image.new()
		_ok(back.load_jpg_from_buffer(raw) == OK and maxi(back.get_width(), back.get_height()) == TextGen.PICTURE_EDGE,
			"a picture is not sent as a JPEG at %d px on its long edge" % TextGen.PICTURE_EDGE)
		var top := back.get_pixel(back.get_width() / 2, 10)
		_ok((top.r > top.b) != flip, "a %s picture arrives %s" % ["reversed" if flip else "upright",
			"right way up" if top.r > top.b else "upside down"])
	var m := TextGen.Claude.compose({"prompt": "Read it.", "images": [{"path": path, "label": "Card 1:", "flip": true}]})
	var content: Array = (((JSON.parse_string(String(m["input"])) as Dictionary)["message"] as Dictionary)["content"]) as Array
	_ok(content.size() == 3 and String((content[0] as Dictionary)["text"]) == "Card 1:"
		and String((content[1] as Dictionary)["type"]) == "image" and String((content[2] as Dictionary)["text"]) == "Read it.",
		"the message is not the label, the picture, then the prompt")
	_ok(String(m["shown"]).begins_with("[picture: Card 1 (turned over, as it lies) - picture.png]"),
		"the prompt record does not list the picture sent: %s" % String(m["shown"]).substr(0, 80))
	var gone := TextGen.Claude.compose({"prompt": "x", "images": [{"path": path + ".missing"}]})
	_ok(not String(gone["error"]).is_empty(), "a picture that cannot be read did not stop the run")
	# CODEX, the other writer: told to work from the message alone, the pictures attached as files
	# in order and named in the message, read-only
	var job := {"dir": "/tmp/job", "system": "The system.", "prompt": "Read it.", "tier": "fast",
		"images": [{"path": path, "label": "Card 1:", "flip": true}, {"path": path, "label": "Card 2:"}]}
	var cm := TextGen.Codex.compose(job)
	var cp := String(cm["prompt"])
	_ok(cp.begins_with(TextGen.Codex.ONLY_THIS) and cp.contains("The system.") and cp.ends_with("Read it.")
		and cp.contains("Attached picture 1 - Card 1") and cp.contains("Attached picture 2 - Card 2"),
		"the Codex writer's message is not the instruction, the system prompt, the pictures, then the prompt")
	var argv := TextGen.Codex.argv(job, cm["pictures"])
	var at := argv.find("--image=/tmp/job/picture_1.jpg")
	_ok(at >= 0 and argv.find("--image=/tmp/job/picture_2.jpg") == at + 1 and argv.find("--") > at + 1
		and argv[argv.find("-s") + 1] == "read-only", "the Codex writer's command line is wrong: %s" % " ".join(argv))
	_ok(bool((cm["pictures"][0] as Dictionary)["flip"]) and not bool((cm["pictures"][1] as Dictionary)["flip"]),
		"a reversed card's picture is not turned over for Codex")
	_ok(TextGen.has("codex") and TextGen.has("claude"), "the writers on offer are not Claude and Codex")
	# THE MODEL: the one chosen for a job wins over its tier's; none chosen is the CLI's own
	_ok(TextGen.Claude.model_of({"tier": "fast"}) == "sonnet" and TextGen.Claude.model_of({"tier": "best"}) == "opus"
		and TextGen.Claude.model_of({"tier": "fast", "model": "fable"}) == "fable",
		"a chosen model does not win over the tier's, or the tiers lost their defaults")
	var chosen := TextGen.Codex.argv(dict_with(job, "model", "gpt-test"), [])
	_ok(chosen.find("-m") >= 0 and chosen[chosen.find("-m") + 1] == "gpt-test" and chosen.find("--") > chosen.find("-m"),
		"the Codex writer is not run on the model chosen for it")
	_ok(TextGen.Codex.argv(job, []).find("-m") < 0, "the Codex writer names a model when none was chosen")
	for list in [TextGen.Claude.models(), TextGen.Codex.models(), ImageGen.Codex.models()]:
		_ok((list as Array).size() >= 1 and String(((list as Array)[0] as Dictionary)["key"]) == "",
			"a model list does not open on the CLI's own default")
	return true


## NO PAINTED OBJECTS: an older plan that names some makes no step for them, and the planner is not
## asked for any - the things on the table are the set dresser's, modeled (see _table_step).
func _no_objects() -> bool:
	var ep := _episode(3)
	var plan: Dictionary = ep.read_json("plan")
	(plan["look"] as Dictionary)["objects"] = [{"what": "a cup", "size": "small"}, {"what": "a lamp", "size": "large"}]
	ep.write_json("plan", plan)
	var objects := ep.steps().filter(func(st: Variant) -> bool: return String(st).contains("object"))
	_ok(objects.is_empty(), "a plan naming objects still makes steps for them: %s" % str(objects))
	_ok(ep.steps().has("image:card:3") and ep.steps().has("image:surface"), "the plan's other steps went too: %s" % str(ep.steps()))
	var p := TarotPrompts.producer("T", "B", 7, 3, true, TarotTable.FACES, TarotTable.FRAMES, [], TarotDeck.standard())
	_ok(p.has("prompt") and not String(p["prompt"]).contains("\"objects\""), "the planner is still asked for objects")
	return true


## THE CARDS MOVE AFTER THE WORDS: every reader prompt says the table acts only between passages,
## and none is told to end ON a move - "End on the moment you stop shuffling to pull the first
## card" got "And there. That's the first one." spoken before the card was out (2026-10-05).
func _moves_after_words() -> bool:
	var n := 3
	var ep := _episode(n)
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief."})
	var plan: Dictionary = ep.read_json("plan")
	for step in ["intro", "1", str(n), "close"]:
		var upto := 0 if step == "intro" else (n if step == "close" else int(step))
		var said: Array = [] if step == "intro" else prod._said(n if step == "close" else int(step) - 1)
		var text := String(TarotPrompts.reader("Test Tarot", "A brief.", plan, step, said, prod._drawn(upto), n)["prompt"])
		_ok(text.contains(TarotPrompts.MOVES), "the %s prompt does not say when the cards move" % step)
		_ok(not text.contains("End on the moment"), "the %s prompt asks for a passage that ends ON a move" % step)
	return true


## A PART ASKED FOR IS THE PART MADE. Run against a queue that refuses every job (no Settings
## here), so what the producer TRIED is the record: one step asked for is one step tried - with
## the shuffle, which costs nothing, made on the way - where Generate tries everything ready.
func _only_what_was_asked() -> bool:
	var ep := _episode(3)
	for f in DirAccess.get_files_at(ep.dir):
		if not String(f).begins_with("plan"):
			DirAccess.remove_absolute(ep.dir.path_join(f))
	var prod := TarotProducer.new(ep, {"title": "T", "brief": "B", "deck": TarotDeck.standard(), "cards": [3, 3]})
	prod.start(["design:2"])
	_ok(prod._tries.keys() == ["design:2"], "asked for design 2, the producer tried %s" % str(prod._tries.keys()))
	_ok(ep.has("draw"), "the shuffle a design needs, which costs nothing, was not made on the way")
	_ok(not prod.running, "a run asked for one step is still running after it")
	var all := TarotProducer.new(ep, {"title": "T", "brief": "B", "deck": TarotDeck.standard(), "cards": [3, 3]})
	all.start()
	for s in ["design:1", "design:3", "image:back", "image:surface", "say:intro"]:
		_ok(all._tries.has(s), "Generate did not try %s (it tried %s)" % [s, str(all._tries.keys())])
	return true


func dict_with(d: Dictionary, k: String, v: Variant) -> Dictionary:
	var o := d.duplicate()
	o[k] = v
	return o


## THE TABLE STEP: where it sits in the episode, what it is made from and what it takes with it,
## what the set dresser is told - and that it is told no card, two-sided like the reader's.
func _table_step() -> bool:
	var n := 3
	var ep := _episode(n)
	var steps := ep.steps()
	_ok(steps.has("table") and steps.find("table") > steps.find("image:backdrop") and steps.find("table") < steps.find("say:intro"),
		"the table is not a step between the room and the reading: %s" % str(steps))
	_ok(ep.needs("table") == ["plan", "image:surface"], "the table is made from %s" % str(ep.needs("table")))
	for s in ["image:back", "image:surface", "image:backdrop"]:
		ep.write_text(s, "png")
	ep.write_json("table", {"things": [{"name": "a cup", "parts": [{"shape": "lathe", "profile": [[0, 0], [3, 0], [3, 5], [0, 5]]}]}]})
	ep.invalidate("image:surface")
	_ok(ep.has("table"), "painting a new cloth took the table with it")
	ep.write_text("image:surface", "png")
	ep.invalidate("table")
	_ok(not ep.has("table") and ep.has("say:intro") and ep.has("image:surface"), "setting the table again took more than the table")
	ep.write_json("table", {"things": []})
	ep.invalidate("plan")
	_ok(not ep.has("table"), "a new plan kept the old plan's table")
	# NO CARD REACHES THE SET DRESSER, through the producer itself
	ep = _episode(n)
	ep.write_text("image:surface", "png")
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief."})
	var names: Array = []
	for k in range(1, n + 1):
		names.append(String(prod._card(k)["name"]))
	var head := TarotTable.headroom(ep.seed)
	var p := TarotPrompts.set_dresser("Test Tarot", "A brief.", ep.read_json("plan"), ep.seed, head, ["a brass bell"], true)
	var text := String(p["system"]) + "\n" + String(p["prompt"])
	var leaks := 0
	for nm in names:
		leaks += 1 if text.contains(String(nm)) else 0
	_ok(leaks == 0, "%d drawn card(s) reached the set dresser" % leaks)
	var cheat := TarotPrompts.set_dresser("Test Tarot", "A brief. " + " ".join(PackedStringArray(names)), ep.read_json("plan"), ep.seed, head, [], true)
	var seen := 0
	for nm in names:
		seen += 1 if (String(cheat["system"]) + String(cheat["prompt"])).contains(String(nm)) else 0
	_ok(seen == names.size(), "the set dresser's leak check is blind (%d of %d seen in a prompt that holds them)" % [seen, names.size()])
	# ...and what it IS told: every word the builder knows, the zones with their headroom, the
	# candles, what earlier tables held
	for k in Props.SHAPES:
		_ok(text.contains("- %s:" % k), "the set dresser is not told the shape %s" % k)
	for k in Props.MATERIALS:
		_ok(text.contains("- %s:" % k), "the set dresser is not told the material %s" % k)
	for k in Props.ORNAMENTS:
		_ok(text.contains("- %s:" % k), "the set dresser is not told the ornament %s" % k)
	for k in Props.PLAYS:
		_ok(text.contains("- %s:" % k), "the set dresser is not told the play of light %s" % k)
	for z in TarotTable.ZONES:
		_ok(text.contains("\"%s\": %s; up to %d cm" % [z, String((TarotTable.ZONES[z] as Dictionary)["about"]), int(head[z])]),
			"the set dresser is not told the zone %s and its headroom" % z)
	_ok(text.contains("Exactly 1 lit thing -"), "the set dresser is not told how many lit things the look has")
	_ok(text.contains("a brass bell"), "the set dresser is not told what earlier tables held")
	_ok(int(head["back"]) < int(head["left"]) and int(head["back"]) > 3, "the back holds less height than the sides: %s" % str(head))
	# THE EXAMPLE IS BUILDABLE, and the format it shows is the format read
	var ex: Variant = TextGen.extract_json(TarotPrompts.SET_EXAMPLE)
	_ok(ex is Dictionary and (TarotTable.sanitize_table(ex as Dictionary, {})["things"] as Array).size() == 1,
		"the format the set dresser is shown does not read back as one thing")
	# LANDING: junk is refused; a buildable reply is kept as it was written
	_ok(prod._land_table("no table here") != "", "a reply with no JSON landed as a table")
	_ok(prod._land_table("{\"things\": [{\"name\": \"a hat\", \"parts\": [{\"shape\": \"hat\"}]}]}") != "",
		"a table with nothing buildable on it landed")
	_ok(prod._land_table(TarotPrompts.SET_EXAMPLE) == "" and ep.has("table")
		and String(((ep.read_json("table") as Dictionary)["things"][0] as Dictionary)["name"]) == "a boxwood chess pawn",
		"a buildable table did not land as written")
	return true


## WHAT IS BUILT IS SAFE, whatever was written: shapes the builder does not know and junk numbers
## are dropped or clamped, no more flames than the table allows, every thing on y = 0 with a foot,
## a candle's flame at the top of its wax, an oversize thing scaled down whole. Built headless (no
## renderer: meshes and materials only).
func _things_built() -> bool:
	var junk := {"materials": {"m": {"kind": "unobtainium", "color": "red", "polish": 9}},
		"things": [
			{"name": "nothing at all", "parts": [{"shape": "hat"}, "not a part", {"shape": "lathe", "profile": [[1]]}]},
			{"name": "a giant", "place": "on the ceiling", "group": 3.0, "turn": "sideways",
				"parts": [{"shape": "box", "size": [900, 2, "x"], "material": "m", "wick": "yes"},
					{"shape": "ball", "smooth": "no", "facets": 1, "copies": {"ring": {"count": 2, "face": "no"}}}]},
			{"name": "candles", "parts": [{"shape": "lathe", "profile": [[0, 0], [1, 0], [1, 8], [0, 8]], "wick": true,
				"copies": {"ring": {"count": 3, "radius": 4}}}, {"shape": "lathe", "profile": [[0, 0], [1, 0], [1, 8], [0, 8]],
				"wick": true, "copies": {"line": {"count": 3, "step": [3, 0, 0]}}}]},
			{"name": "every shape", "parts": [
				{"shape": "ball", "radius": 2, "lumpy": 0.7, "at": [10, 0, 0]},
				{"shape": "ball", "size": [3, 2, 3], "facets": true, "at": [-10, 0, 0]},
				{"shape": "point", "radius": 1, "length": 4, "at": [0, 0, 6]},
				{"shape": "cluster", "count": 9, "radius": 3, "length": [1, 4], "base": "nowhere"},
				{"shape": "ring", "radius": 3, "thickness": 0.3, "arc": 200},
				{"shape": "tube", "path": [[0, 0, 0], [2, 3, 0], [4, 1, 1]], "radii": [0.4, 0.2]},
				{"shape": "sheet", "outline": "feather", "size": [2, 9], "bend": 0.5, "fold": 0.4},
				{"shape": "sheet", "points": [[0, 0], [4, 0], [5, 3], [1, 4]]},
				{"shape": "bloom", "petals": 7, "layers": 3, "radius": 3, "cup": 0.6},
				{"shape": "geode", "radius": 5, "at": [0, 0, -12]},
				{"shape": "ball", "size": [2, 1.2, 1.6], "lumpy": 0.5, "material": ["crystal", "stone", {"kind": "metal", "play": "rainbow"}],
					"copies": {"heap": {"count": 9, "radius": 1.5}, "jitter": 0.8}, "at": [0, 0, 12]},
				{"shape": "lathe", "profile": [[0, 0], [3, 0], [3.5, 5], [0, 5]], "sides": 6, "lobes": 5, "twist": 90,
					"ornament": {"kind": "stars", "count": 6, "color": "#ffcc00"}, "copies": {"scatter": {"count": 4, "radius": 5}, "jitter": 1}}]},
		]}
	var safe := TarotTable.sanitize_table(junk, {"palette": ["#112233", "#445566", "#778899"]})
	var things: Array = safe["things"]
	_ok(things.size() == 3, "junk did not lose exactly the thing with nothing buildable (%d things left)" % things.size())
	var mats: Dictionary = safe["materials"]
	_ok(String((mats["m"] as Dictionary)["kind"]) == "painted" and String((mats["m"] as Dictionary)["color"]).begins_with("#")
		and float((mats["m"] as Dictionary)["polish"]) <= 1.0, "a junk material was kept as written: %s" % str(mats["m"]))
	var giant: Dictionary = things[0]
	_ok(String(giant["place"]) == "back" and String(giant["group"]) == "3" and float(giant["turn"]) == 0.0,
		"a junk place, group or turn was kept: %s, %s, %s" % [giant["place"], giant["group"], giant["turn"]])
	var flames := 0
	for p in (things[1] as Dictionary)["parts"]:
		flames += Props.flames_of(p as Dictionary)
	_ok(flames == 6, "a candle's six flames (two parts of three copies) did not all survive: %d" % flames)
	for i in things.size():
		var b := Props.build(things[i], mats, 7 + i)
		var box: AABB = b["size"]
		_ok(absf(box.position.y) < 0.0005, "%s does not stand on the cloth (its lowest point at %.4f)" % [(things[i] as Dictionary)["name"], box.position.y])
		_ok(maxf(box.size.x, maxf(box.size.y, box.size.z)) <= Props.MAX_SIZE * 0.01 + 0.0005,
			"%s is bigger than anything may be (%s)" % [(things[i] as Dictionary)["name"], str(box.size)])
		_ok((b["foot"] as PackedVector2Array).size() >= 3 and (b["outline"] as PackedVector2Array).size() >= 3,
			"%s has no foot or outline" % (things[i] as Dictionary)["name"])
		_ok(not (b["meshes"] as Array).is_empty(), "%s built no meshes" % (things[i] as Dictionary)["name"])
		var verts := 0
		for m in b["meshes"]:
			var arr := ((m as MeshInstance3D).mesh as ArrayMesh).surface_get_arrays(0)
			var v: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
			var nn: PackedVector3Array = arr[Mesh.ARRAY_NORMAL]
			verts += v.size()
			var bad := 0
			for j in v.size():
				if not v[j].is_finite() or not nn[j].is_finite() or absf(nn[j].length() - 1.0) > 0.01:
					bad += 1
			_ok(bad == 0 and v.size() % 3 == 0, "%s has %d bad vertices or normals" % [(things[i] as Dictionary)["name"], bad])
		(b["node"] as Node).free()
	# A CANDLE'S FLAME at the top of its wax: a 9 cm candle on a 2 cm holder is lit at 11 cm, less its pool
	var cand := TarotTable.sanitize_table({"things": [{"name": "a candle", "parts": [
		{"shape": "lathe", "profile": [[0, 0], [3, 0], [3, 2], [0, 2]]},
		{"shape": "lathe", "profile": [[0, 0], [1.2, 0], [1.2, 9], [0, 9]], "at": [0, 2, 0], "wick": true}]}]}, {})
	var cb := Props.build(cand["things"][0], cand["materials"], 1)
	var w: Array = cb["wicks"]
	_ok(w.size() == 1 and absf((w[0] as Vector3).y - 0.11) < 0.005 and (w[0] as Vector3).y < 0.11
		and Vector2((w[0] as Vector3).x, (w[0] as Vector3).z).length() < 0.001,
		"the flame is not lit at the top of the wax: %s" % str(w))
	(cb["node"] as Node).free()
	# THE TABLE BEFORE THE SET DRESSER: the look's candles, lit
	var dt := TarotTable.default_table({"candles": 3, "palette": ["#112233", "#445566", "#778899"]}, 5)
	var lit := 0
	for t in dt["things"]:
		var b := Props.build(t, dt["materials"], 2)
		lit += (b["wicks"] as Array).size()
		(b["node"] as Node).free()
	_ok(lit == 3, "the table before the set dresser lights %d candles, not the look's 3" % lit)
	return true


## STONES, as a handful is set out. Scattered copies never touch, however crowded the handful;
## heaped ones pile up - some resting on others, none through another or the floor; a part's list
## of materials goes to its copies in turn, a mesh for each; every vertex knows its copy's middle,
## for a pattern to center on; a geode keeps its crystals inside it; a play of light is kept only
## when it is one the shader knows.
func _stones() -> bool:
	var spec := Props.sanitize({"materials": {
			"agate": {"kind": "crystal", "color": "#a0522d", "rings": true, "play": "Flash"},
			"glass": {"kind": "stone", "color": "#336633", "play": "sparkle", "rings": "yes"}},
		"things": [
			{"name": "a crowded handful", "parts": [{"shape": "ball", "size": [2, 1.2, 1.6], "lumpy": 0.5, "material": "agate",
				"copies": {"scatter": {"count": 16, "radius": 1}, "jitter": 1}}]},
			{"name": "a heap", "parts": [{"shape": "ball", "size": [2, 1.2, 1.6], "lumpy": 0.5,
				"material": ["agate", "glass", "crystal"], "copies": {"heap": {"count": 18, "radius": 1.5}, "jitter": 0.6}}]},
			{"name": "a geode", "parts": [{"shape": "geode", "radius": 7, "rind": 1.2, "length": 2.5, "material": "agate"}]}]},
		["#112233"])
	# A REPEAT WRITTEN BESIDE THE PART is read as its copies (a skein's turns were built one of each,
	# the top one floating) - and one under `copies` still wins
	var beside := Props.sanitize({"things": [{"name": "a skein", "parts": [
		{"shape": "ring", "radius": 3.5, "thickness": 0.8, "line": {"count": 4, "step": [0, 0.7, 0]}, "jitter": 0.3},
		{"shape": "ring", "radius": 3.5, "thickness": 0.8, "line": {"count": 4, "step": [0, 0.7, 0]},
			"copies": {"line": {"count": 2, "step": [0, 0.7, 0]}}}]}]}, ["#112233"])
	var bp: Array = (beside["things"][0] as Dictionary)["parts"]
	_ok(String(((bp[0] as Dictionary)["copies"] as Dictionary).get("kind", "")) == "line" and int(((bp[0] as Dictionary)["copies"] as Dictionary).get("count", 0)) == 4
		and float(((bp[0] as Dictionary)["copies"] as Dictionary).get("jitter", 0.0)) == 0.3,
		"a repeat written beside the part was not read as its copies: %s" % str((bp[0] as Dictionary)["copies"]))
	_ok(int(((bp[1] as Dictionary)["copies"] as Dictionary).get("count", 0)) == 2, "a repeat beside the part overrode the one under `copies`")
	var mats: Dictionary = spec["materials"]
	_ok(String((mats["agate"] as Dictionary).get("play", "")) == "flash" and (mats["agate"] as Dictionary)["rings"] == true,
		"a known play or rings was not kept: %s" % str(mats["agate"]))
	_ok(not (mats["glass"] as Dictionary).has("play") and (mats["glass"] as Dictionary)["rings"] == false,
		"an unknown play, or rings that are not true, was kept: %s" % str(mats["glass"]))
	var things: Array = spec["things"]
	_ok(things.size() == 3, "%d of 3 stone things survived sanitizing" % things.size())
	if things.size() != 3:
		return true
	# SCATTERED, never touching; HEAPED, piled without passing through
	for t in [things[0], things[1]]:
		var part: Dictionary = (t as Dictionary)["parts"][0]
		var rng := RandomNumberGenerator.new()
		rng.seed = 4
		var geos := Props._geometry(part, rng)
		var xforms := Props._placements(part, rng, geos)
		var bounds := Props._bounds(geos, Transform3D.IDENTITY)
		var reach := float(bounds["reach"])
		var tall := float(bounds["high"])
		_ok(xforms.size() == int((part["copies"] as Dictionary)["count"]), "%s laid out %d copies, not %d" % [t["name"], xforms.size(), part["copies"]["count"]])
		var touching := 0
		var through := 0
		var stacked := 0
		var below := 0
		for i in xforms.size():
			var a: Transform3D = xforms[i]
			var sa := a.basis.get_scale().x
			if a.origin.y < -0.0005:
				below += 1
			if a.origin.y > tall * 0.4:
				stacked += 1
			for j in range(i + 1, xforms.size()):
				var b: Transform3D = xforms[j]
				var sb := b.basis.get_scale().x
				var d := Vector2(a.origin.x - b.origin.x, a.origin.z - b.origin.z).length()
				if d < (sa + sb) * reach - 0.0002:
					touching += 1
				# two in the same place: near in plan AND at the same height
				if d < (sa + sb) * reach * 0.4 and absf(a.origin.y - b.origin.y) < tall * 0.4:
					through += 1
		if String((part["copies"] as Dictionary)["kind"]) == "scatter":
			_ok(touching == 0, "%d pairs of scattered copies touch" % touching)
		else:
			_ok(stacked > 0, "a heap of %d in a 1.5 cm circle did not pile up (none above the floor)" % xforms.size())
			_ok(through == 0, "%d pairs of heaped copies pass through each other" % through)
		_ok(below == 0, "%d copies of %s sink below the floor" % [below, t["name"]])
	# A LIST OF MATERIALS: one mesh for each, its copies in turn; every vertex knows its copy
	var heap := Props.build(things[1], mats, 3)
	var colors := {}
	var counts: Array = []
	var stray := 0
	for m in heap["meshes"]:
		var mi: MeshInstance3D = m
		colors[str((mi.material_override as ShaderMaterial).get_shader_parameter("color"))] = true
		var arr := (mi.mesh as ArrayMesh).surface_get_arrays(0)
		var v: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
		var c: PackedFloat32Array = arr[Mesh.ARRAY_CUSTOM0] if arr[Mesh.ARRAY_CUSTOM0] is PackedFloat32Array else PackedFloat32Array()
		counts.append(v.size())
		if c.size() != v.size() * 4:
			stray += v.size()
			continue
		for i in range(0, v.size(), 37):
			# a vertex's copy middle is near it: within the reach of one stone
			if Vector3(c[i * 4], c[i * 4 + 1], c[i * 4 + 2]).distance_to(v[i]) > 0.02:
				stray += 1
	_ok((heap["meshes"] as Array).size() == 3 and colors.size() == 3, "a heap in 3 materials built %d meshes in %d colors" % [(heap["meshes"] as Array).size(), colors.size()])
	_ok(counts.size() == 3 and counts[0] == counts[1] and counts[1] == counts[2], "18 copies in 3 materials did not go 6 to each: %s vertices" % str(counts))
	_ok(stray == 0, "%d vertices do not know their copy's middle" % stray)
	(heap["node"] as Node).free()
	# THE GEODE: its lining in the part's material, its rock in plain stone, the crystals inside
	var geo := Props.build(things[2], mats, 5)
	var meshes: Array = geo["meshes"]
	_ok(meshes.size() == 2, "a geode built %d meshes, not its lining and its rock" % meshes.size())
	if meshes.size() == 2:
		var top := [-INF, -INF]
		var wide := [0.0, 0.0]
		for k in 2:
			var v: PackedVector3Array = ((meshes[k] as MeshInstance3D).mesh as ArrayMesh).surface_get_arrays(0)[Mesh.ARRAY_VERTEX]
			for q in v:
				top[k] = maxf(top[k], q.y)
				wide[k] = maxf(wide[k], Vector2(q.x, q.z).length())
		var rock_kind := int(((meshes[1] as MeshInstance3D).material_override as ShaderMaterial).get_shader_parameter("kind"))
		_ok(rock_kind == int((Props.MATERIALS["stone"] as Dictionary)["code"]), "a geode's rock is not plain stone (kind %d)" % rock_kind)
		_ok(float(top[0]) <= float(top[1]) + 0.002 and float(wide[0]) < float(wide[1]),
			"a geode's crystals stand out of it (lining to %.3f m high, %.3f wide; rock %.3f, %.3f)" % [top[0], wide[0], top[1], wide[1]])
	_ok(absf((geo["size"] as AABB).position.y) < 0.0005, "a geode does not rest on the cloth")
	(geo["node"] as Node).free()
	return true


## THE CARD STOCK is the producer's choice (asked only for "card stock", it printed every deck on
## cream): it is told what the stock is and shown the stocks earlier decks used; the name reads on
## any stock; and the booklet is printed in the card's colors.
func _card_stock() -> bool:
	# shown an earlier deck's stock - two-sided: with no history that color is nowhere in the prompt
	var past := [{"seed": 5, "plan": {"episode_title": "E", "look": {"frame": {"stock": "#2b1d3c"}}}}]
	var p := String(TarotPrompts.producer("T", "B", 1, 3, true, TarotTable.FACES, TarotTable.FRAMES, past)["prompt"])
	_ok(p.contains("THE CARD STOCK"), "the producer is not told what the card stock is")
	_ok(p.contains("Card stock #2b1d3c"), "the producer is not shown an earlier deck's stock")
	var fresh := String(TarotPrompts.producer("T", "B", 1, 3, true, TarotTable.FACES, TarotTable.FRAMES, [])["prompt"])
	_ok(not fresh.contains("#2b1d3c"), "control: a stock no earlier episode used is in the prompt")
	_ok(not TarotPrompts.dice(1).has("stock"), "a die still decides the card stock")
	# THE NAME READS ON ANY STOCK: an ink that does not is moved until it does, either way round...
	for pair in [["#14121a", "#1d1a2b"], ["#efe6d2", "#f2e4c4"], ["#7a7a7a", "#808080"]]:
		_ok(TarotTable.contrast(Color.html(pair[1]), Color.html(pair[0])) < TarotTable.INK_CONTRAST,
			"control: %s on %s already reads" % [pair[1], pair[0]])
		var f: Dictionary = TarotTable.sanitize_look({"frame": {"stock": pair[0], "ink": pair[1]}})["frame"]
		_ok(TarotTable.contrast(TarotTable.color(String(f["ink"])), TarotTable.color(String(f["stock"])))
			>= TarotTable.INK_CONTRAST, "ink %s on stock %s was left unreadable (now %s)" % [pair[1], pair[0], f["ink"]])
	# ...and one that already reads is the producer's own, untouched
	var kept: Dictionary = TarotTable.sanitize_look({"frame": {"stock": "#e6d8bb", "ink": "#3A2A20"}})["frame"]
	_ok(String(kept["ink"]) == "#3A2A20", "a readable ink was changed: %s" % kept["ink"])
	# THE BOOKLET IS PRINTED IN THE CARD'S COLORS: its page is the stock, a dark one included, and its
	# type reads on it (the last look's ink and accent do not, as written)
	for look in [{"frame": {"stock": "#1b2a3f", "ink": "#e9dcc0", "accent": "#c9a227"}},
			{"frame": {"stock": "#efe6d2", "ink": "#3a2a20", "accent": "#7a2e3a"}},
			{"frame": {"stock": "#16131c", "ink": "#2a2433", "accent": "#3a2f45"}}]:
		var stock := TarotTable.color(String((look["frame"] as Dictionary)["stock"]))
		for shade in [0.0, 1.0]:
			var col := TarotCards.page_colors(look, shade)
			var paper: Color = col["paper"]
			_ok(TarotTable.contrast(paper, stock) < 1.15, "the booklet page %s is not the card's stock %s"
				% [paper.to_html(false), stock.to_html(false)])
			_ok(TarotTable.contrast(col["ink"], paper) >= TarotTable.TEXT_CONTRAST,
				"the booklet's text does not read on its page: %s" % str(look))
			_ok(TarotTable.contrast(col["accent"], paper) >= TarotTable.INK_CONTRAST,
				"the booklet's accent does not read on its page: %s" % str(look))
	# control: the fixed cream page the booklet had is nothing like a dark stock
	_ok(TarotTable.contrast(Color(0.955, 0.935, 0.885), Color.html("#1b2a3f")) > 3.0,
		"control: the old cream page passes for a dark stock")
	print("tarot_check: card stock - the producer's choice, ink readable, booklet in the card's colors")
	return true


## CANDLES OF EVERY FORM, and what a table allows. A pillar's three wicks stand on its pool, apart and
## inside its top; wicks placed by hand stand where they were put; a tea light in its tin lights one
## flame; a candelabra written once as a GROUP - arm, cup and taper, copied round a ring - holds a
## flame in every cup; groups nest only so deep and their parts count toward a thing's; a thing makes
## no more parts in all than it may; an extruded outline (a star tray, a heart, a hexagonal candle,
## an outline of its own going in and out) stands on the cloth, sound; and a table lights no more
## things than it may, none with more flames than it may.
func _candles() -> bool:
	var mats := {"wax": {"kind": "wax", "color": "#e8dcc0"}, "tin": {"kind": "metal", "color": "#c0c0c0"},
		"brass": {"kind": "metal", "color": "#b48a43"}, "slate": {"kind": "stone", "color": "#444a50"}}
	var arm := {"parts": [
		{"shape": "tube", "path": [[0, 12, 0], [5, 13, 0], [8, 15, 0]], "radius": 0.4, "material": "brass"},
		{"shape": "lathe", "profile": [[0, 0], [1.4, 0.2], [1.6, 1.2], [1.4, 1.2], [1.2, 0.4], [0, 0.4]], "at": [8, 15, 0], "material": "brass"},
		{"shape": "lathe", "profile": [[0, 0], [0.9, 0], [0.9, 9], [0, 9]], "at": [8, 15.4, 0], "material": "wax", "wick": true}],
		"copies": {"ring": {"count": 5, "radius": 0}}}
	var spec := TarotTable.sanitize_table({"materials": mats, "things": [
		{"name": "a three-wick pillar", "parts": [{"shape": "lathe", "profile": [[0, 0], [5, 0], [5, 6], [0, 6]], "material": "wax", "wicks": 3}]},
		{"name": "two wicks by hand", "parts": [{"shape": "lathe", "profile": [[0, 0], [4, 0], [4, 5], [0, 5]], "material": "wax", "wicks": [[-2, 0], [2, 0]]}]},
		{"name": "a tea light", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [1.9, 0], [1.9, 1.6], [1.8, 1.6], [1.8, 0.1], [0, 0.1]], "material": "tin"},
			{"shape": "lathe", "profile": [[0, 0], [1.8, 0], [1.8, 1.3], [0, 1.3]], "at": [0, 0.1, 0], "material": "wax", "wick": true}]},
		{"name": "a candelabra", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [5, 0], [4, 1], [1, 2], [0.8, 14], [0, 14]], "material": "brass"}, arm]},
		{"name": "a hexagonal candle", "parts": [{"shape": "extrude", "outline": "polygon", "sides": 6, "size": [6, 6], "height": 8,
			"taper": 0.1, "material": "wax", "wicks": 2}]},
		{"name": "a star tray", "parts": [{"shape": "extrude", "outline": "star", "sides": 5, "size": [12, 12], "height": 1.5, "wall": 0.4, "material": "brass"}]},
		{"name": "a slate heart", "parts": [{"shape": "extrude", "outline": "heart", "size": [8, 7], "height": 1, "bevel": 0.3, "material": "slate"}]},
		{"name": "an outline of its own", "parts": [{"shape": "extrude", "points": [[0, 0], [6, 0], [6, 4], [3, 1.5], [0, 4]], "height": 2, "material": "slate"}]},
		]}, {})
	var things: Array = spec["things"]
	_ok(things.size() == 8, "%d of 8 candles and extrudes survived sanitizing" % things.size())
	if things.size() != 8:
		return true
	var built: Array = []
	for i in things.size():
		var b := Props.build(things[i], spec["materials"], 11 + i)
		built.append(b)
		var box: AABB = b["size"]
		_ok(absf(box.position.y) < 0.0005, "%s does not stand on the cloth (%.4f)" % [things[i]["name"], box.position.y])
		var bad := 0
		for m in b["meshes"]:
			var arr := ((m as MeshInstance3D).mesh as ArrayMesh).surface_get_arrays(0)
			var v: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
			var nn: PackedVector3Array = arr[Mesh.ARRAY_NORMAL]
			for j in v.size():
				if not v[j].is_finite() or not nn[j].is_finite() or absf(nn[j].length() - 1.0) > 0.01:
					bad += 1
			bad += 0 if v.size() % 3 == 0 else 1
		_ok(bad == 0 and not (b["meshes"] as Array).is_empty(), "%s has %d bad vertices or no mesh" % [things[i]["name"], bad])
	# THREE WICKS on the pillar's pool: apart, inside its 5 cm top, a hair under its 6 cm
	var w3: Array = (built[0] as Dictionary)["wicks"]
	var apart := INF
	var out := 0.0
	var low := INF
	var high := -INF
	for i in w3.size():
		var a: Vector3 = w3[i]
		out = maxf(out, Vector2(a.x, a.z).length())
		low = minf(low, a.y)
		high = maxf(high, a.y)
		for j in range(i + 1, w3.size()):
			apart = minf(apart, a.distance_to(w3[j]))
	_ok(w3.size() == 3 and apart > 0.03 and out < 0.045 and low > 0.054 and high <= 0.0601,
		"a pillar's three wicks do not stand apart on its pool: %d, %.3f apart, %.3f out, %.4f-%.4f high" % [w3.size(), apart, out, low, high])
	var w2: Array = (built[1] as Dictionary)["wicks"]
	_ok(w2.size() == 2 and absf(absf((w2[0] as Vector3).x) - 0.02) < 0.001 and absf((w2[0] as Vector3).z) < 0.001
		and (w2[0] as Vector3).x * (w2[1] as Vector3).x < 0.0, "wicks placed by hand are not where they were put: %s" % str(w2))
	_ok(((built[2] as Dictionary)["wicks"] as Array).size() == 1, "a tea light does not light one flame")
	# THE CANDELABRA: a flame in every cup, round the stem, as high as its tapers
	var w5: Array = (built[3] as Dictionary)["wicks"]
	var axis := Vector2.ZERO           # the stem: the flames' middle (a five-armed thing's box is not)
	for w in w5:
		axis += Vector2((w as Vector3).x, (w as Vector3).z) / float(maxi(w5.size(), 1))
	var placed := 0
	for w in w5:
		var q: Vector3 = w
		placed += 1 if absf(Vector2(q.x, q.z).distance_to(axis) - 0.08) < 0.006 and absf(q.y - 0.244) < 0.006 else 0
	_ok(w5.size() == 5 and placed == 5, "a candelabra written once does not hold a flame in each of its 5 cups (%d flames, %d in place)" % [w5.size(), placed])
	# THE TABLE'S CAPS: four lit things at most - the hexagonal candle, the fifth, stands unlit - and
	# a part past the flames one thing may have is put out, alone
	_ok(Props.flames_of((things[4] as Dictionary)["parts"][0]) == 0, "a fifth lit thing was left burning (the table allows %d)" % TarotTable.MAX_CANDLES)
	var crowd := TarotTable.sanitize_table({"materials": mats, "things": [{"name": "a wall of candles", "parts": [
		{"shape": "lathe", "profile": [[0, 0], [2, 0], [2, 6], [0, 6]], "material": "wax", "wicks": 3, "copies": {"ring": {"count": 6, "radius": 6}}},
		{"shape": "lathe", "profile": [[0, 0], [2, 0], [2, 6], [0, 6]], "material": "wax", "wick": true}]}]}, {})
	var wall_parts: Array = (crowd["things"][0] as Dictionary)["parts"]
	_ok(Props.flames_of(wall_parts[0]) == 0 and Props.flames_of(wall_parts[1]) == 1,
		"a part past the flames one thing may have (%d) was not put out alone" % TarotTable.MAX_FLAMES)
	# GROUPS nest three deep, their parts count toward a thing's, and a thing makes no more parts in
	# all than it may
	var nest := {"shape": "ball", "radius": 1}
	for k in 5:
		nest = {"parts": [{"shape": "ball", "radius": 1}, nest]}
	var many: Array = []
	for k in 20:
		many.append({"shape": "ball", "radius": 1, "at": [k, 0, 0]})
	var deep := Props.sanitize({"things": [{"name": "nested", "parts": [nest]}, {"name": "crowded", "parts": [{"parts": many}]},
		{"name": "repeated", "parts": [{"parts": [{"shape": "ball", "radius": 0.5, "copies": {"line": {"count": 24, "step": [1, 0, 0]}}}],
			"copies": {"ring": {"count": 24, "radius": 20}}}]}]}, ["#112233"])
	var leaves := func(parts: Array, f: Callable) -> int:
		var c := 0
		for q in parts:
			c += int(f.call((q as Dictionary)["parts"], f)) if (q as Dictionary).has("parts") else 1
		return c
	var nested := int(leaves.call((deep["things"][0] as Dictionary)["parts"], leaves))
	_ok(nested == 3, "groups nested past %d deep were kept (%d parts)" % [Props.MAX_DEPTH, nested])
	_ok(int(leaves.call((deep["things"][1] as Dictionary)["parts"], leaves)) == Props.MAX_PARTS, "a group's parts did not count toward the %d" % Props.MAX_PARTS)
	var rep := Props.build(deep["things"][2], deep["materials"], 5)
	var one := Props.build(Props.sanitize({"things": [{"name": "one", "parts": [{"shape": "ball", "radius": 0.5}]}]}, ["#112233"])["things"][0],
		deep["materials"], 5)
	var per := ((one["meshes"][0] as MeshInstance3D).mesh as ArrayMesh).surface_get_array_len(0)
	var total := 0
	for m in rep["meshes"]:
		total += ((m as MeshInstance3D).mesh as ArrayMesh).surface_get_array_len(0)
	_ok(total > 0 and total <= per * Props.MAX_INSTANCES, "24 copies of 24 copies made %d balls (a thing makes %d at most)" % [total / maxi(per, 1), Props.MAX_INSTANCES])
	for b in built + [rep, one]:
		((b as Dictionary)["node"] as Node).free()
	# the set dresser is told all of it
	var words := Props.describe()
	_ok(words.contains("- extrude:") and words.contains("GROUPS:") and words.contains("`wicks`") and words.contains("FLAMES:"),
		"the set dresser is not told of extrudes, groups, wicks and one light a thing")
	return true


## THE ROOM'S PICTURE IS ASKED FOR AS THE TABLE PROJECTS IT: a level camera at a seated eye, its
## horizon across the middle, through the lens the table assumes ([constant TarotTable.BACKDROP_LENS]),
## with the lower third - all the tilted camera sees past the table - carrying the place.
func _room_prompt() -> bool:
	var p := TarotPrompts.backdrop_image({"setting": "a quiet room", "palette": ["#112233"]}, "/tmp/x.png")
	_ok(p.contains("THE CAMERA IS LEVEL") and p.contains("%d mm lens" % int(TarotTable.BACKDROP_LENS))
		and p.contains("exact middle") and p.contains("LOWER THIRD"),
		"the room's picture is not asked for level, through the table's lens, its lower third carrying the place")
	return true


## WHAT THE SHOW HAS ALREADY MADE reaches each agent in the part it decides: the producer sees an
## earlier episode's angle, running bit, what its cards pictured, its stock, cloth and room, and is
## asked to name the show's habits; the designer, how an earlier deck pictured the same card; a
## reader passage, how earlier episodes opened, met a card (a jumper apart) and closed - without
## the words the brief gives every episode, and with every card name masked, so no later card
## reaches a reader through another episode. Two-sided: an episode alone in its show is told none
## of it, and no episode is its own history.
func _archive() -> bool:
	TarotEpisode.root = "user://tarot_check"
	var show := "archive-show"
	var base := ProjectSettings.globalize_path(TarotEpisode.root.path_join(show))
	if DirAccess.dir_exists_absolute(base):
		_remove_tree(base)
	var brief := "A brief. \"Hello, my loves\" opens every episode. Its last words are \"Go and do the thing.\""
	var deck := TarotDeck.standard()
	var names: Array = []
	for c in deck:
		names.append(String((c as Dictionary)["name"]))
	var plan := {"episode_title": "THIS-TITLE", "audience": "everyone", "topic": "waiting",
		"premise": "THIS-PREMISE", "reader_mood": "calm", "running_bit": "THIS-BIT",
		"spread": {"name": "The Spread", "positions": [{"name": "One", "asks": "a"}, {"name": "Two", "asks": "b"},
			{"name": "Three", "asks": "c"}]},
		"look": TarotTable.sanitize_look({"deck_name": "This Deck", "deck_style": "THIS-STYLE"})}
	# this episode draws the Tower, the Star, the Moon; the earlier one drew the Tower, the Fool
	# (a jumper) and the Sun, and its reader named the MOON - a card this episode has not drawn yet
	var ep := TarotEpisode.open(show, 4242)
	ep.write_json("plan", plan)
	var drawn: Array = []
	for nm in ["The Tower", "The Star", "The Moon"]:
		drawn.append({"name": nm, "reversed": false, "jumper": false, "meaning": "m"})
	ep.write_json("draw", {"seed": 4242, "cards": drawn})
	for k in range(1, 4):
		ep.write_json("design:%d" % k, {"art": "this art %d" % k, "booklet": {"keywords": ["k"], "upright": "u"}})
		ep.write_text("say:%d" % k, "Passage for card %d." % k)
	ep.write_text("say:intro", "Opening of this episode.")
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": brief, "deck": deck})
	var producer := func() -> String:
		return String(TarotPrompts.producer("Test Tarot", brief, 4242, 3, true, TarotTable.FACES, TarotTable.FRAMES,
			TarotEpisode.archive(show, 4242), deck)["prompt"])
	var designer := func(k: int) -> String:
		return String(TarotPrompts.designer("Test Tarot", brief, prod._look(), prod._card(k), true,
			TarotEpisode.archive(show, 4242))["prompt"])
	# ALONE IN ITS SHOW: told nothing of other episodes
	_ok(TarotEpisode.archive(show, 4242).is_empty(), "an episode alone in its show has a history")
	var alone := producer.call() as String
	_ok(not alone.contains("EARLIER EPISODES") and not alone.contains("\"habits\""),
		"a producer with no earlier episodes is shown some, or asked for their habits")
	_ok(not (designer.call(1) as String).contains("EARLIER DECKS"), "a designer with no earlier decks is shown one")
	for who in ["intro", "1", "close"]:
		_ok(not String(prod.say_prompt(who)["prompt"]).contains("HOW EARLIER"), "a reader alone in its show is told what earlier ones said (%s)" % who)
	# AN EARLIER EPISODE beside it
	var old := TarotEpisode.open(show, 1111)
	old.write_json("plan", {"episode_title": "OLD-TITLE", "audience": "OLD-AUDIENCE", "topic": "OLD-TOPIC",
		"premise": "OLD-PREMISE", "reader_mood": "OLD-MOOD", "running_bit": "OLD-BIT",
		"spread": {"name": "OLD-SPREAD", "positions": []},
		"look": {"deck_name": "Old Deck", "deck_style": "OLD-STYLE", "frame": {"stock": "#123456"},
			"surface": "OLD-CLOTH", "setting": "OLD-ROOM", "light": {"kind": "OLD-LIGHT"}}})
	old.write_json("draw", {"seed": 1111, "cards": [{"name": "The Tower"}, {"name": "The Fool", "jumper": true}, {"name": "The Sun"}]})
	for k in range(1, 4):
		old.write_json("design:%d" % k, {"art": "OLD-ART-%d of the old deck" % k, "booklet": {"upright": "u"}})
	old.write_text("say:intro", "Hello, my loves. OLD-OPENING, welcome.")
	old.write_text("say:1", "Oh. The Tower. OLD-REACTION, and it is not the Moon either.")
	old.write_text("say:2", "OLD-JUMPER, it flew right out.")
	old.write_text("say:3", "The Sun. OLD-REACTION again.")
	old.write_text("say:close", "So overall, OLD-CLOSING. <!-- hesitation --> Go and do the thing.")
	old.write_json("table", {"things": [{"name": "OLD-THING"}]})
	var past := TarotEpisode.archive(show, 4242)
	_ok(past.size() == 1 and int((past[0] as Dictionary)["seed"]) == 1111, "the archive is not the other episode alone: %s" % str(past))
	_ok(((past[0] as Dictionary)["things"] as Array) == ["OLD-THING"], "the archive does not hold the earlier table's things")
	_ok(TarotEpisode.archive(show, 1111).size() == 1 and int((TarotEpisode.archive(show, 1111)[0] as Dictionary)["seed"]) == 4242,
		"an episode is its own history")
	# THE PRODUCER: every choice the earlier one made, and the habits asked for
	var pp := producer.call() as String
	for mark in ["OLD-TITLE", "OLD-PREMISE", "OLD-MOOD", "OLD-BIT", "OLD-STYLE", "OLD-ART-1", "OLD-ART-3", "#123456", "OLD-CLOTH", "OLD-ROOM", "OLD-LIGHT", "OLD-SPREAD"]:
		_ok(pp.contains(mark), "the producer is not shown the earlier episode's %s" % mark)
	_ok(pp.contains("\"habits\"") and pp.contains("THE SHOW'S HABITS"), "the producer is not asked to name the show's habits")
	_ok(not pp.contains("THIS-TITLE") and not pp.contains("THIS-PREMISE"), "the producer is shown the episode it is planning")
	# THE DESIGNER: the same card as an earlier deck pictured it, and only that card
	var d1 := designer.call(1) as String
	_ok(d1.contains("EARLIER DECKS") and d1.contains("OLD-ART-1") and d1.contains("Old Deck"),
		"the designer of the Tower is not shown how an earlier deck pictured it")
	_ok(not d1.contains("OLD-ART-2") and not d1.contains("OLD-ART-3"), "the designer is shown earlier decks' other cards")
	_ok(not (designer.call(2) as String).contains("EARLIER DECKS"), "the designer of a card no earlier deck drew is shown one")
	# THE READER, through the producer itself: what was heard at the same point, the brief's own words
	# left out, card names masked
	var intro := String(prod.say_prompt("intro")["prompt"])
	_ok(intro.contains("HOW EARLIER EPISODES OPENED") and intro.contains("OLD-OPENING"), "the intro is not shown how earlier episodes opened")
	_ok(not intro.contains("Hello, my loves"), "the intro's record keeps the greeting the brief gives every episode")
	var one := String(prod.say_prompt("1")["prompt"])
	_ok(one.contains("HOW EARLIER EPISODES MET A CARD") and one.contains("OLD-REACTION"), "a card's passage is not shown how earlier ones met a card")
	_ok(not one.contains("OLD-JUMPER"), "a card's passage is shown how a jumper was met")
	_ok(not one.contains("The Moon") and not one.contains("the Moon") and one.contains("[card]"),
		"a later card reached a reader through an earlier episode's words")
	_ok(one.contains("Your running bit: THIS-BIT"), "the reader is not told the episode's running bit")
	var close := String(prod.say_prompt("close")["prompt"])
	_ok(close.contains("HOW EARLIER EPISODES CLOSED") and close.contains("OLD-CLOSING"), "the close is not shown how earlier episodes closed")
	_ok(close.contains("- \"So overall, OLD-CLOSING.\"\n") and not close.contains("Go and do the thing"),
		"the close's record keeps the sign-off the brief gives every episode, or a mark")
	var jumper: Dictionary = (drawn[0] as Dictionary).duplicate()
	jumper["jumper"] = true
	var jp := String(TarotPrompts.reader("Test Tarot", brief, plan, "1", prod._said(0), [jumper], 3, false, past, names)["prompt"])
	_ok(jp.contains("HOW EARLIER EPISODES MET A JUMPER") and jp.contains("OLD-JUMPER") and not jp.contains("OLD-REACTION"),
		"a jumper's passage is not shown how earlier jumpers were met, and only that")
	# CONTROLS: unmasked, the later card is there; with no brief, the greeting stays
	var bare := String(TarotPrompts.reader("Test Tarot", brief, plan, "1", prod._said(0), prod._drawn(1), 3, false, past, [])["prompt"])
	_ok(bare.contains("the Moon"), "control: the mask check is blind - unmasked, the earlier reader's Moon is not there")
	_ok(TarotPrompts.unfixed("Hello, my loves. Okay.", "").contains("Hello, my loves"),
		"control: a greeting the brief does not give is dropped")
	_ok(TarotPrompts.mask_cards("Oh. The Tower, and the Ace of Cups.", ["The Tower", "Ace of Cups", "Ace"]) == "Oh. [card], and [card].",
		"card names are not masked whole: %s" % TarotPrompts.mask_cards("Oh. The Tower, and the Ace of Cups.", ["The Tower", "Ace of Cups", "Ace"]))
	# A PLAN LANDS WITH ITS HABITS AND ITS RUNNING BIT in the shape their readers expect
	var fresh := TarotEpisode.open(show, 2222)
	var lp := TarotProducer.new(fresh, {"title": "T", "brief": brief, "cards": [3, 3]})
	_ok(lp._land_plan(JSON.stringify({"episode_title": "T", "habits": "titles end in brackets, decks show people",
		"running_bit": 7, "spread": {"positions": ["a", "b", "c"]}, "look": {}})).is_empty(), "a plan with habits did not land")
	var landed: Dictionary = fresh.read_json("plan")
	_ok(landed.get("habits") == ["titles end in brackets", "decks show people"] and String(landed.get("running_bit")) == "7",
		"habits and the running bit did not land reshaped: %s / %s" % [str(landed.get("habits")), str(landed.get("running_bit"))])
	_remove_tree(base)
	print("tarot_check: archive - each agent shown what earlier episodes made in its own part; card names masked")
	return true
