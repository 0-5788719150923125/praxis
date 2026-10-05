extends RefCounted
class_name TarotProducer

## TarotProducer - makes whatever an episode is missing, in the order a reading happens.
##
## THE LOOP. Every [method tick] looks at the episode's steps (see [TarotEpisode]) and starts
## each one whose inputs exist and whose own file does not; when a job ends, its output is
## checked and written into the episode, which is what lets the steps after it start. So the
## order is not a script anywhere - it falls out of what each step is made from:
##
##   plan       the producer plans the episode: title, angle, spread, the deck's whole look
##   draw       the deck is shuffled and cut (here, from the seed - no agent touches it)
##   design:K   the deck's creator draws up card K: its illustration and its booklet entry
##   image:*    the painter makes the back, the cloth, the room, then each card in turn
##   say:intro  the reader opens the video, shuffling, knowing no card
##   say:K      the reader turns over card K - knowing cards 1..K and what it said before, and
##              LOOKING AT card K's painting, so the words are about the picture on screen
##   say:close  the reader closes, with the whole spread on the table
##   script     the passages and the marks between them, ready for the voice
##
## NO CHEATING. The reader's passages are made strictly in drawing order, each from the passages
## before it and the cards drawn so far ([method _drawn] stops at K), so an early passage was
## written, and is kept, before any later card was ever put in front of a writer. The painter and
## the designer see the cards they make and nothing of the reading; the reader never sees their
## prompts, only the booklet entry printed for a card that has been turned over.
##
## Nothing runs on its own: [method start] is a person pressing Generate, because every step
## spends the author's quota - and a person asking for ONE part ([param only]) gets that part:
## what follows from it waits for Generate. Only the two steps that cost nothing and are a pure
## function of what they follow - the shuffle and assembling the script - come along on their own.

signal changed

## A step whose job fails is tried this many more times before the episode stops on it.
const RETRIES := 1
## How much of the deck goes into a card's references: the back, the first card (which set the
## hand) and the most recent - the [Illustrations] chain, for the same reason: more attachments
## make a painter worse at matching, not better.
const CHAIN_MAX := 2
## The chance the first card is a jumper, when the show allows them.
const JUMPER_CHANCE := 0.3
## The steps made here, with no agent and no quota - see the class note.
const LOCAL := ["draw", "script"]

var episode: TarotEpisode
## {title, brief, cards: [lo, hi], reversals, jumpers, writer, painter}
var spec: Dictionary = {}
var running := false
## The steps this run was asked for; empty for the whole episode.
var only: Array = []

var _jobs := {}        # step -> AgentJobs id
var _errors := {}      # step -> why it failed, once it is out of tries
var _tries := {}       # step -> runs started


func _init(ep: TarotEpisode, sp: Dictionary) -> void:
	episode = ep
	spec = sp


## How many cards this episode draws: from its seed, within the show's range. Decided before
## anything is planned, so the producer plans a spread of exactly that many.
static func spread_size(seed: int, lo: int, hi: int) -> int:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "tarot-spread"])
	return rng.randi_range(mini(lo, hi), maxi(lo, hi))


## Make what is missing: every step, or only [param steps] (and the free steps that follow them).
func start(steps: Array = []) -> void:
	_errors.clear()
	_tries.clear()
	only = steps.duplicate()
	running = true
	tick()
	changed.emit()


func stop() -> void:
	running = false
	for step in _jobs:
		AgentJobs.cancel(String(_jobs[step]))
		AgentJobs.forget(String(_jobs[step]))
	_jobs.clear()
	changed.emit()


func busy() -> bool:
	return not _jobs.is_empty()


## "ready", "running", "queued", "failed" or "missing".
func state_of(step: String) -> String:
	if episode.has(step):
		return "ready"
	if _jobs.has(step):
		var st := AgentJobs.state(String(_jobs[step]))
		return "running" if st == "running" else "queued"
	if _errors.has(step):
		return "failed"
	return "missing"


func error_of(step: String) -> String:
	return String(_errors.get(step, ""))


## Land what ended; start what can start.
func tick() -> void:
	var moved := false
	for step in _jobs.keys():
		var id := String(_jobs[step])
		var st := AgentJobs.state(id)
		if st != "done" and st != "failed":
			continue
		var res := AgentJobs.result(id)
		AgentJobs.forget(id)
		_jobs.erase(step)
		_land(String(step), res)
		moved = true
	if running:
		for step in episode.steps():
			var s := String(step)
			if episode.has(s) or _jobs.has(s) or _errors.has(s):
				continue
			var ready := true
			for need in episode.needs(s):
				if not episode.has(String(need)):
					ready = false
					break
			if not ready or not (only.is_empty() or only.has(s) or LOCAL.has(s)):
				continue
			_make(s)
			moved = true
		if _jobs.is_empty():
			running = false          # done, or stopped on a failure nothing else can get past
			moved = true
	if moved:
		changed.emit()


# --- making one step ---------------------------------------------------------------------

func _make(step: String) -> void:
	var parts := step.split(":")
	match String(parts[0]):
		"plan":
			_make_plan()
		"draw":
			_finish(step, _make_draw())
		"design":
			_make_design(int(parts[1]))
		"image":
			_make_image(step)
		"say":
			_make_say(String(parts[1]))
		"script":
			_finish(step, _make_script())


func _submit_text(step: String, p: Dictionary, tier: String) -> void:
	_tries[step] = int(_tries.get(step, 0)) + 1
	if not p.has("system") or not p.has("prompt"):
		_errors[step] = "its prompt could not be built (see the log)"
		return
	var id := AgentJobs.submit({"kind": "text", "backend": String(spec.get("writer", "claude")),
		"tier": tier, "dir": episode.job_dir(step), "system": String(p["system"]),
		"prompt": String(p["prompt"]), "images": p.get("images", []), "label": "tarot %s" % step,
		"model": String(spec.get("writer_model", ""))})
	if id.is_empty():
		_errors[step] = "this session cannot start agents (read-only)"
		return
	_jobs[step] = id


func _submit_image(step: String, prompt: String, refs: Array) -> void:
	_tries[step] = int(_tries.get(step, 0)) + 1
	var id := AgentJobs.submit({"kind": "image", "backend": String(spec.get("painter", "codex")),
		"dir": episode.job_dir(step), "prompt": prompt, "refs": refs,
		"target": episode.file_of(step), "label": "tarot %s" % step,
		"model": String(spec.get("painter_model", ""))})
	if id.is_empty():
		_errors[step] = "this session cannot start agents (read-only)"
		return
	_jobs[step] = id


## A local step's outcome: "" written, else why not. A step that says it succeeded and left no
## file failed all the same - a script error returns "" - and must say so, or the producer goes
## idle with the step still missing and nothing on screen saying why.
func _finish(step: String, err: String) -> void:
	if err.is_empty() and not episode.has(step):
		err = "it wrote nothing (see the log)"
	if not err.is_empty():
		_errors[step] = err


func _plan() -> Dictionary:
	var p: Variant = episode.read_json("plan")
	return p if p is Dictionary else {}


func _look() -> Dictionary:
	var l: Variant = _plan().get("look", {})
	return l if l is Dictionary else {}


func _cards_range() -> Array:
	var c: Variant = spec.get("cards", [3, 6])
	if c is Array and (c as Array).size() == 2:
		return [clampi(int(c[0]), 1, 10), clampi(int(c[1]), 1, 10)]
	return [3, 6]


## The deck the show reads with (see [TarotDeck]): handed over in the spec, the standard 78 when
## the spec has none.
func _deck() -> Array:
	var d: Variant = spec.get("deck", [])
	return d if d is Array and not (d as Array).is_empty() else TarotDeck.standard()


## How many cards this episode draws: the seed's pick in the show's range, never more than the
## deck holds.
func _spread_n() -> int:
	var r := _cards_range()
	return mini(spread_size(episode.seed, int(r[0]), int(r[1])), _deck().size())


func _make_plan() -> void:
	var n := _spread_n()
	var past: Array = []
	for h in TarotEpisode.history(episode.show):
		if int((h as Dictionary)["seed"]) != episode.seed:
			past.append(h)
	var p := TarotPrompts.producer(String(spec.get("title", "")), String(spec.get("brief", "")),
		episode.seed, n, bool(spec.get("reversals", true)), TarotTable.FACES,
		TarotTable.FRAMES, past, _deck())
	_submit_text("plan", p, "best")


## The shuffle and the cut, from the seed alone.
func _make_draw() -> String:
	var plan := _plan()
	var n := ((plan.get("spread", {}) as Dictionary).get("positions", []) as Array).size()
	if n <= 0:
		return "the plan has no spread"
	# the show's own deck, shuffled from the seed - and each card WRITTEN INTO the draw, so the
	# episode keeps the cards it drew whatever the show's deck becomes later
	var deck := TarotDeck.shuffled(_deck(), episode.seed, bool(spec.get("reversals", true)))
	if deck.size() < n:
		return "the deck has %d cards and this spread needs %d (the plan was made for a bigger deck: New episode)" \
			% [deck.size(), n]
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([episode.seed, "tarot-jumper"])
	var jumper := bool(spec.get("jumpers", true)) and rng.randf() < JUMPER_CHANCE
	var cards: Array = []
	for i in n:
		var c: Dictionary = (deck[i] as Dictionary).duplicate()
		c["jumper"] = jumper and i == 0
		cards.append(c)
	return episode.write_json("draw", {"seed": episode.seed, "cards": cards})


## Card [param k] (1-based) as drawn: the deck's card, the way up it came, its place in the
## spread and - once designed - its booklet entry.
func _card(k: int) -> Dictionary:
	var draw: Variant = episode.read_json("draw")
	if not (draw is Dictionary):
		return {}
	var list: Array = (draw as Dictionary).get("cards", [])
	if k < 1 or k > list.size():
		return {}
	var card := (list[k - 1] as Dictionary).duplicate()
	card["reversed"] = bool(card.get("reversed", false))
	card["jumper"] = bool(card.get("jumper", false))
	var design: Variant = episode.read_json("design:%d" % k)
	if design is Dictionary:
		card["booklet"] = (design as Dictionary).get("booklet", {})
		card["art"] = String((design as Dictionary).get("art", ""))
	return card


## THE CARDS THE READER MAY KNOW at a step: those turned over so far - 1..k for card k, none for
## the intro, all of them for the close. The only way a card reaches a reader prompt.
func _drawn(upto: int) -> Array:
	var out: Array = []
	for k in range(1, upto + 1):
		out.append(_card(k))
	return out


func _make_design(k: int) -> void:
	var card := _card(k)
	var p := TarotPrompts.designer(String(spec.get("title", "")), String(spec.get("brief", "")),
		_look(), card, bool(spec.get("reversals", true)))
	_submit_text("design:%d" % k, p, "fast")


func _make_image(step: String) -> void:
	var look := _look()
	# the painter saves into its own job directory; the queue moves the picture into the episode
	# whole (see AgentJobs.paint_target)
	var target := AgentJobs.paint_target(episode.job_dir(step))
	var parts := step.split(":")
	if parts.size() == 3:
		var k := int(parts[2])
		var design: Variant = episode.read_json("design:%d" % k)
		var art := String((design as Dictionary).get("art", "")) if design is Dictionary else ""
		var refs: Array = []
		var has_back := episode.has("image:back")
		if has_back:
			refs.append(episode.file_of("image:back"))
		var chain: Array = []
		for j in range(1, k):
			if episode.has("image:card:%d" % j):
				chain.append(episode.file_of("image:card:%d" % j))
		if chain.size() > CHAIN_MAX:
			chain = [chain[0]] + chain.slice(chain.size() - (CHAIN_MAX - 1))
		refs.append_array(chain)
		_submit_image(step, TarotPrompts.card_image(look, _card(k), art, target, has_back,
			chain.size()), refs)
		return
	match String(parts[1]):
		"back":
			_submit_image(step, TarotPrompts.back_image(look, target), [])
		"surface":
			_submit_image(step, TarotPrompts.surface_image(look, target), [])
		"backdrop":
			_submit_image(step, TarotPrompts.backdrop_image(look, target), [])


func _make_say(who: String) -> void:
	_submit_text("say:" + who, say_prompt(who), "best")


## THE READER'S PROMPT for passage [param who] ("intro", "1".."N", "close"): the passages before it
## and [method _drawn] up to its card - nothing else reaches it - and `images`, the PAINTINGS it
## talks about: card K's own for card K, every card's for the close, none for the intro. A
## reversed card is sent upside down, as the viewer sees it. Public so the gate can hold the
## producer itself, not only the prompt builder, to all of that.
func say_prompt(who: String) -> Dictionary:
	var n := episode.card_count()
	var said: Array = []
	var drawn: Array = []
	if who == "intro":
		pass
	elif who == "close":
		said = _said(n)
		drawn = _drawn(n)
	else:
		var k := int(who)
		said = _said(k - 1)
		drawn = _drawn(k)
	var images: Array = []
	if who != "intro":
		var plan_pos: Array = ((_plan().get("spread", {}) as Dictionary).get("positions", [])) as Array
		var first := 1 if who == "close" else int(who)
		for k in range(first, drawn.size() + 1):
			var c: Dictionary = drawn[k - 1]
			var pos: Variant = plan_pos[k - 1] if k - 1 < plan_pos.size() else {}
			var where := String((pos as Dictionary).get("name", "")) if pos is Dictionary else ""
			var label := ("Card %d, just turned over - its painting, as the viewer sees it:" % k) if who != "close" \
				else ("Card %d%s - %s%s:" % [k, (" (" + where + ")") if not where.is_empty() else "",
					String(c.get("name", "")), ", reversed" if bool(c.get("reversed", false)) else ""])
			images.append({"path": episode.file_of("image:card:%d" % k), "label": label,
				"flip": bool(c.get("reversed", false))})
	var p := TarotPrompts.reader(String(spec.get("title", "")), String(spec.get("brief", "")),
		_plan(), who, said, drawn, n, not images.is_empty())
	p["images"] = images
	return p


## The intro and the passages for cards 1..upto, in order.
func _said(upto: int) -> Array:
	var out := [episode.read_text("say:intro")]
	for k in range(1, upto + 1):
		out.append(episode.read_text("say:%d" % k))
	return out


func _make_script() -> String:
	var n := episode.card_count()
	var passages: Array = [{"kind": "shuffle", "card": 0, "text": episode.read_text("say:intro")}]
	for k in range(1, n + 1):
		var c := _card(k)
		passages.append({"kind": "jumper" if bool(c.get("jumper", false)) else "draw", "card": k,
			"text": episode.read_text("say:%d" % k)})
	passages.append({"kind": "spread", "card": 0, "text": episode.read_text("say:close")})
	return episode.write_text("script", TarotScript.compose(passages))


# --- landing a job -------------------------------------------------------------------------

func _land(step: String, res: Dictionary) -> void:
	var err := ""
	if not bool(res.get("ok", false)):
		err = String(res.get("error", "the job failed"))
	else:
		match String(step.split(":")[0]):
			"plan":
				err = _land_plan(String(res.get("text", "")))
			"design":
				err = _land_design(step, String(res.get("text", "")))
			"say":
				var t := clean_spoken(String(res.get("text", "")))
				err = "the reader wrote nothing speakable" if t.split(" ", false).size() < 20 \
					else episode.write_text(step, t)
			"image":
				err = "" if episode.has(step) else "the picture did not arrive"
	if err.is_empty():
		_errors.erase(step)
		return
	push_warning("ghost: tarot %s - %s" % [step, err])
	if int(_tries.get(step, 0)) > RETRIES:
		_errors[step] = err


func _land_plan(text: String) -> String:
	var d: Variant = TextGen.extract_json(text)
	if not (d is Dictionary):
		return "the plan was not JSON"
	var plan: Dictionary = d
	var n := _spread_n()
	var spread: Dictionary = plan.get("spread", {}) if plan.get("spread") is Dictionary else {}
	# EVERY FIELD IN THE SHAPE ITS READERS EXPECT: a position given as a bare name, tags as one
	# string - a reply of the wrong shape is reshaped here, once, or it stalls the episode later
	# inside a prompt builder, where it fails with nothing on screen
	var pos: Array = []
	for q in (spread.get("positions", []) if spread.get("positions") is Array else []):
		if q is Dictionary:
			pos.append({"name": _str((q as Dictionary).get("name", "")), "asks": _str((q as Dictionary).get("asks", ""))})
		elif not _str(q).is_empty():
			pos.append({"name": _str(q), "asks": ""})
	spread["name"] = _str(spread.get("name", ""))
	for k in ["episode_title", "description", "audience", "topic", "premise", "reader_mood"]:
		plan[k] = _str(plan.get(k, ""))
	plan["tags"] = Array(TarotPrompts.strings(plan.get("tags", [])))
	# EXACTLY the size drawn for this seed: a plan that miscounts is trimmed, or filled out
	# with the position readers reach for when they want one more card
	pos = pos.slice(0, n)
	while pos.size() < n:
		pos.append({"name": "Clarifier", "asks": "what the cards before it are really saying"})
	spread["positions"] = pos
	plan["spread"] = spread
	if not (plan.get("look") is Dictionary):
		return "the plan has no look"
	plan["look"] = TarotTable.sanitize_look(plan["look"] as Dictionary)
	plan["seed"] = episode.seed
	plan["dice"] = TarotPrompts.dice(episode.seed)
	return episode.write_json("plan", plan)


func _land_design(step: String, text: String) -> String:
	var d: Variant = TextGen.extract_json(text)
	if not (d is Dictionary) or _str((d as Dictionary).get("art", "")).is_empty():
		return "the design was not JSON with an illustration"
	var b: Variant = (d as Dictionary).get("booklet", {})
	if not (b is Dictionary) or _str((b as Dictionary).get("upright", "")).is_empty():
		return "the design has no booklet entry"
	var booklet := {"keywords": Array(TarotPrompts.strings((b as Dictionary).get("keywords", []))),
		"upright": _str((b as Dictionary).get("upright", ""))}
	if not _str((b as Dictionary).get("reversed", "")).is_empty():
		booklet["reversed"] = _str((b as Dictionary)["reversed"])
	return episode.write_json(step, {"art": _str((d as Dictionary)["art"]), "booklet": booklet})


## A reply's field as text, whatever the writer made it (JSON numbers arrive as floats).
static func _str(v: Variant) -> String:
	if v is String:
		return (v as String).strip_edges()
	if v is float and is_equal_approx(v, roundf(v)):
		return str(int(v))
	return "" if v == null else str(v).strip_edges()


## THE READER'S WORDS AS THE VOICE GETS THEM. The prompt asks for spoken words only; this is
## what is left to do when a writer adds a little anyway - wrapping quotes, a heading, a
## bracketed stage direction, a line that is nothing but an action in asterisks - and the em
## dash, which the voice reads as a pause either way and the subtitles show as a plain dash.
static func clean_spoken(text: String) -> String:
	var t := text.strip_edges()
	if t.length() >= 2 and t.begins_with("\"") and t.ends_with("\""):
		t = t.substr(1, t.length() - 2)
	var keep := PackedStringArray()
	var action := Manuscript._rx("^\\s*(\\*[^*]+\\*|\\([^)]*\\)|\\[[^\\]]*\\])\\s*$")
	for line in t.split("\n"):
		var l := String(line)
		if l.strip_edges().begins_with("#") or action.search(l) != null:
			continue
		keep.append(l)
	t = "\n".join(keep)
	t = Manuscript._rx("\\[[^\\]]*\\]").sub(t, "", true)
	t = t.replace("\u2014", " - ").replace("\u2013", " - ").replace("**", "*")
	t = Manuscript._rx("[ \\t]{2,}").sub(t, " ", true)
	t = Manuscript._rx("\\n{3,}").sub(t, "\n\n", true)
	return t.strip_edges()
