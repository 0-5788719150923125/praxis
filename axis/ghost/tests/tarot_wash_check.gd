extends Node

## THE WASH SHUFFLES. Reported 2026-10-05: "When cards are splayed-out, they are BARELY shuffled at
## all. 3 or 4 cards might shift slightly, and then we start collecting them again... no changing
## of z-order (which should only happen when cards are not touching, to prevent clipping)... 5 or
## 10 seconds long, max - when in the real world, that kind of shuffle should take at least twice
## that long". Measured over many planned washes, straight off their tracks:
##
##   - it is long, and most of it is the mixing (not the spreading or the gathering);
##   - the hands move most of the cards, and far - a card's path through the mixing;
##   - the stacking order CHANGES: two cards that lay one way when they met lie the other way at a
##     later meeting - and NEVER while they touch: two cards overlapping from one step to the next
##     keep their order and a card's thickness between them (that would be a card through a card);
##   - no card jumps: its step at the plan's rate is a hand's speed at most;
##   - and the plan is cheap enough to make at an episode's load;
##   - a JUMPER flies out of a shuffle: when it springs the deck is mid-riffle, even when the
##     shuffle before it had gone quiet ("not when the cards are just sitting there").
##
##   tests/run_boot_probe.sh tests/tarot_wash_check.gd 180
##
## A BOOT probe (the medium reaches the Director); no GPU - the tracks are numbers.

const SEEDS := 12

var _fails := 0
var medium: TarotMedium
var subs: Subtitles


func _ready() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _run() -> void:
	var stage := SubViewport.new()
	stage.size = Vector2i(640, 360)
	stage.own_world_3d = true
	add_child(stage)
	Director.detach()
	medium = Medium.make("tarot") as TarotMedium
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)
	subs = preload("res://scripts/subtitles.gd").new()
	add_child(subs)
	medium.bind_captions(subs)
	var script := TarotScript.compose([{"kind": "shuffle", "card": 0, "text": "Shuffle the deck."},
		{"kind": "draw", "card": 1, "text": "One."}, {"kind": "spread", "card": 0, "text": "Done."}])
	var cards: Array = [{"key": "c0", "name": "Card", "numeral": "0", "reversed": false, "jumper": false,
		"position": {}, "booklet": {}, "art": ""}]
	var n := {"dur": 0.0, "mix": 0.0, "moved": 0.0, "path": 0.0, "flips": 0, "clips": 0, "jumps": 0, "ms": 0.0}
	for s in range(1, SEEDS + 1):
		var doc := {"show": "wash-check", "seed": s, "dir": "", "plan": {"look": {"candles": 2}}, "cards": cards}
		subs.document = {"source": script, "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		var run: Dictionary = TarotMedium.RUNS["wash"]
		var dur := lerpf(float(run["dur"][0]), float(run["dur"][1]), float(s) / float(SEEDS))
		var plan: Dictionary = medium._wash_plan(1000 + s, dur)
		var m := _measure(plan, dur)
		for k in m:
			n[k] = float(n[k]) + float(m[k]) / float(SEEDS) if k in ["dur", "mix", "moved", "path"] else int(n[k]) + int(m[k])
	# THE COST of the longest wash, the least of three: a loaded machine only ever adds to a timing
	var best := INF
	for r in 3:
		var t0 := Time.get_ticks_usec()
		medium._wash_plan(7777, float(TarotMedium.RUNS["wash"]["dur"][1]))
		best = minf(best, float(Time.get_ticks_usec() - t0) / 1000.0)
	n["ms"] = best
	print("wash_check: over %d washes - %.1f s long, %.1f s of it mixing; %.0f%% of the cards travel 10 cm or more, %.2f m a card;"
		% [SEEDS, n["dur"], n["mix"], float(n["moved"]) * 100.0, n["path"]])
	print("            %d changes of order between cards that met again; %d clips; %d jumps; the longest takes %.0f ms to plan"
		% [n["flips"], n["clips"], n["jumps"], n["ms"]])
	_ok(float(n["dur"]) >= 24.0, "a wash is long - at least twice the old 13 s (%.1f s)" % n["dur"])
	_ok(float(n["mix"]) >= 14.0, "most of it is the mixing (%.1f s)" % n["mix"])
	_ok(float(n["moved"]) >= 0.85, "the hands move most of the cards 10 cm or more (%.0f%%)" % (float(n["moved"]) * 100.0))
	_ok(float(n["path"]) >= 0.35, "and far (%.2f m a card through the mixing)" % n["path"])
	_ok(int(n["flips"]) >= SEEDS * 8, "the order changes between cards that meet again (%d)" % n["flips"])
	_ok(int(n["clips"]) == 0, "never while they touch: no card through a card (%d)" % n["clips"])
	_ok(int(n["jumps"]) == 0, "no card jumps (%d)" % n["jumps"])
	_ok(float(n["ms"]) < 700.0, "a plan is cheap enough to make as an episode loads (%.0f ms)" % n["ms"])
	_cut_short()
	_jumper()
	Director.hold(false)
	Director.detach()
	print("tarot_wash_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	get_tree().quit(0 if _fails == 0 else 1)


## A WASH CUT SHORT by the first card: the same wash up to its own gather (every card where the
## whole one had it - nothing jumps at the switch), gathered and squared by its end, and cheap,
## because it is made while the reading plays.
func _cut_short() -> void:
	var full: Dictionary = medium._wash_plan(4242, 30.0)
	var t0 := Time.get_ticks_usec()
	var cut: Dictionary = medium._wash_plan(4242, 17.0, full)
	var ms := float(Time.get_ticks_usec() - t0) / 1000.0
	var g := int(floor(float((cut["mix"] as Vector2).y) * TarotMedium.WASH_HZ))
	var same := true
	for i in (full["tracks"] as Array).size():
		var a: PackedVector4Array = (full["tracks"] as Array)[i]
		var b: PackedVector4Array = (cut["tracks"] as Array)[i]
		for st in g:
			if not a[st].is_equal_approx(b[st]):
				same = false
	# no card jumps where it turns to gathering: a step at the plan's rate is a hand's speed at most
	var worst := 0.0
	for i in (cut["tracks"] as Array).size():
		var b: PackedVector4Array = (cut["tracks"] as Array)[i]
		for st in range(maxi(1, g - 10), mini(b.size(), g + 10)):
			worst = maxf(worst, Vector2(b[st].x - b[st - 1].x, b[st].z - b[st - 1].z).length())
	var last: PackedVector4Array = (cut["tracks"] as Array)[0]
	_ok(same, "a wash cut short is the same wash up to its gather")
	_ok(worst <= 0.6 / TarotMedium.WASH_HZ, "and nothing jumps where it turns to gathering (%.1f mm a step at most)" % (worst * 1000.0))
	_ok(last.size() == int(ceil(17.0 * TarotMedium.WASH_HZ)) + 1, "it ends when the room does (%d steps)" % last.size())
	_ok(ms < 250.0, "and is cheap to make mid-reading (%.0f ms)" % ms)
	# too little room for a wash at all: the deck stays squared rather than snapping a spread back
	var m := {"t0": 10.0, "dur": 30.0, "seed": 4242, "plan": full}
	medium._shuffle_room = 10.0 + TarotMedium.WASH_ROOM - 1.0
	_ok(medium._wash_fit(m).is_empty() and medium._wash(3, 2.0, m).is_equal_approx(medium._rest_xf(3)),
		"with no room to wash, the deck waits squared")
	medium._shuffle_room = 10.0 + 17.0
	_ok(not medium._wash_fit(m).is_empty() and float(m.get("cut_dur", 0.0)) == 17.0, "with room, it is cut to fit")
	medium._shuffle_room = INF


## A JUMPER SPRINGS OUT OF A RIFFLE. The shuffle is made to go quiet long before the jumper (one
## riffle, then nothing), so the deck is squared and still when the jumper's action begins - which
## is exactly when the card used to leave it. At the spring the deck must be riffling, the card off
## the top of a half rather than the resting deck, and the deck square again once it lands.
func _jumper() -> void:
	var script := TarotScript.compose([{"kind": "shuffle", "card": 0, "text": "Shuffle the deck, and talk a while about nothing much at all."},
		{"kind": "jumper", "card": 1, "text": "Oh, a jumper."}, {"kind": "spread", "card": 0, "text": "Done."}])
	var cards: Array = [{"key": "c0", "name": "Card", "numeral": "0", "reversed": false, "jumper": true,
		"position": {}, "booklet": {}, "art": ""}]
	var doc := {"show": "wash-check", "seed": 77, "dir": "", "plan": {"look": {"candles": 2}}, "cards": cards}
	subs.words = load("res://tests/tarot_look_probe.gd").timeline(TarotScript.parse(script), 0.36, Director.intro_hold)
	subs.document = {"source": script, "title": "Wash Check", "tarot": doc}
	medium._ensure_doc()
	medium._follow.extend(subs.words)
	medium._sched = medium._follow.place(medium._parse["actions"], maxf(Director.intro_hold, 0.6), TarotMedium.LEAD, TarotMedium.TAIL)
	medium._moves = [{"kind": "riffle", "t0": 0.15, "dur": 2.4, "pause": 1000.0, "seed": 1}]   # then quiet
	var tm := medium._times()
	var first: Array = tm["first"]
	_ok(not first.is_empty() and String(first[2]) == "jumper", "the reading opens on a jumper")
	if first.is_empty():
		return
	var te := float(first[0])
	var sc := float(first[1])
	var moving := func(t: float) -> int:
		medium._now = t
		medium._pose(t)
		var n := 0
		for i in TarotMedium.DECK_N:
			var at: Transform3D = (medium._deck[i] as MeshInstance3D).transform
			var rest := medium._rest_xf(i)
			n += 1 if at.origin.distance_to(rest.origin) > 0.004 or not at.basis.is_equal_approx(rest.basis) else 0
		return n
	var before: int = moving.call(te - 0.3)
	var at_spring: int = moving.call(te + TarotMedium.JUMP_FLY.x * sc)
	var card: Transform3D = (medium._cards[0] as MeshInstance3D).transform
	var lift := card.origin.distance_to(medium._deck_top_xf(0).origin)
	var after: int = moving.call(te + (TarotMedium.JUMP_RIFFLE + 0.05) * sc)
	_ok(before == 0, "control: the shuffle had gone quiet before the jumper (%d of %d cards moving)" % [before, TarotMedium.DECK_N])
	_ok(at_spring >= 10, "when the card springs, the deck is riffling (%d of %d cards in the riffle)" % [at_spring, TarotMedium.DECK_N])
	_ok(lift > 0.02, "and it springs off a half in the air, not the resting deck (%.0f mm from it)" % (lift * 1000.0))
	_ok(after == 0, "the riffle done, the deck is square again (%d moving)" % after)


## What one wash's tracks do: through its MIXING (the plan's own `mix` window, or between the
## spreading's end and the gathering for a plan that does not say), how far each card goes, which
## lies over which whenever two meet, and whether two touching cards ever swap or share a height.
func _measure(plan: Dictionary, dur: float) -> Dictionary:
	var tracks: Array = plan["tracks"]
	var nc := tracks.size()
	var mix: Vector2 = plan.get("mix", Vector2(2.4, dur - 0.6 - clampf(dur * 0.45, 5.0, 7.0)))
	var a := int(ceil((mix.x + 0.5) * TarotMedium.WASH_HZ))
	var b := int(floor(mix.y * TarotMedium.WASH_HZ))
	var path := PackedFloat32Array()
	path.resize(nc)
	var jumps := 0
	var clips := 0
	var flips := 0
	var met := {}           # pair -> who lay on top when they last met (+1 the lower index, -1 the higher)
	var touching := {}      # pair -> touching at the step before
	for st in range(a, b + 1):
		for i in nc:
			var p: Vector4 = (tracks[i] as PackedVector4Array)[st]
			var q: Vector4 = (tracks[i] as PackedVector4Array)[st - 1]
			var d := Vector2(p.x - q.x, p.z - q.z).length()
			path[i] += d
			if d > 0.6 / TarotMedium.WASH_HZ:
				jumps += 1
		for i in nc:
			var pi_: Vector4 = (tracks[i] as PackedVector4Array)[st]
			for j in range(i + 1, nc):
				var pj: Vector4 = (tracks[j] as PackedVector4Array)[st]
				var key := i * 64 + j
				var over := TarotMedium._cards_overlap(Vector2(pi_.x, pi_.z), pi_.w, Vector2(pj.x, pj.z), pj.w)
				if over:
					var top := 1 if pi_.y > pj.y else -1
					if bool(touching.get(key, false)):
						var qi: Vector4 = (tracks[i] as PackedVector4Array)[st - 1]
						var qj: Vector4 = (tracks[j] as PackedVector4Array)[st - 1]
						if (1 if qi.y > qj.y else -1) != top or absf(pi_.y - pj.y) < TarotMedium.CARD_T * 0.9:
							clips += 1
					elif met.has(key) and int(met[key]) != top:
						flips += 1
					met[key] = top
				touching[key] = over
	var moved := 0
	var total := 0.0
	for i in nc:
		moved += 1 if path[i] >= 0.1 else 0
		total += path[i]
	return {"dur": dur, "mix": mix.y - mix.x, "moved": float(moved) / float(nc), "path": total / float(nc),
		"flips": flips, "clips": clips, "jumps": jumps}
