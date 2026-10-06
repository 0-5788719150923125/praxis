extends Node

## THE WASH SHUFFLES. Reported 2026-10-05: "When cards are splayed-out, they are BARELY shuffled at
## all. 3 or 4 cards might shift slightly, and then we start collecting them again... no changing
## of z-order (which should only happen when cards are not touching, to prevent clipping)... 5 or
## 10 seconds long, max - when in the real world, that kind of shuffle should take at least twice
## that long". Measured over many planned washes, straight off their tracks:
##
##   - it is long, and most of it is the mixing (not the spreading or the gathering);
##   - the hands move the cards far - a card's path through the mixing;
##
## Reported again the same day: "cards barely move... the movements are very small, very
## localized... It would be much more common for cards to sweep back, and forth, back, and forth in
## various directions, crossing large regions of the table, creating chaos along their path. Today,
## these sort of just shift in tiny little circles". The path above was long enough - round and
## round a 6 cm circle. So, besides:
##
##   - a card goes in SWEEPS: half its travel is in straight runs (a run ends where the card's way
##     bends - its chord falls short of nine tenths of its path) at least SWEEP long - the old
##     wash's was 4 cm - and most of it in runs of 10 cm or more (the old wash: 11%);
##   - the sweeps CROSS THE SPREAD: most cards get 15 cm or more from where the mixing found them
##     (the old wash: a quarter);
##   - and leave CHAOS ALONG THEIR PATH: a card lying still that a sweeping card runs over is
##     knocked - moved 1.5 cm or turned 9 degrees within half a second - often (the old wash: a
##     fifth, of the few cards it ever ran over);
##   - the measure can tell: cards made to go round tiny circles fail all three - and the circling
##     wash itself, put through this measure, had 0.39 m of path and failed the rest (4.2 cm runs,
##     11% long, 24% reach, 20% knocked);
##   - the stacking order CHANGES: two cards that lay one way when they met lie the other way at a
##     later meeting - and NEVER while they touch: two cards overlapping from one step to the next
##     keep their order and a card's thickness between them (that would be a card through a card);
##   - no card jumps: its step at the plan's rate is a hand's speed at most (HAND_SPEED);
##   - and the plan is cheap enough to make at an episode's load;
##   - a JUMPER flies out of a shuffle: when it springs the deck is mid-riffle, even when the
##     shuffle before it had gone quiet ("not when the cards are just sitting there").
##
##   tests/run_boot_probe.sh tests/tarot_wash_check.gd 180
##
## A BOOT probe (the medium reaches the Director); no GPU - the tracks are numbers.

const SEEDS := 12
## Half a card's travel is in straight runs at least this long (meters).
const SWEEP := 0.12
## A card going this fast (m/s) is sweeping; one going slower than STILL is lying still.
const FAST := 0.3
const STILL := 0.01

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
	var n := {"dur": 0.0, "mix": 0.0, "path": 0.0, "sweep": 0.0, "long": 0.0, "reach": 0.0, "knocked": 0.0,
		"flips": 0, "clips": 0, "jumps": 0, "ms": 0.0}
	for s in range(1, SEEDS + 1):
		var doc := {"show": "wash-check", "seed": s, "dir": "", "plan": {"look": {"candles": 2}}, "cards": cards}
		subs.document = {"source": script, "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		var run: Dictionary = TarotMedium.RUNS["wash"]
		var dur := lerpf(float(run["dur"][0]), float(run["dur"][1]), float(s) / float(SEEDS))
		var plan: Dictionary = medium._wash_plan(1000 + s, dur)
		var m := _measure(plan, dur)
		for k in m:
			n[k] = float(n[k]) + float(m[k]) / float(SEEDS) if k in ["dur", "mix", "path", "sweep", "long", "reach", "knocked"] \
				else int(n[k]) + int(m[k])
	# THE COST of the longest wash, the least of three: a loaded machine only ever adds to a timing
	var best := INF
	for r in 3:
		var t0 := Time.get_ticks_usec()
		medium._wash_plan(7777, float(TarotMedium.RUNS["wash"]["dur"][1]))
		best = minf(best, float(Time.get_ticks_usec() - t0) / 1000.0)
	n["ms"] = best
	print("wash_check: over %d washes - %.1f s long, %.1f s of it mixing; %.2f m a card, half of it in runs of %.0f cm or more, %.0f%% in runs of 10 cm or more;"
		% [SEEDS, n["dur"], n["mix"], n["path"], float(n["sweep"]) * 100.0, float(n["long"]) * 100.0])
	print("            %.0f%% of the cards go 15 cm or more from where the mixing found them; %.0f%% of the still cards a sweep runs over are knocked"
		% [float(n["reach"]) * 100.0, float(n["knocked"]) * 100.0])
	print("            %d changes of order between cards that met again; %d clips; %d jumps; the longest takes %.0f ms to plan"
		% [n["flips"], n["clips"], n["jumps"], n["ms"]])
	_ok(float(n["dur"]) >= 24.0, "a wash is long - at least twice the old 13 s (%.1f s)" % n["dur"])
	_ok(float(n["mix"]) >= 14.0, "most of it is the mixing (%.1f s)" % n["mix"])
	_ok(float(n["path"]) >= 0.35, "the hands move the cards far (%.2f m a card through the mixing)" % n["path"])
	_ok(float(n["sweep"]) >= SWEEP, "in sweeps, not circles: half a card's travel in straight runs of %.0f cm or more (%.1f cm)"
		% [SWEEP * 100.0, float(n["sweep"]) * 100.0])
	_ok(float(n["long"]) >= 0.55, "most of it in runs of 10 cm or more (%.0f%%)" % (float(n["long"]) * 100.0))
	_ok(float(n["reach"]) >= 0.7, "crossing the spread: most cards go 15 cm or more from where the mixing found them (%.0f%%)"
		% (float(n["reach"]) * 100.0))
	_ok(float(n["knocked"]) >= 0.35, "and the cards a sweep runs over are knocked askew (%.0f%%)" % (float(n["knocked"]) * 100.0))
	_circles()
	_ok(int(n["flips"]) >= SEEDS * 8, "the order changes between cards that meet again (%d)" % n["flips"])
	_ok(int(n["clips"]) == 0, "never while they touch: no card through a card (%d)" % n["clips"])
	_ok(int(n["jumps"]) == 0, "no card jumps (%d)" % n["jumps"])
	_ok(float(n["ms"]) < 700.0, "a plan is cheap enough to make as an episode loads (%.0f ms)" % n["ms"])
	_cut_short()
	_jumper()
	_looks()
	_resting()
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
	_ok(worst <= TarotMedium.HAND_SPEED / TarotMedium.WASH_HZ, "and nothing jumps where it turns to gathering (%.1f mm a step at most)" % (worst * 1000.0))
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


## CARDS IN A WASH REST ON ONE ANOTHER (feedback 0007: "many of these cards are lifted off of the
## table itself... they cannot rest upon each other with a gentle tilt"). Over the mixing of many
## washes, every card spread on the cloth passes through no card under it and not into the cloth
## (its plane at every corner and at every point where it crosses a card under it), and lies AS LOW
## AS WHAT IS UNDER IT LETS IT - its middle exactly as high as the lowest plane over all of those
## points, found here a second, independent way (the highest point any three of them, or two, or
## one, hold up over the middle). Two-sided: the plan's flat layers sit above that, held up in the air.
func _resting() -> void:
	var through := 0
	var afloat := 0
	var cards := 0
	var bare_new := 0.0
	var bare_old := 0.0
	var bare_n := 0
	var tipped := 0
	var worst_ms := 0.0
	var judged := 0
	var off_low := 0
	var flat_high := 0
	for s in range(1, 7):
		var doc := {"show": "wash-check", "seed": s, "dir": "", "plan": {"look": {"candles": 2}}, "cards": []}
		subs.document = {"source": TarotScript.compose([{"kind": "shuffle", "card": 0, "text": "Shuffle."}]), "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		var plan: Dictionary = medium._wash_plan(3000 + s, 30.0)
		var mix: Vector2 = plan["mix"]
		for st in range(int(mix.x * TarotMedium.WASH_HZ) + 5, int(mix.y * TarotMedium.WASH_HZ), 11):
			var t0 := Time.get_ticks_usec()
			var rest: Array = medium._wash_rest(plan, st)
			worst_ms = maxf(worst_ms, float(Time.get_ticks_usec() - t0) / 1000.0)
			var v := float(st) / TarotMedium.WASH_HZ
			var on: Array = []
			for i in rest.size():
				if medium._wash_spread_at(plan, i, v) >= 0.999:
					on.append(i)
			for i in on:
				var qi: Vector4 = ((plan["tracks"] as Array)[i] as PackedVector4Array)[st]
				var mid := Vector2(qi.x, qi.z)
				var r: Vector3 = rest[i]
				var g := Vector2(r.y, r.z)
				var corners := TarotMedium._card_corners(mid, qi.w)
				var least := INF
				cards += 1
				tipped += 1 if g.length() > 0.002 else 0
				var covered := [false, false, false, false]
				var pts := PackedVector2Array()
				var hs := PackedFloat32Array()
				for c in 4:
					var gap := r.x + g.dot(corners[c] - mid) - TarotMedium.WASH_FLOOR
					through += 1 if gap < -1e-5 else 0
					least = minf(least, gap)
					pts.append(corners[c] - mid)
					hs.append(TarotMedium.WASH_FLOOR)
				for j in on:
					var qj: Vector4 = ((plan["tracks"] as Array)[j] as PackedVector4Array)[st]
					if qj.y >= qi.y:
						continue
					var mj := Vector2(qj.x, qj.z)
					var rj: Vector3 = rest[j]
					var cj := TarotMedium._card_corners(mj, qj.w)
					for piece in Geometry2D.intersect_polygons(corners, cj):
						for p in piece:
							var under := rj.x + Vector2(rj.y, rj.z).dot(p - mj) + TarotMedium.CARD_T + TarotMedium.STACK_GAP
							var gap := (r.x + g.dot(p - mid)) - under
							through += 1 if gap < -1e-5 else 0
							least = minf(least, gap)
							pts.append(p - mid)
							hs.append(under)
					for c in 4:
						if Geometry2D.is_point_in_polygon(corners[c], cj):
							covered[c] = true
				afloat += 1 if least > 1e-5 else 0
				if pts.size() > 4 and cards % 3 == 0:
					var lowest := _held_up(pts, hs)
					judged += 1
					off_low += 1 if absf(r.x - lowest) > 2e-6 else 0
					flat_high += 1 if qi.y > lowest + 0.0001 else 0
				for c in 4:
					if not covered[c]:
						bare_new += r.x + g.dot(corners[c] - mid) - TarotMedium.WASH_FLOOR
						bare_old += qi.y - TarotMedium.WASH_FLOOR
						bare_n += 1
	var mean_new := bare_new / maxf(float(bare_n), 1.0) * 1000.0
	var mean_old := bare_old / maxf(float(bare_n), 1.0) * 1000.0
	print("            resting: %d cards over 6 washes, %d tipped; corners over bare cloth %.2f mm up on average (flat layers: %.2f mm); the slowest step %.1f ms"
		% [cards, tipped, mean_new, mean_old, worst_ms])
	_ok(through == 0, "no card in a wash passes through a card under it or into the cloth (%d points)" % through)
	_ok(afloat == 0, "every card rests on something (%d of %d in the air)" % [afloat, cards])
	_ok(judged > 200 and off_low == 0, "every card lies as low as the cards under it let it (%d of %d judged do not)" % [off_low, judged])
	_ok(flat_high > judged / 3, "control: on the plan's flat layers %d of %d sit higher than that" % [flat_high, judged])
	_ok(mean_new < mean_old, "corners over bare cloth come down toward it: %.2f mm up, against %.2f mm on flat layers" % [mean_new, mean_old])
	_ok(tipped > cards / 10, "cards resting on others tip (%d of %d)" % [tipped, cards])
	_ok(worst_ms < 25.0, "a step of rest is cheap enough to make as it plays (%.1f ms)" % worst_ms)


## The lowest plane above every support [param hs] at [param pts] (about a card's middle), at the
## middle - found as the highest point any three supports (or two, or one) hold up over the middle,
## the other way round from [method TarotMedium._rest_on], so the two cannot share a mistake.
func _held_up(pts: PackedVector2Array, hs: PackedFloat32Array) -> float:
	var best := -INF
	var n := pts.size()
	for i in n:
		if pts[i].length() < 1e-7:
			best = maxf(best, hs[i])
		for j in range(i + 1, n):
			var d := pts[j] - pts[i]
			if d.length() > 1e-9 and absf(d.cross(-pts[i])) < 1e-12:
				var u := (-pts[i]).dot(d) / d.length_squared()
				if u >= 0.0 and u <= 1.0:
					best = maxf(best, lerpf(hs[i], hs[j], u))
			for k in range(j + 1, n):
				var area := (pts[j] - pts[i]).cross(pts[k] - pts[i])
				if absf(area) < 1e-12:
					continue
				var l0 := pts[j].cross(pts[k]) / area
				var l1 := pts[k].cross(pts[i]) / area
				var l2 := pts[i].cross(pts[j]) / area
				if l0 >= -1e-9 and l1 >= -1e-9 and l2 >= -1e-9:
					best = maxf(best, l0 * hs[i] + l1 * hs[j] + l2 * hs[k])
	return best


## THE HELD CARD'S LOOKS AT ITS BACK. Reported 2026-10-05: the hold "seems to be the exact same
## length, always" (1.1-1.8 s); and asked for, a PIROUETTE - "rather than just reversing direction
## back to the front... KEEP GOING in the same direction... such that 3 whole rotations are made in
## the same direction (but we hold on the back artwork for the first 180, still)". So: the holds
## spread wide - two-sided, the retired range does not; some looks are pirouettes and most are not;
## a pirouette turns over, holds its back, then turns ONE way on round to face on three whole turns
## from where it began; a plain look comes back the way it went; and over long held stretches the
## card never jumps, a look's start and end included, and never turns as it goes down. Then (the
## user's, 2026-10-05: "4-5 times for a single draw is far too many. I would expect the baseline to
## be 0", and a card twirled 3 times in one draw): most held cards are never turned, few more than
## once, pirouettes are rare and never twice on one card.
func _looks() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 5
	var holds: Array = []
	var spins := 0
	var bad_spin := 0
	var bad_plain := 0
	var draws := 600
	for i in draws:
		var look := TarotMedium._look_of(rng)
		var way := float(look["way"])
		var turn := float(look["turn"])
		var hold := float(look["hold"])
		var total := float(look["total"])
		var spin := float(look["twirl"]) > 0.0
		if spin:
			spins += 1
		else:
			holds.append(hold)
		# along the look: does it ever turn against its way, and where does it end
		var against := false
		var prev := 0.0
		for s in 600:
			var a := TarotMedium._look_angle(look, total * float(s + 1) / 600.0)
			if (a - prev) * way < -1e-5:
				against = true
			prev = a
		var on_back := absf(TarotMedium._look_angle(look, turn + hold * 0.5) - way * PI) < 1e-4
		var end := TarotMedium._look_angle(look, total)
		if spin:
			bad_spin += 0 if (not against and on_back and absf(end - way * 6.0 * PI) < 1e-3) else 1
		else:
			bad_plain += 0 if (against and on_back and absf(end) < 1e-3) else 1
	holds.sort()
	var spread := float(holds[int(holds.size() * 0.9)]) / float(holds[int(holds.size() * 0.1)])
	var old: Array = []
	for i in 600:
		old.append(rng.randf_range(1.1, 1.8))
	old.sort()
	var old_spread := float(old[540]) / float(old[60])
	_ok(spread >= 3.0, "a look's hold varies: the long ones %.1fx the short (%.1f-%.1f s)" % [spread, holds[int(holds.size() * 0.1)], holds[int(holds.size() * 0.9)]])
	_ok(old_spread < 3.0, "control: the retired holds (1.1-1.8 s) vary %.1fx" % old_spread)
	var share := float(spins) / float(draws)
	_ok(share > 0.03 and share < 0.2, "a few looks are pirouettes, most are plain half turns (%.0f%%)" % (share * 100.0))
	_ok(bad_spin == 0, "a pirouette holds its back, then turns one way to face on three whole turns round (%d did not)" % bad_spin)
	_ok(bad_plain == 0, "a plain look holds its back and comes back the way it went (%d did not)" % bad_plain)
	# OVER LONG HELD STRETCHES: never a jump (a pose a whole turn round is the same pose), the
	# twirl seen turning, and nothing as the card goes down
	var worst := 0.0
	var round_seen := 0
	var going_down := 0
	var never := 0
	var many := 0
	var twice_spun := 0
	var cards := 160
	for k in cards:
		var until := 60.0 + 0.75 * float(k)
		var prev := 0.0
		var looks := 0
		var spins_k := 0
		var turned := false
		var past := false
		for f in int(until * 30.0):
			var t := float(f) / 30.0
			var a := medium._turn_of(k, t, 0.0, until)
			worst = maxf(worst, absf(wrapf(a - prev, -PI, PI)))
			round_seen += 1 if absf(a) > TAU else 0
			if t > until - 1.0 and absf(a) > 1e-6:
				going_down += 1
			var now_turned := absf(a) > 1e-6
			if now_turned and not turned:
				looks += 1
				past = false
			if absf(a) > TAU and not past:
				spins_k += 1
				past = true
			turned = now_turned
			prev = a
		never += 1 if looks == 0 else 0
		many += 1 if looks > 2 else 0
		twice_spun += 1 if spins_k > 1 else 0
	_ok(worst < 0.8, "a held card's turning never jumps (%.2f rad a frame at most)" % worst)
	_ok(round_seen > 0, "a pirouette is seen in long held stretches (%d frames past a whole turn)" % round_seen)
	_ok(going_down == 0, "and no card turns as it goes down (%d frames)" % going_down)
	_ok(never > cards * 0.6, "most held cards are never turned (%d of %d)" % [never, cards])
	_ok(many <= cards / 40, "hardly any card looks at its back more than twice (%d of %d)" % [many, cards])
	_ok(twice_spun == 0, "no card pirouettes twice (%d did)" % twice_spun)
	# the measure can see a snap: a pirouette half a turn short snaps from its back to its face
	var short := absf(wrapf(5.0 * PI - 0.0, -PI, PI))
	_ok(short > 3.0, "control: a twirl half a turn short is a snap of %.2f rad" % short)


## What one wash's tracks do: through its MIXING (the plan's own `mix` window, or between the
## spreading's end and the gathering for a plan that does not say), how far each card goes and in
## what runs, how far from where the mixing found it, which lies over which whenever two meet,
## whether two touching cards ever swap or share a height, and what a sweeping card does to the
## still cards it runs over.
func _measure(plan: Dictionary, dur: float) -> Dictionary:
	var tracks: Array = plan["tracks"]
	var nc := tracks.size()
	var mix: Vector2 = plan.get("mix", Vector2(2.4, dur - 0.6 - clampf(dur * 0.45, 5.0, 7.0)))
	var a := int(ceil((mix.x + 0.5) * TarotMedium.WASH_HZ))
	var b := int(floor(mix.y * TarotMedium.WASH_HZ))
	var half := int(0.5 * TarotMedium.WASH_HZ)
	var path := PackedFloat32Array()
	var reach := PackedFloat32Array()
	var speed := PackedFloat32Array()
	var run0: Array = []        # where each card's straight run began
	var walked := PackedFloat32Array()    # and how far it has gone since
	var runs := PackedFloat32Array()
	path.resize(nc)
	reach.resize(nc)
	speed.resize(nc)
	walked.resize(nc)
	for i in nc:
		var p0: Vector4 = (tracks[i] as PackedVector4Array)[a - 1]
		run0.append(Vector2(p0.x, p0.z))
	var jumps := 0
	var clips := 0
	var flips := 0
	var met := {}           # pair -> who lay on top when they last met (+1 the lower index, -1 the higher)
	var touching := {}      # pair -> touching at the step before
	var knocks: Array = []  # [card, step]: a still card a sweeping card has just run over
	for st in range(a, b + 1):
		for i in nc:
			var p: Vector4 = (tracks[i] as PackedVector4Array)[st]
			var q: Vector4 = (tracks[i] as PackedVector4Array)[st - 1]
			var d := Vector2(p.x - q.x, p.z - q.z).length()
			path[i] += d
			speed[i] = d * TarotMedium.WASH_HZ
			if d > TarotMedium.HAND_SPEED / TarotMedium.WASH_HZ:
				jumps += 1
			# a run ends where the card's way stops being straight - turning back, or round a bend
			var r0: Vector2 = run0[i]
			if walked[i] > 0.005 and Vector2(p.x, p.z).distance_to(r0) < 0.9 * (walked[i] + d):
				runs.append(Vector2(q.x, q.z).distance_to(r0))
				run0[i] = Vector2(q.x, q.z)
				walked[i] = 0.0
			walked[i] += d
			var p0: Vector4 = (tracks[i] as PackedVector4Array)[a - 1]
			reach[i] = maxf(reach[i], Vector2(p.x - p0.x, p.z - p0.z).length())
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
					else:
						if met.has(key) and int(met[key]) != top:
							flips += 1
						# just run over: one card sweeping, the other lying still
						if st + half <= b:
							if speed[i] >= FAST and speed[j] < STILL:
								knocks.append([j, st])
							elif speed[j] >= FAST and speed[i] < STILL:
								knocks.append([i, st])
					met[key] = top
				touching[key] = over
	for i in nc:
		var r0: Vector2 = run0[i]
		var e: Vector4 = (tracks[i] as PackedVector4Array)[b]
		runs.append(Vector2(e.x, e.z).distance_to(r0))
	# HALF THE TRAVEL is in runs at least this long (the runs' length-weighted median)
	var sorted_runs := runs.duplicate()
	sorted_runs.sort()
	var total_runs := 0.0
	var long := 0.0
	for r in sorted_runs:
		total_runs += r
		long += r if r >= 0.1 else 0.0
	var sweep := 0.0
	var acc := 0.0
	for r in sorted_runs:
		acc += r
		if acc >= total_runs * 0.5:
			sweep = r
			break
	var knocked := 0
	for k in knocks:
		var c := int((k as Array)[0])
		var st := int((k as Array)[1])
		var p: Vector4 = (tracks[c] as PackedVector4Array)[st]
		var q: Vector4 = (tracks[c] as PackedVector4Array)[st + half]
		if Vector2(q.x - p.x, q.z - p.z).length() >= 0.015 or absf(q.w - p.w) >= deg_to_rad(9.0):
			knocked += 1
	var far := 0
	var total := 0.0
	for i in nc:
		far += 1 if reach[i] >= 0.15 else 0
		total += path[i]
	return {"dur": dur, "mix": mix.y - mix.x, "path": total / float(nc), "sweep": sweep,
		"long": long / maxf(total_runs, 1e-6), "reach": float(far) / float(nc),
		"knocked": float(knocked) / float(maxi(knocks.size(), 1)), "flips": flips, "clips": clips, "jumps": jumps}


## THE MEASURE CAN TELL CIRCLES FROM SWEEPS: the cards of a wash made to go round tiny circles - a
## 6 cm palm's loop, as the old hands made them, and at their pace - fail every one of the checks
## above, however long their path.
func _circles() -> void:
	var r := RandomNumberGenerator.new()
	r.seed = 99
	var tracks: Array = []
	var steps := int(30.0 * TarotMedium.WASH_HZ) + 1
	for i in TarotMedium.DECK_N:
		var c := Vector2(r.randf_range(-0.22, 0.22), r.randf_range(-0.11, 0.09))
		var ph := r.randf() * TAU
		var om := r.randf_range(2.4, 4.0) * (1.0 if r.randf() < 0.5 else -1.0)
		var tr := PackedVector4Array()
		for st in steps:
			var t := float(st) / TarotMedium.WASH_HZ
			var a := ph + om * t
			var at := c + Vector2(cos(a), sin(a)) * 0.06
			tr.append(Vector4(at.x, TarotMedium.WASH_FLOOR, at.y, 0.0))
		tracks.append(tr)
	var m := _measure({"tracks": tracks, "mix": Vector2(2.4, 22.0)}, 30.0)
	_ok(float(m["path"]) >= 0.35 and float(m["sweep"]) < SWEEP and float(m["long"]) < 0.55 and float(m["reach"]) < 0.7
		and float(m["knocked"]) < 0.35, "control: tiny circles have the path (%.2f m) and fail the rest (runs %.1f cm, %.0f%% long, %.0f%% reach, %.0f%% knocked)"
		% [m["path"], float(m["sweep"]) * 100.0, float(m["long"]) * 100.0, float(m["reach"]) * 100.0, float(m["knocked"]) * 100.0])
