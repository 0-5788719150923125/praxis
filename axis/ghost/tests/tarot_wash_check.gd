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
## Reported 2026-10-06: the cards "'repel' each other in a way that makes them almost bouncy - they
## are very unstable and they tilt far too much... the cards often clip through each other... cards
## moving THROUGH each other, which should be impossible"; and asked: "when a lot of repulsion is
## applied, a card can get ejected, it can land face-up, and the reader can choose to draw that
## card... some ejections to land face-down, and thus NOT be drawn". So, posed as the camera sees
## it, frame by frame:
##
##   - no card passes through another or into the cloth - the deck spreading and the pile gathering
##     included - and two cards touching never swap which lies on top (control: posed as two steps'
##     rests blended, cards cross);
##   - cards lie gently: a mixing card tips a few degrees at most, and its tilt does not jump about;
##   - a meeting is a nudge: the card run into never overtakes the card that struck it, which never
##     comes back the way it went;
##   - a hard blow throws a card out of the spread now and then: struck faster than EJECT_SPEED,
##     face down, onto the cloth and wholly in the picture, its way clear of what stands;
##   - a JUMPER OUT OF A WASH: when the first card is a jumper and the deck is being washed as it
##     comes, a blow throws it over, face up and the right way round, clear of every card; it lies
##     there, nothing passing through it, until it is picked up and shown; the deck is gathered one
##     card short, then pushed aside. An episode that washes into its jumper holds its wash back for it.
##
##   tests/run_boot_probe.sh tests/tarot_wash_check.gd 420
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
	var plans: Array = []
	for s in range(1, SEEDS + 1):
		var doc := {"show": "wash-check", "seed": s, "dir": "", "plan": {"look": {"candles": 2}}, "cards": cards}
		subs.document = {"source": script, "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		var run: Dictionary = TarotMedium.RUNS["wash"]
		var dur := lerpf(float(run["dur"][0]), float(run["dur"][1]), float(s) / float(SEEDS))
		var plan: Dictionary = medium._wash_plan(1000 + s, dur)
		plans.append(plan)
		_throw_facts(plan)
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
	_nudge()
	_throws(plans)
	_resting()
	_posed()
	_wash_jumper()
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
			if TarotMedium._flight_of(cut, i, float(st) / TarotMedium.WASH_HZ).is_empty():
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
## table itself... they cannot rest upon each other with a gentle tilt"). At the plan's steps of many
## washes - the deck spreading and the pile gathering as well as the mixing - every card on the table
## passes through no card under it and not into the cloth (its plane at every corner and at every
## point where it crosses a card under it, each card its own thickness), and lies AS LOW AS WHAT IS
## UNDER IT LETS IT - its middle exactly as high as the lowest plane over all of those points, found
## here a second, independent way (the highest point any three of them, or two, or one, hold up over
## the middle). Two-sided: the plan's flat layers sit above that, held up in the air.
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
	var off_most := 0.0
	var flat_high := 0
	for s in range(1, 5):
		var doc := {"show": "wash-check", "seed": s, "dir": "", "plan": {"look": {"candles": 2}}, "cards": []}
		subs.document = {"source": TarotScript.compose([{"kind": "shuffle", "card": 0, "text": "Shuffle."}]), "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		var plan: Dictionary = medium._wash_plan(3000 + s, 30.0)
		var tracks: Array = plan["tracks"]
		var last := (tracks[0] as PackedVector4Array).size() - 1
		for st in range(1, last, 7):
			var t0 := Time.get_ticks_usec()
			var rest: Array = medium._wash_rest(plan, st)
			worst_ms = maxf(worst_ms, float(Time.get_ticks_usec() - t0) / 1000.0)
			var v := float(st) / TarotMedium.WASH_HZ
			var on: Array = []
			for i in rest.size():
				if TarotMedium._flight_of(plan, i, v).is_empty():
					on.append(i)
			for i in on:
				var qi: Vector4 = (tracks[i] as PackedVector4Array)[st]
				var si := medium._wash_spread_at(plan, i, v)
				var thi := lerpf(TarotMedium.DECK_T, TarotMedium.WASH_T, si)
				var low := lerpf(0.0, TarotMedium.CLOTH_TOP + TarotMedium.FLOOR_GAP, si) + thi * 0.5
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
					var gap := r.x + g.dot(corners[c] - mid) - low
					through += 1 if gap < -1e-5 else 0
					least = minf(least, gap)
					pts.append(corners[c] - mid)
					hs.append(low)
				for j in on:
					var qj: Vector4 = (tracks[j] as PackedVector4Array)[st]
					if qj.y > qi.y or (qj.y == qi.y and j >= i):
						continue
					var mj := Vector2(qj.x, qj.z)
					var rj: Vector3 = rest[j]
					var sj := medium._wash_spread_at(plan, j, v)
					var lift := lerpf(TarotMedium.DECK_T, TarotMedium.WASH_T, sj) * 0.5 + TarotMedium.STACK_GAP * maxf(si, sj) + thi * 0.5
					var cj := TarotMedium._card_corners(mj, qj.w)
					var pieces := Geometry2D.intersect_polygons(corners, cj)
					for piece in pieces:
						for p: Vector2 in piece:
							var under := rj.x + Vector2(rj.y, rj.z).dot(p - mj) + lift
							var gap := (r.x + g.dot(p - mid)) - under
							through += 1 if gap < -1e-5 else 0
							least = minf(least, gap)
					# what the table rests it on: where they cross - or, lying squared on it, the face
					# under it carried to its own corners
					if mj.distance_to(mid) < 0.006 and absf(wrapf(qi.w - qj.w, -PI, PI)) < 0.12:
						for c in 4:
							hs[c] = maxf(hs[c], rj.x + Vector2(rj.y, rj.z).dot(corners[c] - mj) + lift)
					else:
						for piece in pieces:
							for p: Vector2 in piece:
								pts.append(p - mid)
								hs.append(rj.x + Vector2(rj.y, rj.z).dot(p - mj) + lift)
					for c in 4:
						if Geometry2D.is_point_in_polygon(corners[c], cj):
							covered[c] = true
				var held := INF
				for k in hs.size():
					held = minf(held, (r.x + g.dot(pts[k])) - hs[k])
				afloat += 1 if held > 1e-5 else 0
				if pts.size() > 4 and cards % 3 == 0:
					var lowest := _held_up(pts, hs)
					judged += 1
					off_low += 1 if absf(r.x - lowest) > 2e-6 else 0
					off_most = maxf(off_most, absf(r.x - lowest))
					flat_high += 1 if qi.y > lowest + 0.0001 else 0
				if si > 0.99:
					for c in 4:
						if not covered[c]:
							bare_new += r.x + g.dot(corners[c] - mid) - low
							bare_old += qi.y - low
							bare_n += 1
	var mean_new := bare_new / maxf(float(bare_n), 1.0) * 1000.0
	var mean_old := bare_old / maxf(float(bare_n), 1.0) * 1000.0
	print("            resting: %d cards over 4 washes, %d tipped; corners over bare cloth %.2f mm up on average (flat layers: %.2f mm); the slowest step %.1f ms"
		% [cards, tipped, mean_new, mean_old, worst_ms])
	_ok(through == 0, "no card in a wash passes through a card under it or into the cloth (%d points)" % through)
	_ok(afloat == 0, "every card rests on something (%d of %d in the air)" % [afloat, cards])
	_ok(judged > 200 and off_low == 0, "every card lies as low as the cards under it let it (%d of %d judged do not, by %.4f mm at most)" % [off_low, judged, off_most * 1000.0])
	_ok(flat_high > judged / 3, "control: on the plan's flat layers %d of %d sit higher than that" % [flat_high, judged])
	_ok(mean_new < mean_old, "corners over bare cloth come down toward it: %.2f mm up, against %.2f mm on flat layers" % [mean_new, mean_old])
	_ok(tipped > cards / 10, "cards resting on others tip (%d of %d)" % [tipped, cards])
	_ok(worst_ms < 25.0, "a step of rest is cheap enough to make as it plays (%.1f ms)" % worst_ms)


## AS THE CAMERA SEES IT (2026-10-06: "the cards often clip through each other... cards moving
## THROUGH each other"; "they tilt far too much"): washes posed frame by frame, as the table poses
## them, between the plan's steps. Wherever two cards lie over one another their faces never cross,
## the upper never dips into the lower, and the pair never swaps which is on top from one frame to
## the next; no card on the cloth dips into it. Mixing cards tip a few degrees at most and their tilt
## does not jump about. Control: posed as two steps' rests blended, cards pass through one another.
func _posed() -> void:
	var fps := 15.0
	var crossings := 0
	var sinks := 0
	var swaps := 0
	var cloth := 0
	var mix_tilts: Array = []
	var jerks := 0
	var mix_frames := 0
	var worst := 0.0
	var blended := 0
	for s in range(1, 4):
		var doc := {"show": "wash-check", "seed": s, "dir": "", "plan": {"look": {"candles": 2}}, "cards": []}
		subs.document = {"source": TarotScript.compose([{"kind": "shuffle", "card": 0, "text": "Shuffle."}]), "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		medium._shuffle_room = INF
		medium._jump_room = -1.0
		var dur := 28.0
		var plan: Dictionary = medium._wash_plan(6000 + s, dur)
		var m := {"kind": "wash", "t0": 0.0, "dur": dur, "pause": 0.0, "seed": 6000 + s, "plan": plan}
		var mix: Vector2 = plan["mix"]
		var prev: Array = []
		var prev_top := {}
		var prev_up: Array = []
		for f in int(dur * fps):
			var v := float(f) / fps
			var xfs: Array = []
			for i in TarotMedium.DECK_N:
				xfs.append(medium._wash(i, v, m))
			var r := _overlaps(xfs, prev_top)
			crossings += int(r["crossings"])
			sinks += int(r["sinks"])
			swaps += int(r["swaps"])
			prev_top = r["tops"]
			var ups: Array = []
			for i in xfs.size():
				var xf: Transform3D = xfs[i]
				var up := xf.basis.y.normalized()
				ups.append(up)
				var aloft := not TarotMedium._flight_of(plan, i, v).is_empty()
				if aloft:
					continue
				var sp := medium._wash_spread_at(plan, i, v)
				if sp > 0.99 and _lowest(xf) < TarotMedium.CLOTH_TOP - 1e-5:
					cloth += 1
				if v > mix.x + 0.5 and v < mix.y:
					var tilt := rad_to_deg(acos(clampf(up.y, -1.0, 1.0)))
					mix_tilts.append(tilt)
					worst = maxf(worst, tilt)
					mix_frames += 1
					if not prev_up.is_empty():
						var turn := rad_to_deg(acos(clampf(up.dot(prev_up[i] as Vector3), -1.0, 1.0))) * fps
						jerks += 1 if turn > 60.0 else 0
			prev_up = ups
			# CONTROL: the same frame posed as two steps' rests blended
			if s == 1:
				var old: Array = []
				for i in TarotMedium.DECK_N:
					old.append(_blended(plan, m, i, v))
				blended += int(_overlaps(old, {})["crossings"])
			prev = xfs
	mix_tilts.sort()
	var p99 := float(mix_tilts[int(mix_tilts.size() * 0.99)]) if not mix_tilts.is_empty() else 0.0
	var jerk_share := float(jerks) / maxf(float(mix_frames), 1.0)
	print("            posed: %d faces crossing, %d dipping into a card, %d swaps, %d into the cloth; mixing tilt 99%% under %.1f deg, at most %.1f; %.2f%% of mixing frames turn faster than 60 deg/s"
		% [crossings, sinks, swaps, cloth, p99, worst, jerk_share * 100.0])
	_ok(crossings == 0 and sinks == 0, "no card passes through another, frame by frame (%d crossing, %d dipping in)" % [crossings, sinks])
	_ok(swaps == 0, "two cards touching never swap which is on top (%d)" % swaps)
	_ok(cloth == 0, "no card on the cloth dips into it (%d)" % cloth)
	_ok(blended > 0, "control: posed as two steps' rests blended, cards cross (%d)" % blended)
	_ok(p99 < 3.0 and worst < 8.0, "mixing cards lie gently: 99%% tip under %.1f deg, the most %.1f" % [p99, worst])
	_ok(jerk_share < 0.01, "and their tilt does not jump about (%.2f%% of frames turn faster than 60 deg/s)" % (jerk_share * 100.0))


## Over cards posed at [param xfs]: wherever two lie over one another, whether their faces cross
## (the lower's face above the upper's somewhere), whether the upper dips into the lower, and which
## is on top - and whether that swapped from [param before] (pair -> lower index on top).
func _overlaps(xfs: Array, before: Dictionary) -> Dictionary:
	var crossings := 0
	var sinks := 0
	var swaps := 0
	var tops := {}
	for i in xfs.size():
		var a: Transform3D = xfs[i]
		var fa := _foot(a)
		var ta := TarotMedium.DECK_T * a.basis.y.length()
		for j in range(i + 1, xfs.size()):
			var b: Transform3D = xfs[j]
			if Vector2(a.origin.x - b.origin.x, a.origin.z - b.origin.z).length() > 0.14:
				continue
			if absf(a.basis.y.normalized().y) < 0.5 or absf(b.basis.y.normalized().y) < 0.5:
				continue
			var pieces := Geometry2D.intersect_polygons(fa, _foot(b))
			if pieces.is_empty():
				continue
			var tb := TarotMedium.DECK_T * b.basis.y.length()
			var hi := -INF
			var lo := INF
			var area := 0.0
			var mid := Vector2.ZERO
			var cnt := 0
			for piece in pieces:
				var pp: PackedVector2Array = piece
				for k in pp.size():
					area += pp[k].cross(pp[(k + 1) % pp.size()]) * 0.5
				for p: Vector2 in pp:
					var d := (_plane_y(a, p) + ta * 0.5) - (_plane_y(b, p) + tb * 0.5)
					hi = maxf(hi, d)
					lo = minf(lo, d)
					mid += p
					cnt += 1
			if absf(area) < 2e-5:
				continue
			mid /= float(cnt)
			var a_top := _plane_y(a, mid) > _plane_y(b, mid)
			tops[i * 64 + j] = a_top
			if hi > 0.0001 and lo < -0.0001:
				crossings += 1
			var dip := INF
			for piece in pieces:
				for p: Vector2 in piece:
					dip = minf(dip, (_plane_y(a, p) - ta * 0.5) - (_plane_y(b, p) + tb * 0.5) if a_top
						else (_plane_y(b, p) - tb * 0.5) - (_plane_y(a, p) + ta * 0.5))
			sinks += 1 if dip < -0.00002 else 0
			if before.has(i * 64 + j) and bool(before[i * 64 + j]) != a_top:
				swaps += 1
	return {"crossings": crossings, "sinks": sinks, "swaps": swaps, "tops": tops}


## A posed card's middle plane at [param p] (x by z).
static func _plane_y(xf: Transform3D, p: Vector2) -> float:
	var n := xf.basis.y.normalized()
	return xf.origin.y - (n.x * (p.x - xf.origin.x) + n.z * (p.y - xf.origin.z)) / maxf(absf(n.y), 1e-4) * signf(n.y)


## A posed card's four corners on the table (x by z).
static func _foot(xf: Transform3D) -> PackedVector2Array:
	var out := PackedVector2Array()
	for sv in [Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)]:
		var s: Vector2 = sv
		var p: Vector3 = xf.origin + xf.basis.x * (TarotMedium.CARD.x * 0.5 * s.x) + xf.basis.z * (TarotMedium.CARD.y * 0.5 * s.y)
		out.append(Vector2(p.x, p.z))
	return out


## The lowest point of a posed card.
static func _lowest(xf: Transform3D) -> float:
	var b := xf.basis
	return xf.origin.y - absf(b.x.y) * TarotMedium.CARD.x * 0.5 - absf(b.y.y) * TarotMedium.DECK_T * 0.5 - absf(b.z.y) * TarotMedium.CARD.y * 0.5


## CONTROL: card [param i] posed as two steps' rests blended - what the table did before posing each
## frame's own rest.
func _blended(plan: Dictionary, m: Dictionary, i: int, v: float) -> Transform3D:
	var track: PackedVector4Array = (plan["tracks"] as Array)[i]
	var f := clampf(v * TarotMedium.WASH_HZ, 0.0, float(track.size() - 1))
	var a := int(floor(f))
	var b := mini(a + 1, track.size() - 1)
	var q := track[a].lerp(track[b], f - float(a))
	var th := lerpf(TarotMedium.DECK_T, TarotMedium.WASH_T, medium._wash_spread_at(plan, i, v))
	var ra: Vector3 = (medium._wash_rest(plan, a) as Array)[i]
	var rb: Vector3 = (medium._wash_rest(plan, b) as Array)[i]
	var r := ra.lerp(rb, f - float(a))
	return Transform3D(TarotMedium._tipped(q.w, Vector2(r.y, r.z)) * Basis.from_scale(Vector3(1.0, th / TarotMedium.DECK_T, 1.0)),
		medium._cur_base + Vector3(q.x, r.x, q.z))


## A MEETING IS A NUDGE (2026-10-06: the cards "'repel' each other in a way that makes them almost
## bouncy"): a card sliding into one lying still, struck at its middle, an edge or a corner, at the
## strongest the hands' feel samples ([constant TarotMedium.WASH_KNOCK] x 1.25) - where they touch,
## the card run into never comes away faster than the one that struck it, which goes on its way.
## Control: at half their speeds apart (the knock that made them bounce), it overtakes.
func _nudge() -> void:
	var bad := 0
	var control := 0
	var cases := 0
	for k in [TarotMedium.WASH_KNOCK * 1.25, 0.5 * 1.25]:
		for off in [Vector2.ZERO, Vector2(0.03, 0.0), Vector2(0.03, 0.05), Vector2(-0.02, 0.055)]:
			var bot := Vector2(0.0, 0.0)
			var top := Vector2(-0.07, 0.0) + (off as Vector2)
			var vel := PackedVector2Array([Vector2(1.0, 0.0), Vector2.ZERO])
			var spin := PackedFloat32Array([0.0, 0.0])
			var c := TarotMedium._touch(bot, 0.0, top)
			var dv := (vel[0] - vel[1]) * float(k)
			TarotMedium._wash_push(vel, spin, 1, c - bot, dv)
			TarotMedium._wash_push(vel, spin, 0, c - top, -dv)
			# where they touch, along the blow
			var rt := c - top
			var rb := c - bot
			var at_top := (vel[0] + Vector2(rt.y, -rt.x) * spin[0]).x
			var at_bot := (vel[1] + Vector2(rb.y, -rb.x) * spin[1]).x
			var ok := at_top > 0.0 and at_bot <= at_top + 1e-6
			if is_equal_approx(float(k), TarotMedium.WASH_KNOCK * 1.25):
				cases += 1
				bad += 0 if ok else 1
			else:
				control += 0 if ok else 1
	_ok(bad == 0, "a meeting is a nudge: the card run into never overtakes the one that struck it (%d of %d did)" % [bad, cases])
	_ok(control > 0, "control: at half their speeds apart it does (%d of %d)" % [control, cases])


## THROWN CARDS (2026-10-06: "when a lot of repulsion is applied, a card can get ejected... some
## ejections to land face-down, and thus NOT be drawn"), over the washes [param plans] (their throws
## checked as each was planned, in [method _throw_facts]): now and then, never more than the most.
func _throws(plans: Array) -> void:
	var some := 0
	var most := 0
	for p in plans:
		var k := ((p as Dictionary)["flights"] as Array).size()
		some += 1 if k > 0 else 0
		most = maxi(most, k)
	print("            throws: %d of %d washes threw a card, at most %d in one; %d thrown in all - %d not struck hard, %d face up, %d late, %d off the cloth, %d out of the picture, %d through or onto what stands"
		% [some, plans.size(), most, _thrown["n"], _thrown["soft"], _thrown["up"], _thrown["late"], _thrown["off"], _thrown["unseen"], _thrown["things"]])
	_ok(some >= plans.size() / 3 and most <= TarotMedium.EJECT_MOST, "a hard blow throws a card now and then (%d of %d washes), never more than %d (%d)"
		% [some, plans.size(), TarotMedium.EJECT_MOST, most])
	_ok(int(_thrown["soft"]) == 0, "every card thrown was struck faster than %.2f m/s (%d were not)" % [TarotMedium.EJECT_SPEED, _thrown["soft"]])
	_ok(int(_thrown["up"]) == 0 and int(_thrown["late"]) == 0, "thrown face down, and in the mixing - down before the gather (%d face up, %d late)"
		% [_thrown["up"], _thrown["late"]])
	_ok(int(_thrown["off"]) == 0 and int(_thrown["unseen"]) == 0 and int(_thrown["things"]) == 0,
		"each lands on the cloth, wholly in the picture, its way and its place clear of what stands (%d, %d, %d)"
		% [_thrown["off"], _thrown["unseen"], _thrown["things"]])


var _thrown := {"n": 0, "soft": 0, "up": 0, "late": 0, "off": 0, "unseen": 0, "things": 0}


## What [param plan]'s throws were, judged on the table it was planned on (the medium's as it is).
func _throw_facts(plan: Dictionary) -> void:
	var mix: Vector2 = plan["mix"]
	var off := Vector2(medium._mid.x, medium._mid.z)
	var cloth := Rect2(-TarotMedium.CLOTH.x * 0.5, -0.02 - TarotMedium.CLOTH.y * 0.5, TarotMedium.CLOTH.x, TarotMedium.CLOTH.y)
	for f in plan["flights"]:
		var fd: Dictionary = f
		_thrown["n"] = int(_thrown["n"]) + 1
		_thrown["soft"] = int(_thrown["soft"]) + (1 if float(fd.get("blow", 0.0)) < TarotMedium.EJECT_SPEED else 0)
		_thrown["up"] = int(_thrown["up"]) + (1 if bool(fd["up"]) else 0)
		_thrown["late"] = int(_thrown["late"]) + (1 if float(fd["t0"]) > mix.y - TarotMedium.EJECT_ROOM + 1e-4 or float(fd["t0"]) < mix.x else 0)
		var card := TarotMedium._card_poly((fd["to"] as Vector2) + off, float(fd["lie"]))
		var off_cloth := false
		var unseen := false
		for c: Vector2 in card:
			off_cloth = off_cloth or not cloth.has_point(c)
			var sc: Variant = TarotTable.project(medium._cam_base, medium._cam.fov, Vector3(c.x, TarotMedium.WASH_FLOOR, c.y))
			unseen = unseen or sc == null or not Rect2(0.0, 0.0, 1.0, 1.0).has_point(sc as Vector2)
		_thrown["off"] = int(_thrown["off"]) + (1 if off_cloth else 0)
		_thrown["unseen"] = int(_thrown["unseen"]) + (1 if unseen else 0)
		var things := not medium._path_clear(fd["from"], fd["to"])
		for foot in medium._standing:
			things = things or TarotMedium._convex_overlap(card, foot, 0.0)
		_thrown["things"] = int(_thrown["things"]) + (1 if things else 0)


## A JUMPER OUT OF A WASH (2026-10-06: "a card can get ejected, it can land face-up, and the reader
## can choose to draw that card"). The first card is a jumper and the deck is being washed as its
## moment comes: a blow throws it - and only it - out of the wash, over onto its face, the right way
## round for its draw, clear of every card; it lies there, no card sliding through or onto it, until
## it is picked up and shown as a drawn card is; the deck is gathered one card short, squared, and
## pushed aside. Then an episode whose jumper the wash is to throw holds its wash back for it, and
## one whose jumper flies from a riffle keeps the shuffle as it was.
func _wash_jumper() -> void:
	var words := "I'm just mixing the cards slowly, no rush at all, while we settle in. Nobody needs anything from you right now. Breathe out, and let the hour be whatever hour it is. There."
	var script := TarotScript.compose([{"kind": "shuffle", "card": 0, "text": words},
		{"kind": "jumper", "card": 1, "text": "Oh, one jumped."}, {"kind": "spread", "card": 0, "text": "Done."}])
	for reversed in [false, true]:
		var cards: Array = [{"key": "c0", "name": "Card", "numeral": "0", "reversed": reversed, "jumper": true,
			"position": {}, "booklet": {}, "art": ""}]
		var doc := {"show": "wash-check", "seed": 78 if reversed else 77, "dir": "", "plan": {"look": {"candles": 2}}, "cards": cards}
		subs.words = load("res://tests/tarot_look_probe.gd").timeline(TarotScript.parse(script), 0.36, Director.intro_hold)
		subs.document = {"source": script, "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		medium._follow.extend(subs.words)
		medium._sched = medium._follow.place(medium._parse["actions"], maxf(Director.intro_hold, 0.6), TarotMedium.LEAD, TarotMedium.TAIL)
		var tm := medium._times()
		var first: Array = tm["first"]
		var ts := float(tm["shuffle"])
		var te := float(first[0])
		var sc := maxf(float(first[1]), 0.05)
		# the deck is being washed as the jumper comes: twelve seconds into it
		var w0 := maxf(0.15, te - ts - 12.0)
		medium._moves = [{"kind": "wash", "t0": w0, "dur": 30.0, "pause": 1000.0, "seed": 4321,
			"plan": medium._wash_plan(4321, 30.0)}]
		medium._now = te
		medium._pose(te)
		var wj := medium._wash_jump(ts)
		_ok(not wj.is_empty(), "the jumper comes out of the wash under way (%s)" % ("reversed" if reversed else "upright"))
		if wj.is_empty():
			continue
		var c := int(wj["card"])
		var card: MeshInstance3D = medium._cards[0]
		var hidden_before := true
		var in_flight := true
		var lies_up := true
		var lies_right := true
		var on_top := true
		var through := 0
		var rest_at := te + TarotMedium.JUMP_REST * sc
		var t := ts + w0 + 2.0
		while t < float(wj["end"]) + 2.0:
			medium._now = t
			medium._pose(t)
			if t < float(wj["eject"]):
				hidden_before = hidden_before and not card.visible and (medium._deck[c] as MeshInstance3D).visible
			elif t < float(wj["land"]):
				in_flight = in_flight and card.visible and not (medium._deck[c] as MeshInstance3D).visible
			elif t < rest_at - 0.05:
				var xf := card.transform
				lies_up = lies_up and xf.basis.y.normalized().y < -0.98
				var top := -xf.basis.z.normalized()
				lies_right = lies_right and top.dot(Vector3(0.0, 0.0, 1.0 if reversed else -1.0)) > 0.5
				# on top of whatever it came down on, nothing through it (as a deck mesh's pose, for
				# its thickness)
				var as_deck := xf
				as_deck.basis.y = xf.basis.y * (TarotMedium.CARD_T / TarotMedium.DECK_T)
				var posed: Array = [as_deck]
				for i in TarotMedium.DECK_N:
					var dm: MeshInstance3D = medium._deck[i]
					if dm.visible:
						posed.append(dm.transform)
				var r := _overlaps(posed, {})
				through += int(r["crossings"]) + int(r["sinks"])
				for key in r["tops"]:
					if int(key) < 64 and not bool(r["tops"][key]):
						on_top = false
			t += 1.0 / 20.0
		_ok(hidden_before, "until it is thrown it is one of the wash's cards")
		_ok(in_flight, "thrown, the drawn card flies in that card's place")
		_ok(lies_up and lies_right, "it lies face up, its top %s the reader (%s, %s)" % ["toward" if reversed else "away from",
			str(lies_up), str(lies_right)])
		_ok(on_top and through == 0, "on top of whatever it came down on, nothing through it while it lies there (%s, %d)" % [str(on_top), through])
		# picked up and shown; the deck one card short, squared and pushed aside
		var up_at := te + TarotMedium.JUMP_RISE * sc
		medium._now = up_at + 0.1
		medium._pose(up_at + 0.1)
		var shown := card.transform.origin.distance_to(medium._present_xf(0, up_at + 0.1, up_at, INF).origin)
		var late := float(wj["end"]) + TarotMedium.PUSH_SLIDE * sc + 0.5
		medium._now = late
		medium._pose(late)
		var visible := 0
		for i in TarotMedium.DECK_N:
			visible += 1 if (medium._deck[i] as MeshInstance3D).visible else 0
		var at_side := medium._cur_base.distance_to(medium._deck_base)
		_ok(shown < 0.002, "then it is picked up and shown (%.1f mm from the shown place)" % (shown * 1000.0))
		_ok(visible == TarotMedium.DECK_N - 1 and at_side < 0.002, "the deck is gathered one card short and pushed aside (%d meshes, %.1f mm off its place)"
			% [visible, at_side * 1000.0])
	# THE SHUFFLE MAKES ROOM FOR IT: a jumper's episode that washes into it holds its one wash back to
	# begin before the jumper is reckoned to come and run past it; another keeps the shuffle as it was
	var held := -1
	var free := -1
	var long_intro := " ".join(PackedStringArray([words, words, words, words]))
	var long_script := TarotScript.compose([{"kind": "shuffle", "card": 0, "text": long_intro},
		{"kind": "jumper", "card": 1, "text": "Oh, one jumped."}, {"kind": "spread", "card": 0, "text": "Done."}])
	for sd in range(1, 60):
		if held >= 0 and free >= 0:
			break
		var doc := {"show": "wash-check", "seed": sd, "dir": "", "plan": {"look": {"candles": 2}},
			"cards": [{"key": "c0", "name": "Card", "numeral": "0", "reversed": false, "jumper": true, "position": {}, "booklet": {}, "art": ""}]}
		subs.document = {"source": long_script, "title": "Wash Check", "tarot": doc}
		medium._ensure_doc()
		if medium._wash_held and held < 0:
			held = sd
			var at := medium._wash_from + TarotMedium.JUMPER_WASH_LEAD
			var m := medium._move_at(at)
			var covers := not m.is_empty() and String(m["kind"]) == "wash"
			var mixing := covers and at - float(m["t0"]) >= float(((m["plan"] as Dictionary)["mix"] as Vector2).x) + TarotMedium.JUMP_MIX \
				and at - float(m["t0"]) <= float(((m["plan"] as Dictionary)["mix"] as Vector2).y)
			var early := 0
			for mv in medium._moves:
				if String((mv as Dictionary)["kind"]) == "wash" and float((mv as Dictionary)["t0"]) < medium._wash_from - 6.0:
					early += 1
			_ok(mixing and early == 0, "a jumper's wash is held back for it: mixing when the jumper is reckoned to come (episode %d, %s)" % [sd, str(mixing)])
		elif not medium._wash_held and free < 0:
			free = sd
	_ok(held >= 0 and free >= 0, "some jumpers' episodes wash into them, some riffle (%d, %d)" % [held, free])


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
				# a triangle all but flat holds nothing up its edges do not (a corner lying in
				# another card is a support twice over, a hair apart)
				var area := (pts[j] - pts[i]).cross(pts[k] - pts[i])
				if absf(area) < 1e-8:
					continue
				var l0 := pts[j].cross(pts[k]) / area
				var l1 := pts[k].cross(pts[i]) / area
				var l2 := pts[i].cross(pts[j]) / area
				if l0 >= -1e-9 and l1 >= -1e-9 and l2 >= -1e-9 and absf(l0 + l1 + l2 - 1.0) < 1e-6:
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
			if d > TarotMedium.HAND_SPEED / TarotMedium.WASH_HZ and TarotMedium._flight_of(plan, i, float(st) / TarotMedium.WASH_HZ).is_empty() \
					and TarotMedium._flight_of(plan, i, float(st - 1) / TarotMedium.WASH_HZ).is_empty():
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
						if (1 if qi.y > qj.y else -1) != top or absf(pi_.y - pj.y) < TarotMedium.WASH_LAYER * 0.9:
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
