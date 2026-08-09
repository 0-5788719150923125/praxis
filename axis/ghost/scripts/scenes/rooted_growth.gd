extends GhostScene

## Rooted growth - crawling roots and tendrils that spread from a seed.
##
## Built on the [Filament] primitive: each root follows a curl-noise [Flow2D] field so
## it meanders and curls, branches into trunk-and-limb structure, tapers, and is
## revealed along a growth front so it *crawls* outward rather than appearing whole.
##
## Roots emerge the way real ones do: mostly as **laterals** sprouting from a point
## partway along an *existing* root (at an angle to that parent), only occasionally as a
## fresh taproot from the central seed. So the system spreads from many points across the
## branches instead of every strand radiating from one origin.
##
## Each root runs an independent **lifecycle**, staggered and rate-varied so the field
## is asynchronous (no uniform bands): it grows slowly to full, holds, then **retires**
## gracefully - either FADING out (alpha to zero) or REWINDING (the front retracts back
## toward its attachment point, collapsing inward) - before regrowing from a fresh site.
## It never clears and pops back in. Audio surges the growth front (a nonlinear `spike`,
## so beats lunge it and quiet barely moves) and carries colour and glow.
##
## Neither the colour nor the silhouette is fixed. A SEASON is the first thing the seed
## chooses, and it bundles everything a season actually changes - the [Scheme] moods on
## offer, how dense the growth is, how thick and how long and how branched, how hard the
## colour turns from base to tip, and what is in the air. Spring is pale and dense and fine,
## summer lush and deep, autumn sparse and bare and hard-turning, winter a few thick
## structural limbs with the saturation drained out. Within the season a growth HABIT sets
## the form, from a few long thick taproots through a dense fine mat to sparse long creepers.

## Growth HABITS - the silhouette choice, sampled per session. A root system is not one
## shape: some are a few long thick taproots, some a dense fine mat, some sparse long
## creepers. The numbers only read as a habit TOGETHER (a mat is many AND short AND fine AND
## mostly lateral), so they are sampled as a set rather than as independent ranges.
const HABITS := {
	"taproot": {"roots": [7, 12],  "len": [0.55, 0.82], "width": [8.0, 13.0],
		"steps": [16, 24], "centre": 0.62, "emit": [0.6, 1.2], "tendril": 0.15},
	"mat":     {"roots": [24, 34], "len": [0.24, 0.38], "width": [3.0, 5.5],
		"steps": [9, 14],  "centre": 0.20, "emit": [0.2, 0.5], "tendril": 0.55},
	"creeper": {"roots": [11, 18], "len": [0.62, 0.95], "width": [3.5, 6.5],
		"steps": [18, 28], "centre": 0.35, "emit": [0.4, 0.9], "tendril": 0.80},
	"thicket": {"roots": [14, 22], "len": [0.42, 0.60], "width": [5.0, 9.0],
		"steps": [12, 20], "centre": 0.42, "emit": [0.4, 0.9], "tendril": 0.40},
}

## SEASONS - the top-level choice, made before mood or habit.
##
## Colour alone could already reach autumn: the mood list has ember, dawn, brass and sodium
## in it. What it could not do was BE an autumn wood, because colour and density and form
## were rolled independently, so a seed was free to produce dense fresh young growth painted
## rust - a spring plant with an orange filter over it. An autumn wood is oranger AND sparser
## AND barer AND reaching further on thicker limbs, and its colour turns hard from trunk to
## tip; a spring one is pale AND dense AND small AND fine; a winter one is the same wood
## drained of saturation, down to a few thick structural limbs. Those belong together, so
## they are one entry rather than five independent ranges.
##
## - `moods`   the [Scheme] moods on offer. Deliberately WIDE - a season biases colour, it
##             does not own it (a rain-soaked November wood is ash; a bare winter branch
##             against a low sun is ember), so each season keeps 8-9 of the 14 moods.
## - `habits`  which HABITS are plausible - a winter wood is not a dense fine mat.
## - `density` scales the root count. The strongest single read of a season.
## - `limbs`   scales path resolution, and so how many times a root forks along its length.
## - `leaf`    scales strand width - fine spring shoots against thick bare winter limbs.
## - `reach`   scales root length.
## - `eager`   scales the gap between new shoots - how fast the frame fills in.
## - `crown`   scales how many shoots come straight from the seed rather than off older wood.
## - `coil`    scales the chance of the coiling tendril form over the woody root form.
## - `bare`    the chance a shoot grows as an unbranching THREAD instead. This is what "barer"
##             actually means here, and it had to be measured to be found: root count alone
##             made autumn structurally sparser but not visibly so, because a few long
##             creepers forking four deep put more line on screen than forty short spring
##             shoots. Branchiness lives in the [Filament] variant, so the season picks the
##             variant - reusing that registry rather than adding a fork-rate of its own.
## - `turn`    how far hue walks toward the mood's accent, per branch depth AND along each
##             strand. Autumn is the season where one plant carries three colours at once, so
##             it turns hardest; the along-strand half is what keeps that true of the bare
##             threads, which have almost no branch depth to walk over.
## - `sat`/`val` scale the mood's own character rather than replacing it, so winter is
##             recognisably the SAME mood with the colour drained out of it.
## - `air`     what drifts through the frame, taken from the shared [Layer] registry so the
##             season inherits components other scenes already use instead of growing its own.
const SEASONS := {
	"spring": {
		"moods": ["verdant", "toxic", "teal", "rose", "dawn", "bone", "magenta", "violet"],
		"habits": ["mat", "thicket", "creeper"],
		"density": 1.30, "limbs": 1.15, "leaf": 0.55, "reach": 0.78,
		"eager": 0.65, "crown": 0.70, "coil": 1.50, "bare": 0.0,
		"turn": [0.04, 0.12], "sat": 0.86, "val": 1.10,
		"air": [{"kind": "petals", "n": [14, 26]}],
	},
	"summer": {
		"moods": ["verdant", "toxic", "teal", "brass", "sodium", "magenta", "violet",
			"abyss", "ember"],
		"habits": ["mat", "thicket", "creeper"],
		"density": 1.50, "limbs": 1.30, "leaf": 0.90, "reach": 1.00,
		"eager": 0.55, "crown": 0.85, "coil": 1.30, "bare": 0.0,
		"turn": [0.06, 0.18], "sat": 1.15, "val": 1.00,
		"air": [{"kind": "fireflies", "n": [10, 22]}, {"kind": "dust", "n": [50, 90]}],
	},
	"autumn": {
		"moods": ["ember", "dawn", "sodium", "brass", "rose", "magenta", "ash",
			"verdant", "violet"],
		"habits": ["taproot", "creeper"],
		"density": 0.60, "limbs": 0.78, "leaf": 1.05, "reach": 1.18,
		"eager": 1.45, "crown": 1.10, "coil": 0.55, "bare": 0.55,
		"turn": [0.20, 0.42], "sat": 1.10, "val": 0.94,
		"air": [{"kind": "petals", "n": [10, 20]}, {"kind": "dust", "n": [40, 70]}],
	},
	"winter": {
		"moods": ["ash", "bone", "glacier", "abyss", "violet", "teal", "ember",
			"rose", "brass"],
		"habits": ["taproot", "thicket"],
		"density": 0.55, "limbs": 0.58, "leaf": 1.45, "reach": 0.95,
		"eager": 1.80, "crown": 1.35, "coil": 0.12, "bare": 0.72,
		"turn": [0.02, 0.07], "sat": 0.42, "val": 1.06,
		"air": [{"kind": "snow", "n": [70, 130]}],
	},
}

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _flow: Flow2D
var _roots: Array = []
var _variant := "root"
var _habit := "thicket"
var _season := "summer"
var _sch: Scheme
var _hue_depth := 0.0
var _hue_along := 0.0        # the season's colour turn ALONG a strand, base to tip
var _sat_mul := 1.0          # the season scaling the mood's saturation - winter is drained
var _val_mul := 1.0
var _air := "dust"           # the season's airborne [Layer] - blossom, pollen, leaves, snow
var _glow := 0.0
var _base_len := 0.5
var _width := 7.0
var _steps := Vector2i(12, 20)   # path resolution/length in segments, per habit
var _centre := 0.42              # chance a new shoot is a fresh taproot rather than a lateral
var _draw_life := 1.0        # current root's alpha, set before its draw, read by _color_for
var _draw_hue := 0.0         # current root's own hue, likewise - a family, not clones
var _max_roots := 20         # the bloom grows the pool up to this, then sustains
var _emit_t := 0.0           # countdown to the next sprout from the seed
var _emit_interval := 0.6    # sampled seconds between new sprouts


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	# The season comes first, and everything below is drawn through it - that is the whole
	# point of the table: colour, density and form cannot disagree about what time of year
	# this is, because they are all read off the same entry.
	_season = String(SEASONS.keys()[rng.randi() % SEASONS.size()])
	var sea: Dictionary = SEASONS[_season]
	_sch = Scheme.among(sea.moods, rng)
	_sat_mul = float(sea.sat)
	_val_mul = float(sea.val)
	# Depth walks the strand toward the scheme's ACCENT instead of an arbitrary literal drift,
	# so trunk and limb are related by the mood rather than by chance. How FAR it walks is the
	# season's call - an autumn plant carries green, gold and rust at once; a winter one is
	# all one colour from trunk to twig.
	var to_accent := fposmod(_sch.accent - _sch.hue + 0.5, 1.0) - 0.5
	var turn := rng.randf_range(float(sea.turn[0]), float(sea.turn[1]))
	_hue_depth = to_accent * turn
	_hue_along = to_accent * turn * 1.6
	var habs: Array = sea.habits
	_habit = String(habs[rng.randi() % habs.size()])
	var hab: Dictionary = HABITS[_habit]
	# The form: a BARE unbranching thread (late-season wood), else the habit's own call between
	# a coiling tendril and a woody forking root - which the season only leans on. Soft coiling
	# shoots are a growing season's business; winter is bare thread almost always.
	if rng.randf() < float(sea.bare):
		_variant = "thread"
	else:
		_variant = "tendril" if rng.randf() < float(hab.tendril) * float(sea.coil) else "root"
	_flow = Flow2D.new(rng.randi(), rng.randf_range(2.0, 3.4), 0.06)
	_width = rng.randf_range(float(hab.width[0]), float(hab.width[1])) * float(sea.leaf)
	_base_len = rng.randf_range(float(hab.len[0]), float(hab.len[1])) * float(sea.reach)
	# Path resolution is also the fork count (a branch is rolled per step), so this is how many
	# limbs a root throws, not just how smooth it is.
	var lim := float(sea.limbs)
	var s0 := maxi(4, int(round(float(hab.steps[0]) * lim)))
	_steps = Vector2i(s0, maxi(s0 + 2, int(round(float(hab.steps[1]) * lim))))
	_centre = clampf(float(hab.centre) * float(sea.crown), 0.05, 0.9)
	_max_roots = maxi(3, int(round(float(rng.randi_range(int(hab.roots[0]), int(hab.roots[1])))
		* float(sea.density))))
	_emit_interval = rng.randf_range(float(hab.emit[0]), float(hab.emit[1])) * float(sea.eager)
	_emit_t = _emit_interval
	# What is in the air, from the shared [Layer] registry - blossom, pollen, dry leaves, snow.
	var air: Array = sea.air
	var pick: Dictionary = air[rng.randi() % air.size()]
	_air = String(pick.kind)
	add_layer(_air, rng, _air_cfg(_air, rng.randi_range(int(pick.n[0]), int(pick.n[1]))))
	# Start SPARSE - a few barely-sprouted shoots from the seed - and bloom outward over time
	# (new shoots keep emitting from the centre in update), rather than filling the frame at
	# once and then holding a static shape.
	var start := rng.randi_range(2, 4)
	for i in start:
		var r := _new_root()
		r.heading = -PI * 0.5 + TAU * float(i) / float(start) + rng.randf_range(-0.4, 0.4)
		_regrow(r, true)                            # from the central seed
		r.grown = rng.randf_range(0.0, 0.25)        # barely out of the ground; they grow from here
		_roots.append(r)
	return {"season": _season, "mood": _sch.name, "habit": _habit, "form": _variant,
		"roots": _max_roots, "width": _width, "air": _air}


# Tint the season's airborne layer to the plant it falls from: blossom and dry leaves take the
# mood's ACCENT (the colour the branch tips are already turning toward), pollen and snow the
# base hue, and everything inherits the season's own saturation scaling so winter snow is not
# more vivid than the wood it settles on.
func _air_cfg(kind: String, n: int) -> Dictionary:
	var s: float = clampf(_sch.sat * _sat_mul, 0.04, 0.95)
	match kind:
		"petals":
			return {"count": n, "hue": _sch.accent, "sat": s}
		"snow":
			return {"count": n, "hue": _sch.hue, "sat": minf(s, 0.12), "size": 0.005}
		"fireflies":
			return {"count": n, "hue": _sch.accent}
		_:
			return {"count": n, "hue": _sch.hue}


# A fresh root record (lifecycle fields); origin/heading/fil are filled in by _regrow.
func _new_root() -> Dictionary:
	return {"fil": null, "grown": 0.0, "life": 1.0, "state": "grow",
		"timer": 0.0, "rate": 1.0, "hold": 2.0, "mode": "fade", "retract_to": 0.0,
		"origin": Vector2.ZERO, "heading": 0.0, "hue": _sch.vary(_rng),
		"trate": _rng.randf_range(4.0, 9.0), "tphase": _rng.randf_range(0.0, TAU)}


# The seed keeps sprouting: a new shoot emerges from the centre heading outward in any
# direction, so the root system continuously blooms over time instead of pausing once full.
func _emit_root() -> void:
	var r := _new_root()
	r.heading = _rng.randf_range(-PI, PI)
	_regrow(r, true)
	r.grown = 0.0
	_roots.append(r)


# Grow a fresh path for one root from a spawn site (a point along an existing root, or
# the central seed), on a heading set by that site, and re-roll its lifecycle constants
# (rate / hold / retire mode) so each life differs from the last. Laterals start shorter
# and finer than taproots, so the network reads as trunks feeding ever-thinner roots.
func _regrow(r: Dictionary, force_centre := false) -> void:
	var steps := _rng.randi_range(_steps.x, _steps.y)
	var site := _spawn_site(force_centre)
	r.origin = site.pos
	r.heading = float(site.ang) + _rng.randf_range(-0.3, 0.3)
	var length := _base_len * _rng.randf_range(0.8, 1.25)
	var w := _width
	if site.lateral:
		length *= _rng.randf_range(0.5, 0.8)
		w *= _rng.randf_range(0.45, 0.7)
	r.fil = Filament.grow(_variant, r.origin, r.heading, length, w, steps, _flow, _rng)
	r.grown = 0.0
	r.life = 1.0
	r.state = "grow"
	r.rate = _gauss_rate()
	r.hold = _rng.randf_range(0.6, 2.2)   # shorter holds: keep crawling, don't freeze full
	r.mode = _roll_mode()   # mostly partial rewind, rarely a full fade-out


# Where a new root sprouts: mostly a point partway along an existing, revealed root
# (a LATERAL, emerging at an angle to its parent), occasionally the central seed (a fresh
# TAPROOT). Returns {pos, ang (heading suggestion), lateral}. Falls back to the centre
# when nothing has grown enough to branch from yet (e.g. the very first roots).
func _spawn_site(force_centre := false) -> Dictionary:
	var centre := {"pos": Vector2.ZERO, "ang": _rng.randf_range(-PI, PI), "lateral": false}
	# How many shoots come straight from the seed is the HABIT's call (a mat spreads almost
	# entirely by laterals; a taproot system keeps sending fresh roots down from the crown),
	# and emitted shoots force it.
	if force_centre or _rng.randf() < _centre:
		return centre
	# Roots with something revealed to branch from.
	var live := []
	for o in _roots:
		if o.fil != null and o.fil.segs.size() > 0 and float(o.grown) > 0.12:
			live.append(o)
	if live.is_empty():
		return centre
	var o = live[_rng.randi() % live.size()]
	var segs: Array = o.fil.segs
	# Find a revealed, fairly basal segment (basal points survive the parent's diebacks and
	# read as laterals off older tissue), then emerge from a point along it at an angle.
	var reveal: float = minf(float(o.grown), 0.65)
	for _attempt in 6:
		var s: Dictionary = segs[_rng.randi() % segs.size()]
		if float(s.born0) <= reveal:
			var t := _rng.randf()
			var pos: Vector2 = (s.a as Vector2).lerp(s.b, t)
			var tang := ((s.b as Vector2) - (s.a as Vector2)).angle()
			var side := 1.0 if _rng.randf() < 0.5 else -1.0
			return {"pos": pos, "ang": tang + side * _rng.randf_range(0.5, 1.1), "lateral": true}
	return centre


# A stuttery, timelapse advance multiplier: cubed sine spends most of its time low (the front
# creeps or pauses) with brief peaks (a lunge), per root, so growth twitches forward.
func _twitch(r: Dictionary) -> float:
	var s := 0.5 + 0.5 * sin(_life * float(r.trate) + float(r.tphase))
	return 0.25 + 1.6 * pow(s, 3.0)


# Per-root growth-speed multiplier from a ~normal distribution (sum of three uniforms),
# wide spread - so roots grow at a real variety of speeds, not all at roughly one rate.
func _gauss_rate() -> float:
	var g := (_rng.randf() + _rng.randf() + _rng.randf()) / 3.0
	return 0.3 + 1.7 * g


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05)
	_flow.advance(delta)
	update_layers(f, delta)
	_glow = Nonlinear.flare(_glow, clampf(0.30 * f.energy + 0.70 * f.beat, 0.0, 1.0), delta, 8.0, 1.5)

	# The seed keeps blooming: emit new shoots from the centre over time until the pool fills,
	# so the system is always sprouting rather than settling into one finished shape.
	_emit_t -= delta
	if _emit_t <= 0.0:
		_emit_t = _emit_interval * _rng.randf_range(0.7, 1.4)
		if _roots.size() < _max_roots:
			_emit_root()

	# A slow base creep, surged by beats/energy through a spike curve.
	var drive := 0.55 + 1.1 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)
	for r in _roots:
		_advance(r, delta, drive)
	queue_redraw()


# One root's lifecycle step: grow -> hold -> (fade | rewind) -> regrow.
func _advance(r: Dictionary, delta: float, drive: float) -> void:
	match r.state:
		"grow":
			# Slow and deliberate, and TWITCHY: the front mostly creeps, with brief lunges and
			# near-pauses, the way a plant-growth timelapse stutters forward rather than gliding.
			r.grown = minf(1.0, r.grown + delta * 0.12 * float(r.rate) * drive * _twitch(r))
			if r.grown >= 1.0:
				r.state = "hold"
				r.timer = r.hold
		"hold":
			r.timer -= delta
			if r.timer <= 0.0:
				r.state = r.mode
				if r.state == "rewind":
					r.retract_to = _roll_retract()
		"fade":
			r.life = maxf(0.0, r.life - delta * 0.55)
			if r.life <= 0.0:
				_regrow(r)
		"rewind":
			# Retract the front back toward the seed - the root collapses partly inward.
			var floor_v: float = r.retract_to
			r.grown = maxf(floor_v, r.grown - delta * 0.4 * (0.6 + 0.5 * drive))
			if r.grown <= floor_v + 0.005:
				if floor_v < 0.08:
					_regrow(r)                  # fully retracted -> fresh path
				else:
					r.state = "grow"            # partial -> regrow the same root back up
					r.hold = _rng.randf_range(0.6, 2.2)
					r.mode = _roll_mode()


# How far a rewinding root pulls back: almost always a shallow dieback (then it grows out
# again on the same root); a deep collapse that nearly erases it is rare.
func _roll_retract() -> float:
	if _rng.randf() < 0.10:
		return _rng.randf_range(0.0, 0.30)    # rare: deep collapse (< 0.08 = fresh path)
	return _rng.randf_range(0.55, 0.88)        # usual: a small partial dieback


# Mostly a partial REWIND (dieback + regrow), only occasionally a full FADE + new path.
func _roll_mode() -> String:
	return "fade" if _rng.randf() < 0.12 else "rewind"


func _draw() -> void:
	begin_draw()
	var u := unit()
	# Timelapse twitch: trunks hold, young tips tremble (more with energy).
	var jitter := 0.008 + 0.018 * _f.energy
	for r in _roots:
		if r.fil == null:
			continue
		_draw_life = float(r.life)
		_draw_hue = float(r.hue)
		var tip := _sch.color(_sch.accent, 0.35 * _sat_mul, 1.15 * _val_mul, 0.85 * _draw_life)
		r.fil.draw_growing(self, u, float(r.grown), _color_for, tip, jitter, _life)
	# The season's air falls in FRONT of the growth, so blossom and snow read as between the
	# viewer and the plant rather than as part of it.
	draw_layers()


# Colour with TEXTURE along the strand, not one flat tone: the hue drifts and the strand
# desaturates from base to tip (a woody-to-fresh gradient), brightness ramps toward the tip,
# and a fine sinusoidal band rides along the length so the line reads as grain / pattern
# rather than a uniform stroke. `along` is 0 at the base, 1 at the growing tip. Brightness is
# still carried by bass + beat glow; alpha by the root's current life.
func _color_for(depth: int, along := 0.0) -> Color:
	# Base hue is this ROOT's own drift off the scheme, so the system is a family; depth walks
	# toward the accent, and `along` walks the same way (plus the scheme's own spread) so a
	# single unbranched limb still turns from base to tip the way a real one does.
	var h := fposmod(_draw_hue + _hue_depth * float(depth)
		+ (_sch.spread * 2.0 + _hue_along) * along, 1.0)
	var band := 0.11 * sin(along * 38.0 + float(depth) * 1.9)        # fine grain along the length
	var v := clampf(0.28 + 0.40 * along + 0.42 * _f.bass + 0.38 * _glow + band, 0.10, 1.0)
	# Saturation/value SCALE the mood rather than restating absolutes, so ash roots stay
	# grey-dry and toxic ones stay lurid; still fresher / paler toward the tip. The SEASON
	# scales them again, which is what makes a winter wood the same mood with the colour
	# drained out of it rather than a different mood entirely.
	return _sch.color(h, _sat_mul * (1.0 - 0.42 * along), v * 1.2 * _val_mul, 0.92 * _draw_life)
