extends RefCounted
class_name Crystal

## Crystal - a BANK of snow crystals, generated once from sampled morphology.
##
## What was wrong with what came before. A flake was drawn by a helper taking a single
## `shape` scalar in [0,1], which fed three derived values - where the side branches sat,
## how long they were, and the size of the terminal fork. Those three are affine
## functions of the same number, and the third is anti-correlated with the other two, so
## the whole morphology space was a ONE-DIMENSIONAL CURVE: there is no flake with
## branches near the centre and also long, and none with long branches and a large tip.
## All six arms were bitwise identical by construction, the fold count was a parameter
## that every call site in the project passed the literal 6, and there was no plate, no
## column, no needle, no riming and no per-arm variation of any kind. One number.
##
## And it barely mattered, because the crystal path was almost never reached: selection
## required a probability roll AND depth > 0.8 AND a squared size roll > 0.6, which
## multiplies out to `crystal_frac * 0.064`. A preset asking for a 10-20% crystal flurry
## delivered 0.6-1.3%, or about one crystal on screen. The other 94-99% of flakes were
## two concentric circles - rotationally symmetric, so their per-flake spin and phase
## were invisible. That is the literal reason every snowflake looked the same.
##
## THE MODEL. Real snow crystal morphology is two-dimensional, and has been since Nakaya
## plotted it: TEMPERATURE selects the habit (plate near freezing, needle and column
## through the middle, plate again and then column when it gets cold enough), and
## SUPERSATURATION selects the complexity (a bare prism at low humidity, a full fernlike
## dendrite at high). Riming - supercooled droplets freezing onto the arms - is a third,
## independent axis that only needs liquid water present. So the sampled space here is
## `habit` x `branch_gens` x `rime`, which is a box rather than a line, and a weather can
## weight the habits it actually produces.
##
## THE BANK. Crystals are generated ONCE at build, as flat line lists in local unit space
## (radius 1, centred on the origin), and every falling flake references one by index.
## That is the cheap way to get variety: a few dozen genuinely different crystals shared
## across hundreds of flakes reads as "no two alike" at any density a viewer can audit,
## while costing one array lookup and a transform per flake instead of a generator. It
## also means the geometry can be pushed straight through [TriBatch] as segments, which
## is what makes drawn crystals affordable as the NORM rather than as a rare accent.

## The habits, and what each one is. Weights are supplied per weather by the caller.
##   plate     - a bare hexagonal plate, no arms. Cold and dry.
##   sectored  - a plate with ridges dividing it into six sectors.
##   stellar   - the classic six-armed star: arms with a couple of branch generations.
##   fern      - a stellar dendrite pushed to its limit; branches carry sub-branches.
##   needle    - a long thin spike. Drawn with TWO folds, which is what the fold
##               parameter was always for and never used on.
##   column    - a stubby hexagonal prism seen with its axis across the frame.
##   capped    - a column that drifted into plate conditions: a plate at each end.
##   bullet    - a rosette of a few short columns sharing an origin at odd angles.
const HABITS := ["plate", "sectored", "stellar", "fern", "needle", "column", "capped", "bullet"]

## How many arms each habit is built on. This is the knob that was inert before.
const FOLDS := {
	"plate": 6, "sectored": 6, "stellar": 6, "fern": 6,
	"needle": 2, "column": 2, "capped": 2, "bullet": 4,
}


## One crystal, ready to draw: `a`/`b` are matched endpoint arrays in LOCAL UNIT SPACE
## (the crystal spans roughly radius 1), `w` is a per-segment width multiplier, and
## `fill` is an optional convex polygon for the solid-bodied habits (a plate is a filled
## hexagon, not an outline).
class Shape:
	extends RefCounted
	var a := PackedVector2Array()
	var b := PackedVector2Array()
	var w := PackedFloat32Array()
	var fill := PackedVector2Array()
	var habit := "stellar"
	var rime := 0.0

	func seg(p: Vector2, q: Vector2, width: float) -> void:
		a.append(p)
		b.append(q)
		w.append(width)

	func size() -> int:
		return a.size()


## Build [param count] crystals. [param weights] maps a habit name to a relative weight;
## anything absent is never generated. [param complexity] biases how many branch
## generations the branching habits get, and [param rime_max] how heavily they can be
## rimed - both are the weather's business, not the individual crystal's.
static func bank(rng: RandomNumberGenerator, count: int, weights: Dictionary,
		complexity := 0.5, rime_max := 0.0) -> Array:
	var names: Array = []
	var total := 0.0
	for k in HABITS:
		var wv := float(weights.get(k, 0.0))
		if wv > 0.0:
			names.append([k, wv])
			total += wv
	if names.is_empty():
		names = [["stellar", 1.0]]
		total = 1.0
	var out: Array = []
	for i in maxi(1, count):
		var pick := rng.randf() * total
		var habit := "stellar"
		for pair in names:
			pick -= float(pair[1])
			if pick <= 0.0:
				habit = String(pair[0])
				break
		out.append(make(rng, habit, complexity, rime_max))
	return out


## One crystal of a named habit.
static func make(rng: RandomNumberGenerator, habit: String, complexity := 0.5,
		rime_max := 0.0) -> Shape:
	var s := Shape.new()
	s.habit = habit
	s.rime = 0.0 if rime_max <= 0.0 else rng.randf_range(0.0, rime_max)
	var folds: int = int(FOLDS.get(habit, 6))
	match habit:
		"plate":
			_plate(s, rng, 1.0, true)
		"sectored":
			_plate(s, rng, 1.0, true)
			_sector_ridges(s, rng)
		"needle":
			_needle(s, rng)
		"column":
			_column(s, rng, false)
		"capped":
			_column(s, rng, true)
		"bullet":
			_bullet(s, rng)
		_:
			_dendrite(s, rng, folds, complexity, habit == "fern")
	if s.rime > 0.01:
		_rime(s, rng)
	return s


# --- habits ------------------------------------------------------------------------

## A regular hexagon. `solid` fills it; otherwise it is an outline. The plate is the
## habit the old drawing code could not express at all - it has no arms, and a flake
## with no arms was simply not in the vocabulary.
static func _plate(s: Shape, rng: RandomNumberGenerator, r: float, solid: bool) -> void:
	var turn := rng.randf() * TAU
	# Real plates are not perfectly regular; a few percent of radius variation per corner
	# is the difference between a drawn hexagon and a grown one.
	var pts := PackedVector2Array()
	for i in 6:
		var ang := turn + TAU * float(i) / 6.0
		var rr := r * rng.randf_range(0.94, 1.03)
		pts.append(Vector2(cos(ang), sin(ang)) * rr)
	for i in 6:
		s.seg(pts[i], pts[(i + 1) % 6], 1.0)
	if solid:
		s.fill = pts


## The ridges that divide a sectored plate: a spoke to each corner, sometimes stopping
## short of it, which is what makes the sectors read as facets rather than as a wheel.
static func _sector_ridges(s: Shape, rng: RandomNumberGenerator) -> void:
	var reach := rng.randf_range(0.55, 0.95)
	var n := s.fill.size()
	for i in n:
		s.seg(Vector2.ZERO, s.fill[i] * reach, 0.7)


## A long thin spike, drawn on TWO folds so it is a bar rather than a star.
static func _needle(s: Shape, rng: RandomNumberGenerator) -> void:
	var half_w := rng.randf_range(0.03, 0.10)
	var taper := rng.randf_range(0.2, 0.7)
	var ang := rng.randf() * TAU
	var d := Vector2(cos(ang), sin(ang))
	var p := Vector2(-d.y, d.x) * half_w
	# a closed sliver: two long sides drawn toward a narrowed tip at each end
	s.seg(-d + p * taper, d + p * taper, 1.0)
	s.seg(-d - p * taper, d - p * taper, 1.0)
	s.seg(-d + p * taper, -d - p * taper, 0.8)
	s.seg(d + p * taper, d - p * taper, 0.8)
	# a few internal striations, which is what makes a needle read as ice
	var lines := rng.randi_range(1, 3)
	for i in lines:
		var f := float(i + 1) / float(lines + 1) * 2.0 - 1.0
		s.seg(d * f + p, d * f - p, 0.5)


## A hexagonal prism seen from the side: two end caps joined by its long faces. `capped`
## puts a full plate on each end - the capped column, which is one of the most
## recognisable real habits and was completely absent.
static func _column(s: Shape, rng: RandomNumberGenerator, capped: bool) -> void:
	var ang := rng.randf() * TAU
	var d := Vector2(cos(ang), sin(ang))
	var n := Vector2(-d.y, d.x)
	var half_len := rng.randf_range(0.55, 1.0)
	var rad := rng.randf_range(0.16, 0.34)
	var c0 := -d * half_len
	var c1 := d * half_len
	for k in [-1.0, 1.0]:
		s.seg(c0 + n * rad * k, c1 + n * rad * k, 1.0)
	s.seg(c0 - n * rad, c0 + n * rad, 0.9)
	s.seg(c1 - n * rad, c1 + n * rad, 0.9)
	# the visible far edge of the prism, which gives it a little solidity
	s.seg(c0 + n * rad * 0.35, c1 + n * rad * 0.35, 0.45)
	if capped:
		var cap := rng.randf_range(0.5, 0.85)
		for c in [c0, c1]:
			var pts := PackedVector2Array()
			for i in 6:
				var a2 := TAU * float(i) / 6.0
				pts.append(c + (n * cos(a2) + d * sin(a2) * 0.25) * cap)
			for i in 6:
				s.seg(pts[i], pts[(i + 1) % 6], 0.8)


## A bullet rosette: a few short columns sharing an origin at independent angles. This
## is the habit that the old single-basis drawing could not express even in principle,
## because every arm had to share one orientation.
static func _bullet(s: Shape, rng: RandomNumberGenerator) -> void:
	var arms := rng.randi_range(3, 6)
	for i in arms:
		var ang := rng.randf() * TAU
		var d := Vector2(cos(ang), sin(ang))
		var n := Vector2(-d.y, d.x)
		var ln := rng.randf_range(0.45, 1.0)
		var rad := ln * rng.randf_range(0.10, 0.22)
		for k in [-1.0, 1.0]:
			s.seg(n * rad * float(k), d * ln + n * rad * float(k) * 0.4, 0.9)
		s.seg(d * ln - n * rad * 0.4, d * ln + n * rad * 0.4, 0.7)


## The branching habits. `gens` generations of side branches on each arm, at sampled
## positions and lengths, with SIBLING JITTER: the branch schedule is drawn once for the
## crystal and then each arm perturbs it slightly, because real arms grew in the same
## conditions but are not stamped copies. That single detail does more for the read than
## any amount of extra branching.
static func _dendrite(s: Shape, rng: RandomNumberGenerator, folds: int, complexity: float,
		fern: bool) -> void:
	var gens := 1
	if fern:
		gens = 2 if rng.randf() < 0.35 else 3
	else:
		gens = 1 if rng.randf() > complexity else 2
	var per_arm := rng.randi_range(2, 5) if fern else rng.randi_range(1, 3)
	var branch_ang := deg_to_rad(rng.randf_range(52.0, 68.0))
	var core := rng.randf_range(0.0, 0.30)
	var decay := rng.randf_range(0.62, 0.9) if fern else rng.randf_range(0.45, 0.72)
	# the shared schedule: where along the arm each branch pair sits, and how long it is
	var at: Array = []
	for k in per_arm:
		var f := core + (1.0 - core) * (float(k + 1) / float(per_arm + 1))
		at.append([f, rng.randf_range(0.22, 0.44) * (1.0 - f * 0.5)])
	if core > 0.02:
		_plate_at(s, rng, core, folds)
	for arm in folds:
		var base := TAU * float(arm) / float(folds)
		# per-arm jitter - a few percent, deliberately small
		var ang := base + deg_to_rad(rng.randf_range(-2.5, 2.5))
		var reach := rng.randf_range(0.95, 1.0)
		var d := Vector2(cos(ang), sin(ang))
		s.seg(d * core, d * reach, 1.0)
		for pair in at:
			var f: float = float(pair[0]) * rng.randf_range(0.96, 1.04)
			var ln: float = float(pair[1]) * rng.randf_range(0.9, 1.12)
			var root := d * (reach * f)
			for sgn in [-1.0, 1.0]:
				var ba: float = ang + float(sgn) * branch_ang
				var bd := Vector2(cos(ba), sin(ba))
				var tip := root + bd * ln
				s.seg(root, tip, 0.75)
				if gens >= 2:
					# sub-branches, which is the whole difference between a stellar
					# dendrite and a fernlike one
					var subs := 2 if fern else 1
					for j in subs:
						var sf := float(j + 1) / float(subs + 1)
						var sroot := root + bd * (ln * sf)
						for sg2 in [-1.0, 1.0]:
							var sa: float = ba + float(sg2) * branch_ang
							s.seg(sroot, sroot + Vector2(cos(sa), sin(sa)) * (ln * decay * 0.5), 0.5)


## A small central hexagon some dendrites grow out of.
static func _plate_at(s: Shape, rng: RandomNumberGenerator, r: float, folds: int) -> void:
	var pts := PackedVector2Array()
	var turn := rng.randf() * TAU
	var n := maxi(3, folds)
	for i in n:
		var ang := turn + TAU * float(i) / float(n)
		pts.append(Vector2(cos(ang), sin(ang)) * r)
	for i in n:
		s.seg(pts[i], pts[(i + 1) % n], 0.8)


## Riming: frozen droplets stuck to the structure. Modelled as short stubs normal to
## existing segments, so a heavily rimed crystal reads as lumpy and blurred at its
## edges - which is exactly what riming does to a real one, and what a blizzard's
## crystals should look like.
static func _rime(s: Shape, rng: RandomNumberGenerator) -> void:
	var n := s.size()
	if n == 0:
		return
	var drops := int(6.0 + 26.0 * s.rime)
	for i in drops:
		var k := rng.randi() % n
		var p := s.a[k]
		var q := s.b[k]
		var f := rng.randf()
		var at := p.lerp(q, f)
		var dir := (q - p).normalized()
		var nrm := Vector2(-dir.y, dir.x)
		var len2 := rng.randf_range(0.03, 0.09) * (0.5 + s.rime)
		var sgn := 1.0 if rng.randf() < 0.5 else -1.0
		s.seg(at, at + nrm * (len2 * sgn), 0.6)
