extends GhostScene

## wallpaper - one motif, one symmetry group, and the whole plane follows.
##
## A flat printed pattern, edge to edge: three to five solid inks, thin keylines, and no
## glow, no gradient and no soft edge anywhere. This is the catalogue's first flat-vector
## poster surface, and it is deliberately set against a project whose whole vocabulary is
## soft blobs, glow fringes and vignetted beds - the drama here has to come from drawing,
## not from light. One small motif - a couple of bars, a wedge, a disc, an arc - is repeated
## by an ACTUAL plane symmetry group, so the mirrors, glides and rotation centres in the
## picture are real rather than suggested: under p6m the frame maps onto itself exactly at
## sixty degrees. The group table, its closure and the lattice walk live in [WallpaperGroup].
##
## NOTHING MOVES AS A PARTICLE, which is the other half of the idea. There is no drift, no
## sway, no per-element wobble. What the music does instead is RE-INK the pattern. Every
## motif element is bound at build to a pitch class; a few times a second the twelve chroma
## bins of [method Spectrum.harmonic_signature] are ranked, and each element takes the ink
## its own pitch class's RANK points at. Because every copy of the motif is the same motif,
## one chord change repaints six hundred of them on the same tick and the whole surface
## flips character in a beat. That is "sound drives colour, not scale" taken to its limit -
## a large, perfectly synchronized change with zero motion, which is a kind of drama nothing
## else here does.
##
## Two smaller audio channels. `f.beat` starts a LATTICE GLIDE: the tiling translates by
## exactly one lattice vector over a fraction of the beat period and lands back in register,
## so it reads as a step rather than a scroll - at the end of the move the picture is
## identical to the one before it. `f.movement` cross-fades the motif between two seeded
## vertex sets (a straight radial lerp with matched counts), so a section change morphs the
## pattern continuously while it stays inside its group. `f.energy` moves the keyline alpha
## and nothing else, inside a narrow sampled band.
##
## WHAT THE SEED DECIDES: which of the seventeen groups (weighted - the richer ones come up
## more), the lattice aspect and, for the oblique groups, its angle; how many motif elements
## and what each one is (bar / wedge / disc / arc), its vertex count, size, rotation and place
## in the fundamental domain; the morph target's per-vertex jitter; the [Scheme] mood and how
## many inks are drawn from it; whether the ground is high-key or near-black; the keyline's
## width and its alpha band; how often the inks are re-ranked; which lattice vectors the glide
## walks, on which beats, over how much of the beat period.
##
## THE ONE GUARD THIS SCENE NEEDS is multiplicative, and it is why the cell size is derived
## rather than sampled: cells x group order x elements is the polygon count, and 18x18 cells
## of a five-element motif under p6m (order 12) would be 19,440 polygons. So a total motif
## budget is sampled first, the element count is capped against the group's order, and the
## cell length is solved backwards from the budget - then clamped so cells-across-frame still
## lands in [5, 18], because a pattern with three cells in it is not a pattern and one with
## forty is a texture.
##
## HOW IT DRAWS. The motif is identical in every cell, so the frame is built ONCE per cell -
## every coset, every element, fills then keylines, a few thousand triangles - and that single
## triangle array is then submitted per cell under a shifted transform. The cost is two
## RenderingServer calls per cell instead of the tens of thousands of GDScript vertex appends
## a naive full-plane batch would need; measured against the budget, that is the difference
## between about a millisecond and about twenty.

const OVER := 1.35            # draw this far beyond the frame, for view drift headroom

## How far past its own cell origin the motif can reach, in cell lengths - element centres sit
## inside roughly the first half of the cell and their radii top out near 0.2, and the glide
## adds a whole lattice vector on top. Cells are enumerated with this as their margin.
const MOTIF_REACH := 2.0

## Total motifs drawn, whatever group was picked. The low end is where a pattern stops reading
## as a pattern; the high end is the frame budget (see the packet note in the header).
const BUDGET := [500.0, 1800.0]

## The keyline is feathered ([constant TriBatch.FEATHER]) only while the pattern is sparse.
## Feathering triples a line's triangle count, and a keyline edge is the single most numerous
## thing on screen here; past this many motifs the hard-edged form is the only affordable one,
## and a hard keyline is the correct look for this language anyway - a printed rule is crisp.
const SOFT_KEYLINE_MAX := 800

## How fast the morph follows f.movement, per second of simulated time. Slow enough that a
## transient does not twitch the whole plane, fast enough that a section change has completed
## by the time the ear has finished registering it.
const MORPH_RATE := 1.6

## Groups, weighted by how much pattern they actually make. p1 is a plain repeat and appears
## once; the mirror and 6-fold groups carry the surface and appear more often.
const GROUP_POOL := [
	"p1", "p2", "p2", "pm", "pm", "pg", "cm", "cm", "pmm", "pmg", "pgg",
	"cmm", "p4", "p4", "p4m", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m", "p6m",
]

## Where in the cell the motif's anchor may sit, in fractional coordinates. This is a stand-in
## for the group's true asymmetric unit, which differs per group and would be a second table
## for very little gain: keeping the anchor inside the first quadrant-ish corner of the cell
## puts it near enough to the fundamental domain that the copies interleave instead of piling
## on top of the rotation centre. The hexagonal region is squatter because a hex cell's
## fundamental wedge lies close to the a1 edge.
const SEED_REGION := {
	"oblique": {"u": [0.08, 0.44], "v": [0.08, 0.44]},
	"rect":    {"u": [0.08, 0.44], "v": [0.08, 0.44]},
	"square":  {"u": [0.10, 0.42], "v": [0.10, 0.42]},
	"hex":     {"u": [0.12, 0.46], "v": [0.04, 0.26]},
}

const KINDS := ["bar", "wedge", "disc", "arc"]

var _f: AudioFeatures = AudioFeatures.new()
var _sim := SimClock.new(60.0)
var _tb := TriBatch.new()
var _group: WallpaperGroup = null
var _offsets := PackedVector2Array()
var _cell := 0.0                     # current cell length, px
var _els: Array = []                 # the motif: one Dictionary per element
var _inks: Array = []                # Color per ink
var _ink_of := PackedInt32Array()    # element index -> ink index, re-ranked on the beat clock
var _key_col := Color.BLACK
var _dirs: Array = []                # glide directions, as lattice coefficient pairs
var _ch := Vector2.ZERO              # live tonal colour (hue, strength)
var _prev_beat := 0.0
var _beat_n := 0
var _dir_i := 0
var _gliding := false
var _glide_p := 0.0
var _glide_dur := 0.5
var _morph := 0.0
var _morph_target := 0.0
var _reink_t := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# A printed pattern is abstract matter - no mood is wrong for it, so take the whole set.
	var sch := Scheme.pick(rng)
	var gname := String(GROUP_POOL[rng.randi() % GROUP_POOL.size()])
	var ratio := rng.randf_range(0.72, 1.40)
	var angle := rng.randf_range(50.0, 90.0)
	# Build the group once at unit cell length purely to learn its ORDER, which the element
	# count has to be capped against before anything else can be sized.
	var probe := WallpaperGroup.make(gname, 1.0, ratio, angle)
	var order := probe.order()
	# Order x elements is the per-cell polygon count. Holding it near 30 keeps a p6m instance
	# from needing forty cells' worth of budget in a single cell.
	var n_el := rng.randi_range(2, clampi(int(30.0 / float(order)), 2, 5))

	var high_key := rng.randf() < 0.5
	var n_hue := rng.randi_range(2, 4)
	_inks = _build_inks(rng, sch, n_hue, high_key)
	_key_col = (Color.from_hsv(sch.hue, 0.22, rng.randf_range(0.05, 0.11))
		if high_key else Color.from_hsv(sch.hue, 0.08, rng.randf_range(0.88, 0.97)))

	_els = _build_motif(rng, probe, n_el, _inks.size())
	_ink_of.resize(n_el)
	for i in n_el:
		var el: Dictionary = _els[i]
		_ink_of[i] = int(el["home"])

	# The glide walks a seeded ring of lattice vectors. It is a RING and not a roll because
	# the direction must never be drawn from the rng on an audio-conditioned event - the live
	# analyzer and the offline bake do not produce identical beat streams, and an export that
	# stepped a different way from its preview would be a different video.
	_dirs = [Vector2i(1, 0), Vector2i(0, 1), Vector2i(-1, 0), Vector2i(0, -1), Vector2i(1, 1)]
	for i in range(_dirs.size() - 1, 0, -1):
		var j := rng.randi_range(0, i)
		var tmp: Vector2i = _dirs[i]
		_dirs[i] = _dirs[j]
		_dirs[j] = tmp
	_dirs.resize(rng.randi_range(2, 4))

	var kinds: Array = []
	var pitches: Array = []
	for i in n_el:
		var el2: Dictionary = _els[i]
		kinds.append(el2["kind"])
		pitches.append(el2["pitch"])

	# Keyline alpha rides f.energy, which is the MEAN over 64 bands and so rarely passes 0.5 -
	# the band is therefore narrow and the drive is scaled up to reach across it.
	var key_lo := rng.randf_range(0.30, 0.68)
	var g_sat := rng.randf_range(0.03, 0.11) if high_key else rng.randf_range(0.18, 0.44)
	var g_val := rng.randf_range(0.88, 0.97) if high_key else rng.randf_range(0.05, 0.12)
	return {
		"group": gname,
		"order": order,
		"lattice": probe.lattice,
		"ratio": ratio,
		"angle": angle,
		"mood": sch.name,
		"elements": n_el,
		"kinds": kinds,
		"pitch_classes": pitches,
		"inks": _inks.size(),
		"ink_hues": n_hue,
		"high_key": high_key,
		"ground_sat": g_sat,
		"ground_val": g_val,
		"ground_hue": sch.hue,
		"tint": rng.randf_range(0.20, 0.45),
		"key_width": rng.randf_range(0.8, 2.4),
		"key_lo": key_lo,
		"key_hi": minf(1.0, key_lo + rng.randf_range(0.10, 0.32)),
		"budget": rng.randf_range(float(BUDGET[0]), float(BUDGET[1])),
		"beat_thr": rng.randf_range(0.35, 0.60),
		"glide_every": rng.randi_range(1, 4),
		"glide_frac": rng.randf_range(0.35, 0.80),
		"reink_hold": rng.randf_range(0.10, 0.30),
		"morph_gain": rng.randf_range(0.8, 1.6),
	}


# The inks: a run of related hues from the Scheme, plus one extreme that the ground decides.
#
# THE BLACK-DIP CHECK. A high-key ground is a val ~0.93 sheet, and an ink that drifts up
# toward it stops being an ink - the shape vanishes and only its keyline survives, which reads
# as a printing fault rather than as a choice. So on paper every ink is capped in value and
# floored in saturation, and the extra ink is a near-black. On a dark ground the inequality
# runs the other way and the extra ink is a near-white. Either way the palette can never
# collapse into the ground it is printed on.
func _build_inks(rng: RandomNumberGenerator, sch: Scheme, n_hue: int, high_key: bool) -> Array:
	var out: Array = []
	# hue_at's own construction, but with the far end pushed out by Scheme.opposed first:
	# several moods keep their accent within 0.04 of the base (sodium's sits there), and two
	# inks that close together read as one ink and lose the whole point of a limited palette.
	var h0 := sch.hue
	var h1 := sch.opposed(h0, rng.randf_range(0.18, 0.34))
	var d := fposmod(h1 - h0 + 0.5, 1.0) - 0.5
	for i in n_hue:
		var t := float(i) / float(maxi(1, n_hue - 1))
		var c := sch.color(fposmod(h0 + d * t, 1.0),
			rng.randf_range(0.85, 1.25), rng.randf_range(0.85, 1.15))
		var s := c.s
		var v := c.v
		if high_key:
			v = minf(v, 0.80)
			s = maxf(s, 0.42)
		else:
			v = maxf(v, 0.48)
			s = maxf(s, 0.30)
		out.append(Color.from_hsv(c.h, s, v))
	if high_key:
		out.append(Color.from_hsv(sch.hue, rng.randf_range(0.10, 0.24), rng.randf_range(0.06, 0.14)))
	else:
		out.append(Color.from_hsv(sch.hue, rng.randf_range(0.03, 0.12), rng.randf_range(0.90, 0.98)))
	return out


# The motif: n_el elements clustered around one anchor inside the fundamental domain. Points
# are stored in CELL LENGTHS, not pixels, so the same motif survives a resize by scaling.
#
# Two vertex sets are built per element with matched counts, and the second is the first with a
# per-vertex RADIAL scale about the element's own centre. Radial rather than free displacement
# for a concrete reason: the fills are fanned from that centre, and a radially perturbed polygon
# is still star-shaped about it however hard the jitter is pushed, so the fan stays exact. Free
# jitter at the sampled 0.5 would turn an 8-gon concave and the fan would fold over itself.
func _build_motif(rng: RandomNumberGenerator, probe: WallpaperGroup, n_el: int, n_inks: int) -> Array:
	var region: Dictionary = SEED_REGION.get(probe.lattice, SEED_REGION["rect"])
	var ru: Array = region["u"]
	var rv: Array = region["v"]
	var anchor := (probe.a1 * rng.randf_range(float(ru[0]), float(ru[1]))
		+ probe.a2 * rng.randf_range(float(rv[0]), float(rv[1])))
	var cluster := rng.randf_range(0.05, 0.20)
	var jit_amp := rng.randf_range(0.10, 0.50)
	var out: Array = []
	for i in n_el:
		var kind := String(KINDS[rng.randi() % KINDS.size()])
		var n := rng.randi_range(3, 8)
		var r := rng.randf_range(0.07, 0.20)
		var rot := rng.randf_range(0.0, TAU)
		var ctr := anchor + Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)) * cluster
		var pts := PackedVector2Array()
		var strip := false
		match kind:
			"bar":
				# An elongated n-gon: at n = 4 with the half-step phase it is a plain rectangle,
				# which is the bar the language is built on; higher n rounds its ends off.
				var hl := rng.randf_range(0.10, 0.28)
				var hw := maxf(0.012, hl / rng.randf_range(2.0, 6.0))
				for k in n:
					var ab := TAU * (float(k) + 0.5) / float(n)
					pts.append(ctr + Vector2(cos(ab) * hl, sin(ab) * hw).rotated(rot))
			"wedge":
				# Apex at the element centre, so the fill fan and the shape agree exactly.
				var span_w := rng.randf_range(0.5, 1.8)
				pts.append(ctr)
				for k in n - 1:
					var aw := rot - span_w * 0.5 + span_w * float(k) / float(n - 2)
					pts.append(ctr + Vector2(cos(aw), sin(aw)) * r)
			"arc":
				# A band, carried as a closed ring of 2n points (outer forward, inner back) and
				# filled as a strip of trapezoids - the one non-convex element in the set.
				strip = true
				var span_a := rng.randf_range(0.9, 2.8)
				var r_in := r * rng.randf_range(0.45, 0.80)
				for k in n:
					var ao := rot - span_a * 0.5 + span_a * float(k) / float(n - 1)
					pts.append(ctr + Vector2(cos(ao), sin(ao)) * r)
				for k in range(n - 1, -1, -1):
					var ai := rot - span_a * 0.5 + span_a * float(k) / float(n - 1)
					pts.append(ctr + Vector2(cos(ai), sin(ai)) * r_in)
			_:
				var asp := rng.randf_range(0.80, 1.20)
				for k in n:
					var ad := TAU * (float(k) + 0.5) / float(n)
					pts.append(ctr + Vector2(cos(ad) * r, sin(ad) * r * asp).rotated(rot))
		# The morph target. One jitter factor per ANGULAR position, shared by an arc's inner and
		# outer point so the band stays a band rather than crossing itself.
		var na := (pts.size() / 2) if strip else pts.size()
		var jf := PackedFloat32Array()
		for k in na:
			jf.append(rng.randf_range(-jit_amp, jit_amp))
		var pts_b := PackedVector2Array()
		for k in pts.size():
			var ang := k
			if strip and k >= na:
				ang = pts.size() - 1 - k
			pts_b.append(ctr + (pts[k] - ctr) * (1.0 + jf[ang]))
		out.append({
			"kind": kind,
			"strip": strip,
			"ctr": ctr,
			"pts_a": pts,
			"pts_b": pts_b,
			"pitch": rng.randi_range(0, 11),
			"home": rng.randi_range(0, maxi(1, n_inks) - 1),
		})
	return out


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# Barely any camera at all: this is a printed surface, and the moment it breathes it stops
	# being one. The overscan and the shot still give a "drift" registration something to do.
	drift_view(f, 0.02, 0.02)
	_ch = chroma_hue()
	_size_lattice()
	var thr: float = params["beat_thr"]
	if f.beat > thr and _prev_beat <= thr:
		_on_beat(f)
	_prev_beat = f.beat
	_morph_target = clampf(f.movement * float(params["morph_gain"]), 0.0, 1.0)
	# The glide, the morph follower and the re-ink clock are STATE, so they advance on
	# wall-clock time through a SimClock rather than once per update() - the Director
	# sub-steps up to fifteen times in a frame and pre-warms every scene twelve times
	# before its first, and a naive glide would have stepped half a cell before anyone
	# had seen the pattern at rest.
	for _i in _sim.ticks(delta):
		_step(_sim.dt)
	queue_redraw()


func _step(dt: float) -> void:
	if _gliding:
		_glide_p += dt / _glide_dur
		if _glide_p >= 1.0:
			# A full lattice vector: the picture at phase 1 IS the picture at phase 0, so
			# resetting here is invisible. That identity is what makes it read as a step.
			_glide_p = 0.0
			_gliding = false
	_morph = lerpf(_morph, _morph_target, 1.0 - exp(-MORPH_RATE * dt))
	_reink_t += dt
	var hold: float = params["reink_hold"]
	if _reink_t >= hold:
		_reink_t = 0.0
		_reink()


func _on_beat(f: AudioFeatures) -> void:
	_beat_n += 1
	var every: int = params["glide_every"]
	if _gliding or _beat_n % every != 0:
		return
	_gliding = true
	_glide_p = 0.0
	_glide_dur = maxf(0.12, f.beat_period * float(params["glide_frac"]))
	_dir_i = (_dir_i + 1) % _dirs.size()


# Rank the twelve chroma bins and hand every element the ink its own pitch class's rank points
# at, offset by that element's seeded resting ink. The offset is the difference between a chord
# change PERMUTING the palette across the motif and a flat chord collapsing the whole surface
# onto one ink - and it means the pattern is still a proper multi-ink poster with no audio at
# all, which is how it gets developed.
func _reink() -> void:
	var n_inks := _inks.size()
	var sig := Spectrum.harmonic_signature()
	var peak := 0.0
	if sig.size() >= 12:
		for k in 12:
			peak = maxf(peak, sig[k])
	if sig.size() < 12 or peak < 1e-4:
		for i in _els.size():
			var el: Dictionary = _els[i]
			_ink_of[i] = int(el["home"]) % n_inks
		return
	var rank := PackedInt32Array()
	rank.resize(12)
	for k in 12:
		var r := 0
		for j in 12:
			# The index tiebreak makes the ranking a strict permutation, so the buckets below
			# always divide the twelve classes evenly however flat the chroma is.
			if sig[j] > sig[k] or (sig[j] == sig[k] and j < k):
				r += 1
		rank[k] = r
	for i in _els.size():
		var el2: Dictionary = _els[i]
		var pc: int = el2["pitch"]
		var bucket := int(float(rank[pc]) * float(n_inks) / 12.0)
		_ink_of[i] = (int(el2["home"]) + bucket) % n_inks


# Solve the cell length backwards from the motif budget, then hold cells-across in range.
func _size_lattice() -> void:
	if size.x < 4.0 or size.y < 4.0:
		return
	var field := size * OVER
	var gname: String = params["group"]
	var ratio: float = params["ratio"]
	var angle: float = params["angle"]
	var per_cell := float(int(params["order"]) * int(params["elements"]))
	var want := maxf(1.0, float(params["budget"]) / per_cell)
	var shape := WallpaperGroup.unit_cell_area(gname, ratio, angle)
	var cell := sqrt(field.x * field.y / (want * shape))
	var across := clampf(minf(field.x, field.y) / maxf(1.0, cell), 5.0, 18.0)
	cell = minf(field.x, field.y) / across
	if _group != null and absf(cell - _cell) < 0.5:
		return
	_cell = cell
	_group = WallpaperGroup.make(gname, cell, ratio, angle)
	_offsets = _group.cell_offsets(size * 0.5 * OVER, cell * MOTIF_REACH)
	params["cell_px"] = cell
	params["cells"] = _offsets.size()
	params["motifs"] = _offsets.size() * int(per_cell)


func _draw() -> void:
	begin_draw()
	if _group == null or _offsets.is_empty():
		return
	paint_ground(float(params["ground_hue"]), float(params["ground_sat"]),
		float(params["ground_val"]), float(params["tint"]) * _ch.y, _ch.x)

	var key := _key_col
	key.a = lerpf(float(params["key_lo"]), float(params["key_hi"]),
		clampf(_f.energy * 2.2, 0.0, 1.0))
	var soft := _offsets.size() * int(params["order"]) * int(params["elements"]) <= SOFT_KEYLINE_MAX
	_build_cell(key, float(params["key_width"]), soft)

	var d: Vector2i = _dirs[_dir_i]
	var p := _glide_p
	var glide := _group.lattice_vector(d.x, d.y) * (p * p * p * (p * (p * 6.0 - 15.0) + 10.0))
	# One prebuilt triangle array, submitted once per cell under a shifted transform. This is
	# TriBatch.flush's call without the reset, so the same buffers serve every cell - the whole
	# reason the pattern can afford to be a pattern.
	var base := view.matrix(size)
	var ci := get_canvas_item()
	for off in _offsets:
		draw_set_transform_matrix(base.translated_local(off + glide))
		RenderingServer.canvas_item_add_triangle_array(ci, _tb.idx, _tb.pts, _tb.cols)
	draw_set_transform_matrix(base)
	_tb.pts.clear()
	_tb.cols.clear()
	_tb.idx.clear()


# One cell: every coset x every element, fills first and keylines second, so a keyline is never
# buried under a neighbour's fill - that stacking is the whole flat-vector look.
func _build_cell(key: Color, key_w: float, soft: bool) -> void:
	var n_el := _els.size()
	var placed: Array = []
	var fans := PackedVector2Array()
	for ci in _group.cosets.size():
		var co: Dictionary = _group.cosets[ci]
		var xf: Transform2D = co["xf"]
		var flip: bool = co["flip"]
		for ei in n_el:
			var el: Dictionary = _els[ei]
			var a: PackedVector2Array = el["pts_a"]
			var b: PackedVector2Array = el["pts_b"]
			var ctr: Vector2 = el["ctr"]
			var out := PackedVector2Array()
			out.resize(a.size())
			for k in a.size():
				out[k] = xf * (a[k].lerp(b[k], _morph) * _cell)
			# A reflected coset comes out wound the other way. Nothing in this path culls by
			# winding, so this is hygiene rather than a fix - but it keeps every triangle the
			# scene emits consistently oriented, and it costs one reverse per coset.
			if flip:
				out.reverse()
			placed.append(out)
			fans.append(xf * (ctr * _cell))

	for i in placed.size():
		var pts: PackedVector2Array = placed[i]
		var el2: Dictionary = _els[i % n_el]
		var col: Color = _inks[_ink_of[i % n_el]]
		var n := pts.size()
		if bool(el2["strip"]):
			var h := n / 2
			for k in h - 1:
				_tb.quad(pts[k], pts[k + 1], pts[2 * h - 2 - k], pts[2 * h - 1 - k], col)
		else:
			var f := fans[i]
			for k in n:
				_tb.tri(f, pts[k], pts[(k + 1) % n], col)

	for i in placed.size():
		var pts2: PackedVector2Array = placed[i]
		var n2 := pts2.size()
		for k in n2:
			_tb.line(pts2[k], pts2[(k + 1) % n2], key, key_w, soft)
