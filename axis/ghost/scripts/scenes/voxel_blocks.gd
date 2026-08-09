extends GhostScene

## Voxel blocks - an isometric heightfield equalizer.
##
## A grid of iso cubes whose stack height tracks the spectrum, a 3D equalizer
## terrain. By seed it is one of three scales (see [constant SCALES]):
##   plot    - a small grid held centred in the frame (the original look).
##   terrace - a mid grid that just overflows the frame, stacks running tall.
##   city    - thousands of blocks spilling off every edge, the camera down among
##             them like a skyline. City blocks carry a structural base height so
##             the skyline stands even in quiet, the spectrum bouncing on top.
## The iso squash and the colour scheme are sampled too, so the same grid can be
## looked down on steeply or almost edge-on, in any of [Scheme]'s moods.
## Whichever it is, an [Activation] decides who moves: with sparsity some columns stay
## rooted (a still skyline) while others rise and fall - so it is not one uniform
## wall of motion (the "some could stay stationary" ask). Three flat faces per cube,
## painted back-to-front; always at least a one-block floor so it reads in silence.

## A mood of about this brightness is what the face shading was tuned against, so
## exposure is expressed relative to it and only the mood's own value shifts it.
const NOMINAL_VAL := 0.85

## The scales this equalizer is built at. `over` > 0 means the grid overflows the
## frame (the camera down among the blocks); `base` is a structural floor so a
## skyline still stands in silence. It was plot-or-city and nothing between, and a
## plot was always small and a city always vast.
const SCALES := {
	"plot": {"grid": [5, 10], "over": [0.0, 0.0], "max_h": [0.09, 0.20], "base": [0.0, 0.0]},
	"terrace": {"grid": [11, 20], "over": [1.0, 1.4], "max_h": [0.18, 0.34], "base": [0.0, 0.14]},
	"city": {"grid": [26, 44], "over": [1.5, 1.9], "max_h": [0.14, 0.28], "base": [0.18, 0.40]},
}

var _f: AudioFeatures = AudioFeatures.new()
var _h := PackedFloat32Array()    # eased column heights, 0..1
var _act: Activation
var _sch: Scheme


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	# Blocks are neutral matter - any mood paints them, so draw from the whole set.
	_sch = Scheme.pick(rng)
	var sname := String(SCALES.keys()[rng.randi() % SCALES.size()])
	var sc: Dictionary = SCALES[sname]
	var gr: Array = sc["grid"]
	var ov: Array = sc["over"]
	var mh: Array = sc["max_h"]
	var bs: Array = sc["base"]
	var grid := rng.randi_range(int(gr[0]), int(gr[1]))
	var base := rng.randf_range(float(bs[0]), float(bs[1]))
	# The bigger the build, the more structural it leans: more rooted (stationary)
	# columns, a standing skyline. A small plot is often all-active instead.
	var sparsity: float
	if sname == "plot":
		sparsity = 0.0 if rng.randf() < 0.4 else rng.randf_range(0.35, 0.7)
	elif sname == "terrace":
		sparsity = rng.randf_range(0.2, 0.65)
	else:
		sparsity = rng.randf_range(0.45, 0.8)
	_act = Activation.new(grid * grid, rng, sparsity)
	# Tall blocks travel toward the accent hue, so height is a colour axis on the
	# scheme rather than an arbitrary shift off an arbitrary hue.
	var to_accent := fposmod(_sch.accent - _sch.hue + 0.5, 1.0) - 0.5
	return {
		"grid": grid,
		"scale": sname,
		"mood": _sch.name,
		"over": rng.randf_range(float(ov[0]), float(ov[1])),   # >0 => span the frame
		"tile": rng.randf_range(0.050, 0.095),                 # plot tile half-width
		"pitch": rng.randf_range(0.40, 0.66),                  # iso squash: 2:1 was fixed
		"max_h": rng.randf_range(float(mh[0]), float(mh[1])),
		"base": base,                                          # structural floor (skyline)
		"hue": _sch.hue,
		"hue_h": to_accent * rng.randf_range(0.35, 1.1),       # hue shift with height
		"hue_pos": to_accent * rng.randf_range(0.0, 0.5),      # hue gradient across the grid
		"swirl": rng.randf() < 0.5,                            # radial vs. linear spectrum map
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	_act.update(f.energy + 0.4 * f.beat, delta)
	var g := int(params.grid)
	if _h.size() != g * g:
		_h.resize(g * g)
	var swirl: bool = params.swirl
	var base: float = params.base
	for gy in g:
		for gx in g:
			var idx := gx * g + gy
			var t: float
			if swirl:
				var dx := float(gx) - (g - 1) * 0.5
				var dy := float(gy) - (g - 1) * 0.5
				t = sqrt(dx * dx + dy * dy) / (g * 0.7)
			else:
				t = float(idx) / float(g * g - 1)
			var spec := _f.sample(clampf(t, 0.0, 1.0))
			# City: a structural base, the spectrum bouncing on top (and only on the
			# activated columns - rooted ones hold the skyline). Plot: as before.
			var target: float
			if base > 0.0:
				target = base + (1.0 - base) * spec * (0.25 + 0.75 * _act.level(idx))
			else:
				target = spec * (0.2 + 0.8 * _act.level(idx))
			_h[idx] = lerpf(_h[idx], target, 0.18)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var g := int(params.grid)
	var over: float = params.over
	# City sizes tiles to overflow the frame (zoomed in among the blocks); plot uses
	# the fixed centred tile.
	var tw: float = (size.x * over) / float(g) * 0.5 if over > 0.0 else float(params.tile) * u
	var th: float = tw * float(params.pitch)            # iso squash: how steeply we look down
	var bh: float = float(params.max_h) * u
	var hue: float = params.hue
	var hue_h: float = params.hue_h
	var hue_pos: float = params.hue_pos

	# Back-to-front: outer gy, inner gx keeps each tile in front of the one behind.
	for gy in g:
		for gx in g:
			var idx := gx * g + gy
			var top := maxf(1.0, _h[idx] * bh)         # min 1px stack -> valid faces
			var cx := float(gx - gy) * tw
			var cy := (float(gx + gy) - float(g - 1)) * th
			var hbias := hue_pos * (float(gx + gy) / float(2 * g))   # gradient across grid
			_draw_cube(Vector2(cx, cy), tw, th, top, _h[idx], fposmod(hue + hbias, 1.0), hue_h)


# One iso cube: front-left and front-right side quads, then the top diamond.
func _draw_cube(base: Vector2, tw: float, th: float, top: float, e: float, hue: float, hue_h: float) -> void:
	var h := fposmod(hue + hue_h * e, 1.0)
	var lift := Vector2(0, -top)
	var n_b := base + Vector2(0, -th)
	var e_b := base + Vector2(tw, 0)
	var s_b := base + Vector2(0, th)
	var w_b := base + Vector2(-tw, 0)
	var n_t := n_b + lift
	var e_t := e_b + lift
	var s_t := s_b + lift
	var w_t := w_b + lift

	var lit := (0.26 + 0.6 * e) / NOMINAL_VAL
	# Faces differ only in how much light they catch; the mood decides the character
	# they catch it WITH, so an ash city is grey and dim where a toxic one glares.
	# Dividing by NOMINAL_VAL keeps a mid-brightness mood at the old exposure.
	var left_face := _sch.color(h, 0.85, lit * 0.55)
	var right_face := _sch.color(h, 0.85, lit * 0.86)
	var top_face := _sch.color(h, 0.62, lit + 0.18 / NOMINAL_VAL)

	draw_colored_polygon(PackedVector2Array([w_t, s_t, s_b, w_b]), left_face)
	draw_colored_polygon(PackedVector2Array([s_t, e_t, e_b, s_b]), right_face)
	draw_colored_polygon(PackedVector2Array([n_t, e_t, s_t, w_t]), top_face)
