extends VortexScene

## Voxel blocks - an isometric heightfield equalizer.
##
## A grid of iso cubes whose stack height tracks the spectrum, a 3D equalizer
## terrain. By seed it is one of two scales:
##   plot - a small grid held centred in the frame (the original look).
##   city - thousands of blocks spilling off every edge, the camera down among them
##          like a skyline. City blocks carry a structural base height so the
##          skyline stands even in quiet, with the spectrum bouncing on top.
## Either way an [Activation] decides who moves: with sparsity some columns stay
## rooted (a still skyline) while others rise and fall - so it is not one uniform
## wall of motion (the "some could stay stationary" ask). Three flat faces per cube,
## painted back-to-front; always at least a one-block floor so it reads in silence.

var _f: AudioFeatures = AudioFeatures.new()
var _h := PackedFloat32Array()    # eased column heights, 0..1
var _act: Activation
var _city := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	_city = rng.randf() < 0.5
	var grid := rng.randi_range(28, 44) if _city else rng.randi_range(6, 9)
	# City leans structural: more rooted (stationary) columns, a standing skyline.
	var sparsity := rng.randf_range(0.45, 0.8) if _city else (0.0 if rng.randf() < 0.4 else rng.randf_range(0.35, 0.7))
	_act = Activation.new(grid * grid, rng, sparsity)
	return {
		"grid": grid,
		"over": rng.randf_range(1.5, 1.9) if _city else 0.0,   # >0 => span the frame (city)
		"tile": rng.randf_range(0.060, 0.085),                 # plot tile half-width
		"max_h": rng.randf_range(0.16, 0.26) if _city else rng.randf_range(0.10, 0.18),
		"base": rng.randf_range(0.18, 0.4) if _city else 0.0,  # structural floor (city skyline)
		"hue": rng.randf(),
		"hue_h": rng.randf_range(0.10, 0.40),                  # hue shift with height
		"hue_pos": rng.randf_range(0.0, 0.25) if _city else 0.0,  # hue gradient across the grid
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
	var th := tw * 0.5                                  # 2:1 iso
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

	var lit := 0.26 + 0.6 * e
	var left_face := Color.from_hsv(h, 0.6, lit * 0.5)
	var right_face := Color.from_hsv(h, 0.6, lit * 0.78)
	var top_face := Color.from_hsv(h, 0.45, lit + 0.18)

	draw_colored_polygon(PackedVector2Array([w_t, s_t, s_b, w_b]), left_face)
	draw_colored_polygon(PackedVector2Array([s_t, e_t, e_b, s_b]), right_face)
	draw_colored_polygon(PackedVector2Array([n_t, e_t, s_t, w_t]), top_face)
