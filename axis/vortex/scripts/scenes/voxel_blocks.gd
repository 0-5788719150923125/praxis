extends VortexScene

## Voxel blocks - an isometric Minecraft heightfield.
##
## A grid of iso cubes whose stack height rises and falls with the spectrum, like
## a 3D equalizer terrain. Each cube is three flat faces (a bright top and two
## shaded sides), painted back-to-front so they overlap correctly. There is always
## a one-block floor, so the grid reads even in silence - no black frames.

var _f: AudioFeatures = AudioFeatures.new()
var _h := PackedFloat32Array()    # eased column heights, 0..1
var _act: Activation


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var grid := rng.randi_range(6, 9)
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.35, 0.7)
	_act = Activation.new(grid * grid, rng, sparsity)
	return {
		"grid": grid,
		"tile": rng.randf_range(0.060, 0.085),   # tile half-width, fraction of unit
		"max_h": rng.randf_range(0.10, 0.18),    # tallest stack, fraction of unit
		"hue": rng.randf(),
		"hue_h": rng.randf_range(0.10, 0.40),    # hue shift with height
		"swirl": rng.randf() < 0.5,              # radial vs. linear spectrum mapping
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	_act.update(f.energy + 0.4 * f.beat, delta)
	var g := int(params.grid)
	if _h.size() != g * g:
		_h.resize(g * g)
	var swirl: bool = params.swirl
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
			# Rooted columns stay low; activated ones rise with the spectrum.
			var target: float = _f.sample(clampf(t, 0.0, 1.0)) * (0.2 + 0.8 * _act.level(idx))
			_h[idx] = lerpf(_h[idx], target, 0.18)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var g := int(params.grid)
	var tw: float = float(params.tile) * u            # tile half width (x)
	var th := tw * 0.5                                 # tile half height (y), 2:1 iso
	var bh: float = float(params.max_h) * u
	var hue: float = params.hue
	var hue_h: float = params.hue_h

	# Back-to-front: outer gy, inner gx keeps each tile in front of the one behind.
	for gy in g:
		for gx in g:
			var idx := gx * g + gy
			var top := maxf(1.0, _h[idx] * bh)         # min 1px stack -> valid faces
			var cx := float(gx - gy) * tw
			var cy := (float(gx + gy) - float(g - 1)) * th
			_draw_cube(Vector2(cx, cy), tw, th, top, _h[idx], hue, hue_h)


# One iso cube: front-left and front-right side quads, then the top diamond.
func _draw_cube(base: Vector2, tw: float, th: float, top: float, e: float, hue: float, hue_h: float) -> void:
	var h := fposmod(hue + hue_h * e, 1.0)
	var lift := Vector2(0, -top)
	# Diamond corners (n=back, e=right, s=front, w=left) at base and at top.
	var n_b := base + Vector2(0, -th)
	var e_b := base + Vector2(tw, 0)
	var s_b := base + Vector2(0, th)
	var w_b := base + Vector2(-tw, 0)
	var n_t := n_b + lift
	var e_t := e_b + lift
	var s_t := s_b + lift
	var w_t := w_b + lift

	var lit := 0.30 + 0.55 * e
	var left_face := Color.from_hsv(h, 0.6, lit * 0.55)
	var right_face := Color.from_hsv(h, 0.6, lit * 0.78)
	var top_face := Color.from_hsv(h, 0.45, lit + 0.15)

	draw_colored_polygon(PackedVector2Array([w_t, s_t, s_b, w_b]), left_face)
	draw_colored_polygon(PackedVector2Array([s_t, e_t, e_b, s_b]), right_face)
	draw_colored_polygon(PackedVector2Array([n_t, e_t, s_t, w_t]), top_face)
