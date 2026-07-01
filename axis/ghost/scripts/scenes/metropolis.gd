extends GhostScene

## Metropolis - a city of thousands of blocks growing over a countryside.
##
## A large isometric landscape: Gaussian hills carry a grid of blocks whose heights
## and colours are driven by a [Swarm] field - no per-block scripting. By seed:
##   growth - development creeps outward from a few seeds, the city spreading across
##            the hills over time, breathing with the music.
##   pulse  - the city is already built; colour pulses are injected on the beat and
##            ripple outward across the blocks as expanding fronts.
## A second WAVE swarm always carries the colour pulses. Fog pools in the low
## ground. The grid is drawn far larger than the frame, so it runs off every edge.

const G := 48          # grid is G x G blocks
const OVER := 1.6      # landscape spans this much beyond the frame

var _f: AudioFeatures = AudioFeatures.new()
var _theme := "growth"
var _hue := 0.0
var _city_hue := 0.0
var _dev: Swarm         # development / building height
var _pulse: Swarm       # colour wave
var _terrain := PackedFloat32Array()
var _detail := PackedFloat32Array()    # per-tile ground/wall grain (texture)
var _district := PackedFloat32Array()  # per-tile low-freq hue zone (city QUADRANTS - siblings share a hue)
var _origins: Array = []
var _beat_prev := 0.0
var _opick := 0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "swarm"
	framing = "field"
	_theme = "growth" if rng.randf() < 0.5 else "pulse"
	_hue = rng.randf()                                   # terrain hue
	_city_hue = fposmod(_hue + rng.randf_range(0.3, 0.6), 1.0)

	# Terrain: a few Gaussian hills.
	_terrain.resize(G * G)
	var bumps: Array = []
	for b in rng.randi_range(3, 6):
		bumps.append({
			"x": rng.randf(), "y": rng.randf(),
			"s": rng.randf_range(0.10, 0.28), "a": rng.randf_range(0.4, 1.0)})
	# A fine detail field (ground/wall grain) and a low-frequency DISTRICT field (broad hue zones, so
	# neighbouring buildings share a colour and the city reads as quadrants rather than noise).
	var detf := Field.make("fbm", rng.randi(), 9.0, 4)
	var distf := Field.make("fbm", rng.randi(), 2.0, 3)
	_detail.resize(G * G)
	_district.resize(G * G)
	for y in G:
		for x in G:
			var height := 0.0
			for b: Dictionary in bumps:
				var dx := float(x) / G - float(b.x)
				var dy := float(y) / G - float(b.y)
				height += float(b.a) * exp(-(dx * dx + dy * dy) / (2.0 * float(b.s) * float(b.s)))
			_terrain[y * G + x] = height
			var p := Vector2(float(x) / float(G), float(y) / float(G)) * 2.0 - Vector2.ONE
			_detail[y * G + x] = detf.at(p)
			_district[y * G + x] = distf.at(p)

	_dev = Swarm.new(G, G, Swarm.GROW, rng, rng.randi_range(2, 4))
	if _theme == "pulse":
		for i in G * G:                                  # already a built city
			_dev.f[i] = 1.0
	_pulse = Swarm.new(G, G, Swarm.WAVE, rng, 0)
	for o in rng.randi_range(2, 4):                       # pulse origins
		_origins.append(Vector2i(rng.randi_range(4, G - 4), rng.randi_range(4, G - 4)))
	return {}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	var drive := f.energy * 0.8 + f.beat * 0.6
	if _theme == "growth":
		_dev.step(drive, delta, 0.02)    # grows, slight decay so it breathes
	# Inject a colour pulse at one origin on each beat, then let it ripple outward.
	if f.beat > 0.5 and _beat_prev <= 0.5 and _origins.size() > 0:
		var o: Vector2i = _origins[_opick % _origins.size()]
		_pulse.inject(o.x, o.y, 1.0)
		_opick += 1
	_pulse.step(0.0, delta, 1.1)
	_beat_prev = f.beat
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var span := size.x * OVER
	var tw := span / float(G) * 0.5         # tile half-width
	var th := tw * 0.5                        # 2:1 iso
	var z_ter := u * 0.16                     # hill height scale
	var z_build := u * 0.12                   # building height scale
	var build_gain := 0.55 + 0.3 * _f.energy

	# Back-to-front so nearer blocks overlap farther ones correctly.
	for gy in G:
		for gx in G:
			var i := gy * G + gx
			var dev := _dev.f[i]
			var pulse := _pulse.f[i]
			# Each block bounces with its own spectral band - responsiveness, so the
			# city moves with the music instead of standing as one static slab.
			var react := _f.sample(clampf(_terrain[i] * 0.85, 0.0, 1.0))
			var building := dev * (build_gain + 0.7 * react)
			var top_z := _terrain[i] * z_ter + building * z_build
			var cx := float(gx - gy) * tw
			var cy := (float(gx + gy) - float(G - 1)) * th - top_z
			var side := building * z_build + 2.0
			var det: float = _detail[i]
			# Hue: a DISTRICT base (broad zones so neighbours match) drifting with height, development
			# and the colour pulse - a real varied field, and buildings vs ground read differently.
			var district: float = _district[i] - 0.5
			var is_bldg := building > 0.12
			var base_h: float = _city_hue if is_bldg else _hue
			var hue := fposmod(base_h + 0.34 * district + 0.12 * _terrain[i] + 0.22 * dev + 0.20 * pulse, 1.0)
			# Value: terrain + development + audio + pulse, grained by the detail texture.
			var lit := 0.08 + 0.52 * _terrain[i] + 0.28 * dev + 0.5 * react + 0.7 * pulse + 0.28 * (det - 0.5)
			# Cast shadow: taller buildings drop a soft dark diamond onto the ground toward the back-left
			# (light reads from the front-right), grounding them instead of floating on flat colour.
			if building > 0.25:
				var so := building * z_build
				var shp := Vector2(cx - tw * 0.5 - so * 0.35, cy + top_z - th * 0.4 - so * 0.15)
				draw_colored_polygon(PackedVector2Array([
					shp + Vector2(0, -th), shp + Vector2(tw, 0), shp + Vector2(0, th), shp + Vector2(-tw, 0)]),
					Color(0, 0, 0, clampf(0.10 + 0.22 * clampf(building, 0.0, 1.5), 0.0, 0.4)))
			_block(Vector2(cx, cy), tw, th, side, hue, clampf(lit, 0.05, 1.15), building, det)

	_draw_fog(-span * 0.5, span, size.y * 0.5, u)


# One iso block: front-left and front-right side quads (with floor/window detail on tall ones),
# then the top diamond.
func _block(base: Vector2, tw: float, th: float, side: float, hue: float, lit: float,
		building := 0.0, det := 0.5) -> void:
	var down := Vector2(0, side)
	var n_t := base + Vector2(0, -th)
	var e_t := base + Vector2(tw, 0)
	var s_t := base + Vector2(0, th)
	var w_t := base + Vector2(-tw, 0)
	draw_colored_polygon(PackedVector2Array([w_t, s_t, s_t + down, w_t + down]),
		Color.from_hsv(hue, 0.5, lit * 0.55))
	draw_colored_polygon(PackedVector2Array([s_t, e_t, e_t + down, s_t + down]),
		Color.from_hsv(hue, 0.5, lit * 0.75))
	# Building features: FLOOR bands + a few LIT WINDOWS on the two visible walls, so a tall block
	# reads as a building with structure instead of a flat coloured slab. Only tall blocks, capped.
	if building > 0.3 and side > 14.0:
		var floors := clampi(int(side / 8.0), 2, 7)
		_wall(w_t, s_t, down, floors, hue, lit * 0.55, det)          # left wall
		_wall(s_t, e_t, down, floors, hue, lit * 0.75, det + 0.37)   # right wall
	draw_colored_polygon(PackedVector2Array([n_t, e_t, s_t, w_t]),
		Color.from_hsv(hue, 0.4, clampf(lit + 0.12, 0.0, 1.0)))


# Draw floor bands + scattered lit windows on one skewed wall face (top edge a->b, height `down`).
func _wall(a: Vector2, b: Vector2, down: Vector2, floors: int, hue: float, wlit: float, seed: float) -> void:
	var fade := Color(0, 0, 0, 0.18)
	for f in range(1, floors):
		var v := float(f) / float(floors)
		draw_line(a + down * v, b + down * v, fade, 1.0, true)        # floor band (a thin dark line)
	# A few windows, lit or dark by a stable per-cell hash (no flicker), on the mid floors.
	var cols := 2
	for f in floors:
		for c in cols:
			var h := fposmod(sin((seed + float(f) * 3.3 + float(c) * 1.7) * 12.9898) * 43758.5, 1.0)
			if h < 0.45:
				continue                                             # a dark (unlit) window - skip
			var uc := (float(c) + 0.5) / float(cols)
			var vf := (float(f) + 0.5) / float(floors)
			var hw := 0.5 / float(cols) * 0.5
			var hh := 0.5 / float(floors) * 0.5
			var p0 := a.lerp(b, uc - hw) + down * (vf - hh)
			var p1 := a.lerp(b, uc + hw) + down * (vf - hh)
			var p2 := a.lerp(b, uc + hw) + down * (vf + hh)
			var p3 := a.lerp(b, uc - hw) + down * (vf + hh)
			draw_colored_polygon(PackedVector2Array([p0, p1, p2, p3]),
				Color.from_hsv(fposmod(hue + 0.04, 1.0), 0.3, clampf(wlit * (1.4 + 0.8 * h), 0.0, 1.0)))


# Translucent fog flowing across the low ground.
func _draw_fog(left: float, w: float, foot: float, u: float) -> void:
	var cols := 40
	for layer in 3:
		var pts := PackedVector2Array()
		pts.resize(cols + 2)
		for c in cols:
			var fx := float(c) / float(cols - 1)
			var ripple := sin((fx * 3.5 + _f.time * 0.25 + layer * 0.7) * TAU)
			var y := foot - u * (0.20 - 0.05 * layer) + ripple * u * 0.02
			pts[c] = Vector2(left + fx * w, minf(y, foot - 1.0))
		pts[cols] = Vector2(left + w, foot)
		pts[cols + 1] = Vector2(left, foot)
		draw_colored_polygon(pts, Color.from_hsv(_hue, 0.08, 0.9, 0.05 + 0.02 * _f.low_mid))
