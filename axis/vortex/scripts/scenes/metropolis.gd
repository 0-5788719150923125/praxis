extends VortexScene

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
var _origins: Array = []
var _beat_prev := 0.0
var _opick := 0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
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
	for y in G:
		for x in G:
			var height := 0.0
			for b: Dictionary in bumps:
				var dx := float(x) / G - float(b.x)
				var dy := float(y) / G - float(b.y)
				height += float(b.a) * exp(-(dx * dx + dy * dy) / (2.0 * float(b.s) * float(b.s)))
			_terrain[y * G + x] = height

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
	var drive := f.energy * 0.7 + f.beat * 0.5
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
	var build_gain := 0.3 + 0.7 * _f.energy

	# Back-to-front so nearer blocks overlap farther ones correctly.
	for gy in G:
		for gx in G:
			var i := gy * G + gx
			var dev := _dev.f[i]
			var pulse := _pulse.f[i]
			var building := dev * build_gain
			var top_z := _terrain[i] * z_ter + building * z_build
			var cx := float(gx - gy) * tw
			var cy := (float(gx + gy) - float(G - 1)) * th - top_z
			var side := building * z_build + 2.0
			var hue := fposmod(lerpf(_hue, _city_hue, dev) + 0.12 * pulse, 1.0)
			var lit := 0.14 + 0.45 * _terrain[i] + 0.35 * dev + 0.6 * pulse
			_block(Vector2(cx, cy), tw, th, side, hue, clampf(lit, 0.05, 1.1))

	_draw_fog(-span * 0.5, span, size.y * 0.5, u)


# One iso block: front-left and front-right side quads, then the top diamond.
func _block(base: Vector2, tw: float, th: float, side: float, hue: float, lit: float) -> void:
	var down := Vector2(0, side)
	var n_t := base + Vector2(0, -th)
	var e_t := base + Vector2(tw, 0)
	var s_t := base + Vector2(0, th)
	var w_t := base + Vector2(-tw, 0)
	draw_colored_polygon(PackedVector2Array([w_t, s_t, s_t + down, w_t + down]),
		Color.from_hsv(hue, 0.5, lit * 0.55))
	draw_colored_polygon(PackedVector2Array([s_t, e_t, e_t + down, s_t + down]),
		Color.from_hsv(hue, 0.5, lit * 0.75))
	draw_colored_polygon(PackedVector2Array([n_t, e_t, s_t, w_t]),
		Color.from_hsv(hue, 0.4, clampf(lit + 0.12, 0.0, 1.0)))


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
