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
##
## What KIND of place it is is one roll, not several. The hue was already free here,
## but the character never was: always a 48x48 grid on the same rolling hills at the
## same mid saturation, so every seed built the same city in a different colour. A
## [constant CHARACTER] now decides palette, block size, how flat the land is, how
## tall the buildings stand and how broad the hue districts are together - because
## those are the same decision. Ground takes the [Scheme]'s base and the buildings its
## counter hue, which is what the old "+0.3 to 0.6" offset was reaching for.

const OVER := 1.6      # landscape spans this much beyond the frame

## The place. `grid` is blocks per side (block SIZE, so a coarse grid is a village of
## big lots and a fine one a dense downtown); `relief`/`rise` split the vertical budget
## between the land and the buildings; `district` is the hue-zone frequency (low = a
## few broad quarters, high = many small ones) and `vary` how far the hue moves across
## them; `hills`/`hill_s` shape the land the blocks stand on (two broad swells read as
## nearly flat, five tight ones as rolling); `sat` and `fog` finish the air.
const CHARACTER := {
	"countryside": {
		"moods": ["verdant", "dawn", "sodium", "brass", "toxic", "teal", "rose", "bone"],
		"grid": [32, 40], "hills": [4, 6], "hill_s": [0.18, 0.30], "district": [1.3, 2.1],
		"vary": 0.30, "sat": 0.60, "relief": [0.19, 0.25], "rise": [0.06, 0.09], "fog": 0.07},
	"megacity": {
		"moods": ["glacier", "abyss", "ash", "violet", "teal", "bone", "brass", "sodium"],
		"grid": [46, 54], "hills": [2, 3], "hill_s": [0.22, 0.36], "district": [2.6, 3.6],
		"vary": 0.16, "sat": 0.40, "relief": [0.07, 0.12], "rise": [0.15, 0.21], "fog": 0.05},
	"dustbowl": {
		"moods": ["ash", "bone", "sodium", "dawn", "ember", "brass", "rose", "verdant"],
		"grid": [36, 44], "hills": [3, 5], "hill_s": [0.14, 0.24], "district": [1.8, 2.6],
		"vary": 0.12, "sat": 0.20, "relief": [0.14, 0.20], "rise": [0.08, 0.12], "fog": 0.14},
	"neon": {
		"moods": ["magenta", "violet", "toxic", "rose", "abyss", "teal", "ember", "glacier"],
		"grid": [44, 52], "hills": [3, 5], "hill_s": [0.16, 0.26], "district": [3.0, 4.2],
		"vary": 0.42, "sat": 0.80, "relief": [0.10, 0.16], "rise": [0.12, 0.18], "fog": 0.06},
}

var _f: AudioFeatures = AudioFeatures.new()
var _theme := "growth"
var _char := "countryside"
var _g := 48           # grid is _g x _g blocks
var _sat := 0.5        # block saturation - the mood's own character
var _vary := 0.34      # how far the district field swings the hue
var _relief := 0.16    # hill height scale
var _rise := 0.12      # building height scale
var _fog := 0.05       # fog density over the low ground
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
var _forge := FrameForge.new()   # the city is built OFF-THREAD (see FrameForge):
                                 # update() ships a snapshot, _draw() submits the
                                 # finished packet - the UI never waits on tiles


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "swarm"
	framing = "field"
	_theme = "growth" if rng.randf() < 0.5 else "pulse"
	_char = String(CHARACTER.keys()[rng.randi() % CHARACTER.size()])
	var ch: Dictionary = CHARACTER[_char]
	var grid: Array = ch["grid"]
	_g = rng.randi_range(int(grid[0]), int(grid[1]))
	_sat = float(ch["sat"])
	_vary = float(ch["vary"])
	_relief = _band(ch["relief"], rng)
	_rise = _band(ch["rise"], rng)
	_fog = float(ch["fog"])
	# Ground and buildings are the two hues of one mood - the relationship the old
	# "somewhere 0.3 to 0.6 away" offset was approximating, now named.
	var sch := Scheme.among(ch["moods"], rng)
	_hue = sch.vary(rng)                                 # terrain hue
	_city_hue = sch.opposed(_hue)                  # the built city stands against it

	# Terrain: a few Gaussian hills.
	_terrain.resize(_g * _g)
	var bumps: Array = []
	var hills: Array = ch["hills"]
	for b in rng.randi_range(int(hills[0]), int(hills[1])):
		bumps.append({
			"x": rng.randf(), "y": rng.randf(),
			"s": _band(ch["hill_s"], rng), "a": rng.randf_range(0.4, 1.0)})
	# A fine detail field (ground/wall grain) and a low-frequency DISTRICT field (broad hue zones, so
	# neighbouring buildings share a colour and the city reads as quadrants rather than noise).
	var detf := Field.make("fbm", rng.randi(), 9.0, 4)
	var distf := Field.make("fbm", rng.randi(), _band(ch["district"], rng), 3)
	_detail.resize(_g * _g)
	_district.resize(_g * _g)
	for y in _g:
		for x in _g:
			var height := 0.0
			for b: Dictionary in bumps:
				var dx := float(x) / _g - float(b.x)
				var dy := float(y) / _g - float(b.y)
				height += float(b.a) * exp(-(dx * dx + dy * dy) / (2.0 * float(b.s) * float(b.s)))
			_terrain[y * _g + x] = height
			var p := Vector2(float(x) / float(_g), float(y) / float(_g)) * 2.0 - Vector2.ONE
			_detail[y * _g + x] = detf.at(p)
			_district[y * _g + x] = distf.at(p)

	_dev = Swarm.new(_g, _g, Swarm.GROW, rng, rng.randi_range(2, 4))
	if _theme == "pulse":
		for i in _g * _g:                                # already a built city
			_dev.f[i] = 1.0
	_pulse = Swarm.new(_g, _g, Swarm.WAVE, rng, 0)
	for o in rng.randi_range(2, 4):                       # pulse origins
		_origins.append(Vector2i(rng.randi_range(4, _g - 4), rng.randi_range(4, _g - 4)))
	return {"character": _char, "mood": sch.name, "theme": _theme, "grid": _g,
		"hue": _hue, "city_hue": _city_hue, "sat": _sat, "relief": _relief,
		"rise": _rise, "hills": bumps.size()}


# A value from a [lo, hi] table band.
func _band(b, rng: RandomNumberGenerator) -> float:
	var a: Array = b
	return rng.randf_range(float(a[0]), float(a[1]))




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
	# ship the frame's inputs to the worker: swarm fields DUPLICATED (main
	# keeps stepping them), terrain/detail/district by reference (immutable
	# after build), AudioFeatures fresh-per-frame upstream. The forge
	# queue_redraws when the packet lands.
	_forge.kick(Callable(get_script(), "_build_city"), {
		"f": f, "g": _g, "sizex": size.x, "u": unit(),
		"dev": _dev.f.duplicate(), "pulse": _pulse.f.duplicate(),
		"terrain": _terrain, "detail": _detail, "district": _district,
		"hue": _hue, "city_hue": _city_hue,
		"sat": _sat, "vary": _vary, "relief": _relief, "rise": _rise,
	}, self)


func _draw() -> void:
	begin_draw()
	_forge.submit(self)              # the whole city: prebuilt off-thread
	_draw_fog(-size.x * OVER * 0.5, size.x * OVER, size.y * 0.5, unit())


## The city builder - STATIC and PURE (the FrameForge contract): touches only
## the snapshot, runs on a worker thread live (synchronously in exports, for
## determinism), returns one packet chunk. This is the code that used to
## block the UI for the whole tile loop.
static func _build_city(s: Dictionary) -> Array:
	var f: AudioFeatures = s.f
	var g: int = s.g
	var tb := TriBatch.new()
	var span: float = float(s.sizex) * OVER
	var tw := span / float(g) * 0.5           # tile half-width
	var th := tw * 0.5                        # 2:1 iso
	var z_ter: float = float(s.u) * float(s.relief)   # hill height scale
	var z_build: float = float(s.u) * float(s.rise)   # building height scale
	var sat: float = float(s.sat)
	var vary: float = float(s.vary)
	var build_gain := 0.55 + 0.3 * f.energy
	var dev_f: PackedFloat32Array = s.dev
	var pulse_f: PackedFloat32Array = s.pulse
	var terrain: PackedFloat32Array = s.terrain
	var detail: PackedFloat32Array = s.detail
	var district_f: PackedFloat32Array = s.district
	var hue0: float = s.hue
	var city_hue: float = s.city_hue

	# Back-to-front so nearer blocks overlap farther ones correctly.
	for gy in g:
		for gx in g:
			var i := gy * g + gx
			var dev := dev_f[i]
			var pulse := pulse_f[i]
			# Each block bounces with its own spectral band - responsiveness, so the
			# city moves with the music instead of standing as one static slab.
			var react := f.sample(clampf(terrain[i] * 0.85, 0.0, 1.0))
			var building := dev * (build_gain + 0.7 * react)
			var top_z := terrain[i] * z_ter + building * z_build
			var cx := float(gx - gy) * tw
			var cy := (float(gx + gy) - float(g - 1)) * th - top_z
			var side := building * z_build + 2.0
			var det: float = detail[i]
			# Hue: a DISTRICT base (broad zones so neighbours match) drifting with height, development
			# and the colour pulse - a real varied field, and buildings vs ground read differently.
			var district: float = district_f[i] - 0.5
			var is_bldg := building > 0.12
			var base_h: float = city_hue if is_bldg else hue0
			var hue := fposmod(base_h + vary * district + 0.12 * terrain[i] + 0.22 * dev + 0.20 * pulse, 1.0)
			# Value: terrain + development + audio + pulse, grained by the detail texture.
			var lit := 0.08 + 0.52 * terrain[i] + 0.28 * dev + 0.5 * react + 0.7 * pulse + 0.28 * (det - 0.5)
			# Cast shadow: taller buildings drop a soft dark diamond onto the ground toward the back-left
			# (light reads from the front-right), grounding them instead of floating on flat colour.
			if building > 0.25:
				var so := building * z_build
				var shp := Vector2(cx - tw * 0.5 - so * 0.35, cy + top_z - th * 0.4 - so * 0.15)
				tb.quad(shp + Vector2(0, -th), shp + Vector2(tw, 0),
					shp + Vector2(0, th), shp + Vector2(-tw, 0),
					Color(0, 0, 0, clampf(0.10 + 0.22 * clampf(building, 0.0, 1.5), 0.0, 0.4)))
			_block(tb, Vector2(cx, cy), tw, th, side, hue, clampf(lit, 0.05, 1.15), sat, building, det)
	return [{"pts": tb.pts, "cols": tb.cols, "idx": tb.idx}]


# One iso block: front-left and front-right side quads (with floor/window detail on tall ones),
# then the top diamond.
static func _block(tb: TriBatch, base: Vector2, tw: float, th: float, side: float,
		hue: float, lit: float, sat := 0.5, building := 0.0, det := 0.5) -> void:
	var down := Vector2(0, side)
	var n_t := base + Vector2(0, -th)
	var e_t := base + Vector2(tw, 0)
	var s_t := base + Vector2(0, th)
	var w_t := base + Vector2(-tw, 0)
	# Saturation comes from the character's mood, so a dustbowl block is concrete and a
	# neon one is lit signage; the top face stays a little paler, as it always did.
	tb.quad(w_t, s_t, s_t + down, w_t + down, Color.from_hsv(hue, sat, lit * 0.55))
	tb.quad(s_t, e_t, e_t + down, s_t + down, Color.from_hsv(hue, sat, lit * 0.75))
	# Building features: FLOOR bands + a few LIT WINDOWS on the two visible walls, so a tall block
	# reads as a building with structure instead of a flat coloured slab. Only tall blocks, capped.
	if building > 0.3 and side > 14.0:
		var floors := clampi(int(side / 8.0), 2, 7)
		_wall(tb, w_t, s_t, down, floors, hue, lit * 0.55, sat, det)          # left wall
		_wall(tb, s_t, e_t, down, floors, hue, lit * 0.75, sat, det + 0.37)   # right wall
	tb.quad(n_t, e_t, s_t, w_t, Color.from_hsv(hue, sat * 0.8, clampf(lit + 0.12, 0.0, 1.0)))


# Draw floor bands + scattered lit windows on one skewed wall face (top edge a->b, height `down`).
static func _wall(tb: TriBatch, a: Vector2, b: Vector2, down: Vector2, floors: int,
		hue: float, wlit: float, sat: float, seed: float) -> void:
	var fade := Color(0, 0, 0, 0.18)
	for f in range(1, floors):
		var v := float(f) / float(floors)
		# Hard-edged on purpose: these are faint dark bands, several thousand of them on a dense
		# grid, and they gain nothing from the feather that a wireframe edge does (see TriBatch.line).
		tb.line(a + down * v, b + down * v, fade, 1.0, false)   # floor band (a thin dark line)
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
			tb.quad(p0, p1, p2, p3,
				Color.from_hsv(fposmod(hue + 0.04, 1.0), sat * 0.6, clampf(wlit * (1.4 + 0.8 * h), 0.0, 1.0)))


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
		# The fog is the ground's own colour, thinned - a dustbowl's air hangs heavier
		# than a megacity's, so the character sets how much of it there is.
		draw_colored_polygon(pts, Color.from_hsv(_hue, _sat * 0.2, 0.9,
			_fog + 0.02 * _f.low_mid))
