extends GhostScene

## Falling sand - a world of matter you never draw, only rule.
##
## A chamber seen straight on, its walls running off every edge of the frame. Two to five
## spouts pour into it and never stop: dry sand builds a cone that holds its flank angle
## until one grain too many starts an avalanche running down the side; water finds its
## level and soaks into whatever it lands on; oil, being lighter, floats on the water as a
## thin film; an ember dropped into that oil eats through it and leaves ash. The pile builds
## all song. On a beat a hatch opens in the floor and the column drains, a collapse front
## travelling UPWARD through material that was settled a second ago.
##
## It is the catalogue's first scene whose every frame is a STATE rather than a picture.
## Everything else here draws something - a skyline, a lattice, a flame. This draws a
## [Grains] grid where each cell holds exactly one thing, and its silhouette is produced by
## nothing but local rules applied a few thousand times. The angle of repose is a slide
## probability, the avalanche is what happens when a cone gets steeper than it, the oil
## film is a density comparison, and the drainage collapse is the absence of a floor. None
## of the four has a branch of its own anywhere in either file, which is the entire point:
## it is a world with rules rather than a picture of one.
##
## WHAT THE SEED DECIDES. The chamber (a straight silo, an hourglass whose waist chokes the
## pour, stepped ledges the material cascades down, or open sides it walks off), its grid
## and wall thickness; a two-to-four material subset of [constant Grains.MATTER] with every
## property of every material sampled from its own range - so this sand is not the last
## sand, it has its own repose, density, colour and susceptibility to the draft; the ember
## and ash chain, added only when a flammable material was drawn; the spouts, each with its
## own column, width, material, spectral band, gain and slow duty cycle; the hatch
## positions, widths and dwell; how far gravity may tilt and for how long; a steady ambient
## draft and its direction; the wall strata ramp; and whether the air behind it all is a
## bright ground or a dark one.
##
## AUDIO. Each spout owns a FIXED band, and its rate in grains per second is its gain times
## that band's level, so the pours move against each other rather than throbbing together -
## the bass spout runs while the treble spout is dry. A beat on an armed latch opens one
## floor hatch. f.movement past a sampled gate eases gravity off vertical by a sampled
## ANGLE (0.12-0.35 rad, applied as its tangent, which is the per-step sideways probability
## that a cellular gravity actually takes), so a section change shears the whole pile and it
## re-settles into a different landscape. f.flux scales ember ignition. The tonal centre
## from chroma_hue() drags every material's hue by a quarter of its strength, and f.energy
## feeds [method Nonlinear.flare] into the emissive materials' VALUE only. Nothing anywhere
## touches grain size or grid scale: the sound colours this world and gates its activity, it
## does not resize it.
##
## MEASURED CONSTANTS. f.energy is a mean over 64 bands and rarely passes 0.5, and f.flux
## lives around 0.01-0.05, so the gains here are written for those ranges rather than for a
## nominal 0-1 (the ember drive is 0.35 + 8*flux, not 1 + flux). Spout rates carry a 0.12
## floor so a quiet passage still trickles instead of freezing the world.
##
## COST. The sim and the batching BOTH run inside a [FrameForge] job that owns the grid, and
## the scene never touches a cell - a 200x115 chamber is 23000 cells, and at even half a
## microsecond each that is past the main thread's whole budget. The sim advances on a
## [SimClock] rather than per update() call, which is not optional: the Director sub-steps
## update() up to fifteen times in one frame, pre-warms every new scene twelve times before
## it is ever drawn, and an Echo re-localize can fast-forward hundreds of calls - any of the
## three would have the chamber full before the first frame anyone sees.

const OVER := 1.12          # the chamber runs this much past the frame's width

## Ceiling on sim ticks handed to one build. SimClock already caps each update() call; this
## caps what several capped calls may ACCUMULATE between two drawn frames, which is what the
## pre-warm and an Echo jump actually produce.
const MAX_PENDING := 8

## The chamber, which is the thing that decides what the material DOES on the way down.
## A silo is a plain column and the pile is one cone; an hourglass chokes the pour at a
## waist and holds a second pile above it; ledges break the fall into cascades; open sides
## let everything walk off the edge, so it never fills and reads as an outdoor drift.
## The chamber roll, weighted. See where it is used.
const SHAPE_BAG := ["ledges", "ledges", "ledges", "hourglass", "hourglass", "silo", "open"]

const CHAMBERS := {
	"silo": {"wall": [2, 4], "pillars": [0, 2], "hatch_n": [1, 2], "hatch_w": [0.09, 0.24]},
	"hourglass": {"wall": [2, 3], "waist": [0.44, 0.62], "gap": [0.045, 0.13], "span": [0.16, 0.30],
		"hatch_n": [1, 2], "hatch_w": [0.10, 0.26]},
	"ledges": {"wall": [2, 4], "shelves": [2, 5], "shelf": [0.20, 0.46], "hatch_n": [1, 3],
		"hatch_w": [0.07, 0.19]},
	"open": {"wall": [2, 3], "hatch_n": [1, 3], "hatch_w": [0.08, 0.20]},
}

## What can be poured. The rest of the matter table (ember, ash, steam) arrives only as a
## CONSEQUENCE - nothing pours ash, ash is what is left over.
const POURABLE := ["sand", "water", "oil", "salt", "dust"]

## Elevation ramps for the chamber wall, so the stone reads as strata rather than as one
## flat grey behind the material.
const WALL_RAMPS := ["earth", "desert", "alpine", "ocean"]

var _w := 192
var _h := 108
var _cell := 6.0
var _org := Vector2.ZERO
var _clock: SimClock
var _forge := FrameForge.new()
var _job: SandJob
var _spouts: Array = []            # [{x0,x1,mat,band,gain,period,phase,duty}]
var _hatches: Array = []           # [{x0,x1}]
var _hatch_i := 0
var _hatch_gap := 5.0
var _hatch_dwell := 0.6
var _hatch_left := 0.0
var _hatch_arm := 0.0
var _drain_x0 := 0
var _drain_x1 := -1
var _beat_prev := 0.0
var _tilt := 0.0
var _tilt_target := 0.0
var _tilt_mag := 0.2
var _tilt_gate := 0.5
var _tilt_dwell := 3.0
var _tilt_rate := 1.2
var _tilt_left := 0.0
var _tilt_signs: Array = []
var _tilt_i := 0
var _wind := 0.0
var _wind_dir := 1
var _ignite := 0.5
var _glow := 0.0
var _hue_base := 0.1
## Guaranteed separation in VALUE between the air and the mean of the materials falling
## through it. 0.34 is a shade over a third of the axis: comfortably visible at every
## saturation, and still leaving both ends room to be sampled rather than pinned.
const AIR_CONTRAST := 0.34

var _air_sat := 0.08
var _air_val := 0.9
var _air_hue := 0.5
var _depth := 0.28
var _run_cap := 8
var _ch := Vector2.ZERO


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "swarm"
	framing = "field"
	var sch := Scheme.pick(rng)
	_hue_base = sch.hue
	# The grid stays landscape and modest: the sweep is O(cells) and runs every tick, and
	# past ~25k the worker stops keeping up with a 30 Hz clock on a busy chamber.
	# FINER GRAIN. This was 160-216 wide, which puts a cell at 9-12 screen pixels on a
	# 1920 frame - coarse enough that the sand read as tiles rather than as sand, which is
	# what "rather large particles" meant. 240-300 lands a cell at 6.4-8.0 px. The sweep is
	# O(cells) and measured at 0.53 us/cell worst case (a half-full grid, every row
	# active), so the ceiling is real - but a chamber is mostly empty or settled and the
	# per-row activity mask skips those, and the whole sim runs inside a FrameForge job
	# where falling behind costs a slower pour rather than a dropped frame.
	_w = rng.randi_range(240, 300)
	_h = clampi(int(round(float(_w) * rng.randf_range(0.55, 0.60))), 132, 180)
	_run_cap = rng.randi_range(5, 11)
	_depth = rng.randf_range(0.16, 0.36)
	# 20-30 ticks a second. A granular automaton does not need 60: the grid IS the
	# quantization, and a slower clock buys back the cells it costs.
	_clock = SimClock.new(rng.randf_range(20.0, 30.0), 2)

	# WEIGHTED, not uniform. "ledges" was one of four shapes picked evenly, so three
	# sessions in four were a plain box and the sand simply heaped in the bottom of it -
	# reported, fairly, as boring. The cascading shapes are what this scene is FOR: matter
	# that runs along a surface, reaches an edge, and pours off it.
	var shape := String(SHAPE_BAG[rng.randi() % SHAPE_BAG.size()])
	var ch: Dictionary = CHAMBERS[shape]
	var thick := _rint(ch["wall"], rng)

	var sim := Grains.new(_w, _h, rng.randi())
	var stone := sim.add("stone", rng)
	sim.wall_id = stone
	sim.floor_h = thick
	sim.spill = shape == "open"

	# --- the matter subset -----------------------------------------------------------
	var bag: Array = POURABLE.duplicate()
	for i in range(bag.size() - 1, 0, -1):          # a seeded shuffle, not a sort
		var j := rng.randi_range(0, i)
		var tmp = bag[i]
		bag[i] = bag[j]
		bag[j] = tmp
	if rng.randf() < 0.8 and bag.find("sand") > 0:  # sand is the scene's name, usually
		var k := bag.find("sand")
		bag[k] = bag[0]
		bag[0] = "sand"
	var pours: Array = []
	for i in mini(rng.randi_range(2, 4), bag.size()):
		pours.append(String(bag[i]))
	var fuelly := false
	for n in pours:
		if Grains.MATTER[String(n)].has("fuel"):
			fuelly = true
	var mats: Array = []
	for n in pours:
		mats.append(String(n))
		sim.add(String(n), rng)
	# Ember and ash exist only as the consequence of a flammable pour, and steam only
	# where an ember can reach water.
	if fuelly:
		sim.add("ember", rng)
		sim.add("ash", rng)
		mats.append("ember")
		mats.append("ash")
		if pours.has("water"):
			sim.add("steam", rng)
			mats.append("steam")
	sim.finalize()

	# --- the chamber -----------------------------------------------------------------
	sim.fill_rect(0, 0, _w - 1, thick - 1, stone)
	sim.fill_rect(0, _h - thick, _w - 1, _h - 1, stone)
	if shape != "open":
		sim.fill_rect(0, 0, thick - 1, _h - 1, stone)
		sim.fill_rect(_w - thick, 0, _w - 1, _h - 1, stone)
	match shape:
		"hourglass":
			var wy := int(float(_h) * _rand(ch["waist"], rng))
			var gap := int(float(_w) * _rand(ch["gap"], rng))
			var span := maxi(3, int(float(_h) * _rand(ch["span"], rng)))
			var mouth := int(float(_w) * 0.42)
			for y in range(wy - span, wy + span + 1):
				if y < thick or y >= _h - thick:
					continue
				var u := float(absi(y - wy)) / float(span)
				var half := int(lerpf(float(gap) * 0.5, float(mouth), u * u))
				sim.fill_rect(0, y, _w / 2 - half, y, stone)
				sim.fill_rect(_w / 2 + half, y, _w - 1, y, stone)
		"ledges":
			_cascade(sim, stone, thick, _rint(ch["shelves"], rng), ch, rng)
		"silo":
			for k in _rint(ch["pillars"], rng):
				var px := int(float(_w) * rng.randf_range(0.25, 0.75))
				var py := int(float(_h) * rng.randf_range(0.35, 0.72))
				sim.fill_rect(px, py, px + 1, py + int(float(_h) * rng.randf_range(0.08, 0.20)), stone)

	# EVERY chamber gets something to spill over, not just the one named for it. A silo
	# with a couple of shelves in it is still a silo, and it gives the pour somewhere to
	# break; an hourglass with a shelf under its waist catches the throat's output and
	# throws it sideways. Only the count differs by shape.
	if shape != "ledges":
		var extra := 1 if shape == "open" else rng.randi_range(1, 3)
		if extra > 0:
			_cascade(sim, stone, thick, extra, CHAMBERS["ledges"], rng)

	# Strata: the wall takes its colour from an elevation ramp by row, which is the one
	# place a [Palette] belongs here - everything else is matter, and matter has its own.
	var ramp := Palette.named(String(WALL_RAMPS[rng.randi() % WALL_RAMPS.size()]), rng)
	var wall_v := rng.randf_range(0.30, 0.62)
	var band := PackedColorArray()
	band.resize(_h)
	for y in _h:
		var c := ramp.at(1.0 - float(y) / float(_h))
		band[y] = Color.from_hsv(c.h, clampf(c.s * 0.75, 0.0, 1.0),
			clampf(c.v * wall_v, 0.0, 1.0))
	sim.wall_band = band

	# --- spouts ----------------------------------------------------------------------
	var ns := rng.randi_range(2, 5)
	for i in ns:
		var mname := String(pours[i % pours.size()])
		var cx := int(float(_w) * rng.randf_range(0.12, 0.88))
		var half_w := rng.randi_range(0, 2)
		var x0 := clampi(cx - half_w, thick, _w - 1 - thick)
		var x1 := clampi(cx + half_w, thick, _w - 1 - thick)
		# Bands are SPREAD across the spectrum rather than drawn freely, so two spouts
		# cannot end up on the same frequency and pour as one.
		var band_t := clampf((float(i) + 0.5) / float(ns) + rng.randf_range(-0.08, 0.08), 0.0, 1.0)
		_spouts.append({
			"x0": x0, "x1": x1, "y": 0, "mat": sim.id_of(mname), "name": mname,
			"band": band_t, "gain": rng.randf_range(70.0, 240.0),
			"period": rng.randf_range(5.0, 16.0), "phase": rng.randf(),
			"duty": rng.randf_range(0.55, 1.0)})
		sim.fill_rect(x0, 0, x1, thick - 1, 0)         # a pipe through the ceiling
	if fuelly:
		# The ignition source is its own thin, intermittent spout on the top band - a
		# torch, not a firehose. Rate is deliberately an order of magnitude below a pour.
		var ex := int(float(_w) * rng.randf_range(0.18, 0.82))
		var ex0 := clampi(ex, thick, _w - 1 - thick)
		_spouts.append({
			"x0": ex0, "x1": ex0, "y": 0, "mat": sim.id_of("ember"), "name": "ember",
			"band": rng.randf_range(0.7, 0.97), "gain": rng.randf_range(4.0, 18.0),
			"period": rng.randf_range(6.0, 18.0), "phase": rng.randf(),
			"duty": rng.randf_range(0.15, 0.45)})
		sim.fill_rect(ex0, 0, ex0, thick - 1, 0)

	# --- hatches, tilt, draft ---------------------------------------------------------
	var nh := _rint(ch["hatch_n"], rng)
	for i in nh:
		var hw := maxi(2, int(float(_w) * _rand(ch["hatch_w"], rng)))
		var hx := int(float(_w) * rng.randf_range(0.18, 0.82))
		_hatches.append({"x0": maxi(1, hx - hw / 2), "x1": mini(_w - 2, hx + hw / 2)})
	_hatch_gap = rng.randf_range(3.5, 10.0)
	_hatch_dwell = rng.randf_range(0.35, 1.3)
	# The tilt is sampled as an ANGLE off vertical and applied as its tangent, because a
	# cellular gravity cannot rotate - it can only take a sideways step with some
	# probability, and tan(angle) IS that probability for a given lean.
	_tilt_mag = tan(rng.randf_range(0.12, 0.35))
	_tilt_gate = rng.randf_range(0.34, 0.68)
	_tilt_dwell = rng.randf_range(1.5, 5.0)
	_tilt_rate = 1.0 / rng.randf_range(0.6, 2.0)
	for i in 6:
		_tilt_signs.append(1 if rng.randf() < 0.5 else -1)
	_wind = 0.0 if rng.randf() < 0.35 else rng.randf_range(0.05, 0.30)
	_wind_dir = 1 if rng.randf() < 0.5 else -1
	_ignite = rng.randf_range(0.25, 0.75)

	# --- air ----------------------------------------------------------------------------
	# CHOSEN AGAINST THE SAND, never blind of it. The air used to be a coin between a
	# near-white and a near-black at almost no saturation, drawn with no reference to what
	# was going to fall through it - so the one thing that actually matters here, that the
	# grains and the chamber stay legible, was left to luck. It also read as a blank page
	# far more often than the 45% coin suggests, because white is what the eye reports
	# when it cannot find the picture.
	#
	# So: measure the materials this chamber actually got, then put the air a guaranteed
	# distance away in VALUE - which is the axis legibility lives on - and let hue and
	# saturation open right up now that contrast is no longer their job. The hue is the
	# mood's OPPOSED one rather than its base, so the background separates from the sand
	# in chroma as well as in brightness instead of sitting a shade off it.
	var sand_v := 0.0
	for m in mats:
		var mv = Grains.MATTER.get(String(m), {}).get("val", [0.5, 0.7])
		sand_v += (float(mv[0]) + float(mv[1])) * 0.5
	sand_v /= maxf(1.0, float(mats.size()))
	# TAKE THE SIDE THAT HAS ROOM. Dark is the better picture - the sand glows out of it
	# and an ember actually reads as hot - so it is the default rather than a coin flip.
	# But "always darker than the sand" is not a guarantee that can always be honoured:
	# with a chamber of ash and stone the mean material value is already down at 0.2, and
	# there is no room left underneath it for a full margin. Measured over 400 seeds, the
	# first version of this promised 0.34 and delivered 0.15 on those chambers. So the
	# side is chosen by which one can actually pay, and dark only wins where it can.
	var room_dark := sand_v - AIR_CONTRAST      # the highest air value the dark side allows
	var room_lit := sand_v + AIR_CONTRAST       # ...and the lowest the lit side allows
	var can_dark := room_dark >= 0.03
	var can_lit := room_lit <= 0.98
	var lit := (rng.randf() < 0.18) if (can_dark and can_lit) else can_lit
	_air_sat = rng.randf_range(0.04, 0.62) if rng.randf() < 0.75 else rng.randf_range(0.0, 0.08)
	_air_hue = sch.opposed(_hue_base)
	if lit:
		_air_val = rng.randf_range(clampf(room_lit, 0.55, 0.97), 0.99)
	else:
		# A saturated background carries more apparent weight than its value alone, so the
		# deeper the colour the further down it sits - taken off the CEILING of the range
		# rather than off the result, which is what keeps the margin a guarantee.
		_air_val = rng.randf_range(0.02, maxf(0.03, minf(room_dark, 0.55) - 0.10 * _air_sat))
	if rng.randf() < 0.5:
		add_layer("dust", rng, {"count": rng.randi_range(40, 90), "shaft": false,
			"hue": sch.accent, "alpha": 0.16, "speed": 0.015})

	var acc := PackedFloat32Array()
	acc.resize(_spouts.size())
	_job = SandJob.new()
	_job.g = sim
	_job.dt = _clock.dt
	_job.spouts = _spouts
	_job.acc = acc
	_job.run_cap = _run_cap
	_job.depth = _depth
	_job.wind_dir = _wind_dir

	return {"chamber": shape, "mood": sch.name, "grid_w": _w, "grid_h": _h,
		"materials": mats, "pours": pours, "spouts": _spouts.size(),
		"hatches": nh, "hatch_dwell": _hatch_dwell, "tilt": _tilt_mag,
		"wind": _wind, "wind_dir": _wind_dir, "ignite": _ignite,
		"rate": _clock.rate, "wall": thick, "run_cap": _run_cap,
		"air": "lit" if lit else "dim", "air_val": _air_val,
		"air_sat": _air_sat, "sand_val": sand_v}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	_ch = chroma_hue()

	# The hatch: a beat on an ARMED latch, so it opens on the music rather than on a timer
	# but cannot chatter on a dense passage.
	_hatch_arm += delta
	var beat_up := f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	if _hatch_left > 0.0:
		_hatch_left -= delta
		if _hatch_left <= 0.0:
			_drain_x0 = 0
			_drain_x1 = -1
	elif beat_up and _hatch_arm >= _hatch_gap and not _hatches.is_empty():
		# The WHICH is an index step, never an rng draw: drawing on an audio-conditioned
		# event would desynchronise an export from its preview for the rest of the song.
		var hh: Dictionary = _hatches[_hatch_i % _hatches.size()]
		_hatch_i += 1
		_hatch_arm = 0.0
		_hatch_left = _hatch_dwell
		_drain_x0 = int(hh["x0"])
		_drain_x1 = int(hh["x1"])

	# Gravity shear on a section change, held for a while and then eased back.
	if _tilt_left > 0.0:
		_tilt_left -= delta
		if _tilt_left <= 0.0:
			_tilt_target = 0.0
	elif f.movement >= _tilt_gate and not _tilt_signs.is_empty():
		_tilt_target = float(_tilt_signs[_tilt_i % _tilt_signs.size()]) * _tilt_mag
		_tilt_i += 1
		_tilt_left = _tilt_dwell
	_tilt = lerpf(_tilt, _tilt_target, 1.0 - exp(-_tilt_rate * delta))

	# Emissive materials brighten with the mix - fast attack, slow release, VALUE only.
	_glow = Nonlinear.flare(_glow, clampf(f.energy * 1.7 + f.beat * 0.35, 0.0, 1.0),
		delta, 7.0, 1.1)

	# Each spout's rate is its own band's level, so the pours move AGAINST each other - the
	# bass spout runs while the treble one is dry. The 0.12 floor is measured, not taste:
	# a band sits near zero for long stretches and a chamber that stops pouring is dead.
	var rates := PackedFloat32Array()
	rates.resize(_spouts.size())
	for i in _spouts.size():
		var sp: Dictionary = _spouts[i]
		var b: float = sp["band"]
		var gain: float = sp["gain"]
		rates[i] = gain * (0.12 + 0.95 * f.sample(b))

	# Sized to overscan BOTH axes, whichever binds: the chamber must run off the sides and
	# off the top whatever the aspect ratio, or the camera's drift exposes a corner of a
	# room that is supposed to continue past the frame.
	_cell = maxf(size.x * OVER / float(_w), size.y * OVER / float(_h))
	# The floor sits one cell inside the bottom edge so the drain is visible; the ceiling
	# and the side walls run off the frame, which is what makes it read as an interior.
	_org = Vector2(-float(_w) * _cell * 0.5, size.y * 0.5 - _cell - float(_h) * _cell)
	var drag := fposmod(_ch.x - _hue_base + 0.5, 1.0) - 0.5

	# Everything the audio decides travels in a FRESH snapshot dictionary each frame; only
	# the grid and the tick debt live on the job. Mutating a field on the job while the
	# worker is mid-build would be writing to memory another thread is reading, and for a
	# refcounted value (a packed array) that is not merely stale, it is unsafe.
	_job.pending = mini(_job.pending + _clock.ticks(delta), MAX_PENDING)
	_forge.kick(_job.run, {
		"rate": rates,
		"tilt": _tilt,
		# The draft breathes on its own slow clock rather than on the audio - it is weather
		# in the chamber, not a reaction to the track.
		"wind": _wind * (0.7 + 0.45 * mod.unit("draft")),
		"ignite": clampf(_ignite * (0.35 + 8.0 * f.flux), 0.0, 1.0),
		"glow": _glow,
		"hue_rot": fposmod(_hue_base + drag * 0.25 * _ch.y, 1.0),
		"drain_x0": _drain_x0,
		"drain_x1": _drain_x1,
		"org": _org,
		"cell": _cell,
	}, self, _job)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	# Flat and full-bleed, never a vignette: the air in this chamber is a back wall, and the
	# moment it has a gradient the pile reads as lit by a spotlight instead of standing in a
	# room. Layer's `bed` cannot do this (and its brightest tone is a mid one).
	paint_ground(_air_hue, _air_sat, _air_val, _ch.y * 0.5, _ch.x)
	draw_layers("back")
	_forge.submit(self)          # the whole chamber: simulated AND batched off-thread
	draw_layers("front")


func _rand(v, rng: RandomNumberGenerator) -> float:
	var a: Array = v
	return rng.randf_range(float(a[0]), float(a[1]))


func _rint(v, rng: RandomNumberGenerator) -> int:
	var a: Array = v
	return rng.randi_range(int(a[0]), int(a[1]))


## The off-thread job (the [FrameForge] job-object form): it OWNS the grid, runs the
## simulation, and batches the result, all on the worker. The scene never reads a cell -
## which is exactly what makes it safe, since a Director cut can free the scene mid-build.
##
## Pending ticks accumulate here rather than in the forge's snapshot because kick() is
## idempotent and keeps only the NEWEST snapshot: the Director's several update() calls per
## frame would otherwise have their ticks silently discarded, and the sim would run at
## whatever rate the display happened to be, which is the one thing [SimClock] exists to
## prevent.
class SandJob:
	extends RefCounted

	var g: Grains
	var spouts: Array = []       # immutable after build - safe to read from the worker
	var acc := PackedFloat32Array()
	var pending := 0
	var dt := 1.0 / 25.0
	var t_sim := 0.0
	var wind_dir := 1
	var depth := 0.28
	var run_cap := 8

	func run(s: Dictionary) -> Array:
		var n := pending
		pending -= n          # consume; a concurrent increment losing a race costs one tick
		var rate: PackedFloat32Array = s["rate"]
		g.tilt = float(s["tilt"])
		g.wind = float(s["wind"])
		g.wind_dir = wind_dir
		g.ignite = float(s["ignite"])
		g.set_drain(int(s["drain_x0"]), int(s["drain_x1"]))
		for k in n:
			t_sim += dt
			for si in mini(spouts.size(), rate.size()):
				var sp: Dictionary = spouts[si]
				# The duty cycle runs on the SIM clock, so a spout's on/off phase is a
				# property of the world rather than of the frame rate.
				var period: float = sp["period"]
				var duty: float = sp["duty"]
				var phase: float = sp["phase"]
				if fposmod(t_sim / maxf(0.5, period) + phase, 1.0) >= duty:
					continue
				acc[si] += rate[si] * dt
				var cnt := int(acc[si])
				if cnt <= 0:
					continue
				acc[si] -= float(cnt)
				g.emit(int(sp["x0"]), int(sp["x1"]), int(sp["y"]), int(sp["mat"]),
					mini(cnt, 32), si)
			g.step()
		var tb := TriBatch.new()
		var org: Vector2 = s["org"]
		g.paint(tb, org, float(s["cell"]), run_cap, float(s["hue_rot"]),
			float(s["glow"]), depth)
		return [{"pts": tb.pts, "cols": tb.cols, "idx": tb.idx}]


## Sloped shelves, alternating sides, for the material to run along and pour off.
##
## A FLAT shelf is a shelf: sand lands on it, heaps at its angle of repose, and eventually
## spills off both ends at once. That is what the old ledges did and it reads as a pile
## with a step in it. A SLOPED one is a chute - the grains keep taking the downhill
## sideways step the automaton already offers them, so the material travels the length of
## the shelf and leaves it at one known edge, in a stream. That stream falling to the next
## shelf down is the waterfall this scene was missing.
##
## The slope is one cell of drop per `run` cells of length, built as a staircase because
## the grid has no other way to express a slope - and a staircase is right anyway, since
## each step is a tiny lip that keeps the stream lively instead of laminar.
func _cascade(sim, stone: int, thick: int, count: int, ch: Dictionary,
		rng: RandomNumberGenerator) -> void:
	if count <= 0:
		return
	for k in count:
		# Spread down the chamber, avoiding the very top (nothing has fallen yet) and the
		# floor (the bottom pile wants room to build).
		var ly := int(float(_h) * (0.24 + 0.54 * (float(k) + 0.5) / float(count)))
		var ln := maxi(6, int(float(_w) * _rand(ch["shelf"], rng)))
		var left := (k % 2) == 0
		# A shallow slope only: steep enough that the sand runs, shallow enough that it
		# still piles a little on the way, which is what makes the stream pulse rather
		# than pour evenly.
		var run := rng.randi_range(4, 9)
		var thickness := rng.randi_range(1, 2)
		for i in ln:
			var x := (thick + i) if left else (_w - 1 - thick - i)
			if x < thick or x > _w - 1 - thick:
				continue
			var y := ly + int(i / run)
			if y >= _h - thick - 2:
				break
			sim.fill_rect(x, y, x, y + thickness, stone)
