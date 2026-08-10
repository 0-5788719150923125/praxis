extends Scene3D

## Canopy - trees growing on real terrain, from taproot to leaf, through one season.
##
## A wooded slope under a low raking sun. Thirty to a hundred and ten trees stand on a real
## [Terrain] heightfield - not a decal, not a sprite: each one is a genuine 3D branching
## structure grown by [Branch3D], a tapered trunk lofted as a closed 6-sided tube with limbs
## forking away at species-specific angles, a divergence roll between successive whorls, and
## twigs thinning into soft leaf masses. Trunks and crowns rasterize into a [ShadowField], so
## the hillside is striped with the wood's own shadows, and the whole thing merges into
## [method Terrain.collect_surface]'s quad list before the depth sort - which is what makes a
## hill occlude the trees standing behind it instead of the wood floating over the land.
##
## THE FOREST IS INSTANCED, and that is the only reason it fits. A handful of MODELS (five to
## nine) are grown once in [method build_params]; every tree is one of them, yawed, scaled and
## BENT at draw time. That is furry's trick - grow once in local space, re-lean at draw -
## lifted into three dimensions, and it turns a hundred L-systems into eight.
##
## WIND IS A TRAVELLING FRONT, not a global multiplier. A gust has a real position, a real
## crossing speed and a gaussian width, so it sweeps the hillside as a visible wave and the
## far trees answer LATE. That delay is physical - it falls out of the geometry rather than
## being scheduled - and it is the single detail that stops a field of trees reading as one
## object jiggling. The two frequencies of wind are separated the way real wind separates
## them: `f.bass` drives the long-period trunk sway as an ANGULAR cantilever displacement
## (never a scale - a tree that grows and shrinks on the beat is a joke), while `f.high` plus
## `f.treble` drive the fast small flutter of the leaf clusters alone.
##
## THE SEASON is the first thing the seed chooses, and everything is read off it - the mood on
## offer, the climate the terrain may take, leaf density, how hard the foliage turns from base
## to tip, whether the saturation is drained out, and what is in the air. Leaves come off over
## the scene, but shedding is NOT an rng draw: each leaf carries a fixed release threshold set
## at build and detaches when the running peak of an EMA of `f.beat` crosses it. That matters
## because the live analyzer and the offline export bake do not produce identical feature
## streams - anything that rolled dice on an audio event would render a different wood in the
## video than in the preview. Being a pure function of (threshold, shed level) it also costs
## no per-leaf state at all: sixty trees share eight models and the bare ones are computed,
## not stored.
##
## THE BUDGET IS THE DESIGN. Sixty trees at ~200 segments lofted six-sided is ninety thousand
## quads, which is not a frame. So LOD is structural: the nearest dozen get full tubes on the
## heavy wood and screen-space ribbons on the twigs, the middle band drops the finest order and
## thins its foliage, and everything beyond collapses to three to six crown puffs over a single
## trunk ribbon - a silhouette on the ridge. The whole frame is built off the main thread
## through [FrameForge], because under `--export` that builder runs synchronously and the LOD
## budget IS the render wall time.
##
## What the seed decides: the season and its mood; the landform and a climate that suits both;
## the sun's elevation (8-35 degrees, kept low for long shadows) and its drift direction; one
## species rule table (fork angle, divergence roll, length/width ratios, depth, children,
## crown fraction, buttress flare, trunk lean, leaf size); how many models and how many trees;
## the distribution law the trees scatter under; the slope they refuse to root on; the gust's
## speed, width and strength; and the LOD distances.

## SEASONS - the top-level choice, made before anything else, adapted from rooted_growth's
## table for a canopy rather than a root system.
##
## The point of the table is that colour, density and form cannot disagree about what time of
## year it is, because they are all read off one entry. An autumn wood is not a summer wood
## with an orange filter: it is thinner in the leaf, harder-turning from base to tip, and
## actively shedding, and those belong together.
##   `climates` the [Terrain] climates this season may take - a verdant winter is a lie.
##   `leaf`     foliage density, 0..1. Winter is nearly bare wood.
##   `bare`     the fraction of leaves that carry a reachable release threshold at all.
##   `turn`     how far foliage hue walks from base to tip. Autumn turns hardest.
##   `sat`/`val` scale the mood's own character rather than replacing it.
##   `ground`   what drifts through the frame, from the shared [Layer] registry.
##   `veil`     the chance of a soft coverage haze over the wood.
const SEASONS := {
	"spring": {
		"moods": ["verdant", "toxic", "teal", "rose", "dawn", "bone", "glacier"],
		"climates": ["verdant", "temperate"],
		"leaf": 0.85, "bare": 0.04, "turn": [0.04, 0.12], "sat": 0.88, "val": 1.12,
		"ground": [{"kind": "petals", "n": [14, 28]}, {"kind": "dust", "n": [40, 80]}],
		"veil": 0.25,
	},
	"summer": {
		"moods": ["verdant", "toxic", "teal", "brass", "sodium", "abyss", "ember"],
		"climates": ["verdant", "temperate", "arid"],
		"leaf": 1.00, "bare": 0.08, "turn": [0.05, 0.16], "sat": 1.15, "val": 1.00,
		"ground": [{"kind": "dust", "n": [50, 95]}, {"kind": "fog", "n": [4, 8]}],
		"veil": 0.12,
	},
	"autumn": {
		"moods": ["ember", "dawn", "sodium", "brass", "rose", "ash", "verdant"],
		"climates": ["temperate", "arid", "tundra"],
		"leaf": 0.72, "bare": 0.62, "turn": [0.20, 0.42], "sat": 1.10, "val": 0.94,
		"ground": [{"kind": "petals", "n": [12, 24]}, {"kind": "fog", "n": [4, 9]}],
		"veil": 0.35,
	},
	"winter": {
		"moods": ["ash", "bone", "glacier", "abyss", "violet", "teal", "ember"],
		"climates": ["tundra", "temperate"],
		"leaf": 0.16, "bare": 0.92, "turn": [0.02, 0.07], "sat": 0.42, "val": 1.06,
		"ground": [{"kind": "snow", "n": [70, 130]}, {"kind": "fog", "n": [5, 9]}],
		"veil": 0.60,
	},
}

## Which climates suit which landform - a mesa is not verdant, an island is not tundra. The
## season's own list is intersected with this, so the wood, the ground and the weather agree.
const TERRAINS := {
	"hills":   ["temperate", "verdant", "tundra", "arid"],
	"valleys": ["verdant", "temperate", "tundra"],
	"mesa":    ["arid", "temperate"],
	"islands": ["verdant", "temperate"],
}

## How the wood is distributed over the land. Scatter is by REJECTION against the terrain
## itself - slope first (nothing roots on a cliff or under water), then the law's weight.
const LAWS := ["even", "ridge", "valley", "glade"]

var _terrain: Terrain
var _models: Array = []              # the grown tree models (immutable after build)
var _trees: Array = []               # instances: model index, place, yaw, scale, phase
var _forge := FrameForge.new()
var _sim := SimClock.new(60.0)       # the wind / season / shed simulation clock
var _light: Lighting
var _sch: Scheme
var _species: Dictionary = {}
var _season := "summer"
var _law := "even"
var _ground_layer := "dust"

# Wind. `_front` is the gust's position along `_wind`, in world units; it crosses and wraps.
var _wind := Vector3(1, 0, 0)
var _front := 0.0
var _span := 5.4
var _gust := 0.0
var _gust_speed := 1.6
var _gust_width := 0.9
var _gust_amp := 0.18
var _breeze := 0.0
var _flutter := 0.0

# The season's own clock, and the shedding level (a running peak of an EMA of the beat).
var _season_t := 0.0
var _beat_ema := 0.0
var _shed := 0.0

var _glow := 0.0
var _chroma := Vector2.ZERO
var _yaw := 0.0
var _yaw_dir := 1.0
var _dist := 8.0
var _pitch := 0.45
var _look_y := 0.35
var _light_az := 0.0
var _light_el := 0.35
var _light_drift := 1.0
var _bark_hue := 0.08
var _bark_sat := 0.34
var _bark_val := 0.62
var _leaf_hue := 0.32
var _leaf_sat := 0.60
var _leaf_val := 0.82
var _leaf_turn := 0.10
var _leaf_density := 1.0
var _fog_col := Color(0.62, 0.66, 0.72)
var _lod_near := 3.2
var _lod_far := 5.4
var _near_max := 12
var _mid_max := 34
var _embed := 0.03


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	# The season first: every colour, density and climate decision below is read through it.
	_season = String(SEASONS.keys()[rng.randi() % SEASONS.size()])
	var sea: Dictionary = SEASONS[_season]
	_sch = Scheme.among(sea["moods"], rng)
	var sat_mul: float = float(sea["sat"])
	var val_mul: float = float(sea["val"])
	_leaf_density = float(sea["leaf"])
	var bare: float = float(sea["bare"])

	# Landform crossed with a climate BOTH the land and the season accept.
	var ttype := String(TERRAINS.keys()[rng.randi() % TERRAINS.size()])
	var land_cl: Array = TERRAINS[ttype]
	var sea_cl: Array = sea["climates"]
	var ok: Array = []
	for c in land_cl:
		if sea_cl.has(c):
			ok.append(c)
	if ok.is_empty():
		ok = land_cl
	var climate := String(ok[rng.randi() % ok.size()])
	_terrain = Terrain.new()
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.42, 0.72), null, climate)
	# Sink each base a little BELOW the surface so the trunk rises out of the ground with a
	# ragged root line (the merged terrain hides the buried stub) instead of meeting it on a
	# clean seam. Scaled by relief, because a flatter map needs less burial to stay hidden.
	_embed = _terrain.relief * rng.randf_range(0.03, 0.10)

	# A LOW sun: 8 to 35 degrees. Long shadows are the whole reason to stripe a hillside with
	# trunks, and at these elevations the terrain's own shadow march still resolves (it gives
	# up below 0.02 in y, i.e. about 1.1 degrees, so there is real margin).
	_light_el = deg_to_rad(rng.randf_range(8.0, 35.0))
	_light_az = rng.randf() * TAU
	_light_drift = 1.0 if rng.randf() < 0.5 else -1.0
	_terrain.set_light(_light_az, _light_el)
	var sun := Vector3(cos(_light_el) * cos(_light_az), sin(_light_el),
		cos(_light_el) * sin(_light_az))

	# ONE species for the whole wood - a stand is a stand - and a handful of models grown from
	# it, so the trees are a family rather than clones or a hundred separate L-systems. The
	# phototropic bias is baked into the model and then rotated with each instance's yaw, so a
	# yawed copy leans a little off-sun; at 8-35 degrees of elevation that is invisible, and
	# the alternative is one model per tree, which is the cost this whole design exists to avoid.
	_species = Branch3D.sample_species(rng)
	var nmodels := rng.randi_range(5, 9)
	var seg_budget := rng.randi_range(140, 230)
	for _mi in nmodels:
		var h := rng.randf_range(0.30, 0.56)
		var model: Dictionary = Branch3D.grow(_species, rng, h,
			h * rng.randf_range(0.020, 0.042), sun, seg_budget)
		# Release thresholds: which leaves are ever sheddable, and in what order. Sampled HERE,
		# at build, off the seeded rng - never on an audio event.
		var lv: Array = model["leaves"]
		for entry in lv:
			var lf: Dictionary = entry
			lf["release"] = rng.randf_range(0.03, 0.45) if rng.randf() < bare else 99.0
		_models.append(model)

	# THE SCATTER. Rejection against the real surface: never in the water, never on ground too
	# steep to root, then weighted by the distribution law, then spaced by a Poisson-disc test
	# through a hash grid (the naive all-pairs version is 480k distance checks at this count,
	# and build_params runs synchronously on the main thread at every cut).
	_law = String(LAWS[rng.randi() % LAWS.size()])
	var want := rng.randi_range(30, 110)
	var margin: float = _terrain.half * 0.94
	# SLOPE REJECTION, as a QUANTILE of this landform's own steepness rather than an absolute
	# normal. An absolute threshold is untunable across the recipes: the same number that
	# rejects half a mesa rejects nothing at all on rolling hills, because a heightfield's
	# gradient scales with its relief and its frequency content. Sampling the surface first and
	# refusing the steepest fraction of it means "trees do not root on cliffs" stays true of
	# every landform, and the fraction itself is the sampled tunable.
	var slopes := PackedFloat32Array()
	slopes.resize(256)
	for i in 256:
		slopes[i] = _terrain.normal_world(rng.randf_range(-margin, margin),
			rng.randf_range(-margin, margin)).y
	slopes.sort()
	var reject := rng.randf_range(0.08, 0.40)
	var slope_min: float = slopes[clampi(int(reject * 256.0), 0, 255)]
	var spacing: float = _terrain.half * 2.0 / sqrt(float(want)) * rng.randf_range(0.45, 0.78)
	var glade_c := Vector2(rng.randf_range(-1.4, 1.4), rng.randf_range(-1.4, 1.4))
	var glade_r: float = _terrain.half * rng.randf_range(0.22, 0.42)
	var hmax := 0.001
	for hv in _terrain.hgrid:
		hmax = maxf(hmax, hv)
	hmax = maxf(0.05, hmax - _terrain.water)
	var cells: Dictionary = {}
	var inv_cell := 1.0 / maxf(0.001, spacing)
	var tries := 0
	while _trees.size() < want and tries < want * 20:
		tries += 1
		var wx := rng.randf_range(-margin, margin)
		var wz := rng.randf_range(-margin, margin)
		var hh: float = _terrain.height_at(wx, wz)
		if hh <= 0.015:
			continue                                   # in the water, or on the shoreline
		var nrm: Vector3 = _terrain.normal_world(wx, wz)
		if nrm.y < slope_min:
			continue                                   # too steep to hold a root plate
		var hn := clampf(hh / hmax, 0.0, 1.0)
		var weight := 1.0
		match _law:
			"ridge":
				weight = pow(hn, 1.7)
			"valley":
				weight = pow(1.0 - hn, 1.7)
			"glade":
				weight = 0.03 if Vector2(wx, wz).distance_to(glade_c) < glade_r else 1.0
		if rng.randf() > weight:
			continue
		var cx := int(floor(wx * inv_cell))
		var cz := int(floor(wz * inv_cell))
		var clash := false
		for dz in [-1, 0, 1]:
			for dx in [-1, 0, 1]:
				var key := Vector2i(cx + dx, cz + dz)
				if not cells.has(key):
					continue
				var occ: Array = cells[key]
				for o in occ:
					if (o as Vector2).distance_to(Vector2(wx, wz)) < spacing:
						clash = true
						break
				if clash:
					break
			if clash:
				break
		if clash:
			continue
		var key2 := Vector2i(cx, cz)
		if not cells.has(key2):
			cells[key2] = []
		(cells[key2] as Array).append(Vector2(wx, wz))
		var mi := rng.randi() % _models.size()
		var model: Dictionary = _models[mi]
		var sc := rng.randf_range(0.72, 1.30)
		var yaw := rng.randf() * TAU
		var gy: float = hh * _terrain.relief - _embed
		_trees.append({
			"pos": Vector3(wx, gy, wz), "model": mi, "scale": sc,
			"cy": cos(yaw), "sy": sin(yaw),
			"h": float(model["height"]) * sc,
			"trunk": float(model["radius"]) * sc * float(_species["flare"]),
			"crown_r": float(model["crown_r"]) * sc,
			"phase": rng.randf() * TAU,
			"rate": rng.randf_range(2.6, 4.6),
			"delay": rng.randf_range(0.0, 0.55),
			"shed_bias": rng.randf_range(0.7, 1.35),
			"hue": rng.randf_range(-0.04, 0.04),
			# Pinned to a spectral position by ELEVATION, so the ridge line rings with the top
			# of the spectrum and the valley floor with the bottom.
			"t": hn})

	# THE GUST. A front with a real crossing speed and width; `_span` is a little wider than
	# the land so it enters and leaves rather than blinking on in the middle of the wood.
	var wa := rng.randf() * TAU
	_wind = Vector3(cos(wa), 0.0, sin(wa))
	_gust_speed = rng.randf_range(0.9, 2.6)
	_gust_width = rng.randf_range(0.5, 1.4)
	_gust_amp = rng.randf_range(0.10, 0.26)
	_span = _terrain.half * 1.8
	_front = -_span

	# Colour. Bark takes the mood's base darkened and drained; foliage takes the accent, and
	# the season scales both rather than replacing them.
	_bark_hue = fposmod(_sch.hue + rng.randf_range(-0.02, 0.02), 1.0)
	_bark_sat = clampf(_sch.sat * sat_mul * rng.randf_range(0.32, 0.55), 0.05, 0.8)
	_bark_val = clampf(_sch.val * val_mul * rng.randf_range(0.42, 0.62), 0.08, 1.0)
	_leaf_hue = fposmod(_sch.accent + rng.randf_range(-0.03, 0.03), 1.0)
	_leaf_sat = clampf(_sch.sat * sat_mul * rng.randf_range(0.7, 1.05), 0.05, 0.95)
	_leaf_val = clampf(_sch.val * val_mul * rng.randf_range(0.78, 1.05), 0.1, 1.2)
	var to_accent := fposmod(_sch.opposed(_leaf_hue, 0.18) - _leaf_hue + 0.5, 1.0) - 0.5
	_leaf_turn = to_accent * rng.randf_range(float(sea["turn"][0]), float(sea["turn"][1])) * 1.6
	_fog_col = Color.from_hsv(fposmod(_sch.hue + rng.randf_range(0.38, 0.58), 1.0),
		rng.randf_range(0.04, 0.16), rng.randf_range(0.55, 0.86))

	# Camera. Far enough back to read a whole hillside, pitched down enough that the far rim of
	# the heightfield never becomes the horizon (there is no sky behind it - only the frame).
	lens.fov = rng.randf_range(44.0, 56.0)
	_dist = rng.randf_range(6.4, 9.2)
	_pitch = rng.randf_range(0.42, 0.66)
	_look_y = rng.randf_range(0.22, 0.5)
	_yaw = rng.randf() * TAU
	_yaw_dir = 1.0 if rng.randf() < 0.5 else -1.0

	# LOD. Sampled, because how far "near" reaches decides both the look and the frame cost.
	# These are measured from the NEAREST tree rather than from the camera: the orbit distance
	# and the landform between them decide where the wood actually starts, and an absolute
	# threshold would either articulate nothing (camera pulled back) or everything (pushed in).
	# So this is the depth INTO the wood that detail reaches, which is what a LOD distance
	# means when the subject is a field rather than an object.
	_lod_near = rng.randf_range(2.5, 4.0)
	_lod_far = _lod_near + rng.randf_range(1.5, 2.6)
	_near_max = rng.randi_range(9, 14)
	_mid_max = _near_max + rng.randi_range(16, 26)

	_light = Lighting.new(rng, 3)

	# A SKY behind the land, and it is not decoration. A heightfield has no side skirt and no
	# horizon: past its far rim there is simply nothing, and at these low pitches the rim sits
	# below the top of the frame, so a bare canopy scene would put a band of void above its own
	# ridge. `bed` (as a BACK layer - it defaults to front) fills that band with the same pale
	# distance the fog already fades the far trees toward, so the ridge line reads as a skyline.
	add_layer("bed", rng, {"z": "back", "hue": _fog_col.h, "sat": _fog_col.s + 0.06,
		"val": rng.randf_range(0.22, 0.42), "pools": rng.randi_range(2, 4)})

	# The season's air, and an optional coverage veil / low sun-shafts.
	var ground: Array = sea["ground"]
	var pick: Dictionary = ground[rng.randi() % ground.size()]
	_ground_layer = String(pick["kind"])
	add_layer(_ground_layer, rng,
		_air_cfg(_ground_layer, rng.randi_range(int(pick["n"][0]), int(pick["n"][1]))))
	if rng.randf() < float(sea["veil"]):
		add_layer("veil", rng, {"hue": _fog_col.h, "sat": 0.05, "val": _fog_col.v,
			"max": rng.randf_range(0.18, 0.38)})
	if _light_el < deg_to_rad(18.0) and rng.randf() < 0.55:
		add_layer("rays", rng, {"count": rng.randi_range(3, 6),
			"hue": fposmod(_sch.accent + 0.02, 1.0), "z": "front"})
	return {"season": _season, "mood": _sch.name, "terrain": ttype, "climate": climate,
		"law": _law, "trees": _trees.size(), "models": _models.size(),
		"depth": int(_species["depth"]), "children": int(_species["children"]),
		"angle": float(_species["angle"]), "roll": float(_species["roll"]),
		"ratio": float(_species["ratio"]), "taper": float(_species["taper"]),
		"crown": float(_species["crown"]), "leaf_density": _leaf_density,
		"sun_deg": rad_to_deg(_light_el), "gust_speed": _gust_speed,
		"gust_width": _gust_width, "lod_near": _lod_near, "slope_reject": reject,
		"air": _ground_layer}


# Tint the season's airborne layer to the wood it falls from: blossom and dry leaves take the
# foliage hue, snow and fog the pale sky colour the distance already fades to.
func _air_cfg(kind: String, n: int) -> Dictionary:
	match kind:
		"petals":
			return {"count": n, "hue": _leaf_hue, "sat": clampf(_leaf_sat, 0.05, 0.9)}
		"snow":
			return {"count": n, "hue": _fog_col.h, "sat": 0.06, "size": 0.005}
		"fog":
			return {"count": n, "hue": _fog_col.h, "sat": 0.10, "z": "front"}
		_:
			return {"count": n, "hue": _fog_col.h, "shaft": false}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.012, 0.02)
	_light.update(f, delta)
	update_layers(f, delta)
	_glow = Nonlinear.flare(_glow, clampf(0.3 * f.energy + 0.7 * f.beat, 0.0, 1.0),
		delta, 9.0, 1.6)
	# The wind, the season and the shedding are a SIMULATION - they integrate state - so they
	# advance on wall-clock ticks at a fixed rate, never once per update() call. The Director
	# sub-steps up to fifteen times in a frame, pre-warms every scene twelve times before its
	# first, and an Echo re-localize can fast-forward hundreds of calls: a naive gust would
	# have crossed the hillside four times before anyone saw it.
	for _i in _sim.ticks(delta):
		_step(f, _sim.dt)
	_chroma = chroma_hue()
	_yaw += delta * (0.028 + 0.075 * f.energy) * _yaw_dir
	lens.orbit(Vector3(0.0, _look_y, 0.0), _dist, _yaw, _pitch + 0.02 * sin(_life * 0.09))
	_light_az += delta * 0.02 * _light_drift
	_terrain.set_light(_light_az, _light_el)
	_terrain.step_light(delta)

	# The 12-step pre-warm runs BEFORE the scene enters the tree, so `size` is still zero and
	# unit() with it - a packet built now would be the whole wood collapsed onto a point, and
	# being the newest finished packet it is exactly what the first drawn frame would submit.
	# Everything above (the sim, the light, the camera ease) is what pre-warm is for; the build
	# is not, so skip only that.
	if size.x < 1.0 or size.y < 1.0:
		return

	# Snapshot the frame into a job. Models and tree records are immutable after build so they
	# ride by reference; the terrain does the same (its per-frame light writes are in-place
	# float updates, at worst a frame stale); the lens is copied because main re-orbits it.
	var job := CanopyJob.new()
	job.f = f
	job.terrain = _terrain
	job.tex_rid = Terrain.detail_texture().get_rid()
	job.models = _models
	job.trees = _trees
	job.u = unit()
	job.clock = _sim.elapsed()
	job.life = _life
	job.reveal = smoothstep(0.8, 1.0, view.presence)
	job.glow = _glow
	# [Lighting]'s glow drives the bark RIM and nothing else - the one place a beat is allowed
	# to touch the wood. It moves light across a trunk; it does not move the trunk.
	job.rim = _light.glow()
	job.lit = clampf(0.62 + 0.4 * _glow + 0.3 * f.energy, 0.4, 1.4)
	job.front = _front
	job.gust = _gust
	job.breeze = _breeze
	job.flutter = _flutter
	job.season_t = _season_t
	job.shed = _shed
	job.wind = _wind
	job.gust_width = _gust_width
	job.gust_amp = _gust_amp
	job.bark_hue = _bark_hue
	job.bark_sat = _bark_sat
	job.bark_val = _bark_val
	# Foliage is pulled toward the music's own tonality - the hue of its key - by a quarter of
	# how tonal the moment actually is. Sound drives colour.
	job.leaf_hue = _blend_hue(_leaf_hue, _chroma.x, 0.25 * _chroma.y)
	job.leaf_sat = _leaf_sat
	job.leaf_val = _leaf_val
	job.leaf_turn = _leaf_turn
	job.leaf_density = _leaf_density
	job.fog_col = _fog_col
	job.fog_near = _dist * 0.5
	job.fog_far = _dist * 2.2
	job.lod_near = _lod_near
	job.lod_far = _lod_far
	job.near_max = _near_max
	job.mid_max = _mid_max
	job.lens = Lens3D.new()
	job.lens.eye = lens.eye
	job.lens.look = lens.look
	job.lens.up = lens.up
	job.lens.fov = lens.fov
	job.lens.near = lens.near
	_forge.kick(job.run, {}, self, job)
	queue_redraw()


# One fixed simulation tick: the gust crosses, the season advances, the wood sheds.
func _step(f: AudioFeatures, dt: float) -> void:
	_front = wrapf(_front + _gust_speed * dt, -_span, _span)
	# Movement (a section change) is what raises a real gust; the beat only ruffles it. Both
	# run well under 1.0 in practice, so the gains are written for the range they actually
	# occupy and a floor keeps a steady passage breezy rather than dead still.
	_gust = Nonlinear.flare(_gust,
		clampf(0.16 + 1.7 * f.movement + 0.45 * f.beat, 0.0, 1.0), dt, 8.0, 1.5)
	_breeze = Nonlinear.flare(_breeze, clampf(0.95 * f.bass, 0.0, 1.0), dt, 3.0, 1.0)
	_flutter = Nonlinear.flare(_flutter,
		clampf(0.6 * f.high + 0.7 * f.treble, 0.0, 1.0), dt, 12.0, 4.0)
	# The season's own front: beats lunge it forward through a spike curve (the rooted_growth
	# idiom), quiet barely moves it. It runs past 1.0 so the last trees still finish.
	_season_t = minf(1.7, _season_t + dt * 0.055
		* (0.25 + 1.6 * Nonlinear.apply("spike", f.energy, 2.2)))
	_beat_ema = lerpf(_beat_ema, f.beat, 1.0 - exp(-0.7 * dt))
	_shed = maxf(_shed, _beat_ema)      # a running PEAK: a wood does not re-leaf mid-autumn


# Walk `h` toward `toward` by `amount`, the short way round the wheel.
func _blend_hue(h: float, toward: float, amount: float) -> float:
	var d := toward - h
	return fposmod(h + (d - round(d)) * clampf(amount, 0.0, 1.0), 1.0)


func _draw() -> void:
	begin_draw()
	draw_layers("back")                  # the sky, behind the land
	texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED
	_forge.submit(self)
	draw_layers("front")                 # what falls through the wood, in front of it


## The whole frame, built off the main thread (the [FrameForge] contract): the shadow pass,
## the terrain merge, every tree at its LOD, the painter sort, and the batch runs. Reads only
## its own members - never the scene node - so a mid-job Director cut is harmless.
class CanopyJob:
	extends RefCounted

	const SIDES := 6                 # facets around a lofted limb
	const LEAF_SIDES := 5            # rim points on a leaf-mass billboard
	const SHADOW_TREES := 64         # how many trunks rasterize into the shadow map

	var f: AudioFeatures
	var lens: Lens3D
	var terrain: Terrain
	var tex_rid := RID()
	var models: Array = []
	var trees: Array = []
	var u := 1.0
	var clock := 0.0
	var life := 0.0
	var reveal := 1.0
	var glow := 0.0
	var rim := 0.0
	var lit := 1.0
	var front := 0.0
	var gust := 0.0
	var breeze := 0.0
	var flutter := 0.0
	var season_t := 0.0
	var shed := 0.0
	var wind := Vector3(1, 0, 0)
	var gust_width := 1.0
	var gust_amp := 0.18
	var bark_hue := 0.08
	var bark_sat := 0.34
	var bark_val := 0.62
	var leaf_hue := 0.32
	var leaf_sat := 0.6
	var leaf_val := 0.82
	var leaf_turn := 0.1
	var leaf_density := 1.0
	var fog_col := Color(0.62, 0.66, 0.72)
	var fog_near := 4.0
	var fog_far := 16.0
	var lod_near := 3.2
	var lod_far := 5.4
	var near_max := 12
	var mid_max := 34

	var _cam_r := Vector3.RIGHT
	var _cam_u := Vector3.UP

	func run(_s: Dictionary) -> Array:
		lens.prepare()
		var axes := Branch3D.cam_axes(lens)
		_cam_r = axes[0]
		_cam_u = axes[1]
		var ldir: Vector3 = terrain.light_dir()
		# Rank the wood NEAR-FIRST through the native-key sort (a sort_custom lambda over a
		# hundred trees is a hundred interpreter calls a frame for nothing).
		var order: Array = []
		for i in trees.size():
			var tr: Dictionary = trees[i]
			var p: Vector3 = tr["pos"]
			order.append({"d": -(p - lens.eye).length(), "i": i})
		order = TriBatch.painter_sort(order)

		# Pass A: rasterize trunks (and, where there is foliage, crowns) into a light-space
		# shadow map, so the hillside is striped with the wood's own shadows and a tree
		# standing behind another is shaded by it.
		var shadow := ShadowField.new()
		shadow.build(ldir, Vector3(-terrain.half, -terrain.relief, -terrain.half),
			Vector3(terrain.half, terrain.relief + 1.4, terrain.half))
		if reveal >= 0.02:
			var sn := mini(order.size(), SHADOW_TREES)
			for oi in sn:
				var it: Dictionary = order[oi]
				var tr: Dictionary = trees[int(it["i"])]
				var g := _grown(tr)
				if g < 0.1:
					continue
				var p: Vector3 = tr["pos"]
				var th: float = float(tr["h"]) * g
				shadow.add_box(p, Vector3.UP, Vector3.RIGHT, Vector3.BACK,
					float(tr["trunk"]), th * 0.7)
				if leaf_density > 0.25:
					shadow.add_box(p + Vector3.UP * (th * 0.6), Vector3.UP, Vector3.RIGHT,
						Vector3.BACK, float(tr["crown_r"]) * g * leaf_density, th * 0.34)

		# ONE merged list: terrain quads (already shadowed by the wood) plus every tree face,
		# depth-sorted together, so the land occludes the buried trunk bases and the trees
		# occlude one another correctly.
		var faces: Array = terrain.collect_surface(lens, u, lit, life, shadow)
		if reveal >= 0.02:
			# The detail bands start at the near edge of the wood, not at the camera - see the
			# scene's note on _lod_near. Both a distance test AND a rank cap: the distance
			# decides what LOOKS right, the rank is the hard ceiling that decides what the
			# frame COSTS, and under --export the frame cost is the render wall time.
			var d0 := 0.0
			if not order.is_empty():
				var first: Dictionary = order[0]
				d0 = -float(first["d"])
			var near_d := d0 + lod_near
			var far_d := d0 + lod_far
			for oi in order.size():
				var it: Dictionary = order[oi]
				var dst := -float(it["d"])
				var mode := 2
				if dst < near_d and oi < near_max:
					mode = 0
				elif dst < far_d and oi < mid_max:
					mode = 1
				_emit_tree(faces, trees[int(it["i"])], mode, dst, ldir)
		faces = TriBatch.painter_sort(faces)
		var tb := TriBatch.new()
		for entry in faces:
			var fc: Dictionary = entry
			if fc.has("fan"):                                # a soft leaf mass / crown puff
				tb.mark_run(false, RID())
				var fp: PackedVector2Array = fc["fan"]
				var fcl: PackedColorArray = fc["fcols"]
				for k in range(1, fp.size() - 1):
					tb.tri_colored(fp[0], fp[k], fp[k + 1], fcl[0], fcl[k], fcl[k + 1])
			elif fc.has("uvs"):                              # terrain land (textured run)
				tb.mark_run(true, tex_rid)
				tb.quad_textured(fc["poly"], fc["cols"], fc["uvs"])
			else:                                            # wood
				tb.mark_run(false, RID())
				tb.quad_colored(fc["poly"], fc["cols"])
		return tb.take_chunks()

	# How far this tree has come this season. Staggered per tree, so the wood fills in as a
	# stand rather than as one synchronized organism.
	func _grown(tr: Dictionary) -> float:
		return clampf((season_t - float(tr["delay"])) / 0.45, 0.0, 1.0)

	# Place a model-local point into the world: yaw, scale, then the wind BEND - a cantilever
	# displacement growing as the square of height above the base, which is how a trunk
	# actually deflects. It moves the wood; it never resizes it.
	func _pose(p: Vector3, base: Vector3, s: float, cy: float, sy: float,
			bend: float, inv_h: float) -> Vector3:
		var x := p.x * cy - p.z * sy
		var z := p.x * sy + p.z * cy
		var hy := p.y * s
		var t := clampf(hy * inv_h, 0.0, 1.0)
		var k := bend * t * t
		return Vector3(base.x + x * s + wind.x * k, base.y + hy, base.z + z * s + wind.z * k)

	func _fog(c: Color, t: float) -> Color:
		if t <= 0.002:
			return c
		return Color(lerpf(c.r, fog_col.r, t), lerpf(c.g, fog_col.g, t),
			lerpf(c.b, fog_col.b, t), c.a)

	# One tree at LOD `mode`: 0 = articulated (tubes on the heavy wood, ribbons on the twigs,
	# full foliage), 1 = ribbons with the finest order dropped and thinned foliage, 2 = a
	# silhouette (trunk ribbon plus crown puffs).
	func _emit_tree(faces: Array, tree_entry, mode: int, dst: float, ldir: Vector3) -> void:
		var tr: Dictionary = tree_entry
		var g := _grown(tr)
		if g <= 0.02:
			return
		var model: Dictionary = models[int(tr["model"])]
		var base: Vector3 = tr["pos"]
		var s: float = float(tr["scale"])
		var cy: float = float(tr["cy"])
		var sy: float = float(tr["sy"])
		var th: float = maxf(0.001, float(tr["h"]))
		var inv_h := 1.0 / th
		# THE GUST, arriving here and not everywhere. `along` is this tree's coordinate on the
		# wind axis; the front is a gaussian window travelling along it, so a tree on the far
		# side answers late by exactly the crossing time - physics, not a schedule.
		var along := wind.x * base.x + wind.z * base.z
		var ph := (along - front) / maxf(0.05, gust_width)
		var local := gust * exp(-ph * ph)
		var swing := sin(clock * float(tr["rate"]) + float(tr["phase"]))
		var ripple := sin(clock * float(tr["rate"]) * 1.7 + float(tr["phase"]) * 1.3)
		var bend := th * (gust_amp * (0.16 + 0.30 * breeze) * swing
			+ gust_amp * 1.5 * local * (0.65 + 0.35 * ripple))
		var shade := 0.42 + 0.58 * terrain.shadow_at(base.x, base.z)
		var band := clampf(f.sample(float(tr["t"])), 0.0, 1.0)
		var fogt := clampf(smoothstep(fog_near, fog_far, dst), 0.0, 1.0)
		var hue_off: float = float(tr["hue"])
		var segs: Array = model["segs"]
		var lends: PackedInt32Array = model["level_end"]
		# How much of the structure this LOD walks AT ALL. Segments are stored in level order,
		# so one index is the whole cull - no per-segment depth test, and a silhouette touches
		# four dictionaries instead of two hundred.
		var limit := segs.size()
		if mode == 2:
			limit = int(lends[0]) if lends.size() > 0 else 0
		elif mode == 1 and lends.size() > 1:
			limit = int(lends[clampi(lends.size() - 3, 1, lends.size() - 1)])
		var barkv := clampf(bark_val * shade * (0.72 + 0.30 * band) * lit, 0.05, 1.4)
		for si in limit:
			var sg: Dictionary = segs[si]
			var b0: float = float(sg["born0"])
			if b0 > g:
				continue
			var sd: int = int(sg["depth"])
			# The segment straddling the growth front is SHORTENED, so the wood extends rather
			# than popping into place a limb at a time.
			var la: Vector3 = sg["a"]
			var lb: Vector3 = sg["b"]
			var frac := clampf((g - b0) / maxf(1e-4, float(sg["born"]) - b0), 0.0, 1.0)
			if frac < 0.999:
				lb = la.lerp(lb, frac)
			var wa := _pose(la, base, s, cy, sy, bend, inv_h)
			var wb := _pose(lb, base, s, cy, sy, bend, inv_h)
			var w0: float = float(sg["w0"]) * s
			var w1: float = float(sg["w1"]) * s
			# Finer wood is paler and less saturated - young bark against old.
			var fine := clampf(float(sd) / 4.0, 0.0, 1.0)
			var col := Color.from_hsv(fposmod(bark_hue + hue_off * 0.5, 1.0),
				clampf(bark_sat * (1.0 - 0.3 * fine), 0.02, 0.95),
				clampf(barkv * (1.0 + 0.22 * fine), 0.03, 1.4), reveal)
			col = _fog(col, fogt * 0.85)
			if mode == 0 and sd <= 1:
				# Only the heavy wood is worth a closed tube; a twig's tube is three
				# sub-pixel quads where a ribbon is one.
				Branch3D.loft_tube(faces, lens, u, wa, wb, w0, w1, SIDES, col, ldir,
					0.42, rim * 0.55)
			else:
				var dim := Color(col.r * 0.78, col.g * 0.78, col.b * 0.78, col.a)
				Branch3D.loft_ribbon(faces, lens, u, wa, wb, w0, w1, dim, col)
		if mode == 2:
			_emit_puffs(faces, tr, model, base, s, cy, sy, bend, inv_h, g, band, shade, fogt)
			return
		if leaf_density <= 0.02:
			return
		# FOLIAGE. Density thins the cluster count by striding the model's leaf sites, and the
		# middle LOD band thins it again - a tree two thirds of the way to the ridge does not
		# need forty billboards to read as leafy.
		var lv: Array = model["leaves"]
		var stride := maxi(1, int(round(1.0 / maxf(0.05, leaf_density))))
		if mode == 1:
			stride *= 3
		var shed_level := shed * float(tr["shed_bias"])
		var lsat := clampf(leaf_sat * (0.85 + 0.3 * band), 0.03, 0.98)
		for li in range(0, lv.size(), stride):
			var lf: Dictionary = lv[li]
			if float(lf["born"]) > g:
				continue
			# The leaf is on the tree until the season's shed level passes ITS threshold, then
			# it fades over a short band. No state, no dice: a pure function of the two.
			var rel: float = float(lf["release"])
			var alive := 1.0 - smoothstep(rel, rel + 0.07, shed_level)
			if alive <= 0.02:
				continue
			var lpos: Vector3 = lf["p"]
			var lp := _pose(lpos, base, s, cy, sy, bend, inv_h)
			# The fast half of the wind: the leaf masses flutter where the trunk only sways.
			var fw := flutter * float(lf["size"]) * s * 1.4
			var fa := clock * 9.0 + float(lf["phase"])
			lp += _cam_r * (sin(fa) * fw) + _cam_u * (cos(fa * 1.3) * fw * 0.6)
			var h := fposmod(leaf_hue + hue_off + float(lf["hue"])
				+ leaf_turn * float(lf["along"]), 1.0)
			var v := clampf(leaf_val * shade * (0.68 + 0.36 * band + 0.22 * glow) * lit,
				0.05, 1.35)
			var core := Color.from_hsv(h, lsat, v, reveal * alive * 0.92)
			Branch3D.billboard_fan(faces, lens, u, lp, _cam_r, _cam_u,
				float(lf["size"]) * s, _fog(core, fogt * 0.8), LEAF_SIDES,
				0.28, float(lf["ragged"]))

	# The silhouette: a distant tree is its crown, not its architecture. Three to six puffs,
	# sized off the model's real leaf mass so a push-in does not change the wood's shape.
	func _emit_puffs(faces: Array, tr: Dictionary, model: Dictionary, base: Vector3, s: float,
			cy: float, sy: float, bend: float, inv_h: float, g: float, band: float,
			shade: float, fogt: float) -> void:
		if leaf_density <= 0.05 or g < 0.5:
			return
		var open := smoothstep(0.45, 1.0, g) * clampf(leaf_density * 1.2, 0.0, 1.0)
		var alive := 1.0 - smoothstep(0.35, 0.85, shed * float(tr["shed_bias"]))
		var a := reveal * (0.35 + 0.5 * alive) * open
		if a <= 0.02:
			return
		var v := clampf(leaf_val * shade * (0.62 + 0.3 * band) * lit, 0.05, 1.3)
		var col := _fog(Color.from_hsv(fposmod(leaf_hue + float(tr["hue"]) + leaf_turn, 1.0),
			clampf(leaf_sat * 0.85, 0.03, 0.95), v, a), fogt * 0.9)
		var puffs: Array = model["puffs"]
		for entry in puffs:
			var pf: Dictionary = entry
			var pp: Vector3 = pf["p"]
			var wp := _pose(pp, base, s, cy, sy, bend, inv_h)
			Branch3D.billboard_fan(faces, lens, u, wp, _cam_r, _cam_u,
				float(pf["r"]) * s * open, col, LEAF_SIDES, 0.32, float(pf["ragged"]))
