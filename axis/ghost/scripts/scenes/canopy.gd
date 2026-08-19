extends Scene3D

## Canopy - trees growing on real terrain, from taproot to leaf, through one season.
##
## A wooded slope under a low raking sun. Two to six hundred trees stand on a real
## [Terrain] heightfield - not a decal, not a sprite: each one is a genuine 3D branching
## structure grown by [Branch3D], a tapered trunk lofted as a closed 6-sided tube with limbs
## forking away at species-specific angles, a divergence roll between successive whorls, and
## twigs thinning into soft leaf masses. Trunks and crowns rasterize into a [ShadowField], so
## the hillside is striped with the wood's own shadows, and the whole thing merges into
## [method Terrain.collect_surface]'s quad list before the depth sort - which is what makes a
## hill occlude the trees standing behind it instead of the wood floating over the land.
##
## THE FOREST IS INSTANCED, and that is the only reason it fits. A handful of MODELS (three to
## five per stand) are grown once in [method build_params]; every tree is one of them, yawed,
## scaled and BENT at draw time. That is furry's trick - grow once in local space, re-lean at
## draw - lifted into three dimensions, and it turns six hundred L-systems into eight.
##
## WHAT A WOOD LOOKS LIKE, which is the whole subject of this scene and was got wrong in four
## separate ways at once. Reported: "the trees always look kind of weird. The way they grow is
## weird, their size seems too large, relative to the terrain, and their placement is far too
## sparse. Proper tree coverage will be dense in some places, sparse in others - and it will
## track features of the terrain." Each of those is a number, and tests/canopy_scatter_check.gd
## holds all of them:
##
##   SIZE. A tree is a small fraction of the landscape's own vertical range - a mean of 0.11 of
##   the relief here, against 0.78 before, when the average tree was three quarters as tall as
##   the whole hillside and the tallest was half again taller than it. Height is sampled AS a
##   fraction of relief now, so it cannot drift when a landform changes.
##
##   DENSITY. Two to six hundred trees rather than thirty to a hundred and ten.
##
##   DISTRIBUTION. Stands, built as stands: clump centres rejection-sampled against a density
##   field, trees scattered around them with a gaussian falloff, and a minority of stragglers on
##   the open ground between. A minimum-distance test cannot do this - forbidding close
##   neighbours is its whole job - and one measured as a lattice (nearest-neighbour spread 0.20,
##   where a random scatter is 0.52 and stands are above 0.6). It reads 0.9 to 1.3 now.
##
##   WHERE. [method _density_at]: a tree line the wood thins out under, wetness that gathers it
##   in the valleys (or up the ridges - the elevation preference is one signed sample), a grove
##   field that opens clearings, and a slope term. The four rules this replaced were each a
##   power of height, which is monotonic and so cannot know where a valley is.
##
##   FOLIAGE. [constant FOLIAGE] - broadleaf, needle, frond, blossom, scrub - deciding cluster
##   size (as a fraction of the crown, so it scales with the tree), rim shape, and the band the
##   mood may move its colour inside. A wood carries TWO STANDS, a canopy and an understory,
##   each with its own architecture, hue and leaf form, and a share of clusters in a second hue
##   taken along the arc toward the scheme's own base. Crowns are shaded by which side of the
##   crown a cluster sits on, which is what makes a cluster cloud read as a lit mass.
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
## What the seed decides: the season and its mood; the wooded landform and a climate that suits
## both;
## the sun's elevation (8-35 degrees, kept low for long shadows) and its drift direction; one
## species rule table (fork angle, divergence roll, length/width ratios, depth, children,
## crown fraction, buttress flare, trunk lean, leaf size); how many models and how many trees;
## the density field the trees scatter under (tree line, wetness, groves, elevation preference)
## and the stands they gather into; the slope they refuse to root on; the gust's speed, width and
## strength; and the LOD distances.

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
	# `mesa` is deliberately absent from canopy's own list below: a fractured plateau of scarps
	# is not where a wood grows, and at the relief this scene needs it renders as spikes.
	"mesa":    ["arid", "temperate"],
	"islands": ["verdant", "temperate"],
}

## FOLIAGE ARCHETYPES - what the leaf mass IS, which is a different question from what colour
## it is and was previously not asked at all: every tree in every wood carried the same round
## cluster at the same size, and the size was small enough that a crown read as a handful of
## specks on a stick. Reported as trees that "look kind of weird... the way they grow is weird".
##
## `size`   cluster radius as a fraction of the model's CROWN radius, against the ~140 sites
##          Branch3D lays through the outer orders. The two numbers are one decision: a few big
##          clusters overlap into a smooth mass (a cloud), and many small ones read as grain (a
##          canopy). Both extremes have been rendered and looked at - specks on a stick at one
##          end, plates of coloured card at the other.
## `sides`  rim points on the billboard fan, and `ragged` how far the rim wanders: a needle
##          cluster is a spiky little tuft, a broadleaf mass is a round blob, a frond is torn.
## `hard`   how much alpha the cluster's RIM keeps (see Branch3D.billboard_fan). This is the
##          coarse/fluffy dial: at 0 a cluster is a soft radial gradient - a cloud puff - and a
##          crown of them is a cloud, which is what "a little too fluffy... proper foliage would
##          be more coarse" was. Needles are hardest (a spruce tuft is a solid dark spike),
##          fronds softest, and nothing is 0.
## `sat`/`val` the band the mood is allowed to move the foliage inside. NOT a multiplier on the
##          mood - that is how a `bone` or `ash` scheme produced white-cream leaves. Leaves are
##          saturated things; the mood chooses where in the plausible band, not whether.
## `turn`   scales the season's base-to-tip hue walk.
## `acc`    the share of clusters that take a SECOND hue (the scheme's opposed colour), which is
##          what gives a wood flowers, new growth, or a turning crown that is not uniform.
## `stride` thins the cluster count (a conifer has many small tufts, a palm a few big fronds).
const FOLIAGE := {
	"broadleaf": {"size": [0.15, 0.24], "sides": 8, "ragged": [0.26, 0.44], "hard": [0.55, 0.78],
		"sat": [0.42, 0.78], "val": [0.52, 0.86], "turn": 1.0, "acc": [0.0, 0.12], "stride": 1},
	"needle":    {"size": [0.09, 0.15], "sides": 6, "ragged": [0.50, 0.80], "hard": [0.72, 0.92],
		"sat": [0.34, 0.62], "val": [0.30, 0.54], "turn": 0.5, "acc": [0.0, 0.05], "stride": 1},
	"frond":     {"size": [0.20, 0.32], "sides": 6, "ragged": [0.55, 0.85], "hard": [0.34, 0.55],
		"sat": [0.40, 0.72], "val": [0.46, 0.78], "turn": 0.8, "acc": [0.0, 0.10], "stride": 2},
	"blossom":   {"size": [0.13, 0.22], "sides": 9, "ragged": [0.18, 0.34], "hard": [0.48, 0.70],
		"sat": [0.35, 0.70], "val": [0.62, 0.95], "turn": 1.3, "acc": [0.25, 0.55], "stride": 1},
	"scrub":     {"size": [0.12, 0.20], "sides": 7, "ragged": [0.40, 0.65], "hard": [0.60, 0.85],
		"sat": [0.26, 0.52], "val": [0.38, 0.62], "turn": 0.7, "acc": [0.0, 0.08], "stride": 1},
}

## Which foliage a season may put in a wood. Winter has no blossom and autumn no fresh frond.
const SEASON_FOLIAGE := {
	"spring": ["broadleaf", "broadleaf", "blossom", "needle", "frond", "scrub"],
	"summer": ["broadleaf", "broadleaf", "needle", "frond", "scrub", "blossom"],
	"autumn": ["broadleaf", "broadleaf", "needle", "scrub"],
	"winter": ["needle", "needle", "broadleaf", "scrub"],
}

## How the wood is distributed over the land - a LABEL for the sampled weights below, kept for
## the feedback record. The distribution itself is no longer one of four rules: see
## [method _density_at].
const LAWS := ["even", "ridge", "valley", "glade"]

var _terrain: Terrain
var _models: Array = []              # the grown tree models (immutable after build)
var _stands: Array = []              # one entry per stand: foliage archetype + its colour
# The distribution field (see _density_at).
var _elev_bias := 0.0                # <0 the wood prefers valleys, >0 it climbs the ridges
var _tree_line := 0.8                # normalized height where the wood gives up
var _line_soft := 0.2
var _wet_k := 1.4
var _grove_thr := 0.5
var _grove_soft := 0.1
var _grove_f: Field
var _wet_f: Field
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
var h_cam_clear := 0.4               # how far above the ground the eye is kept (see update)
var _focus := Vector3.ZERO           # the densest part of the wood - what the orbit looks at
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
	# WOODED LANDFORMS ONLY. `mesa` is in the TERRAINS table (other scenes use it) and out of this
	# bag: a fractured plateau of scarps is not where a wood grows, and at the relief a canopy
	# needs it renders as a wall of spikes with the trees hidden in the cracks.
	var wooded := ["hills", "hills", "valleys", "valleys", "islands"]
	var ttype := String(wooded[rng.randi() % wooded.size()])
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
	# RELIEF. A tree may only be a small fraction of the landscape's vertical range (that is what
	# tests/canopy_scatter_check.gd measures), so the range has to be big enough for a small
	# fraction of it to still be a tree: at the old 0.42-0.72 a correctly-proportioned tree was a
	# tenth of a world unit and eleven pixels tall. Deeper relief also gives the wood real slopes
	# to climb and real valleys to gather in, which is the whole subject of the density field.
	# HALF-EXTENT DOWN WITH THE CAMERA. The heightfield is a fixed 112 vertices across whatever
	# it spans, so a six-unit world puts a terrain facet at 0.054 units - a quarter of a tree -
	# and from inside the wood the land reads as bumpy plates. Less ground, same resolution: the
	# facets go under the trees where they belong, and the same 300-600 trees cover it as a
	# forest instead of an orchard.
	# EXTENT BACK OUT, RELIEF UP, AND THE CAMERA IN. Three numbers that only make sense together.
	# A tree may only be a small fraction of the relief (tests/canopy_scatter_check.gd), so the
	# relief has to be deep enough for that fraction to be a tree: at the old 0.42-0.72 it was a
	# tenth of a unit and eleven pixels. But relief over a SMALL extent is a wall of spikes - the
	# slope of this land is relief/half and the recipes are tuned around a sixth - so shrinking
	# the world to compensate (which the first attempt did) made the mesa recipe read as crumpled
	# paper. The answer is a full-size world with deeper relief and a camera a handful of tree
	# heights out, which is also the only framing where a wood reads as a wood.
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.85, 1.35), null, climate)
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
	# TWO STANDS, not one species. A real wood is a canopy and an understory - and even where it
	# is one species, it is not one COLOUR. The dominant stand takes most of the trees; the
	# second is shorter, differently leaved and differently tinted, so the wood has depth and
	# variety without a second L-system per tree.
	_species = Branch3D.sample_species(rng)
	var fol_bag: Array = SEASON_FOLIAGE[_season]
	var nstands := 1 if rng.randf() < 0.18 else 2
	var seg_budget := rng.randi_range(140, 230)
	# HEIGHT IS A FRACTION OF THE LANDSCAPE'S RELIEF, not an absolute. It was 0.30 to 0.56 world
	# units against a relief of 0.42 to 0.72, which made the average tree 78% of the whole
	# hillside's vertical range and the tallest 150% of it - a wood of trees taller than the
	# mountain they stand on. Measured, in tests/canopy_scatter_check.gd, which is where the
	# 0.18-of-relief mean and 0.30 ceiling come from.
	var h_hi: float = _terrain.relief * rng.randf_range(0.13, 0.185)
	for si in nstands:
		var arche := String(fol_bag[rng.randi() % fol_bag.size()])
		var fo: Dictionary = FOLIAGE[arche]
		# The understory is shorter, and reliably: a second stand the same height as the first
		# is not an understory, it is the same wood twice.
		var band := 1.0 if si == 0 else rng.randf_range(0.45, 0.72)
		var sp: Dictionary = _species.duplicate()
		if si > 0:
			# Its own architecture too, within the same genome - a different fork angle and
			# crown fraction is what stops the understory reading as scaled-down copies.
			sp["angle"] = clampf(float(sp["angle"]) * rng.randf_range(0.75, 1.30), 14.0, 62.0)
			sp["crown"] = clampf(float(sp["crown"]) * rng.randf_range(0.7, 1.35), 0.18, 0.78)
			sp["droop"] = clampf(float(sp["droop"]) + rng.randf_range(-0.1, 0.18), 0.0, 0.5)
		var nmodels := rng.randi_range(3, 5)
		# The stand's foliage look, resolved once. Saturation and value come from the
		# archetype's own band with the mood choosing WHERE in it - see the FOLIAGE table.
		var fsat := lerpf(float(fo["sat"][0]), float(fo["sat"][1]),
			clampf(_sch.sat * rng.randf_range(0.8, 1.2), 0.0, 1.0))
		var fval := lerpf(float(fo["val"][0]), float(fo["val"][1]),
			clampf(_sch.val * rng.randf_range(0.8, 1.15), 0.0, 1.0))
		var stand := {
			"arche": arche,
			"dh": 0.0 if si == 0 else rng.randf_range(-0.09, 0.09),
			"sat": fsat * sat_mul,
			"val": fval * val_mul,
			"sides": int(fo["sides"]),
			"ragged": rng.randf_range(float(fo["ragged"][0]), float(fo["ragged"][1])),
			"hard": rng.randf_range(float(fo["hard"][0]), float(fo["hard"][1])),
			"turn": float(fo["turn"]),
			"acc": rng.randf_range(float(fo["acc"][0]), float(fo["acc"][1])),
			# The accent clusters' hue shift, taken along the arc from the foliage hue toward the
			# scheme's OWN base rather than as a free offset: at +-0.3 of a turn a green wood grew
			# cyan blossom, which is the sort of colour nothing in a landscape has.
			"acc_dh": (fposmod(_sch.hue - _sch.accent + 0.5, 1.0) - 0.5)
				* rng.randf_range(0.45, 1.0),
			"stride": int(fo["stride"]),
			"scale": [0.78 * band, 1.22 * band],
			"share": 1.0 if si == 0 else rng.randf_range(0.18, 0.45),
		}
		_stands.append(stand)
		for _mi in nmodels:
			var h := h_hi * band * rng.randf_range(0.72, 1.0)
			var sp2: Dictionary = sp.duplicate()
			# Cluster size is set from the CROWN this model will actually have, which is why the
			# archetype speaks in fractions of it: a leaf size in world units is meaningless
			# across models that differ in height by a factor of three.
			sp2["leaf_size"] = float(sp2["leaf_size"])
			# Trunk radius as a fraction of height. Real proportion is nearer 2%, but at the
			# framing this scene uses that is a two-pixel line in shadow: the trees came out as
			# floating crowns with no visible wood under them at all.
			var model: Dictionary = Branch3D.grow(sp2, rng, h,
				h * rng.randf_range(0.032, 0.052), sun, seg_budget)
			var crad: float = maxf(0.001, float(model["crown_r"]))
			var csize := crad * rng.randf_range(float(fo["size"][0]), float(fo["size"][1]))
			var lv: Array = model["leaves"]
			for entry in lv:
				var lf: Dictionary = entry
				# The cluster, re-sized off the crown, keeping the per-leaf variation Branch3D
				# rolled. THIS is what turns specks on a twig into a canopy.
				lf["size"] = csize * rng.randf_range(0.72, 1.28)
				# Release thresholds: which leaves are ever sheddable, and in what order.
				# Sampled HERE, at build, off the seeded rng - never on an audio event.
				lf["release"] = rng.randf_range(0.03, 0.45) if rng.randf() < bare else 99.0
				# ... and which of them carry the stand's SECOND hue.
				lf["acc"] = 1.0 if rng.randf() < float(stand["acc"]) else 0.0
				# DAPPLE. Every cluster taking the same value is the other half of the cloud
				# look: a smooth mass. Real foliage is a hundred separate things catching the
				# light separately, so each cluster is a little brighter or darker than its
				# neighbour for good.
				lf["dap"] = rng.randf_range(0.74, 1.26)
			model["stand"] = si
			_models.append(model)

	# THE SCATTER. Rejection against the real surface: never in the water, never on ground too
	# steep to root, then weighted by the distribution law, then spaced by a Poisson-disc test
	# through a hash grid (the naive all-pairs version is 480k distance checks at this count,
	# and build_params runs synchronously on the main thread at every cut).
	# THE DENSITY FIELD (see _density_at) - sampled weights, not one of four rules.
	_elev_bias = rng.randf_range(-1.0, 0.55)     # negative: valleys. positive: ridges.
	_tree_line = rng.randf_range(0.55, 0.95)     # where the wood gives up, in normalized height
	_line_soft = rng.randf_range(0.10, 0.30)
	_wet_k = rng.randf_range(0.8, 2.2)
	_grove_f = Field.make("fbm", rng.randi(), rng.randf_range(0.55, 1.15), 3)
	_wet_f = Field.make("fbm", rng.randi(), rng.randf_range(0.30, 0.70), 2)
	_grove_thr = rng.randf_range(0.40, 0.56)
	_grove_soft = rng.randf_range(0.06, 0.16)
	_law = "ridge" if _elev_bias > 0.25 else ("valley" if _elev_bias < -0.35 else "even")
	if _grove_thr > 0.52:
		_law = "glade"
	# A WOOD, not a specimen collection. 30 to 110 trees over a six-unit hillside is what
	# "placement is far too sparse" was; the LOD bands (rank-capped, so the frame cost is
	# bounded whatever this is) carry the rest as silhouettes.
	var want := rng.randi_range(260, 620)
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
	# THE WOOD GROWS IN STANDS, and that has to be built rather than hoped for. A minimum-
	# distance (Poisson-disc) test, however its radius is modulated, CANNOT produce stands: its
	# whole job is to forbid close neighbours, so what comes out is evenly spread by
	# construction. Measured on the nearest-neighbour coefficient of variation, where a random
	# scatter reads 0.52 and real stands read above 0.6: one radius everywhere gave 0.20, and
	# scaling the radius by the density field only reached 0.34. That is the arithmetic behind
	# "placement is far too sparse... proper tree coverage will be dense in some places, sparse
	# in others".
	#
	# So: CLUMP CENTRES first, rejection-sampled against the density field (so a stand sits
	# where a stand would - in the wet valley, inside a grove, below the tree line), then trees
	# scattered around each centre with a gaussian falloff, each still tested against the land
	# and against the field. A minority are STRAGGLERS, placed the old way, which is what keeps
	# the gaps between stands from reading as walls. The only distance test left is a trunk
	# guard - trees may crowd, they may not intersect.
	var spacing: float = _terrain.half * 2.0 / sqrt(float(want))
	# The trunk guard, and it is DELIBERATELY SMALL. It is the only distance test left, and its
	# job is to stop two trees occupying one trunk - not to space the wood. Set anywhere near the
	# nominal spacing it saturates every stand instead: a stand packed to its limit has every
	# tree exactly one guard from its neighbour, which measures as a lattice however far apart
	# the stands are (0.39-0.52 on the nearest-neighbour spread, against 0.55 for a wood).
	var guard := spacing * rng.randf_range(0.07, 0.16)
	var hmax := 0.001
	for hv in _terrain.hgrid:
		hmax = maxf(hmax, hv)
	hmax = maxf(0.05, hmax - _terrain.water)
	var cells: Dictionary = {}
	var inv_cell := 1.0 / maxf(0.001, guard * 3.0)
	# Stand size and spread, sampled: a wood of a few big stands is a different wood from one of
	# many small copses, and both are woods.
	var clump_n := rng.randi_range(5, 14)
	# THE STANDS MUST NOT TILE THE HILL. Sampling a count and a radius independently let the two
	# multiply into full coverage - eighteen stands of a quarter-hill radius is one continuous
	# wood, and its nearest-neighbour spread measured 0.21, back at a lattice. So the radius is
	# SOLVED from a target coverage: whatever the count, the stands together claim a third or so
	# of the usable ground, which is what guarantees the gaps between them.
	var cover := rng.randf_range(0.09, 0.26)
	var usable := (2.0 * margin) * (2.0 * margin)
	var clump_r: float = sqrt(cover * usable / (PI * float(clump_n)))
	var stragglers := rng.randf_range(0.06, 0.18)
	var centres: Array = []
	var ctries := 0
	while centres.size() < clump_n and ctries < clump_n * 60:
		ctries += 1
		var cxw := rng.randf_range(-margin, margin)
		var czw := rng.randf_range(-margin, margin)
		var chh: float = _terrain.height_at(cxw, czw)
		if chh <= 0.02:
			continue
		var cn: Vector3 = _terrain.normal_world(cxw, czw)
		if cn.y < slope_min:
			continue
		var cd := _density_at(cxw, czw, clampf(chh / hmax, 0.0, 1.0), cn.y)
		# A stand centre is held to a HIGHER bar than a single tree: this is where the wood is
		# thickest, so it belongs on ground that would carry a wood.
		if rng.randf() > cd * cd:
			continue
		centres.append(Vector2(cxw, czw))
	# PER-STAND QUOTAS, and unequal ones. A stand filled to its packing limit is locally a
	# lattice - every tree exactly the trunk guard from its neighbour - so a wood of saturated
	# stands measures as evenly spread however far apart the stands are. Weighting the quotas
	# heavily (a squared uniform) makes some stands thick and others a handful of trees, which
	# is what a real wood looks like and what the nearest-neighbour spread actually reads.
	var quotas := PackedFloat32Array()
	var wsum := 0.0
	for ci in centres.size():
		var w := rng.randf()
		w = w * w * rng.randf_range(0.5, 1.5) + 0.05
		quotas.append(w)
		wsum += w
	var placed_target := int(round(float(want) * (1.0 - stragglers)))
	var tries := 0
	for ci in centres.size():
		var c: Vector2 = centres[ci]
		var quota := int(round(placed_target * quotas[ci] / maxf(0.0001, wsum)))
		# Each stand crowds to its own degree - understory thickets sit tighter than a stand of
		# mature crowns.
		var guard_i := guard * rng.randf_range(0.7, 1.7)
		# ...and its own SPREAD. One radius for every stand is the same saturation trap one
		# level up: a tight copse and a broad open stand are different things, and the mix of
		# the two is most of what the nearest-neighbour spread is measuring.
		var clump_ri := clump_r * rng.randf_range(0.45, 1.85)
		var got := 0
		var t2 := 0
		while got < quota and t2 < quota * 14 + 40:
			t2 += 1
			tries += 1
			# Gaussian about the centre (three uniforms summed is close enough and cheap), so a
			# stand has a dense heart and a ragged edge rather than a hard rim.
			var off := Vector2(rng.randf() + rng.randf() + rng.randf() - 1.5,
				rng.randf() + rng.randf() + rng.randf() - 1.5) * clump_ri
			if _place_tree(rng, clampf(c.x + off.x, -margin, margin),
					clampf(c.y + off.y, -margin, margin), slope_min, hmax, guard_i,
					inv_cell, cells, true):
				got += 1
	# ... and the stragglers, on open ground between the stands, which is what keeps the gaps from
	# reading as walls. CAPPED AT THEIR OWN SHARE, and not used to backfill: when the stands
	# under-deliver (a landform with little rootable ground, a field that rejects most of it) the
	# wood is simply smaller. Filling the shortfall with uniform scatter was what left one seed
	# in ten reading as a lattice - the stragglers had quietly become most of the wood.
	var straggle_want := int(round(float(want) * stragglers))
	var straggled := 0
	while straggled < straggle_want and tries < want * 45:
		tries += 1
		if _place_tree(rng, rng.randf_range(-margin, margin), rng.randf_range(-margin, margin),
				slope_min, hmax, guard, inv_cell, cells, false):
			straggled += 1


	# WHERE THE SHOT LOOKS. The wood is in stands now, so the middle of the map is as likely to be
	# bare ground as trees - and a canopy scene framed on an empty hillside is a bug however good
	# the wood is elsewhere. The orbit centres on the densest part of the wood: every tree votes,
	# weighted by how many neighbours it has, so the target lands inside a stand rather than at
	# the mean of two stands with a gap between them.
	if not _trees.is_empty():
		var best := Vector3.ZERO
		var best_w := -1.0
		var probe := mini(_trees.size(), 96)
		for k in probe:
			var tr: Dictionary = _trees[(k * 7 + 3) % _trees.size()]
			var pp: Vector3 = tr["pos"]
			var w := 0.0
			for j in _trees.size():
				var qq: Vector3 = (_trees[j] as Dictionary)["pos"]
				if absf(qq.x - pp.x) < clump_r and absf(qq.z - pp.z) < clump_r:
					w += 1.0
			if w > best_w:
				best_w = w
				best = pp
		_focus = Vector3(best.x, 0.0, best.z)

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
	# BARK IS BROWN, tinted toward the scheme rather than taken from it. Straight off the mood
	# it produced teal trunks under a `teal` scheme and green ones under `verdant` - visible in
	# the render as sticks the colour of the grass.
	_bark_hue = _blend_hue(rng.randf_range(0.045, 0.10), _sch.hue,
		rng.randf_range(0.12, 0.30))
	# BARK IS WOOD. Drained off the mood it came out near-white on any pale scheme (bone, ash,
	# glacier), so the trees read as bleached sticks - visible in every render of this scene.
	# The mood now chooses where inside a plausible bark band, the way the foliage does.
	_bark_sat = clampf(lerpf(0.30, 0.58, clampf(_sch.sat, 0.0, 1.0))
		* rng.randf_range(0.85, 1.15), 0.12, 0.8)
	# Dark enough to be wood, light enough to READ against a shaded hillside: at 0.16-0.38 the
	# trunks disappeared into the forest floor entirely, which is its own kind of wrong.
	_bark_val = clampf(lerpf(0.34, 0.58, clampf(_sch.val, 0.0, 1.0)) * val_mul
		* rng.randf_range(0.9, 1.12), 0.14, 0.72)
	_leaf_hue = fposmod(_sch.accent + rng.randf_range(-0.03, 0.03), 1.0)
	_leaf_sat = clampf(_sch.sat * sat_mul * rng.randf_range(0.7, 1.05), 0.05, 0.95)
	_leaf_val = clampf(_sch.val * val_mul * rng.randf_range(0.78, 1.05), 0.1, 1.2)
	var to_accent := fposmod(_sch.opposed(_leaf_hue, 0.18) - _leaf_hue + 0.5, 1.0) - 0.5
	_leaf_turn = to_accent * rng.randf_range(float(sea["turn"][0]), float(sea["turn"][1])) * 1.6
	_fog_col = Color.from_hsv(fposmod(_sch.hue + rng.randf_range(0.38, 0.58), 1.0),
		rng.randf_range(0.04, 0.16), rng.randf_range(0.55, 0.86))

	# CAMERA - IN THE WOOD, not above the range. This moved with the tree size and had to: a
	# tree a fifth of the relief is a tenth of a world unit tall, and from eight units back that
	# is eleven pixels of tree. A forest is photographed from inside it - a slope filling the
	# frame - which is also the only framing in which a canopy reads as a canopy rather than as
	# green texture. The heightfield is six units across, so this still sees a third of it and
	# the far rim stays above the frame at these pitches (no void band, which is what the old
	# distance was protecting against).
	lens.fov = rng.randf_range(46.0, 58.0)
	# Off the TERRAIN, not absolute: the orbit has to clear hills that are now up to 1.65 units
	# tall, and the framing has to hold whatever relief the seed rolled.
	# MEASURED IN TREE HEIGHTS, which is the only unit that frames a wood. Anything expressed as
	# a fraction of the terrain scales with the land, so raising the relief to make the trees
	# correctly proportioned pushed the camera back by exactly as much and every tree stayed
	# twenty pixels of green blob however the numbers moved. Six to fourteen tree-heights back,
	# near the ground: the near rank reads as architecture, the wood recedes behind it, and the
	# 500 trees the hillside carries are mostly BEHIND the frame rather than in it.
	# FRAMED ON THE WOOD'S OWN SPACING, which is the thing that decides how many trees are in
	# shot. Tree-heights alone could not: relief and height are sampled independently, so the
	# same multiplier framed six trees on one seed and the inside of a crown on the next. This
	# puts roughly eight to eighteen tree-spacings across the frame, floored so a tree is never
	# sub-pixel and capped so the shot never backs out into the aerial view this scene came from.
	_dist = clampf(spacing * rng.randf_range(7.0, 16.0),
		maxf(0.9, h_hi * 7.0), _terrain.half * 1.45)
	h_cam_clear = h_hi * rng.randf_range(1.0, 2.2)
	_pitch = rng.randf_range(0.16, 0.34)
	_look_y = h_hi * rng.randf_range(0.45, 0.95)
	_yaw = rng.randf() * TAU
	_yaw_dir = 1.0 if rng.randf() < 0.5 else -1.0

	# LOD. Sampled, because how far "near" reaches decides both the look and the frame cost.
	# These are measured from the NEAREST tree rather than from the camera: the orbit distance
	# and the landform between them decide where the wood actually starts, and an absolute
	# threshold would either articulate nothing (camera pulled back) or everything (pushed in).
	# So this is the depth INTO the wood that detail reaches, which is what a LOD distance
	# means when the subject is a field rather than an object.
	# Scaled with the camera above: these are depths INTO the wood, and the wood now starts a
	# couple of units away rather than eight.
	_lod_near = rng.randf_range(0.9, 1.7)
	_lod_far = _lod_near + rng.randf_range(0.8, 1.6)
	_near_max = rng.randi_range(8, 12)
	_mid_max = _near_max + rng.randi_range(18, 30)

	_light = Lighting.new(rng, 3)

	# A SKY behind the land, and it is not decoration. A heightfield has no side skirt and no
	# horizon: past its far rim there is simply nothing, and at these low pitches the rim sits
	# below the top of the frame, so a bare canopy scene would put a band of void above its own
	# ridge. `bed` (as a BACK layer - it defaults to front) fills that band with the same pale
	# distance the fog already fades the far trees toward, so the ridge line reads as a skyline.
	add_layer("bed", rng, {"z": "back", "hue": _fog_col.h, "sat": _fog_col.s + 0.06,
		"val": rng.randf_range(0.34, 0.58), "pools": rng.randi_range(2, 4)})

	# The season's air, and an optional coverage veil / low sun-shafts.
	var ground: Array = sea["ground"]
	var pick: Dictionary = ground[rng.randi() % ground.size()]
	_ground_layer = String(pick["kind"])
	# THINNED FOR THE NEW FRAMING. These counts were set for an orbit eight units back, where a
	# mote is a speck; from inside the wood the same count is a snowstorm of pale dots that reads
	# as popcorn over the hillside (and was mistaken, looking at a render, for the trees).
	var air_n := int(round(rng.randi_range(int(pick["n"][0]), int(pick["n"][1])) * 0.42))
	add_layer(_ground_layer, rng, _air_cfg(_ground_layer, maxi(3, air_n)))
	if rng.randf() < float(sea["veil"]):
		add_layer("veil", rng, {"hue": _fog_col.h, "sat": 0.05, "val": _fog_col.v,
			"max": rng.randf_range(0.18, 0.38)})
	# Rarer and thinner than before: from inside the wood these are full-height curtains across
	# the frame rather than the distant shafts they were at an eight-unit orbit.
	if _light_el < deg_to_rad(18.0) and rng.randf() < 0.15:
		add_layer("rays", rng, {"count": rng.randi_range(2, 3),
			"hue": fposmod(_sch.accent + 0.02, 1.0), "z": "front"})
	return {"season": _season, "mood": _sch.name, "terrain": ttype, "climate": climate,
		"law": _law, "trees": _trees.size(), "models": _models.size(),
		"stands": _stands.size(),
		"foliage": ", ".join(_stand_names()), "elev_bias": _elev_bias,
		"tree_line": _tree_line,
		"depth": int(_species["depth"]), "children": int(_species["children"]),
		"angle": float(_species["angle"]), "roll": float(_species["roll"]),
		"ratio": float(_species["ratio"]), "taper": float(_species["taper"]),
		"crown": float(_species["crown"]), "leaf_density": _leaf_density,
		"sun_deg": rad_to_deg(_light_el), "gust_speed": _gust_speed,
		"gust_width": _gust_width, "lod_near": _lod_near, "slope_reject": reject,
		"air": _ground_layer}


## HOW DENSE THE WOOD IS AT ONE PLACE, in 0..1, and the reason the scatter now tracks the land.
##
## What was here was one of four rules - even, ridge, valley, glade - each a single power of the
## normalized height, and the result was reported as looking wrong for exactly the reason it was:
## "proper tree coverage will be dense in some places, sparse in others - and it will track
## features of the terrain. For example, valleys might have dense trees, while peaks may have
## none." A power of height cannot do that: it is monotonic, so it thins in one direction and
## has no idea where a valley is.
##
## Four terms, each a thing that actually decides where trees grow:
##
##   THE TREE LINE. Above it, nothing - and it is a soft edge, so the wood thins into scattered
##   stragglers before it stops. This is the term that empties the peaks.
##
##   WETNESS. Low ground holds water, so the valleys carry the dense wood. A monotonic
##   preference on its own reads as a gradient, so it is mixed with a low-frequency moisture
##   field: some high ground is wet and some low ground is dry, which is what stops the
##   elevation preference being visible AS a rule.
##
##   GROVES. A low-frequency field thresholded hard: the wood is in stands with real gaps
##   between them. This is the term that produces "dense in some places, sparse in others"
##   independently of the terrain, and it is what a rejection test against a monotonic weight
##   could never do.
##
##   SLOPE. Gentle ground holds more than a steep flank (which the hard slope rejection has
##   already cleared of anything unrootable).
##
## `_elev_bias` runs the elevation term from valley-loving through indifferent to ridge-loving,
## so a wood that climbs is one sample rather than a separate branch.
## One candidate tree, tested against the land and the density field and (if it survives) added.
## Returns whether it took.
func _place_tree(rng: RandomNumberGenerator, wx: float, wz: float, slope_min: float,
		hmax: float, guard: float, inv_cell: float, cells: Dictionary,
		in_clump: bool) -> bool:
	var hh: float = _terrain.height_at(wx, wz)
	if hh <= 0.015:
		return false                                   # in the water, or on the shoreline
	var nrm: Vector3 = _terrain.normal_world(wx, wz)
	if nrm.y < slope_min:
		return false                                   # too steep to hold a root plate
	var hn := clampf(hh / hmax, 0.0, 1.0)
	var dens := _density_at(wx, wz, hn, nrm.y)
	# A tree INSIDE a stand is held to a gentler bar than a straggler on open ground: the stand
	# is already established here, so the field only has to not forbid it. pow(d, 0.45) lifts the
	# middle of the range and still returns zero where the field does - so the tree line stays a
	# tree line and a stand cannot creep over a peak.
	if rng.randf() > (pow(dens, 0.45) if in_clump else dens):
		return false
	var cx := int(floor(wx * inv_cell))
	var cz := int(floor(wz * inv_cell))
	for dz in [-1, 0, 1]:
		for dx in [-1, 0, 1]:
			var key := Vector2i(cx + dx, cz + dz)
			if not cells.has(key):
				continue
			for o in (cells[key] as Array):
				if (o as Vector2).distance_to(Vector2(wx, wz)) < guard:
					return false                       # trees may crowd; they may not intersect
	var key2 := Vector2i(cx, cz)
	if not cells.has(key2):
		cells[key2] = []
	(cells[key2] as Array).append(Vector2(wx, wz))
	var mi := rng.randi() % _models.size()
	if _stands.size() > 1:
		var want_s := 0 if rng.randf() > float((_stands[1] as Dictionary)["share"]) else 1
		for k in _models.size():
			var cand: Dictionary = _models[(mi + k) % _models.size()]
			if int(cand["stand"]) == want_s:
				mi = (mi + k) % _models.size()
				break
	var model: Dictionary = _models[mi]
	var st: Dictionary = _stands[int(model["stand"])]
	var srange: Array = st["scale"]
	var sc := rng.randf_range(float(srange[0]), float(srange[1]))
	var yaw := rng.randf() * TAU
	var gy: float = hh * _terrain.relief - _embed
	_trees.append({
		"pos": Vector3(wx, gy, wz), "model": mi, "scale": sc,
		"cy": cos(yaw), "sy": sin(yaw),
		"h": float(model["height"]) * sc,
		"trunk": float(model["radius"]) * sc * float(_species["flare"]),
		"crown_r": float(model["crown_r"]) * sc,
		"stand": int(model["stand"]),
		"phase": rng.randf() * TAU,
		"rate": rng.randf_range(2.6, 4.6),
		"delay": rng.randf_range(0.0, 0.55),
		"shed_bias": rng.randf_range(0.7, 1.35),
		"hue": rng.randf_range(-0.04, 0.04),
		# Pinned to a spectral position by ELEVATION, so the ridge line rings with the top
		# of the spectrum and the valley floor with the bottom.
		"t": hn})
	return true


func _stand_names() -> PackedStringArray:
	var out := PackedStringArray()
	for st in _stands:
		out.append(String((st as Dictionary)["arche"]))
	return out


func _density_at(wx: float, wz: float, hn: float, ny: float) -> float:
	var line := 1.0 - smoothstep(_tree_line - _line_soft, _tree_line + _line_soft, hn)
	var wet_pref := pow(clampf(1.0 - hn, 0.0, 1.0), _wet_k) if _elev_bias <= 0.0 \
		else pow(hn, _wet_k)
	var strength := absf(_elev_bias)
	var moisture := clampf(_wet_f.at(Vector2(wx, wz)) * 1.15, 0.0, 1.0)
	var wet := lerpf(0.55 + 0.45 * moisture, wet_pref * (0.45 + 0.75 * moisture), strength)
	var grove := smoothstep(_grove_thr - _grove_soft, _grove_thr + _grove_soft,
		_grove_f.at(Vector2(wx, wz)))
	var slope := smoothstep(0.0, 0.35, ny - 0.45)
	return clampf(line * wet * (0.10 + 0.95 * grove) * (0.45 + 0.65 * slope), 0.0, 1.0)


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
	lens.orbit(Vector3(_focus.x, _terrain.height_at(_focus.x, _focus.z) * _terrain.relief
		+ _look_y, _focus.z), _dist, _yaw, _pitch + 0.02 * sin(_life * 0.09))
	# THE EYE MAY NOT BE INSIDE A HILL. At a couple of tree-heights out and a ten-degree pitch the
	# orbit passes through ground that is up to the whole relief tall, and a camera under the
	# surface renders the inside of the heightfield - a wall of dark quads. Lifting the eye to
	# clear the land beneath it (plus a couple of tree-heights of headroom) keeps the low framing
	# without ever putting the camera in the dirt; the look point is unchanged, so the shot only
	# rises over the high ground and settles back down over the low.
	var ex: float = lens.eye.x
	var ez: float = lens.eye.z
	var ground: float = _terrain.height_at(ex, ez) * _terrain.relief
	var floor_y: float = ground + h_cam_clear
	if lens.eye.y < floor_y:
		# AND THE LOOK POINT RISES WITH IT. Lifting the eye alone turns a shallow shot into a
		# top-down one whenever the orbit passes over high ground - the target stays down at the
		# focus stand and the camera ends up looking at the tops of everything. Carrying most of
		# the lift into the target keeps the view roughly parallel to the hillside, which is the
		# framing that shows trees standing rather than a canopy from above.
		# CAPPED, and only half of it carried. Raising the target by the whole lift aims the shot
		# clean over the stand it was framed on - the eye climbs a ridge, the look point climbs
		# with it, and the wood ends up below the bottom of the frame with bare hillside in the
		# middle of it. Half the lift keeps the view shallow without losing the subject, and the
		# cap stops a hill the camera happens to cross from re-aiming the shot at all.
		var lift: float = minf(floor_y - lens.eye.y, _dist * 0.35)
		lens.eye = Vector3(ex, maxf(lens.eye.y, floor_y), ez)
		lens.look = lens.look + Vector3(0.0, lift * 0.5, 0.0)
	# LINE OF SIGHT TO THE WOOD. Orbiting a point inside hill country puts a hill between the
	# camera and the stand it was framed on for part of every revolution, and the shot is then
	# bare foreground with the subject hidden behind it - rendered and looked at, twice. Marching
	# the ray and lifting the eye until the ground clears it costs eight height samples a frame
	# and means the wood is always actually in the picture. The lift is converted from "clearance
	# needed at this point along the ray" back to "lift at the eye" by the remaining fraction, and
	# capped, so a mountain in the way raises the shot rather than teleporting it.
	var need := 0.0
	for k in range(1, 9):
		var t := float(k) / 9.0
		var gy: float = _terrain.height_at(lerpf(lens.eye.x, lens.look.x, t),
			lerpf(lens.eye.z, lens.look.z, t)) * _terrain.relief + h_cam_clear * 0.6
		var ry: float = lerpf(lens.eye.y, lens.look.y, t)
		if gy > ry:
			need = maxf(need, (gy - ry) / maxf(0.12, 1.0 - t))
	if need > 0.001:
		lens.eye = lens.eye + Vector3(0.0, minf(need, _dist * 1.1), 0.0)
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
	job.size = size
	job.clock = _sim.elapsed()
	job.life = _life
	job.reveal = smoothstep(0.8, 1.0, view.presence)
	job.glow = _glow
	# [Lighting]'s glow drives the bark RIM and nothing else - the one place a beat is allowed
	# to touch the wood. It moves light across a trunk; it does not move the trunk.
	job.rim = _light.glow()
	# Raised with the tree count: five hundred crowns cast five hundred shadows into the same
	# map, so the floor they stand on is far darker than it was at sixty and the wood needs the
	# key light back up to read at all.
	job.lit = clampf(0.82 + 0.4 * _glow + 0.3 * f.energy, 0.5, 1.5)
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
	job.stands = _stands
	job.fog_col = _fog_col
	# AERIAL PERSPECTIVE IS A PROPERTY OF THE LANDSCAPE, not of where the camera happens to
	# stand. Tied to `_dist` (which it was), pulling the camera into the wood pulled the fog in
	# with it: at a three-unit orbit the haze began 1.4 units away and was total by 6, so most of
	# the wood was better than half faded to the pale sky colour and the foliage measured at
	# saturation 0.12 - grey popcorn on a hillside, which is what it looked like. Anchored to the
	# terrain's own extent the far rim hazes and the wood in front of it keeps its colour.
	job.fog_near = _dist + _terrain.half * 0.45
	job.fog_far = _dist + _terrain.half * 2.3
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
	## Shadow-map resolution, and it goes DOWN, which is the opposite of the instinct.
	##
	## The map is only ever READ at terrain vertices - Terrain.collect_surface calls
	## shadow.factor() once per vertex of a 112x112 lattice over 2 * half world units, so the
	## samples sit 0.054 world units apart and the result is Gouraud-interpolated across each
	## quad. The ShadowField default of 220 puts a texel at 0.043 units: FINER than the lattice
	## that reads it. A shadow map sampled below its own resolution can only alias, and that is
	## the blockiness - a trunk 0.042 to 0.059 units wide sits right at the lattice's Nyquist
	## limit and pops in and out of single vertices as the sun drifts.
	##
	## 128 puts a texel at 0.074 units, about 1.4 vertex spacings, so ShadowField's bilinear tap
	## has something to interpolate BETWEEN and a shadow edge feathers over roughly two quads
	## instead of snapping between them. It also costs a quarter of the memory: 16k floats
	## against 48k, cleared once per built frame.
	const SHADOW_RES := 128

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
	## One entry per stand (see the scene's FOLIAGE table): the foliage look each tree draws
	## with. Hue/sat/val here are DELTAS and multipliers on the job's global leaf colour, so the
	## live chroma modulation still reaches every stand.
	var stands: Array = []
	var size := Vector2(1920, 1080)  # the frame, for the frustum cull in run()
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
		# FRUSTUM CULL FIRST, and it is not an optimization detail - it is what makes a wood of
		# five hundred trees affordable at all. The camera stands among them now, so most of the
		# hillside's trees are behind it or off to the side: measured at 1920x1080, emitting all
		# of them cost 283 ms a frame (and under --export a frame's build IS the render's wall
		# time). One projection per tree replaces five billboards per tree for everything out of
		# shot. Trees are culled from the DRAW only - every tree still casts into the shadow map,
		# because a tree behind the camera still throws its shadow across what is in front of it.
		var hx := 0.5 * float(size.x) / maxf(1.0, u)
		var hy := 0.5 * float(size.y) / maxf(1.0, u)
		var order: Array = []
		for i in trees.size():
			var tr: Dictionary = trees[i]
			var p: Vector3 = tr["pos"]
			var mid := p + Vector3.UP * (float(tr["h"]) * 0.6)
			var pc := lens.project(mid)
			if pc.z <= lens.near:
				continue                                  # behind the camera
			# A generous screen radius for the whole tree, from the crown and the trunk height.
			var rad := (float(tr["crown_r"]) + float(tr["h"]) * 0.5) * 2.4 / pc.z
			if absf(pc.x) > hx + rad or absf(pc.y) > hy + rad:
				continue                                  # off the side of the frame
			if rad * u < 0.8:
				continue                                  # sub-pixel: nothing to draw
			order.append({"d": -(p - lens.eye).length(), "i": i})
		order = TriBatch.painter_sort(order)

		# Pass A: rasterize trunks (and, where there is foliage, crowns) into a light-space
		# shadow map, so the hillside is striped with the wood's own shadows and a tree
		# standing behind another is shaded by it.
		var shadow := ShadowField.new()
		shadow.build(ldir, Vector3(-terrain.half, -terrain.relief, -terrain.half),
			Vector3(terrain.half, terrain.relief + 1.4, terrain.half), SHADOW_RES)
		if reveal >= 0.02:
			# EVERY tree casts. This used to take the 64 nearest the CAMERA off the painter order,
			# which is a set that changes as the camera moves - so a tree's shadow blinked out of
			# the hillside the moment it fell to 65th and back in when it rose again, with nothing
			# about the tree or the light having changed. That was the other half of "the shadows
			# aren't stable". A wood is 30 to 110 trees and add_box only fills a small rect, so
			# rasterizing all of them costs less than the map's own clear - spires.gd already
			# rasterizes its whole cast for the same reason.
			for i in trees.size():
				var tr: Dictionary = trees[i]
				var g := _grown(tr)
				if g < 0.1:
					continue
				var p: Vector3 = tr["pos"]
				var th: float = float(tr["h"]) * g
				# ROUND silhouettes (the last argument): a trunk is a cylinder and a crown is a
				# ball, and add_box otherwise rasterizes their bounding SQUARE - which under a wood
				# is a hillside striped with rectangles. See ShadowField.add_box.
				shadow.add_box(p, Vector3.UP, Vector3.RIGHT, Vector3.BACK,
					float(tr["trunk"]), th * 0.7, true)
				if leaf_density > 0.25:
					# 0.72 of the crown, not all of it. The wood is five hundred trees now rather
					# than sixty, and at full radius their casts merge into one blanket that takes
					# the whole hillside down to a fifth of its brightness - a forest floor IS
					# dark, but the trees have to be visible standing on it.
					shadow.add_box(p + Vector3.UP * (th * 0.6), Vector3.UP, Vector3.RIGHT,
						Vector3.BACK, float(tr["crown_r"]) * g * leaf_density * 0.72,
						th * 0.34, true)

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
			_emit_puffs(faces, tr, model, base, s, cy, sy, bend, inv_h, g, band, shade, fogt, ldir)
			return
		if leaf_density <= 0.02:
			return
		# FOLIAGE. Density thins the cluster count by striding the model's leaf sites, and the
		# middle LOD band thins it again - a tree two thirds of the way to the ridge does not
		# need forty billboards to read as leafy.
		var lv: Array = model["leaves"]
		var crown_cen: Vector3 = model["crown"]
		var st: Dictionary = stands[int(tr["stand"])] if not stands.is_empty() else {}
		var st_stride := int(st.get("stride", 1))
		var stride := maxi(1, int(round(float(st_stride) / maxf(0.05, leaf_density))))
		if mode == 1:
			stride *= 3
		var shed_level := shed * float(tr["shed_bias"])
		var st_sat := float(st.get("sat", 0.6))
		var st_val := float(st.get("val", 0.8))
		var st_dh := float(st.get("dh", 0.0))
		var st_turn := float(st.get("turn", 1.0))
		var st_acc := float(st.get("acc_dh", 0.2))
		var st_sides := int(st.get("sides", 7))
		var st_ragged := float(st.get("ragged", 0.25))
		var st_hard := float(st.get("hard", 0.6))
		var lsat := clampf(st_sat * (0.85 + 0.3 * band), 0.03, 0.98)
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
			# The stand's hue, this leaf's own jitter, the base-to-tip turn - and, on the
			# archetype's accent share, a second hue entirely (blossom, new growth, a crown
			# part-turned). `leaf_hue` still carries the live chroma blend for the whole wood.
			var h := fposmod(leaf_hue + st_dh + hue_off + float(lf["hue"])
				+ leaf_turn * st_turn * float(lf["along"])
				+ st_acc * float(lf.get("acc", 0.0)), 1.0)
			# CROWN VOLUME. Every cluster took the same colour, so a crown was a flat green
			# splotch however many billboards went into it. Shading each one by which SIDE of
			# the crown it sits on - toward the key light or away from it - is what turns the
			# cluster cloud into a lit mass, and it costs one dot product per billboard. The
			# offset is yawed with the instance so a rotated copy is lit consistently.
			var away := lpos - crown_cen
			var ax := away.x * cy - away.z * sy
			var az := away.x * sy + away.z * cy
			var facing := clampf((ax * ldir.x + away.y * ldir.y + az * ldir.z)
				/ maxf(0.0001, away.length()) * 0.5 + 0.5, 0.0, 1.0)
			var v := clampf(st_val * shade * (0.68 + 0.36 * band + 0.22 * glow) * lit
				* (0.62 + 0.62 * facing) * float(lf.get("dap", 1.0)), 0.05, 1.35)
			var core := Color.from_hsv(h, lsat, v, reveal * alive * 0.92)
			Branch3D.billboard_fan(faces, lens, u, lp, _cam_r, _cam_u,
				float(lf["size"]) * s, _fog(core, fogt * 0.8), st_sides,
				st_ragged, float(lf["ragged"]), st_hard)

	# The silhouette: a distant tree is its crown, not its architecture. Three to six puffs,
	# sized off the model's real leaf mass so a push-in does not change the wood's shape.
	func _emit_puffs(faces: Array, tr: Dictionary, model: Dictionary, base: Vector3, s: float,
			cy: float, sy: float, bend: float, inv_h: float, g: float, band: float,
			shade: float, fogt: float, ldir := Vector3.UP) -> void:
		if leaf_density <= 0.05 or g < 0.5:
			return
		var open := smoothstep(0.45, 1.0, g) * clampf(leaf_density * 1.2, 0.0, 1.0)
		var alive := 1.0 - smoothstep(0.35, 0.85, shed * float(tr["shed_bias"]))
		var a := reveal * (0.35 + 0.5 * alive) * open
		if a <= 0.02:
			return
		var st: Dictionary = stands[int(tr["stand"])] if not stands.is_empty() else {}
		var v := clampf(float(st.get("val", 0.8)) * shade * (0.62 + 0.3 * band) * lit, 0.05, 1.3)
		var col := _fog(Color.from_hsv(fposmod(leaf_hue + float(st.get("dh", 0.0))
			+ float(tr["hue"]) + leaf_turn * float(st.get("turn", 1.0)), 1.0),
			clampf(float(st.get("sat", 0.6)) * 0.9, 0.03, 0.95), v, a), fogt * 0.9)
		var puffs: Array = model["puffs"]
		# ROUNDER AND WIDER-FACETED THAN A LEAF CLUSTER, because it is standing in for a whole
		# crown: a five-point fan with a leaf's raggedness reads as a cream STAR floating over
		# the ridge, which is what the far half of every wood looked like. Same colour as the
		# near trees' foliage now, too - it used to take the global leaf tint while the near
		# trees took the stand's.
		var cen: Vector3 = model["crown"]
		for entry in puffs:
			var pf: Dictionary = entry
			var pp: Vector3 = pf["p"]
			var wp := _pose(pp, base, s, cy, sy, bend, inv_h)
			# The same crown shading the near trees get, so a tree does not change how it is lit
			# when it crosses a LOD band.
			var aw := pp - cen
			var axx := aw.x * cy - aw.z * sy
			var azz := aw.x * sy + aw.z * cy
			var fac := clampf((axx * ldir.x + aw.y * ldir.y + azz * ldir.z)
				/ maxf(0.0001, aw.length()) * 0.5 + 0.5, 0.0, 1.0)
			var pcol := Color(col.r * (0.66 + 0.58 * fac), col.g * (0.66 + 0.58 * fac),
				col.b * (0.66 + 0.58 * fac), col.a)
			# THE FAR HALF OF THE WOOD IS MOST OF IT, so a soft puff here undoes the coarse near
			# foliage: at 0.4 of the stand's hardness the ridge was still cotton wool while the
			# trees in front of it had grain. A distant crown is hazier, not fluffier - the haze
			# is the fog term's job, and this keeps its edge.
			Branch3D.billboard_fan(faces, lens, u, wp, _cam_r, _cam_u,
				float(pf["r"]) * s * open, pcol, 8, 0.42, float(pf["ragged"]),
				0.82 * float(st.get("hard", 0.6)))
