extends Scene3D

## Terrain city - blocks rising as a city over real 3D terrain, growing nonlinearly.
##
## The metropolis idea on the [Terrain] foundation: a [Swarm] development field creeps
## across a landscape (rolling hills / mesa), and where it has grown, blocks stand on the
## surface - **PLUMB**, because real buildings are vertical whatever the ground does, and what keeps
## the field from being a rigid lattice is the STREET GRID rather than any lean (see _heading). Heights are driven
## by development x a per-block spectral band (nonlinear), so the skyline rises with the
## music. Some plots **detach**, their blocks floating a little off the ground. Camera
## orbits under a wide lens; the city grows over time from a few seeds.
##
## Land, layout and colour are all sampled per session rather than fixed. The terrain is any
## of four landforms under any climate that suits it; the LAYOUT decides whether the city
## reads as sparse dendritic arms, a broad sprawl, thin ribbons along the ridges or several
## separate towns; the SKYLINE law decides whether heights are even or a rare few spires
## tower over everything; and a [Scheme] mood colours the blocks, developed districts
## walking toward its accent.

## Which climates suit which landform - a mesa is not verdant, a canyon is not tundra.
const TERRAINS := {
	"hills":   ["temperate", "verdant", "tundra", "arid"],
	"mesa":    ["arid", "temperate"],
	"valleys": ["verdant", "temperate", "tundra"],
	"canyon":  ["arid", "temperate"],
}
## City layouts. The ridged "arm" field's frequency and how hard off-ridge plots are
## penalised decide the whole plan together, so they are sampled as a set: high frequency
## plus a hard penalty gives thin ribbons, low frequency plus a soft one gives sprawl.
const LAYOUTS := {
	"dendritic": {"arm": [1.6, 2.4], "penalty": 0.90, "band": [0.46, 0.64], "cores": [1, 2], "detach": 0.12},
	"sprawl":    {"arm": [0.9, 1.4], "penalty": 0.45, "band": [0.35, 0.60], "cores": [2, 3], "detach": 0.08},
	"ribbon":    {"arm": [2.6, 3.6], "penalty": 1.10, "band": [0.52, 0.70], "cores": [1, 1], "detach": 0.15},
	"towns":     {"arm": [2.0, 3.0], "penalty": 0.70, "band": [0.44, 0.62], "cores": [3, 4], "detach": 0.10},
}
## Skyline laws - how tall potential is DISTRIBUTED over the plots. A high exponent means
## almost everything is low with a rare tower; a low one means an even mid-rise mass.
const SKYLINES := {
	"spires":  {"exp": 4.6, "hbase": 0.7, "hspan": 3.4, "foot": 0.9, "sky": 0.06},
	"even":    {"exp": 1.4, "hbase": 1.1, "hspan": 1.2, "foot": 0.7, "sky": 0.02},
	"stepped": {"exp": 3.2, "hbase": 0.9, "hspan": 2.5, "foot": 1.0, "sky": 0.05},
	"slabs":   {"exp": 2.0, "hbase": 0.8, "hspan": 1.8, "foot": 1.5, "sky": 0.03},
}

var C := 30                      # city grid is C x C plots - density is sampled, not fixed

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _dev: Swarm
var _detach := PackedFloat32Array()   # per-plot float-off height (0 = grounded)
var _thresh := PackedFloat32Array()   # per-plot development level needed before a building rises
var _grown := PackedFloat32Array()    # per-plot BUILT height 0..1, eased up over time (starts small)
var _foot := PackedFloat32Array()     # per-plot footprint scale (skewed: many small, a few big anchors)
var _hclass := PackedFloat32Array()   # per-plot MAX height potential (reached only at critical mass)
var _sky := PackedFloat32Array()      # per-plot "tower from the start" propensity (rare, for variety)
var _phase := PackedFloat32Array()    # per-plot phase for the slow rearrange wobble
## Per-plot heading, in radians - which way the block's footprint faces. See the roll in
## [method build_params]: this is what stops the city being one rigid lattice, and it replaces a
## faint lean toward the terrain normal that used to do that job.
##
## THAT LEAN WAS A MISTAKE and it is worth writing down why, because it looked reasonable in the
## code. It was small - eight percent of the surface normal, five degrees at the very steepest -
## but the normal is a nearest-cell finite difference of a fractal heightfield, so it points a
## different way for every plot, and neighbouring towers leaned INDEPENDENTLY. A vertical edge is
## the most sensitive thing the eye has for reading tilt (it has the frame's own edges to judge
## against), and a hundred of them disagreeing by a few degrees does not read as "not a rigid
## grid" - it reads as a city that is falling over. Reported exactly that way.
##
## A heading does the same job honestly. Real cities are not laid out on one lattice either; they
## have a dominant grid, districts platted on their own alignment, and plots that sit a degree or
## two off true - and every one of those buildings is plumb.
var _heading := PackedFloat32Array()
var _maturity := 0.0                  # 0..1, rises over the scene: thresholds drop -> arms thicken, gaps fill
var _forge := FrameForge.new()        # off-thread frame builder (FrameForge contract)
var _cores: Array = []                # the 1-2 valley cells the city grows out from (re-pinned each frame)
var _light_az := 0.0
var _light_el := 0.5
var _light_dir := 1.0
var _hue := 0.0
var _hue_dev := 0.25      # hue walk toward the scheme accent per unit development
var _hue_elev := 0.12     # ... and per unit terrain height
var _sat := 0.45          # block saturation, from the mood
var _vmul := 1.0          # block value multiplier, from the mood
var _glow := 0.0
var _yaw := 0.0
var _dist := 7.5
var _pitch := 0.5
var _beat_prev := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var ttype := String(TERRAINS.keys()[rng.randi() % TERRAINS.size()])
	var climates: Array = TERRAINS[ttype]
	var climate := String(climates[rng.randi() % climates.size()])
	# Plot size is the city's grain: a coarse grid gives big blocks on a small town, a fine
	# one a dense field of smaller buildings over the same land.
	C = rng.randi_range(26, 34)
	_terrain = Terrain.new()
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.35, 0.55), null, climate)
	# Buildings take a [Scheme] mood - concrete-grey, brass, bone, or a lit-up violet or teal
	# night - rather than one arbitrary hue rotated off the terrain's.
	var sch := Scheme.among(["ash", "bone", "brass", "sodium", "dawn", "glacier",
		"abyss", "violet", "teal", "ember", "rose"], rng)
	_hue = sch.hue
	_sat = sch.sat * 0.65
	_vmul = sch.val * 1.15
	# Development walks the blocks toward the accent (the short way round), so the built-up
	# core reads as a different, related colour from the frontier instead of a fixed swing.
	var to_accent := fposmod(sch.accent - sch.hue + 0.5, 1.0) - 0.5
	_hue_dev = to_accent * rng.randf_range(0.35, 0.9)
	_hue_elev = to_accent * rng.randf_range(0.1, 0.4)
	var layout := String(LAYOUTS.keys()[rng.randi() % LAYOUTS.size()])
	var lay: Dictionary = LAYOUTS[layout]
	var skyline := String(SKYLINES.keys()[rng.randi() % SKYLINES.size()])
	var sky_law: Dictionary = SKYLINES[skyline]
	# The city grows from a few cores, each seeded in a LOW valley (easy ground): sample a
	# handful of central cells and keep the lowest. Development creeps outward from there; the cores
	# are re-pinned every frame so the origin never fades. How MANY cores is the layout's call -
	# one origin gives a single town, several give separate settlements that grow together.
	_dev = Swarm.new(C, C, Swarm.GROW, rng, 0)
	for k in rng.randi_range(int(lay.cores[0]), int(lay.cores[1])):
		var bx := C / 2
		var by := C / 2
		var blo := 1e9
		for _t in 10:
			var cx := rng.randi_range(int(C * 0.28), int(C * 0.72))
			var cy := rng.randi_range(int(C * 0.28), int(C * 0.72))
			var hh: float = _terrain.height_at(_plot_wx(cx), _plot_wz(cy))
			if hh < blo:
				blo = hh
				bx = cx
				by = cy
		_cores.append(Vector2i(bx, by))
		_dev.inject(bx, by, 1.0)
	# Per-plot BUILD THRESHOLD, biased by TERRAIN HEIGHT: valleys (low, easy ground) need only a
	# little development to build, so they fill FIRST; hillsides need more; peaks stay bare. Plus a
	# little noise so the frontier is ragged, not a clean contour.
	_thresh.resize(C * C)
	_grown.resize(C * C)
	_foot.resize(C * C)
	_hclass.resize(C * C)
	_sky.resize(C * C)
	_phase.resize(C * C)
	_heading.resize(C * C)
	# THE STREET GRID. One dominant alignment for the city, a second one for a district platted on
	# its own (which nearly every real city has, and which is the thing that stops a grid reading as
	# wallpaper), and a couple of degrees of per-plot slop on top. A square footprint repeats every
	# quarter turn, so the second grid is offset by well inside one - anything more just lands back
	# on the first.
	var grid_a := rng.randf() * TAU
	var grid_b := grid_a + rng.randf_range(0.30, 1.10)
	var slop := rng.randf_range(0.012, 0.045)
	# The district is a SMOOTH function of position, not a per-plot coin toss: two low harmonics
	# across the grid, so the second alignment comes out as a contiguous quarter of the city rather
	# than as salt and pepper through it.
	var dk := Vector2(rng.randf_range(1.2, 3.0), rng.randf_range(1.2, 3.0))
	var dp := Vector2(rng.randf() * TAU, rng.randf() * TAU)
	var dthr := rng.randf_range(0.35, 1.15)
	# A ridged "arm" field: its branching high ridges become the channels the city builds ALONG, so
	# development reads as dendritic ARMS reaching out from the core rather than a filled blob.
	var armf := Field.make("ridged", rng.randi(), rng.randf_range(float(lay.arm[0]), float(lay.arm[1])), 3)
	var band0: float = float(lay.band[0])
	var band1: float = float(lay.band[1])
	var penalty: float = float(lay.penalty)
	var hexp: float = float(sky_law.exp)
	for cy in C:
		for cx in C:
			var i := cy * C + cx
			var p := Vector2(float(cx) / float(C - 1) - 0.5, float(cy) / float(C - 1) - 0.5) * 2.0
			var elev: float = clampf(_terrain.height_at(_plot_wx(cx), _plot_wz(cy)), 0.0, 1.0)
			var arm: float = armf.at(p)
			# Threshold = valley bias + an OFF-ARM penalty. On an arm ridge in a valley: builds first.
			# Between the arms / up the hills: needs far more development (fills only as the city matures).
			# Valley bias + a GENTLE off-arm bias: the arm ridges build a bit sooner so the frontier
			# reaches out in dendritic arms, but off-arm plots still fill in as the base matures (this
			# is a preference, not a hard gate - a hard gate starved a rugged map of any city at all).
			# Threshold is dominated by the ridged ARM field: only cells ON a branching ridge of that
			# field can build (a strong OFF-ridge penalty keeps the rest bare whatever the development),
			# so the city is a sparse DENDRITIC network of arms rather than a solid blob. Elevation adds
			# a gentle bias (lower ground a touch likelier); noise ragged-ifies the frontier.
			_thresh[i] = 0.05 + 0.2 * elev + (1.0 - smoothstep(band0, band1, arm)) * penalty + rng.randf_range(-0.03, 0.05)
			_grown[i] = 0.0
			# Footprint + MAX-height potential, distributed by the SKYLINE law: a steep exponent
			# leaves most plots modest with a rare big anchor, a shallow one gives an even mid-rise
			# mass. Either way the height is only REALISED once the district hits critical mass (see
			# the draw): blocks start SMALL and grow taller as their surroundings develop.
			var big := pow(rng.randf(), hexp)
			_foot[i] = 0.5 + float(sky_law.foot) * big
			_hclass[i] = float(sky_law.hbase) + float(sky_law.hspan) * big + rng.randf_range(-0.1, 0.3)
			# A rare few plots are skyscrapers FROM THE START (variety) - most are 0 (grow up naturally).
			_sky[i] = rng.randf_range(0.55, 1.0) if rng.randf() < float(sky_law.sky) else 0.0
			_phase[i] = rng.randf() * TAU
			var district := sin(p.x * dk.x + dp.x) + sin(p.y * dk.y + dp.y)
			_heading[i] = (grid_b if district > dthr else grid_a) + rng.randf_range(-slop, slop)
	# Per-plot detach: a few districts float off the ground (how many is the layout's).
	_detach.resize(C * C)
	for i in C * C:
		_detach[i] = rng.randf_range(0.10, 0.30) if rng.randf() < float(lay.detach) else 0.0
	lens.fov = rng.randf_range(56.0, 72.0)
	_dist = rng.randf_range(6.5, 8.5)
	_pitch = rng.randf_range(0.34, 0.55)
	_yaw = rng.randf() * TAU
	# A low key light so the mountains (and the skyline) cast long, gently sweeping shadows.
	_light_az = rng.randf() * TAU
	_light_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_el = rng.randf_range(0.34, 0.52)
	_terrain.set_light(_light_az, _light_el)
	return {"type": ttype, "climate": climate, "mood": sch.name,
		"layout": layout, "skyline": skyline, "grid": C}


func _plot_wx(cx: int) -> float:
	return (float(cx) / float(C - 1) - 0.5) * 2.0 * _terrain.half


func _plot_wz(cy: int) -> float:
	return (float(cy) / float(C - 1) - 0.5) * 2.0 * _terrain.half


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.012, 0.018)
	# BURST then decay: growth is FAST in the opening seconds and eases to a slow crawl after. A scene's
	# hold is drive-scaled and can be short (a few seconds on an energetic passage), so front-loading the
	# growth means even a brief scene BURSTS into a small city up front; a long one keeps creeping after.
	# `_life` is the scene's age (starts ~0.6 after the pre-warm, which already banks some burst growth).
	# A quick initial POP (a handful of blocks fast, so even a short scene isn't empty), decaying sharply
	# to a SLOW crawl - the city keeps developing gently for the rest of the scene, never filling all at
	# once. Fast decay (~1s) so the burst is a brief opener, not a fill.
	var burst: float = 1.0 + 3.5 * exp(-_life * 0.7)
	# Nonlinear growth drive: beats lunge development outward through a spike curve, bursting at the start.
	var drive := (1.0 + 1.1 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)) * (0.7 + 0.3 * burst)
	for core in _cores:
		_dev.inject(core.x, core.y, 1.0)          # keep the origin cores alive
	_dev.step(drive, delta, 0.015)
	# The city matures over the scene: the effective thresholds drop, so the arms THICKEN a little - but
	# kept modest so it stays SPARSE and dendritic (bands/arms reaching along the terrain), never a solid
	# filled blob. Slow BASE rate (the burst supplies the opening pop; the rest is a gentle creep).
	_maturity = minf(1.0, _maturity + delta * 0.05 * burst)
	# Ease each plot's BUILT height up toward its current maturity, so buildings START SMALL and grow
	# taller as their district matures - and the densest (most-developed) plots grow tallest.
	var rise := delta * 0.35 * burst
	for cy in C:
		for cx in C:
			var i := cy * C + cx
			var thr: float = maxf(0.04, float(_thresh[i]) - _maturity * 0.1)
			var target: float = clampf((_dev.at(cx, cy) - thr) / maxf(0.05, 1.0 - thr), 0.0, 1.0)
			_grown[i] = move_toward(float(_grown[i]), target, rise)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.5 * f.beat, 0.0, 1.0), 1.0 - exp(-5.0 * delta))
	_yaw += delta * (0.08 + 0.16 * f.energy)
	lens.orbit(Vector3(0.0, _terrain.relief * 0.25, 0.0), _dist, _yaw, _pitch + 0.04 * sin(_life * 0.13))
	# Drift the key light and refresh the terrain's sweeping cast shadows.
	_light_az += delta * 0.035 * _light_dir
	_terrain.set_light(_light_az, _light_el)
	_terrain.step_light(delta)
	# Snapshot the frame into a job for the worker (see FrameForge): swarm and
	# grown fields DUPLICATED (main keeps stepping them), immutable-after-build
	# plot arrays by reference, terrain by reference (its light writes are
	# benign in-place float updates), the lens copied (main re-orbits it).
	var job := CityJob.new()
	job.f = f
	job.c = C
	job.u = unit()
	job.life = _life
	job.glow = _glow
	job.hue = _hue
	job.hue_dev = _hue_dev
	job.hue_elev = _hue_elev
	job.sat = _sat
	job.vmul = _vmul
	job.maturity = _maturity
	job.reveal = view.reveal
	job.terrain = _terrain
	job.tex_rid = Terrain.detail_texture().get_rid()
	job.dev = _dev.f.duplicate()
	job.grown = _grown.duplicate()
	job.detach = _detach
	job.foot = _foot
	job.hclass = _hclass
	job.sky = _sky
	job.phase = _phase
	job.yaw = _heading
	job.lens = Lens3D.new()
	job.lens.eye = lens.eye
	job.lens.look = lens.look
	job.lens.up = lens.up
	job.lens.fov = lens.fov
	job.lens.near = lens.near
	_forge.kick(job.run, {}, self, job)   # retain: a Callable alone will not keep the job alive


func _draw() -> void:
	begin_draw()
	texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED
	_forge.submit(self)


## The whole frame off the main thread (the FrameForge job): shadow raster,
## building realization, terrain merge, painter sort, batch runs. Reads only
## its own members - a mid-job Director cut is harmless.
class CityJob:
	extends RefCounted

	var f: AudioFeatures
	var c := 30
	var u := 1.0
	var life := 0.0
	var glow := 0.0
	var hue := 0.0
	var hue_dev := 0.25
	var hue_elev := 0.12
	var sat := 0.45
	var vmul := 1.0
	var maturity := 0.0
	var reveal := 1.0
	var terrain: Terrain
	var tex_rid := RID()
	var lens: Lens3D
	var dev := PackedFloat32Array()
	var grown := PackedFloat32Array()
	var detach := PackedFloat32Array()
	var foot := PackedFloat32Array()
	var hclass := PackedFloat32Array()
	var sky := PackedFloat32Array()
	var phase := PackedFloat32Array()
	var yaw := PackedFloat32Array()

	func run(_s: Dictionary) -> Array:
		lens.prepare()
		var lit := clampf(0.7 + 0.4 * glow + 0.3 * f.energy, 0.4, 1.4)
		var bw := terrain.half / float(c) * 0.62          # block half-footprint (world)
		var bgain := 0.5 + 0.3 * f.energy
		# How deep each building is sunk INTO the terrain: the merged land hides the buried part,
		# so the visible base is ragged (cut by the ground), never a clean line.
		var embed: float = 0.35 * terrain.relief + 0.14
		# Pass A: compute every building and RASTERIZE it into the light-space shadow map - it must
		# finish before shading, since a building shadows the ground and other buildings.
		var shadow := ShadowField.new()
		shadow.build(terrain.light_dir(), Vector3(-terrain.half, -terrain.relief, -terrain.half),
			Vector3(terrain.half, terrain.relief + 3.0, terrain.half))
		var blds: Array = []
		for cy in c:
			for cx in c:
				if reveal < 0.02:                       # still fading the terrain in - no buildings yet
					break
				var i := cy * c + cx
				var grown_i: float = grown[i]
				if grown_i < 0.02:                      # nothing built here yet (a gap / bare peak)
					continue
				var wx := (float(cx) / float(c - 1) - 0.5) * 2.0 * terrain.half
				var wz := (float(cy) / float(c - 1) - 0.5) * 2.0 * terrain.half
				var ground := terrain.height_at(wx, wz) * terrain.relief
				var float_off: float = detach[i]
				var dv := dev[i]
				var react := f.sample(clampf(terrain.height_at(wx, wz) + 0.5, 0.0, 1.0))
				# A slow per-plot wobble keeps the built skyline REARRANGING over time.
				var wob := 0.85 + 0.28 * sin(life * 0.12 + float(phase[i]))
				# CRITICAL MASS: blocks are built small and only reach their tall potential as their
				# district densifies and the city matures; a rare few (`sky`) tower from the start.
				var crit := clampf(dv * lerpf(0.28, 1.0, maturity), 0.0, 1.0)
				var realize := clampf(maxf(Nonlinear.apply("spike", crit, 2.4), float(sky[i])), 0.0, 1.0)
				var tall := lerpf(0.5, float(hclass[i]), realize)
				var h := grown_i * tall * (0.42 + 0.4 * bgain + 0.5 * react) * wob
				var bw_i := bw * float(foot[i])
				# PLUMB, and turned on its own heading. The ground under a city does what it likes;
				# the buildings on it do not follow it (see the scene's _heading for what leaning toward
				# the terrain normal actually looked like).
				var up := Vector3.UP
				var ya: float = yaw[i]
				var cy_ := cos(ya)
				var sy_ := sin(ya)
				var bx := Vector3(cy_, 0.0, sy_)
				var bz := Vector3(-sy_, 0.0, cy_)
				# Sink the base BELOW the surface so its bottom is buried; the top stays where it was.
				var base := Vector3(wx, ground + float_off - embed, wz)
				var htot := h + embed
				var bhue := fposmod(hue + hue_elev * terrain.height_at(wx, wz) + hue_dev * dv, 1.0)
				# The TERRAIN also shadows the building (a block in a hill's cast shadow darkens).
				var tsh: float = terrain.shadow_at(wx, wz)
				var blit := clampf(0.18 + 0.5 * dv + 0.5 * react + 0.6 * glow, 0.05, 1.2) * lit * (0.35 + 0.65 * tsh)
				var ext := shadow.add_box(base, up, bx, bz, bw_i, htot)   # rasterize + self-shadow bias
				blds.append({"base": base, "up": up, "bx": bx, "bz": bz, "w": bw_i, "h": htot,
					"hue": bhue, "lit": blit, "ext": ext})

		# ONE merged list: terrain quads + building faces, depth-sorted together so the land occludes
		# the buried building bases and neighbours' shadows layer correctly.
		var faces: Array = terrain.collect_surface(lens, u, lit, life, shadow)
		for b in blds:
			_block_faces(faces, shadow, b.base, b.up, b.bx, b.bz, b.w, b.h, b.hue, b.lit, float(b.ext))
		faces = TriBatch.painter_sort(faces)
		var tb := TriBatch.new()
		for fc in faces:
			if fc.has("uvs"):
				tb.mark_run(true, tex_rid)
				tb.quad_textured(fc.poly, fc.cols, fc.uvs)
			else:
				tb.mark_run(false, RID())
				tb.quad_colored(fc.poly, fc.cols)
		return tb.take_chunks()

	# Append the camera-facing faces of one oriented box to `out` (each {poly, cols, d}), per-VERTEX
	# shaded by the key light AND the cast-shadow map.
	func _block_faces(out: Array, shadow: ShadowField, base: Vector3, up: Vector3, bx: Vector3,
			bz: Vector3, w: float, h: float, bhue: float, lit: float, ext: float) -> void:
		var top := base + up * h
		var corners := [
			base - bx * w - bz * w, base + bx * w - bz * w, base + bx * w + bz * w, base - bx * w + bz * w,
			top - bx * w - bz * w, top + bx * w - bz * w, top + bx * w + bz * w, top - bx * w + bz * w]
		var pr: Array = []
		for cwld in corners:
			var pj := lens.project(cwld)
			pr.append(Vector3(pj.x * u, pj.y * u, pj.z))
		var quads := [[4, 5, 6, 7, up],                     # top
			[0, 1, 5, 4, -bz], [1, 2, 6, 5, bx], [2, 3, 7, 6, bz], [3, 0, 4, 7, -bx]]
		for q in quads:
			var i0: int = q[0]
			var i1: int = q[1]
			var i2: int = q[2]
			var i3: int = q[3]
			var fn: Vector3 = q[4]
			var fc: Vector3 = (corners[i0] + corners[i1] + corners[i2] + corners[i3]) * 0.25
			if fn.dot(lens.eye - fc) <= 0.0:                 # facing away
				continue
			var p0: Vector3 = pr[i0]
			var p1: Vector3 = pr[i1]
			var p2: Vector3 = pr[i2]
			var p3: Vector3 = pr[i3]
			if p0.z <= lens.near or p1.z <= lens.near or p2.z <= lens.near or p3.z <= lens.near:
				continue
			var fpoly := PackedVector2Array([Vector2(p0.x, p0.y), Vector2(p1.x, p1.y),
				Vector2(p2.x, p2.y), Vector2(p3.x, p3.y)])
			if Terrain._quad_area(fpoly) < 1.0:        # edge-on face - skip
				continue
			var shade := 0.34 + 0.72 * clampf(fn.dot(terrain.light_dir()), 0.0, 1.0)
			var cols := PackedColorArray()
			for idx in [i0, i1, i2, i3]:
				var sf: float = shadow.factor(corners[idx], ext + 0.06)
				# Saturation and brightness carry the mood, so an ash city is grey concrete
				# and a violet one glows - the lighting maths is untouched either way.
				var cc := Color.from_hsv(bhue, sat, clampf(lit * shade * sf * vmul, 0.0, 1.0))
				# A plot below the waterline is IN the lake, not on it (see Terrain.submerged).
				cc = terrain.submerged(cc, corners[idx].y)
				cc.a = reveal                          # fade the buildings in AFTER the terrain
				cols.append(cc)
			out.append({"d": (p0.z + p1.z + p2.z + p3.z) * 0.25, "poly": fpoly, "cols": cols})
