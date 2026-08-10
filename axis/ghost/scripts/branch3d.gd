extends RefCounted
class_name Branch3D

## Branch3D - a bracketed L-system that emits segments in THREE dimensions, plus the
## tapered-tube loft that turns them into drawable faces.
##
## [Filament] already grows probabilistic bracketed structures - its `_build` recursion
## over (depth, ratio, spread, steps * ratio) IS an L-system, and a good one - but it is
## strictly two-dimensional: the whole vocabulary is Vector2 unit-fractions, so a tree
## could not be assembled from it as a parts list however many Filaments were stacked.
## This is that same known shape lifted into 3D, with the two rules a plant needs that a
## flat tendril does not:
##
##   MULTI-CHILD WHORLS. A node throws 2-4 children at once from one point, deterministically,
##   rather than flipping a coin per step to fork. That single change is the difference
##   between a vine and a crown.
##
##   A DIVERGENCE ROLL. Successive whorls are rotated about the parent's own axis by a fixed
##   angle. This is the reason a real branch system reads as a spiral staircase rather than
##   as a flat fan, and it is why [constant DIVERGENCE] holds the four angles botany actually
##   uses instead of a continuous range - sampling 137.5 vs 90 degrees produces two SPECIES,
##   while sampling 131 vs 134 produces the same tree twice.
##
## WHAT THE RULE TABLE DECIDES. [method sample_species] rolls one dictionary that is the
## whole genome: fork angle and how it tightens with depth, the divergence roll, the length
## and width ratios between parent and child, how many children, how much clear trunk sits
## below the crown, a leader fraction (how nearly straight the CONTINUING axis runs, which
## is what gives a tree a trunk at all instead of a broccoli head), gravitropism toward
## vertical, phototropism toward the sun, droop on the outer orders, buttress flare at the
## base, and trunk lean.
##
## THE BUDGET IS PART OF THE ALGORITHM, not a safety valve bolted on. Axis count is
## children^depth, so a 4-child 6-deep tree is 5461 axes before a single face is drawn -
## and sixty of those is not a scene, it is a hang. So growth proceeds LEVEL BY LEVEL and a
## level is accepted only if it fits whole: a partially-expanded level looks broken (half the
## crown missing on one side), while a level short is simply a younger tree. [method
## sample_species] also trades depth against width for the same reason, because the two
## together are what explodes.
##
## COORDINATES. Everything grows in LOCAL space with the base at the origin and +y up, and
## the finished model is rescaled so its real height is exactly the height asked for (the
## crown adds length above the trunk axis, so the raw structure always overshoots). A caller
## instances one model many times by scaling, yawing and BENDING it at draw time - the furry
## trick, grow once in local space and re-lean per frame, lifted into 3D.
##
## The three emitters below all append the same face dictionaries the terrain merge path
## uses ({d, poly, cols}, or {d, fan, fcols} for a soft billboard), so tree faces depth-sort
## against terrain quads in one list and the hillside occludes what stands behind it.

const UP := Vector3(0, 1, 0)

## The divergence angles between successive whorls that plants actually use. 137.5 degrees is
## the golden angle (spiral phyllotaxis, far and away the commonest); 90 is decussate (maple,
## ash); 180 distichous (elm, hazel); 120 the three-ranked whorl of many conifers. Four
## discrete choices rather than a range, because the angle is a species trait, not a tolerance.
const DIVERGENCE := [137.5, 90.0, 180.0, 120.0]


## Roll one species - the per-depth rule table, and every tunable of it sampled.
##
## Depth and children are NOT independent draws. Axis count is children^depth and the level
## budget would simply discard the outer orders of a 4-child 6-deep tree, so a wide species
## is capped shallow and only the narrow ones are allowed to reach - which is also true of
## real crowns, where a tree that forks four ways does it fewer times.
static func sample_species(rng: RandomNumberGenerator) -> Dictionary:
	var children := rng.randi_range(2, 4)
	var depth := rng.randi_range(3, 6)
	if children >= 4:
		depth = mini(depth, 4)
	elif children == 3:
		depth = mini(depth, 5)
	return {
		"depth": depth,
		"children": children,
		# The fork angle off the parent axis. Narrow reads as a poplar, wide as an old oak.
		"angle": rng.randf_range(18.0, 55.0),
		# How the fork angle scales per order - below 1 the twigs close up toward the light,
		# above 1 they splay out at the ends.
		"angle_falloff": rng.randf_range(0.78, 1.12),
		"roll": DIVERGENCE[rng.randi() % DIVERGENCE.size()],
		"ratio": rng.randf_range(0.55, 0.82),
		# The LEADER: child 0 of every whorl continues the parent's axis at this fraction of
		# the fork angle and a little longer than its siblings. Without it there is no trunk.
		"leader": rng.randf_range(0.10, 0.45),
		"leader_ratio": rng.randf_range(1.0, 1.35),
		"taper": rng.randf_range(0.62, 0.78),
		"internodes": rng.randi_range(2, 4),
		"crown": rng.randf_range(0.25, 0.70),
		# Gravitropism: how hard each internode re-aims toward vertical. Slightly negative is
		# allowed - a weeping form that gives up on the sky.
		"tropism": rng.randf_range(-0.05, 0.34),
		"photo": rng.randf_range(0.0, 0.22),
		"droop": rng.randf_range(0.0, 0.34),
		"lean": rng.randf_range(0.0, 0.16),
		"flare": rng.randf_range(1.15, 1.95),
		"jitter": rng.randf_range(0.05, 0.22),
		"leaf_size": rng.randf_range(0.030, 0.085),
		"leaf_ragged": rng.randf_range(0.12, 0.38),
		# How readily the TRUNK throws side branches between whorls, so the crown is not one
		# ring of limbs stuck on the top of a pole.
		"lateral": rng.randf_range(0.35, 1.0),
	}


## Grow one model. Returns
## `{segs, leaves, puffs, level_end, height, radius, depth, crown, crown_r}` in local space
## (base at the origin, +y up, real height == [param height]).
##
## Each segment is `{a, b, w0, w1, depth, born0, born}`. `born0`/`born` are the segment's
## start and end as a fraction of the longest path from the base, so a caller reveals the
## structure with one scalar and the growth front sweeps outward from the trunk exactly the
## way a season fills a tree in - and a segment straddling the front can be shortened rather
## than popping.
static func grow(sp: Dictionary, rng: RandomNumberGenerator, height: float,
		radius: float, sun: Vector3, budget := 200) -> Dictionary:
	var depth_max: int = int(sp["depth"])
	var children: int = int(sp["children"])
	var ratio: float = float(sp["ratio"])
	var taper: float = float(sp["taper"])
	var base_angle: float = deg_to_rad(float(sp["angle"]))
	var falloff: float = float(sp["angle_falloff"])
	var roll_step: float = deg_to_rad(float(sp["roll"]))
	var tropism: float = float(sp["tropism"])
	var photo: float = float(sp["photo"])
	var droop: float = float(sp["droop"])
	var jitter: float = float(sp["jitter"])
	var lead_frac: float = float(sp["leader"])
	var lead_ratio: float = float(sp["leader_ratio"])
	var crown: float = float(sp["crown"])
	var flare: float = float(sp["flare"])
	var internodes: int = int(sp["internodes"])
	var lateral: float = float(sp["lateral"])
	var sun_dir := sun.normalized() if sun.length() > 1e-6 else UP
	# The heading every internode is pulled toward: vertical, biased toward the sun.
	var want := UP.lerp(sun_dir, clampf(photo, 0.0, 1.0)).normalized()

	var segs: Array = []
	# The trunk axis is only the CLEAR part of the tree; the crown is built above it by the
	# leader, so the axis is shortened by the crown fraction and the rescale at the end puts
	# the finished silhouette back at the requested height.
	var level: Array = [{
		"p": Vector3.ZERO, "dir": _jog(UP, float(sp["lean"]), rng),
		"len": height * (1.0 - crown * 0.55), "w": radius,
		"roll": rng.randf() * TAU, "path": 0.0}]
	var d := 0
	var deepest := 0
	while d <= depth_max and not level.is_empty():
		var next: Array = []
		for entry in level:
			var ax: Dictionary = entry
			# Finer wood is drawn with fewer internodes - it curves less and costs less.
			var n := maxi(1, internodes - int(d / 2))
			var alen: float = float(ax["len"])
			var aw: float = float(ax["w"])
			var pos: Vector3 = ax["p"]
			var dir: Vector3 = ax["dir"]
			var path: float = float(ax["path"])
			var roll: float = float(ax["roll"])
			var seg_len := alen / float(n)
			# Tropism is spent over the whole axis, so a 2-internode twig and a 4-internode
			# trunk bend by the same total amount rather than by the same amount per step.
			var pull := clampf(tropism - droop * float(d) / float(maxi(1, depth_max)),
				-0.5, 0.8) / float(n)
			for i in n:
				dir = _jog(dir.lerp(want, pull).normalized(), jitter / float(n), rng)
				var t0 := float(i) / float(n)
				var t1 := float(i + 1) / float(n)
				# The buttress: only the trunk flares, and only near the ground (cubed, so the
				# swelling is confined to the base rather than fattening the whole shaft).
				var fl0 := 1.0
				var fl1 := 1.0
				if d == 0:
					fl0 = 1.0 + (flare - 1.0) * pow(1.0 - t0, 3.0)
					fl1 = 1.0 + (flare - 1.0) * pow(1.0 - t1, 3.0)
				var a: Vector3 = pos
				var b: Vector3 = pos + dir * seg_len
				var p0 := path
				path += seg_len
				segs.append({"a": a, "b": b,
					"w0": aw * lerpf(1.0, taper, t0) * fl0,
					"w1": aw * lerpf(1.0, taper, t1) * fl1,
					"depth": d, "path0": p0, "path": path})
				pos = b
				# A side branch off the trunk between whorls (the rooted_growth lateral, in 3D).
				if d == 0 and i < n - 1 and t1 > crown * 0.6 and rng.randf() < lateral:
					next.append(_child_at(pos, dir,
						roll + roll_step * float(i + 1) + rng.randf_range(-jitter, jitter),
						base_angle * rng.randf_range(0.85, 1.30),
						alen * ratio * rng.randf_range(0.6, 0.9),
						aw * lerpf(1.0, taper, t1) * rng.randf_range(0.45, 0.68),
						roll + roll_step, path))
			deepest = maxi(deepest, d)
			if d >= depth_max:
				continue
			var tip_w := aw * taper
			var ang := base_angle * pow(falloff, float(d))
			for k in children:
				var phi := roll + TAU * float(k) / float(children) \
					+ rng.randf_range(-jitter, jitter)
				var theta := ang * rng.randf_range(0.8, 1.2)
				var clen := alen * ratio * rng.randf_range(0.85, 1.15)
				var cw := tip_w * rng.randf_range(0.5, 0.72)
				if k == 0:
					theta *= lead_frac                 # the leader continues the axis
					clen *= lead_ratio
					cw = tip_w * 0.92
				next.append(_child_at(pos, dir, phi, theta, clen, cw, roll + roll_step, path))
		# THE LEVEL BUDGET. Accept the next order only if it fits whole - a level short is a
		# younger tree, a level half-expanded is a broken one.
		var est := next.size() * maxi(1, internodes - int((d + 1) / 2))
		if segs.size() + est > budget:
			next = []
		level = next
		d += 1

	# Normalize the growth front and put the silhouette back at the requested height. Widths
	# are deliberately NOT rescaled: the caller passes a radius as a fraction of the final
	# height and gets exactly that, instead of a thickness that drifts with how tall the
	# structure happened to grow.
	var pmax := 0.001
	var ymax := 0.001
	for entry in segs:
		var sg: Dictionary = entry
		pmax = maxf(pmax, float(sg["path"]))
		var av: Vector3 = sg["a"]
		var bv: Vector3 = sg["b"]
		ymax = maxf(ymax, maxf(av.y, bv.y))
	var k := height / ymax
	for entry in segs:
		var sg: Dictionary = entry
		sg["a"] = (sg["a"] as Vector3) * k
		sg["b"] = (sg["b"] as Vector3) * k
		sg["born0"] = clampf(float(sg["path0"]) / pmax, 0.0, 1.0)
		sg["born"] = clampf(float(sg["path"]) / pmax, 0.0, 1.0)

	# LEVEL INDEX. Segments are appended level by level, so the array is already ordered by
	# depth and one integer per order is enough for a caller to stop early: `level_end[k]` is
	# the index past the last segment of order k. That is what makes the silhouette LOD free -
	# a distant tree walks four trunk segments instead of skipping two hundred it will not draw,
	# and at a hundred trees a frame the skipping was the cost, not the drawing.
	var level_end := PackedInt32Array()
	level_end.resize(deepest + 1)
	for si in segs.size():
		var sg: Dictionary = segs[si]
		level_end[int(sg["depth"])] = si + 1
	for lk in range(1, level_end.size()):
		level_end[lk] = maxi(level_end[lk], level_end[lk - 1])

	# LEAF SITES: the outer two orders only, strided down to a cap. Every twig carrying its
	# own cluster is both unaffordable and wrong - real foliage is held in masses.
	var leaves: Array = []
	var outer: Array = []
	var leaf_order := maxi(0, deepest - 1)
	for entry in segs:
		var sg: Dictionary = entry
		if int(sg["depth"]) >= leaf_order:
			outer.append(sg)
	var cap := 46
	var stride := maxi(1, int(ceil(float(outer.size()) / float(cap))))
	var lsize: float = float(sp["leaf_size"]) * height
	# `along` is renormalized over the leaves' OWN range of born-fractions. Raw, they all sit
	# near 1.0 (they are the tips, by construction), so a base-to-tip hue turn keyed to it
	# would be a constant offset rather than a gradient across the crown.
	var blo := 1.0
	var bhi := 0.0
	for entry in outer:
		var sg: Dictionary = entry
		blo = minf(blo, float(sg["born"]))
		bhi = maxf(bhi, float(sg["born"]))
	var bspan := maxf(1e-3, bhi - blo)
	for oi in range(0, outer.size(), stride):
		var sg: Dictionary = outer[oi]
		var av: Vector3 = sg["a"]
		var bv: Vector3 = sg["b"]
		var tw := bv - av
		tw = tw.normalized() if tw.length() > 1e-6 else UP
		leaves.append({
			"p": bv, "dir": tw,
			"size": lsize * rng.randf_range(0.7, 1.35),
			"phase": rng.randf() * TAU,
			"hue": rng.randf_range(-0.035, 0.035),
			"ragged": rng.randf() * TAU,
			"along": clampf((float(sg["born"]) - blo) / bspan, 0.0, 1.0),
			"born": float(sg["born"])})

	# The SILHOUETTE form: a handful of crown puffs a distant tree collapses to. Measured off
	# the leaf mass rather than invented, so the far tree occupies the same space the near one
	# would and a push-in does not make the wood change shape.
	var cen := Vector3.ZERO
	var crad := height * 0.22
	if not leaves.is_empty():
		for entry in leaves:
			var lf: Dictionary = entry
			cen += lf["p"] as Vector3
		cen /= float(leaves.size())
		crad = 0.001
		for entry in leaves:
			var lf: Dictionary = entry
			crad = maxf(crad, ((lf["p"] as Vector3) - cen).length())
	else:
		cen = Vector3(0.0, height * 0.72, 0.0)
	var puffs: Array = []
	for _pi in rng.randi_range(3, 6):
		puffs.append({
			"p": cen + Vector3(rng.randf_range(-1.0, 1.0), rng.randf_range(-0.7, 0.7),
				rng.randf_range(-1.0, 1.0)) * crad * 0.55,
			"r": crad * rng.randf_range(0.45, 0.80),
			"ragged": rng.randf() * TAU})
	return {"segs": segs, "leaves": leaves, "puffs": puffs, "height": height,
		"radius": radius, "depth": deepest, "crown": cen, "crown_r": crad,
		"level_end": level_end}


# One child axis: a heading built from the parent's own frame - `phi` around the parent axis
# (the divergence roll), `theta` away from it (the fork angle).
static func _child_at(pos: Vector3, dir: Vector3, phi: float, theta: float,
		clen: float, cw: float, roll: float, path: float) -> Dictionary:
	var fx := dir.cross(UP)
	if fx.length() < 1e-4:
		fx = dir.cross(Vector3(1, 0, 0))
	fx = fx.normalized()
	var fz := fx.cross(dir).normalized()
	var cd := (dir * cos(theta) + (fx * cos(phi) + fz * sin(phi)) * sin(theta)).normalized()
	return {"p": pos, "dir": cd, "len": clen, "w": cw, "roll": roll, "path": path}


# A small random deflection of a heading. Every node in a plant is a little off its ideal
# angle, and without this the whole structure reads as machined.
static func _jog(v: Vector3, amount: float, rng: RandomNumberGenerator) -> Vector3:
	var n := v.normalized() if v.length() > 1e-6 else UP
	if amount <= 0.0:
		return n
	var r := Vector3(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0),
		rng.randf_range(-1.0, 1.0))
	var out := n + r * amount
	return out.normalized() if out.length() > 1e-6 else n


## The camera's screen-aligned right / up axes in world space, for billboards. Returns
## `[right, up]`. [Lens3D] caches this internally but does not expose it, and a scene that
## billboards thousands of things should not recompute it per item.
static func cam_axes(lens: Lens3D) -> Array:
	var fwd := lens.look - lens.eye
	fwd = fwd.normalized() if fwd.length() > 1e-6 else Vector3(0, 0, -1)
	var right := fwd.cross(lens.up)
	right = right.normalized() if right.length() > 1e-6 else Vector3(1, 0, 0)
	return [right, right.cross(fwd).normalized()]


## THE TAPERED-TUBE LOFT, generalised out of spires' private one (which lofted SIDES x LEVELS
## rings for a whole tower and could not be reused for anything else). One segment becomes a
## closed `sides`-gon tube from radius [param w0] to [param w1], back-face culled - which
## halves the face count for free, and is safe here precisely because the tube IS closed,
## unlike a hollow spire shaft where the far wall has to fill the near one in.
##
## Faces append as `{d, poly, cols}`, the same shape [method Terrain.collect_surface] returns,
## so a caller merges wood and land into ONE depth-sorted list and the hill occludes the trunk
## behind it. Shading is the key light's n.l over an [param ambient] floor, plus [param rim] -
## a term that peaks where the surface turns away from the eye, which is the only channel the
## audio glow drives on bark (sound moves the light, never the geometry).
static func loft_tube(faces: Array, lens: Lens3D, u: float, a: Vector3, b: Vector3,
		w0: float, w1: float, sides: int, col: Color, ldir: Vector3,
		ambient := 0.42, rim := 0.0) -> void:
	var axis := b - a
	var alen := axis.length()
	if alen < 1e-6 or sides < 3:
		return
	var dir := axis / alen
	var fx := dir.cross(UP)
	if fx.length() < 1e-4:
		fx = dir.cross(Vector3(1, 0, 0))
	fx = fx.normalized()
	var fz := fx.cross(dir).normalized()
	var vdir := (a + b) * 0.5 - lens.eye
	if vdir.length_squared() < 1e-12:
		return
	vdir = vdir.normalized()
	var offs: Array = []
	for s in sides:
		var ang := TAU * float(s) / float(sides)
		offs.append(fx * cos(ang) + fz * sin(ang))
	for s in sides:
		var o0: Vector3 = offs[s]
		var o1: Vector3 = offs[(s + 1) % sides]
		var nrm := (o0 + o1).normalized()
		if nrm.dot(vdir) > 0.02:
			continue                                  # facing away - the tube is closed
		var r0 := lens.project(a + o0 * w0)
		var r1 := lens.project(a + o1 * w0)
		var r2 := lens.project(b + o1 * w1)
		var r3 := lens.project(b + o0 * w1)
		if r0.z <= lens.near or r1.z <= lens.near or r2.z <= lens.near or r3.z <= lens.near:
			continue
		var poly := PackedVector2Array([Vector2(r0.x, r0.y) * u, Vector2(r1.x, r1.y) * u,
			Vector2(r2.x, r2.y) * u, Vector2(r3.x, r3.y) * u])
		if Terrain._quad_area(poly) < 0.30:
			continue
		var sh := ambient + (1.0 - ambient) * clampf(nrm.dot(ldir), 0.0, 1.0)
		sh += rim * pow(clampf(1.0 - absf(nrm.dot(vdir)), 0.0, 1.0), 3.0)
		var lo := Color(col.r * sh, col.g * sh, col.b * sh, col.a)
		var k := sh * 1.07
		var hi := Color(col.r * k, col.g * k, col.b * k, col.a)
		faces.append({"d": (r0.z + r1.z + r2.z + r3.z) * 0.25, "poly": poly,
			"cols": PackedColorArray([lo, lo, hi, hi])})


## The LOD form of the same segment: one camera-facing tapered ribbon instead of a tube.
##
## Built in SCREEN space on purpose. A twig a few millimetres thick at eight world units
## projects to well under a pixel, and a sub-pixel quad is either discarded by the area guard
## or rasterized as nothing - so a lofted twig simply vanishes and the crown goes bald at
## exactly the distance where it matters most. Widening to a floor of [param min_px] after
## projection keeps every twig a visible hairline, which is what a real branch does to the eye.
static func loft_ribbon(faces: Array, lens: Lens3D, u: float, a: Vector3, b: Vector3,
		w0: float, w1: float, ca: Color, cb: Color, min_px := 0.55) -> void:
	var axis := b - a
	if axis.length_squared() < 1e-12:
		return
	var view := (a + b) * 0.5 - lens.eye
	var side := axis.cross(view)
	if side.length_squared() < 1e-14:
		return
	side = side.normalized()
	var pa := lens.project(a)
	var pb := lens.project(b)
	if pa.z <= lens.near or pb.z <= lens.near:
		return
	var sa := lens.project(a + side)
	var sb := lens.project(b + side)
	var oa := Vector2(pa.x, pa.y) * u
	var ob := Vector2(pb.x, pb.y) * u
	var ea := (Vector2(sa.x, sa.y) * u - oa) * w0
	var eb := (Vector2(sb.x, sb.y) * u - ob) * w1
	var la := ea.length()
	var lb := eb.length()
	if la < min_px:
		ea = (ea / la) * min_px if la > 1e-6 else Vector2(min_px, 0.0)
	if lb < min_px:
		eb = (eb / lb) * min_px if lb > 1e-6 else Vector2(min_px, 0.0)
	# A point just past the near plane projects to coordinates in the millions, and a vertex
	# buffer full of those is at best wasted triangles and at worst non-finite. The near-plane
	# test above catches most of it; this catches the rest.
	if not (is_finite(oa.x) and is_finite(oa.y) and is_finite(ob.x) and is_finite(ob.y)):
		return
	if maxf(absf(oa.x), absf(oa.y)) > 1.0e6 or maxf(absf(ob.x), absf(ob.y)) > 1.0e6:
		return
	var poly := PackedVector2Array([oa - ea, oa + ea, ob + eb, ob - eb])
	faces.append({"d": (pa.z + pb.z) * 0.5, "poly": poly,
		"cols": PackedColorArray([ca, ca, cb, cb])})


## A soft billboard cluster - one leaf mass, or one crown puff of a distant tree.
##
## A fan with an opaque core and a fully transparent rim, its radius modulated per vertex so
## the silhouette is ragged rather than a disc. This is deliberately NOT a textured atlas
## quad: leaves interleave with terrain quads in the merged depth sort, and a second texture
## in that list cuts the batch at every crossing between land and foliage - the exact failure
## the water sheet cost 894 batch cuts a frame to prove. A fan is five triangles in the run
## everything else is already in, and the soft rim is the part of a leaf that reads anyway.
##
## Faces append as `{d, fan, fcols}`; the caller draws them with [method TriBatch.tri_colored].
static func billboard_fan(faces: Array, lens: Lens3D, u: float, c: Vector3,
		right: Vector3, up: Vector3, r: float, core: Color, sides := 5,
		ragged := 0.25, phase := 0.0) -> void:
	if r <= 0.0 or core.a <= 0.004:
		return
	var pc := lens.project(c)
	if pc.z <= lens.near:
		return
	var pr := lens.project(c + right * r)
	var pu := lens.project(c + up * r)
	var o := Vector2(pc.x, pc.y) * u
	var ex := Vector2(pr.x, pr.y) * u - o
	var ey := Vector2(pu.x, pu.y) * u - o
	if ex.length() < 0.8 and ey.length() < 0.8:
		return                                        # sub-pixel: not worth the triangles
	var pts := PackedVector2Array()
	var cols := PackedColorArray()
	pts.append(o)
	cols.append(core)
	var edge := Color(core.r, core.g, core.b, 0.0)
	for s in sides + 1:
		var ang := TAU * float(s) / float(sides) + phase
		var rr := 1.0 + ragged * sin(3.0 * ang + phase * 2.0)
		pts.append(o + ex * (cos(ang) * rr) + ey * (sin(ang) * rr))
		cols.append(edge)
	faces.append({"d": pc.z, "fan": pts, "fcols": cols})
