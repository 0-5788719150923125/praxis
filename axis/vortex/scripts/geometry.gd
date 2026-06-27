extends RefCounted
class_name Geo

## Geo - small reusable polygon helpers (area, centroid, convex split, fracture).
## Shared geometry, not a scene's private code - the fracture that shatters glass
## can shatter anything.


static func area(poly: PackedVector2Array) -> float:
	var a := 0.0
	var n := poly.size()
	for i in n:
		var p := poly[i]
		var q := poly[(i + 1) % n]
		a += p.x * q.y - q.x * p.y
	return absf(a) * 0.5


static func centroid(poly: PackedVector2Array) -> Vector2:
	if poly.is_empty():
		return Vector2.ZERO
	var s := Vector2.ZERO
	for p in poly:
		s += p
	return s / float(poly.size())


## Split a convex polygon by the line through `pt` with normal `nrm`.
## Returns [front, back]; either side may have fewer than 3 points.
static func split(poly: PackedVector2Array, pt: Vector2, nrm: Vector2) -> Array:
	var front := PackedVector2Array()
	var back := PackedVector2Array()
	var n := poly.size()
	for i in n:
		var a := poly[i]
		var b := poly[(i + 1) % n]
		var da := nrm.dot(a - pt)
		var db := nrm.dot(b - pt)
		if da >= 0.0:
			front.append(a)
		else:
			back.append(a)
		if (da > 0.0) != (db > 0.0):
			var denom := da - db
			if absf(denom) > 0.00001:
				var ip := a.lerp(b, da / denom)
				front.append(ip)
				back.append(ip)
	return [front, back]


## Fracture a convex polygon into ~count convex shards by repeatedly splitting
## the largest shard along a random line biased toward `impact`, so the cracks
## read as radiating from a strike rather than a uniform grid.
static func fracture(base: PackedVector2Array, count: int, impact: Vector2, jitter: float, rng: RandomNumberGenerator) -> Array:
	var shards: Array = [base]
	var guard := 0
	while shards.size() < count and guard < count * 8:
		guard += 1
		var idx := 0
		var best := -1.0
		for i in shards.size():
			var ar: float = area(shards[i])
			if ar > best:
				best = ar
				idx = i
		var poly: PackedVector2Array = shards[idx]
		var c := centroid(poly)
		var pt := c.lerp(impact, rng.randf() * 0.4) + Vector2(
			rng.randf_range(-1, 1), rng.randf_range(-1, 1)) * jitter
		var ang := rng.randf() * PI
		var res := split(poly, pt, Vector2(cos(ang), sin(ang)))
		var fr: PackedVector2Array = res[0]
		var bk: PackedVector2Array = res[1]
		if fr.size() >= 3 and bk.size() >= 3:
			shards.remove_at(idx)
			shards.append(fr)
			shards.append(bk)
	return shards
