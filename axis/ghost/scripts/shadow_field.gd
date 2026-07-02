extends RefCounted
class_name ShadowField

## A cheap CPU SHADOW MAP in light space - the honest version of "cast shadows" for the canvas 3D
## scenes. Occluders (buildings, spires) are rasterized into a grid on the plane perpendicular to the
## light; each cell keeps the occluder surface CLOSEST to the light. A surface point is then in shadow
## if something is nearer the light at its cell - so a WHOLE building casts a real volumetric shadow
## onto the ground AND onto other buildings (it layers on the blocks), instead of a flat footprint decal.
##
## Self-shadowing (a building darkening its own sunlit face) is avoided by biasing each building's own
## query by its own light-depth extent: moving a point along the light never changes its cell (the axis
## is perpendicular), only its depth, so a large-enough bias lifts it clear of its own body while a
## TALLER neighbour still occludes it.

const SHADOW_MIN := 0.5          # brightness of a fully cast-shadowed surface (0 = black, 1 = lit)

var _l := Vector3(0, 1, 0)       # unit direction TOWARD the light
var _lr := Vector3(1, 0, 0)      # light-space right / up (perpendicular to _l)
var _lu := Vector3(0, 0, 1)
var _res := 220
var _grid := PackedFloat32Array()
var _minx := 0.0
var _miny := 0.0
var _sx := 1.0
var _sy := 1.0
var _ready := false


## Set up the light frame + grid bounds from the scene's world AABB, and clear the map.
func build(light_dir: Vector3, wmin: Vector3, wmax: Vector3, res := 220) -> void:
	_l = light_dir.normalized()
	_lr = _l.cross(Vector3.UP)
	if _lr.length() < 1e-4:
		_lr = _l.cross(Vector3(1, 0, 0))
	_lr = _lr.normalized()
	_lu = _lr.cross(_l).normalized()
	var lo := Vector2(1e20, 1e20)
	var hi := Vector2(-1e20, -1e20)
	for i in 8:
		var c := Vector3(wmin.x if i & 1 == 0 else wmax.x, wmin.y if i & 2 == 0 else wmax.y,
			wmin.z if i & 4 == 0 else wmax.z)
		var lx := c.dot(_lr)
		var ly := c.dot(_lu)
		lo.x = minf(lo.x, lx); lo.y = minf(lo.y, ly)
		hi.x = maxf(hi.x, lx); hi.y = maxf(hi.y, ly)
	var pad := 0.5
	_minx = lo.x - pad
	_miny = lo.y - pad
	_res = res
	_sx = float(res) / maxf(0.001, (hi.x - lo.x) + 2.0 * pad)
	_sy = float(res) / maxf(0.001, (hi.y - lo.y) + 2.0 * pad)
	_grid = PackedFloat32Array()
	_grid.resize(res * res)
	_grid.fill(-1e20)
	_ready = true


## Rasterize an oriented box occluder (its light-space silhouette bounding rect, storing the box's
## surface nearest the light). Returns the box's light-DEPTH extent, which the caller passes back as the
## self-shadow bias when shading THAT box's own faces.
func add_box(base: Vector3, up: Vector3, bx: Vector3, bz: Vector3, w: float, h: float) -> float:
	if not _ready:
		return 0.0
	var top := base + up * h
	var corners := [
		base - bx * w - bz * w, base + bx * w - bz * w, base + bx * w + bz * w, base - bx * w + bz * w,
		top - bx * w - bz * w, top + bx * w - bz * w, top + bx * w + bz * w, top - bx * w + bz * w]
	var minlx := 1e20
	var maxlx := -1e20
	var minly := 1e20
	var maxly := -1e20
	var maxld := -1e20
	var minld := 1e20
	for c in corners:
		var lx: float = c.dot(_lr)
		var ly: float = c.dot(_lu)
		var ld: float = c.dot(_l)
		minlx = minf(minlx, lx); maxlx = maxf(maxlx, lx)
		minly = minf(minly, ly); maxly = maxf(maxly, ly)
		maxld = maxf(maxld, ld); minld = minf(minld, ld)
	var cx0 := clampi(int((minlx - _minx) * _sx), 0, _res - 1)
	var cx1 := clampi(int((maxlx - _minx) * _sx), 0, _res - 1)
	var cy0 := clampi(int((minly - _miny) * _sy), 0, _res - 1)
	var cy1 := clampi(int((maxly - _miny) * _sy), 0, _res - 1)
	for cy in range(cy0, cy1 + 1):
		var row := cy * _res
		for cx in range(cx0, cx1 + 1):
			var idx := row + cx
			if maxld > _grid[idx]:
				_grid[idx] = maxld
	return maxld - minld


## The light-space DEPTH of a world point (higher = nearer the light). Lets a caller measure an
## occluder's own depth extent for its self-shadow bias.
func light_depth(p: Vector3) -> float:
	return p.dot(_l)


## Light factor at a world point: 1 = full sun, down to SHADOW_MIN if an occluder is nearer the light
## at this point's cell. `bias` lifts the point clear of its own occluder (pass the box's depth extent
## for its own faces; a small value for the ground).
func factor(p: Vector3, bias := 0.06) -> float:
	if not _ready:
		return 1.0
	var cx := int((p.dot(_lr) - _minx) * _sx)
	var cy := int((p.dot(_lu) - _miny) * _sy)
	if cx < 0 or cx >= _res or cy < 0 or cy >= _res:
		return 1.0
	var occ := _grid[cy * _res + cx]
	var t := clampf((occ - p.dot(_l) - bias) / 0.35, 0.0, 1.0)   # soft ramp into shadow
	return lerpf(1.0, SHADOW_MIN, t)
