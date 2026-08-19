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
var _cov := PackedFloat32Array()      # per-cell silhouette COVERAGE, 0..1 (see add_box)
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
	# STABLE TEXELS - the fix for "the shadows aren't stable either". The bounds used to be the
	# world box PROJECTED into light space, so both the origin AND the scale changed every frame as
	# the light drifted (Terrain.set_light turns the azimuth continuously). Every texel therefore
	# landed on a new world position each frame and a stationary tree's shadow crawled and shimmered
	# along its own edges - textbook shadow swimming.
	#
	# The extent comes from the box's bounding SPHERE instead, which is the same number whatever
	# direction the light comes from, and the origin is snapped DOWN to a whole texel. The texel then
	# has a fixed size and a fixed phase in world space, so a turning light rotates the map instead of
	# re-quantizing it, and geometry that has not moved keeps the shadow it had.
	var pad := 0.5
	var ctr := (wmin + wmax) * 0.5
	var radius := (wmax - wmin).length() * 0.5 + pad
	_res = res
	_sx = float(res) / maxf(0.001, 2.0 * radius)
	_sy = _sx
	var texel := 1.0 / _sx
	_minx = floor((ctr.dot(_lr) - radius) / texel) * texel
	_miny = floor((ctr.dot(_lu) - radius) / texel) * texel
	_grid = PackedFloat32Array()
	_grid.resize(res * res)
	_grid.fill(-1e20)
	_cov = PackedFloat32Array()
	_cov.resize(res * res)
	_cov.fill(0.0)
	_ready = true


## Rasterize an oriented box occluder (its light-space silhouette bounding rect, storing the box's
## surface nearest the light) and its per-cell COVERAGE. Returns the box's light-DEPTH extent, which
## the caller passes back as the self-shadow bias when shading THAT box's own faces.
##
## COVERAGE IS THE ANTIALIASING, and it is the other half of "the shadows are blocky". The rect used
## to be snapped to whole cells - `int((minlx - _minx) * _sx)` - so an occluder's silhouette both
## STARTED and ENDED on a texel boundary. Two things follow from that and both were visible: the edge
## is a hard staircase however smoothly [method factor] then interpolates the verdicts, because there
## is no partial cell anywhere to interpolate; and as the light drifts the rect's bounds cross whole
## texels one at a time, so the whole patch STEPS instead of travelling. Storing the fraction of each
## border cell the silhouette actually covers puts the edge back at its true sub-texel position, and
## it moves continuously because the fraction does.
##
## [param round_] treats the silhouette as the inscribed ELLIPSE rather than the rect, feathering to
## nothing at its rim. A tree crown and a spire shaft are round; their bounding rect is a square, and
## a square shadow under a round thing is the blockiest object in the frame. Buildings stay square.
func add_box(base: Vector3, up: Vector3, bx: Vector3, bz: Vector3, w: float, h: float,
		round_ := false) -> float:
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
	# The silhouette in CELL units, kept fractional - the whole point of the coverage pass.
	var fx0 := (minlx - _minx) * _sx
	var fx1 := (maxlx - _minx) * _sx
	var fy0 := (minly - _miny) * _sy
	var fy1 := (maxly - _miny) * _sy
	var cx0 := clampi(int(floor(fx0)), 0, _res - 1)
	var cx1 := clampi(int(floor(fx1)), 0, _res - 1)
	var cy0 := clampi(int(floor(fy0)), 0, _res - 1)
	var cy1 := clampi(int(floor(fy1)), 0, _res - 1)
	var mx := (fx0 + fx1) * 0.5
	var my := (fy0 + fy1) * 0.5
	var rx := maxf(0.5 * (fx1 - fx0), 1e-4)
	var ry := maxf(0.5 * (fy1 - fy0), 1e-4)
	for cy in range(cy0, cy1 + 1):
		var row := cy * _res
		# Fraction of this cell's height the silhouette spans (1 in the interior, a fraction at the rim).
		var covy := clampf(minf(fy1, float(cy) + 1.0) - maxf(fy0, float(cy)), 0.0, 1.0)
		var dy := (float(cy) + 0.5 - my) / ry
		for cx in range(cx0, cx1 + 1):
			var cov := covy * clampf(minf(fx1, float(cx) + 1.0) - maxf(fx0, float(cx)), 0.0, 1.0)
			if round_:
				# Inscribed ellipse, feathered over its outer third so the rim is a gradient and not
				# a circle of hard cells - the same discipline as the rect edge above.
				var dx := (float(cx) + 0.5 - mx) / rx
				cov *= clampf((1.0 - sqrt(dx * dx + dy * dy)) * 3.0, 0.0, 1.0)
			if cov <= 0.0:
				continue
			var idx := row + cx
			if maxld > _grid[idx]:
				_grid[idx] = maxld
			# Coverage accumulates as a MAX over occluders rather than travelling with the depth that
			# won: a cell grazed by one tree and filled by another is filled. It can therefore only
			# ever soften what the old whole-cell rasterizer wrote, never darken it.
			if cov > _cov[idx]:
				_cov[idx] = cov
	return maxld - minld


## The light-space DEPTH of a world point (higher = nearer the light). Lets a caller measure an
## occluder's own depth extent for its self-shadow bias.
func light_depth(p: Vector3) -> float:
	return p.dot(_l)


## Light factor at a world point: 1 = full sun, down to SHADOW_MIN if an occluder is nearer the light
## at this point's cell. `bias` lifts the point clear of its own occluder (pass the box's depth extent
## for its own faces; a small value for the ground).
##
## BILINEAR, and that is half the fix for "very blocky" shadows. The lookup used to be one nearest
## -neighbour tap, so a shadow edge was a hard staircase on the cell grid: at canopy's world span
## (about 8.5 units across `_res` cells) one cell is roughly a dozen screen pixels at 1920x1080,
## which is exactly the size of step the wood's shadows were reported to have. `Terrain.shadow_at`
## - the terrain's own, older shadow map - already samples bilinearly for this reason; this is the
## same discipline applied to the prop map. The other half is [method add_box]'s coverage: filtering
## between four cells that are each fully lit or fully shadowed can only ever produce a one-cell
## ramp, and a one-cell ramp on a grid this size IS the staircase.
##
## The FOUR FACTORS are blended, not the four depths: neighbouring cells hold the depth of
## DIFFERENT occluders, and the average of two unrelated occluder depths is not a surface. Each
## cell's own lit/shadowed verdict is computed first and only those are mixed, which soft-edges the
## silhouette without inventing geometry between two trees.
func factor(p: Vector3, bias := 0.06) -> float:
	if not _ready:
		return 1.0
	# Cell CENTRES sit at +0.5, so subtracting it puts the sample point in the frame where the
	# integer part is the lower-left neighbour and the fraction is the blend weight.
	var fx := (p.dot(_lr) - _minx) * _sx - 0.5
	var fy := (p.dot(_lu) - _miny) * _sy - 0.5
	var x0 := floori(fx)
	var y0 := floori(fy)
	if x0 < -1 or x0 >= _res or y0 < -1 or y0 >= _res:
		return 1.0
	var ax := fx - float(x0)
	var ay := fy - float(y0)
	var x1 := mini(x0 + 1, _res - 1)
	var y1 := mini(y0 + 1, _res - 1)
	x0 = maxi(x0, 0)
	y0 = maxi(y0, 0)
	var d := p.dot(_l) + bias
	var r0 := y0 * _res
	var r1 := y1 * _res
	# Each cell's verdict is its depth test SCALED BY how much of it the silhouette covered, so a
	# grazed cell casts a partial shadow and the edge sits where the occluder's outline really is.
	var t00 := _cov[r0 + x0] * clampf((_grid[r0 + x0] - d) / 0.35, 0.0, 1.0)     # soft ramp into shadow
	var t10 := _cov[r0 + x1] * clampf((_grid[r0 + x1] - d) / 0.35, 0.0, 1.0)
	var t01 := _cov[r1 + x0] * clampf((_grid[r1 + x0] - d) / 0.35, 0.0, 1.0)
	var t11 := _cov[r1 + x1] * clampf((_grid[r1 + x1] - d) / 0.35, 0.0, 1.0)
	var t := lerpf(lerpf(t00, t10, ax), lerpf(t01, t11, ax), ay)
	return lerpf(1.0, SHADOW_MIN, t)
