extends GhostScene

## Projection - a PCA-style density map of a latent geometry, eye-shaped.
##
## A nod to the research paper's geometry figure and the dashboard's projection maps: a
## 3D point cloud shaped like a CALM model's latent centers - an elongated blob with a
## dense pupil nucleus and an iris ring, the "single eye" those gaussian-ish latents
## settle into - projected through a slowly tumbling 3D pose down to 2D (the top-2
## projection) and rendered as a **binned density grid with log colour**, exactly the
## figure's look. Audio drives DRAMATIC poses: the pupil dilates, the eye stretches, and
## beats snap the projection to a new angle. Nonlinear activations shape the cloud and
## the drive. One eye, varied by seed - its structure ([constant FORMS]: a plain iris,
## two concentric bands, or an iris with a sparse corona), its elongation, where the
## pupil sits (so it can look aside), how many latents there are and how coarsely they
## bin ([constant BINS]) are all sampled, and the colormap is one [Scheme]'s ramp.

## How the binning reads. The paper bins to 64x64 and that was the only resolution
## this scene ever used, so the map always had the same grain; coarse bins read as
## blocky cells, fine ones as a smooth density cloud. The number is the count across
## the SHORT screen axis; the long axis gets proportionally more of them (see _fit).
const BINS := {"coarse": 40, "paper": 64, "fine": 96}

## Object half-extent mapped onto one bin-count's worth of grid. Fixed, not sampled:
## it is the projection's scale contract, not a look tunable - the sampled geometry
## (elongation, iris radius, pupil spread) is what varies the picture.
const SPAN := 2.4

## The latent's own structure. It is one eye every time - that is the scene - but a
## latent geometry is not obliged to be a single iris ring around a single nucleus.
const FORMS := ["plain", "banded", "haloed"]

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _pts := PackedVector3Array()    # base cloud (object space)
var _grid := PackedFloat32Array()
var _g := 64                        # bins across the SHORT screen axis (sampled per scene)
var _gx := 64                       # grid columns ...
var _gy := 64                       # ... and rows - the LONG axis gets MORE bins, not bigger ones
var _win := Rect2i(0, 0, 0, 0)      # bins written last frame: the only ones that need clearing
var _n := 2600                      # points in the latent cloud
var _pose := Vector3.ZERO           # current pose euler (the projection angle)
var _spin := Vector3.ZERO
var _dilate := 0.0                  # pupil dilation drive (nonlinear, audio)
var _stretch := 1.0                 # horizontal stretch drive (audio)
var _sch: Scheme
var _hue := 0.0
var _beat_prev := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	_rng.seed = rng.randi()
	# A density map takes any palette - it is a colormap, not a thing with a nature.
	_sch = Scheme.pick(rng)
	_hue = _sch.hue
	var bname := String(BINS.keys()[rng.randi() % BINS.size()])
	_g = int(BINS[bname])
	_n = rng.randi_range(1500, 4200)
	# A placeholder shape only: the frame's size is not known until the node is in the tree,
	# so the real grid is sized on the first draw (see _fit).
	_gx = _g
	_gy = _g
	_grid.resize(_g * _g)
	# A dramatic starting angle and a slow tumble.
	_pose = Vector3(rng.randf_range(-0.6, 0.6), rng.randf() * TAU, rng.randf_range(-0.4, 0.4))
	_spin = Vector3(rng.randf_range(-0.22, 0.22),
		rng.randf_range(0.08, 0.30) * (1.0 if rng.randf() < 0.5 else -1.0),
		rng.randf_range(-0.15, 0.15))
	# Sampled eye geometry: elongation (round through to very wide), iris ring, pupil
	# nucleus, depth. The pupil can also sit OFF centre, so the eye looks aside.
	var form := String(FORMS[rng.randi() % FORMS.size()])
	var ex := rng.randf_range(1.0, 2.3)
	var ey := rng.randf_range(0.55, 1.10)
	var iris_r := rng.randf_range(0.45, 0.95)
	var iris_w := rng.randf_range(0.06, 0.26)
	var z_spread := rng.randf_range(0.10, 0.38)
	var pupil_frac := rng.randf_range(0.14, 0.40)
	var pupil_sig := rng.randf_range(0.08, 0.24)
	var gaze := Vector3(rng.randf_range(-0.30, 0.30), rng.randf_range(-0.18, 0.18), 0.0)
	# "banded" splits the iris into two concentric rings (two latent clusters at one
	# radius apart); "haloed" keeps one ring and scatters a sparse corona outside it.
	var band_gap := rng.randf_range(0.16, 0.34)
	var halo_r := iris_r * rng.randf_range(1.35, 1.9)
	var halo_frac := rng.randf_range(0.12, 0.28)
	_pts.resize(_n)
	for i in _n:
		var p: Vector3
		if rng.randf() < pupil_frac:
			p = _gauss3(rng) * pupil_sig + gaze                # the dense pupil nucleus
		else:
			var a := rng.randf() * TAU
			var rad := iris_r
			if form == "banded" and rng.randf() < 0.45:
				rad = maxf(0.12, iris_r - band_gap)            # the inner of two rings
			elif form == "haloed" and rng.randf() < halo_frac:
				rad = halo_r                                   # a sparse corona outside the iris
			var rr := rad + _gauss(rng) * iris_w
			var z := _gauss(rng) * z_spread
			p = Vector3(cos(a) * rr, sin(a) * rr, z)
		p.x *= ex
		p.y *= ey
		_pts[i] = p
	return {"mood": _sch.name, "form": form, "bins": bname, "points": _n,
		"elongation": ex / ey}


# Standard normal (Box-Muller) and a 3-vector of them.
func _gauss(rng: RandomNumberGenerator) -> float:
	return sqrt(-2.0 * log(maxf(1e-6, rng.randf()))) * cos(TAU * rng.randf())

func _gauss3(rng: RandomNumberGenerator) -> Vector3:
	return Vector3(_gauss(rng), _gauss(rng), _gauss(rng))


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	# Tumble the projection, faster with energy.
	_pose += _spin * delta * (0.6 + 0.9 * f.energy)
	# A beat snaps the pose to a fresh dramatic angle (a saccade of the projection).
	var beat_edge: bool = f.beat > 0.6 and _beat_prev <= 0.6
	_beat_prev = f.beat
	if beat_edge:
		_pose += Vector3(_rng.randf_range(-0.5, 0.5), _rng.randf_range(-0.7, 0.7),
			_rng.randf_range(-0.3, 0.3)) * (0.4 + 0.6 * f.energy)
	# Nonlinear drives: the pupil dilates on energy (spike), the eye stretches on bass.
	var dil := Nonlinear.apply("spike", clampf(0.8 * f.energy + f.beat, 0.0, 1.0), 2.0)
	_dilate = lerpf(_dilate, dil, 1.0 - exp(-4.0 * delta))
	_stretch = lerpf(_stretch, 1.0 + 0.5 * Nonlinear.apply("tanh", f.bass * 1.6, 1.0),
		1.0 - exp(-3.0 * delta))
	queue_redraw()


## Size the density grid to the FRAME rather than to a square, and hand back the bin size
## in pixels.
##
## THE 4:3 BUG. The grid was `_g` x `_g` bins drawn over a `unit() * 1.25` box, and unit() is
## the SHORTER screen axis - so on the 2880x1620 export the map only ever covered a 2025 px
## square in the middle of the frame and left 427 px of untouched ground down each side. Black
## on black while the cloud was small, and a hard vertical cut straight through it the moment
## the cloud was large: "the left and right sides were truncated by black... as if it were 4:3".
## The covered region measured 2025x1620, which is 1.25:1 - a hair off 4:3 inside a 16:9 frame.
##
## The BIN keeps the size it had, so the paper's grain and the eye's size on screen are both
## unchanged; the long axis simply gets more bins. The 1.25 overdraw is the original and stays:
## the view drifts and zooms, so the map has to reach past every edge.
func _fit() -> float:
	var cell := maxf(1.0, minf(size.x, size.y)) * 1.25 / float(_g)
	var gx := maxi(2, ceili(size.x * 1.25 / cell))
	var gy := maxi(2, ceili(size.y * 1.25 / cell))
	if gx != _gx or gy != _gy:
		_gx = gx
		_gy = gy
		_grid.resize(_gx * _gy)
		_grid.fill(0.0)
		_win = Rect2i(0, 0, 0, 0)      # nothing of the new grid has been written yet
	return cell


func _draw() -> void:
	begin_draw()
	var cell := _fit()
	# Clear only the window written last frame. The grid now spans the whole frame (16416 bins
	# at 1920x1080 on the fine setting, against 9216 before), and clearing and re-scanning all
	# of it every frame would cost more than the few thousand bins the cloud actually lands in.
	for wy in range(_win.position.y, _win.end.y):
		var row := wy * _gx
		for wx in range(_win.position.x, _win.end.x):
			_grid[row + wx] = 0.0
	# Project the posed, audio-shaped cloud into the grid.
	var basis := Basis.from_euler(_pose)
	var per := 2.0 * SPAN / float(_g)         # object units per bin - the SAME on both axes
	var dscale := 1.0 + 0.25 * _dilate        # dilation spreads the whole cloud a touch
	var hx := float(_gx) * 0.5
	var hy := float(_gy) * 0.5
	var x0 := _gx
	var x1 := -1
	var y0 := _gy
	var y1 := -1
	for k in _n:
		var p: Vector3 = _pts[k]
		var sp := Vector3(p.x * _stretch, p.y, p.z) * dscale
		var r := basis * sp
		# floori, not int(): int() truncates TOWARD ZERO, so a point just off the left or top
		# edge landed on bin 0 instead of being rejected and drew a false bright rank there.
		var gx := floori(r.x / per + hx)
		var gy := floori(r.y / per + hy)
		if gx < 0 or gx >= _gx or gy < 0 or gy >= _gy:
			continue
		_grid[gy * _gx + gx] += 1.0
		x0 = mini(x0, gx)
		x1 = maxi(x1, gx)
		y0 = mini(y0, gy)
		y1 = maxi(y1, gy)
	if x1 < x0 or y1 < y0:
		_win = Rect2i(0, 0, 0, 0)
		return
	_win = Rect2i(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
	# Log-normalise.
	var maxd := 1.0
	for wy in range(y0, y1 + 1):
		var row := wy * _gx
		for wx in range(x0, x1 + 1):
			maxd = maxf(maxd, _grid[row + wx])
	var lmax := log(1.0 + maxd)
	# Render the grid as a heatmap, centred on the frame and covering the whole of it.
	var origin := Vector2(-float(_gx) * cell * 0.5, -float(_gy) * cell * 0.5)
	var glow: float = 0.3 + 0.7 * _f.energy
	for wy in range(y0, y1 + 1):
		var row := wy * _gx
		for wx in range(x0, x1 + 1):
			var d: float = _grid[row + wx]
			if d <= 0.0:
				continue
			var ti := log(1.0 + d) / lmax
			var pos := origin + Vector2(wx, wy) * cell
			draw_rect(Rect2(pos, Vector2(cell, cell) * 1.04), _cmap(ti, glow))


# Log-density colormap: low density = dark, saturated; high = bright, desaturating to
# white. The RAMP is the scheme's own two hues - the sparse fringe sits on the accent
# and the dense core on the base, so the map is a colormap of one mood rather than a
# hue plus a fixed 0.55 leap that landed wherever it landed.
func _cmap(ti: float, glow: float) -> Color:
	var d := fposmod(_sch.accent - _sch.hue + 0.5, 1.0) - 0.5
	var h := fposmod(_sch.hue + d * (1.0 - ti), 1.0)
	var s := lerpf(clampf(_sch.sat * 1.2, 0.0, 1.0), _sch.sat * 0.15, ti)
	var v := clampf(0.10 + 1.15 * ti, 0.0, 1.0) * (0.55 + 0.6 * glow) * (0.7 + 0.35 * _sch.val)
	return Color.from_hsv(h, s, clampf(v, 0.0, 1.0), clampf(0.14 + 0.95 * ti, 0.0, 1.0))
