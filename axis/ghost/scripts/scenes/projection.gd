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
## blocky cells, fine ones as a smooth density cloud.
const BINS := {"coarse": 40, "paper": 64, "fine": 96}

## The latent's own structure. It is one eye every time - that is the scene - but a
## latent geometry is not obliged to be a single iris ring around a single nucleus.
const FORMS := ["plain", "banded", "haloed"]

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _pts := PackedVector3Array()    # base cloud (object space)
var _grid := PackedFloat32Array()
var _g := 64                        # density-grid resolution (sampled per scene)
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


func _draw() -> void:
	begin_draw()
	var u := unit()
	# Reset the density grid.
	for i in _g * _g:
		_grid[i] = 0.0
	# Project the posed, audio-shaped cloud into the grid.
	var basis := Basis.from_euler(_pose)
	var span := 2.4                          # object half-extent mapped to the grid
	var dscale := 1.0 + 0.25 * _dilate        # dilation spreads the whole cloud a touch
	for k in _n:
		var p: Vector3 = _pts[k]
		var sp := Vector3(p.x * _stretch, p.y, p.z) * dscale
		var r := basis * sp
		var gx := int((r.x / span * 0.5 + 0.5) * _g)
		var gy := int((r.y / span * 0.5 + 0.5) * _g)
		if gx >= 0 and gx < _g and gy >= 0 and gy < _g:
			_grid[gy * _g + gx] += 1.0
	# Log-normalise.
	var maxd := 1.0
	for i in _g * _g:
		maxd = maxf(maxd, _grid[i])
	var lmax := log(1.0 + maxd)
	# Render the grid as a heatmap centred in the frame.
	var grid_px := u * 1.25
	var cell := grid_px / float(_g)
	var origin := Vector2(-grid_px * 0.5, -grid_px * 0.5)
	var glow: float = 0.3 + 0.7 * _f.energy
	for gy in _g:
		for gx in _g:
			var d: float = _grid[gy * _g + gx]
			if d <= 0.0:
				continue
			var ti := log(1.0 + d) / lmax
			var pos := origin + Vector2(gx, gy) * cell
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
