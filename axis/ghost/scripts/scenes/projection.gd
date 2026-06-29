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
## the drive. One eye, varied by seed.

const G := 64                       # density-grid resolution (the paper bins to 64x64)
const N := 2600                     # points in the latent cloud

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _pts := PackedVector3Array()    # base cloud (object space)
var _grid := PackedFloat32Array()
var _pose := Vector3.ZERO           # current pose euler (the projection angle)
var _spin := Vector3.ZERO
var _dilate := 0.0                  # pupil dilation drive (nonlinear, audio)
var _stretch := 1.0                 # horizontal stretch drive (audio)
var _hue := 0.0
var _beat_prev := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	_rng.seed = rng.randi()
	_hue = rng.randf()
	_grid.resize(G * G)
	# A dramatic starting angle and a slow tumble.
	_pose = Vector3(rng.randf_range(-0.6, 0.6), rng.randf() * TAU, rng.randf_range(-0.4, 0.4))
	_spin = Vector3(rng.randf_range(-0.22, 0.22),
		rng.randf_range(0.08, 0.30) * (1.0 if rng.randf() < 0.5 else -1.0),
		rng.randf_range(-0.15, 0.15))
	# Sampled eye geometry: elongation (wider than tall), iris ring, pupil nucleus, depth.
	var ex := rng.randf_range(1.2, 1.9)
	var ey := rng.randf_range(0.7, 1.0)
	var iris_r := rng.randf_range(0.55, 0.85)
	var iris_w := rng.randf_range(0.10, 0.22)
	var z_spread := rng.randf_range(0.12, 0.30)
	var pupil_frac := rng.randf_range(0.18, 0.32)
	var pupil_sig := rng.randf_range(0.10, 0.20)
	_pts.resize(N)
	for i in N:
		var p: Vector3
		if rng.randf() < pupil_frac:
			p = _gauss3(rng) * pupil_sig                       # the dense pupil nucleus
		else:
			var a := rng.randf() * TAU
			var rr := iris_r + _gauss(rng) * iris_w            # the iris ring
			var z := _gauss(rng) * z_spread
			p = Vector3(cos(a) * rr, sin(a) * rr, z)
		p.x *= ex
		p.y *= ey
		_pts[i] = p
	return {}


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
	for i in G * G:
		_grid[i] = 0.0
	# Project the posed, audio-shaped cloud into the grid.
	var basis := Basis.from_euler(_pose)
	var span := 2.4                          # object half-extent mapped to the grid
	var dscale := 1.0 + 0.25 * _dilate        # dilation spreads the whole cloud a touch
	for k in N:
		var p: Vector3 = _pts[k]
		var sp := Vector3(p.x * _stretch, p.y, p.z) * dscale
		var r := basis * sp
		var gx := int((r.x / span * 0.5 + 0.5) * G)
		var gy := int((r.y / span * 0.5 + 0.5) * G)
		if gx >= 0 and gx < G and gy >= 0 and gy < G:
			_grid[gy * G + gx] += 1.0
	# Log-normalise.
	var maxd := 1.0
	for i in G * G:
		maxd = maxf(maxd, _grid[i])
	var lmax := log(1.0 + maxd)
	# Render the grid as a heatmap centred in the frame.
	var grid_px := u * 1.25
	var cell := grid_px / float(G)
	var origin := Vector2(-grid_px * 0.5, -grid_px * 0.5)
	var glow: float = 0.3 + 0.7 * _f.energy
	for gy in G:
		for gx in G:
			var d: float = _grid[gy * G + gx]
			if d <= 0.0:
				continue
			var ti := log(1.0 + d) / lmax
			var pos := origin + Vector2(gx, gy) * cell
			draw_rect(Rect2(pos, Vector2(cell, cell) * 1.04), _cmap(ti, glow))


# Log-density colormap: low density = dark, saturated; high = bright, desaturating to
# white. Hue shifts with density and rides the scene's palette - a perceptual ramp.
func _cmap(ti: float, glow: float) -> Color:
	var h := fposmod(_hue + 0.55 * (1.0 - ti), 1.0)
	var s := lerpf(0.85, 0.12, ti)
	var v := clampf(0.10 + 1.15 * ti, 0.0, 1.0) * (0.55 + 0.6 * glow)
	return Color.from_hsv(h, s, v, clampf(0.14 + 0.95 * ti, 0.0, 1.0))
