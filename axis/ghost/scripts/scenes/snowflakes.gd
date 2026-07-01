extends GhostScene

## Snowflakes - a field of several dozen crystal dendrites, restored and multiplied.
##
## The hero crystal scene (distinct from `snowfall`, the soft atmospheric one). The old
## single hard-coded Koch flake is gone; this is the elegant-procedural answer: dozens
## of six-fold dendrites, each generated with sampled branch geometry ([method
## Layer.draw_flake]) so no two are alike, in varied sizes across a cool bed. Motion
## follows the flat-subject discipline (in-plane spin + gentle sway, never fake depth):
## every flake turns on its *own* signed angular velocity, varied in direction and
## speed, so the field is organic rather than locked in lockstep. Mode (by seed) sets
## only the translation - a stationary field (`drift`) or one carried on a curl-noise
## breeze (`wind`). Colour, not size, carries the audio: the crystals brighten with energy.

const FlowField := preload("res://scripts/flow.gd")

var _flakes: Array = []
var _mode := "independent"
var _flow: FlowField
var _spin := 0.0
var _f: AudioFeatures = AudioFeatures.new()


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = [0.58, 0.62, 0.55, 0.5][rng.randi() % 4]
	# Mode is only about translation now: a stationary field, or one carried on the wind.
	# Rotation is always per-flake (see below), never globally locked.
	_mode = "wind" if rng.randf() < 0.4 else "drift"
	add_layer("bed", rng, {"hue": hue, "sat": 0.4, "val": 0.18, "pools": 2})
	_flow = FlowField.new(rng.randi(), 1.6, 0.05)
	var n := rng.randi_range(26, 50)
	for i in n:
		var depth := rng.randf_range(0.35, 1.0)
		# Each flake is a thin plate TUMBLING in 3D: a full starting orientation and a signed
		# angular velocity on all THREE axes, so it turns in space and foreshortens as it goes
		# (edge-on it thins to a line). A flat in-plane spin of a six-fold-symmetric flake barely
		# reads; a 3D tumble unmistakably does.
		_flakes.append({
			"pos": Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)),
			"size": rng.randf_range(0.02, 0.07) * depth,
			"depth": depth,
			"shape": rng.randf(),
			"ax": rng.randf() * TAU, "ay": rng.randf() * TAU, "az": rng.randf() * TAU,
			"wx": rng.randf_range(0.35, 1.1) * (1.0 if rng.randf() < 0.5 else -1.0),
			"wy": rng.randf_range(0.35, 1.1) * (1.0 if rng.randf() < 0.5 else -1.0),
			"wz": rng.randf_range(0.35, 1.1) * (1.0 if rng.randf() < 0.5 else -1.0),
			# Each flake tumbles faster to its OWN harmonic band (high / treble / mid / energy)
			# with its own sensitivity, so the surges are independent, not one global wobble.
			"band": rng.randi() % 4,
			"spin": rng.randf_range(0.9, 2.6),
			"phase": rng.randf() * TAU,
			"hue": fposmod(hue + rng.randf_range(-0.04, 0.05), 1.0),
			"sway": rng.randf() * TAU,
		})
	return {"hue": hue, "mode": _mode}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	update_layers(f, delta)
	_spin += delta * (0.15 + 0.3 * f.energy)
	# Advance each flake's 3D tumble on all three axes: a steady signed base rate plus an
	# independent surge keyed to that flake's own harmonic band, so they tumble independently
	# rather than turning in one locked direction.
	var bands := [f.high, f.treble, f.mid, f.energy]
	for fl in _flakes:
		var rate: float = 0.5 + float(bands[int(fl.band)]) * float(fl.spin)
		fl.ax += float(fl.wx) * delta * rate
		fl.ay += float(fl.wy) * delta * rate
		fl.az += float(fl.wz) * delta * rate
	if _mode == "wind":
		_flow.advance(delta)
		for fl in _flakes:
			fl.pos += _flow.at(fl.pos) * 0.05 * delta * float(fl.depth)
			fl.pos.x = wrapf(fl.pos.x, -1.15, 1.15)
			fl.pos.y = wrapf(fl.pos.y, -1.15, 1.15)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
	var u := unit()
	var hx := size.x / (2.0 * maxf(1.0, u))
	var hy := size.y / (2.0 * maxf(1.0, u))
	var glow_b: float = 0.3 + 0.7 * _f.energy
	for fl in _flakes:
		var swayx: float = 0.02 * sin(_spin * 0.5 + fl.sway)
		var px: float = fl.pos.x * hx + swayx
		var py: float = fl.pos.y * hy
		var c := Vector2(px, py) * u
		var r: float = fl.size * u
		# The flake's local plane, tumbled in 3D and projected orthographically (drop z): the
		# screen images of its local x / y axes foreshorten as it turns, so it reads as tumbling.
		var b := Basis.from_euler(Vector3(float(fl.ax), float(fl.ay), float(fl.az)))
		var ex := Vector2(b.x.x, b.x.y) * r
		var ey := Vector2(b.y.x, b.y.y) * r
		var bright: float = clampf(0.55 + 0.45 * glow_b, 0.0, 1.0)
		var col := Color.from_hsv(float(fl.hue), 0.12, 1.0, clampf(0.5 + 0.4 * float(fl.depth), 0.4, 0.95))
		Layer.glow(self, c, r * 1.5, Color(col.r, col.g, col.b, 0.10 * bright), 4)
		Layer.draw_flake_basis(self, c, ex, ey,
			Color(col.r, col.g, col.b, col.a * bright), maxf(1.0, r * 0.06), 6, float(fl.shape))
