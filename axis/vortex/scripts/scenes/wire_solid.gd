extends VortexScene

## Wire solid - a translucent polyhedron in real 3D.
##
## A true 3D cube / octahedron / tetrahedron / icosahedron ([Mesh3D]): perspective
## projected, depth sorted, faces faint and edges bright so you see *through* it.
## Because it is genuinely 3D, it rotates slowly and continuously - that is how the
## volume reveals itself (a real solid turning, not a flat shape flickering). The
## rotation is deliberately gentle; energy only nudges its pace.

var _f: AudioFeatures = AudioFeatures.new()
var _mesh: Mesh3D
var _rot := Vector3.ZERO
var _hue := 0.0
var _glow := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	match rng.randi_range(0, 3):
		0: _mesh = Mesh3D.cube()
		1: _mesh = Mesh3D.octahedron()
		2: _mesh = Mesh3D.tetrahedron()
		_: _mesh = Mesh3D.icosphere(0)
	_rot = Vector3(rng.randf() * TAU, rng.randf() * TAU, rng.randf() * TAU)
	_hue = rng.randf()
	return {
		"radius": rng.randf_range(0.24, 0.36),
		# Slow tumble axis (rad/s) - a full turn takes the better part of a minute.
		"spin": Vector3(
			rng.randf_range(0.04, 0.12),
			rng.randf_range(0.04, 0.12),
			rng.randf_range(-0.06, 0.06)),
		"face_alpha": rng.randf_range(0.16, 0.30),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05)
	var spin: Vector3 = params.spin
	# Slow continuous 3D rotation reveals the solid; energy only nudges the pace.
	_rot += spin * delta * (0.7 + 0.5 * f.energy)
	# Audio drives the *glow*, not the size - the solid holds its shape.
	_glow = 0.35 * f.beat + 0.18 * f.energy
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var sz := float(params.radius) * unit()
	_mesh.draw_shaded(self, Basis.from_euler(_rot), Vector2.ZERO, sz,
		fposmod(_hue + 0.04 * _f.energy, 1.0), 0.5, 0.0, 2, float(params.face_alpha), _glow)
