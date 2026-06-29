extends Scene3D

## Wire solid - a translucent polyhedron on the unified 3D path.
##
## A true cube / octahedron / tetrahedron / icosahedron ([Mesh3D]) projected through
## a [Lens3D]: perspective, depth sorted, faces faint and edges bright so you see
## *through* it. Because it is genuinely 3D it rotates slowly and continuously - that
## is how the volume reveals itself. Migrated off the old centred draw_shaded
## projector onto [Scene3D]: the body now lives in a camera world (it could share
## the frame with planes or other bodies), and the lens eases in instead of a fixed
## focal. Rotation is gentle; energy only nudges its pace, and audio drives the glow.

var _f: AudioFeatures = AudioFeatures.new()
var _mesh: Mesh3D
var _rot := Vector3.ZERO
var _hue := 0.0
var _glow := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"                       # the lens frames it; keep the 2D view square-on
	match rng.randi_range(0, 3):
		0: _mesh = Mesh3D.cube()
		1: _mesh = Mesh3D.octahedron()
		2: _mesh = Mesh3D.tetrahedron()
		_: _mesh = Mesh3D.icosphere(0)
	_rot = Vector3(rng.randf() * TAU, rng.randf() * TAU, rng.randf() * TAU)
	_hue = rng.randf()
	lens.fov = rng.randf_range(42.0, 56.0)
	lens.eye = Vector3(0.0, 0.0, rng.randf_range(3.4, 4.4))   # a touch of perspective
	return {
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
	bodies.clear()
	add_body(_mesh, Basis.from_euler(_rot), Vector3.ZERO, 0.7,
		fposmod(_hue + 0.04 * f.energy, 1.0), 0.5, 2, float(params.face_alpha), _glow)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	render_world()
