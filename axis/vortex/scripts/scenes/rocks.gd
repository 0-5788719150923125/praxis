extends VortexScene

## Rocks - faceted stones in real 3D.
##
## Each rock is a [Mesh3D] (a displaced icosphere) rotated by a genuine 3D basis
## and drawn depth-sorted and shaded - so tumbling reads as dimensional, not a
## sheared flat polygon. An [Activation] decides which rocks stir and which stay
## rooted. Style (seeded) sets the surface:
##   plain   - smooth, many faces, no edges.
##   rough   - heavily displaced lumps, dark facet relief.
##   crystal - low-poly gem, bright edges, high contrast.
## Mode (seeded) sets the motion: `pulse` (breathe), `explode` (faces burst out on
## the beat), `crumble` (faces push apart once, then the scene ends).

enum Mode { PULSE, EXPLODE, CRUMBLE }
const STYLES := ["plain", "rough", "crystal"]

var _f: AudioFeatures = AudioFeatures.new()
var _mode := Mode.PULSE
var _style := "plain"
var _rocks: Array = []
var _act: Activation
var _edge := 0
var _sat := 0.35
var _crumble_t := 0.0
var _done := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "mesh3d"
	_mode = rng.randi_range(0, 2)
	_style = STYLES[rng.randi_range(0, STYLES.size() - 1)]
	lifecycle = "oneshot" if _mode == Mode.CRUMBLE else "loop"

	_edge = 0 if _style == "plain" else (1 if _style == "rough" else 2)
	_sat = 0.30 if _style == "plain" else (0.42 if _style == "rough" else 0.5)

	var base_hue := rng.randf()
	var count := rng.randi_range(2, 4)
	for i in count:
		var mesh := Mesh3D.rock(_style, rng)   # coherent fractal mass + fracture facets
		var spin := Vector3(
			rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-0.4, 0.4))
		_rocks.append({
			"mesh": mesh,
			"center": Vector2(rng.randf_range(-0.30, 0.30), rng.randf_range(-0.24, 0.24)),
			"radius": rng.randf_range(0.10, 0.17),
			"hue": fposmod(base_hue + 0.08 * rng.randf(), 1.0),
			"basis": Basis.from_euler(Vector3(rng.randf() * TAU, rng.randf() * TAU, 0.0)),
			"spin": spin.normalized() * rng.randf_range(0.07, 0.16),   # gentle
			"e": 0.0,
			"glow": 0.0,
		})
	# Some instances have everyone stir; others keep most rocks rooted.
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.3, 0.7)
	_act = Activation.new(count, rng, sparsity)
	return {}


func finished() -> bool:
	return _done


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.04, 0.03, 0.05)
	var drive := clampf(f.energy * 0.8 + f.beat * 0.7, 0.0, 1.3)
	_act.update(drive, delta)

	for ri in _rocks.size():
		var rock: Dictionary = _rocks[ri]
		var a := _act.level(ri)
		# Rooted rocks barely turn; activation earns rotation (structure is the bias).
		rock.basis = rock.basis * Basis.from_euler(rock.spin * delta * (0.1 + 0.9 * a))
		match _mode:
			Mode.PULSE:
				# Pulse the light, not the size - the rock holds its form.
				rock.glow = (0.25 * f.energy + 0.40 * f.beat) * a
			Mode.EXPLODE:
				rock.e = maxf(rock.e, f.beat * a * 0.5)
				rock.e = maxf(0.0, rock.e - delta * 0.5)
			Mode.CRUMBLE:
				rock.e = minf(0.55, rock.e + delta * 0.22)

	if _mode == Mode.CRUMBLE:
		_crumble_t += delta
		if _crumble_t > 4.5:
			_done = true
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	for rock: Dictionary in _rocks:
		var mesh: Mesh3D = rock.mesh
		mesh.draw_shaded(self, rock.basis, Vector2(rock.center) * u,
			float(rock.radius) * u, float(rock.hue), _sat,
			float(rock.e), _edge, 1.0, float(rock.glow))
