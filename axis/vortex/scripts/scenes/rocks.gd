extends VortexScene

## Rocks - faceted stones in real 3D, sampled from a small material/geometry spec.
##
## Each rock is a [Mesh3D] rotated by a genuine 3D basis and drawn depth-sorted and
## shaded. The look is a *sampled configuration* of composable layers rather than a
## fixed scene: a geometry family, a surface texture, and a material (gloss /
## roughness). An [Activation] decides which rocks stir and which stay rooted. Style
## (seeded) sets the character:
##   plain   - smooth rounded mass, satin sheen.
##   rough   - craggy boulder, matte, dark facet relief.
##   crystal - faceted gem, bright edges, glossy.
##   hybrid  - a geometric base (cube / octa / tetra) with rock crusting over part of
##             it (gaussian-masked growth) - part machined, part grown.
## Mode (seeded) sets the motion: `pulse` (breathe), `explode` (faces burst out on
## the beat), `crumble` (faces push apart once, then the scene ends).

enum Mode { PULSE, EXPLODE, CRUMBLE }
const STYLES := ["plain", "rough", "crystal", "hybrid"]
# Per-style material: [edge, sat, gloss, roughness].
const MATERIAL := {
	"plain":   {"edge": 0, "sat": 0.30, "gloss": 0.18, "rough": 0.6},
	"rough":   {"edge": 1, "sat": 0.42, "gloss": 0.05, "rough": 0.95},
	"crystal": {"edge": 2, "sat": 0.50, "gloss": 0.55, "rough": 0.18},
	"hybrid":  {"edge": 1, "sat": 0.38, "gloss": 0.30, "rough": 0.45},
}

var _f: AudioFeatures = AudioFeatures.new()
var _mode := Mode.PULSE
var _style := "plain"
var _rocks: Array = []
var _act: Activation
var _edge := 0
var _sat := 0.35
var _gloss := 0.0
var _rough := 0.6
var _crumble_t := 0.0
var _done := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "mesh3d"
	_mode = rng.randi_range(0, 2)
	_style = STYLES[rng.randi_range(0, STYLES.size() - 1)]
	lifecycle = "oneshot" if _mode == Mode.CRUMBLE else "loop"

	var mat: Dictionary = MATERIAL[_style]
	_edge = int(mat.edge)
	_sat = float(mat.sat)
	_gloss = float(mat.gloss)
	_rough = float(mat.rough)

	var base_hue := rng.randf()
	var count := rng.randi_range(2, 4)
	for i in count:
		# Sample the geometry family for this rock (the start of the spec pattern).
		var mesh := Mesh3D.hybrid(rng) if _style == "hybrid" else Mesh3D.rock(_style, rng)
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
			float(rock.e), _edge, 1.0, float(rock.glow), _gloss, _rough)
