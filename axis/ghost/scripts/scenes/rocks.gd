extends GhostScene

## Rocks - faceted stones in real 3D, sampled from a small material/geometry spec.
##
## Each rock is a [Mesh3D] rotated by a genuine 3D basis and drawn depth-sorted and
## shaded. The look is a *sampled configuration* of composable layers rather than a
## fixed scene: a geometry family, a surface texture, a material (gloss / roughness),
## and - sometimes - a partial **wireframe reveal**. Style (seeded) sets the character:
##   plain   - smooth rounded mass, satin sheen.
##   rough   - craggy boulder, matte, dark facet relief.
##   crystal - faceted gem, bright edges, glossy.
##   hybrid  - a geometric base (cube / octa / tetra) with rock crusting over part of
##             it (gaussian-masked growth) - part machined, part grown.
## Independently, any rock may be **partially revealed**: a gaussian alpha mask punches
## holes in its coat (a sampled masking threshold, sparse bare patches through to
## half-and-half) so the wireframe lattice shows through - half-realistic, half-skeletal.
## Mode (seeded) sets the motion: `pulse` (breathe), `explode` (faces burst out on the
## beat), `crumble` (faces push apart once, then the scene ends).
##
## Nothing here is one fixed constant per style: the material, responsiveness, and reveal
## are all *sampled per rock* around the style's centre, so two rocks of a kind still
## differ - every computation perturbed by sampling.

enum Mode { PULSE, EXPLODE, CRUMBLE }
const STYLES := ["plain", "rough", "crystal", "hybrid"]
# Per-style material CENTRES (each rock samples around these, below): [edge, sat, gloss, roughness].
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
var _crumble_t := 0.0
var _done := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "mesh3d"
	_mode = rng.randi_range(0, 2)
	_style = STYLES[rng.randi_range(0, STYLES.size() - 1)]
	lifecycle = "oneshot" if _mode == Mode.CRUMBLE else "loop"

	var mat: Dictionary = MATERIAL[_style]
	_edge = int(mat.edge)
	# How much of this scene leans skeletal is itself sampled: some scenes are all solid,
	# others mostly wireframe-revealed - the masking is a spectrum, not a flag.
	var reveal_chance := 0.0 if rng.randf() < 0.3 else rng.randf_range(0.3, 0.9)

	var base_hue := rng.randf()
	var count := rng.randi_range(2, 4)
	for i in count:
		# Sample the geometry family for this rock (the start of the spec pattern).
		var mesh := Mesh3D.hybrid(rng) if _style == "hybrid" else Mesh3D.rock(_style, rng)
		var spin := Vector3(
			rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-0.4, 0.4))
		var rock := {
			"mesh": mesh,
			"center": Vector2(rng.randf_range(-0.30, 0.30), rng.randf_range(-0.24, 0.24)),
			"radius": rng.randf_range(0.10, 0.17),
			"hue": fposmod(base_hue + 0.08 * rng.randf(), 1.0),
			"basis": Basis.from_euler(Vector3(rng.randf() * TAU, rng.randf() * TAU, 0.0)),
			"spin": spin.normalized() * rng.randf_range(0.07, 0.16),   # gentle
			"e": 0.0,
			"glow": 0.0,
			# Material sampled around the style centre - perturbed, not a shared constant.
			"sat": clampf(float(mat.sat) + rng.randf_range(-0.08, 0.08), 0.0, 1.0),
			"gloss": clampf(float(mat.gloss) * rng.randf_range(0.7, 1.3), 0.0, 1.0),
			"rough": clampf(float(mat.rough) + rng.randf_range(-0.12, 0.12), 0.05, 1.0),
			"react": rng.randf_range(0.75, 1.3),   # per-rock responsiveness to the audio
			"reveal": false,
			"rtex": null,
			"wire": Color.WHITE,
		}
		# Partial wireframe reveal, with a sampled masking threshold across the spectrum:
		# low threshold = mostly coat with sparse bare patches, near 0 = roughly half-and-half.
		if rng.randf() < reveal_chance:
			rock.reveal = true
			# -0.35 = mostly coat with a few sparse bare patches; 0.0 = roughly half-and-half
			# (the heaviest reveal - never more skeletal than that).
			var threshold := rng.randf_range(-0.35, 0.0)
			rock.rtex = Mesh3D.reveal_texture(rng, threshold,
				rng.randf_range(0.10, 0.22), rng.randf_range(0.03, 0.07))
			rock.wire = Color.from_hsv(
				float(rock.hue), rng.randf_range(0.0, 0.2), 1.0, rng.randf_range(0.6, 0.85))
		_rocks.append(rock)
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
		var react: float = rock.react
		# Rooted rocks barely turn; activation earns rotation (structure is the bias).
		rock.basis = rock.basis * Basis.from_euler(rock.spin * delta * (0.1 + 0.9 * a))
		match _mode:
			Mode.PULSE:
				# Pulse the light, not the size - the rock holds its form.
				rock.glow = (0.25 * f.energy + 0.40 * f.beat) * a * react
			Mode.EXPLODE:
				rock.e = maxf(rock.e, f.beat * a * 0.5 * react)
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
		var c := Vector2(rock.center) * u
		var rad := float(rock.radius) * u
		if rock.reveal:
			# The exploded-faces modes don't apply to a revealed shell (it would tear the
			# lattice apart); the reveal rocks breathe with light only.
			mesh.draw_revealed(self, rock.basis, c, rad,
				float(rock.hue), float(rock.sat), float(rock.glow), rock.wire, rock.rtex)
		else:
			mesh.draw_shaded(self, rock.basis, c, rad, float(rock.hue), float(rock.sat),
				float(rock.e), _edge, 1.0, float(rock.glow), float(rock.gloss), float(rock.rough))
