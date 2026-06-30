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
	texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED   # so the panned reveal mask wraps seamlessly
	_mode = rng.randi_range(0, 2)
	_style = STYLES[rng.randi_range(0, STYLES.size() - 1)]
	lifecycle = "oneshot" if _mode == Mode.CRUMBLE else "loop"

	var mat: Dictionary = MATERIAL[_style]
	_edge = int(mat.edge)
	# How much of this scene leans skeletal is itself sampled: some scenes are all solid,
	# others mostly wireframe-revealed - the masking is a spectrum, not a flag.
	var reveal_chance := 0.0 if rng.randf() < 0.3 else rng.randf_range(0.3, 0.9)

	var base_hue := rng.randf()
	# Each rock gets a DISTINCT colour, spread across a sampled span around the base hue (so
	# some scenes are a tight family, others a wide spread) and distributed by index so no
	# two stones read the same - per the note that different shapes want different colours.
	var hue_span := rng.randf_range(0.22, 0.72)
	var hue_dir := 1.0 if rng.randf() < 0.5 else -1.0
	# Fewer, bigger, more spread-out rocks: a zoomed-in cluster where the largest run off
	# the edges rather than sitting tidily centred. The overall scale is sampled per scene
	# so some shows are a few colossal boulders, others a looser scatter of mid stones.
	var count := rng.randi_range(2, 4)
	var zoom := rng.randf_range(1.5, 2.6)            # > 1 pushes rocks bigger / off-frame
	for i in count:
		var hue_t := float(i) / float(maxi(1, count - 1))   # 0..1 across the rocks
		# Sample the geometry family for this rock (the start of the spec pattern).
		var mesh := Mesh3D.hybrid(rng) if _style == "hybrid" else Mesh3D.rock(_style, rng)
		var spin := Vector3(
			rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-0.4, 0.4))
		# Spread wide - well past the frame edges, so big rocks are only partly on screen.
		var spread := rng.randf_range(0.0, 0.65)
		var ang := rng.randf() * TAU
		var rock := {
			"mesh": mesh,
			"verts0": mesh.verts.duplicate(),   # pristine geometry, for the collision dent
			"center": Vector2(cos(ang), sin(ang)) * spread + Vector2(
				rng.randf_range(-0.12, 0.12), rng.randf_range(-0.10, 0.10)),
			"radius": rng.randf_range(0.10, 0.20) * zoom,   # zoomed in; the biggest overflow the frame
			"hue": fposmod(base_hue + hue_dir * (hue_t - 0.5) * hue_span + rng.randf_range(-0.04, 0.04), 1.0),
			"basis": Basis.from_euler(Vector3(rng.randf() * TAU, rng.randf() * TAU, 0.0)),
			"spin": spin.normalized() * rng.randf_range(0.07, 0.16),   # gentle
			"e": 0.0,
			"glow": 0.0,
			# Material sampled around the style centre - perturbed, not a shared constant.
			"sat": clampf(float(mat.sat) + rng.randf_range(-0.14, 0.14), 0.0, 1.0),
			"gloss": clampf(float(mat.gloss) * rng.randf_range(0.7, 1.3), 0.0, 1.0),
			"rough": clampf(float(mat.rough) + rng.randf_range(-0.12, 0.12), 0.05, 1.0),
			"react": rng.randf_range(0.75, 1.3),   # per-rock responsiveness to the audio
			"reveal": false,
			"rtex": null,
			"wire": Color.WHITE,
			"pan": 0.0, "pan_rate": 0.0,   # reveal rocks: continuous mask drift (set below)
			"pan_seed": 0.0,               # per-rock phase for the spatially-varying drift field
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
				# Each revealed rock's crust drifts at its own slow rate/direction, so the mask
				# is always gently panning rather than holding a fixed (looping) pattern.
			rock.pan_rate = rng.randf_range(0.015, 0.055) * (1.0 if rng.randf() < 0.5 else -1.0)
			rock.pan_seed = rng.randf_range(0.0, TAU)
		_rocks.append(rock)
	# Settle the cluster so the stones rest against each other instead of passing through
	# one another (which read as broken collision); a light overlap is left so the contact
	# dent still shows where they press.
	_relax_positions(rng)
	# Some instances have everyone stir; others keep most rocks rooted.
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.3, 0.7)
	_act = Activation.new(count, rng, sparsity)
	return {}


# Push overlapping rock centres apart over a few relaxation passes until they only lightly
# overlap - a believable touching pile rather than a heap of interpenetrating shapes. The
# pushes are symmetric so the cluster stays centred (the biggest stones still overflow the
# frame). Positions are static after this, so one settle at build time is enough.
func _relax_positions(rng: RandomNumberGenerator) -> void:
	for _iter in 32:
		var moved := false
		for i in _rocks.size():
			for j in range(i + 1, _rocks.size()):
				var a: Dictionary = _rocks[i]
				var b: Dictionary = _rocks[j]
				var d: Vector2 = Vector2(b.center) - Vector2(a.center)
				var dist := d.length()
				var mind: float = (float(a.radius) + float(b.radius)) * 0.86   # ~14% overlap kept
				if dist < 1e-4:
					d = Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1)).normalized()
					dist = 0.001
				if dist < mind:
					var push: Vector2 = (d / dist) * (mind - dist) * 0.5
					a.center = Vector2(a.center) - push
					b.center = Vector2(b.center) + push
					moved = true
		if not moved:
			break


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
		# Continuously drift the reveal mask (slightly faster with activation) so the crust
		# keeps panning instead of holding a static pattern.
		rock.pan += delta * float(rock.pan_rate) * (1.0 + 0.6 * a)
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
	_deform_collisions()
	queue_redraw()


# Where two rocks overlap (their screen circles intersect), dent the contact-facing
# faces inward so the panels bend, as if pressed together. The dent is in WORLD space
# at the contact: each frame we restore the pristine geometry and push the vertices that
# currently rotate into the contact direction back along it - so as a rock spins, its
# surface flows through a dent that stays put at the contact, rather than the whole rock
# carrying a fixed flat spot. Bounded and only while overlapping.
func _deform_collisions() -> void:
	var n := _rocks.size()
	for i in n:
		var rock: Dictionary = _rocks[i]
		var verts: PackedVector3Array = (rock.verts0 as PackedVector3Array).duplicate()
		var ci: Vector2 = rock.center
		var ri: float = rock.radius
		var basinv := (rock.basis as Basis).inverse()
		for j in n:
			if j == i:
				continue
			var other: Dictionary = _rocks[j]
			var d: Vector2 = Vector2(other.center) - ci
			var dist := d.length()
			var overlap: float = (ri + float(other.radius)) - dist
			if overlap <= 0.0 or dist < 1e-4:
				continue
			# Contact direction (toward the neighbour), mapped from world/screen into this
			# rock's object space; dent depth grows with the overlap.
			var ldir := (basinv * Vector3(d.x / dist, d.y / dist, 0.0)).normalized()
			var dent := clampf(overlap / ri, 0.0, 0.7) * 0.45
			for k in verts.size():
				var v := verts[k]
				var vl := v.length()
				if vl < 1e-5:
					continue
				var facing := (v / vl).dot(ldir)
				if facing > 0.15:
					verts[k] = v - ldir * ((facing - 0.15) * dent * vl)
		rock.mesh.verts = verts


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
				float(rock.hue), float(rock.sat), float(rock.glow), rock.wire, rock.rtex,
				float(rock.pan), float(rock.pan_seed))
		else:
			mesh.draw_shaded(self, rock.basis, c, rad, float(rock.hue), float(rock.sat),
				float(rock.e), _edge, 1.0, float(rock.glow), float(rock.gloss), float(rock.rough))
