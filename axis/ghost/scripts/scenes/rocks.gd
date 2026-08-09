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
## differ - every computation perturbed by sampling. Colour is a [Scheme] mood chosen to
## suit the style (matte stone gets the earthy moods, the gem gets the jewel ones), and the
## stones either share a tight hue FAMILY or spread base-to-accent across the set. The
## composition is sampled too - a couple of colossal boulders, a resting cluster, or a wide
## scatter of scree - so the same subject is framed differently each time it is seeded.

enum Mode { PULSE, EXPLODE, CRUMBLE }
## ARRANGEMENTS - how many stones and how big, sampled as a set. The old build was always
## 2-4 zoomed-in stones, so every rocks scene had the same composition however it was
## seeded; these are three genuinely different readings of the same subject (a couple of
## colossi filling the frame, a resting cluster, a wide scatter of small stones).
## A scatter carries its own style list: scree is chips and shards, not polished masses -
## and the smooth style is the heaviest mesh (subdivided twice further), which a dozen of
## would cost real frame time for a look that scale hides anyway.
const ARRANGEMENTS := {
	"boulders": {"count": [1, 3],  "zoom": [1.9, 2.8], "spread": [0.0, 0.45], "radius": [0.12, 0.22],
		"styles": ["plain", "rough", "crystal", "hybrid"]},
	"cluster":  {"count": [3, 5],  "zoom": [1.4, 2.2], "spread": [0.0, 0.65], "radius": [0.10, 0.20],
		"styles": ["plain", "rough", "crystal", "hybrid"]},
	"scree":    {"count": [6, 10], "zoom": [0.8, 1.2], "spread": [0.20, 1.0], "radius": [0.07, 0.16],
		"styles": ["rough", "crystal", "hybrid"]},
}
## Which moods a stone of each style can be. Stone is mineral, so the earthy moods carry the
## matte styles while the faceted gem gets the jewel ones - within that, any of them.
const STYLE_MOODS := {
	"plain":   ["ash", "bone", "brass", "dawn", "sodium", "abyss", "glacier", "verdant"],
	"rough":   ["ash", "bone", "brass", "ember", "dawn", "sodium", "verdant", "abyss"],
	"crystal": ["violet", "magenta", "teal", "glacier", "rose", "toxic", "abyss", "ember"],
}
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
	# Composition first: it decides which stone styles belong in it (see ARRANGEMENTS).
	var arrangement := String(ARRANGEMENTS.keys()[rng.randi() % ARRANGEMENTS.size()])
	var arr: Dictionary = ARRANGEMENTS[arrangement]
	var styles: Array = arr.styles
	_style = String(styles[rng.randi() % styles.size()])
	lifecycle = "oneshot" if _mode == Mode.CRUMBLE else "loop"

	var mat: Dictionary = MATERIAL[_style]
	_edge = int(mat.edge)
	# How much of this scene leans skeletal is itself sampled: some scenes are all solid,
	# others mostly wireframe-revealed - the masking is a spectrum, not a flag.
	var reveal_chance := 0.0 if rng.randf() < 0.3 else rng.randf_range(0.3, 0.9)

	# Colour comes from a shared [Scheme] mood suited to the style, so a session of rocks is
	# ash-grey or bone or lurid gem rather than one arbitrary hue every time.
	var sch := Scheme.pick(rng) if _style == "hybrid" else Scheme.among(STYLE_MOODS[_style], rng)
	# Two ways a set of stones can relate: a tight FAMILY (all near the base hue, one stone
	# type) or a SPREAD from base to accent (a mixed deposit, each stone its own mineral).
	var family := rng.randf() < 0.45
	var count := rng.randi_range(int(arr.count[0]), int(arr.count[1]))
	var zoom := rng.randf_range(float(arr.zoom[0]), float(arr.zoom[1]))   # > 1 pushes rocks off-frame
	for i in count:
		# Sample the geometry family for this rock (the start of the spec pattern).
		var mesh := Mesh3D.hybrid(rng) if _style == "hybrid" else Mesh3D.rock(_style, rng)
		var spin := Vector3(
			rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-0.4, 0.4))
		# Spread wide - well past the frame edges, so big rocks are only partly on screen.
		var spread := rng.randf_range(float(arr.spread[0]), float(arr.spread[1]))
		var ang := rng.randf() * TAU
		# Family: everyone drifts a little off the base. Spread: the i-th of `count` related
		# hues, base through accent, so no two stones read the same.
		var hue := sch.vary(rng) if family else fposmod(
			sch.hue_at(i, count) + rng.randf_range(-0.02, 0.02), 1.0)
		var rock := {
			"mesh": mesh,
			"verts0": mesh.verts.duplicate(),   # pristine geometry, for the collision dent
			"center": Vector2(cos(ang), sin(ang)) * spread + Vector2(
				rng.randf_range(-0.12, 0.12), rng.randf_range(-0.10, 0.10)),
			"radius": rng.randf_range(float(arr.radius[0]), float(arr.radius[1])) * zoom,
			"hue": hue,
			"basis": Basis.from_euler(Vector3(rng.randf() * TAU, rng.randf() * TAU, 0.0)),
			"spin": spin.normalized() * rng.randf_range(0.07, 0.16),   # gentle
			"e": 0.0,
			"glow": 0.0,
			# Material sampled around the style centre, then SCALED by the mood's own
			# saturation - ash stones come out grey, toxic ones lurid, from the same style.
			"sat": clampf(float(mat.sat) * (0.35 + 1.3 * sch.sat)
				+ rng.randf_range(-0.10, 0.10), 0.0, 1.0),
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
			# The lattice under the coat reads as the mood's ACCENT, near-white, so the
			# skeletal half still belongs to the same scene as the stone half.
			rock.wire = sch.color(sch.accent, rng.randf_range(0.1, 0.4),
				1.4, rng.randf_range(0.6, 0.85))
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
	return {"mood": sch.name, "style": _style, "arrangement": arrangement,
		"count": count, "hues": "family" if family else "spread"}


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
