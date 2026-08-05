extends RefCounted
class_name VoiceField

## VoiceField - the belt as a field of interfering sources.
##
## Every seed on the belt is a SOURCE sitting at its own coordinate, radiating
## an influence that falls off with distance at its own rate. A candidate voice
## sits somewhere in that field and is modulated by what reaches it. Drift far
## from the belt and the influences vanish, so the voice is free and stable;
## sit among several seeds and the modulation is strong and structured.
##
## HOW INFLUENCE COMBINES, and why it is a choice rather than an accident.
## Three different combination rules were already in this codebase, none of them
## chosen deliberately:
##   - `_field_reception` summed amplitudes ARITHMETICALLY but blended their
##     frequencies as an amplitude-weighted GEOMETRIC mean - additive in one
##     quantity, multiplicative in another, in the same function.
##   - `_adreno_step` summed SIGNED forces, the sign a hard +-1 threshold on
##     whether a seed's acceptance beat the belt mean.
##   - `_party_vector` and `_background_traits` take positive-weighted
##     arithmetic means, so nothing can ever subtract.
## Here the rule is named and selectable (see COMBINE), because the four modes
## are audibly different instruments: superposition is a chord, product is a
## gate, max is a spotlight, and ring is a beat.
##
## ON NEGATIVE INFLUENCE. Amplitudes here are SIGNED and continuous. A seed the
## belt likes less than average radiates a negative amplitude and SUBTRACTS,
## so combining two seeds can reduce a trait rather than only ever inflating it.
## That was previously a binary +-1 flag inside the anneal and absent everywhere
## else. But the sign flag is the weaker half of the idea: the stronger half is
## PHASE. Because each source carries a wave vector, `cos(theta_i)` already
## spans [-1, +1] as a function of WHERE THE CANDIDATE STANDS - so two seeds can
## reinforce here, partially cancel a step away, and null completely a step
## further. That is interference rather than the addition of signed numbers, and
## it is what makes distance and angle matter instead of just membership.
##
## Everything is derived from what a seed already has - its traits, its genome,
## its lineage, its acceptance - so no per-seed constant is ever rolled or
## hand-tuned. Deterministic per (belt contents, position): no wall clock.

## The combination rules. Each takes the per-source contributions and returns a
## single Vector3. Named, so a voice can carry which one its water obeys.
const COMBINE := ["sum", "product", "max", "ring"]

## Genome deltas are folded into the 3-axis frame through a fixed hyperplane
## bank so that the 16 genes contribute without any one of them dominating.
## Baked from a constant seed - the frame must never move between runs, or a
## saved belt would silently change shape.
const PLANE_SEED := 0x5EED_F1E1
## Genome contributes at this weight relative to the trait axes: present, but
## the traits are what a listener can name, so they lead.
const GENOME_WEIGHT := 0.45
## How sharply acceptance turns into polarity. tanh, so it is continuous and
## bounded: a seed at the belt mean radiates nothing, above it attracts, below
## it repels, and neither runs away.
const POLARITY_GAIN := 1.2
## Below this the source is not worth summing (and, for `product`, must not be
## allowed to gate everything to nothing).
const AMP_FLOOR := 0.004
## Total radiated amplitude a belt may carry, however many seeds are on it.
## Same shared-budget discipline as ProsodyWalk's modulators: a growing belt
## steals from itself rather than stacking, so catches do not get louder and
## more homogeneous as the game goes on.
const FIELD_BUDGET := 2.4

static var _planes: Array = []


## The 3 legible axes, the same grouping the HUD compass already uses:
##   X brightness - pitch, lilt, and song (a sung voice reads as lifted)
##   Y damage     - grit, air, breath: smooth vs abraded
##   Z drive      - pace, drawl, tract: urgent vs drawn-out
## Genome deltas fold in through the baked planes at GENOME_WEIGHT.
##
## Three axes, not the full 25, and the reason is measurable rather than
## aesthetic: a random offset's projection onto a source's wave vector has a
## standard deviation of about 0.58 in 3-D and 0.20 in 25-D. In 25-D nearly
## every pair of directions is close to orthogonal, so `cos(theta)` sits near
## zero almost everywhere and ANGLE STOPS DISCRIMINATING - the field would
## degenerate into pure distance falloff, which is the thing we are trying to
## fix. 3-D is also where the existing visual vocabulary already lives.
static func frame(v: PackedFloat32Array) -> Vector3:
	_bake()
	var n := Voice.TRAIT_KEYS.size()
	var g := func(i: int) -> float:
		return float(v[i]) if i < v.size() else 0.0
	# TRAIT_KEYS: pitch0 lilt1 tract2 pace3 breath4 grit5 drawl6 air7 song8
	var out := Vector3(
		(g.call(0) + g.call(1) + 0.7 * g.call(8)) * 0.55,
		(g.call(5) + g.call(7) + g.call(4)) * 0.55,
		(g.call(3) + g.call(6) + g.call(2)) * 0.55)
	for k in _planes.size():
		var d: float = g.call(n + k)
		if absf(d) > 0.0:
			out += (_planes[k] as Vector3) * (d * GENOME_WEIGHT)
	return out


static func _bake() -> void:
	if not _planes.is_empty():
		return
	var rng := RandomNumberGenerator.new()
	rng.seed = PLANE_SEED
	for _i in Voice.ProsodyWalk.PRIOR.size():
		_planes.append(Vector3(rng.randfn(0.0, 1.0), rng.randfn(0.0, 1.0),
			rng.randfn(0.0, 1.0)).normalized())


## One source, everything derived.
##
##   pos    - where the seed sits, in the frame above
##   amp    - SIGNED: tanh of how far its acceptance beats the belt mean, so a
##            disliked seed subtracts. Continuous, unlike the +-1 it replaces.
##   range  - from the seed's OWN `ring` and `damp` genes: `ring` is how far its
##            resonance carries and `damp` is how hard it absorbs, so
##            `ring * (1 - damp)` is already the seed's reach and needs no new
##            roll. Normalized against the belt's own scale so the units are the
##            belt's, not a constant's. The gene ranges give a ~40x span, which
##            is where "some seeds reach across the map, some only work at point
##            blank" comes from.
##   k      - wave VECTOR: direction is the seed's own bearing, magnitude comes
##            from its strongest lineage modulator's rate. A vector, so the
##            phase depends on the candidate's angle to the seed and not only
##            its distance.
##   phase  - that modulator's own phase.
static func source(traits: Dictionary, genome: Dictionary, lineage: Array,
		acceptance: float, seed_vec: PackedFloat32Array,
		reach_norm: float, scale: float) -> Dictionary:
	var pos := frame(seed_vec)
	var ring: float = float(genome.get("ring", 0.6))
	var damp: float = float(genome.get("damp", 0.35))
	var reach: float = maxf(ring * (1.0 - damp), 0.02)
	var rate := 0.5
	var phase := 0.0
	var best := 0.0
	for m in Voice.ProsodyWalk._lineage_mods(lineage):
		if float(m.depth) > best:
			best = float(m.depth)
			rate = float(m.rate)
			phase = float(m.phase)
	var dir := pos.normalized() if pos.length() > 0.0001 else Vector3(1.0, 0.0, 0.0)
	return {
		"pos": pos,
		"amp": tanh((acceptance - 1.0) * POLARITY_GAIN),
		"range": maxf(scale * reach / maxf(reach_norm, 0.0001), 0.05),
		"k": dir * (TAU * rate / maxf(scale, 0.0001)),
		"phase": phase,
		"key": hash(str(lineage)),
	}


## Evaluate the field at `p`. Returns the modulation vector, its energy, and the
## proximity/centre-frequency the reception filter wants - one field, so the
## water a player HEARS and the forces a candidate FEELS agree by construction.
static func evaluate(sources: Array, p: Vector3, mode := "sum") -> Dictionary:
	# stable order: the belt is an array and eviction reorders it, but a field
	# is a property of the SET of sources, not of how they are stored
	var srcs := sources.duplicate()
	srcs.sort_custom(func(a, b): return int(a.key) < int(b.key))
	var contrib: Array = []
	var atot := 0.0
	for s in srcs:
		var d: Vector3 = p - (s.pos as Vector3)
		var dist := d.length()
		var a: float = float(s.amp) * exp(-dist / float(s.range))
		if absf(a) < AMP_FLOOR:
			continue
		atot += absf(a)
		contrib.append({
			"a": a,
			"th": (s.k as Vector3).dot(d) + float(s.phase),
			"u": (-d).normalized() if dist > 0.0001 else Vector3.ZERO,
			"dist": dist,
		})
	if contrib.is_empty():
		return {"vec": Vector3.ZERO, "energy": 0.0, "prox": 0.0, "freq": 400.0}
	# the shared budget: scale the whole set down rather than letting it stack
	var trim: float = 1.0 if atot <= FIELD_BUDGET else FIELD_BUDGET / atot
	var vec := Vector3.ZERO
	match mode:
		"product":
			# GATING. Each source multiplies what came before, so one source
			# sitting on a null can silence the whole belt - the field goes
			# quiet in specific places rather than merely getting weaker.
			var m := 1.0
			for c in contrib:
				m *= 1.0 + float(c.a) * trim * cos(float(c.th))
			var dirp := Vector3.ZERO
			for c in contrib:
				dirp += (c.u as Vector3) * absf(float(c.a))
			vec = dirp.normalized() * (m - 1.0)
		"max":
			# SPOTLIGHT. The single strongest source wins outright; the belt is
			# a set of territories rather than a blend.
			var bestc: Dictionary = contrib[0]
			for c in contrib:
				if absf(float(c.a)) > absf(float(bestc.a)):
					bestc = c
			vec = (bestc.u as Vector3) * (float(bestc.a) * trim * cos(float(bestc.th)))
		"ring":
			# BEATS. Only the PAIRWISE products survive, so a lone seed radiates
			# nothing and the field is made entirely of relationships between
			# seeds - the most literal reading of "interference".
			for i in contrib.size():
				for j in range(i + 1, contrib.size()):
					var ci: Dictionary = contrib[i]
					var cj: Dictionary = contrib[j]
					var amp: float = float(ci.a) * float(cj.a) * trim * trim
					vec += ((ci.u as Vector3) + (cj.u as Vector3)).normalized() \
						* (amp * cos(float(ci.th) - float(cj.th)))
		_:
			# SUPERPOSITION, the default and the physical one. `cos` has zero
			# spatial mean, so this perturbs WITHOUT biasing toward the belt's
			# centroid - which is the structural reason the field cannot
			# collapse every candidate onto the party centre the way a fourth
			# attraction term would.
			for c in contrib:
				vec += (c.u as Vector3) * (float(c.a) * trim * cos(float(c.th)))
	# reception: proximity is how much total amplitude arrives, the centre
	# frequency the amplitude-weighted geometric blend of the sources reached
	var prox := clampf(atot * trim, 0.0, 1.0)
	var flog := 0.0
	var fw := 0.0
	for i in contrib.size():
		var w: float = absf(float(contrib[i].a))
		fw += w
		flog += w * log(maxf(120.0 / maxf(float(contrib[i].dist), 0.05), 60.0))
	var freq: float = exp(flog / maxf(fw, 0.0001)) if fw > 0.0 else 400.0
	return {"vec": vec, "energy": vec.length(), "prox": prox,
		"freq": clampf(freq, 90.0, 3000.0)}
