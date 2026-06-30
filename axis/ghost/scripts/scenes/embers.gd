extends GhostScene

## Embers - a drift of sparks that twinkle and flare, each on its own.
##
## A cloud of point particles riding the wind (per-particle, so it drifts and rises
## asynchronously) with a weak spring + drag so they wander about home instead of
## escaping. The life is in the *light*, and it is async by construction: each spark
## twinkles on its own phase, and a beat lights only the sparks whose own threshold
## it crosses (through a nonlinear `spike`), so a hit ripples a *subset* alight - not
## the whole cloud throbbing in unison. Size barely moves; colour and brightness
## carry the audio. (The old version pulsed every spark's size with the global beat
## and burst them all from centre together - one synchronized throb. Fixed.)

var _f: AudioFeatures = AudioFeatures.new()
var _sys: ParticleSystem
var _t := 0.0
var _beat_prev := 0.0
var _ch := Vector2.ZERO       # live tonal colour (hue, strength) from the harmonic signature


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "particles"
	_sys = ParticleSystem.new()
	var base_hue := rng.randf_range(0.02, 0.12)     # warm
	var n := rng.randi_range(80, 160)
	for i in n:
		var a := rng.randf() * TAU
		# A wider, looser cloud (not a tight central clump) so the sparks no longer feel
		# bunched; the spawn radius itself varies per spark.
		var r := sqrt(rng.randf()) * rng.randf_range(0.22, 0.40)
		var p := Particle.new()
		p.home = Vector2(cos(a), sin(a)) * r
		p.radius = rng.randf_range(0.004, 0.012)
		p.hue = fposmod(base_hue + rng.randf_range(-0.04, 0.07), 1.0)
		p.noise = Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1))
		# Per-particle flare threshold (some embers catch on a faint beat, some need a
		# hard one), a twinkle phase/rate, and a mobility multiplier so each rides the
		# wind at its own speed (the Wind force reads "mobility") - varied, not uniform.
		p.data = {"thresh": 0.18 + 0.5 * absf(p.noise.y), "phase": rng.randf() * TAU,
			"rate": rng.randf_range(0.8, 2.4), "mobility": rng.randf_range(0.45, 1.7),
			# Each ember listens to its own spectral BAND (drives its transparency), and has a
			# high LAUNCH threshold: a strong harmonic moment flings it far outward (a mega
			# activation), still tethered by the weak spring. `h` is its smoothed band level.
			"band": rng.randf(), "launch": rng.randf_range(0.7, 1.5), "h": 0.0}
		_sys.add(p)

	# Per-particle wind drift + a *gentle* pull home (weak spring, so they wander wide
	# instead of clumping back to centre) + drag for decay. No synchronized burst.
	_sys.add_force("wind", {"amp": 0.24, "freq": 0.3, "lift": -0.14})   # negative y = rise
	_sys.add_force("spring", {"k": 0.5})
	_sys.add_force("drag", {"k": 1.1})
	return {"hue": base_hue}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05, 0.04, 0.08)
	_t += delta
	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	_ch = chroma_hue()          # tonal hue + strength, to tint the cloud toward the music's key
	for p: Particle in _sys.particles:
		var d: Dictionary = p.data
		# Each ember tracks its own band - its harmonic strength right now (drives transparency).
		d.h = lerpf(float(d.h), f.sample(float(d.band)), 1.0 - exp(-6.0 * delta))
		# MEGA ACTIVATION: on a beat, an ember whose harmonic-weighted drive clears its high
		# launch threshold is flung far OUTWARD from the attractor (with a swirl), the further
		# the harder the hit - then the weak spring reels it slowly back, so it fills the screen
		# yet stays loosely coupled. Only a subset fires per beat - never the whole cloud.
		if beat_edge:
			var drive: float = (f.beat + 0.6 * f.energy) * (0.35 + float(d.h))
			var excess: float = drive - float(d.launch)
			if excess > 0.0:
				var mag: float = (1.2 + 6.0 * Nonlinear.apply("spike", clampf(excess, 0.0, 1.0), 2.0)) * float(d.mobility)
				var dir: Vector2 = p.pos()
				if dir.length() < 0.05:
					dir = p.noise
				dir = dir.normalized()
				var tang := Vector2(-dir.y, dir.x) * (1.0 if p.nspin >= 0.0 else -1.0)   # swirl out
				p.vel += (dir + tang * 0.45) * mag
	_sys.step(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var beat_drive := _f.beat + 0.4 * _f.energy
	for p: Particle in _sys.particles:
		var d: Dictionary = p.data
		# Async twinkle + a flare that only fires when this spark's own threshold is
		# crossed, shaped by a spike so the onset is emphatic.
		var twinkle := 0.5 + 0.5 * sin(_t * float(d.rate) + float(d.phase))
		var flare := Nonlinear.apply("spike", clampf(beat_drive - float(d.thresh), 0.0, 1.0), 3.0)
		var harm := float(d.h)                                # this ember's own harmonic strength
		var v := clampf(0.28 + 0.34 * twinkle + 0.75 * flare, 0.05, 1.0)
		# Transparency is INDEPENDENT per ember, tied to its band: embers on a live harmonic
		# stay present, those on a quiet one fade toward glass - so the cloud reads the spectrum.
		var alpha := clampf(0.10 + 0.85 * harm + 0.45 * flare, 0.04, 1.0)
		var c := p.pos() * u
		var r := p.radius * u * (0.85 + 0.3 * twinkle)        # gentle, async - not beat-synced
		# Tint each ember toward the live tonal hue (circular nudge, scaled by tonal strength), so
		# the warm cloud drifts in colour with the music's key.
		var dh: float = _ch.x - p.hue
		dh = dh - round(dh)
		var col := Color.from_hsv(fposmod(p.hue + dh * 0.4 * _ch.y, 1.0), 0.6, v)
		draw_circle(c, r * (2.0 + 2.0 * flare), Color(col.r, col.g, col.b, (0.08 + 0.18 * flare) * (0.4 + 0.6 * harm)))
		draw_circle(c, r, Color(col.r, col.g, col.b, alpha))
