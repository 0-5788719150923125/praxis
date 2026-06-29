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


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "particles"
	_sys = ParticleSystem.new()
	var base_hue := rng.randf_range(0.02, 0.12)     # warm
	var n := rng.randi_range(80, 160)
	for i in n:
		var a := rng.randf() * TAU
		var r := sqrt(rng.randf()) * 0.14
		var p := Particle.new()
		p.home = Vector2(cos(a), sin(a)) * r
		p.radius = rng.randf_range(0.004, 0.012)
		p.hue = fposmod(base_hue + rng.randf_range(-0.04, 0.07), 1.0)
		p.noise = Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1))
		# Per-particle flare threshold (some embers catch on a faint beat, some need
		# a hard one) and a twinkle phase/rate - this is what makes it async.
		p.data = {"thresh": 0.18 + 0.5 * absf(p.noise.y), "phase": rng.randf() * TAU,
			"rate": rng.randf_range(0.8, 2.4)}
		_sys.add(p)

	# Per-particle wind drift + a gentle pull home. No synchronized burst.
	_sys.add_force("wind", {"amp": 0.20, "freq": 0.3, "lift": -0.14})   # negative y = rise
	_sys.add_force("spring", {"k": 1.1})
	_sys.add_force("drag", {"k": 1.2})
	return {"hue": base_hue}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05, 0.04, 0.08)
	_t += delta
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
		var v := clampf(0.28 + 0.34 * twinkle + 0.75 * flare, 0.05, 1.0)
		var c := p.pos() * u
		var r := p.radius * u * (0.85 + 0.3 * twinkle)        # gentle, async - not beat-synced
		var col := Color.from_hsv(p.hue, 0.6, v)
		draw_circle(c, r * (2.0 + 2.0 * flare), Color(col.r, col.g, col.b, 0.10 + 0.18 * flare))
		draw_circle(c, r, col)
