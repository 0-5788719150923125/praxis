extends VortexScene

## Embers - a drift of sparks that flare on the beat and ride the wind.
##
## This scene is almost no code: it is the primitive registry recombined. A cloud
## of point particles is driven by `scatter` (burst out on the beat - from glass),
## `wind` with upward lift (drift sideways and rise - from the landscape), and a
## weak `spring` + `drag` (so they ease back instead of escaping). Different
## physics, same parts. That is the point of the kit.

var _f: AudioFeatures = AudioFeatures.new()
var _sys: ParticleSystem


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_sys = ParticleSystem.new()
	var base_hue := rng.randf_range(0.02, 0.12)     # warm
	var n := rng.randi_range(70, 150)
	for i in n:
		var a := rng.randf() * TAU
		var r := sqrt(rng.randf()) * 0.12
		var p := Particle.new()
		p.home = Vector2(cos(a), sin(a)) * r
		p.radius = rng.randf_range(0.004, 0.012)
		p.hue = fposmod(base_hue + rng.randf_range(-0.04, 0.07), 1.0)
		p.noise = Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1))
		_sys.add(p)

	_sys.add_force("scatter", {"strength": 0.5, "jitter": 0.9, "period": 1.6})
	_sys.add_force("wind", {"amp": 0.22, "freq": 0.3, "lift": -0.16})   # negative y = rise
	_sys.add_force("spring", {"k": 1.4})
	_sys.add_force("drag", {"k": 1.0})
	return {"hue": base_hue}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05, 0.04, 0.08)
	_sys.step(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var bright := 0.5 + 0.5 * _f.energy
	for p: Particle in _sys.particles:
		var c := p.pos() * u
		var r := p.radius * u * (1.0 + 2.0 * _f.beat)
		var col := Color.from_hsv(p.hue, 0.6, bright)
		draw_circle(c, r * 2.2, Color(col.r, col.g, col.b, 0.12))   # soft glow
		draw_circle(c, r, col)
