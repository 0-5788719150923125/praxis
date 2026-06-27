extends VortexScene

## Shatter glass - a pane that fractures and bursts on the beat.
##
## Geometry: a disc or rectangular pane fractured into irregular angular shards by
## recursive splitting (see [Geo]), with cracks biased to radiate from an impact
## point - planes and fragments, not pizza slices. Physics: composed from the
## primitive registry. The shards burst on the beat and spin hard. Lifecycle is
## seeded:
##   loop    - scatter + spring + drag: burst out, get pulled home, settle, repeat.
##   oneshot - scatter(once) + drag: burst out, spinning, drift to rest, finished().

var _f: AudioFeatures = AudioFeatures.new()
var _sys: ParticleSystem
var _moved := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	_sys = ParticleSystem.new()
	lifecycle = "oneshot" if rng.randf() < 0.4 else "loop"

	# Usually one elegant pane; sometimes a few smaller ones so a lone plane isn't
	# stranded. Each pane fractures independently and bursts from its own center.
	var pane_count := 1 if rng.randf() < 0.6 else rng.randi_range(2, 3)
	var base_hue := rng.randf()
	for pane in pane_count:
		var pane_center := Vector2.ZERO
		var radius := rng.randf_range(0.30, 0.46)
		if pane_count > 1:
			pane_center = Vector2(rng.randf_range(-0.28, 0.28), rng.randf_range(-0.24, 0.24))
			radius = rng.randf_range(0.16, 0.26)
		var base := PackedVector2Array()
		if rng.randf() < 0.5:
			var n := rng.randi_range(10, 16)             # disc
			for i in n:
				var a := TAU * float(i) / float(n)
				base.append(Vector2(cos(a), sin(a)) * radius)
		else:
			var hw := radius * rng.randf_range(0.8, 1.1)  # rectangular pane
			var hh := radius * rng.randf_range(0.6, 1.0)
			base = PackedVector2Array([
				Vector2(-hw, -hh), Vector2(hw, -hh), Vector2(hw, hh), Vector2(-hw, hh)])
		var impact := Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1)) * radius * 0.4
		var shards := Geo.fracture(base, rng.randi_range(14, 30), impact, radius * 0.12, rng)
		for poly: PackedVector2Array in shards:
			var cen := Geo.centroid(poly)
			var local := PackedVector2Array()
			for v in poly:
				local.append(v - cen)
			var p := Particle.new()
			p.home = pane_center + cen
			p.poly = local
			p.hue = fposmod(base_hue + 0.3 * cen.length(), 1.0)
			p.noise = Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1))
			p.nspin = rng.randf_range(-1, 1)
			p.data = {"center": pane_center}
			_sys.add(p)

	if lifecycle == "loop":
		_sys.add_force("scatter", {"strength": 0.55, "jitter": 0.5, "spin": 2.2, "period": 3.5})
		_sys.add_force("spring", {"k": 2.6})
		_sys.add_force("drag", {"k": 1.3})
	else:
		_sys.add_force("scatter", {"strength": 0.7, "jitter": 0.6, "spin": 3.0, "once": true})
		_sys.add_force("drag", {"k": 0.7})
	return {}


func finished() -> bool:
	return lifecycle == "oneshot" and _moved and _sys.settled()


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05, 0.04, 0.08)
	_sys.step(f, delta)
	if not _sys.settled():
		_moved = true
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	for p in _sys.particles:
		var poly := _sys.world_poly(p, u)
		var scatter := clampf(p.off.length() * 3.0, 0.0, 1.0)
		var val := 0.35 + 0.5 * _f.energy + 0.15 * (1.0 - scatter)
		draw_colored_polygon(poly, Color.from_hsv(p.hue, 0.45, val, 0.85))
		var edge := poly.duplicate()
		edge.append(poly[0])
		draw_polyline(edge, Color.from_hsv(p.hue, 0.2, 0.9, 0.45 + 0.4 * scatter), 1.0, true)
