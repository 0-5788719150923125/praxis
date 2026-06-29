extends RefCounted
class_name Swarm

## Swarm - a scalar field over a grid that evolves by *local* rules.
##
## The transferable mechanism behind many-item scenes. Instead of scripting each of
## thousands of items, you seed a field and let it spread: development creeps
## outward from a few origins (a city growing over a countryside), or injected
## pulses diffuse across the lattice (colour rippling through the blocks). Cellular,
## seeded, cheap - and the same field drives any abstract grid of many items.
## (Pheromone trails / ant-colony rules are a natural next rule to add here.)

const GROW := 0   # development spreads from seeds via logistic neighbor growth, and stays
const WAVE := 1   # injected pulses diffuse outward and decay (colour fronts)

var w: int
var h: int
var f: PackedFloat32Array     # the field, row-major
var _g: PackedFloat32Array    # double buffer
var rule: int
var _seeds: PackedInt32Array  # cells pinned to 1.0 (growth origins)


func _init(width: int, height: int, rule_: int, rng: RandomNumberGenerator, seed_count := 3) -> void:
	w = width
	h = height
	rule = rule_
	var n := w * h
	f = PackedFloat32Array()
	f.resize(n)
	_g = PackedFloat32Array()
	_g.resize(n)
	_seeds = PackedInt32Array()
	if rule == GROW:
		for s in seed_count:
			var x := rng.randi_range(int(w * 0.25), int(w * 0.75))
			var y := rng.randi_range(int(h * 0.25), int(h * 0.75))
			var i := y * w + x
			f[i] = 1.0
			_seeds.append(i)


func at(x: int, y: int) -> float:
	return f[y * w + x]


func inject(x: int, y: int, v: float) -> void:
	if x < 0 or x >= w or y < 0 or y >= h:
		return
	var i := y * w + x
	f[i] = maxf(f[i], v)


func step(drive: float, dt: float, decay := 0.0) -> void:
	match rule:
		GROW:
			_step_grow(drive, dt, decay)
		WAVE:
			_step_wave(dt, decay)


# Logistic spread: a cell rises toward 1 at a rate set by its developed neighbors
# (so development creeps outward from the seeds), accelerated by the audio drive.
func _step_grow(drive: float, dt: float, decay: float) -> void:
	var rate := (0.3 + drive) * 1.4 * dt
	for y in h:
		for x in w:
			var i := y * w + x
			var na := _neighbor_avg(x, y)
			var grow := rate * na * (1.0 - f[i])
			_g[i] = clampf(f[i] + grow - decay * dt * f[i], 0.0, 1.0)
	for s in _seeds:
		_g[s] = 1.0
	_swap()


# Diffuse the field outward and fade it - injected pulses become expanding rings.
func _step_wave(dt: float, decay: float) -> void:
	var keep := exp(-decay * dt)
	for y in h:
		for x in w:
			var i := y * w + x
			_g[i] = (_neighbor_avg(x, y) * 0.72 + f[i] * 0.28) * keep
	_swap()


func _neighbor_avg(x: int, y: int) -> float:
	var s := 0.0
	var c := 0
	if x > 0:
		s += f[y * w + x - 1]; c += 1
	if x < w - 1:
		s += f[y * w + x + 1]; c += 1
	if y > 0:
		s += f[(y - 1) * w + x]; c += 1
	if y < h - 1:
		s += f[(y + 1) * w + x]; c += 1
	return s / float(maxi(1, c))


func _swap() -> void:
	var n := f.size()
	for i in n:
		f[i] = _g[i]
