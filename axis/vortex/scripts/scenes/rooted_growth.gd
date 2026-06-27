extends VortexScene

## Rooted growth - trees / roots that grow, twist, and contract.
##
## A handful of arms radiate from the center, each a recursive branching
## structure. A growth value rises sharply on beats and bass hits, then decays
## back toward zero - so the tree surges outward into complexity on every pulse
## and relaxes to a bare stub between them. New branch levels are *born* as the
## growth crosses each depth, easing in by length and brightness, with a tip bud
## glowing at the live edge. A modulator twists the whole structure as it grows,
## so no two surges trace the same shape.

var _f: AudioFeatures = AudioFeatures.new()
var _growth := 0.0      # 0..1 envelope: attack on hits, slow decay
var _twist := 0.0       # organic swirl applied along the branches
var _hue := 0.0
var _bud := Color.WHITE


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	return {
		"arms": rng.randi_range(2, 5),
		"max_depth": rng.randi_range(5, 7),
		"ratio": rng.randf_range(0.62, 0.78),     # child length / parent length
		"spread": rng.randf_range(0.40, 0.85),    # base fork half-angle
		"base_len": rng.randf_range(0.12, 0.19),  # root length, fraction of unit
		"decay": rng.randf_range(0.30, 0.65),     # growth lost per second
		"rest": rng.randf_range(0.22, 0.40),      # never contracts below this
		"hue": rng.randf(),
		"hue_depth": rng.randf_range(-0.06, 0.10), # hue shift per level
		"width0": rng.randf_range(3.5, 6.0),
		"twist_amt": rng.randf_range(0.3, 1.0),
		"third": rng.randf() < 0.5,               # an occasional middle branch
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.04, 0.06, 0.05, 0.10)

	# Attack on transients, exponential-ish decay back to simplicity.
	var attack := clampf(f.beat * 0.7 + f.bass * 0.45 + f.energy * 0.2, 0.0, 1.0)
	_growth = maxf(_growth, attack)
	_growth = maxf(0.0, _growth - delta * float(params.decay))
	# Contract back toward a small resting tree, never to bare nothing - so a cut
	# to this scene always shows structure instead of a black frame.
	_growth = maxf(_growth, float(params.rest))

	_twist = mod.value("twist") * float(params.twist_amt) + 0.4 * mod.value("swirl")
	_hue = float(params.hue)
	_bud = Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.3, 1.0, 0.9)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var arms := int(params.arms)
	var base_len: float = unit() * float(params.base_len) * (0.7 + 0.6 * _f.energy)
	var budget := _growth * float(params.max_depth)
	for a in arms:
		var ang := -PI * 0.5 + TAU * float(a) / float(arms) + 0.3 * mod.value("sway")
		_branch(Vector2.ZERO, ang, base_len, 0, budget)


# Recursive branch. `budget` is the live growth depth; a level fades/extends in
# as the budget crosses it, and everything past the budget is simply not drawn.
func _branch(p: Vector2, ang: float, len: float, depth: int, budget: float) -> void:
	var a := clampf(budget - float(depth) + 1.0, 0.0, 1.0)
	if a <= 0.0:
		return
	var dir := Vector2(cos(ang), sin(ang))
	var tip := p + dir * len * a
	var w := maxf(1.0, float(params.width0) * pow(0.7, depth))
	var h := fposmod(_hue + float(params.hue_depth) * depth, 1.0)
	var col := Color.from_hsv(h, 0.6, 0.45 + 0.55 * _f.bass, a)
	draw_line(p, tip, col, w, true)

	# A glowing bud at the live edge (the level currently being born).
	if a < 1.0:
		draw_circle(tip, w * 0.9, Color(_bud.r, _bud.g, _bud.b, _bud.a * a))

	if depth >= int(params.max_depth) or len < 3.0:
		return
	var sp: float = float(params.spread) + _twist * 0.12 * float(depth)
	var cl := len * float(params.ratio)
	var lean := _twist * 0.1
	_branch(tip, ang - sp + lean, cl, depth + 1, budget)
	_branch(tip, ang + sp + lean, cl, depth + 1, budget)
	if bool(params.third) and depth % 2 == 0:
		_branch(tip, ang + lean, cl * 0.8, depth + 1, budget)
