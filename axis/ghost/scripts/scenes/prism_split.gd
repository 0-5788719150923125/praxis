extends GhostScene

## Prism split - one prism strains, then breaks into two (from "the-point").
##
## It begins as a SINGLE blue [PrismBody] at centre. Energy builds a TENSION: a red clone is pulled
## out of it as a faint SHADE, stretching toward its anchor but held back by an attractor BOND
## between them (a taut filament). When the tension crosses its breaking point, the bond SNAPS - a
## flash - and the freed clone SPRINGS to its anchor with an overshoot, both prisms settling to the
## left/right anchors. Thereafter each prism strains against its own anchor (the rubber-band core in
## [PrismBody]), so the pair keeps straining and springing with the music. The whole split is driven
## by energy: a quiet passage barely stretches it; a surge breaks it.

var _f: AudioFeatures = AudioFeatures.new()
var _blue: PrismBody
var _red: PrismBody
var _tension := 0.0        # attractor strain: builds with energy, breaks the bond at >= 1
var _broke := false
var _bx := 0.0             # blue centre, in anchor fractions (-1 = left anchor)
var _rx := 0.0             # red centre, in anchor fractions (+1 = right anchor)
var _bxv := 0.0
var _rxv := 0.0
var _rop := 0.0            # red opacity (a faint shade before the break, full after)
var _snap := 0.0           # break-flash envelope


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "field"
	_blue = PrismBody.new(rng.randi())
	_red = PrismBody.new(rng.randi())
	return {"radius": rng.randf_range(0.26, 0.34)}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.02)
	var drive := clampf(f.energy * 0.85 + f.beat * 0.6, 0.0, 1.0)
	_blue.update(delta, drive)
	_red.update(delta, drive)
	# While still coupled, the red is a SHADE of the same prism: phase-lock it to the blue so the two
	# spin as ONE. It only starts turning on its own once the bond breaks and it becomes real.
	if not _broke:
		_red.lock_pose_to(_blue)
	var follow := 1.0 - exp(-7.0 * delta)
	if not _broke:
		# Tension accumulates with energy; a surge tips it past the breaking point.
		_tension = clampf(_tension + delta * (0.05 + 0.6 * drive), 0.0, 1.15)
		var t := smoothstep(0.0, 1.0, _tension)
		_bx = lerpf(_bx, -0.05 * _tension, follow)     # the original strains a little
		_rx = lerpf(_rx, 0.5 * t, follow)              # the shade is pulled out, held by the bond
		_rop = lerpf(_rop, 0.08 + 0.3 * t, follow)     # ... a faint ghost, growing
		if _tension >= 1.0:
			_broke = true
			_snap = 1.0
			_bxv = -1.6                                # recoil kick
			_rxv = 3.0                                 # the freed clone springs out hard
	else:
		# Post-break springs: settle at the symmetric anchors with a little overshoot.
		_bxv += (-1.0 - _bx) * 95.0 * delta
		_bxv *= exp(-11.0 * delta)
		_bx += _bxv * delta
		_rxv += (1.0 - _rx) * 110.0 * delta
		_rxv *= exp(-11.0 * delta)
		_rx += _rxv * delta
		_rop = minf(1.0, _rop + delta * 3.5)           # the clone becomes fully real
	_snap = maxf(0.0, _snap - delta * 2.2)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var anchor := 0.32 * u
	var spread := clampf(maxf(absf(_bx), absf(_rx)), 0.0, 1.0)
	var sc := float(params.radius) * u * (1.0 - 0.12 * spread)   # shrink a touch as they part
	var bc := Vector2(_bx * anchor, 0.0)
	var rc := Vector2(_rx * anchor, 0.0)
	# The attractor bond: a taut filament from the original to the emerging clone - brightening as
	# the tension winds up and FLASHING white as it snaps.
	if _tension > 0.02 or _snap > 0.01:
		_draw_bond(bc, rc, clampf(0.1 * _tension + 0.95 * _snap, 0.0, 1.0))
	_blue.draw(self, bc, sc, 0.6, 1.0)
	_red.draw(self, rc, sc, 0.0, clampf(_rop, 0.0, 1.0))


# The stretching attractor bond between the two centres: a slightly bowed, glowing filament whose
# brightness `k` tracks the tension (and spikes on the snap).
func _draw_bond(a: Vector2, b: Vector2, k: float) -> void:
	if k <= 0.001:
		return
	var mid := (a + b) * 0.5 + Vector2(0.0, 5.0 * sin(_life * 2.5))
	var pts := PackedVector2Array([a, mid, b])
	draw_polyline(pts, Color(0.7, 0.85, 1.0, k * 0.3), 8.0, true)
	draw_polyline(pts, Color(1.0, 1.0, 1.0, k * 0.85), 2.0, true)
