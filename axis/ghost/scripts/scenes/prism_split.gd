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
##
## Colour and geometry both come off one roll. The pair takes a [Scheme] mood - the
## original on its base hue, the clone on the mood's counter hue (see [method
## Scheme.opposed]) - so "blue splits into red" is now one outcome of many rather than
## the only one. The structural liberty is the SPLIT ITSELF: which way the thing comes
## apart, which is this scene's whole composition.

## How the prism comes apart. `angle` is the axis the two halves separate along (0 =
## sideways), `throw` how far each ends up from centre, `size` the body scale that
## still reads at that separation - a tight vertical cleave wants smaller prisms than
## a wide lateral break, so the three travel together rather than being rolled apart.
const SPLIT := {
	"lateral":  {"angle": [0.00, 0.12], "throw": [0.30, 0.38], "size": [0.26, 0.34]},
	"diagonal": {"angle": [0.45, 0.85], "throw": [0.28, 0.34], "size": [0.24, 0.30]},
	"cleave":   {"angle": [1.30, 1.75], "throw": [0.20, 0.27], "size": [0.20, 0.26]},
}

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
var _axis := Vector2.RIGHT # the direction the two halves separate along
var _throw := 0.32         # how far each settles from centre, in unit-fractions
var _lead := 0.6           # the original's hue ..
var _counter := 0.0        # .. and the clone's
var _bond_col := Color(0.7, 0.85, 1.0)   # the taut filament, tinted by the mood


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "field"
	_blue = PrismBody.new(rng.randi())
	_red = PrismBody.new(rng.randi())
	var sch := Scheme.pick(rng)
	_lead = sch.vary(rng)
	_counter = sch.opposed(_lead)
	_bond_col = Color.from_hsv(_lead, 0.25, 1.0)      # a pale filament of the original's colour
	var split := String(SPLIT.keys()[rng.randi() % SPLIT.size()])
	var s: Dictionary = SPLIT[split]
	var ang := _band(s["angle"], rng) * (1.0 if rng.randf() < 0.5 else -1.0)   # mirrored half the time
	_axis = Vector2(cos(ang), sin(ang))
	_throw = _band(s["throw"], rng)
	return {"radius": _band(s["size"], rng), "mood": sch.name, "split": split,
		"angle": ang, "throw": _throw, "lead": _lead, "counter": _counter,
		"forms": [_blue.form, _red.form]}


# A value from a [lo, hi] table band.
func _band(b, rng: RandomNumberGenerator) -> float:
	var a: Array = b
	return rng.randf_range(float(a[0]), float(a[1]))


# The prism family's second hue: the mood's own accent, never closer to the lead than


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
	var anchor := _throw * u
	var spread := clampf(maxf(absf(_bx), absf(_rx)), 0.0, 1.0)
	var sc := float(params.radius) * u * (1.0 - 0.12 * spread)   # shrink a touch as they part
	# The halves travel along the rolled split axis; -1/+1 stay the two ends of it.
	var bc := _axis * (_bx * anchor)
	var rc := _axis * (_rx * anchor)
	# The attractor bond: a taut filament from the original to the emerging clone - brightening as
	# the tension winds up and FLASHING white as it snaps.
	if _tension > 0.02 or _snap > 0.01:
		_draw_bond(bc, rc, clampf(0.1 * _tension + 0.95 * _snap, 0.0, 1.0))
	_blue.draw(self, bc, sc, _lead, 1.0)
	_red.draw(self, rc, sc, _counter, clampf(_rop, 0.0, 1.0))


# The stretching attractor bond between the two centres: a slightly bowed, glowing filament whose
# brightness `k` tracks the tension (and spikes on the snap).
func _draw_bond(a: Vector2, b: Vector2, k: float) -> void:
	if k <= 0.001:
		return
	# The bow is PERPENDICULAR to the split axis, so a vertical cleave sags sideways
	# rather than along its own line (where the bow would be invisible).
	var perp := Vector2(-_axis.y, _axis.x)
	var mid := (a + b) * 0.5 + perp * (5.0 * sin(_life * 2.5))
	var pts := PackedVector2Array([a, mid, b])
	draw_polyline(pts, Color(_bond_col.r, _bond_col.g, _bond_col.b, k * 0.3), 8.0, true)
	draw_polyline(pts, Color(1.0, 1.0, 1.0, k * 0.85), 2.0, true)
