extends Control
class_name DialWidget

## DialWidget - the on-screen face of a [Dial] (see that class for what turning does).
##
## A drawn instrument, not a themed knob: an outer ring with this revolution's wedge
## ticks, a needle at the current phase, a hub that glows with the total deposited
## energy, and the turn counter. Drag anywhere on it to rotate (the needle follows
## your angle around the centre); the mouse wheel steps it finely. The widget reads
## the Director's primary dial every frame, so it survives session changes.

const WHEEL_STEP := 0.11          # radians per wheel notch

var dial_index := 0
var _dragging := false
var _last_angle := 0.0            # pointer angle at the previous drag event


func _ready() -> void:
	custom_minimum_size = Vector2(132, 132)
	mouse_filter = Control.MOUSE_FILTER_STOP


func _process(_delta: float) -> void:
	queue_redraw()


func _dial() -> Dial:
	return Director.dial(dial_index)


func _gui_input(event: InputEvent) -> void:
	var d := _dial()
	if d == null:
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			d.turn(WHEEL_STEP)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			d.turn(-WHEEL_STEP)
			accept_event()
		elif mb.button_index == MOUSE_BUTTON_LEFT:
			_dragging = mb.pressed
			if _dragging:
				_last_angle = _pointer_angle(mb.position)
			accept_event()
	elif event is InputEventMouseMotion and _dragging:
		var a := _pointer_angle((event as InputEventMouseMotion).position)
		var delta := wrapf(a - _last_angle, -PI, PI)     # shortest way around
		_last_angle = a
		d.turn(delta)
		accept_event()


func _pointer_angle(p: Vector2) -> float:
	var c := size * 0.5
	return (p - c).angle()


func _draw() -> void:
	var d := _dial()
	var c := size * 0.5
	var r := minf(size.x, size.y) * 0.5 - 6.0
	if d == null:
		draw_arc(c, r, 0, TAU, 48, Color(0.3, 0.34, 0.4, 0.4), 2.0, true)
		return
	var glow := clampf(d.glow() * 0.22, 0.0, 1.0)
	var spin := clampf(d.spin() * 0.8, 0.0, 1.0)

	# Hub glow: the deposited energy, breathing a little brighter while turning.
	var hub := Color(0.55, 0.7, 1.0, 0.10 + 0.5 * glow + 0.25 * spin)
	draw_circle(c, r * (0.30 + 0.16 * glow), hub)
	draw_circle(c, r * 0.16, Color(0.85, 0.92, 1.0, 0.25 + 0.6 * glow))

	# Ring.
	draw_arc(c, r, 0, TAU, 64, Color(0.5, 0.58, 0.72, 0.55 + 0.3 * spin), 2.0, true)

	# This revolution's wedge ticks (5 or 6 - the transformation zones).
	var turn_i := d.turn_count()
	var n := d.wedges_of(turn_i)
	var cur := d.wedge()
	for w in n:
		var a0 := TAU * float(w) / float(n)
		var p0 := c + Vector2(cos(a0), sin(a0)) * (r - 7.0)
		var p1 := c + Vector2(cos(a0), sin(a0)) * (r + 1.0)
		var lit := w == cur
		draw_line(p0, p1, Color(0.9, 0.95, 1.0, 0.9) if lit else Color(0.5, 0.58, 0.72, 0.5),
			2.5 if lit else 1.5, true)

	# Needle at the phase.
	var na := d.phase() * TAU
	var nd := Vector2(cos(na), sin(na))
	draw_line(c + nd * r * 0.30, c + nd * (r - 9.0), Color(1, 1, 1, 0.9), 2.0, true)
	draw_circle(c + nd * (r - 9.0), 3.0, Color(1, 1, 1, 0.9))

	# Turn counter, below the hub.
	var font := ThemeDB.fallback_font
	var txt := str(turn_i)
	var ts := font.get_string_size(txt, HORIZONTAL_ALIGNMENT_CENTER, -1, 13)
	draw_string(font, c + Vector2(-ts.x * 0.5, r * 0.72), txt,
		HORIZONTAL_ALIGNMENT_CENTER, -1, 13, Color(0.6, 0.68, 0.8, 0.85))
