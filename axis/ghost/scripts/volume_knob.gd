extends Control
class_name VolumeKnob

## A pull-rope volume control. Click-and-HOLD the knob, then drag AWAY from it: a rope
## stretches from the knob (the anchor) to your cursor, and the farther you pull the
## louder it gets - but the ceiling is ASYMPTOTIC (v = 1 - e^(-d/D0)), so "max" is
## approached, never reached. Let go and the level is PEGGED exactly where you left it -
## no ongoing decay, ever; the value only ever changes while you are actively pulling -
## see feedback/0005. Releasing within _RELEASE_SNAP_EPS of true 0 snaps to exact 0, as a
## floating-point safety margin on top of the geometric fix below - see feedback/0003.
## Volume is an energy you pull out, continuous 0..1, not a two-state toggle. Read/
## written through get_v/set_v so one knob drives a track's volume or the main clip's.
## The rope tracks the cursor anywhere on screen (motion is caught via _input, not just
## _gui_input), drawn on top via a raised z_index.
##
## The fill split (the "internal horizon" between full and empty) stays perpendicular to
## a pull axis - the empty side always faces out along that axis, the full side always
## faces back - so the control reads correctly when pulled from any angle, not just
## top-down - see feedback/0003. That axis is FROZEN for the duration of any active pull
## (grabbed from whatever the resting ambient direction was the instant you clicked) so it
## never spirals along with your own drag - see feedback/0007. At rest (not pulling) it
## instead eases toward the cursor through a VERY long EMA (see _AMBIENT_TAU), same as the
## fill depth, so rapid mouse movement barely registers and it reads as ambient drift
## rather than the control spinning to track the cursor - see feedback/0008, feedback/0006,
## feedback/0005. This drift is purely visual: the depth only ever eats into the knob's own
## HEADROOM (v + ema * (1 - v)), never past the true level and never written back through
## set_v, so a knob pegged at 100% shows no animation at all - the feature only reveals
## itself on knobs that aren't already maxed.
##
## Volume itself is a SIGNED push/pull along that same frozen axis, not a raw radial
## distance from the anchor - see feedback/0007. Pulling further out along the axis (away
## from the anchor) raises it, asymptotically; pushing back in along the axis (toward and
## past the anchor) lowers it, reaching exactly 0 anywhere at or beyond the anchor on that
## line - not just a single pixel-perfect point at the anchor's center, which is what made
## 0% nearly unreachable before.

var get_v: Callable
var set_v: Callable
var accent := Color(0.6, 0.95, 0.7)
## Draw a speaker glyph inside the block - reserved for the main track's knob only
## (a one-time visual hint; other lanes stay unlabeled so exploring them is how the
## user learns what the control is - see feedback/0008).
var show_icon := false

const _D0 := 90.0    # pixels of pull for ~63% - the "feel" scale of the rope
const _AMBIENT_TAU := 18.0   # seconds - how slowly the resting horizon eases toward the cursor
const _RELEASE_SNAP_EPS := 0.03   # release-time pull below this snaps to exact 0 - keeps 0 reachable
var _pulling := false
var _anchor := Vector2.ZERO   # knob centre in global (screen) space, fixed at grab
var _ambient_ema := 0.0      # 0..1, eased fraction of headroom the resting horizon eats into
var _ambient_dir := Vector2.UP   # eased split-axis direction for the resting horizon (see class doc)
var _pull_axis := Vector2.UP    # the horizon direction, frozen for the duration of an active pull


func _ready() -> void:
	custom_minimum_size = Vector2(22, 18)
	mouse_filter = Control.MOUSE_FILTER_STOP
	tooltip_text = "Volume: hold and pull away to set (max is approached, never reached; push back toward the knob for 0)"


func _process(delta: float) -> void:
	var vp := get_viewport()
	if vp == null:
		return
	var vh: float = maxf(vp.get_visible_rect().size.y, 1.0)
	var target := clampf(1.0 - vp.get_mouse_position().y / vh, 0.0, 1.0)
	var alpha := 1.0 - exp(-delta / _AMBIENT_TAU)
	_ambient_ema = lerpf(_ambient_ema, target, alpha)
	_ambient_dir = _ambient_dir.lerp(_cursor_dir(), alpha).normalized()
	queue_redraw()


func _v() -> float:
	return clampf(float(get_v.call()), 0.0, 1.0) if get_v.is_valid() else 0.0


## Pull distance (px) -> volume. Asymptotic: large pulls approach 1 but never reach it.
func _v_from_pull(d: float) -> float:
	return clampf(1.0 - exp(-d / _D0), 0.0, 0.999)


## Signed pull (px) of the cursor along the frozen `_pull_axis`, measured from `_anchor`.
## Positive = pulled out along the horizon (louder); zero or negative = at or past the
## anchor along that line (silent) - a whole half-plane reaches exactly 0, not one point.
func _signed_pull(cursor: Vector2) -> float:
	return (cursor - _anchor).dot(_pull_axis)


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		_pulling = true
		_anchor = global_position + size * 0.5
		_pull_axis = _ambient_dir   # freeze the horizon for this whole drag - see feedback/0007
		z_index = 200          # the rope draws over the video/lanes while pulling
		queue_redraw()
		accept_event()


func _input(event: InputEvent) -> void:
	if not _pulling:
		return
	if event is InputEventMouseMotion:
		if set_v.is_valid():
			set_v.call(_v_from_pull(maxf(_signed_pull(get_global_mouse_position()), 0.0)))
		queue_redraw()
		get_viewport().set_input_as_handled()
	elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and not event.pressed:
		_pulling = false
		if set_v.is_valid():
			var pv := _v_from_pull(maxf(_signed_pull(get_global_mouse_position()), 0.0))
			if pv < _RELEASE_SNAP_EPS:
				set_v.call(0.0)
		z_index = 0
		queue_redraw()
		get_viewport().set_input_as_handled()


func _draw() -> void:
	var v := _v()
	# The resting horizon: the true level PLUS a slow ambient drift into whatever
	# headroom is left (see class doc) - nil effect once v is pegged near 1.0.
	var vd := clampf(v + _ambient_ema * (1.0 - v), 0.0, 1.0)
	var c := size * 0.5
	var dir := _pull_axis if _pulling else _ambient_dir
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.05, 0.06, 0.08, 0.85))
	if vd > 0.001:
		var full := _fill_polygon(c, dir, vd)
		if full.size() >= 3:
			draw_colored_polygon(full, Color(accent.r, accent.g, accent.b, 0.85))
	draw_rect(Rect2(Vector2.ZERO, size), Color(accent.r, accent.g, accent.b, 0.9), false, 1.0)
	if show_icon:
		_draw_speaker_icon()
	if not _pulling:
		return
	# The rope: from the knob centre to the cursor, thickening/brightening as it's pulled
	# louder, a grabbed end that swells with the level, and a live readout.
	var m := get_local_mouse_position()
	var pv := _v_from_pull(maxf(_signed_pull(get_global_mouse_position()), 0.0))
	draw_line(c, m, Color(accent.r, accent.g, accent.b, 0.45 + 0.5 * pv), 1.0 + 3.0 * pv)
	draw_circle(m, 3.0 + 9.0 * pv, Color(accent.r, accent.g, accent.b, 0.92))
	draw_string(ThemeDB.fallback_font, m + Vector2(11, -9), "%d%%" % int(round(pv * 100.0)),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(1, 1, 1, 0.95))


## Direction from the knob's centre to the cursor (global space, unit length) - the axis
## the fill split stays perpendicular to. Falls back to straight up if the cursor sits
## exactly on the centre (normalize guard).
func _cursor_dir() -> Vector2:
	var d := get_global_mouse_position() - (global_position + size * 0.5)
	return d.normalized() if d.length_squared() > 0.0001 else Vector2.UP


## The "full" region of the box: the rect clipped to the half-plane away from the
## cursor, cut by a line perpendicular to dir positioned so the full side's extent
## along dir grows from nothing (vd = 0) to the whole box (vd = 1). This is the
## "internal horizon" - see class doc.
func _fill_polygon(c: Vector2, dir: Vector2, vd: float) -> PackedVector2Array:
	var reach := absf(dir.x) * c.x + absf(dir.y) * c.y
	var threshold := reach * (2.0 * vd - 1.0)
	var poly := PackedVector2Array([-c, Vector2(c.x, -c.y), c, Vector2(-c.x, c.y)])
	var out := PackedVector2Array()
	var n := poly.size()
	for i in n:
		var cur: Vector2 = poly[i]
		var prev: Vector2 = poly[(i - 1 + n) % n]
		var cur_s := cur.dot(dir)
		var prev_s := prev.dot(dir)
		var cur_in := cur_s <= threshold
		if cur_in != (prev_s <= threshold):
			var t := (threshold - prev_s) / (cur_s - prev_s)
			out.append(prev.lerp(cur, t) + c)
		if cur_in:
			out.append(cur + c)
	return out


## A small speaker glyph (body + cone + sound arcs) centred in the block - the one-time
## "this is audio" hint for the main track's knob only (see show_icon / feedback/0008).
func _draw_speaker_icon() -> void:
	var h := size.y
	var col := Color(1.0, 1.0, 1.0, 0.85)
	draw_rect(Rect2(3.0, h * 0.375, 3.0, h * 0.25), col)
	draw_colored_polygon(PackedVector2Array([
		Vector2(6.0, h * 0.375), Vector2(6.0, h * 0.625),
		Vector2(11.0, h * 0.85), Vector2(11.0, h * 0.15),
	]), col)
	draw_arc(Vector2(11.0, h * 0.5), 3.5, -0.7, 0.7, 8, col, 1.0)
	draw_arc(Vector2(11.0, h * 0.5), 6.0, -0.6, 0.6, 8, col, 1.0)
