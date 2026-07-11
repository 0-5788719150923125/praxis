extends Control
class_name TrackLane

## TrackLane - one clip's block on the shared timeline: drag its LEFT/RIGHT edges
## to trim (in/out points), drag its BODY to shift where it sits on the master
## timeline. Used for both the primary clip's own trim block (offset fixed at 0 -
## it IS master time, so `movable` is false there) and one lane per imported
## track (offset draggable). Every lane shares the same TimelineView as the
## marker strip below them (see mask_timeline.gd), so a second lines up
## identically in every row - and, critically, dragging THIS lane's own edges
## never touches that shared view's cached extent (see TimelineView.refresh):
## the ruler a trim handle is measured against never moves under it mid-drag.
##
## Backed by Callables, not a fixed field name - a lane doesn't care whether
## it's reading/writing session.clip_in/clip_out (the primary) or one
## session.tracks[i] dict (a secondary track); the caller wires whichever.

signal changed
## Fired once per drag gesture, BEFORE the first mutation - the undo hook,
## mirroring MaskTimeline.marker_drag_started.
signal drag_started
## Fired once a drag gesture ENDS (mouse-up) - the ONLY safe point to refresh
## the shared TimelineView's cached extent (see TimelineView.refresh): doing it
## mid-drag would rescale the very ruler this lane's handle is being measured
## against, while it's still being dragged.
signal drag_ended

var tview: TimelineView = null
var label := ""
var color := Color(0.5, 0.7, 1.0)
var movable := true         # false for the primary lane - its offset is always 0
var full_duration := 0.0    # the SOURCE clip's own full length; trim can't exceed it
## Px of the lane's left edge already occupied by another overlaid control (the
## volume knob mask_editor.gd anchors top-left on every lane) - the label's start
## x is pushed past this so it never draws underneath that knob, whatever the
## block's own left edge happens to be (offset 0 puts them right on top of each
## other - see feedback/0009).
var reserved_left := 0.0

var get_in: Callable
var set_in: Callable
var get_out: Callable
var set_out: Callable
var get_offset: Callable = func(): return 0.0
var set_offset: Callable = Callable()   # invalid = shift disabled (the primary lane)
## Fade envelope (seconds in from each trimmed edge). The fade BOUNDARY is a second
## handle at each edge - see the class doc / _hit. Invalid = fades disabled.
var get_fade_in: Callable = Callable()
var set_fade_in: Callable = Callable()
var get_fade_out: Callable = Callable()
var set_fade_out: Callable = Callable()
## Master-timeline seconds to snap a dragged clip's start/end to (other clips' edges,
## the playhead, 0). Invalid = no snapping.
var get_snap_targets: Callable = Callable()
## Current master-timeline playhead position, in seconds. Invalid = no playhead drawn
## (the primary lane skips this - its own timestamp already sits right below in the
## marker strip; this is for secondary tracks, whose OWN source position at the shared
## playhead isn't otherwise visible - see feedback/0007).
var get_playhead: Callable = Callable()

const _EDGE_W := 8.0
const _SNAP_PX := 8.0   # a dragged clip edge within this many px of a target snaps to it
# A fade handle sitting exactly on top of its trim handle (fade == 0) used to be
# separated by a vertical hit BAND - a sliver in the top ~22% of a 26px-tall lane.
# That made the fade handle nearly unhittable (had to land inside a ~16x6px box) and
# an ordinary click anywhere else on that edge silently grabbed the trim handle and
# resized instead - see feedback/0011. Replaced with a horizontal nudge instead: a
# coincident fade handle draws/hits _HANDLE_GAP px in from the trim edge, so the two
# are always two adjacent, full-height, independently-grabbable targets (matching the
# original two-markers-stacked spec) rather than one pixel gated behind a vertical band.
const _HANDLE_GAP := 7.0
var _drag_mode := ""   # "" | "in" | "out" | "fade_in" | "fade_out" | "body"
var _drag_started_emitted := false
var _body_grab_dx := 0.0


func _ready() -> void:
	custom_minimum_size = Vector2(0, 26)
	mouse_filter = Control.MOUSE_FILTER_STOP


func _process(_dt: float) -> void:
	# Hide the whole row - not just skip drawing the clip block, which left an
	# empty strip on screen - once its clip's span is entirely outside the
	# current zoom/pan window: there's nothing left to look at or grab, and the
	# VBoxContainer collapses the reclaimed space automatically. Same bounds
	# check _draw already uses to skip the block itself, just promoted to gate
	# the row's visibility too - see feedback/0023.
	if tview != null:
		var b := _bounds()
		visible = tview.x_of(b.y) >= 0.0 and tview.x_of(b.x) <= size.x
	queue_redraw()


func _gui_input(event: InputEvent) -> void:
	if tview == null:
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_drag_mode = _hit(mb.position)
				_drag_started_emitted = false
				if _drag_mode == "body":
					_body_grab_dx = mb.position.x - tview.x_of(float(get_offset.call()))
				if _drag_mode != "":
					accept_event()
			else:
				var was_dragging := _drag_mode != ""
				_drag_mode = ""
				if was_dragging and _drag_started_emitted:
					drag_ended.emit()
				accept_event()
	elif event is InputEventMouseMotion and _drag_mode != "":
		if not _drag_started_emitted:
			_drag_started_emitted = true
			drag_started.emit()
		_apply_drag(event.position.x)
		accept_event()


func _fade_in_s() -> float:
	return maxf(0.0, float(get_fade_in.call())) if get_fade_in.is_valid() else 0.0


func _fade_out_s() -> float:
	return maxf(0.0, float(get_fade_out.call())) if get_fade_out.is_valid() else 0.0


## The clip's [left, right] edges in MASTER-timeline seconds. A movable (secondary)
## lane sits at its own independently-placed `offset`, unaffected by where clip_in/
## clip_out happen to fall in ITS OWN source's timeline. The fixed primary lane has
## no such independent placement - clip_in/clip_out already ARE master time (offset
## is always 0, non-negotiable) - so its bounds are the trim points directly. Every
## hit-test/draw/drag below reads through this: using `offset` alone for the primary
## pinned its left edge to time 0 regardless of clip_in, so trimming that edge never
## visibly moved it - only the untouched right edge shifted in response, which read
## as backwards - see feedback/0023.
func _bounds() -> Vector2:
	var cin: float = get_in.call()
	var cout: float = get_out.call()
	if movable:
		var offset: float = get_offset.call()
		return Vector2(offset, offset + (cout - cin))
	return Vector2(cin, cout)


## The x-position each fade boundary HANDLE draws/hits at. Equal to its real position
## (b.x + fade_in_s / b.y - fade_out_s) once that's dragged out past _HANDLE_GAP, but
## nudged in from a coincident trim edge below that - see the _HANDLE_GAP doc above.
## The underlying fade seconds (and so where the envelope itself actually starts
## fading, drawn separately in _draw) are never touched by this - only where the
## little marker itself draws/hits.
func _fade_handle_xs(x0: float, x1: float, b: Vector2, span: float) -> Vector2:
	var xfi := x0
	var xfo := x1
	if set_fade_in.is_valid():
		xfi = maxf(tview.x_of(b.x + clampf(_fade_in_s(), 0.0, span)), x0 + _HANDLE_GAP)
	if set_fade_out.is_valid():
		xfo = minf(tview.x_of(b.y - clampf(_fade_out_s(), 0.0, span)), x1 - _HANDLE_GAP)
	return Vector2(xfi, xfo)


func _hit(pos: Vector2) -> String:
	var span: float = float(get_out.call()) - float(get_in.call())
	var b := _bounds()
	var x0 := tview.x_of(b.x)
	var x1 := tview.x_of(b.y)
	var fx := _fade_handle_xs(x0, x1, b, span)
	# Whichever handle CENTER is closest wins - full lane height counts now, not just
	# a sliver top band, so a click dead-on the trim edge still grabs the trim edge
	# even though the fade handle sits only _HANDLE_GAP px away from it - see
	# feedback/0011.
	var best_mode := ""
	var best_d := _EDGE_W
	if set_fade_in.is_valid():
		var d: float = absf(pos.x - fx.x)
		if d <= best_d:
			best_d = d
			best_mode = "fade_in"
	if set_fade_out.is_valid():
		var d: float = absf(pos.x - fx.y)
		if d <= best_d:
			best_d = d
			best_mode = "fade_out"
	var d_in: float = absf(pos.x - x0)
	if d_in <= best_d:
		best_d = d_in
		best_mode = "in"
	var d_out: float = absf(pos.x - x1)
	if d_out <= best_d:
		best_d = d_out
		best_mode = "out"
	if best_mode != "":
		return best_mode
	if pos.x > x0 and pos.x < x1 and movable:
		return "body"
	return ""


func _apply_drag(x: float) -> void:
	var t := tview.t_of(x)
	var offset: float = get_offset.call()
	var span: float = float(get_out.call()) - float(get_in.call())
	var b := _bounds()
	match _drag_mode:
		"in":
			var cur_out: float = get_out.call()
			var cur_in: float = get_in.call()
			var delta := 0.0
			if movable:
				# A movable lane's right edge is offset + (clip_out - clip_in) - moving
				# clip_in alone (leaving offset put) shrinks THAT, i.e. drags the RIGHT
				# edge instead of the left one. Shift clip_in and offset by the same
				# delta so the right edge stays put and the left edge tracks the cursor -
				# see feedback/0026.
				delta = t - offset
				var delta_min: float = -minf(cur_in, offset)
				var delta_max: float = cur_out - 0.1 - cur_in
				delta = clampf(delta, delta_min, delta_max)
				set_in.call(cur_in + delta)
				set_offset.call(offset + delta)
			else:
				var new_in := clampf(t - offset, 0.0, cur_out - 0.1)
				delta = new_in - cur_in
				set_in.call(new_in)
			# Rescale the fade, don't just shift it: if a fade boundary is already set,
			# keep ITS absolute position fixed as the trim edge slides past it, so the
			# ramp stretches/shrinks to match instead of translating along with the edge
			# and needing manual recorrection afterward - see feedback/0011. Only kicks
			# in once a fade has actually been dragged out; a fresh (duration 0) edge
			# stays at 0 rather than spontaneously growing a fade.
			if set_fade_in.is_valid() and _fade_in_s() > 0.0:
				set_fade_in.call(clampf(_fade_in_s() - delta, 0.0, maxf(0.0, cur_out - (cur_in + delta))))
		"out":
			var cur_in: float = get_in.call()
			var cur_out: float = get_out.call()
			var new_out: float
			if movable:
				# Left edge (offset, clip_in) stays put; only clip_out follows the cursor.
				new_out = clampf(t - offset + cur_in, cur_in + 0.1, full_duration)
			else:
				new_out = clampf(t - offset, cur_in + 0.1, full_duration)
			set_out.call(new_out)
			# Same automatic rescale as the "in" edge, mirrored - see above.
			if set_fade_out.is_valid() and _fade_out_s() > 0.0:
				var delta_out := new_out - cur_out
				set_fade_out.call(clampf(_fade_out_s() + delta_out, 0.0, maxf(0.0, new_out - cur_in)))
		"fade_in":
			if set_fade_in.is_valid():
				set_fade_in.call(clampf(t - b.x, 0.0, span))
		"fade_out":
			if set_fade_out.is_valid():
				set_fade_out.call(clampf(b.y - t, 0.0, span))
		"body":
			if set_offset.is_valid():
				var raw := _snap_offset(tview.t_of(x - _body_grab_dx), span)
				set_offset.call(clampf(raw, 0.0, maxf(0.0, tview.extent - span)))
	changed.emit()


## Gentle snap: if the dragged clip's START or END lands within _SNAP_PX of any snap
## target (other clip edges, the playhead, 0), align it exactly. Returns the offset.
func _snap_offset(raw: float, span: float) -> float:
	if not get_snap_targets.is_valid():
		return raw
	var tol: float = absf(tview.t_of(_SNAP_PX) - tview.t_of(0.0))   # px threshold, in seconds
	var best := raw
	var best_d := tol
	for tv in get_snap_targets.call():
		var tgt := float(tv)
		var d0 := absf(raw - tgt)                # snap the clip's start
		if d0 < best_d:
			best_d = d0
			best = tgt
		var d1 := absf((raw + span) - tgt)       # snap the clip's end
		if d1 < best_d:
			best_d = d1
			best = tgt - span
	return best


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.09, 0.10, 0.13, 0.94))
	if tview == null:
		return
	var offset: float = get_offset.call()
	var cin: float = get_in.call()
	var cout: float = get_out.call()
	var b := _bounds()
	var x0 := tview.x_of(b.x)
	var x1 := tview.x_of(b.y)
	if x1 < 0.0 or x0 > size.x:
		return
	var xa := clampf(x0, -4.0, size.x + 4.0)
	var xb := clampf(x1, -4.0, size.x + 4.0)
	draw_rect(Rect2(xa, 2, maxf(1.0, xb - xa), size.y - 4), Color(color.r, color.g, color.b, 0.35))
	draw_rect(Rect2(xa, 2, maxf(1.0, xb - xa), size.y - 4), Color(color.r, color.g, color.b, 0.9), false, 2.0)
	# Fade envelope: a triangle darkening the clip from full at the fade boundary down
	# to silence at the trimmed edge (the same envelope drives audio + video). Only the
	# ramp is drawn; the flat middle is left alone.
	var span := cout - cin
	var top := 2.0
	var bot := size.y - 2.0
	var xfi := tview.x_of(b.x + clampf(_fade_in_s(), 0.0, span))
	var xfo := tview.x_of(b.y - clampf(_fade_out_s(), 0.0, span))
	if _fade_in_s() > 0.0 and xfi > x0:
		draw_colored_polygon(PackedVector2Array([Vector2(x0, bot), Vector2(xfi, top), Vector2(xfi, bot)]),
			Color(0.02, 0.02, 0.04, 0.62))
	if _fade_out_s() > 0.0 and xfo < x1:
		draw_colored_polygon(PackedVector2Array([Vector2(xfo, top), Vector2(x1, bot), Vector2(xfo, bot)]),
			Color(0.02, 0.02, 0.04, 0.62))
	# Grip handles at each trimmable edge (drawn even for the immovable primary -
	# in/out are always draggable, only BODY shift is what `movable` gates).
	if x0 >= -_EDGE_W and x0 <= size.x + _EDGE_W:
		draw_rect(Rect2(x0 - 2.5, 2, 5, size.y - 4), color.lightened(0.35))
	if x1 >= -_EDGE_W and x1 <= size.x + _EDGE_W:
		draw_rect(Rect2(x1 - 2.5, 2, 5, size.y - 4), color.lightened(0.35))
	# The SECOND handle at each edge - the fade boundary - as a full-height bright bar,
	# nudged _HANDLE_GAP px in from a coincident trim edge (see _fade_handle_xs) so it
	# always draws as its own distinct, easily-grabbed target instead of a sliver
	# hiding on top of the trim handle - see feedback/0011.
	var fade_col := Color(1.0, 0.95, 0.55, 0.95)
	var fx := _fade_handle_xs(x0, x1, b, span)
	if set_fade_in.is_valid() and fx.x >= -_EDGE_W and fx.x <= size.x + _EDGE_W:
		draw_rect(Rect2(fx.x - 2.0, 2, 4, size.y - 4), fade_col)
	if set_fade_out.is_valid() and fx.y >= -_EDGE_W and fx.y <= size.x + _EDGE_W:
		draw_rect(Rect2(fx.y - 2.0, 2, 4, size.y - 4), fade_col)
	if label != "":
		var lx := clampf(maxf(xa + 6, reserved_left), 4, size.x - 60)
		draw_string(ThemeDB.fallback_font, Vector2(lx, size.y * 0.5 + 4),
			label, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.9, 0.92, 0.96, 0.9))
	# Playhead, same style as MaskTimeline's - but tagged with THIS track's own local
	# source position (cin + how far the master playhead sits into this clip's span),
	# not the master time, since knowing where the alt track itself is tracking is the
	# whole point (see feedback/0007). Only drawn while the master playhead is actually
	# over this clip's span - outside it the track isn't playing (see _sync_tracks).
	if get_playhead.is_valid():
		var master_t: float = get_playhead.call()
		var local_t := master_t - offset + cin
		if local_t >= cin and local_t < cout:
			var px := tview.x_of(master_t)
			if px >= -2.0 and px <= size.x + 2.0:
				draw_line(Vector2(px, 0), Vector2(px, size.y), Color(1, 1, 1, 0.92), 2.0)
				var tag := MaskTimeline.format_time(local_t)
				var tag_w := ThemeDB.fallback_font.get_string_size(tag, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x + 8.0
				var tag_x := clampf(px - tag_w * 0.5, 0.0, size.x - tag_w)
				draw_rect(Rect2(tag_x, 2, tag_w, 18), Color(1, 1, 1, 0.92))
				draw_string(ThemeDB.fallback_font, Vector2(tag_x + 4, 15), tag,
					HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.05, 0.06, 0.08))
