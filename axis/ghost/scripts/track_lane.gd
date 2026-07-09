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

var get_in: Callable
var set_in: Callable
var get_out: Callable
var set_out: Callable
var get_offset: Callable = func(): return 0.0
var set_offset: Callable = Callable()   # invalid = shift disabled (the primary lane)

const _EDGE_W := 8.0
var _drag_mode := ""   # "" | "in" | "out" | "body"
var _drag_started_emitted := false
var _body_grab_dx := 0.0


func _ready() -> void:
	custom_minimum_size = Vector2(0, 26)
	mouse_filter = Control.MOUSE_FILTER_STOP


func _process(_dt: float) -> void:
	queue_redraw()


func _gui_input(event: InputEvent) -> void:
	if tview == null:
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_drag_mode = _hit(mb.position.x)
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


func _hit(x: float) -> String:
	var offset: float = get_offset.call()
	var x0 := tview.x_of(offset)
	var x1 := tview.x_of(offset + (float(get_out.call()) - float(get_in.call())))
	if absf(x - x0) <= _EDGE_W:
		return "in"
	if absf(x - x1) <= _EDGE_W:
		return "out"
	if x > x0 and x < x1 and movable:
		return "body"
	return ""


func _apply_drag(x: float) -> void:
	var t := tview.t_of(x)
	var offset: float = get_offset.call()
	match _drag_mode:
		"in":
			var cur_out: float = get_out.call()
			set_in.call(clampf(t - offset, 0.0, cur_out - 0.1))
		"out":
			var cur_in: float = get_in.call()
			set_out.call(clampf(t - offset, cur_in + 0.1, full_duration))
		"body":
			if set_offset.is_valid():
				var span: float = float(get_out.call()) - float(get_in.call())
				set_offset.call(clampf(tview.t_of(x - _body_grab_dx), 0.0, maxf(0.0, tview.extent - span)))
	changed.emit()


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.09, 0.10, 0.13, 0.94))
	if tview == null:
		return
	var offset: float = get_offset.call()
	var cin: float = get_in.call()
	var cout: float = get_out.call()
	var x0 := tview.x_of(offset)
	var x1 := tview.x_of(offset + (cout - cin))
	if x1 < 0.0 or x0 > size.x:
		return
	var xa := clampf(x0, -4.0, size.x + 4.0)
	var xb := clampf(x1, -4.0, size.x + 4.0)
	draw_rect(Rect2(xa, 2, maxf(1.0, xb - xa), size.y - 4), Color(color.r, color.g, color.b, 0.35))
	draw_rect(Rect2(xa, 2, maxf(1.0, xb - xa), size.y - 4), Color(color.r, color.g, color.b, 0.9), false, 2.0)
	# Grip handles at each trimmable edge (drawn even for the immovable primary -
	# in/out are always draggable, only BODY shift is what `movable` gates).
	if x0 >= -_EDGE_W and x0 <= size.x + _EDGE_W:
		draw_rect(Rect2(x0 - 2.5, 2, 5, size.y - 4), color.lightened(0.35))
	if x1 >= -_EDGE_W and x1 <= size.x + _EDGE_W:
		draw_rect(Rect2(x1 - 2.5, 2, 5, size.y - 4), color.lightened(0.35))
	if label != "":
		draw_string(ThemeDB.fallback_font, Vector2(clampf(xa + 6, 4, size.x - 60), size.y * 0.5 + 4),
			label, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.9, 0.92, 0.96, 0.9))
