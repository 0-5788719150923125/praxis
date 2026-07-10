extends Control
class_name MaskTimeline

## MaskTimeline - the mask editor's scrub strip.
##
## A bespoke-drawn instrument, same idiom as [DialWidget]: click-drag on the body
## to scrub (emits `scrubbed`), press-drag a marker's triangle HANDLE (the top band
## only - grabbing anywhere on the full tick made it far too easy to snag a marker
## while aiming for the playhead) to pick it up and move it (emits `marker_picked`
## on grab, `marker_moved` while held). Session markers are mutated directly (time,
## plus a re-sort); MaskEditor stays the source of truth for everything else.
##
## Ctrl+scroll ZOOMS around the cursor (plain scroll pans when zoomed), for
## granular work across short sequences - all drawing and hit-testing goes through
## the shared TimelineView (see timeline_view.gd), so this strip and any track
## lane above it always agree on where a second sits on screen.
##
## Every marker is a ramp or a decay (see MaskSession.MARKER_KINDS) and draws as
## three things together: the anchor line, the triangle handle pointing the
## direction its span runs (back into the past for a ramp, forward for a decay),
## and a translucent tint across exactly the span itself (MaskSession.marker_span).

signal scrubbed(t: float)
signal marker_picked(m: Dictionary)
## Fired once per drag gesture, BEFORE the first time-mutation - the hook point
## for undo (see MaskEditor._push_undo). A plain click-to-select never fires
## this at all (no motion, no mutation, nothing to undo).
signal marker_drag_started(m: Dictionary)
signal marker_moved(m: Dictionary)

var session: MaskSession = null
var player: VideoStreamPlayer = null
var selected: Variant = null   # the currently-selected marker Dictionary, or null
var waveform_texture: Texture2D = null   # set by MaskEditor once generated; null draws nothing
## The shared pixel<->time mapping - see timeline_view.gd. Set by MaskEditor
## (one instance shared with every track lane, so they all agree on where a
## second sits on screen). Never owned here anymore - multi-track needs the
## SAME zoom/pan read by more than one Control.
var tview: TimelineView = null

var _dragging := false                 # scrubbing the playhead (grabbed empty space)
var _dragging_marker: Variant = null   # the marker Dictionary being dragged, or null
var _drag_started_emitted := false     # per-gesture latch for marker_drag_started

const _HANDLE_H := 16.0    # the triangle-handle band; also the only marker-grab region


func _ready() -> void:
	custom_minimum_size = Vector2(0, 84)
	mouse_filter = Control.MOUSE_FILTER_STOP
	clip_contents = true   # zoomed-in drawing runs past both edges; never bleed out


func _process(_dt: float) -> void:
	if tview != null:
		tview.width = maxf(1.0, size.x)
	queue_redraw()


func _gui_input(event: InputEvent) -> void:
	if session == null or session.duration <= 0.0 or tview == null:
		return
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			if mb.ctrl_pressed:
				tview.zoom_at(mb.position.x, 1.25)
			else:
				tview.pan(-0.1)
			accept_event()
			return
		if mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			if mb.ctrl_pressed:
				tview.zoom_at(mb.position.x, 1.0 / 1.25)
			else:
				tview.pan(0.1)
			accept_event()
			return
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				var m: Variant = _marker_near(mb.position)
				if m != null:
					marker_picked.emit(m)
					_dragging_marker = m
					_drag_started_emitted = false
				else:
					_dragging = true
					scrubbed.emit(tview.t_of(mb.position.x))
			else:
				_dragging = false
				_dragging_marker = null
			accept_event()
	elif event is InputEventMouseMotion:
		if _dragging_marker != null:
			_drag_marker_to_x(event.position.x)
			accept_event()
		elif _dragging:
			scrubbed.emit(tview.t_of(event.position.x))
			accept_event()


## Move the dragged marker to the time under `x`, keeping the array sorted (the
## interpolation in MaskSession.at_time assumes sorted order) - re-sorting every
## motion event is trivial at marker-list sizes, no need to defer it to release.
func _drag_marker_to_x(x: float) -> void:
	if not _drag_started_emitted:
		_drag_started_emitted = true
		marker_drag_started.emit(_dragging_marker)
	_dragging_marker.time = clampf(tview.t_of(x), 0.0, session.duration)
	session.markers.sort_custom(func(a, b): return a.time < b.time)
	marker_moved.emit(_dragging_marker)


## Markers are grabbable ONLY by their triangle handle in the top band - clicking
## the strip's body always scrubs, however close to a marker's anchor line it lands.
func _marker_near(pos: Vector2) -> Variant:
	if pos.y > _HANDLE_H + 4.0:
		return null
	for m in session.markers:
		var mx := tview.x_of(float(m.time))
		if pos.x > mx - 7.0 and pos.x < mx + _HANDLE_H:   # anchor edge through triangle tip
			return m
	return null


## m:ss (or h:mm:ss past an hour) - shared with MaskEditor's panel readout so the
## two timestamps displayed on screen always agree on format.
static func format_time(t: float) -> String:
	var s := maxi(0, int(t))
	if s >= 3600:
		return "%d:%02d:%02d" % [s / 3600, (s / 60) % 60, s % 60]
	return "%d:%02d" % [s / 60, s % 60]


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color(0.09, 0.10, 0.13, 0.94))
	if session == null or session.duration <= 0.0 or tview == null:
		# Distinct from "session is null" (no clip picked at all) so a clip that IS
		# loaded but came back with a bad/zero probe reads as its own state, not the
		# same blank message as never having opened anything.
		var msg := "no clip loaded" if session == null else "clip loaded, but duration came back 0 - re-preparing…"
		draw_string(ThemeDB.fallback_font, Vector2(12, size.y * 0.5 + 4),
			msg, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.5, 0.55, 0.62))
		return
	var span := tview.visible_span()
	# The amplitude waveform (rendered once via ffmpeg's showwavespic, cached next to
	# the clip - see MaskEditor._ensure_waveform). Drawn across wherever the clip's
	# OWN span [0, duration] currently maps to on screen - not the full widget width -
	# so it stays correctly placed once the ruler can extend past a single clip
	# (secondary tracks, the overflow margin) instead of always filling edge to edge.
	if waveform_texture != null:
		var wx0 := tview.x_of(0.0)
		var wx1 := tview.x_of(session.duration)
		if wx1 > 0.0 and wx0 < size.x:
			var tw := float(waveform_texture.get_width())
			var th := float(waveform_texture.get_height())
			draw_texture_rect_region(waveform_texture, Rect2(wx0, 4, wx1 - wx0, size.y - 20),
				Rect2(0, 0, tw, th), Color(0.55, 0.68, 0.85, 0.85))
	# Marker spans - a translucent tint across exactly the time range each marker's
	# own ramp/decay occupies (see MaskSession.marker_span). Ramp = cool/green,
	# decay = warm/orange; overlapping spans layer visibly.
	for m in session.markers:
		var mspan: Vector2 = session.marker_span(m)
		var x0 := tview.x_of(clampf(mspan.x, 0.0, session.duration))
		var x1 := tview.x_of(clampf(mspan.y, 0.0, session.duration))
		if x1 < 0.0 or x0 > size.x:
			continue
		var is_ramp: bool = int(m.get("kind", 0.0)) == 0
		var tint := Color(0.45, 0.85, 0.55, 0.16) if is_ramp else Color(0.95, 0.55, 0.3, 0.16)
		draw_rect(Rect2(x0, 0, maxf(1.0, x1 - x0), size.y), tint)
	# The ruler adapts its tick step to the visible window, so zooming in brings
	# finer gradations instead of one lonely label.
	var step := 60.0
	for s in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]:
		if span / s <= 14.0:
			step = s
			break
	var tt := ceilf(tview.view_start / step) * step
	while tt < tview.view_start + span:
		var x := tview.x_of(tt)
		draw_line(Vector2(x, size.y - 14), Vector2(x, size.y), Color(0.3, 0.34, 0.4, 0.6), 1.0)
		var tick_label := format_time(tt) if step >= 1.0 else "%.1f" % tt
		draw_string(ThemeDB.fallback_font, Vector2(x + 3, size.y - 4),
			tick_label, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.45, 0.5, 0.58))
		tt += step
	# Markers: an anchor line (taller/brighter when selected) plus a triangle handle
	# pointing back (ramp - it's pulling the timeline in from the past) or forward
	# (decay - pushing away into the future), colored to match its span tint. The
	# handle is also the ONLY grab region (see _marker_near).
	for m in session.markers:
		var x := tview.x_of(float(m.time))
		if x < -20.0 or x > size.x + 20.0:
			continue
		var on: bool = selected != null and selected == m
		var is_ramp: bool = int(m.get("kind", 0.0)) == 0
		draw_line(Vector2(x, 0), Vector2(x, size.y - 16),
			Color(1.0, 0.82, 0.4, 0.95) if on else Color(0.55, 0.75, 1.0, 0.85), 3.0 if on else 2.0)
		var col := Color(0.5, 0.95, 0.6) if is_ramp else Color(0.98, 0.6, 0.35)
		if on:
			col = col.lightened(0.35)
		var h := _HANDLE_H
		var tip_x := x - h * 0.7 if is_ramp else x + h * 0.7
		var pts := PackedVector2Array([Vector2(x, 1), Vector2(x, h), Vector2(tip_x, h * 0.5)])
		draw_colored_polygon(pts, col)
	# Playhead, with its own time tag riding right above it - the point being you
	# never have to look away from "where the line is" to read "what time that is".
	var t: float = player.stream_position if player != null else 0.0
	var px := tview.x_of(t)
	if px >= -2.0 and px <= size.x + 2.0:
		draw_line(Vector2(px, 0), Vector2(px, size.y), Color(1, 1, 1, 0.92), 2.0)
		var tag := format_time(t)
		var tag_w := ThemeDB.fallback_font.get_string_size(tag, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x + 8.0
		var tag_x := clampf(px - tag_w * 0.5, 0.0, size.x - tag_w)
		draw_rect(Rect2(tag_x, 2, tag_w, 18), Color(1, 1, 1, 0.92))
		draw_string(ThemeDB.fallback_font, Vector2(tag_x + 4, 15), tag,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.05, 0.06, 0.08))
	# Zoom badge, so a zoomed view never masquerades as the whole clip.
	if tview.zoom > 1.01:
		draw_string(ThemeDB.fallback_font, Vector2(size.x - 74, 14),
			"%.0fx zoom" % tview.zoom, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.7, 0.78, 0.9, 0.8))
