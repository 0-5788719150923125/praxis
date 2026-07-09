extends RefCounted
class_name TimelineView

## TimelineView - the shared pixel<->time mapping for the mask editor's whole
## timeline stack: the primary clip's trim lane, any imported track's lane, and
## the marker-editing strip all read the SAME zoom/pan state, so a second's
## position agrees across every row.
##
## `extent` is DELIBERATELY a cached value, refreshed only by an explicit
## refresh() call - never derived live from whatever a drag is currently
## touching. See MaskSession.timeline_extent(): the ruler's total length is
## `duration` (which trimming never changes) extended to cover the furthest
## track, plus a fixed overflow margin. Recomputing it mid-drag is exactly the
## "drag a clip's edge and the whole ruler rescales under your cursor" bug every
## timeline editor seems to hit eventually - refresh() is called once a gesture
## ENDS (mouse-up), never during one, so the ruler a handle is being measured
## against never moves while that same handle is what's moving it.

const _MAX_ZOOM := 64.0

var zoom := 1.0
var view_start := 0.0      # left edge of the visible window, seconds
var extent := 1.0          # cached total ruler length, seconds - see refresh()
var width := 1.0           # widest lane's pixel width; kept current by callers each frame


func refresh(session: MaskSession) -> void:
	if session == null:
		return
	extent = maxf(1.0, session.timeline_extent())
	_clamp_view()


func visible_span() -> float:
	return extent / maxf(1.0, zoom)


func x_of(t: float) -> float:
	return (t - view_start) / maxf(0.001, visible_span()) * width


func t_of(x: float) -> float:
	return view_start + clampf(x / maxf(1.0, width), 0.0, 1.0) * visible_span()


## Zoom around the time under the cursor, so the point being studied stays put.
func zoom_at(x: float, factor: float) -> void:
	var anchor_t := t_of(x)
	zoom = clampf(zoom * factor, 1.0, _MAX_ZOOM)
	view_start = anchor_t - (x / maxf(1.0, width)) * visible_span()
	_clamp_view()


## Pan by a fraction of the visible window (a no-op at zoom 1, where the whole
## extent is already on screen and _clamp_view pins the window).
func pan(frac: float) -> void:
	view_start += visible_span() * frac
	_clamp_view()


func _clamp_view() -> void:
	view_start = clampf(view_start, 0.0, maxf(0.0, extent - visible_span()))
