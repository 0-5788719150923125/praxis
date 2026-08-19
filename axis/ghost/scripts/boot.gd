extends Node

## Boot - the earliest hook (first autoload), for things that must happen before the
## window is ever drawn into. In export-render mode (--export) it keeps the render
## window out of the way as early as GDScript can - off-screen, no focus - so it
## barely flickers into view before the render takes over. (The OS maps the window
## during engine init, before any script runs, so a sub-frame flash isn't fully
## avoidable without a virtual display; this gets it as close as possible.)
##
## NEVER minimize a Movie Maker window. Godot skips rendering while a window is
## minimized (window_can_draw() is false), and the movie writer then re-captures the
## last rendered buffer while the audio keeps advancing - the recording freezes for
## as long as the compositor holds the window minimized (on Wayland it can be every
## frame: a fully black video). This was the "4K exports partially freeze for
## seconds" bug: the old code minimized here to hide the window on Wayland, and the
## slow 4K frames gave the compositor every chance to throttle it. The exporter now
## shrinks the OS window instead (window_*_override in override.cfg) - the movie
## records the VIEWPORT, whose resolution is independent of the window in the
## export's "viewport" stretch mode, so a tiny window costs nothing and stays
## drawable.

func _enter_tree() -> void:
	if not OS.get_cmdline_user_args().has("--export"):
		return
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_NO_FOCUS, true)
	DisplayServer.window_set_position(Vector2i(-5000, -5000))   # X11 honors this


## TOOLTIP WRAPPING, applied to the whole app.
##
## Godot's built-in tooltip is a Label with autowrap OFF, and there is no theme item to change
## that: the default theme exposes only `shadow_offset_x/y`, `outline_size` and `font_size` for
## `TooltipLabel`, so no Theme resource can fix it. The label simply grows to fit, and a long
## description runs off the side of the screen - measured, 79 characters is 636 px, so ghost's
## longest tooltip renders about 3,400 px wide on a 1920 px display.
##
## Since the WIDTH cannot be constrained, the TEXT is re-flowed instead: real newlines inserted at
## a measured pixel width, using the same font and size the tooltip will draw with. Hooked from
## `node_added` so it covers every mode, every dialog and anything added later without each call
## site having to remember - the request was for this to be global, and 68 scattered
## `tooltip_text` assignments is exactly the kind of thing that goes stale.
##
## Re-flowing is IDEMPOTENT (a wrapped paragraph collapses back to one line and re-wraps to the
## same result), which is what lets it run again on every hover - so a tooltip whose text is
## rewritten at runtime, as the mask editor's readouts are, stays wrapped too. Blank-line
## paragraph breaks an author wrote on purpose are preserved.
const TIP_MAX_PX := 480.0


func _ready() -> void:
	get_tree().node_added.connect(_hook_tooltip)


## THE BACKSTOP for background programs. Every child ghost starts goes through
## [Subprocess], which remembers it; this kills whatever is still registered when the app
## closes, so a mode that forgets to clean up (or is torn down in an order that skips its
## own `_exit_tree`) still cannot leave an `ffmpeg` writing to the disk with nothing on
## screen. Boot is the first autoload, so it is the last thing standing at shutdown.
##
## This is the CLEAN-QUIT half only. If ghost is killed outright no script runs at all,
## which is why `Subprocess` also binds each child to this process at the kernel level -
## see its class doc.
func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST or what == NOTIFICATION_EXIT_TREE:
		Subprocess.reap_all()


func _hook_tooltip(n: Node) -> void:
	if not (n is Control):
		return
	var c: Control = n
	# On hover, because that also catches tooltips rewritten after the control was built.
	if not c.mouse_entered.is_connected(_wrap_tooltip):
		c.mouse_entered.connect(_wrap_tooltip.bind(c))
	# And once now, for controls that never emit mouse_entered themselves (a Label defaults to
	# MOUSE_FILTER_IGNORE) but whose tooltip is surfaced through an ancestor.
	_wrap_tooltip.call_deferred(c)


func _wrap_tooltip(c: Control) -> void:
	if not is_instance_valid(c) or c.tooltip_text.is_empty():
		return
	var wrapped := wrap_tip(c.tooltip_text, TIP_MAX_PX)
	if wrapped != c.tooltip_text:
		c.tooltip_text = wrapped


## Re-flow `text` so no line exceeds `max_px` in the tooltip's own font. Public, so anything that
## builds a tooltip string by hand can pre-wrap it the same way.
static func wrap_tip(text: String, max_px := TIP_MAX_PX) -> String:
	var f: Font = ThemeDB.fallback_font
	if f == null or max_px <= 0.0:
		return text
	var fsize := ThemeDB.get_default_theme().get_font_size("font_size", "TooltipLabel")
	if fsize <= 0:
		fsize = ThemeDB.fallback_font_size
	var paras: Array = []
	for para in text.split("\n\n"):
		var lines: Array = []
		var line := ""
		for word in String(para).replace("\n", " ").split(" ", false):
			var w := String(word)
			if line.is_empty():
				line = w
				continue
			var probe := line + " " + w
			if f.get_string_size(probe, HORIZONTAL_ALIGNMENT_LEFT, -1, fsize).x > max_px:
				lines.append(line)
				line = w
			else:
				line = probe
		if not line.is_empty():
			lines.append(line)
		paras.append("\n".join(lines))
	return "\n\n".join(paras)
