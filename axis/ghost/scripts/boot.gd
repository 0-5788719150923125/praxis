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
