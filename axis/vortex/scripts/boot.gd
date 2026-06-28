extends Node

## Boot - the earliest hook (first autoload), for things that must happen before the
## window is ever drawn into. In export-render mode (--export) it hides the render
## window as early as GDScript can - minimized, off-screen, no focus - so it barely
## flickers into view before the render takes over. (The OS maps the window during
## engine init, before any script runs, so a sub-frame flash isn't fully avoidable
## without a virtual display; this gets it as close as possible.)

func _enter_tree() -> void:
	if not OS.get_cmdline_user_args().has("--export"):
		return
	DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_NO_FOCUS, true)
	DisplayServer.window_set_position(Vector2i(-5000, -5000))   # X11 honors this
	DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_MINIMIZED)  # Wayland honors this
