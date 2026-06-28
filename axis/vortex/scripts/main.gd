extends Node2D

## vortex entry point.
##
## Almost nothing lives here on purpose: the audio is owned by [Spectrum] and the
## visuals by [Director]. main owns the *session lifecycle* - showing the splash,
## starting a session in Auto or Manual mode, and tearing it back down to the splash
## when the song ends - and maps the global keys. Scenes are added as children here.

const Bake := preload("res://scripts/bake.gd")

var _splash: Node = null
var _feedback: Node = null
var _workspace: Node = null
var _exporter: Node = null
var _status_t := 0.0     # throttle for writing render progress (export mode)

# Export render mode: this instance was relaunched by the exporter in Movie Maker
# mode (--export). It runs the session clean (no overlays) and quits when the song
# ends, so the recorded movie starts and stops with the music.
var _export_mode := false


func _ready() -> void:
	_export_mode = OS.get_cmdline_user_args().has("--export")
	if _export_mode:
		# Background render: Boot (first autoload) already hid the window as early as
		# possible. Run clean and quit when the song ends, finalizing the movie.
		Spectrum.song_finished.connect(func(): get_tree().quit())
		_begin_session()
		return
	# The exporter is persistent (created once, never torn down): an in-flight render
	# and its status must survive the song ending and the return to the home screen.
	_exporter = preload("res://scripts/exporter.gd").new()
	add_child(_exporter)
	# Return to the home screen whenever the current song finishes.
	Spectrum.song_finished.connect(_end_session)
	if _wants_direct_boot():
		_begin_session()                 # CLI flags / --no-splash: skip the splash
	else:
		_show_splash()


# A CLI flag means "I know what I want" (authoring, automation, headless): boot
# straight in, past the splash, so the existing --audio/--scene/--runbook flows and
# the headless tests are unchanged.
func _wants_direct_boot() -> bool:
	var args := OS.get_cmdline_user_args()
	return args.has("--audio") or args.has("--scene") \
		or args.has("--runbook") or args.has("--no-splash")


func _show_splash() -> void:
	var splash := preload("res://scripts/splash.gd").new()
	splash.start_session = _on_splash_start
	_splash = splash
	add_child(splash)


# The splash hands back the chosen song and whether Manual was clicked. Manual opens
# the workspace over a session running the default runbook; Auto just runs.
func _on_splash_start(audio_path: String, manual: bool) -> void:
	if manual:
		Director.load_runbook("default")
		_begin_session(audio_path)
		_workspace = preload("res://scripts/workspace.gd").new()
		add_child(_workspace)
	else:
		_begin_session(audio_path)


func _begin_session(audio_path := "") -> void:
	Spectrum.begin(audio_path)
	Director.attach(self)
	if _export_mode:
		return                         # render clean: no overlays in the recorded movie
	_feedback = preload("res://scripts/feedback.gd").new()
	add_child(_feedback)               # press ` to critique a scene


# Tear the session down and return to the splash (the song ended, or a future
# "exit to home" control). The persistent exporter is intentionally NOT freed - a
# render in progress and its status carry across to the home screen.
func _end_session() -> void:
	for n in [_feedback, _workspace, _splash]:
		if n != null and is_instance_valid(n):
			n.queue_free()
	_feedback = null
	_workspace = null
	Director.detach()
	Spectrum.stop()
	_show_splash()


func _process(delta: float) -> void:
	# The background render reports its progress (playback position / length) so the
	# live app's exporter can show a percentage in the status notification.
	if _export_mode:
		_status_t += delta
		if _status_t >= 0.3:
			_status_t = 0.0
			var length := Spectrum.song_length()
			if length > 0.0:
				Bake.write_progress(Spectrum.current.time / length)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_F11:
				_toggle_fullscreen()
			KEY_SPACE:
				Director.next()
			KEY_ESCAPE:
				get_tree().quit()


func _toggle_fullscreen() -> void:
	var mode := DisplayServer.window_get_mode()
	if mode == DisplayServer.WINDOW_MODE_FULLSCREEN:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
	else:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
