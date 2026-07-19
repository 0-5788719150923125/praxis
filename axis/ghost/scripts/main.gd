extends Node2D

## ghost entry point.
##
## Almost nothing lives here on purpose: the audio is owned by [Spectrum] and the
## visuals by [Director]. main owns the *session lifecycle* - showing the splash,
## starting a session in Auto or Manual mode, and tearing it back down to the splash
## when the song ends - and maps the global keys. Scenes are added as children here.

const Bake := preload("res://scripts/bake.gd")

var _splash: Node = null
var _feedback: Node = null
var _assistant: Node = null
var _workspace: Node = null
var _exporter: Node = null
var _mask_editor: Node = null
var _synth_editor: Node = null
var _synth_active := false           # a synth take is playing as the session
var _stream: Node = null             # the live VoiceStream feeding the session
var _subtitles: Node = null          # karaoke overlay, present when the audio has a sidecar
var _status_t := 0.0     # throttle for writing render progress (export mode)

# Export render mode: this instance was relaunched by the exporter in Movie Maker
# mode (--export). It runs the session clean (no overlays) and quits when the song
# ends, so the recorded movie starts and stops with the music.
var _export_mode := false

func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	# Mask mode is a standalone authoring tool (see mask_editor.gd) - tied to one
	# specific external clip, not the audio-reactive show - so it does not touch
	# Director/Spectrum at all and is checked before everything else.
	if args.has("--mask-render"):
		_begin_mask_render(_arg_value(args, "--mask-render"))
		return
	if args.has("--mask-edit"):
		_open_mask_editor(_arg_value(args, "--mask-edit"))
		return
	# Synthesis mode: the voice editor (see synth_editor.gd). Unlike mask mode it
	# DOES use Director/Spectrum - a rendered take plays as a normal session so the
	# scenes react to the narration - but the session starts on Speak, not at boot.
	if args.has("--synth"):
		_open_synth_editor()
		return
	_export_mode = args.has("--export")
	if _export_mode:
		# Background render: Boot (first autoload) already hid the window as early as
		# possible, and the exporter's override.cfg has set the output resolution + stretch
		# mode at engine startup (Movie Maker locks resolution before any script runs, so
		# it cannot be set from here). Run clean and quit when the song ends.
		Spectrum.song_finished.connect(func(): get_tree().quit())
		_begin_session()
		return
	# The live window stretches in "canvas_items" mode (set in project.godot): 2D is
	# rasterized at the window's native pixel resolution - so fullscreen is crisp, not an
	# upscale of the 1080p base - while the coordinate system scales *proportionally*, so
	# UI and scene content keep their relative size and snap back exactly when the window
	# returns to its original size. (Export overrides this; see _apply_export_resolution.)
	# The exporter is persistent (created once, never torn down): an in-flight render
	# and its status must survive the song ending and the return to the home screen.
	_exporter = preload("res://scripts/exporter.gd").new()
	add_child(_exporter)
	# Always present, regardless of the splash's Assistant dropdown (see
	# splash.gd) - it's also the feedback browser (review/delete old
	# submissions), which shouldn't require opting into AI dispatch to use.
	# Assistant itself gates actually DISPATCHING anything on the persisted
	# backend choice; this just controls whether the console exists at all.
	# Persistent (created once, never torn down) for the same reason as the
	# exporter: a queued/running claude subprocess and its conversation
	# history must survive a song ending and a new session beginning, not get
	# torn down and duplicated by the next _begin_session() call - see the
	# signal (re)connect there.
	_assistant = preload("res://scripts/assistant.gd").new()
	add_child(_assistant)
	# When the current song finishes: an AUTO session returns to the home screen; a
	# MANUAL session is endless - loop the audio and leave the visualization alone.
	Spectrum.song_finished.connect(_on_song_finished)
	if _wants_direct_boot():
		_begin_session()                 # CLI flags / --no-splash: skip the splash
	else:
		_show_splash()


# A CLI flag means "I know what I want" (authoring, automation, headless): boot
# straight in, past the splash, so the existing --audio/--scene/--storyboard flows and
# the headless tests are unchanged.
func _wants_direct_boot() -> bool:
	var args := OS.get_cmdline_user_args()
	return args.has("--audio") or args.has("--scene") \
		or args.has("--storyboard") or args.has("--no-splash")


func _show_splash() -> void:
	var splash := preload("res://scripts/splash.gd").new()
	splash.start_session = _on_splash_start
	splash.start_mask = _on_splash_mask
	splash.start_synth = _open_synth_editor
	_splash = splash
	add_child(splash)


# The splash hands back the chosen song and whether Manual was clicked. Manual opens
# the workspace over a session running the default storyboard; Auto just runs.
func _on_splash_start(audio_path: String, manual: bool) -> void:
	if manual:
		Director.load_storyboard("default")
		_begin_session(audio_path)
		_workspace = preload("res://scripts/workspace.gd").new()
		add_child(_workspace)
	else:
		_begin_session(audio_path)


# The splash's Mask button. Mask mode never touches Director/Spectrum (see
# _open_mask_editor), so this is a straight handoff, not a _begin_session() variant -
# there is no "session" here in the Auto/Manual sense, just the editor opening on
# whatever clip (or lack of one) the splash handed over.
func _on_splash_mask(video_path: String) -> void:
	_open_mask_editor(video_path)


func _begin_session(audio_path := "") -> void:
	Spectrum.begin(audio_path)
	Director.attach(self)
	_attach_subtitles()
	if _export_mode:
		return                         # render clean: no overlays (the Director fades the video ends)
	_feedback = preload("res://scripts/feedback.gd").new()
	add_child(_feedback)               # press ` to critique a scene
	# _assistant itself is created once in _ready() (see the exporter's own
	# persistence comment) - a fresh _feedback node needs a fresh connection
	# to it every _begin_session() call, but the assistant instance and its
	# in-flight/queued work must not be recreated.
	if _assistant != null and is_instance_valid(_assistant):
		_feedback.submitted.connect(_assistant.enqueue)


# The song ended. Manual mode never exits to the home screen: restart the audio in
# place and keep the show running - the Director's walk, the storyboard's tail, and
# the dial's deposited modulations all carry across the loop untouched. (Whether the
# SEQUENCE replays is the board's own `loop` field; the-point's `loop: false` keeps
# streaming its tail.) Auto mode returns home as before.
func _on_song_finished() -> void:
	# A synth take is endless like a manual session: loop the narration in place
	# so the show stays up while the user iterates in the editor.
	if _synth_active:
		Spectrum.replay()
		print("ghost: take looped (synthesis session continues)")
		return
	if Director.is_manual():
		Spectrum.replay()
		print("ghost: song looped (manual session continues)")
		return
	_end_session()


# Tear the session down and return to the splash (the song ended, or a future
# "exit to home" control). The persistent exporter is intentionally NOT freed - a
# render in progress and its status carry across to the home screen.
func _end_session() -> void:
	# Never yank the feedback console out from under the user. If the song ends while
	# they have it open and are typing, defer the whole teardown until they submit or
	# cancel - the held scene stays frozen behind the modal until then.
	if _feedback != null and is_instance_valid(_feedback) and _feedback.is_open():
		if not _feedback.closed.is_connected(_end_session):
			_feedback.closed.connect(_end_session, CONNECT_ONE_SHOT)
		return
	for n in [_feedback, _workspace, _splash, _subtitles]:
		if n != null and is_instance_valid(n):
			n.queue_free()
	_feedback = null
	_workspace = null
	_subtitles = null
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
				# Mask mode owns Space itself (play/pause the clip, see mask_editor.gd) -
				# it never attaches Director, so Director.next() would be meaningless
				# there anyway; this just makes the skip explicit rather than relying on
				# Director to no-op safely while unattached.
				if _mask_editor == null or not is_instance_valid(_mask_editor):
					Director.next()
			KEY_ESCAPE:
				get_tree().quit()


## The value following `flag` in the cmdline args, or "" if the flag is bare/absent
## (mask mode's two entry points both take an optional path this way).
func _arg_value(args: PackedStringArray, flag: String) -> String:
	var i := args.find(flag)
	if i >= 0 and i + 1 < args.size():
		return args[i + 1]
	return ""


## --synth: the voice-synthesis editor. Each Speak renders a WAV take and plays it
## as a fresh session (new fingerprint, new show); the take loops when it ends,
## like a manual session, so the show stays up while the user iterates.
func _open_synth_editor() -> void:
	# The exporter is persistent here for the same reason as in the normal flow:
	# a synth take exports exactly like a song (the relaunch boots
	# `--audio take.wav --export`, and the sidecar gives the render subtitles).
	# Guarded: reached both from the splash (exporter already exists) and from a
	# direct `--synth` boot (it doesn't).
	if _exporter == null or not is_instance_valid(_exporter):
		_exporter = preload("res://scripts/exporter.gd").new()
		add_child(_exporter)
	var editor := preload("res://scripts/synth_editor.gd").new()
	editor.begin_stream = _begin_synth_stream
	_synth_editor = editor
	add_child(editor)
	if not Spectrum.song_finished.is_connected(_on_song_finished):
		Spectrum.song_finished.connect(_on_song_finished)


## Start (or restart) the session on a live VoiceStream: audio begins the same
## frame Speak was clicked - the stream synthesizes ahead of the playhead and
## the analyzer bus hears it like a song. Streamed takes loop endlessly inside
## the stream itself (song_finished never fires for a generator).
func _begin_synth_stream(stream: Node) -> void:
	if _synth_active:
		Director.detach()
		Spectrum.stop()
	if _stream != null and is_instance_valid(_stream):
		_stream.queue_free()
	_stream = stream
	add_child(stream)
	var pb: AudioStreamGeneratorPlayback = Spectrum.begin_stream(stream.fingerprint(), Voice.SR)
	stream.attach_playback(pb)
	Director.attach(self)
	_attach_live_subtitles(stream)
	stream.completed.connect(_on_stream_completed)
	stream.restarted.connect(_on_stream_restarted)
	_synth_active = true


## Subtitles for a live stream: share the stream's growing word array directly;
## the sidecar file doesn't exist yet (the take is still being synthesized).
func _attach_live_subtitles(stream: Node) -> void:
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.queue_free()
	var subs := preload("res://scripts/subtitles.gd").new()
	subs.words = stream.words
	_subtitles = subs
	add_child(subs)


func _on_stream_completed(dur: float, wav_path: String) -> void:
	Spectrum.set_stream_info(wav_path, dur)
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.loop_length = dur
	print("ghost: take complete (%.1fs) -> %s" % [dur, wav_path])


## The stream replaced its content in place (an implicit re-speak): rebase the
## karaoke clock and forget the previous take's loop length until the new take
## finishes.
func _on_stream_restarted(base: float) -> void:
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.time_base = base
		_subtitles.loop_length = 0.0


## Karaoke subtitles are session content, not editor chrome: whenever the
## session's audio has a sidecar timing map (a synth take), attach the overlay -
## live sessions, reopened takes, and export renders alike. Music without a
## sidecar gets nothing.
func _attach_subtitles() -> void:
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.queue_free()
	_subtitles = null
	var side: String = Subtitles.sidecar_for(Spectrum.audio_path())
	if side.is_empty():
		return
	var subs := preload("res://scripts/subtitles.gd").new()
	if subs.load_sidecar(side):
		_subtitles = subs
		add_child(subs)
		print("ghost: subtitles attached (%d words)" % (subs.words as Array).size())


## --mask-render <session.json>: the export relaunch. No splash, no Director - just
## the clip + shader + audio, autoplaying, quitting when the audio ends (mirrors
## _export_mode's Spectrum.song_finished -> quit, one layer down).
func _begin_mask_render(session_path: String) -> void:
	var editor := preload("res://scripts/mask_editor.gd").new()
	editor.render_mode = true
	add_child(editor)
	editor.open_source(session_path)


## --mask-edit [path]: the interactive editor. `path` is a session .json, a raw
## source video (transcoded once and cached), or empty (prompts via file dialog).
func _open_mask_editor(path: String) -> void:
	var editor := preload("res://scripts/mask_editor.gd").new()
	_mask_editor = editor
	add_child(editor)
	editor.open_source(path)


func _toggle_fullscreen() -> void:
	var mode := DisplayServer.window_get_mode()
	if mode == DisplayServer.WINDOW_MODE_FULLSCREEN:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
	else:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
