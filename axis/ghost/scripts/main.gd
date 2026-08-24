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
var _chrome: Node = null             # shared session furniture (exporter/assistant/feedback)
var _workspace: Node = null
var _mask_editor: Node = null
var _synth_editor: Node = null
var _synth_active := false           # a synth take is playing as the session
var _generative: Node = null         # the generative editor, when that mode is up
var _stream: Node = null             # the live VoiceStream feeding the session
var _subtitles: Node = null          # karaoke overlay, present when the audio has a sidecar
var _status_t := 0.0     # throttle for writing render progress (export mode)

# THE STAGE (2026-07-27): scenes no longer render in the root viewport. They
# live inside a SubViewport presented through a TextureRect UNDER the UI, so
# a heavy scene's cost can be governed independently: when the frame rate
# sags, the governor renders the STAGE every 2nd/3rd frame (and at reduced
# resolution) while the UI keeps compositing at full rate. The scenes get
# slower; the instrument stays playable. Export mode bypasses the governor
# entirely - renders are always full-rate, full-resolution.
var _stage: SubViewport = null
var _stage_view: TextureRect = null
# The governor's metric is the cost of STAGE-ACTIVE frames ONLY. The blended
# frame rate is a LIE while throttling: the skipped frames are cheap, the
# average looks healthy, the governor de-escalates straight back into the
# wall (measured: 7 fps -> level 2 -> "86 fps" -> level 0 -> 7 fps, forever).
# Active-frame cost keeps reading the scene's true weight at any level.
var _stage_cost := 0.0               # ms EMA of frames where the stage ran
var _stage_good_t := 0.0             # seconds the cost has stayed comfortable
var _stage_change_t := 0.0           # cooldown between level changes
var _stage_was_active := true
var _stage_level := 0
var _stage_frame := 0
# Capped at slow-motion (period 6): a deeper freeze-frame level was tried
# and read as a broken scene - one engine tick per refresh advances the
# SIMULATION 16 ms per 1.5 s, so the picture repainted but nothing moved.
# 1/6 rate is visibly alive; the scene-side optimization work is what
# raises the ceiling for real.
# NO dynamic resolution: per-level render-target scaling was tried and
# every level change REALLOCATED the SubViewport target while the backdrop
# sampled it - black triangular chunks, split/stretched banding, and
# uninitialized-VRAM colour noise on every heavy scene ENTRY (the cut
# resets the level, then the escalation ladder resizes repeatedly). The
# scenes are CPU-bound anyway; resolution bought little and cost all that.
const STAGE_PERIODS := [1, 2, 3, 6]
const STAGE_HEAVY_MS := 24.0         # active frame over this -> escalate
const STAGE_GOOD_MS := 14.0          # under this, sustained -> de-escalate

# Export render mode: this instance was relaunched by the exporter in Movie Maker
# mode (--export). It runs the session clean (no overlays) and quits when the song
# ends, so the recorded movie starts and stops with the music.
var _export_mode := false

func _ready() -> void:
	# window-close is handled by _shutdown (see _notification): the WM close
	# button must run our teardown, not the engine's default instant quit
	get_tree().set_auto_accept_quit(false)
	var args := OS.get_cmdline_user_args()
	# `--deps` before everything: it is the thing you run when nothing else works,
	# so it must not depend on a mode, a window or an audio device coming up. Pairs
	# with `--headless`, and exits non-zero when something a feature needs is
	# missing, so it can gate an install script.
	if args.has("--deps"):
		var rows := Deps.report()
		print(Deps.format_report(rows))
		get_tree().quit(0 if Deps.verdict(rows).is_empty() else 1)
		return
	# Mask mode is a standalone authoring tool (see mask_editor.gd) - tied to one
	# specific external clip, not the audio-reactive show - so it does not touch
	# Director/Spectrum at all and is checked before everything else.
	if args.has("--mask-render"):
		_begin_mask_render(_arg_value(args, "--mask-render"))
		return
	if args.has("--mask-edit"):
		_open_mask_editor(_arg_value(args, "--mask-edit"))
		return
	_export_mode = args.has("--export")
	if _export_mode:
		# Background render: Boot (first autoload) already hid the window as early as
		# possible, and the exporter's override.cfg has set the output resolution + stretch
		# mode at engine startup (Movie Maker locks resolution before any script runs, so
		# it cannot be set from here). Run clean and quit when the song ends.
		Spectrum.song_finished.connect(func(): get_tree().quit())
		_begin_session()
		# A render with no audio has NO END: song_finished never fires, so Movie
		# Maker records silence until something kills it (one such run reached
		# 5.7 GB). An export is defined by its audio - if it didn't load, say so
		# loudly and exit non-zero so the exporter reports a failure instead of
		# leaving the user watching 0% forever.
		if not Spectrum.has_audio():
			printerr("ghost: EXPORT ABORTED - the audio did not load (%s)" %
				_arg_value(args, "--audio"))
			get_tree().quit(1)
			return
		# "Automate the Synthesis game (record the UI)": the render otherwise runs
		# clean, but with this flag the Synthesis panel opens over the take and
		# plays itself (see synth_editor.gd's autopilot), so the UI is IN the
		# video. It generates no audio - the take is the audio - and persists
		# nothing, so a render never touches the player's real save.
		if args.has("--synth-autopilot"):
			_open_synth_autopilot()
		return
	# The live window stretches in "canvas_items" mode (set in project.godot): 2D is
	# rasterized at the window's native pixel resolution - so fullscreen is crisp, not an
	# upscale of the 1080p base - while the coordinate system scales *proportionally*, so
	# UI and scene content keep their relative size and snap back exactly when the window
	# returns to its original size. (Export overrides this; see _apply_export_resolution.)
	# The CHROME is the shared session furniture - exporter, assistant, and the
	# per-session ` feedback console - created ONCE here for every interactive
	# mode, so a mode can never again ship without them (see chrome.gd for the
	# lesson that forced this). Persistent across sessions by design: an
	# in-flight export and a queued assistant dispatch must survive session
	# churn and the return to the home screen.
	_chrome = preload("res://scripts/chrome.gd").new()
	add_child(_chrome)
	# When the current song finishes: an AUTO session returns to the home screen;
	# MANUAL and SYNTH sessions are endless - handled inside _on_song_finished.
	Spectrum.song_finished.connect(_on_song_finished)
	# Synthesis mode: the voice editor (see synth_editor.gd). Unlike mask mode it
	# DOES use Director/Spectrum - a rendered take plays as a normal session so
	# the scenes react to the narration - but the session starts on the first
	# throw, not at boot. Checked here, BELOW the chrome, deliberately: synth is
	# an interactive mode and gets the full furniture like every other.
	if args.has("--synth"):
		_open_synth_editor()
		return
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
	# THE BOOKEND, decided before the first sample - Spectrum.begin starts playback, so
	# it cannot be set afterwards. Whether the silence is HELD here or already baked into
	# the audio is resolved inside begin(), which is the only place that knows which file
	# actually got loaded.
	Spectrum.lead_in = Director.intro_hold
	Spectrum.tail = Director.outro_hold
	Spectrum.begin(audio_path)
	Director.attach(_stage_host())
	_attach_subtitles()
	if _export_mode:
		return                         # render clean: no overlays (the Director fades the video ends)
	_feedback = _chrome.attach_feedback()   # press ` to critique a scene


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
	for n in [_workspace, _splash, _subtitles]:
		if n != null and is_instance_valid(n):
			n.queue_free()
	_chrome.detach_feedback()
	_feedback = null
	_workspace = null
	_subtitles = null
	Director.set_game_paced(false)
	Director.set_aura(0.0)
	Director.detach()
	Spectrum.stop()
	_show_splash()


## The stage the Director attaches scenes to (created on first session). A
## SubViewport with its own 3D world, shown through a TextureRect at child
## index 0 - everything UI stacks above it in the ROOT viewport, which is
## what keeps the instrument responsive when a scene gets heavy.
func _stage_host() -> Node:
	if _stage != null:
		return _stage
	_stage = SubViewport.new()
	_stage.own_world_3d = true
	_stage.transparent_bg = false
	_stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_stage)
	_stage_view = TextureRect.new()
	_stage_view.texture = _stage.get_texture()
	_stage_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_stage_view.stretch_mode = TextureRect.STRETCH_SCALE
	# the backdrop must never eat input: a fullscreen Control's default
	# MOUSE_FILTER_STOP swallows every click that misses a panel
	_stage_view.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_stage_view)
	move_child(_stage_view, 0)       # the stage is the backdrop, always
	get_viewport().size_changed.connect(_sync_stage_size)
	# every scene gets a FRESH measurement: without this, a light scene that
	# follows a heavy one stays imprisoned at the old level for the ~15 s the
	# sparse active samples need to forgive (measured, and it read as broken)
	Director.scene_cut.connect(func():
		_stage_cost = 0.0
		_stage_good_t = 0.0
		if _stage_level > 0:
			_stage_level = 0
			_stage_change_t = 0.0
			_sync_stage_size()
			_stage.process_mode = Node.PROCESS_MODE_INHERIT
			_stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
			print("ghost: stage governor -> level 0 (scene cut)"))
	_sync_stage_size()
	return _stage


func _sync_stage_size() -> void:
	if _stage == null:
		return
	var base := get_viewport().get_visible_rect().size
	_stage.size = Vector2i(base.round())
	_stage_view.position = Vector2.ZERO
	_stage_view.size = base


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
		return
	# ---- the stage governor: sacrifice SCENE rate, never UI rate ----------
	# A heavy scene's active frame blocks the whole main loop for its full
	# cost - Godot cannot paint UI mid-frame - so the only lever short of
	# rewriting the scene is SPACING those frames out: the UI runs fluid
	# between hitches, and at the freeze-frame level the backdrop becomes a
	# slideshow while the instrument stays fully playable. The metric is the
	# active-frame cost (see the var block: the blended fps is a lie here).
	# Escalation is fast; de-escalation needs the ACTIVE frames themselves
	# to have been cheap for a sustained stretch - a still-heavy scene keeps
	# measuring heavy at any level, so the loop cannot oscillate.
	if _stage == null:
		return
	var ms := delta * 1000.0
	if _stage_was_active:
		if _stage_cost <= 0.0:
			_stage_cost = ms
		else:
			# asymmetric EMA: quick to forgive (a scene cut should release
			# the throttle in a few active samples), steady to accuse
			_stage_cost = lerpf(_stage_cost, ms, 0.35 if ms < _stage_cost else 0.12)
	_stage_change_t += delta
	var want := _stage_level
	if _stage_cost > STAGE_HEAVY_MS:
		_stage_good_t = 0.0
		if _stage_level < STAGE_PERIODS.size() - 1:
			want = _stage_level + 1
	elif _stage_cost < STAGE_GOOD_MS:
		_stage_good_t += delta
		if _stage_good_t > 2.0 and _stage_level > 0:
			want = _stage_level - 1
			_stage_good_t = 0.0
	else:
		_stage_good_t = 0.0
	if want != _stage_level and _stage_change_t > 0.4:
		_stage_level = want
		_stage_change_t = 0.0
		_sync_stage_size()
		print("ghost: stage governor -> level %d (active frame %.0f ms)" % [want, _stage_cost])
		if want == 0:
			_stage.process_mode = Node.PROCESS_MODE_INHERIT
			_stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	_stage_frame += 1
	var active: bool = _stage_level == 0 \
		or _stage_frame % int(STAGE_PERIODS[_stage_level]) == 0
	if _stage_level > 0:
		if active:
			_stage.process_mode = Node.PROCESS_MODE_INHERIT
			_stage.render_target_update_mode = SubViewport.UPDATE_ONCE
		else:
			_stage.process_mode = Node.PROCESS_MODE_DISABLED
	_stage_was_active = active


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		# DEFERRED: this notification is still propagating through the tree,
		# so the children are LOCKED ("calling or emitting") - freeing them
		# here errors out. One idle frame later they are free to free.
		call_deferred("_shutdown")


## WORKAROUND (2026-07-27): with audio input enabled (the voice sampler's
## microphone; project.godot driver/enable_input), the engine's audio-driver
## teardown aborts in glibc on this system ("double free" / "corrupted size
## vs. prev_size" - PipeWire input stream finalization; appeared the day
## enable_input landed and reproduces without ever recording). Every piece
## of app state persists on change, and the stateful editors save in
## _exit_tree - so: free main's children (their save hooks run), then leave
## without the driver teardown. GHOST_CLEAN_EXIT=1 restores the normal quit
## path for debugging; remove this whole workaround when upstream is fixed.
func _shutdown() -> void:
	if not OS.get_environment("GHOST_CLEAN_EXIT").is_empty():
		get_tree().quit()
		return
	# queue_free, never free(): something in the tree is reliably mid-call at
	# close time (even from a deferred hook) and free() errors on locked
	# objects; the engine's own end-of-frame free handles that correctly and
	# still runs every _exit_tree save hook (mask playhead etc.)
	for child in get_children():
		child.queue_free()
	await get_tree().process_frame
	# THE HARD EXIT BELOW IS WHY THIS LINE IS HERE. `OS.kill(OS.get_process_id())` is a
	# SIGKILL on ourselves, so nothing after it runs: no autoload `_exit_tree`, and none of
	# the engine's own child reaping. Every background program ghost started - the export's
	# bake / Movie Maker render / `ffmpeg` transcode, the mask editor's prep passes, the
	# voice host - would simply be inherited by init and carry on. That is the reported
	# bug: both windows shut and ffmpeg still writing to the file, with no "godot" in `ps`.
	#
	# It has to be HERE and not only in `Boot`, because this function is the single funnel
	# for BOTH ways out (the window's close request and Escape), and the Escape path never
	# raises WM_CLOSE_REQUEST at all. `Subprocess` also binds each child to this process in
	# the kernel, which covers a ghost that is killed from outside; this covers the ghost
	# that kills itself.
	Subprocess.reap_all()
	print("ghost: bye (hard exit - audio-input teardown workaround, see main._shutdown)")
	OS.kill(OS.get_process_id())


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
				call_deferred("_shutdown")   # outside the input propagation
											 # (children are locked during it)


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
func _open_synth_editor(mode := "fishing") -> void:
	if mode == "neural":
		_open_generative_editor()
		return
	# No furniture wiring here: the chrome (exporter + assistant, created in
	# _ready for every interactive mode) already covers it - a synth take
	# exports exactly like a song, and ` feedback works over the running show.
	var editor := preload("res://scripts/synth_editor.gd").new()
	editor.begin_stream = _begin_synth_stream
	editor.end_stream = _end_synth_stream
	_synth_editor = editor
	add_child(editor)
	# exports are ready from the very first moment: the exporter can ask the
	# editor to render the current text into a take on demand
	if _chrome.exporter != null:
		_chrome.exporter.take_provider = editor.export_take
		_chrome.exporter.take_ready = editor.can_export_take
		_chrome.exporter.automation_available = true
	_feedback = _chrome.attach_feedback()


## The generative path: a small local neural voice, run by the voice host
## subprocess. Deliberately a SEPARATE editor rather than a backend swap inside
## synth_editor - the fishing game's economy is defined over the procedural
## engine's 25-dimensional genome and a neural backend exposes a speaker id and
## three scalars, so one UI cannot serve both. See VOICE_PLAN.md section 6.
func _open_generative_editor() -> void:
	var editor := preload("res://scripts/generative_editor.gd").new()
	editor.begin_stream = _begin_generative_stream
	editor.end_stream = _end_generative_stream
	_synth_editor = editor
	_generative = editor
	add_child(editor)
	# Export was left wired to the fishing game's rules, so the button stayed
	# greyed with "catch a seed first" - a gate that has no meaning on this path.
	# Here the requirement is simply text and a loaded voice.
	if _chrome.exporter != null:
		_chrome.exporter.take_provider = editor.export_take
		_chrome.exporter.take_ready = editor.can_export_take
		_chrome.exporter.automation_available = false
		# no fishing game on this path, so nothing to record
	_feedback = _chrome.attach_feedback()


## ONE session for the whole chapter. The generative path used to start a fresh
## session per chunk, which meant the Director re-cut and the harmonic seed
## re-derived every few sentences - reported as "the scene is changing too
## often". A generator playback instead lets the editor push each chunk's PCM
## into a single continuous stream, so the show sees one unbroken take however
## many chunks it was made from. Subtitles ride the same array, which
## Subtitles.words explicitly allows to keep growing.
## The closing half of _begin_generative_stream. Everything that function attaches, this one
## lets go of - the stage, the audio stream and the subtitle overlay - so the editor's Stop can
## actually end a reading. It was missing entirely: the generative path was wired `begin_stream`
## and nothing else, so a reading could be started and restarted but never stopped, and the only
## way out was to quit ghost.
func _end_generative_stream() -> void:
	Director.detach()
	Spectrum.stop()
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.queue_free()
	_subtitles = null
	if _generative != null and is_instance_valid(_generative):
		_generative.subtitles = null


func _begin_generative_stream(fp: int, sr: int, words: Array) -> AudioStreamGeneratorPlayback:
	# TEAR DOWN FIRST. This is called again every time the reading restarts - a
	# new text, a voice change, a tone change - and it used to attach a fresh
	# Director stage on top of the previous one without detaching. The old scene
	# stayed in the tree with nothing driving it, so it sat frozen behind the
	# new one and they piled up as settings were changed.
	Director.detach()
	Spectrum.stop()
	var pb: AudioStreamGeneratorPlayback = Spectrum.begin_stream(fp, sr)
	Director.attach(_stage_host())
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.queue_free()
	var subs := preload("res://scripts/subtitles.gd").new()
	subs.words = words          # shared by reference; the editor appends to it
	_subtitles = subs
	add_child(subs)
	if _generative != null and is_instance_valid(_generative):
		_generative.subtitles = subs   # the editor re-bases its clock per frame
	return pb


## Export autopilot: the Synthesis panel, playing itself over the take being
## rendered. Deliberately NOT wired to begin_stream/end_stream - the editor must
## not synthesize live audio (the export's --audio take IS the audio); it only
## drives the panel, HUD, and Director scene jumps. The editor self-activates
## its autopilot from the --synth-autopilot flag in its own _ready.
func _open_synth_autopilot() -> void:
	var editor := preload("res://scripts/synth_editor.gd").new()
	_synth_editor = editor
	add_child(editor)


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
	Director.attach(_stage_host())
	# synthesis is GAME PACED: the fishing owns the cuts - a scene changes when
	# a catch jumps it, so each new scene reads as a reward, not weather
	Director.set_game_paced(true)
	_attach_live_subtitles(stream)
	stream.completed.connect(_on_stream_completed)
	stream.restarted.connect(_on_stream_restarted)
	_synth_active = true


## Release the line: tear the synthesis session back down to the state the
## editor boots in - no voice, no show, no Director - while the editor itself
## stays open. The mirror of _begin_synth_stream; the next throw rebuilds
## everything from scratch.
func _end_synth_stream() -> void:
	if not _synth_active:
		return
	if _stream != null and is_instance_valid(_stream):
		_stream.queue_free()
	_stream = null
	if _subtitles != null and is_instance_valid(_subtitles):
		_subtitles.queue_free()
	_subtitles = null
	Director.set_game_paced(false)
	Director.set_aura(0.0)
	Director.detach()
	Spectrum.stop()
	_synth_active = false


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
