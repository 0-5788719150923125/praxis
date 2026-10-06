extends CanvasLayer
class_name Exporter

## Exporter - render the visualization to a video, in the background, in two steps.
##
## Persistent (main creates it once, never frees it), so an in-flight export and its
## status survive the song ending and the return to the home screen. In a song's
## window it shows an Export button; clicking it asks for a quality (720p@30 /
## 1080p@60 / 4K@60) and a path, then:
##
##   1. BAKE (headless, no window): a one-shot `bake_runner` process analyzes the song
##      into a spectrum cache. This is the slow part, and it now runs windowless and
##      off the render's critical path - so there is no gray, frozen render window.
##      Cached per song, so only the first export of a song pays for it.
##   2. RENDER (Movie Maker): a second process loads that cache (`--bake-file`, no
##      in-process baking) and draws immediately, recording the visualization + audio
##      to a scratch AVI. Its window is moved off-screen by main.
##   3. TRANSCODE (ffmpeg): the scratch AVI is re-encoded to the chosen MP4 (H.264 + AAC)
##      WHILE it is written, and on Linux what has been encoded is given back as it goes and
##      the scratch leaves the folder (see [member _live_encode]). Godot only writes AVI, and
##      AVI is a 32-bit/RIFF container that corrupts past ~4 GB (the 4K exports had a broken
##      index + glitchy audio); the MP4 we ship uses 64-bit offsets, is ~10-20x smaller, and
##      plays everywhere.
##   4. UPLOAD (only when the menu's "Upload to YouTube" is ticked): the saved MP4 goes up to
##      the author's channel, unlisted, described as the mode describes it - see
##      [member upload_provider] and youtube.gd. The sign-in it needs is settled as the export
##      starts, beside the render, while the person is still at the machine.
##
## All three steps are separate processes, polled by PID; status ("Analyzing… / Rendering… /
## Finalizing… / Saved ✓") shows here in the main window. Nothing to watch, nothing to force-quit.

const Bake := preload("res://scripts/bake.gd")
const YouTube := preload("res://scripts/youtube.gd")

# Output quality presets, offered when the Export button is pressed. `w`/`h` is the file's
# resolution and `fps` what Movie Maker records at, so the file is exactly what is chosen here.
#
# `ss` is the SUPERSAMPLE factor: the render runs at ss x the output and ffmpeg resolves it back
# down. That is ghost's antialiasing, because it has no other. Everything is drawn as flat
# CanvasItem triangles (see TriBatch) - a wireframe edge is a thin QUAD, not a line primitive, so
# it gets no smoothing of its own - and Godot's msaa_2d, which would otherwise cover it, is inert
# in the export: rendering the same clip at msaa_2d 0, 4x and 8x under the export's own
# "viewport" stretch override produces BYTE-IDENTICAL frames. (It does work in the live window,
# which is why an export has always looked worse than the session it came from.) Measured at 1.5x:
# the excess energy near Nyquist - the direct spectral signature of aliasing - falls from +114% of
# ground truth to +16% before the codec and from +62% to +2.6% in the delivered file, and the
# fraction of edge pixels with no partial coverage at all goes from 61-73% down to 12-19%.
#
# 1.5x rather than 2x because the cost is the SCRATCH FILE, not the time: drawing is CPU-geometry
# bound and flat with resolution (+8% end to end at 1.5x), but the MJPEG intermediate grows 80%,
# which pulls Godot's 4 GiB AVI wrap in from ~5.7 to ~3.1 minutes of 60 fps video. 2x is better
# still (85% of visibly-wrong edge pixels removed against 48%) but wraps at ~2.0 minutes.
# 4K stays native: 1.5x of it is 5760x3240, which wraps in about a minute.
const QUALITIES := [
	{"label": "720p · 30 fps  (HD, smaller file)", "w": 1280, "h": 720, "fps": 30, "tag": "720p", "ss": 1.5},
	{"label": "1080p · 60 fps  (Full HD)", "w": 1920, "h": 1080, "fps": 60, "tag": "1080p", "ss": 1.5},
	{"label": "4K · 60 fps  (UHD, full resolution)", "w": 3840, "h": 2160, "fps": 60, "tag": "4k", "ss": 1.0},
]
const DEFAULT_QUALITY := 1

# Checkable item in the quality menu: for a synthesis take, record the Synthesis
# workspace itself (panel and all) instead of a silent narration - and since
# nobody is there to click Throw/Pull in a background render, the game plays
# itself (see synth_editor.gd's `autopilot`). Meaningless for a plain song
# export (no game to automate), so it's disabled when there's no take_provider.
const UI_TOGGLE_ID := 1000
# Generous but arbitrary: a catch's own reel alone can run 45-240s (see
# synth_editor.gd's _begin_hook), so no fixed window guarantees a full throw
# -> catch -> hold/fold cycle. This is long enough to usually show one.
const SYNTH_AUTOPLAY_DURATION := 150.0
## The status line's widest (see [method _place_status]).
const STATUS_W := 588.0
## The menu's YouTube items (see [method _refresh_upload_items]).
const UPLOAD_ID := 1001
const RESUME_ID := 1002
const SIGN_OUT_ID := 1003
const CLIENT_ID := 1004
## What the sign-in dialog says. Google's "Access blocked" page is a dead end that never comes back
## to Ghost Notes, so the dialog names it rather than waiting the whole timeout out in silence.
const SIGN_BLOCKED := ("\"Access blocked\" or \"has not completed the Google verification process\"? "
	+ "While your Google Cloud project is in Testing, only its test users can sign in: add your Google "
	+ "account as a test user (Google Cloud, Google Auth Platform, Audience), then open the page again.")
const SIGN_WAIT := ("Your browser should be showing Google's sign-in. Sign in with the Google account "
	+ "that owns your channel and allow Ghost Notes to upload videos - this window closes by itself.\n\n"
	+ "No page? It may have opened behind another window: open it again, or copy the link into the "
	+ "browser you use.\n\n" + SIGN_BLOCKED)

## Set by a mode that owns its OWN export path, so the shared button steps aside
## instead of sitting dead on top of it. Masking is the one such mode: its export
## is a headless RELAUNCH of the app against the session json, not this bake
## pipeline, so it builds its own button - at the same bottom-right offsets,
## because that is where an export button lives in every mode. This layer is 250
## and the editor's is 100, so without this the shared button is drawn OVER the
## working one and eats its clicks while grayed out, which is exactly how it was
## reported: "the export button is grayed-out and I cannot click it".
var suppressed := false
## How far up the current mode has asked the bottom-right row to sit. See
## Chrome.bottom_inset.
var _inset := 0.0

var _btn: Button
var _status: Label
var _dialog: FileDialog
var _quality_menu: PopupMenu
var _state := "idle"     # idle | baking | rendering | transcoding | upload_wait | uploading | done
var _bake_pid := -1
var _render_pid := -1
var _transcode_pid := -1
## THE SLIDING WINDOW. The transcode runs DURING the render, reading the AVI as Movie Maker
## writes it (`-follow`), and on Linux the bytes it has already read are released from disk
## behind it (`fallocate --punch-hole`), so the scratch AVI holds seconds of video, not the film.
## A 250 GB intermediate for a few-GB result was the report; measured on a 1080p render: 47 MB
## written, 9.4 MB ever on disk, every frame and the audio intact.
var _live_encode := false        # the transcode is following the render
var _punched := 0                # bytes released from the head of the AVI so far
var _punch_t := 0.0
## ...AND THE SCRATCH LEAVES THE FOLDER. A release frees blocks but cannot shorten the file:
## Movie Maker writes at its own offset, so the AVI's LENGTH still grows to the whole film, and
## the length is what a file manager shows - a second file beside the MP4, "growing constantly"
## (reported 2026-10-06; measured on that render: 13 GB listed, 40 MB on disk). So once a
## release has worked, the name is unlinked. The render and the encoder go on writing and
## reading the same file through the handles they hold, ghost holds one more ([member _hold],
## reached by path as [member _held]) to measure and release through, and the kernel frees what
## is left when the last of them closes - however ghost ends.
var _hold: FileAccess = null
var _held := ""                  # /proc/<ghost>/fd/<n> for _hold; "" = reach the scratch by name
var _unlinked := false           # the scratch's name is gone from the folder
var _released := false           # some of it was given back, so it can never be encoded again
## True when this render is running on the real desktop rather than a display of its own -
## the state in which burying the window corrupts the picture. Shown to the user, because a
## six-hour job that can be spoiled by another window needs to say so BEFORE it is spoiled.
var _note_no_virtual_display := false
var _out := ""           # the final file the user chose (.mp4)
var _avi := ""           # the intermediate Movie Maker AVI (transcoded away, then deleted)
var _song := ""
var _cache := ""
var _done_t := 0.0
var _pct := 0            # last progress read from the render/bake process
var _song_dur := 0.0     # song length captured at export START (the live song may end mid-transcode)
var _quality: Dictionary = QUALITIES[DEFAULT_QUALITY]
var _announced := false  # one-shot console note the first time export becomes ready
var _note_t := 0.0       # countdown for self-clearing informational notes
var _prepping := false   # a provider take is rendering (async) - ignore re-clicks
var _stall_t := 0.0      # seconds since the render last showed ANY sign of life
var _stall_frac := -1.0  # high-water fractional progress the watchdog has seen
var _stall_size := 0     # high-water size of the movie file being written
var _synth_autoplay := false   # UI_TOGGLE_ID checked at export time (synth takes only)
var _yt: YouTube               # the sign-in and the upload
var _client_dialog: FileDialog # the Google client file, asked for once
var _upload := false           # the menu's "Upload to YouTube" box
var _upload_this := false      # this export goes to YouTube once it is saved
var _upload_meta := {}         # what it goes up as: asked of the mode once the take was rendered
var _upload_file := ""         # the file going up (an earlier export's, on a resume)
var _sign := ""                # the sign-in an upload needs: "" | checking | signing_in | ok | failed
var _sign_why := ""
var _sign_attempt := 0         # which sign-in may still report: a newer one, or a cancel, moves it on
var _sign_then := Callable()   # what the sign-in leads on to, kept for "Try again"
var _sign_dialog: AcceptDialog # what Ghost Notes is waiting for while the browser signs in
var _sign_again: Button
var _sign_copy: Button
var _reopen := false           # the menu reopens after the sign-in: its box stays ticked

# The watchdog exists for ONE failure: a render that can never finish (the
# audio failed to load, so the session has no end and Movie Maker records
# silence until the disk fills). It must never fire on a render that is merely
# SLOW - heavy scenes at 4K can spend minutes on a few seconds of video.
#
# Liveness is two independent FINE-GRAINED signals - the fractional playback
# position and the movie file GROWING on disk - and either one counts, with a
# long fuse. Integer percent is far too coarse to mean "alive": one percent point
# of a 344 s take is 3.4 s of video, which a heavy scene can spend over a minute
# producing, and a healthy 720p render was killed for it.
const STALL_LIMIT := 300.0
## How long the following encoder waits on a file that has stopped growing before it decides the
## render is over (microseconds, ffmpeg's unit). Longer than any pause a live render makes; a
## render that does pause longer is caught by [method _encoder_quit_early].
const FOLLOW_TIMEOUT_US := 60000000
## Seconds the render's virtual display outlives its last client (see [method virtual_display]).
const XVFB_LINGER := 10
## The head of the AVI is never released: Movie Maker seeks back there to finish its header.
const PUNCH_KEEP := 1 << 20
## ...and the release stays this far behind the encoder's read position.
const PUNCH_BEHIND := 8 << 20
const STALL_MIN_GROWTH := 65536   # bytes; below this the file is not really moving

## Synthesis hook: a mode whose audio is REPRODUCIBLE ON DEMAND (the voice is
## a pure function of the text and the seed) registers a provider that renders
## the current take to disk and returns its path. Export then works from the
## very first moment - the click IS the trigger that makes the audio.
var take_provider := Callable()

## Optional companion to [member take_provider]: returns true when the provider
## has something worth rendering (in synthesis: at least one seed on the belt,
## or a voice already cast). When it returns false the button grays out - a
## click could only produce a video of nothing.
var take_ready := Callable()

## What the video is called, when the mode knows: returns a name without an extension (an
## episode's title, say), or "" for the exporter's own `ghost_notes_<quality>`. Asked as the save
## dialog opens; whatever it returns goes through [method safe_name].
var name_provider := Callable()

## Whether this mode has a UI worth recording. Only the fishing game does; the
## generative path has a text box and some sliders, and an option offering to
## "record the game" there is noise.
var automation_available := false

## What an upload of the take at the given path says: `{title, description, tags, record}` -
## `record` is a file the upload's result is kept in - or {} when the mode has nothing to upload.
## Asked with "" for a look at what an upload would be now: the menu offers "Upload to YouTube"
## only when that finds something. Asked again with the take once it is rendered, so a mode can
## time chapters from it and describe the episode the take was made of.
var upload_provider := Callable()


func _ready() -> void:
	layer = 250          # above the splash (200), so status shows on the home screen too
	_clear_override()    # remove a stale override.cfg left by a crashed/killed render
	_build_ui()
	_place_status()
	get_viewport().size_changed.connect(_place_status)


## AN EXPORT DOES NOT SURVIVE THE APP, and until this existed it did. Every step of the
## export is a separate process (see the class doc) and `OS.create_process` detaches them
## completely, so closing ghost mid-export left the bake, the Movie Maker render or the
## `ffmpeg` transcode running with nothing on screen, no way to stop them, and no "godot"
## in `ps` to explain what was still writing to the disk. Reported from the transcode: both
## windows shut, ffmpeg still finalizing the file.
##
## The pids go through [Subprocess], so `Boot` would reap them anyway - this is the owner
## doing it first and at the right moment, and it also clears the render's `override.cfg`,
## which is the one piece of shutdown state the generic reap cannot know about. The scratch AVI
## goes with the render: nothing resumes a render, and what the encoder read has already been
## given back, so a scratch left behind is only a large file in the author's folder.
##
## WM_CLOSE_REQUEST reaches every node when the window is asked to close; EXIT_TREE covers
## the programmatic-quit paths. Both, because either can be the one that happens.
func _notification(what: int) -> void:
	if what != NOTIFICATION_WM_CLOSE_REQUEST and what != NOTIFICATION_EXIT_TREE:
		return
	if _state == "idle" or _state == "done":
		return
	if _state in ["preparing", "upload_wait", "uploading"]:
		# the video is saved and the upload waits in pending.json: nothing to stop or clear
		print("ghost: YouTube upload interrupted - ghost is closing; resume it from the ⤓ menu")
		return
	for pid in [_bake_pid, _render_pid, _transcode_pid]:
		Subprocess.stop(int(pid))
	_bake_pid = -1
	_render_pid = -1
	_transcode_pid = -1
	_clear_override()
	_drop_scratch()
	print("ghost: export stopped - ghost is closing")


func _build_ui() -> void:
	_btn = Button.new()
	_btn.text = "⤓"                    # icon-only - matches assistant.gd's chat-bubble toggle
	_btn.tooltip_text = "Render this visualization + audio to a video file (in the background)"
	_btn.focus_mode = Control.FOCUS_NONE
	_btn.custom_minimum_size = Vector2(40, 40)
	_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# Same 40x40 box, same row (-28/-68), as assistant.gd's toggle, right of this one -
	# see that file's _TOGGLE_SIZE/_TOGGLE_ROW_BOTTOM doc for why the numbers match.
	_btn.offset_left = -112
	_btn.offset_top = -68
	_btn.offset_right = -72
	_btn.offset_bottom = -28
	_btn.visible = false
	_btn.modulate.a = 0.0       # fade in elegantly when it becomes eligible
	_btn.pressed.connect(_on_export)
	add_child(_btn)

	_status = Label.new()
	_status.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_status.vertical_alignment = VERTICAL_ALIGNMENT_BOTTOM
	# WRAPPED, and growing UP and LEFT - never toward the buttons or past the screen's edge
	# (see _place_status)
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	_status.grow_vertical = Control.GROW_DIRECTION_BEGIN
	_status.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.7))
	_status.add_theme_constant_override("shadow_offset_x", 1)
	_status.add_theme_constant_override("shadow_offset_y", 1)
	_status.visible = false
	add_child(_status)
	_place_status()

	# Quality picker - shown first when Export is pressed, before the save dialog.
	_quality_menu = PopupMenu.new()
	for i in QUALITIES.size():
		_quality_menu.add_item(QUALITIES[i].label, i)
	_quality_menu.add_separator()
	_quality_menu.add_check_item("Automate the Synthesis game (record the UI)", UI_TOGGLE_ID)
	_quality_menu.set_item_tooltip(_quality_menu.get_item_index(UI_TOGGLE_ID),
		"Synthesis takes only: instead of a silent narration, plays the fishing game\n"
		+ "itself - Throw, Pull, hold or fold - live, with its panel visible in the video.")
	# Checkable items close the popup like any other by default - that would eat the
	# toggle on the very click meant to set it, so only quality items (uncheckable)
	# close it.
	_quality_menu.hide_on_checkable_item_selection = false
	_quality_menu.id_pressed.connect(_on_quality)
	add_child(_quality_menu)

	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.use_native_dialog = true
	_dialog.title = "Export video"
	_dialog.filters = PackedStringArray(["*.mp4 ; Video (MP4, H.264 + AAC)"])
	_dialog.current_file = "ghost_notes.mp4"
	# Default to the Downloads folder so an export lands somewhere predictable (the native dialog
	# otherwise opens in its last-used directory, which is easy to lose track of).
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_dialog.current_dir = downloads
	_dialog.size = Vector2i(800, 560)
	_dialog.file_selected.connect(_on_path)
	add_child(_dialog)

	# The author's Google client file, asked for the first time an upload is ticked and kept from
	# then on (see youtube.gd).
	_client_dialog = FileDialog.new()
	_client_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_client_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_client_dialog.use_native_dialog = true
	_client_dialog.title = "Import your Google OAuth client (the JSON file from Google Cloud)"
	_client_dialog.filters = PackedStringArray(["*.json ; Google OAuth client (JSON)"])
	if not downloads.is_empty():
		_client_dialog.current_dir = downloads
	_client_dialog.size = Vector2i(800, 560)
	_client_dialog.file_selected.connect(_on_client_file)
	add_child(_client_dialog)
	_yt = YouTube.new()
	add_child(_yt)
	# The sign-in, visible while it waits: the page can open behind another window, and Google's
	# "Access blocked" never comes back, so the person needs to see what is awaited and act on it.
	_sign_dialog = AcceptDialog.new()
	_sign_dialog.title = "Sign in to YouTube"
	_sign_dialog.dialog_autowrap = true
	_sign_again = _sign_dialog.add_button("Open the page again", false, "again")
	_sign_copy = _sign_dialog.add_button("Copy the link", false, "copy")
	_sign_dialog.custom_action.connect(_on_sign_action)
	_sign_dialog.confirmed.connect(_on_sign_close)
	_sign_dialog.canceled.connect(_on_sign_close)
	add_child(_sign_dialog)


func _process(dt: float) -> void:
	match _state:
		"baking":
			if Subprocess.alive(_bake_pid):
				_poll_pct()
				_set_status("⏳  Analyzing audio…  %d%%" % _pct, Color(0.95, 0.92, 0.7))
			elif FileAccess.file_exists(_cache):
				_start_render()                  # bake finished -> render from the cache
			else:
				_fail("⚠  Bake failed (is ffmpeg on PATH?)")
		"rendering":
			_tick_render(dt)
		"transcoding":
			_tick_transcode()
		"preparing":
			_set_status("⇪  Taking the thumbnail for %s…" % _out.get_file(), Color(0.95, 0.92, 0.7))
		"upload_wait":
			# the video is saved and queued; it goes up once the sign-in settles
			match _sign:
				"ok":
					_run_upload()
				"failed":
					if YouTube.signed_in():
						_run_upload()        # a sign-in is kept: what failed was reaching Google - try now
					else:
						_state = "done"
						_done_t = 60.0
						_set_status("⚠  Not uploaded: %s - resume it from the ⤓ menu" % _sign_why,
							Color(1.0, 0.7, 0.6))
				"signing_in":
					_set_status("⇪  %s waits for the YouTube sign-in in your browser" % _upload_file.get_file(),
						Color(0.95, 0.92, 0.7))
				_:
					_set_status("⇪  Checking the YouTube sign-in…", Color(0.95, 0.92, 0.7))
		"uploading":
			_set_status("⇪  Uploading %s to YouTube …  %d%%" % [_upload_file.get_file(),
				int(round(_yt.progress * 100.0))], Color(0.95, 0.92, 0.7))
		"done":
			_done_t -= dt
			if _done_t <= 0.0:
				_status.visible = false
				_state = "idle"
	# the take renders with the state still idle: keep its line, and the sign-in beside it, current
	if _prepping and _upload_this:
		_set_status("⏳  Rendering the take…", Color(0.95, 0.92, 0.7))
	# The button fades in once eligible (idle, not mid-export, past the delay) and
	# fades out otherwise - never a hard pop.
	# an informational note (e.g. "nothing to export yet") clears itself
	if _note_t > 0.0 and _state == "idle":
		_note_t -= dt
		if _note_t <= 0.0:
			_status.visible = false
	if suppressed:
		_btn.visible = false
		_btn.modulate.a = 0.0
		# The STATUS line stays: an export started before the mode took over is
		# still running, and its progress is worth seeing.
		return
	var want := _state == "idle" and _can_export()
	if want and not _announced:
		_announced = true
		print("ghost: export ready (⤓ bottom-right)")
	_btn.modulate.a = lerpf(_btn.modulate.a, 1.0 if want else 0.0, 1.0 - exp(-6.0 * dt))
	_btn.visible = _btn.modulate.a > 0.02
	# grayed, never gone: nothing to render yet (see _can_export's history)
	var content := _has_content()
	_btn.disabled = not content
	_btn.tooltip_text = ("Render this visualization + audio to a video file (in the background)"
		if content else
		"Nothing to render yet - catch a seed (or play a song) first")


func _tick_render(dt: float) -> void:
	if not Subprocess.alive(_render_pid):
		_clear_override()                # render finished -> restore live resolution
		# The render only reports success by PID exit; make sure it actually produced the AVI
		# (a crashed Movie Maker exits too) before spending minutes transcoding nothing.
		if _live_encode:
			# the encoder has been following all along; it finishes by itself once the
			# file stops growing (FOLLOW_TIMEOUT_US)
			_state = "transcoding"
		elif FileAccess.file_exists(_avi) and _file_size(_avi) > 65536:
			_repair_avi_sizes(_avi)
			_start_transcode()
		else:
			_fail("⚠  Render produced no file (see console)")
		return
	_poll_pct()
	# STALL WATCHDOG (see the consts): alive = the playback
	# position advanced at ALL, or the movie file grew. A slow
	# render satisfies both; only a render that cannot finish
	# satisfies neither.
	_stall_t += dt
	var frac := Bake.read_progress()
	if frac > _stall_frac:
		_stall_frac = frac
		_stall_t = 0.0
	var sz := _scratch_len()
	if not _live_encode and _transcode_pid <= 0 and sz > 65536:
		_start_transcode(true)
	if _live_encode and not Subprocess.alive(_transcode_pid) and _riff_size() == 0:
		_encoder_quit_early()
		if _state != "rendering":
			return
	if _live_encode:
		_punch_t -= dt
		if _punch_t <= 0.0:
			_punch_t = 2.0
			_punch_behind_encoder()
	if sz > _stall_size + STALL_MIN_GROWTH:
		_stall_size = sz
		_stall_t = 0.0
	if _stall_t > STALL_LIMIT:
		Subprocess.stop(_render_pid)
		_clear_override()
		_drop_scratch()
		_fail("⚠  Render stalled at %d%% - frozen for %d min (see console)"
			% [_pct, int(STALL_LIMIT / 60.0)])
		push_warning("ghost export: render stalled (no frames, no progress); killed pid %d"
			% _render_pid)
		return
	# one decimal, deliberately: whole percents on a long take
	# sit still for minutes on heavy scenes, which reads as a
	# freeze. A moving number is the difference between "slow"
	# and "hung" for the person watching it.
	# THE WARNING RIDES THE PROGRESS LINE, not a one-shot notice: this is
	# the only text on screen for hours, and "leave the render window
	# alone" is advice that has to still be visible at hour five.
	var how := " …  %.1f%%" % (maxf(_stall_frac, 0.0) * 100.0)
	if _note_no_virtual_display:
		how += "   ⚠ leave the render window visible"
	_set_status("⏺  Rendering %s%s" % [_out.get_file(), how],
		Color(0.95, 0.92, 0.7))


func _tick_transcode() -> void:
	if Subprocess.alive(_transcode_pid):
		_set_status("⏳  Finalizing %s …  %d%%" % [_out.get_file(), _read_transcode_pct()], Color(0.95, 0.92, 0.7))
	elif FileAccess.file_exists(_out) and _file_size(_out) > 4096:
		_drop_scratch()                   # transcode ok -> the scratch goes, and its space with it
		_state = "done"
		_done_t = 30.0
		_set_status("✓  Saved  %s" % _out, Color(0.82, 0.95, 0.86))
		print("ghost: exported -> ", _out)
		if _upload_this:
			_queue_upload()
	elif _released:
		# what the encoder read was given back as it went, so there is no whole AVI to keep
		_upload_this = false
		_drop_scratch()
		_state = "done"
		_done_t = 30.0
		_set_status("⚠  Transcode failed (see console)", Color(1.0, 0.7, 0.6))
		push_warning("ghost export: transcode failed")
	else:
		# Transcode failed (ffmpeg missing/errored). Keep the raw AVI so the work isn't lost.
		_upload_this = false
		_drop_scratch(false)
		_state = "done"
		_done_t = 30.0
		_set_status("⚠  Transcode failed; raw file kept: %s" % _avi, Color(1.0, 0.7, 0.6))
		push_warning("ghost export: transcode failed; kept AVI at " + _avi)


# THE BUTTON IS ALWAYS VISIBLE. History, because this exact gate regressed
# repeatedly: eligibility was time-gated (30s of playback / half the song),
# then length-gated for streams, then time-gated again by a concurrent edit
# whose premise (that synthesis paces to real time) was wrong - and in
# synthesis sessions EVERY throw/edit resets the playback clock via the
# stream restart cycle, so a time gate can NEVER be satisfied while the user
# iterates. Each version HID the button from someone.
#
# So the rule is: never hide it, and never gate it on TIMING. It may only be
# DISABLED (grayed, still there, with a tooltip that says why) for the one
# honest reason - there is no content: no song loaded, and no provider that
# could make one (synthesis with an empty belt and nothing cast).
func _can_export() -> bool:
	return true


func _has_content() -> bool:
	if not Spectrum.audio_path().is_empty():
		return true
	if not take_provider.is_valid():
		return false
	return not take_ready.is_valid() or bool(take_ready.call())


# Refresh the cached percentage from the worker process (ignore mid-write misreads).
func _poll_pct() -> void:
	var p := Bake.read_progress()
	if p >= 0.0:
		_pct = int(round(p * 100.0))


# Step 0: pick the output resolution / fps. Pop the menu up by the button.
# THE PICKERS COME FIRST - the same order as every other session: quality,
# then path, then the background pipeline. A synthesis take that still needs
# rendering is made in _on_path, AFTER the user commits - doing it here made
# the click sit on "Rendering the take…" with no dialog in sight, which read
# as a button that never asks anything (and a cancel cost a wasted render).
# A click with nothing to export and no way to make it explains itself
# instead of opening a doomed dialog.
func _on_export() -> void:
	if _prepping:
		return
	# Every export ends in an ffmpeg transcode. ghost downloads FFmpeg itself, so on a
	# first run it may still be on its way - say so now rather than after the render.
	if not Deps.has("ffmpeg"):
		Provision.ensure("ffmpeg")
		_note_t = 6.0
		_set_status("⏳  " + Deps.hint("ffmpeg"), Color(1.0, 0.85, 0.6))
		return
	_song = Spectrum.audio_path()
	if _song.is_empty() and not take_provider.is_valid():
		_note_t = 4.0
		_set_status("⚠  Nothing to export yet - play or speak something first",
			Color(1.0, 0.85, 0.6))
		return
	# Only the FISHING game has a game to automate. take_provider is no longer a
	# proxy for that - the generative path sets one too - so the option is driven
	# by an explicit flag and REMOVED rather than grayed where it is meaningless,
	# since a permanently disabled item just invites the question of what it is.
	var ui_idx := _quality_menu.get_item_index(UI_TOGGLE_ID)
	if automation_available and ui_idx < 0:
		_quality_menu.add_check_item("Automate the Synthesis game (record the UI)", UI_TOGGLE_ID)
	elif not automation_available and ui_idx >= 0:
		_quality_menu.remove_item(ui_idx)
	# THE UPLOAD BOX STARTS CLEAR every time - an upload is public-facing - except when the menu
	# comes back after the client file was imported, which happened because it was ticked
	if not _reopen:
		_upload = false
	_refresh_upload_items()
	var btn_rect := _btn.get_global_rect()
	_quality_menu.reset_size()
	var pos := Vector2i(btn_rect.position) + Vector2i(0, -int(_quality_menu.get_contents_minimum_size().y) - 8)
	_quality_menu.position = pos
	_quality_menu.popup()


func _on_quality(id: int) -> void:
	if id == UI_TOGGLE_ID:
		_synth_autoplay = not _synth_autoplay
		_quality_menu.set_item_checked(_quality_menu.get_item_index(UI_TOGGLE_ID), _synth_autoplay)
		return
	match id:
		UPLOAD_ID:
			_toggle_upload()
			return
		RESUME_ID:
			_resume_upload()
			return
		SIGN_OUT_ID:
			_sign_out()
			return
		CLIENT_ID:
			_client_dialog.popup_centered()
			return
	_quality = QUALITIES[id]
	var named := safe_name(String(name_provider.call())) if name_provider.is_valid() else ""
	_dialog.current_file = ("%s.mp4" % named) if not named.is_empty() else "ghost_notes_%s.mp4" % _quality.tag
	_dialog.popup_centered()


## [param title] as a file name every system accepts, still reading as the title: a colon becomes
## " -", the characters Windows refuses go, whitespace collapses, no trailing dot or space, and
## it stops at a word before 120 characters. "" when nothing is left.
static func safe_name(title: String) -> String:
	var t := title.replace(":", " -")
	var out := ""
	for ch in t:
		if ch.unicode_at(0) >= 32 and not "/\\*?\"<>|".contains(ch):
			out += ch
	out = Manuscript._rx("\\s+").sub(out, " ", true).strip_edges()
	if out.length() > 120:
		out = out.substr(0, 120)
		var cut := out.rfind(" ")
		if cut > 60:
			out = out.substr(0, cut)
	while out.ends_with(".") or out.ends_with(" "):
		out = out.left(-1)
	return out


func _on_path(out_path: String) -> void:
	# RE-ENTRANCY GUARD, and it is load-bearing: this method awaits a take
	# render, and the native file dialog can deliver file_selected more than
	# once (and the user can re-export while a take is still rendering). Two
	# overlapping runs rendered the SAME take file from two threads - a reader
	# in another process (the export render itself) opened it mid-write and
	# saw a truncated WAV, which is how a render ended up recording silence
	# forever. write_wav is atomic now; this keeps the work from doubling too.
	if _prepping:
		return
	# THE UPLOAD IS DECIDED WITH THE PATH: the box is read once and cleared for the next export, and
	# the sign-in starts now - while the person is still at the machine to answer the browser - and
	# runs beside the render rather than in front of it.
	_upload_this = _upload and not _upload_peek().is_empty()
	_upload = false
	_upload_meta = {}
	if _upload_this:
		_check_sign_in()
	# _song was resolved at click time (_on_export) - the live audio path. A
	# synthesis session without a finished take renders one NOW, after the
	# quality and path are committed: the provider is a coroutine that runs
	# the synthesis on a worker thread (rendering it on the main thread froze
	# the window while the audio kept playing), and awaiting a coroutine
	# through a Callable is verified to work - the await is required.
	if _song.is_empty() and take_provider.is_valid():
		_prepping = true
		_set_status("⏳  Rendering the take…", Color(0.95, 0.92, 0.7))
		_song = String(await take_provider.call())
		_prepping = false
	if _song.is_empty():
		_song = Spectrum.audio_path()
	if _song.is_empty():
		_fail("⚠  No song to export")
		return
	_out = out_path
	if _out.get_extension().to_lower() != "mp4":
		_out += ".mp4"
	# what the upload says, asked of the mode now that the take exists (a mode times chapters from it)
	if _upload_this:
		var meta: Variant = upload_provider.call(_song) if upload_provider.is_valid() else {}
		_upload_meta = meta if meta is Dictionary else {}
		if _upload_meta.is_empty():
			_upload_this = false
			push_warning("ghost export: the mode described nothing to upload - the video is only saved")
	# Capture the duration NOW, while the song is loaded: the transcode (esp. 4K) runs for minutes, by
	# which point the live song may have ended/unloaded and Spectrum.song_length() would read 0.
	_song_dur = Spectrum.song_length()
	if _song_dur <= 0.0 and _song.get_extension().to_lower() == "wav":
		# a provider-rendered take Spectrum hasn't finished streaming: PCM16
		# mono WAV, so the file itself says how long it is.
		#
		# READ THE RATE FROM THE HEADER. This assumed Voice.SR (44100), which is
		# only true for the procedural synthesizer's own takes; a neural voice
		# renders at its model's rate - Piper is 22050 - so a generative export
		# reported exactly HALF its length and the render stopped halfway
		# through, which is what "it only produced 5 seconds" was.
		_song_dur = maxf(0.0, float(_file_size(_song) - 44)
			/ (2.0 * float(_wav_rate(_song))))
	# Say what was decided, in the terminal. Every export failure so far has been
	# a disagreement about WHICH file and HOW LONG - a stale path, or a duration
	# read at the wrong sample rate that cut the render in half - and neither is
	# visible from the UI.
	print("ghost/export: take=%s  %.2fs @ %d Hz  ->  %s"
		% [_song, _song_dur, _wav_rate(_song) if _song.get_extension().to_lower() == "wav" else 0, _out])
	if _song_dur <= 0.5:
		printerr("ghost/export: WARNING - the take reads as %.2fs. The render stops "
			% _song_dur + "when the audio ends, so the video will be that short.")

	# Say what was decided, in the terminal. Every export problem so far has been
	# a disagreement about WHICH file or HOW LONG - a duration read at the wrong
	# sample rate halved one render - and neither is visible from the UI.
	print("ghost/export: take=%s" % _song)
	print("ghost/export: %.2fs @ %d Hz -> %s" % [_song_dur,
		_wav_rate(_song) if _song.get_extension().to_lower() == "wav" else 0, _out])
	if _song_dur <= 1.0:
		printerr("ghost/export: WARNING - the take reads as %.2fs, so the render "
			% _song_dur + "will stop there. Check the text in the panel.")

	# Movie Maker records to this intermediate AVI (beside the final file, on the same disk), and it
	# is encoded to the chosen .mp4 as it is written - on Linux given back as it goes and taken out
	# of the folder (see _hold). The AVI is only ever scratch - it never ships, so its 4 GB/RIFF
	# index limit (which corrupts 4K exports) can't reach the user.
	_avi = _out.get_basename() + ".render.avi"
	_cache = Bake.cache_path(_song)
	if FileAccess.file_exists(_cache):
		_start_render()                          # already analyzed -> straight to render
	else:
		_start_bake()


# Step 1: headless analysis (no window). Writes the spectrum cache, then exits.
func _start_bake() -> void:
	_pct = 0
	Bake.write_progress(0.0)
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	_bake_pid = Subprocess.start(exe, PackedStringArray([
		"--headless", "--path", project, "--script", "res://scripts/bake_runner.gd",
		"--", "--bake-song", _song, "--bake-out", _cache]), "bake")
	if _bake_pid > 0:
		_state = "baking"
		print("ghost: analyzing audio (pid ", _bake_pid, ") -> ", _cache)
	else:
		_fail("⚠  Could not start the analysis process")


# Step 2: Movie Maker render that loads the cache (--bake-file) and draws at once.
func _start_render() -> void:
	_pct = 0
	_begin_scratch()
	Bake.write_progress(0.0)
	# Movie Maker locks its output resolution to the project's viewport size at engine
	# startup, before any script runs - so the only way to drive it is override.cfg, which
	# Godot reads from the project root at boot. We write the chosen resolution there (in
	# "viewport" stretch mode, so the render is an offscreen buffer of exactly that size,
	# independent of the physical display - true 4K on a 1080p monitor). It is removed
	# when the render finishes, restoring the live window to its native canvas_items mode.
	# Render at the supersampled size; the transcode resolves it back to w x h (see QUALITIES).
	# Kept EVEN, because yuv420p needs even dimensions and a half-pixel would be rejected.
	_write_override(_render_w(), _render_h())
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	var args := PackedStringArray([
		"--path", project, "--write-movie", _avi, "--fixed-fps", str(_quality.fps),
		"--", "--export", "--bake-file", _cache, "--audio", _song])
	# THE SEED: pin it only when there is a live session to reproduce. Without
	# one (exporting a take straight from the belt - no cast, no show watched
	# yet) passing session_seed() would pin the render to 0, which is not a
	# session, just a constant. Omitting --seed lets the render resolve the
	# seed the way every songless boot does: from the AUDIO'S OWN fingerprint
	# (see Director._resolve_seed) - spectral determinism, so the same take
	# always renders the same show, and the show belongs to the voice.
	var seed_val := Director.session_seed()
	if seed_val != 0:
		args.append("--seed")
		args.append(str(seed_val))
	# THE MEDIUM, passed explicitly rather than left to the render's own config read. The
	# saved setting would usually agree - the render process loads the same user://ghost.cfg
	# this one wrote - but `--medium` overrides that setting for one run, and a render
	# started from such a session must reproduce THE SESSION, not the remembered default.
	args.append("--medium")
	args.append(Director.resolved_medium())
	if Director.is_manual():
		args.append("--storyboard")
		args.append(Director.storyboard_source())   # the loadable name/path, NOT the display name
	# "Automate the Synthesis game (record the UI)" was checked: tell the render
	# to open the Synthesis panel over the take and let it play itself (Throw /
	# Pull / reel / hold-or-fold on timers). Without this flag the render runs
	# clean, exactly as a plain song does.
	if _synth_autoplay:
		args.append("--synth-autopilot")
	# A DISPLAY OF ITS OWN, so the desktop cannot freeze the recording.
	#
	# THE BUG THIS IS FOR, reported as "the newly-exported video is freezing in some places -
	# the scene and subtitles pause while the audio continues uninterrupted, and it recovers
	# after 20 seconds or so". Measured on the delivered file: a subtitle card sitting
	# nineteen seconds past the end of its own sentence with the karaoke fill frozen
	# mid-word, then snapping back into sync, while the audio ran smoothly through it.
	#
	# Godot does not render while the compositor is not drawing its window
	# (`window_can_draw()`), and the movie writer then re-captures the LAST BUFFER while the
	# audio clock keeps advancing - so the film holds a still for exactly as long as the
	# window was minimized, covered or throttled. boot.gd has carried that explanation since
	# the "4K exports partially freeze for seconds" bug, along with the remedy it could not
	# reach at the time: "avoidable without a virtual display; this gets it as close as
	# possible". Shrinking the window to a 480x270 floater got close - it is drawable, and it
	# is also easy to bury under another window for an hour without thinking about it.
	#
	# A render that takes six hours cannot ask for the machine to be left alone for six
	# hours. On its own X display there is no compositor, nothing to minimize and nothing to
	# throttle: the NVIDIA driver still renders on the real GPU (tests/run_quiet.sh has
	# relied on exactly that for the pixel gates), and the desktop is free.
	#
	# It is NOT required. Without xvfb the render works as it always has and the caller is
	# told what that costs, because a silent fallback here is a six-hour job that may quietly
	# come out wrong.
	var runner := exe
	var run_args := args
	var virtual := virtual_display(exe, args)
	if not virtual.is_empty():
		runner = virtual[0]
		run_args = virtual.slice(1)
		print("ghost export: rendering on a virtual display - the desktop cannot freeze it")
	else:
		# xvfb-run is Linux-only; on Windows and macOS there is no virtual display to offer,
		# so the advice is the whole message there.
		var fix := ("  " + Deps.hint("xvfb")) if OS.get_name() == "Linux" else ""
		push_warning("ghost export: no virtual display; the render window must stay drawable "
			+ "for the WHOLE render. Minimizing or burying it freezes the recorded picture "
			+ "while the audio keeps going." + fix)
		_note_no_virtual_display = true
	_render_pid = Subprocess.start(runner, run_args, "render")
	if _render_pid > 0:
		_state = "rendering"
		_stall_t = 0.0
		_stall_frac = -1.0
		_stall_size = 0
		if _render_w() != int(_quality.w):
			print("ghost: rendering %dx%d -> %dx%d @ %d fps (pid %d) -> %s" % [
				_render_w(), _render_h(), _quality.w, _quality.h, _quality.fps, _render_pid, _avi])
		else:
			print("ghost: rendering %dx%d @ %d fps (pid %d) -> %s" % [
				_quality.w, _quality.h, _quality.fps, _render_pid, _avi])
	else:
		_clear_override()
		_fail("⚠  Could not start the render process")


## The argv that runs `exe args` on a virtual display of its own (see [method _start_render]), or
## [] without xvfb-run. `-a` picks a free display number; the screen must be at least the size of
## the WINDOW (the 480x270 floater), never of the recorded viewport - the movie records the
## viewport, which is independent of the window in "viewport" stretch mode.
##
## GODOT DIES WITH ITS WRAPPER. Subprocess binds xvfb-run's shell to ghost, but a pact does not
## pass to the shell's children: stopping a render killed the shell and left Godot rendering and
## Xvfb running (measured 2026-10-06) - a render that could not be stopped, writing on into a
## scratch nothing reads. So Godot gets a pact with the shell, and Xvfb ends [constant
## XVFB_LINGER] seconds after its last client leaves (the delay covers any connection a client
## opens and closes before its own).
static func virtual_display(exe: String, args: PackedStringArray) -> PackedStringArray:
	var xvfb := Deps.resolve("xvfb-run")
	if xvfb.is_empty():
		return PackedStringArray()
	var argv := PackedStringArray([xvfb, "-a", "-s", "-screen 0 960x540x24 -terminate %d" % XVFB_LINGER])
	argv.append_array(Subprocess.pact_prefix())
	argv.append(exe)
	argv.append_array(args)
	return argv


# Step 3: transcode the scratch AVI into the chosen MP4 (H.264 + AAC) via ffmpeg. This is what the
# user actually keeps: MP4 uses 64-bit offsets so its index is valid at any size (Godot's AVI is
# 32-bit and corrupts past 4 GB, which is why 4K exports had a broken index and glitchy audio), and
# H.264 is ~10-20x smaller than the MJPEG intermediate. `-fflags +genpts` re-derives timestamps so a
# damaged AVI index is bypassed; audio is re-encoded from decoded PCM, so it comes out clean.
func _start_transcode(follow := false) -> void:
	var dur := _song_dur
	if not follow:
		_pct = 0              # reset from the render's 100% so "Finalizing" starts fresh, not stuck full
	_progress_reset()
	# COLOR SIGNALING. Godot's MJPEG is yuvj420p - FULL range, BT.601 matrix - and ffmpeg passes
	# those pixels through untouched (measured: blacks bit-exact, no gamma, mean delta -0.33 of a
	# code value). What it did NOT do was SAY so: primaries and transfer came out "unspecified" and
	# the container had no `colr` box at all, so a player reading only the container guesses - and
	# for 1080p the guess is BT.709 LIMITED range. On this content that guess is ruinous rather
	# than subtle, because ghost is nearly all shadow: 46-91% of pixels sit below code 16, so a
	# limited-range decoder crushes a quarter to a half of the frame to pure black (measured 25.0%
	# of one frame lost outright, mean luma 17.7 -> 8.3). That is the "muddy, washed out, something
	# is wrong" half of a bad-looking export, with nothing actually wrong in the pixels.
	#
	# `setparams` is required rather than decorative: the bare -color_primaries/-color_trc OUTPUT
	# options are silently ignored by ffmpeg 8.x (verified - they left primaries=2, trc=2 in the
	# VUI). setparams stamps the FRAMES, which is what the encoder reads. Pixels are bit-identical
	# either way; the file grows by 19 bytes.
	#
	# The matrix stays bt470bg (BT.601) because that is genuinely what Godot's writer used -
	# converting it to 709 costs 6.2 dB for nothing. And do NOT reach for ffmpeg's `colorspace`
	# filter to force limited range: it CLAMPS instead of scaling here, pinning 92% of the frame to
	# code 16 and collapsing PSNR from 46.8 to 29.0 dB while exiting 0.
	var gop := int(_quality.fps) * 2       # 2 s keyframe interval - YouTube's guidance (x264 defaults to 250)
	# THE DOWNSCALE is the antialiasing resolve (see QUALITIES). lanczos rather than bicubic: bicubic
	# scores marginally higher on edge PSNR but lands ~18% BELOW ground truth in near-Nyquist energy,
	# i.e. slightly soft - and the complaint here was blurry AND aliased, so the kernel that keeps the
	# most true detail wins. It must come BEFORE setparams, which stamps the outgoing frames.
	var vf := "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt470bg:range=pc"
	if _render_w() != int(_quality.w):
		vf = "scale=%d:%d:flags=lanczos," % [int(_quality.w), int(_quality.h)] + vf
	# FOLLOWING THE RENDER: read the AVI as it grows, and take a quiet file as the end of it
	var input := PackedStringArray(["-i", _avi])
	if follow:
		input = PackedStringArray(["-nostdin", "-follow", "1", "-rw_timeout", str(FOLLOW_TIMEOUT_US),
			"-i", "file:" + _avi])
	var args := PackedStringArray(["-y", "-fflags", "+genpts"]) + input + PackedStringArray([
		"-vf", vf,
		# crf 20 is NOT the quality floor - the MJPEG intermediate is, and crf 16 measured +0.1 dB
		# for +51% size. What DOES pay is `-tune grain`: on bright frames the default settings smear
		# ghost's fine procedural detail, retaining 93.3% of the source's high-frequency energy,
		# and grain tuning brings that to 99.6% for +1% file size. Its parts do far less alone
		# (no-dct-decimate +0.03 dB, deadzone +0.19, chroma-qp-offset +0.35) - the gain is the set.
		# `medium` rather than `fast` because grain tuning needs the extra analysis to pay off; the
		# encode is a small share of export time next to the render itself.
		"-c:v", "libx264", "-crf", "20", "-preset", "medium", "-tune", "grain", "-pix_fmt", "yuv420p",
		"-color_range", "pc", "-colorspace", "bt470bg",
		"-color_primaries", "bt709", "-color_trc", "bt709", "-chroma_sample_location", "center",
		"-g", str(gop), "-keyint_min", str(gop), "-movflags", "+write_colr",
		"-c:a", "aac", "-b:a", "192k",
		"-progress", ProjectSettings.globalize_path(_PROGRESS_FILE), "-nostats", "-loglevel", "error",
		_out])
	_transcode_pid = Subprocess.start("ffmpeg", args, "transcode")
	if follow:
		_live_encode = _transcode_pid > 0
		_punched = PUNCH_KEEP
		_punch_t = 2.0
		if _live_encode:
			_hold_scratch()
			print("ghost: encoding while rendering (pid %d) %s -> %s" % [_transcode_pid, _avi, _out])
		return
	if _transcode_pid > 0:
		_state = "transcoding"
		print("ghost: transcoding (pid %d, %.0fs) %s -> %s" % [_transcode_pid, dur, _avi, _out])
	else:
		# No ffmpeg: we can't produce the MP4. Leave the raw AVI so the render isn't wasted.
		_state = "done"
		_done_t = 30.0
		_set_status("⚠  ffmpeg not found; raw file kept: %s" % _avi, Color(1.0, 0.7, 0.6))


## Release the AVI's bytes the following encoder has already read. Linux only (a punched hole
## frees the blocks while the file keeps its length, so Movie Maker writes on undisturbed); the
## encoder's read position comes from /proc. Elsewhere the encode still follows the render, it
## just cannot give the space back until the end.
func _punch_behind_encoder() -> void:
	if OS.get_name() != "Linux" or _transcode_pid <= 0 or not Deps.has("fallocate"):
		return
	# ONLY WHILE THE HEADER SAYS "SIZE UNKNOWN". Mid-render Movie Maker writes 0 for the RIFF
	# size (measured), so the encoder can only read straight through and everything behind its
	# position is done with. A finished header names the index at the end, and a reader that
	# jumps there would have its unread data released under it - never release then.
	if _riff_size() != 0:
		return
	var pos := _encoder_read_pos()
	var upto := (pos - PUNCH_BEHIND) / 4096 * 4096
	if pos < 0 or upto <= _punched:
		return
	var target := _release_target()
	if target.is_empty():
		return
	var out: Array = []
	var code := Deps.execute("fallocate", ["--punch-hole", "--offset", str(_punched),
		"--length", str(upto - _punched), target], out)
	if code == 0:
		_punched = upto
		_released = true
		_unlink_scratch()
	else:
		push_warning("ghost export: could not release scratch space (%s) - the file will grow" % str(out))
		_punched = 1 << 62               # stop trying


## How far into the AVI the encoder has read: its open descriptor's offset, from /proc.
func _encoder_read_pos() -> int:
	var dir := "/proc/%d/fd" % _transcode_pid
	var d := DirAccess.open(dir)
	if d == null:
		return -1
	var want := ProjectSettings.globalize_path(_avi)
	for fd in DirAccess.get_files_at(dir):
		var link := d.read_link(dir.path_join(fd))
		if link != want and link != want + " (deleted)":
			continue
		# READ, NOT SIZED: /proc files report a length of 0, so a whole-file read returns nothing
		var f := FileAccess.open("/proc/%d/fdinfo/%s" % [_transcode_pid, fd], FileAccess.READ)
		if f == null:
			return -1
		for line in f.get_buffer(4096).get_string_from_utf8().split("\n"):
			if line.begins_with("pos:"):
				return int(line.substr(4).strip_edges())
	return -1


## A fresh scratch for a new render. A file already at the path goes first: the live encoder
## starts on whatever is there, and one left by an earlier run would be taken for this render's.
func _begin_scratch() -> void:
	_drop_scratch(false)
	_unlinked = false
	_released = false
	_transcode_pid = -1
	_live_encode = false
	_punched = PUNCH_KEEP
	if not _avi.is_empty() and FileAccess.file_exists(_avi):
		DirAccess.remove_absolute(_avi)


## Hold the scratch open and learn which descriptor holds it, so it can still be measured and
## released once its name is gone (see [member _hold]). The render and the encoder are other
## processes, so the one descriptor of ghost's that names the scratch is this one. Linux only:
## elsewhere there is no /proc to reach it by, and it keeps its name to the end.
func _hold_scratch() -> void:
	_drop_scratch(false)
	if OS.get_name() != "Linux":
		return
	_hold = FileAccess.open(_avi, FileAccess.READ)
	var dir := "/proc/%d/fd" % OS.get_process_id()
	var d := DirAccess.open(dir)
	if _hold == null or d == null:
		return
	var want := ProjectSettings.globalize_path(_avi)
	for fd in DirAccess.get_files_at(dir):
		if d.read_link(dir.path_join(fd)) == want:
			_held = dir.path_join(fd)
			return


## Take the scratch's name out of the folder. Only once a release has worked through ghost's own
## handle: a scratch that cannot shrink keeps its name, so it can at least be seen and deleted.
func _unlink_scratch() -> void:
	if _unlinked or _held.is_empty():
		return
	if DirAccess.remove_absolute(_avi) == OK:
		_unlinked = true
		print("ghost: the scratch is given back as it is encoded - %s is out of the folder" % _avi.get_file())


## Where fallocate reaches the scratch: through ghost's own descriptor when there is one -
## checked to still be the scratch, so a release can never land on another file - else by name.
func _release_target() -> String:
	var want := ProjectSettings.globalize_path(_avi)
	if _held.is_empty():
		return "" if _unlinked else want
	var d := DirAccess.open(_held.get_base_dir())
	var link := d.read_link(_held) if d != null else ""
	return _held if link == want or link == want + " (deleted)" else ""


## Let go of the scratch: close ghost's handle - once its name is gone, the kernel frees it when
## the render and the encoder have closed theirs too - and with [param remove], delete it if it
## still has a name.
func _drop_scratch(remove := true) -> void:
	if _hold != null:
		_hold.close()
		_hold = null
	_held = ""
	if remove and not _unlinked and not _avi.is_empty() and FileAccess.file_exists(_avi):
		DirAccess.remove_absolute(_avi)


func _scratch_len() -> int:
	return _hold.get_length() if _hold != null else _file_size(_avi)


## The RIFF size in the scratch's header: 0 while Movie Maker is still writing (it finishes the
## header last), -1 when it cannot be read.
func _riff_size() -> int:
	var f := _hold if _hold != null else FileAccess.open(_avi, FileAccess.READ)
	if f == null:
		return -1
	f.seek(4)
	return f.get_32()


## THE ENCODER QUIT WHILE THE RENDER WENT ON - the header is unfinished, so Movie Maker is still
## writing. A following encoder ends itself only after FOLLOW_TIMEOUT_US with nothing new, so this
## is a render that stalled that long, or ffmpeg failing. Left alone, the render fills a file
## nobody reads (unseen, once its name is gone) and the export then calls the short MP4 saved.
func _encoder_quit_early() -> void:
	if not _released:
		# nothing given back yet, so the AVI is whole: let the render finish and encode it then
		push_warning("ghost export: the encoder stopped early - encoding once the render finishes")
		_live_encode = false
		_drop_scratch(false)
		return
	var at := _read_transcode_pct()
	Subprocess.stop(_render_pid)
	_clear_override()
	_drop_scratch()
	_fail("⚠  Encoding stopped at %d%% while the render went on - export abandoned (see console)" % at)


const _PROGRESS_FILE := "user://transcode_progress.txt"


func _progress_reset() -> void:
	var f := FileAccess.open(_PROGRESS_FILE, FileAccess.WRITE)
	if f != null:
		f.store_string("")
		f.close()


# ffmpeg writes `out_time_us=<microseconds>` lines to the progress file; the fraction of the song's
# duration (captured at export start) it has reached is the transcode percent. Robust to partial/mid-
# write reads, and to the live song having ended (we use the captured `_song_dur`, not a live read).
func _read_transcode_pct() -> int:
	if _song_dur <= 0.0 or not FileAccess.file_exists(_PROGRESS_FILE):
		return _pct
	var text := FileAccess.get_file_as_string(_PROGRESS_FILE)
	var best := -1.0
	for line in text.split("\n"):
		if line.begins_with("out_time_us="):
			best = maxf(best, line.substr(12).to_float() / 1_000_000.0)
		elif line.begins_with("out_time_ms="):     # older ffmpeg (value is microseconds despite the name)
			best = maxf(best, line.substr(12).to_float() / 1_000_000.0)
	if best >= 0.0:
		_pct = clampi(int(round(best / _song_dur * 100.0)), 0, 99)
	return _pct


# Godot's AVI writer keeps 32-bit RIFF/LIST size fields, and a 4K render crosses
# 4 GiB in a few minutes of video - past that the written sizes WRAP (mod 2^32) and
# the container lies about where the frame data ends, even though every 00db/01wb
# chunk after it is written correctly all the way to EOF (verified by walking a 5 GB
# artifact chunk-by-chunk). Demuxers that trust those fields (players especially -
# their seeks also hit the equally-wrapped idx1 offsets) stall or repeat frames.
# The repair is two words: RIFF size and the movi LIST size become 0 - "size
# unknown, read to end of file" - turning any demux into a clean sequential walk of
# the intact chunks. No-op for files under 4 GiB (their sizes are correct).
func _repair_avi_sizes(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ_WRITE)
	if f == null:
		return
	if f.get_length() < 4294967296:
		f.close()
		return
	f.seek(4)
	f.store_32(0)                       # RIFF size -> unknown
	var pos := 12
	for i in 64:                        # walk top-level chunks to the movi LIST
		f.seek(pos)
		var tag := f.get_buffer(4).get_string_from_ascii()
		var csize := f.get_32()
		if tag == "LIST" and f.get_buffer(4).get_string_from_ascii() == "movi":
			f.seek(pos + 4)
			f.store_32(0)               # movi size -> unknown
			print("ghost export: repaired wrapped >4GiB AVI sizes in ", path.get_file())
			break
		if csize <= 0 or tag.is_empty():
			break
		pos += 8 + csize + (csize & 1)
	f.close()


## The sample rate a WAV declares, from its fmt chunk (canonical 44-byte header,
## rate at byte 24). Falls back to the procedural synthesizer's rate if the file
## cannot be read, which is the only rate that was ever assumed before.
static func _wav_rate(path: String) -> int:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null or f.get_length() < 44:
		return Voice.SR
	f.seek(24)
	var rate := f.get_32()
	f.close()
	return rate if rate > 0 else Voice.SR


func _file_size(path: String) -> int:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return 0
	var s := f.get_length()
	f.close()
	return s


# override.cfg lives in the project root only for the duration of a render; Godot reads
# it at startup to override project.godot (here: the export's output resolution + stretch
# mode). _ready() also clears any stale copy left by a render that never exited cleanly.
func _override_path() -> String:
	return ProjectSettings.globalize_path("res://override.cfg")


## The size the SCENE is rendered at - the output size times the preset's supersample factor,
## rounded to an even number of pixels (yuv420p cannot encode an odd dimension). The output size
## itself is `_quality.w/h`; ffmpeg resolves one to the other.
func _render_w() -> int:
	return _even(int(_quality.w) * float(_quality.get("ss", 1.0)))


func _render_h() -> int:
	return _even(int(_quality.h) * float(_quality.get("ss", 1.0)))


static func _even(v: float) -> int:
	return int(round(v * 0.5)) * 2


func _write_override(w: int, h: int) -> void:
	var f := FileAccess.open(_override_path(), FileAccess.WRITE)
	if f == null:
		push_warning("ghost export: could not write override.cfg (resolution may default)")
		return
	# viewport_* set the RENDERED (recorded) resolution; window_*_override shrink the
	# OS window itself to an unobtrusive floater. In "viewport" stretch mode the two
	# are independent - true 4K on any monitor, tiny window. The window must stay a
	# normal, drawable window: minimizing it makes Godot skip rendering and the movie
	# records frozen frames (see boot.gd).
	f.store_string("[display]\n\nwindow/size/viewport_width=%d\nwindow/size/viewport_height=%d\nwindow/size/window_width_override=480\nwindow/size/window_height_override=270\nwindow/stretch/mode=\"viewport\"\n" % [w, h])
	# AND THE INTERMEDIATE'S QUALITY - see the matching note in
	# MaskEditor._write_render_override. Godot writes .avi as MJPEG in yuvj420p at
	# `video_quality` 0.75, so the scratch file is a lossy, chroma-subsampled
	# generation before x264 ever sees a frame. Measured on a real frame: 42.6 dB
	# through the default intermediate against 45.5 for a single generation, and
	# 44.3 at 1.0. The bigger scratch file is deleted seconds later.
	f.store_string("\n[editor]\n\nmovie_writer/video_quality=1.0\n")
	f.close()


func _clear_override() -> void:
	var path := _override_path()
	if FileAccess.file_exists(path):
		DirAccess.remove_absolute(path)


func _fail(msg: String) -> void:
	_clear_override()
	_upload_this = false
	_state = "done"
	_done_t = 8.0
	_set_status(msg, Color(1.0, 0.7, 0.6))
	push_warning("ghost export: " + msg)


func _set_status(text: String, color: Color) -> void:
	var t := text + _sign_note()
	if t != _status.text:
		_status.text = t
		# back to one line's height, AFTER the text: a label grown by a long line does not shrink
		# by itself, and reset before the new text it grows straight back to the old one's height
		_status.offset_top = -110.0 - _inset
	_status.add_theme_color_override("font_color", color)
	_status.visible = true


# --- YouTube -------------------------------------------------------------------------------------

## While an export that is going to YouTube is being made, how its sign-in stands - on the progress
## line, the one thing on screen for the length of the render.
func _sign_note() -> String:
	if not _upload_this or not (_prepping or _state in ["baking", "rendering", "transcoding"]):
		return ""
	match _sign:
		"signing_in":
			return "   ⇪ sign in to YouTube in your browser"
		"ok":
			return "   ⇪ YouTube: signed in"
		"failed":
			return "   ⇪ no upload: " + _sign_why
	return ""


## What the mode would upload now, {} for nothing (see [member upload_provider]).
func _upload_peek() -> Dictionary:
	if not upload_provider.is_valid():
		return {}
	var m: Variant = upload_provider.call("")
	return m if m is Dictionary else {}


## The menu's YouTube items as things stand: the box when the mode has something to upload, a
## resume while an upload waits (from any mode - it is an earlier export's), a sign-out while a
## sign-in is kept, and another client file once one is.
func _refresh_upload_items() -> void:
	for id in [UPLOAD_ID, RESUME_ID, SIGN_OUT_ID, CLIENT_ID]:
		var at := _quality_menu.get_item_index(id)
		if at >= 0:
			_quality_menu.remove_item(at)
	var peek := _upload_peek()
	if not peek.is_empty():
		var before: Array = YouTube.uploads_in(str(peek.get("record", "")))
		_quality_menu.add_check_item("Upload to YouTube again (unlisted)" if not before.is_empty()
			else "Upload to YouTube (unlisted)", UPLOAD_ID)
		var at := _quality_menu.get_item_index(UPLOAD_ID)
		_quality_menu.set_item_checked(at, _upload)
		_quality_menu.set_item_tooltip(at, _upload_tip(peek, before))
	var p: Dictionary = YouTube.pending()
	if not p.is_empty() and FileAccess.file_exists(str(p.get("file", ""))):
		_quality_menu.add_item("Resume the YouTube upload of %s" % str(p["file"]).get_file(), RESUME_ID)
	if YouTube.signed_in():
		_quality_menu.add_item("Sign out of YouTube", SIGN_OUT_ID)
	if YouTube.has_client():
		_quality_menu.add_item("Use a different Google client file…", CLIENT_ID)
		_quality_menu.set_item_tooltip(_quality_menu.get_item_index(CLIENT_ID),
			"Import another Google OAuth client file (a \"Desktop app\" client's JSON). It replaces the one "
			+ "Ghost Notes keeps; a sign-in made with a different client is forgotten.")


func _upload_tip(peek: Dictionary, before: Array) -> String:
	var tip := ("Once the video is saved, upload it to your YouTube channel as an unlisted video "
		+ "titled \"%s\". The title, description and tags are the episode's - edit them in the panel.") \
		% YouTube.fit_title(str(peek.get("title", "")))
	if not YouTube.has_client():
		tip += ("\n\nTicking it asks for your Google OAuth client file first (the JSON from Google Cloud) "
			+ "and keeps it for every later sign-in; then your browser opens to sign in.")
	elif not YouTube.signed_in():
		tip += "\n\nTicking it opens your browser to sign in to YouTube first."
	else:
		tip += "\n\nGhost Notes is signed in: the upload starts by itself once the video is saved."
	if not before.is_empty():
		var last: Dictionary = before[before.size() - 1]
		tip += "\n\nAlready uploaded: %s (%s, %s)." % [str(last.get("url", "")), str(last.get("privacy", "")),
			str(last.get("at", "")).get_slice("T", 0)]
	return tip


## TICKING THE BOX SETS UPLOADS UP, as far as they need: the Google client file the first time,
## then the sign-in while none is kept - each visibly, before the quality and the save path - and
## the menu comes back with the box ticked. Once signed in it simply ticks.
func _toggle_upload() -> void:
	var at := _quality_menu.get_item_index(UPLOAD_ID)
	if at < 0:
		return
	if not _upload and not YouTube.has_client():
		_quality_menu.hide()
		_client_dialog.popup_centered()
		return
	if not _upload and not YouTube.signed_in():
		_quality_menu.hide()
		_check_sign_in(_reopen_ticked)
		return
	_upload = not _upload
	_quality_menu.set_item_checked(at, _upload)


func _on_client_file(path: String) -> void:
	var err: String = YouTube.import_client(path)
	if not err.is_empty():
		_note_t = 10.0
		_set_status("⚠  " + err, Color(1.0, 0.7, 0.6))
		return
	_note_t = 6.0
	_set_status("✓  Google client imported - Ghost Notes keeps it for every sign-in", Color(0.82, 0.95, 0.86))
	_check_sign_in(_reopen_ticked)


## Back to the menu with the box ticked, to choose the quality.
func _reopen_ticked() -> void:
	_upload = true
	_reopen = true
	_on_export()
	_reopen = false


## THE SIGN-IN AN UPLOAD NEEDS, settled where the person can see it: a kept sign-in is renewed
## (which proves Google still honors it), and one that is missing or has lapsed opens the browser
## beside a dialog that says what is awaited, opens the page again, copies its link, names Google's
## usual block, and cancels. A newer check replaces one still waiting. [param then] runs once signed
## in (the menu reopened, ticked); the export's own check runs nothing - its upload waits on [member _sign].
func _check_sign_in(then := Callable()) -> void:
	_sign_attempt += 1
	var mine := _sign_attempt
	_sign_then = then
	_sign = "checking"
	_sign_why = ""
	var tok: Dictionary = await _yt.access_token()
	if mine != _sign_attempt:
		return
	if not tok.has("error"):
		_signed_in(then)
		return
	if not tok.has("signed_out"):
		# the client file is bad, or Google could not be reached: a browser would not help
		_sign = "failed"
		_sign_why = str(tok["error"])
		if then.is_valid():
			_show_sign_failed(_sign_why)
		return
	_sign = "signing_in"
	_show_sign_wait()
	var err: String = await _yt.sign_in()
	if mine != _sign_attempt:
		return
	if err.is_empty():
		_signed_in(then)
		return
	_sign = "failed"
	_sign_why = err
	_show_sign_failed(err)


func _signed_in(then: Callable) -> void:
	_sign = "ok"
	_sign_dialog.hide()
	if _state == "idle":
		_note_t = 5.0
		_set_status("✓  Signed in to YouTube", Color(0.82, 0.95, 0.86))
	if then.is_valid():
		then.call()


func _show_sign_wait() -> void:
	_sign_dialog.dialog_text = SIGN_WAIT
	_sign_dialog.ok_button_text = "Cancel"
	_sign_again.text = "Open the page again"
	_sign_copy.text = "Copy the link"
	_sign_copy.visible = true
	if not _sign_dialog.visible:
		_sign_dialog.popup_centered(Vector2i(560, 0))


func _show_sign_failed(why: String) -> void:
	var hint := SIGN_BLOCKED if why.contains("timed out") or why.contains("access_denied") else ""
	_sign_dialog.dialog_text = "The YouTube sign-in did not finish: %s.%s" % [why, ("\n\n" + hint) if not hint.is_empty() else ""]
	_sign_dialog.ok_button_text = "Close"
	_sign_again.text = "Try again"
	_sign_copy.visible = false
	if not _sign_dialog.visible:
		_sign_dialog.popup_centered(Vector2i(560, 0))


func _on_sign_action(action: StringName) -> void:
	match String(action):
		"again":
			if _yt.phase == "signing_in":
				_yt.reopen_sign_in()
			else:
				_check_sign_in(_sign_then)
		"copy":
			DisplayServer.clipboard_set(_yt.sign_in_url)
			_sign_copy.text = "Link copied"


## The dialog closed: while a sign-in waits, that cancels it. An export goes on and saves its video;
## the upload then waits to be resumed from the menu.
func _on_sign_close() -> void:
	_sign_copy.text = "Copy the link"
	if _sign != "signing_in":
		return
	_sign_attempt += 1
	_yt.stop_sign_in()
	_sign = "failed"
	_sign_why = "the sign-in was cancelled"
	if _state == "idle":
		_note_t = 5.0
		_set_status("YouTube sign-in cancelled", Color(0.95, 0.92, 0.7))


## The saved export joins the queue at once - so a quit, a lapsed sign-in or a dropped connection
## leaves it resumable - and goes up once the sign-in settles (see the `upload_wait` state).
func _queue_upload() -> void:
	_upload_this = false
	# THE THUMBNAIL FIRST, when the mode names a moment for one (the tarot's title screen): a frame of
	# the saved video itself, so it is exactly what the video shows. A frame that cannot be taken
	# only costs the thumbnail.
	var thumb := ""
	var at := float(_upload_meta.get("thumbnail_at", -1.0))
	if at >= 0.0:
		_state = "preparing"
		thumb = await _take_thumbnail(_out, at)
	var q: Dictionary = YouTube.queue(_out, YouTube.video_body(_upload_meta, _out.get_file().get_basename()),
		str(_upload_meta.get("record", "")), thumb)
	if q.has("error"):
		_state = "done"
		_done_t = 30.0
		_set_status("✓  Saved  %s   ⚠ not uploaded: %s" % [_out, q["error"]], Color(1.0, 0.85, 0.6))
		return
	_upload_file = _out
	_state = "upload_wait"


## A frame of [param video] at [param at] seconds, as a 1280x720 JPEG beside it (YouTube's thumbnail
## size): its path, or "" when ffmpeg could not make one.
func _take_thumbnail(video: String, at: float) -> String:
	var jpg := video.get_basename() + ".thumbnail.jpg"
	DirAccess.remove_absolute(jpg)
	var pid := Subprocess.start("ffmpeg", thumbnail_args(video, jpg, at), "thumbnail")
	if pid <= 0:
		return ""
	var t0 := Time.get_ticks_msec()
	while Subprocess.alive(pid) and Time.get_ticks_msec() - t0 < 30000:
		await (Engine.get_main_loop() as SceneTree).process_frame
	if Subprocess.alive(pid):
		Subprocess.stop(pid)
		return ""
	if _file_size(jpg) < 1024:
		push_warning("ghost export: no thumbnail could be taken from %s at %.1f s" % [video.get_file(), at])
		return ""
	print("ghost: thumbnail taken at %.1f s -> %s" % [at, jpg])
	return jpg


## ffmpeg's arguments for [method _take_thumbnail]: seek, one frame, scaled to 1280x720.
static func thumbnail_args(video: String, jpg: String, at: float) -> PackedStringArray:
	return PackedStringArray(["-y", "-nostdin", "-loglevel", "error", "-ss", "%.3f" % maxf(at, 0.0), "-i", video,
		"-frames:v", "1", "-vf", "scale=1280:720:flags=lanczos", "-q:v", "2", jpg])


func _resume_upload() -> void:
	var p: Dictionary = YouTube.pending()
	if p.is_empty():
		return
	_upload_file = str(p.get("file", ""))
	_state = "upload_wait"
	_check_sign_in()


## Send the queued upload; the state is set before the first await, so the frame loop starts it once.
func _run_upload() -> void:
	_state = "uploading"
	var res: Dictionary = await _yt.resume()
	_state = "done"
	_done_t = 60.0
	if res.has("error"):
		var later := "" if YouTube.pending().is_empty() else " - resume it from the ⤓ menu"
		_set_status("⚠  The YouTube upload stopped: %s%s" % [res["error"], later], Color(1.0, 0.7, 0.6))
		return
	var url := str(res.get("url", ""))
	var privacy := str(res.get("privacy", ""))
	DisplayServer.clipboard_set(url)
	if not privacy.is_empty() and privacy != YouTube.PRIVACY:
		# what YouTube does with uploads from a Cloud project that has not passed its API audit
		_set_status("✓  On YouTube, but YouTube kept it %s: %s (link copied)" % [privacy, url], Color(1.0, 0.85, 0.6))
		push_warning("ghost: YouTube kept the upload %s - uploads from a Cloud project that has not "
			% privacy + "passed YouTube's API audit stay private until it does")
		return
	var thumb_err := str(res.get("thumbnail_error", ""))
	if not thumb_err.is_empty():
		push_warning("ghost: YouTube - the thumbnail was not set: " + thumb_err)
	_set_status("✓  On YouTube (%s): %s  (link copied)%s" % [privacy if not privacy.is_empty() else YouTube.PRIVACY, url,
		("   ⚠ thumbnail not set: " + thumb_err) if not thumb_err.is_empty() else ""],
		Color(0.82, 0.95, 0.86) if thumb_err.is_empty() else Color(1.0, 0.85, 0.6))


func _sign_out() -> void:
	await _yt.sign_out()
	_note_t = 6.0
	_set_status("Signed out of YouTube (the Google client file stays imported)", Color(0.95, 0.92, 0.7))


## Lift the button and its status line clear of whatever the mode has put along
## the bottom of the frame. See Chrome.bottom_inset.
func set_bottom_inset(v: float) -> void:
	_inset = v
	if _btn != null:
		_btn.offset_top = -68.0 - _inset
		_btn.offset_bottom = -28.0 - _inset
	_place_status()


## THE STATUS LINE HAS A ROW OF ITS OWN, ABOVE THE BUTTONS, AND WRAPS. It used to get that row
## only when a mode claimed the bottom of the frame (see [method set_bottom_inset]) and otherwise
## shared the button row - and a Label that does not wrap grows to its right, so a long export
## line ("⏺ Rendering <an episode's whole title>.mp4 … 12.3%") ran under the buttons and off the
## screen, hiding the very progress it showed (reported 2026-10-06). Now it is as wide as the
## window allows, never wider than [constant STATUS_W], ends at the row's right edge, and extra
## lines grow UPWARD, away from the buttons.
func _place_status() -> void:
	if _status == null:
		return
	var w := STATUS_W
	var vp := get_viewport() if is_inside_tree() else null
	if vp != null:
		w = minf(STATUS_W, maxf(160.0, vp.get_visible_rect().size.x - 56.0))
	_status.offset_left = -28.0 - w
	_status.offset_right = -28.0
	_status.offset_top = -110.0 - _inset
	_status.offset_bottom = -74.0 - _inset
