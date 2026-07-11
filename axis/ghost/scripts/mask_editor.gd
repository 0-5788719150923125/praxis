extends CanvasLayer
class_name MaskEditor

## MaskEditor - the mask-mode authoring surface.
##
## Load a clip, key two colors apart (per-pixel hue classification, see
## shaders/mask_split.gdshader), place MARKERS where the split/effect should
## change, scrub the timeline, export. See scripts/mask_session.gd for the data
## model - a session's markers are fixed-schema scalar vectors, not free-form
## params, and every one is either a RAMP (eases in before its anchor) or a DAMP
## (accumulates after it) - there's no third, undifferentiated "marker" kind.
##
## Standalone by design: a mask session is tied to one specific external clip, not
## the audio-reactive show, so this does NOT route through Director/Spectrum. Two
## entry points (see main.gd):
##   --mask-edit [path]    interactive editor (this file's normal mode)
##   --mask-render <json>  the export relaunch's headless-ish render mode
##                         ([member render_mode] = true; no panel, autoplay, quits
##                         when the audio ends - mirrors --export/_export_mode).
##
## Widget choices mirror the rest of ghost: the timeline is a bespoke-drawn Control
## ([MaskTimeline], the [DialWidget] idiom), the export button/status/FileDialog are
## the exact pattern from exporter.gd, color entry is Godot's native
## ColorPickerButton (precision input, not an "instrument" worth hand-drawing).
##
## The preview's view modes (see MaskSession.VIEW_MODES + _build_video_composition)
## are the main screen and the inset as independent axes - raw, inset-raw, inset-fx,
## both-fx, full-fx - cycled in that "evolution" order by one button (VIEW_CYCLE).
## Only one video ever decodes ([member _player], always raw); the fx layers just
## re-draw its decoded texture through [const SHADER], each only while visible - so
## "raw" costs no shader pass at all.
##
## view_mode is a per-MARKER field (see MaskSession.VECTOR_FIELDS), not an editing-
## only preference: the toggle button edits the marker at the playhead exactly like
## every other panel control (see _edit), the live preview always renders whatever
## session.at_time() resolves to, and the export relaunch (render_mode) runs the
## identical per-frame logic - so a marker set to "raw" plays raw in the rendered
## file too, not just live. What you see while editing is what you get.

const SHADER := preload("res://shaders/mask_split.gdshader")
const MASKS_DIR := "res://masks"
const PANEL_W := 320
# Picked via _sort_dropdown - see _apply_sort for what each one does.
const _SORT_MODES := ["A → Z", "Z → A", "Energy"]

## Set by main.gd before open_source() for the --mask-render relaunch: skip the
## editing panel, autoplay from t=0, quit when the audio finishes.
var render_mode := false

var session: MaskSession = null
var _session_path := ""       # res://-relative or absolute; wherever it was loaded from

var _player: VideoStreamPlayer     # always the RAW decode - never carries the shader
var _audio: AudioStreamPlayer
var _audio_thread: Thread = null   # loads the (large, uncompressed) main WAV off the main thread
var _render_t := 0.0               # accumulated MOVIE time in render mode - the deterministic
                                   #   export clock (sum of the fixed-fps _dt), authoritative over
                                   #   the video/audio stream clocks, which can drift or end early
var _autostart_pending := false    # live autostart is HELD (video paused on frame 1) until the
                                   #   threaded audio attaches, so the intro never plays audio-less
                                   #   and skips - _poll_audio_thread begins playback, synced (below)
var _pending_restore := -1.0       # playhead seconds to seek to once the player is ready (see _process)
var _pending_restore_tries := 0
var _reload_check_pid := -1        # headless compile check gating a reload (see _do_restart)
var _reload_check_log := ""
## Set when a reload was requested while an export was mid-flight (_render_state !=
## "idle") - see _reload_requested. _poll_render re-fires the request once the export
## finishes, same deference the assistant already gets in reload_when_idle: a restart
## quits this process, and Godot kills the child processes it created (see assistant.gd's
## _closing doc) - including the render/transcode subprocess an export is waiting on.
var _reload_after_export := false
## Set the instant _restart_now actually commits to quitting (after its own final
## _save_session capture) - see _save_session and _exit_tree for why this exists.
var _restarting := false
var _track_audio_jobs: Array = []  # background sidecar-.ogg extractions, {pid, index, ogg}
# One material PER LAYER: the main overlay and the inset can be mid-transition at
# different presences (e.g. fx-inset -> both: the inset holds full while the main
# overlay fades in), and a layer's presence multiplies into its own intensities -
# impossible with one shared material.
var _mat_main := ShaderMaterial.new()
var _mat_inset := ShaderMaterial.new()
var _playing := false
var _audio_holding := false   # main audio paused-in-place, waiting for video to catch up (see _process)
var _cursor_idle_t := 0.0
const _CURSOR_HIDE_DELAY := 1.5   # seconds of stillness during playback before the mouse cursor hides

var _fx_overlay: TextureRect       # full-frame fx layer - shaded copy of whichever source is active
var _cont_view: TextureRect        # full-frame RAW layer for an active continuation track (see _sync_tracks) -
                                   # _player's own raw picture is only valid while session.main_visible_at's
                                   # own-clip half holds; once a continuation track owns time t, this shows
                                   # THAT track's own independently-decoded frame instead (never _player's -
                                   # see continuation_track_at's doc for why the two used to be conflated)
var _pip_view: TextureRect         # the inset's content - shaded or raw per view mode
var _mask_wrap: PanelContainer     # the inset's border/placement box (holds _pip_view)
var _view_label: Label     # passive view-mode readout (the old cycle button; V cycles now)
var _help_panel: PanelContainer
var _peek_raw := false     # DISPLAY-ONLY raw override; never touches session data (hold P)
var _last_inset_show := 0.0   # this frame's resolved inset_show - _sync_tracks reads it too,
                               # so a track's own PiP box respects the same view-mode gate as _mask_wrap
var _pip_track := 0           # this frame's resolved pip_track: 0 = main clip in the PiP, k = track (k-1)

var _timeline: MaskTimeline
var _tview: TimelineView          # shared pixel<->time mapping - see timeline_view.gd
var _video_area: AspectRatioContainer  # letterboxed video slot - see _refresh_lanes
var _lanes_col: VBoxContainer     # primary clip's trim lane + one per imported track
var _composition_parent: Control  # holds _player/_fx_overlay/_mask_wrap - track PiP views land here too
## Runtime state per session.tracks[i] - NOT persisted (session.tracks holds only
## the data; players/views are rebuilt from it every _ready_with_session). Each:
## {player: VideoStreamPlayer, view: TextureRect, active: bool}. `active` tracks
## whether the track is CURRENTLY the one playing (see _sync_tracks) so entering/
## leaving its window on the master timeline only seeks+starts/pauses it once,
## not every frame.
var _track_runtime: Array = []
var _import_dialog: FileDialog
var _import_pid := -1
var _import_pending := {}   # {source, video, index} - the lane is added up front, this tracks its transcode
var _selected: Variant = null   # the marker Dictionary currently shown in the panel

var _color_a: ColorPickerButton
var _hue_a: HSlider   # numeric grading twin of _color_a - same hue_a field, precise dragging
var _threshold: HSlider
var _threshold_label: Label
var _grp_color: VBoxContainer   # "Key color" swatch, pinned above the sortable options below
var _grp_options: VBoxContainer   # every effect option (label+slider pairs), reordered by _apply_sort
var _options: Array = []          # [{label: Label, control: Control}], creation order - see _register_option
var _sort_mode := 2                # index into _SORT_MODES - defaults to Energy, see _apply_sort
var _sort_dropdown: OptionButton
var _feather: HSlider
var _sat_floor: HSlider
var _fx_x: HSlider
var _fx_y: HSlider
var _fx_x_label: Label   # "Pan X/Y" relabeled "Wind X/Y" for snow - direction, not placement
var _fx_y_label: Label
var _fx_scale: HSlider
var _fx_density: HSlider
var _fx_density_label: Label   # "Coverage" relabeled "Stickiness" for crystal (feature conformance)
var _fx_contrast: HSlider
var _fx_contrast_label: Label   # "Contrast" relabeled "Sensitivity" for snow (no keying group of its own)
var _fx_speed: HSlider
var _fx_lag: HSlider
var _fx_lag_label: Label            # "Lag (s)" relabeled "Lead (s)" for oracle - same field, opposite sense
var _fx_smooth: HSlider
var _gust: HSlider   # snow's own Gust slider - a second, independent view onto fx_smooth (see _fx_smooth)
var _undul: HSlider  # fur's Undulation - fur's view onto fx_smooth (same stored-field reuse as _gust)
var _coil: HSlider   # fur's Coil - fur's view onto fx_lag (pushed raw as u_l_lagf; echo bakes its lag into u_l_ew)
var _stick: HSlider  # fur's Stickiness - its OWN field (fx_stick, u_l_stick); 0 = today's free coat
var _resonance: HSlider
var _effect_a: OptionButton
var _intensity_a: HSlider
var _intensity_label: Label   # tooltip swaps meaning for restore/clear, see _update_effect_controls
var _kind: OptionButton     # ramp / damp - see MaskSession.MARKER_KINDS
var _marker_duration: HSlider
var _marker_label: Label
var _time_label: Label
var _marker_list: VBoxContainer   # sequential ramp/damp list, pinned to the panel's bottom
var _history_label: Label   # "Undo: <last action>" preview above the +Ramp/+Damp row, see _refresh_history_label

var _feedback: Node = null    # backtick console (see _build_feedback); editor mode only
var _was_playing_before_feedback := false

var _status: Label            # shared bottom-right notification - prep AND export
var _export_btn: Button
var _dialog: FileDialog
var _open_dialog: FileDialog

var _prep_video_pid := -1
var _prep_audio_pid := -1
var _prep_src_dur := 0.0         # source clip duration (seconds), for the video step's %
var _prep_state := "idle"        # idle / prepping_video / prepping_audio
var _pending := {}               # source/dir/video/audio paths mid-prep
const _PREP_PROGRESS_FILE := "user://mask_prep_progress.txt"

# Temporal capture (echo ring + whisp anchor): quarter-res snapshots of past
# frames, taken every _ECHO_INTERVAL seconds of PLAYBACK time (slot = position /
# interval, so live preview and the export relaunch capture at the same clip
# positions - the deterministic-clock discipline, one level up from u_time).
# Only runs while a whisp/echo/snow/oracle/serpent/chimera layer is actually on
# screen THIS frame (_temporal_active, set alongside _chimera_active in
# _apply_frame_state) - a session-wide "does this ever use one" check made the
# synchronous GPU readback below run for the ENTIRE session once any one of
# those markers appeared anywhere in it, stuttering unrelated stretches of a
# long timeline (feedback/0016).
var _echo_ring: Array = [null, null, null, null, null, null, null, null]   # ImageTexture, slot-indexed
var _echo_slot := -1
var _prev_pos := -1.0         # last frame's playhead position - lets capture skip an ACTIVE
                              #   scrub drag (position moving) and fire once it settles
var _chimera_active := false  # is a chimera layer on screen THIS frame - gates the track readback
                              #   so it never runs while the (distant) chimera marker isn't rendering
var _temporal_active := false # is a whisp/echo/snow/oracle/serpent/chimera layer on screen THIS
                              #   frame - gates _maybe_capture_echo's readback entirely
var _meta_amount := 0.0       # strongest meta-layer weight on screen THIS frame (0 = none);
                              #   drives the workspace capture and the render-mode chrome reveal
var _workspace_tex: ImageTexture = null  # the editor's own previous frame, captured for the META
                              #   mirror (see _capture_workspace) - null until the first capture
var _meta_chrome: Control = null  # render-mode only: the editor-chrome overlay a META section
                              #   fades in over the clean video (the recorded product demo)
var _chrome_parent: Node = null   # where _build_chrome() parents (self, or _meta_chrome in export)
# The whisp anchor is double-buffered: _anchor_ema is the EMA at the LATEST
# capture, _anchor_prev the one before. The uniform pushed per frame lerps
# between them by position-within-slot (see _push_anchor) - pushing the EMA
# directly stepped the whole pattern once per capture, a visible jump amplified
# by pattern zoom ("jittery, it resets, it jumps"). Position-keyed, so live and
# export trace the identical glide.
var _anchor_prev := Vector2(0.5, 0.5)
var _anchor_ema := Vector2(0.5, 0.5)
# Chimera's landmark FRAME (position + size), double-buffered and glided exactly
# like the centroid above. The MAIN face's modeled size (RMS radius of its
# key/motion mass) and the imported TRACK face's own centroid + size, each an
# EMA over the same thresholds. The shader maps every main pixel through the two
# frames so the graft phase-locks to the main head instead of drifting (see the
# chimera branch in mask_split.gdshader). Defaults degrade to ~the old centered
# clone when a face can't be modeled.
var _anchor_scale_prev := 0.28
var _anchor_scale_ema := 0.28
var _track_anchor_prev := Vector2(0.5, 0.5)
var _track_anchor_ema := Vector2(0.5, 0.5)
var _track_scale_prev := 0.28
var _track_scale_ema := 0.28
var _track_prev_lum := PackedFloat32Array()   # track frame's previous-luminance grid (motion fallback)
const _ECHO_INTERVAL := 0.35

# Wave impulses (whisp only): a fast head turn shows up as a big frame-to-frame
# luminance jolt in the same 48x27 grid _update_whisp_anchor already samples -
# an ONSET detector (motion vs. an adaptive EMA baseline + deviation, not a
# fixed magic threshold) fires an impulse at the anchor's current position, and
# the shader drops a decaying blob of paint there that drifts off along
# whisp's own local current, confined to the volumetric field like the rest of
# whisp (see u_wave_* / wave_wash in mask_split.gdshader) - a drop carried by
# the water, not a ring detached from it. WAVE_SLOTS must match WAVEN in the
# shader.
const _WAVE_SLOTS := 3
const _WAVE_COOLDOWN := 1.1   # seconds; keeps a shaky run from piling up waves
var _wave_prev_lum := PackedFloat32Array()   # last capture's 48x27 luminance grid
var _wave_motion_ema := 0.0
var _wave_dev_ema := 0.02     # seeded so the first few ticks aren't hyper-sensitive
var _wave_last_time := -100.0
var _wave_pos := PackedVector2Array()
var _wave_time := PackedFloat32Array()
var _wave_amp := PackedFloat32Array()
var _wave_slot := 0

var _waveform_pid := -1
var _waveform_path := ""         # set once we know it; polled in _process until it exists
var _audio_env := PackedFloat32Array()   # per-column amplitude from the waveform image (resonance)

var _render_state := "idle"      # idle / rendering / transcoding / done
var _render_pid := -1
var _transcode_pid := -1
var _out := ""
var _avi := ""

# Auto-save: every edit marks the session dirty; it saves shortly after the last
# change in a burst (and unconditionally on close), so work persists across
# reloads without a save button and without writing once per slider-drag pixel.
var _dirty := false
var _autosave_cooldown := 0.0
const _AUTOSAVE_DELAY := 0.4

# Undo/redo: whole-array snapshots of session.markers (small, plain data - cheap
# enough to duplicate wholesale, no need for a diff/command log). A slider drag
# or a marker being dragged along the timeline fires the same mutation dozens of
# times a second; without coalescing, one drag would fragment into dozens of undo
# steps and a single Ctrl+Z would barely nudge the value back. `_undo_coalesce_key`
# identifies the CURRENT gesture (which marker, which field) - a repeat push with
# the same key inside the cooldown window just refreshes the window instead of
# snapshotting again, so an entire drag - however long - is one undo step, and
# the boundary lands cleanly the moment you touch something else.
var _undo_stack: Array = []
var _redo_stack: Array = []
# Parallel to _undo_stack/_redo_stack - a human-readable description of the
# action each entry captured, so the panel can preview what Ctrl+Z would
# revert (see _push_undo/_refresh_history_label).
var _undo_descs: Array = []
var _redo_descs: Array = []
const _UNDO_LIMIT := 200
var _undo_coalesce_key := ""
var _undo_coalesce_cooldown := 0.0
const _UNDO_COALESCE_WINDOW := 0.9
var _select_generation := 0   # bumped on every selection change; folded into the coalesce key


func _ready() -> void:
	layer = 100
	_mat_main.shader = SHADER
	_mat_inset.shader = SHADER
	if not render_mode:
		add_to_group("mask_editor")   # so the Assistant can trigger a reload here (see _do_restart)
		_build_status_label()   # built up front - prep needs it before a session exists


## `path` is either a prepared session .json, a raw source video (transcoded once
## and cached under masks/<slug>/), or empty (prompt via a native file dialog).
func open_source(path: String) -> void:
	if path.is_empty():
		_prompt_for_source()
		return
	if path.get_extension().to_lower() == "json":
		_session_path = path
		session = MaskSession.load(path if path.begins_with("res://") else path)
		if session != null:
			_ready_with_session()
		else:
			_set_status("⚠  Could not read session: " + path)
		return
	var slug := _slugify(path)
	var dir := MASKS_DIR + "/" + slug
	var video := dir + "/video.ogv"
	var audio := dir + "/audio.wav"
	_session_path = dir + "/session.json"
	_pending = {"source": path, "dir": dir, "video": video, "audio": audio}
	# Check for a live prep BEFORE trusting anything cached on disk - a file that
	# merely EXISTS may just be mid-write (this is why the real clip's session ended
	# up with duration 0.0: it was opened while an external transcode was still
	# appending to video.ogv, and "exists" isn't "finished"). See _finish_session for
	# the matching duration>0 validation on the fast paths below.
	if _prep_looks_live(dir, video):
		_prep_state = "waiting_external"
		_set_status("⏳  Preparing clip (already in progress elsewhere)…")
		return
	var abs_session := ProjectSettings.globalize_path(_session_path)
	if FileAccess.file_exists(abs_session):
		var loaded := MaskSession.load(abs_session)
		if loaded != null and loaded.duration > 0.0:
			session = loaded
			_ready_with_session()
			return
	var abs_video := ProjectSettings.globalize_path(video)
	var abs_audio := ProjectSettings.globalize_path(audio)
	if FileAccess.file_exists(abs_video) and FileAccess.file_exists(abs_audio):
		_finish_session(path, video, audio)   # validates duration itself; re-preps if bad
		return
	_prep(path, dir, video, audio)


func _prompt_for_source() -> void:
	_open_dialog = FileDialog.new()
	_open_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_open_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_open_dialog.use_native_dialog = true
	_open_dialog.title = "Open a clip for mask mode"
	_open_dialog.filters = PackedStringArray(["*.mp4,*.mov,*.mkv,*.webm ; Video"])
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_open_dialog.current_dir = downloads
	_open_dialog.size = Vector2i(800, 560)
	_open_dialog.file_selected.connect(open_source)
	add_child(_open_dialog)
	_open_dialog.popup_centered()


# --- one-time prep: ffmpeg -> masks/<slug>/{video.ogv, audio.wav} -----------------
# Two real ffmpeg processes (not a shell chain) so each can be polled and reported
# like exporter.gd's bake/render/transcode steps: video first (the slow part - can
# be minutes on a long clip), then audio (fast, PCM decode). Progress comes from the
# same `-progress <file>` mechanism the export transcode step already uses.

func _prep(source: String, dir: String, video: String, audio: String) -> void:
	var abs_dir := ProjectSettings.globalize_path(dir)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	_prep_src_dur = _probe_duration(source)
	_progress_reset(_PREP_PROGRESS_FILE)
	# Small GOP (-g 25 = ~1 keyframe/sec at typical fps) keeps scrubbing responsive on
	# long clips; theora is the only format VideoStreamPlayer decodes natively (see
	# next/mask_mode_spike notes - WebM/H.264 need a GDExtension, so every source
	# clip gets this one-time transcode regardless of its original codec).
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", source, "-an",
		"-c:v", "libtheora", "-q:v", "6", "-g", "25",
		"-progress", ProjectSettings.globalize_path(_PREP_PROGRESS_FILE), "-nostats",
		ProjectSettings.globalize_path(video)])
	_prep_video_pid = OS.create_process("ffmpeg", args)
	_prep_state = "prepping_video" if _prep_video_pid > 0 else "idle"
	if _prep_video_pid <= 0:
		_set_status("⚠  Could not start ffmpeg (is it on PATH?)")
	else:
		_touch_lock(abs_dir)
		_set_status("⏳  Preparing clip (video)…  0%")


func _start_prep_audio() -> void:
	_touch_lock(ProjectSettings.globalize_path(_pending.dir))   # bridge the gap while
	# video.ogv's mtime goes stale and audio.wav doesn't exist yet - see _prep_looks_live.
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", String(_pending.source), "-vn",
		"-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
		ProjectSettings.globalize_path(_pending.audio)])
	_prep_audio_pid = OS.create_process("ffmpeg", args)
	_prep_state = "prepping_audio" if _prep_audio_pid > 0 else "idle"
	_set_status("⏳  Preparing clip (audio)…")


func _progress_reset(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string("")


# --- prep liveness: is SOMETHING actively writing this clip's output right now? ---
# Two writers on the SAME -y output path interleave and corrupt it - not
# hypothetical, this is here because it happened (twice: a second launch mid-prep
# raced the first, and separately a manual external ffmpeg run raced the app). The
# first version of this guard tracked a PID and called OS.is_process_running() on
# it - which crashes (engine-level ECHILD) the moment the PID belongs to a process
# this instance didn't spawn itself, e.g. a PREVIOUS launch's child, exactly the
# cross-instance case the guard exists for. An mtime check answers the actual
# question - "did anything touch this file recently" - without caring who the
# writer is or whether this instance could ever see its PID.
const _STALE_AFTER := 10.0   # seconds since last write before a prep counts as abandoned

func _lock_path(dir: String) -> String:
	return dir.path_join(".prep.lock")


func _touch_lock(abs_dir: String) -> void:
	var f := FileAccess.open(_lock_path(abs_dir), FileAccess.WRITE)
	if f != null:
		f.store_string(str(Time.get_unix_time_from_system()))


func _clear_lock(dir: String) -> void:
	var p := _lock_path(ProjectSettings.globalize_path(dir))
	if FileAccess.file_exists(p):
		DirAccess.remove_absolute(p)


func _fresh(path: String, now: float) -> bool:
	return FileAccess.file_exists(path) and now - FileAccess.get_modified_time(path) < _STALE_AFTER


## True if the lock file, the video output, or the audio output was modified within
## the last _STALE_AFTER seconds - covers both prep sub-steps (the lock's own touch
## bridges the gap before video.ogv exists yet; the outputs' own mtimes take over,
## and cover each other, for the rest).
func _prep_looks_live(dir: String, video: String) -> bool:
	var now := Time.get_unix_time_from_system()
	var abs_dir := ProjectSettings.globalize_path(dir)
	return _fresh(_lock_path(abs_dir), now) \
		or _fresh(ProjectSettings.globalize_path(video), now) \
		or _fresh(abs_dir.path_join("audio.wav"), now)


## Same `out_time_us=` polling exporter.gd's transcode step uses, against the
## source clip's OWN duration (captured before the video step starts).
func _read_prep_pct() -> int:
	if _prep_src_dur <= 0.0 or not FileAccess.file_exists(_PREP_PROGRESS_FILE):
		return 0
	var text := FileAccess.get_file_as_string(_PREP_PROGRESS_FILE)
	var best := -1.0
	for line in text.split("\n"):
		if line.begins_with("out_time_us=") or line.begins_with("out_time_ms="):
			best = maxf(best, line.substr(12).to_float() / 1_000_000.0)
	if best < 0.0:
		return 0
	return clampi(int(round(best / _prep_src_dur * 100.0)), 0, 99)


func _finish_session(source: String, video: String, audio: String) -> void:
	var abs_video := ProjectSettings.globalize_path(video)
	var dur := _probe_duration(abs_video)
	# A theora/ogg file caught mid-write can still probe successfully - ffmpeg
	# doesn't flush on every frame, so a writer can look "stale" by mtime alone
	# during a buffering gap and still be actively growing. duration <= 0.0 catches
	# a totally unparseable file; this catches the sneakier case where it parses
	# fine but is truncated (this is exactly how a real 16:22 clip got cached at
	# 2:14 - the audio step's own duration is ground truth, extracted whole, so a
	# video duration meaningfully short of it means "not actually done yet", not
	# "shorter clip").
	var audio_dur := _probe_duration(ProjectSettings.globalize_path(audio))
	var incomplete := dur <= 0.0 or (audio_dur > 0.0 and dur < audio_dur - 2.0)
	if incomplete:
		var dir := video.get_base_dir()
		if _prep_looks_live(dir, video):
			_prep_state = "waiting_external"
			_set_status("⏳  Preparing clip (already in progress elsewhere)…")
			return
		_set_status("⚠  Cached clip looked incomplete - re-preparing…")
		_prep(source, dir, video, audio)
		return
	session = MaskSession.new()
	session.source_path = source
	session.video_path = video
	session.audio_path = audio
	session.duration = dur
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))
	_ready_with_session()


## Robust video-duration probe. The fast path is the container's own duration,
## but our own libtheora transcode (see _start_track_import) routinely produces an
## .ogv whose header carries NO duration - ffprobe returns "N/A" for both
## format=duration AND stream=duration. That returned 0 here, and _finish_track_import
## reads 0 as "transcode failed" and silently adds no lane (the "Track import does
## nothing" bug). So when both are absent, fall back to counting video packets (fast,
## no decode) and dividing by the frame rate - every playable file has a definite
## packet count even when it lacks a duration field.
func _probe_duration(path: String) -> float:
	var d := _ffprobe_float(["-show_entries", "format=duration"], path)
	if d > 0.0:
		return d
	d = _ffprobe_float(["-select_streams", "v:0", "-show_entries", "stream=duration"], path)
	if d > 0.0:
		return d
	var out := []
	OS.execute("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_packets",
		"-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "default=nw=1", path], out)
	if out.size() > 0:
		var packets := 0.0
		var fps := 0.0
		for line in String(out[0]).split("\n"):
			var s := line.strip_edges()
			if s.begins_with("nb_read_packets="):
				packets = s.substr(16).to_float()
			elif s.begins_with("r_frame_rate="):
				var fr := s.substr(13).split("/")
				if fr.size() == 2 and fr[1].to_float() > 0.0:
					fps = fr[0].to_float() / fr[1].to_float()
				else:
					fps = s.substr(13).to_float()
		if packets > 0.0 and fps > 0.0:
			return packets / fps
	return 0.0


## One ffprobe query for a single float value (csv, no keys), 0.0 if it comes back
## empty or "N/A". See _probe_duration for why the caller layers several of these.
func _ffprobe_float(entries: Array, path: String) -> float:
	var args := PackedStringArray(["-v", "error"])
	for e in entries:
		args.append(String(e))
	args.append("-of")
	args.append("csv=p=0")
	args.append(path)
	var out := []
	OS.execute("ffprobe", args, out)
	if out.size() > 0:
		var s := String(out[0]).strip_edges()
		if s.is_valid_float():
			return s.to_float()
	return 0.0


static func _slugify(path: String) -> String:
	var base := path.get_file().get_basename().to_lower()
	var out := ""
	for c in base:
		out += c if (c >= "a" and c <= "z") or (c >= "0" and c <= "9") else "_"
	while out.contains("__"):
		out = out.replace("__", "_")
	return out.trim_suffix("_").substr(0, 32)


# --- session ready: build preview + (unless render_mode) the editing UI -----------

func _ready_with_session() -> void:
	if _status != null:
		_status.visible = false
	_player = VideoStreamPlayer.new()
	_player.stream = load(ProjectSettings.globalize_path(session.video_path))
	_player.expand = true
	# No .material here, ever - the fx layers carry the shader (see
	# _build_video_composition). Which layer combination is shown is a per-marker
	# field, applied every frame in _process identically whether this is the live
	# editor or a render_mode export relaunch - export renders exactly what the
	# timeline says, not a hardcoded "always full masked" look.

	_audio = AudioStreamPlayer.new()
	add_child(_audio)
	# The main audio is a big uncompressed WAV (~170MB / ~3s to read). Loading it on
	# the main thread froze startup for that whole time. Live: load it on a worker and
	# attach it the moment it's ready (_process polls the thread), so the video + UI
	# come up instantly and audio joins a beat later, synced to the current position.
	# Export (render_mode) still loads synchronously - it must have audio from frame 0,
	# deterministically. AudioStreamWAV's static loader is the runtime-safe path (plain
	# load() has no loader for a raw .wav outside the import pipeline).
	var abs_audio := ProjectSettings.globalize_path(session.audio_path)
	if render_mode:
		_audio.stream = AudioStreamWAV.load_from_file(abs_audio)
		_apply_main_volume()
	else:
		_audio_thread = Thread.new()
		_audio_thread.start(_load_wav_threaded.bind(abs_audio))

	if render_mode:
		_build_render_view()
		# The resonance envelope must exist BEFORE the first recorded frame, or the
		# export's early frames disagree with the live preview. Nearly always cached
		# already (editing generated it); if not, block briefly on ffmpeg - an export
		# is a batch job, a one-time pause is fine where it wouldn't be live.
		_waveform_path = session.audio_path.get_base_dir().path_join("waveform_sqrt.png")
		var abs_wave := ProjectSettings.globalize_path(_waveform_path)
		if not FileAccess.file_exists(abs_wave):
			OS.execute("ffmpeg", _waveform_args(abs_wave))
		_load_waveform(abs_wave)
	else:
		_build_editor_ui()
		_ensure_waveform()
	# Export loaded its audio synchronously (it must have sound from frame 0), so it
	# can start immediately. Live loads the WAV on a worker thread: starting now would
	# advance the video (the master clock) while the audio is still unattached, and the
	# audio would then join wherever the video already got to - the intro skip on every
	# first run. Hold the autostart instead - show frame 1 paused - and let
	# _poll_audio_thread begin playback the moment the audio is ready, synced from 0.
	if render_mode or _audio_thread == null or not _audio_thread.is_started():
		_play(true)
	else:
		_autostart_pending = true
		_play(false)   # decode + show the first frame, but hold the clock at the start
	# Land back where the playhead was last time (persisted per session) - only live;
	# an export always starts at clip_in. Deferred to the first _process tick: a
	# VideoStreamPlayer won't accept a seek the same frame it starts playing.
	if not render_mode and session.playhead > 0.05:
		_pending_restore = session.playhead


## Kick off (or discover already-cached) the timeline's waveform image - fully
## decoupled from playback readiness. Generating it (an ffmpeg pass over the whole
## audio track) can take a couple seconds on a long clip; blocking session-ready on
# that would turn "instant" cache hits back into a wait, so this fires async and the
## timeline just draws nothing until _process() notices the PNG exists.
func _ensure_waveform() -> void:
	# "waveform_sqrt": the filename carries the rendering recipe, so clips whose
	# cache predates a recipe change regenerate instead of loading the stale look.
	# sqrt scaling lifts small amplitudes into visibility - a linear plot of
	# ordinary speech/music sat near the axis and was barely visible on the strip.
	# 4096px wide so it still resolves when the timeline is zoomed well in.
	_waveform_path = session.audio_path.get_base_dir().path_join("waveform_sqrt.png")
	var abs_wave := ProjectSettings.globalize_path(_waveform_path)
	if FileAccess.file_exists(abs_wave):
		_load_waveform(abs_wave)
		return
	_waveform_pid = OS.create_process("ffmpeg", _waveform_args(abs_wave))


func _waveform_args(abs_out: String) -> PackedStringArray:
	return PackedStringArray([
		"-y", "-loglevel", "error", "-i", ProjectSettings.globalize_path(session.audio_path),
		"-filter_complex", "showwavespic=s=4096x160:colors=white:scale=sqrt",
		"-frames:v", "1", abs_out])


## Load the waveform image once and derive BOTH consumers from it: the timeline's
## strip texture, and the resonance envelope (per-column occupancy = amplitude).
## One file, one recipe - so the wisps breathe with exactly the wave the user sees
## on the strip, and live/export read identical values (file-deterministic; no
## real-time analyzer to drift between the two).
func _load_waveform(abs_path: String) -> void:
	var img := Image.load_from_file(abs_path)
	if img == null:
		return
	if _timeline != null:
		_timeline.waveform_texture = ImageTexture.create_from_image(img)
	var w := img.get_width()
	var h := img.get_height()
	_audio_env = PackedFloat32Array()
	_audio_env.resize(w)
	for x in w:
		var count := 0
		for y in range(0, h, 2):    # every 2nd row - envelope precision, half the cost
			if img.get_pixel(x, y).a > 0.1:
				count += 1
		# showwavespic draws a symmetric column; its filled fraction IS the (sqrt-
		# scaled) amplitude. 0.9 headroom so full-scale audio still reaches ~1.0.
		_audio_env[x] = clampf(float(count) / (float(h) * 0.5 * 0.9), 0.0, 1.0)


## The audio envelope at clip-time `t` (0 when unavailable), lightly smoothed so
## the wisps swell rather than flicker frame-to-frame.
func _env_at(t: float) -> float:
	if _audio_env.is_empty() or session == null or session.duration <= 0.0:
		return 0.0
	var n := _audio_env.size()
	var i := clampi(int(t / session.duration * float(n)), 1, n - 2)
	return (_audio_env[i - 1] + _audio_env[i] + _audio_env[i + 1]) / 3.0


## Stacked layers in `parent`'s full rect, shared by both the live editor and
## the render_mode export so they composite identically:
##   _player      raw video, visible underneath everything while the main clip's
##                own kept range covers the current time
##   _cont_view   raw video, same full-rect slot as _player - visible instead of it
##                while a continuation track (see MaskSession.continuation_track_at)
##                owns the current time; never both at once
##   _fx_overlay  full-frame shaded copy of whichever of the above is active (its
##                own material) - the MAIN fx layer
##   _mask_wrap   the bordered inset holding _pip_view (its own material)
## Which layers show, and how strongly, comes from the per-frame AMOUNTS
## (MaskSession.mode_amounts, blended through ramp/damp windows by at_time) -
## applied every frame in _apply_frame_state. A layer's presence multiplies into
## its own material's intensities, which is why each has its own material: the
## inset can hold full fx while the main overlay is still fading in.
func _build_video_composition(parent: Control) -> void:
	_composition_parent = parent   # secondary tracks' PiP views land here too - see _build_track_view
	parent.add_child(_player)
	_player.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	# Sits at the same full-rect slot as _player, right underneath it in the same
	# z-position - only one of the two is ever visible at once (see _process), so
	# _fx_overlay's shaded copy always has exactly one raw picture beneath it to draw.
	_cont_view = TextureRect.new()
	_cont_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_cont_view.stretch_mode = TextureRect.STRETCH_SCALE
	_cont_view.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_cont_view.visible = false
	parent.add_child(_cont_view)

	_fx_overlay = TextureRect.new()
	_fx_overlay.material = _mat_main
	_fx_overlay.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_fx_overlay.stretch_mode = TextureRect.STRETCH_SCALE
	_fx_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	parent.add_child(_fx_overlay)

	_mask_wrap = PanelContainer.new()
	var border := StyleBoxFlat.new()
	border.bg_color = Color(0, 0, 0, 0)
	border.set_border_width_all(2)
	border.border_color = Color(1.0, 1.0, 1.0, 0.85)
	_mask_wrap.add_theme_stylebox_override("panel", border)
	# The inset's placement is fixed (bottom-right corner box); only its
	# visibility/presence animates.
	_mask_wrap.anchor_left = 0.66
	_mask_wrap.anchor_top = 0.64
	_mask_wrap.anchor_right = 0.98
	_mask_wrap.anchor_bottom = 0.96
	parent.add_child(_mask_wrap)

	_pip_view = TextureRect.new()
	_pip_view.material = _mat_inset
	_pip_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_pip_view.stretch_mode = TextureRect.STRETCH_SCALE
	_mask_wrap.add_child(_pip_view)


## A secondary track's own composited view: a raw (unshaded - the masking
## effects system keys off ONE frame's colors via the primary's shader chain,
## not built for a second independent source yet) picture-in-picture box. It
## takes over the SAME inset slot _mask_wrap occupies (bottom-right) rather
## than getting its own corner - there is only ever one PiP box on screen at
## a time; see _sync_tracks, which hides _mask_wrap for as long as a track's
## own box is showing (feedback/0056: a stray, always-empty second box used
## to sit at a separate default corner on top of the real one).
func _build_track_view(i: int) -> void:
	var track: Dictionary = session.tracks[i]
	var player := VideoStreamPlayer.new()
	player.stream = load(ProjectSettings.globalize_path(String(track.video_path)))
	player.expand = true
	_composition_parent.add_child(player)

	var wrap := Control.new()
	wrap.anchor_left = _mask_wrap.anchor_left
	wrap.anchor_top = _mask_wrap.anchor_top
	wrap.anchor_right = _mask_wrap.anchor_right
	wrap.anchor_bottom = _mask_wrap.anchor_bottom
	_composition_parent.add_child(wrap)

	var view := TextureRect.new()
	view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view.stretch_mode = TextureRect.STRETCH_SCALE
	view.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	wrap.add_child(view)

	var border := StyleBoxFlat.new()
	border.bg_color = Color(0, 0, 0, 0)
	border.set_border_width_all(2)
	border.border_color = Color(0.6, 0.9, 1.0, 0.85)
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", border)
	panel.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	wrap.add_child(panel)

	# The track's audio plays through a real AudioStreamPlayer (the SAME proven path as
	# the main clip) - VideoStreamPlayer's own embedded audio does NOT play when the
	# video is driven by manual seeking, which is why the track was silent. We demux the
	# track .ogv's Vorbis into a sidecar .ogg once (fast stream-copy, cached), and mute
	# the video player's audio so nothing double-plays.
	player.volume = 0.0

	while _track_runtime.size() <= i:
		_track_runtime.append({})
	_track_runtime[i] = {"player": player, "view": view, "wrap": wrap, "audio": null, "active": false}
	player.paused = true
	player.play()   # must be playing before .stream_position can be set (see _sync_tracks)
	_ensure_track_audio(i)   # attaches .audio now (cached sidecar) or when ffmpeg finishes


## Give track `i` its AudioStreamPlayer, from a sidecar .ogg demuxed from the track's
## .ogv. If the sidecar already exists we attach immediately; otherwise ffmpeg runs in
## the BACKGROUND (was OS.execute - a synchronous main-thread stall that could hang the
## whole editor while audio kept playing) and _poll_track_audio attaches it on finish.
## The sidecar lives beside the .ogv (same basename, .ogg). If it exists, attach it.
## Otherwise extract it - preferring the ORIGINAL SOURCE, which always has the audio:
## the old -an transcode stripped it from the .ogv, so demuxing the .ogv gave nothing.
## Re-encode to Vorbis (the source is usually aac/mp3, not copyable to ogg). Background
## + quiet; if even the source has no audio, _poll_track_audio records no_audio so we
## don't retry forever. Once a sidecar exists it's cached and this is a no-op.
func _ensure_track_audio(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	var track: Dictionary = session.tracks[i]
	var video_path := String(track.get("video_path", ""))
	if video_path.is_empty():
		return
	var abs_ogg := ProjectSettings.globalize_path(video_path.get_basename() + ".ogg")
	if FileAccess.file_exists(abs_ogg):
		_attach_track_audio(i, abs_ogg)
		return
	if bool(track.get("no_audio", false)):
		return   # already learned there's no audio anywhere for this track
	# Prefer the original import (has audio); fall back to the .ogv (only new imports
	# embed audio) when the source is gone or was never recorded.
	var src := String(track.get("source_path", ""))
	var abs_src := ProjectSettings.globalize_path(src) if src.begins_with("res://") else src
	if src.is_empty() or not FileAccess.file_exists(abs_src):
		abs_src = ProjectSettings.globalize_path(video_path)
	if not FileAccess.file_exists(abs_src):
		return
	var pid := OS.create_process("ffmpeg", ["-y", "-loglevel", "quiet", "-i", abs_src,
		"-vn", "-c:a", "libvorbis", "-q:a", "4", abs_ogg])
	if pid > 0:
		_track_audio_jobs.append({"pid": pid, "index": i, "ogg": abs_ogg})


## Load a track's sidecar .ogg into an AudioStreamPlayer and hang it on the runtime.
func _attach_track_audio(i: int, abs_ogg: String) -> void:
	if i < 0 or i >= _track_runtime.size() or not (_track_runtime[i] is Dictionary):
		return
	var stream := AudioStreamOggVorbis.load_from_file(abs_ogg)
	if stream == null:
		return
	var ap := AudioStreamPlayer.new()
	ap.stream = stream
	add_child(ap)
	_track_runtime[i]["audio"] = ap


## Poll the background sidecar extractions; attach each track's audio as ffmpeg finishes.
func _poll_track_audio() -> void:
	for j in range(_track_audio_jobs.size() - 1, -1, -1):
		var job: Dictionary = _track_audio_jobs[j]
		if not OS.is_process_running(int(job.pid)):
			var idx := int(job.index)
			if FileAccess.file_exists(String(job.ogg)):
				_attach_track_audio(idx, String(job.ogg))
			elif idx >= 0 and idx < session.tracks.size():
				# ffmpeg produced nothing - this track has no audio stream. Record it so
				# we don't re-run (and re-fail) the extraction on every future load.
				session.tracks[idx]["no_audio"] = true
				_mark_dirty()
			_track_audio_jobs.remove_at(j)


# --- multi-track: trim lanes, import, playback sync ----------------------------

# Per-lane reserved height. MUST cover a TrackLane's own minimum (26px, see
# track_lane.gd) PLUS the _lanes_col VBox's inter-lane separation (2px), or the
# lane stack overflows its allocated rect and - drawing above the marker strip it
# abuts - spills over the timeline's top edge, clipping the marker flags and the
# playhead timestamp tag (worsening with each imported track). 26 + 2 = 28.
const _LANE_H := 28.0
# The volume knob every lane gets (see _volume_knob) is anchored top-left, offset_right
# 34 - a lane's own label must start past that plus a small gap, or the two draw on top
# of each other whenever the lane's own left edge sits near local x0 (the primary lane's
# offset is always 0, so this bites it every time - see feedback/0009).
const _LANE_LABEL_LEFT := 38.0

func _track_getter(i: int, field: String) -> Callable:
	return func(): return float(session.tracks[i].get(field, 0.0))


func _track_setter(i: int, field: String) -> Callable:
	return func(v): session.tracks[i][field] = v


## Rebuild every trim/track lane from session state (the primary clip's own trim
## block, plus one per session.tracks entry) and recompute the shared
## TimelineView's cached extent. Called after any STRUCTURAL change (import,
## delete, undo/redo) - never mid-drag; see TrackLane.drag_ended and
## TimelineView.refresh's own doc for why that distinction is load-bearing here.
func _refresh_lanes() -> void:
	if _lanes_col == null:
		return
	for c in _lanes_col.get_children():
		c.queue_free()

	var primary := TrackLane.new()
	primary.tview = _tview
	primary.label = String(session.video_path).get_file()
	primary.reserved_left = _LANE_LABEL_LEFT
	primary.color = Color(0.55, 0.75, 1.0)
	primary.movable = false
	primary.full_duration = session.duration
	primary.get_in = func(): return session.clip_in
	primary.set_in = func(v): session.clip_in = v
	primary.get_out = func(): return session.effective_clip_out()
	primary.set_out = func(v): session.clip_out = v
	primary.get_fade_in = func(): return session.main_fade_in
	primary.set_fade_in = func(v): session.main_fade_in = v
	primary.get_fade_out = func(): return session.main_fade_out
	primary.set_fade_out = func(v): session.main_fade_out = v
	primary.get_snap_targets = _snap_targets_for.bind(-1)
	primary.drag_started.connect(func(): _push_undo("", "trimmed the main clip"))
	primary.drag_ended.connect(func(): _tview.refresh(session))
	primary.changed.connect(_mark_dirty)
	_lanes_col.add_child(primary)
	# The main clip's own pull-rope volume (mirrors each track's), so its level in the
	# mix is set independently.
	primary.add_child(_volume_knob(
		func(): return float(session.main_volume),
		func(v): session.main_volume = v,
		Color(0.55, 0.75, 1.0), true))
	# Split, same corner/behavior as a secondary track's - the main clip is otherwise
	# just track -1 (see _snap_targets_for), so it gets the same cut at the playhead
	# (see _split_main). No delete button here: unlike an imported track, the primary
	# clip is never optional - there's always exactly one.
	var main_split := Button.new()
	main_split.text = "✂"
	main_split.custom_minimum_size = Vector2(20, 18)
	main_split.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	main_split.offset_left = -22
	main_split.offset_top = 2
	main_split.offset_right = -2
	main_split.offset_bottom = 20
	main_split.focus_mode = Control.FOCUS_NONE
	main_split.tooltip_text = "Split the main track at the playhead"
	main_split.pressed.connect(_split_main)
	primary.add_child(main_split)

	for i in session.tracks.size():
		var lane := TrackLane.new()
		lane.tview = _tview
		lane.label = String(session.tracks[i].get("video_path", "")).get_file()
		lane.reserved_left = _LANE_LABEL_LEFT
		lane.color = Color(0.6, 0.95, 0.7)
		lane.movable = true
		lane.full_duration = float(session.tracks[i].get("duration", 0.0))
		lane.get_in = _track_getter(i, "clip_in")
		lane.set_in = _track_setter(i, "clip_in")
		lane.get_out = _track_getter(i, "clip_out")
		lane.set_out = _track_setter(i, "clip_out")
		lane.get_offset = _track_getter(i, "offset")
		lane.set_offset = _track_setter(i, "offset")
		lane.get_fade_in = _track_getter(i, "fade_in")
		lane.set_fade_in = _track_setter(i, "fade_in")
		lane.get_fade_out = _track_getter(i, "fade_out")
		lane.set_fade_out = _track_setter(i, "fade_out")
		lane.get_snap_targets = _snap_targets_for.bind(i)
		lane.get_playhead = func(): return _player.stream_position if _player != null else 0.0
		lane.drag_started.connect(func(): _push_undo("", "trimmed a track"))
		lane.drag_ended.connect(func(): _tview.refresh(session))
		lane.changed.connect(_mark_dirty)
		_lanes_col.add_child(lane)
		# The delete button OVERLAYS the lane's own top-right corner rather than
		# sitting beside it in an HBoxContainer - an inline sibling would shrink
		# the lane's own width below the primary lane's/_timeline's, and every
		# lane's x_of() must span the exact same pixel width or their blocks
		# stop lining up against a shared second.
		var del := Button.new()
		del.text = "✕"
		del.custom_minimum_size = Vector2(20, 18)
		del.set_anchors_preset(Control.PRESET_TOP_RIGHT)
		del.offset_left = -22
		del.offset_top = 2
		del.offset_right = -2
		del.offset_bottom = 20
		del.focus_mode = Control.FOCUS_NONE
		del.tooltip_text = "Remove this track"
		var idx := i
		del.pressed.connect(func(): _delete_track(idx))
		lane.add_child(del)
		# Split, just left of delete - cuts this track in two at the playhead (see
		# _split_track). Same overlay approach as delete, for the same reason: an
		# inline sibling would shrink the lane's own width out from under the
		# shared timeline's pixel mapping.
		var split := Button.new()
		split.text = "✂"
		split.custom_minimum_size = Vector2(20, 18)
		split.set_anchors_preset(Control.PRESET_TOP_RIGHT)
		split.offset_left = -46
		split.offset_top = 2
		split.offset_right = -26
		split.offset_bottom = 20
		split.focus_mode = Control.FOCUS_NONE
		split.tooltip_text = "Split this track at the playhead"
		split.pressed.connect(func(): _split_track(idx))
		lane.add_child(split)
		# Pull-rope volume knob, on the LEFT of the lane so it never sits under the floating
		# assistant chat button (bottom-right, where the old right-side controls landed).
		# The track's own AudioStreamPlayer mixes with the main clip; _sync_tracks reads
		# `volume` every frame, so pulling it changes the level live.
		lane.add_child(_volume_knob(
			func(): return float(session.tracks[idx].get("volume", 1.0)),
			func(v): session.tracks[idx]["volume"] = v,
			Color(0.6, 0.95, 0.7)))

	# The lane stack sits directly above the marker strip, and the video's own
	# letterboxed slot (see _build_editor_ui) has to shrink to match - otherwise
	# an imported track (or several) grows this stack tall enough to push its
	# lanes/delete-buttons over the bottom of the video instead of the reserved
	# strip below it.
	_apply_lane_reserved(1 + session.tracks.size())
	_tview.refresh(session)


## Collapse the lane stack (and expand the video letterbox into the space that
## frees up) to fit exactly `count` rows. Called with the full track count
## right after any structural change (above), and every frame with however
## many lanes are actually on screen right now - once TrackLane starts hiding
## itself for clips scrolled entirely out of the current zoom/pan window, the
## fixed-for-the-whole-track-count reservation left a dead black band where
## those rows used to be instead of actually collapsing - see feedback/0023.
func _apply_lane_reserved(count: int) -> void:
	var reserved := 90.0 + float(count) * _LANE_H
	_lanes_col.offset_top = -reserved
	_lanes_col.offset_bottom = -90
	if _video_area != null:
		_video_area.offset_bottom = -reserved


func _delete_track(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	_push_undo("", "deleted a track")
	if i < _track_runtime.size():
		var rt: Dictionary = _track_runtime[i]
		if rt.has("player"):
			(rt.player as Node).queue_free()
		if rt.has("wrap"):
			(rt.wrap as Node).queue_free()
		if rt.has("audio") and rt.audio != null:
			(rt.audio as Node).queue_free()
		_track_runtime.remove_at(i)
	session.tracks.remove_at(i)
	_refresh_lanes()
	_mark_dirty()


## Cut track `i` into two independent lanes at the playhead, both still pointing
## at the SAME source video/audio (video_path is shared - _ensure_track_audio's
## sidecar .ogg is keyed off that path, so the new half finds the existing
## extraction and attaches instantly, no re-demux). The left half keeps this
## track's own identity and index (just trims its clip_out); the right half is
## a full duplicate of the fields - same clip_in..clip_out span, same volume -
## then re-pointed to start exactly at the split: its clip_in advances to the
## split's LOCAL position in the source, and its offset (where that lands on
## the MASTER timeline) is set to the playhead itself, so the two blocks sit
## edge-to-edge with no gap or overlap. From there each is an ordinary lane -
## TrackLane's own drag handles already shift/resize them independently.
## Appended at the END of session.tracks (never inserted at i+1): background
## sidecar-audio jobs (_track_audio_jobs) and _track_runtime are index-keyed,
## and an insert would silently misalign any in-flight job for a LATER track.
func _split_track(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	var track: Dictionary = session.tracks[i]
	var offset := float(track.get("offset", 0.0))
	var cin := float(track.get("clip_in", 0.0))
	var cout := float(track.get("clip_out", 0.0))
	var master_t: float = _player.stream_position if _player != null else 0.0
	var split_local := master_t - offset + cin
	if split_local <= cin + 0.05 or split_local >= cout - 0.05:
		_set_status("✂  Move the playhead inside this track's span to split it there")
		return
	_push_undo("", "split a track")
	var right: Dictionary = track.duplicate()
	right["clip_in"] = split_local
	right["offset"] = master_t
	right["fade_in"] = 0.0
	track["clip_out"] = split_local
	track["fade_out"] = 0.0
	session.tracks.append(right)
	_reconcile_track_runtime()
	_refresh_lanes()
	_mark_dirty()


## Cut the MAIN clip in two at the playhead, same as _split_track but for the
## primary lane: the main clip has no `offset` of its own (master time and its
## source time are the same clock - see TrackLane._bounds), so the playhead
## position IS the split point directly, no offset arithmetic needed. The left
## half stays the main clip (just trims clip_out, same as dragging its own out
## handle inward). The right half can't stay "main" - there's only ever one -
## so it's appended to session.tracks as an ordinary track pointing at the same
## video_path/source_path, picking up at the split with its own offset. From
## there it's indistinguishable from any imported track: draggable, deletable,
## splittable again - the main clip is really just track -1 (see
## _snap_targets_for), and after a split its tail end is a track in fact, not
## just in spirit.
func _split_main() -> void:
	if _player == null:
		return
	var cin := session.clip_in
	var cout := session.effective_clip_out()
	var split_t: float = _player.stream_position
	if split_t <= cin + 0.05 or split_t >= cout - 0.05:
		_set_status("✂  Move the playhead inside the main clip's span to split it there")
		return
	_push_undo("", "split the main clip")
	var right := {
		"video_path": session.video_path,
		"source_path": session.source_path,
		"duration": session.duration,
		"clip_in": split_t,
		"clip_out": cout,
		"offset": split_t,
		"fade_in": 0.0,
		"fade_out": session.main_fade_out,
		"volume": session.main_volume,
	}
	session.clip_out = split_t
	session.main_fade_out = 0.0
	session.tracks.append(right)
	_reconcile_track_runtime()
	_refresh_lanes()
	_mark_dirty()


func _prompt_import_track() -> void:
	# Already open (double-tap of T) - just bring it forward, don't stack a
	# second dialog behind the first.
	if _import_dialog != null and is_instance_valid(_import_dialog):
		_import_dialog.popup_centered()
		return
	# An import is already transcoding - don't let a second one clobber the
	# first's pending state mid-flight (they share _import_pid/_import_pending).
	if _import_pid > 0:
		_set_status("⏳  A track is still importing - one at a time")
		return
	# Immediate feedback that T registered, BEFORE the dialog - if the dialog
	# itself somehow fails to show, at least the keypress isn't silent.
	_set_status("📁  Choose a video to import as a second track…")
	_import_dialog = FileDialog.new()
	_import_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_import_dialog.access = FileDialog.ACCESS_FILESYSTEM
	# In-window dialog, NOT native. The native (portal) dialog silently shows
	# nothing on a Linux box without xdg-desktop-portal - the exact "I press T
	# and nothing happens" report. Godot's own dialog always renders in-window.
	_import_dialog.use_native_dialog = false
	_import_dialog.title = "Import a second track (picture-in-picture)"
	_import_dialog.filters = PackedStringArray(["*.mp4, *.mov, *.mkv, *.webm, *.ogv ; Video"])
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_import_dialog.current_dir = downloads
	_import_dialog.size = Vector2i(820, 560)
	_import_dialog.file_selected.connect(_start_track_import)
	# Free the dialog whichever way it closes (pick or cancel), so the next T
	# opens a fresh one and the is_instance_valid guard above reads false
	# again. _start_track_import is connected first, so it runs before the
	# close on a pick.
	_import_dialog.file_selected.connect(func(_p): _close_import_dialog())
	_import_dialog.canceled.connect(_close_import_dialog)
	add_child(_import_dialog)
	_import_dialog.popup_centered()


func _close_import_dialog() -> void:
	if _import_dialog != null and is_instance_valid(_import_dialog):
		_import_dialog.queue_free()
	_import_dialog = null


## Same one-time ffmpeg->theora transcode _prep() does for the primary clip,
## minus the audio extraction step (v1 tracks are silent - see the class doc's
## scope note) - PID-polled in _process(), never blocking, matching the
## project's one established pattern for external subprocesses.
## Add the track's LANE first, transcode second. The lane is pure metadata
## (duration/offset/trim) - it doesn't need the transcoded video at all - so it
## appears the instant you pick a file, independent of the background ffmpeg run.
## The old order (transcode -> probe -> only THEN add the lane) meant any hiccup in
## that async tail left the timeline looking like nothing happened. The PiP VIDEO
## still needs the .ogv, so _build_track_view is deferred to _finish_track_import;
## until then _sync_tracks just skips this index (no "player" in its runtime slot).
func _start_track_import(source: String) -> void:
	var slug := _slugify(source)
	var dir := _session_path.get_base_dir()
	var idx := session.tracks.size()
	var video := dir + "/track_%s_%d.ogv" % [slug, idx]
	# The SOURCE file has reliable duration metadata (unlike our libtheora output),
	# so the lane gets its real length up front; the transcoded .ogv re-confirms it
	# in _finish_track_import. Floor keeps the lane a visible width if a probe fails.
	var dur := _probe_duration(source)
	if dur <= 0.0:
		dur = maxf(session.duration, 1.0)
	_push_undo("", "imported a track")
	session.tracks.append({
		"video_path": video, "source_path": source, "duration": dur,
		"clip_in": 0.0, "clip_out": dur, "offset": 0.0,
		"x": 0.68, "y": 0.04, "w": 0.28, "h": 0.28,
		"volume": 1.0,
	})
	while _track_runtime.size() <= idx:
		_track_runtime.append({})   # empty slot: _sync_tracks skips it until the video loads
	_refresh_lanes()                # the new timeline lane is on screen NOW
	_mark_dirty()
	_import_pending = {"source": source, "video": video, "index": idx}
	_set_status("⏳  Track added - transcoding its video in the background…")
	# Keep the audio this time (was -an): the track's own VideoStreamPlayer plays the
	# embedded Vorbis, mixed alongside the main clip, with per-track mute/volume (see
	# _sync_tracks). A source with no audio just yields a silent stream - no failure.
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", source,
		"-c:v", "libtheora", "-q:v", "6", "-g", "25",
		"-c:a", "libvorbis", "-q:a", "4",
		ProjectSettings.globalize_path(video)])
	_import_pid = OS.create_process("ffmpeg", args)
	if _import_pid <= 0:
		_cancel_pending_track()     # couldn't even start ffmpeg - roll the lane back out
		_set_status("⚠  Could not start ffmpeg for track import")


## Roll back the placeholder lane _start_track_import added (ffmpeg failed to start,
## or the transcode produced nothing usable). The pending track is always the last
## one (imports are serialized - see _prompt_import_track's guard).
func _cancel_pending_track() -> void:
	if _import_pending.has("index"):
		var idx := int(_import_pending.index)
		if idx >= 0 and idx < session.tracks.size():
			session.tracks.remove_at(idx)
			if idx < _track_runtime.size():
				_track_runtime.remove_at(idx)
			_refresh_lanes()
			_mark_dirty()
	_import_pending = {}
	_import_pid = -1


func _finish_track_import() -> void:
	if not _import_pending.has("index"):
		_import_pending = {}
		return
	var idx := int(_import_pending.index)
	var abs_video := ProjectSettings.globalize_path(String(_import_pending.video))
	var dur := _probe_duration(abs_video)
	if dur <= 0.0 or idx < 0 or idx >= session.tracks.size():
		# The lane is already on screen; the transcode still failed, so take it back
		# out rather than leave a lane whose PiP video would never load.
		_cancel_pending_track()
		_set_status("⚠  Track import failed (transcode produced no usable video)")
		return
	# Correct the lane to the actual transcoded length, keeping it full if untrimmed.
	var t: Dictionary = session.tracks[idx]
	var was_full: bool = float(t.get("clip_out", 0.0)) >= float(t.get("duration", 0.0)) - 0.05
	t["duration"] = dur
	if was_full:
		t["clip_out"] = dur
	_build_track_view(idx)          # now the .ogv exists: load its PiP player
	_refresh_lanes()
	_mark_dirty()
	_import_pending = {}
	if _status != null:
		_status.visible = false


## Each imported track is driven off the PRIMARY player's own clock (never its
## own independent playback state) - the same discipline the audio sync already
## follows (see _play's class doc), so live preview and an export relaunch trace
## the identical picture. A track that isn't currently inside its
## [offset, offset+span) window on the master timeline is paused and hidden;
## entering it seeks the track's own player to the matching local position and
## lets it run from there, only re-seeking again if it drifts (video seeks are
## heavy - constant tiny re-seeks would stutter, same reasoning as the 0.15s
## audio tolerance, just a bit wider since a video seek is coarser than an
## audio one).
func _sync_tracks() -> void:
	if _player == null:
		return
	var master_t := _player.stream_position
	for i in session.tracks.size():
		if i >= _track_runtime.size():
			continue
		var rt: Dictionary = _track_runtime[i]
		if not rt.has("player"):
			continue
		var track: Dictionary = session.tracks[i]
		var offset := float(track.get("offset", 0.0))
		var cin := float(track.get("clip_in", 0.0))
		var cout := float(track.get("clip_out", 0.0))
		var local_t := master_t - offset + cin
		var inside: bool = cout > cin and local_t >= cin and local_t < cout
		var tplayer: VideoStreamPlayer = rt.player
		var wrap: Control = rt.wrap
		var taudio: AudioStreamPlayer = rt.get("audio")   # may be null (silent track)
		# This track owns the PiP only when V has selected it (pip_track == i+1). The
		# box is still gated by inset_show so track modes fade like the main PiP does.
		var selected := _pip_track == i + 1
		# The clip's fade envelope: audio AND video ramp together (coupled) over the
		# fade_in seconds after the clip's start and the fade_out seconds before its end.
		# g is 0 at a faded edge, 1 across the middle; see _clip_fade_gain / _track_level_db.
		var g := _clip_fade_gain(master_t - offset, cout - cin,
			float(track.get("fade_in", 0.0)), float(track.get("fade_out", 0.0)))
		if inside:
			wrap.visible = _last_inset_show > 0.001 and selected
			wrap.modulate.a = g   # VIDEO fade, coupled to the audio one below
			if not bool(rt.active) or absf(tplayer.stream_position - local_t) > 0.2:
				tplayer.stream_position = local_t
				rt.active = true
			tplayer.paused = not _playing
			var view: TextureRect = rt.view
			if view != null:
				view.texture = tplayer.get_video_texture()
			if taudio != null:
				# pull-rope volume × the fade envelope, as a log/dB level
				taudio.volume_db = _track_level_db(float(track.get("volume", 1.0)), g)
				if _playing:
					if not taudio.playing:
						taudio.play(local_t)
					taudio.stream_paused = false
					if absf(taudio.get_playback_position() - local_t) > 0.2:
						taudio.seek(local_t)
				else:
					taudio.stream_paused = true   # paused / scrubbing: freeze its sound
		else:
			wrap.visible = false
			tplayer.paused = true   # outside its window: no picture...
			rt.active = false
			if taudio != null and taudio.playing:
				taudio.stop()        # ...and no sound; re-entry restarts it clean
	# The main clip's PiP shows only when it is the selected source (pip_track == 0);
	# a selected track replaces it. _apply_frame_state set _mask_wrap by inset_show
	# alone, so correct it here now that we know which source V picked.
	if _pip_track != 0:
		_mask_wrap.visible = false


func _build_render_view() -> void:
	var full := Control.new()
	full.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(full)
	_build_video_composition(full)
	for i in session.tracks.size():
		_build_track_view(i)
	# Product-demo chrome: if this session uses META anywhere, build the REAL editor
	# chrome (timeline, lanes, control panel) on top of the clean composition, fully
	# transparent. _apply_meta_chrome fades it in per frame with the meta envelope, so
	# the export shows clean video normally and the working editor during a meta
	# section (the recorded demo). Purely additive - no meta markers, nothing built,
	# the clean-video export path is untouched. mouse_filter IGNORE: never interactive.
	if _session_uses_meta():
		_meta_chrome = Control.new()
		_meta_chrome.set_anchors_preset(Control.PRESET_FULL_RECT)
		_meta_chrome.mouse_filter = Control.MOUSE_FILTER_IGNORE
		_meta_chrome.modulate.a = 0.0
		add_child(_meta_chrome)
		_chrome_parent = _meta_chrome
		_build_chrome()
		_chrome_parent = null
		_refresh_panel()
	# The export starts at the TIMELINE start (0), exactly like the live editor: live
	# playback begins at 0 and main_visible_at() never gates on clip_in, so the main
	# clip is shown from source 0 and markers/scenes can (and here do) sit before
	# clip_in. Seeking the export to clip_in instead silently dropped everything before
	# it - the "main video starts ~30s too late, early scenes truncated" report, on a
	# session whose clip_in was 37s. clip_in still bounds the restore clamp and the
	# kept-range END via content_end(); it is simply not the export's start.
	_player.play()
	_player.stream_position = 0.0
	_apply_frame_state(session.at_time(0.0))
	# The export quits on the deterministic movie clock reaching content_end() (see
	# _process), NOT on the audio finishing - an audio track shorter than the session
	# would otherwise cut the movie (and its trailing raw video) short.


func _build_editor_ui() -> void:
	_video_area = AspectRatioContainer.new()
	_video_area.set_anchors_preset(Control.PRESET_FULL_RECT)
	_video_area.offset_left = PANEL_W
	_video_area.ratio = 16.0 / 9.0
	add_child(_video_area)

	# A plain Control fills the AspectRatioContainer's one centered/letterboxed slot.
	var inner := Control.new()
	inner.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_video_area.add_child(inner)
	_build_video_composition(inner)
	for i in session.tracks.size():
		_build_track_view(i)

	_build_chrome()
	_build_export_ui()
	_build_feedback()
	_refresh_panel()
	_apply_frame_state(session.at_time(_player.stream_position))


## Where the interactive chrome (timeline, lanes, control panel) parents. Normally
## `self` (the live editor); during a render-mode export that uses META, it's the
## fading _meta_chrome overlay so a meta section can reveal the real working editor
## over the clean video (the recorded product demo).
func _chrome_host() -> Node:
	return _chrome_parent if _chrome_parent != null else self


## The editor chrome - the marker timeline, the trim/track lane stack, and the
## control panel - built into _chrome_host(). Split out of _build_editor_ui so the
## export path can build the SAME real widgets into an overlay (see _build_render_view
## / _apply_meta_chrome). Prereqs: session and _player already exist.
func _build_chrome() -> void:
	_tview = TimelineView.new()
	_tview.zoom = session.timeline_zoom          # restore the last zoom/pan (see _save_session)
	_tview.view_start = session.timeline_view_start

	_timeline = MaskTimeline.new()
	_timeline.session = session
	_timeline.player = _player
	_timeline.tview = _tview
	_timeline.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_timeline.offset_left = PANEL_W
	_timeline.offset_bottom = -90
	_timeline.offset_top = -90
	_timeline.scrubbed.connect(_on_scrub)
	_timeline.marker_picked.connect(_select_marker)
	_timeline.marker_drag_started.connect(func(_m): _push_undo("", "moved a marker"))
	_timeline.marker_moved.connect(func(_m):
		_refresh_marker_label()
		_mark_dirty())
	_chrome_host().add_child(_timeline)

	# The trim/track lane stack - the primary clip's own trim block, plus one
	# lane per imported track - sits directly above the marker strip, sharing
	# its ruler via the same _tview (see _refresh_lanes for why its height is
	# computed and set explicitly rather than left to container auto-sizing).
	_lanes_col = VBoxContainer.new()
	_lanes_col.add_theme_constant_override("separation", 2)
	_lanes_col.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_lanes_col.offset_left = PANEL_W
	_chrome_host().add_child(_lanes_col)
	_refresh_lanes()

	_build_panel()


## The META mirror source: the editor's OWN previous frame, read back from the main
## viewport into u_workspace. A full-window GPU->CPU readback (expensive), so the
## caller (_process) only runs it while a meta layer is actually on screen. Feeding
## this window back into the video surface - which lives IN this window - is the
## infinite mirror; the one-frame delay is inherent and wanted (each frame nests one
## level deeper). Downscaled before upload since the mirror draws small anyway.
func _capture_workspace() -> void:
	# Headless has no real framebuffer to read back (the dummy renderer's viewport
	# texture is null - get_image would error every frame). It also never records a
	# movie, so there is nothing to mirror; the windowed Movie Maker export IS a real
	# GPU context, so this only ever skips genuine no-op cases.
	if DisplayServer.get_name() == "headless":
		return
	var vtex := get_viewport().get_texture()
	if vtex == null:
		return
	var img := vtex.get_image()
	if img == null or img.is_empty():
		return
	if img.get_width() > 960:
		var h := int(round(960.0 * float(img.get_height()) / float(maxi(1, img.get_width()))))
		img.resize(960, maxi(1, h), Image.INTERPOLATE_BILINEAR)
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var sz := Vector2i(img.get_width(), img.get_height())
	if _workspace_tex == null or _workspace_tex.get_size() != Vector2(sz):
		_workspace_tex = ImageTexture.create_from_image(img)
	else:
		_workspace_tex.update(img)
	_mat_main.set_shader_parameter("u_workspace", _workspace_tex)
	_mat_inset.set_shader_parameter("u_workspace", _workspace_tex)


## Render-mode only: fade the editor-chrome overlay in with the meta envelope, so the
## export shows clean video normally and the real working editor during a meta section.
func _apply_meta_chrome(amount: float) -> void:
	if _meta_chrome == null:
		return
	_meta_chrome.modulate.a = smoothstep(0.0, 1.0, clampf(amount, 0.0, 1.0))


func _session_uses_meta() -> bool:
	for m in session.markers:
		if int(m.get("effect_a", 0)) == MaskSession.EFFECT_META:
			return true
	return false


## The same backtick feedback console the auto/manual show has (see feedback.gd),
## with mask-specific plumbing injected: the descriptor snapshots everything needed
## to debug a masking complaint (playhead, the fully-resolved layer stack at that
## instant, every marker, keying globals, audio envelope), freeze pauses playback
## while typing (restoring the prior play state after), and advance is a no-op -
## the playhead is the user's business, not the console's.
func _build_feedback() -> void:
	_feedback = preload("res://scripts/feedback.gd").new()
	_feedback.describe = _feedback_descriptor
	_feedback.freeze = func(on: bool):
		if on:
			_was_playing_before_feedback = _playing
			_play(false)
		elif _was_playing_before_feedback:
			_play(true)
	_feedback.advance = func(): pass
	add_child(_feedback)
	# Always present, same reasoning as main.gd's own _assistant: it's also
	# the feedback browser (review/delete old submissions), which shouldn't
	# require an assistant backend selected to use. Assistant itself gates
	# actually DISPATCHING anything on the splash's persisted backend choice
	# (see splash.gd) - this just wires the console up. One editor session =
	# one Assistant instance for its whole lifetime, no re-entrancy to guard
	# against (open_source() runs once per process here).
	var assistant := preload("res://scripts/assistant.gd").new()
	add_child(assistant)
	_feedback.submitted.connect(assistant.enqueue)


## Everything I'd want to know about "this frame looks wrong": where we are, what
## the timeline resolved to (including each live layer's full parameter set and
## envelope), the raw marker list, and the session's file identity. The screenshot
## the console pairs with this carries the artifact itself.
func _feedback_descriptor() -> Dictionary:
	var t: float = _player.stream_position if _player != null else 0.0
	var p := session.at_time(t)
	return {
		"mode": "mask",
		"time": t,
		"time_str": MaskTimeline.format_time(t),
		"session_path": _session_path,
		"video_path": session.video_path,
		"source_path": session.source_path,
		"duration": session.duration,
		"resolved_state": p,             # globals + amounts + the live layer stack
		"layer_count": (p.get("layers", []) as Array).size(),
		"markers": session.markers,
		"marker_count": session.markers.size(),
		"audio_env": _env_at(t),
		"peek_raw": _peek_raw,
		"playing": _playing,
	}


func _build_panel() -> void:
	var panel := PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_LEFT_WIDE)
	panel.offset_right = PANEL_W
	panel.clip_contents = true   # belt-and-suspenders: a child's minimum size can
	# never visually push the panel past PANEL_W and over the timeline, whatever
	# happens inside (see the autowrap fix below for the actual root cause this
	# guards - a long unwrapped Label's natural width was doing exactly that).
	_chrome_host().add_child(panel)

	# Two independently-scrolling regions, stacked: the controls above (which can
	# get tall - color pickers, a dozen sliders) scroll in whatever space is left,
	# and the sequential ramp/damp list is pinned to the bottom with its own fixed-
	# height scroll, so it's always reachable without paging through everything above.
	var outer := VBoxContainer.new()
	outer.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	panel.add_child(outer)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer.add_child(scroll)

	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 14)
	scroll.add_child(margin)

	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 8)
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	margin.add_child(col)

	# Two buttons up here: Help and Import track. Everything ELSE that used to
	# be a button (play, view cycle, peek, undo, redo) was pure duplication of
	# a key and moved to the keyboard + the help overlay. Import stayed a
	# button on purpose: it's the ONE action with no on-screen equivalent and
	# no way to discover (a hidden T shortcut is exactly the "I don't see a
	# clear way to do this" complaint - a keyboard-only import that also
	# depends on nothing having eaten the keystroke is not good enough for the
	# single essential action). Both a visible button AND the T key now.
	var title_row := HBoxContainer.new()
	title_row.add_theme_constant_override("separation", 6)
	col.add_child(title_row)
	var title := Label.new()
	title.text = "ghost-mask"
	title.add_theme_font_size_override("font_size", 22)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(title)
	var import_btn := Button.new()
	import_btn.text = "⬆ Track"
	import_btn.tooltip_text = "Import a second video as a picture-in-picture track (T)"
	import_btn.focus_mode = Control.FOCUS_NONE
	import_btn.pressed.connect(_prompt_import_track)
	title_row.add_child(import_btn)
	var help_btn := Button.new()
	help_btn.text = "?  Help"
	help_btn.tooltip_text = "Keyboard map (F1)"
	help_btn.focus_mode = Control.FOCUS_NONE
	help_btn.pressed.connect(_toggle_help)
	title_row.add_child(help_btn)

	var time_row := HBoxContainer.new()
	time_row.add_theme_constant_override("separation", 10)
	col.add_child(time_row)
	_time_label = Label.new()
	_time_label.add_theme_font_size_override("font_size", 15)
	_time_label.add_theme_color_override("font_color", Color(0.85, 0.9, 1.0))
	time_row.add_child(_time_label)
	_view_label = Label.new()
	_view_label.add_theme_font_size_override("font_size", 13)
	_view_label.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	time_row.add_child(_view_label)
	_build_help_overlay()

	# One channel by default (a channel = one target color + one effect + one
	# strength); the second is opt-in behind a toggle. Channels are independent
	# layers in the shader, not competing "sides" - so there's no swap control any
	# more either (swapping meant something when every pixel was forced to one side
	# or the other; independent channels just re-pick their own colors).
	_grp_color = VBoxContainer.new()
	_grp_color.add_theme_constant_override("separation", 8)
	col.add_child(_grp_color)
	_grp_color.add_child(_label("Key color", "The color this channel targets - what it keys or paints"))
	_color_a = ColorPickerButton.new()
	_color_a.focus_mode = Control.FOCUS_NONE
	_color_a.custom_minimum_size = Vector2(0, 40)
	_color_a.edit_alpha = false
	_color_a.tooltip_text = "The color this channel targets - what it keys or paints"
	_color_a.color_changed.connect(func(c):
		_hue_a.set_value_no_signal(c.h)
		_edit("hue_a", c.h))
	_grp_color.add_child(_color_a)
	col.add_child(_label("Effect", "Which visual treatment this layer applies"))
	_effect_a = _effect_menu(col, func(id): _edit("effect_a", float(id)))

	# Every option below - this channel's grading through this effect's own
	# pattern/echo/weather/tendril knobs - lives in one flat, sortable list instead
	# of fixed titled groups (see feedback/0011: the group boundaries were drawn as
	# separator lines that "made no sense" and just ate space). Which rows are
	# actually visible still follows the selected effect exactly as before
	# (MaskSession.EFFECT_CONTROLS / PATTERN_KNOBS, see _update_effect_controls) -
	# only their ON-SCREEN ORDER is now driven by _sort_mode. _sort_dropdown picks
	# it directly - same shape as the Effect dropdown just above; see _apply_sort.
	var sort_tip := "How the options below are ordered - alphabetical (either " + \
		"direction), or energy: the fullest slider floats to the top, pick-type " + \
		"options with no slider sink to the bottom (A → Z among themselves)"
	col.add_child(_label("Sort", sort_tip))
	_sort_dropdown = OptionButton.new()
	_sort_dropdown.focus_mode = Control.FOCUS_NONE
	_sort_dropdown.tooltip_text = sort_tip
	for i in _SORT_MODES.size():
		_sort_dropdown.add_item(_SORT_MODES[i], i)
	_sort_dropdown.select(_sort_mode)
	_sort_dropdown.item_selected.connect(func(id):
		_sort_mode = id
		_apply_sort())
	col.add_child(_sort_dropdown)

	_grp_options = VBoxContainer.new()
	_grp_options.add_theme_constant_override("separation", 8)
	col.add_child(_grp_options)

	_intensity_a = _slider(_grp_options, "Strength", 0.0, 1.0, func(v): _edit("intensity_a", v),
		"How strongly this layer's effect applies")
	_intensity_label = _intensity_a.get_meta("field_label")
	_register_option(_intensity_a)
	# A dedicated grading slider on the same hue_a field the swatch above sets -
	# color grading (the feedback that drove this) wants fine numeric dragging,
	# not just a color-wheel pick. Kept in sync both ways: see _refresh_panel and
	# _color_a's color_changed above.
	_hue_a = _slider(_grp_options, "Hue", 0.0, 1.0, func(v):
		_color_a.color = Color.from_hsv(v, 0.85, 0.9)
		_edit("hue_a", v), "Shifts this layer's color - drag for fine color grading")
	_register_option(_hue_a)

	# Option rows, shown per the selected effect's needs (the control hierarchy,
	# MaskSession.EFFECT_CONTROLS): a slider that does nothing for the current
	# effect is not on screen for it. Erase shows none of these (projection is
	# gate-free); restore shows only the threshold, relabeled as its reach;
	# the volumetrics show everything. See _update_effect_controls.
	_threshold = _slider(_grp_options, "Threshold", 0.0, 1.0, func(v): _edit("threshold", v),
		"How far a pixel's hue may drift from the key color and still be masked")
	_threshold_label = _threshold.get_meta("field_label")
	_register_option(_threshold)

	_feather = _slider(_grp_options, "Feather", 0.0, 0.5, func(v): _edit("feather", v),
		"Softness of the mask's edge - 0 is a hard cutoff")
	_register_option(_feather)
	_sat_floor = _slider(_grp_options, "Min colorfulness", 0.0, 1.0, func(v): _edit("sat_floor", v),
		"Minimum saturation a pixel needs before it can be keyed at all")
	_register_option(_sat_floor)

	# The wisp field's placement - pan/zoom the pattern over the frame (keyframe a
	# tendril onto an eye), and dial its coverage from one wisp to an engulfing.
	# All continuous marker fields, so they blend through ramps/damps.
	_fx_x = _slider(_grp_options, "Pan X", -2.0, 2.0, func(v): _edit("fx_x", v),
		"Shifts the pattern horizontally over the frame")
	_fx_x.step = 0.01
	_fx_x_label = _fx_x.get_meta("field_label")
	_register_option(_fx_x)
	_fx_y = _slider(_grp_options, "Pan Y", -2.0, 2.0, func(v): _edit("fx_y", v),
		"Shifts the pattern vertically over the frame")
	_fx_y.step = 0.01
	_fx_y_label = _fx_y.get_meta("field_label")
	_register_option(_fx_y)
	_fx_scale = _slider(_grp_options, "Scale", 0.1, 8.0, func(v): _edit("fx_scale", v),
		"Zoom of the effect's pattern - 1 is nominal size")
	_fx_scale.exp_edit = true
	_register_option(_fx_scale)
	_fx_density = _slider(_grp_options, "Coverage", 0.0, 1.0, func(v): _edit("fx_density", v),
		"How much of the keyed region the pattern consumes - 0 untouched, 1 fully devoured")
	_fx_density_label = _fx_density.get_meta("field_label")
	_register_option(_fx_density)
	_fx_contrast = _slider(_grp_options, "Contrast", 0.0, 1.0, func(v): _edit("fx_contrast", v),
		"Edge hardness of the pattern - 0.5 is neutral")
	_fx_contrast_label = _fx_contrast.get_meta("field_label")
	_register_option(_fx_contrast)
	_fx_speed = _slider(_grp_options, "Velocity", 0.1, 4.0, func(v): _edit("fx_speed", v),
		"Speed multiplier for the pattern's motion")
	_fx_speed.exp_edit = true
	_register_option(_fx_speed)
	_resonance = _slider(_grp_options, "Resonance", 0.0, 1.0, func(v): _edit("resonance", v),
		"Audio drive - how strongly this layer reacts to the track's live energy")
	_register_option(_resonance)

	_fx_lag = _slider(_grp_options, "Lag (s)", 0.05, 2.4, func(v): _edit("fx_lag", v),
		"How far back the lagged frame reaches")
	_fx_lag.exp_edit = true
	_fx_lag_label = _fx_lag.get_meta("field_label")
	_register_option(_fx_lag)
	_fx_smooth = _slider(_grp_options, "Smoothing", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"Stutter → smear: 0 is a discrete stutter, 1 is a wide temporal blend")
	_register_option(_fx_smooth)

	# Snow's own view onto fx_smooth - a separate widget from Smoothing above (same
	# stored field, different meaning; the two rows never show together, see
	# _update_effect_controls, so there's no risk of them fighting over what the
	# slider looks like).
	_gust = _slider(_grp_options, "Gust", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"0 is a steady drift, 1 is chaotic gusts")
	_register_option(_gust)

	# Fur's tendril dynamics - fur-only views onto fx_smooth/fx_lag, the same
	# stored-field reuse as Gust above (the rows never show together, see
	# _update_effect_controls).
	_undul = _slider(_grp_options, "Undulation", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"Traveling waves along each strand")
	_register_option(_undul)
	_coil = _slider(_grp_options, "Coil", 0.0, 1.0, func(v): _edit("fx_lag", v),
		"Eddies and spiral curl")
	_register_option(_coil)
	# Stickiness - 0 keeps today's free coat exactly; higher values thin the strands
	# away from natural anchors so the hair clings to the keyed surface, the tracked
	# landmark/motion centroid, and brighter regions (see the shader's fur branch).
	_stick = _slider(_grp_options, "Stickiness", 0.0, 1.0, func(v): _edit("fx_stick", v),
		"0 is a free coat, 1 clings to the face/motion")
	_register_option(_stick)

	# Every marker is a ramp or a damp - there is no plain/neutral marker (see
	# MaskSession class doc). Both transition TO this marker's values; the kind is
	# which side of the anchor the transition occupies: a ramp eases in BEFORE,
	# complete at the anchor; a damp begins AT the anchor and accumulates after.
	# Lives in _grp_options (not a fixed spot below it) so it's part of the same
	# sortable list as the effect knobs - see _apply_sort/_register_option; a
	# pick-type control with no slider, it sorts to the bottom under Energy mode.
	var _kind_label := _label("Kind",
		"Which way this marker's change runs - ramp eases in before it, damp accumulates after")
	_grp_options.add_child(_kind_label)
	_kind = OptionButton.new()
	_kind.focus_mode = Control.FOCUS_NONE
	_kind.tooltip_text = "Ramp eases in before this marker; damp accumulates after it"
	_kind.set_meta("field_label", _kind_label)
	for i in MaskSession.MARKER_KINDS.size():
		_kind.add_item(MaskSession.MARKER_KINDS[i].capitalize(), i)
	_kind.item_selected.connect(func(id): _edit("kind", float(id)))
	_grp_options.add_child(_kind)
	_register_option(_kind)
	# Exponential response: fine-grained fractions of a second on the left, whole
	# minutes on the right - one slider covers a subtle 0.2s blend and a transition
	# spanning the entire clip. (exp_edit needs a strictly positive min.)
	_marker_duration = _slider(_grp_options, "Ramp/damp span (s)", 0.05, maxf(8.0, session.duration),
		func(v): _edit("duration", v),
		"How long the ramp (before) or damp (after) transition takes, in seconds")
	_marker_duration.exp_edit = true
	_marker_duration.step = 0.01
	_register_option(_marker_duration)

	# --- create/delete + the sequential list, pinned to the panel's bottom with its
	# --- own scroll - the whole "manage markers" workflow stays visible together,
	# --- rather than the create buttons living up in the scrolling edit area where
	# --- reaching them means scrolling past everything else first.
	var list_margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		list_margin.add_theme_constant_override("margin_" + side, 10)
	outer.add_child(list_margin)

	var list_col := VBoxContainer.new()
	list_col.add_theme_constant_override("separation", 4)
	list_margin.add_child(list_col)

	_marker_label = Label.new()
	_marker_label.add_theme_font_size_override("font_size", 12)
	_marker_label.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	_marker_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	list_col.add_child(_marker_label)

	# See feedback/0019: a preview of what Ctrl+Z would revert, so undo isn't a
	# blind guess - kept right above the buttons that create the history it
	# describes. Populated by _refresh_history_label, driven off _undo_descs.
	_history_label = Label.new()
	_history_label.add_theme_font_size_override("font_size", 12)
	_history_label.add_theme_color_override("font_color", Color(0.55, 0.6, 0.7))
	_history_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_history_label.tooltip_text = "What Ctrl+Z would revert right now"
	list_col.add_child(_history_label)
	_refresh_history_label()

	var mrow := HBoxContainer.new()
	list_col.add_child(mrow)
	var ramp_btn := Button.new()
	ramp_btn.text = "+ Ramp"
	ramp_btn.tooltip_text = "Eases IN before the playhead, arriving here complete"
	ramp_btn.focus_mode = Control.FOCUS_NONE
	ramp_btn.pressed.connect(func(): _add_marker_at_playhead(0))
	mrow.add_child(ramp_btn)
	var damp_btn := Button.new()
	damp_btn.text = "+ Damp"
	damp_btn.tooltip_text = "Begins here and accumulates over the span that follows"
	damp_btn.focus_mode = Control.FOCUS_NONE
	damp_btn.pressed.connect(func(): _add_marker_at_playhead(1))
	mrow.add_child(damp_btn)
	var del_btn := Button.new()
	del_btn.text = "Delete"
	del_btn.focus_mode = Control.FOCUS_NONE
	del_btn.pressed.connect(_delete_selected)
	mrow.add_child(del_btn)

	list_col.add_child(_label("In order"))
	var list_scroll := ScrollContainer.new()
	list_scroll.custom_minimum_size = Vector2(0, 150)
	list_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	list_col.add_child(list_scroll)

	_marker_list = VBoxContainer.new()
	_marker_list.add_theme_constant_override("separation", 2)
	_marker_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	list_scroll.add_child(_marker_list)


## `tip` is the full explanation shown on mouse-over (Godot's native tooltip) -
## labels themselves stay short so the panel reads clean; see the feedback that
## drove this (0005): inline "Label - long explanation" text was the ambiguity
## complaint, not just missing descriptions.
func _label(text: String, tip: String = "") -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	# A long, unwrapped label's natural width becomes the whole column's minimum
	# width - that's exactly what pushed the panel wider than PANEL_W and over the
	# timeline (a real bug: "no marker selected..." is longer than most other panel
	# text, so it only showed up after deleting one). Word-wrap caps the minimum to
	# the longest WORD instead of the longest sentence.
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	l.mouse_filter = Control.MOUSE_FILTER_PASS   # let the tooltip trigger over the label itself
	l.tooltip_text = tip
	return l


func _slider(col: VBoxContainer, text: String, lo: float, hi: float, cb: Callable, tip: String = "") -> HSlider:
	var lbl := _label(text, tip)
	col.add_child(lbl)
	var s := HSlider.new()
	# Remember the label so a single knob (label + slider) can be hidden together
	# when the effect doesn't consume it - see _show_field / _update_effect_controls.
	s.set_meta("field_label", lbl)
	s.focus_mode = Control.FOCUS_NONE
	# The wheel belongs to the panel's ScrollContainer, never to a slider you
	# happen to pass over on the way down - the panel is long enough now that
	# scrolling it drags random knobs (and silently edits the marker under
	# them). Drag-only.
	s.scrollable = false
	s.min_value = lo
	s.max_value = hi
	s.step = (hi - lo) / 200.0
	s.tooltip_text = tip   # hovering the slider itself explains it too, not just the label above
	s.value_changed.connect(cb)
	col.add_child(s)
	return s


## A left-anchored pull-rope volume knob for a lane (see VolumeKnob): hold and drag
## away to set a continuous 0..1 level with an asymptotic ceiling. `getter` reads the
## stored volume, `setter` writes it - the same widget drives a track's `volume` and
## the main clip's `main_volume`. Left side deliberately: the right corner sits under
## the floating assistant chat button. `icon` shows the one-time speaker-glyph hint -
## reserved for the main clip's own knob only (see VolumeKnob.show_icon).
func _volume_knob(getter: Callable, setter: Callable, accent: Color, icon: bool = false) -> VolumeKnob:
	var k := VolumeKnob.new()
	k.accent = accent
	k.show_icon = icon
	k.set_anchors_preset(Control.PRESET_TOP_LEFT)
	# offset_left starts past TrackLane._EDGE_W (8px): the primary lane's own trim/fade
	# "in" handles sit at local x0 = 0 (its offset is always 0, and the default view
	# starts at t=0 too), so a knob planted right at the corner silently ate every click
	# meant for that edge - "drag the left edge" looked broken specifically on the main
	# track, since only ITS x0 is pinned to 0 - see feedback/0017.
	k.offset_left = 12
	k.offset_top = 2
	k.offset_right = 34
	k.offset_bottom = 20
	k.get_v = getter
	k.set_v = func(v):
		setter.call(v)
		_mark_dirty()   # cheap + debounced (autosave cooldown), fine to fire while pulling
	return k


## Show/hide one knob - the slider AND the label above it (stored on the slider
## by _slider, or passed for the hand-built pattern rows). Used to prune pattern
## knobs an effect doesn't read (MaskSession.PATTERN_KNOBS).
func _show_field(slider: Control, vis: bool, label: Control = null) -> void:
	if slider != null:
		slider.visible = vis
	var lbl: Variant = label
	if lbl == null and slider != null and slider.has_meta("field_label"):
		lbl = slider.get_meta("field_label")
	if lbl != null:
		(lbl as Control).visible = vis


## Registers a slider or pick-type control (label stashed via set_meta, either by
## _slider or by hand for non-Range controls like an OptionButton) into the flat
## sortable list - see _apply_sort.
func _register_option(ctrl: Control) -> void:
	_options.append({"label": ctrl.get_meta("field_label"), "control": ctrl})


## Re-orders _options within _grp_options per the active sort mode: alphabetical by
## each row's CURRENT label text (which itself changes per effect - e.g. Contrast ->
## Sensitivity for snow), or "energy" - how wide each slider's fill currently reads,
## fullest first (see _option_energy). Pick-type controls with no slider (e.g. Kind)
## have no fill to compare, so they sink below every slider and break ties alphabetically
## among themselves. Hidden rows (this effect doesn't use them) are left trailing after
## the visible ones - their order doesn't matter, they're off screen. Re-run whenever
## the effect/visibility changes (_update_effect_controls) or a mode is picked in
## _sort_dropdown - never on a live drag, so a slider never jumps out from under the
## mouse mid-edit.
func _apply_sort() -> void:
	if _grp_options == null:
		return
	var shown: Array = _options.filter(func(o): return o.control.visible)
	var hidden: Array = _options.filter(func(o): return not o.control.visible)
	match _sort_mode:
		1:
			shown.sort_custom(func(a, b): return a.label.text.nocasecmp_to(b.label.text) > 0)
		2:
			shown.sort_custom(func(a, b):
				var ea: float = _option_energy(a.control)
				var eb: float = _option_energy(b.control)
				if ea == eb:
					return a.label.text.nocasecmp_to(b.label.text) < 0
				return ea > eb)
		_:
			shown.sort_custom(func(a, b): return a.label.text.nocasecmp_to(b.label.text) < 0)
	var idx := 0
	for o in shown + hidden:
		_grp_options.move_child(o.label, idx)
		idx += 1
		_grp_options.move_child(o.control, idx)
		idx += 1


## How "present" a control's current value is, for the "Energy" sort mode. For a
## slider this is its actual fill fraction - the handle's position between the
## track's left and right ends, min at 0 and max at 1 - matching the bar you
## actually see, so a slider barely nudged off its floor reads as low energy even
## if that floor sits far from the range's midpoint. Range.get_as_ratio() (rather
## than a hand-rolled linear (value-min)/span) is what makes this match what's on
## screen for the exp_edit sliders (Scale/Velocity/Lag/Ramp-damp span): those are
## drawn on a logarithmic track, so a linear fraction would read a handle sitting
## visibly right-of-center as barely-there. Pick-type controls with no slider (not
## a Range, e.g. Kind's OptionButton) have nothing to compare, so they return a
## sentinel below any real slider value - see _apply_sort, where that sinks them to
## the bottom and falls back to alphabetical among themselves.
func _option_energy(ctrl: Control) -> float:
	if not (ctrl is Range):
		return -1.0
	return (ctrl as Range).get_as_ratio()


func _effect_menu(col: VBoxContainer, cb: Callable) -> OptionButton:
	var ob := OptionButton.new()
	ob.focus_mode = Control.FOCUS_NONE
	ob.tooltip_text = "Which visual treatment this layer applies"
	for i in MaskSession.MASK_EFFECTS.size():
		ob.add_item(MaskSession.MASK_EFFECTS[i], i)
	ob.item_selected.connect(cb)
	col.add_child(ob)
	return ob


# --- marker editing -----------------------------------------------------------

## Every panel edit targets the selected marker; if none is selected yet, planting
## one at the current playhead is the edit's first move (a knob you touch becomes a
## marker - no separate "create" step needed for the common case). Defaults to a
## ramp when auto-created this way; press +Damp explicitly for the other kind.
func _edit(field: String, value: float) -> void:
	var m: Variant = _selected
	if m == null:
		_push_undo("", "created a marker")   # about to create one - always its own boundary
		m = session.add_marker(_player.stream_position if _player != null else 0.0)
		_selected = m
		_select_generation += 1
	else:
		_push_undo("marker:%d:%s" % [_select_generation, field], "adjusted %s" % field.capitalize())
	m[field] = value
	# Assigning a drawing effect to a marker whose OWN view shows no fx surface is
	# a silent foot-gun: the layer holds forever but this marker's view hides it -
	# "I configured fire and nothing shows" (a real session lost its fire exactly
	# this way: the view button had been cycled to Raw on the same marker). Bump
	# the view to the nearest fx-showing mode, preserving the surface choice:
	# raw -> masked, pip_raw -> pip. Only on effect/intensity edits - never behind
	# the user's back on unrelated knobs, and never for restore (draws nothing).
	if field == "effect_a" or field == "intensity_a":
		var vm := int(m.get("view_mode", 2.0))
		var eid := int(m.get("effect_a", 0))
		var drawing: bool = eid != MaskSession.EFFECT_RESTORE \
			and eid != MaskSession.EFFECT_CLEAR \
			and float(m.get("intensity_a", 0.0)) > 0.0
		if drawing and (vm == 2 or vm == 3):
			m["view_mode"] = 1.0 if vm == 2 else 0.0
	if field == "effect_a":
		_update_effect_controls(int(value))
	_timeline.selected = _selected
	_refresh_marker_label()
	_mark_dirty()


func _add_marker_at_playhead(kind_id: int) -> void:
	_push_undo("", "added a %s marker" % MaskSession.MARKER_KINDS[kind_id].capitalize())
	_selected = session.add_marker(_player.stream_position if _player != null else 0.0, kind_id)
	_select_generation += 1
	_timeline.selected = _selected
	_refresh_panel()
	_mark_dirty()


func _delete_selected() -> void:
	if _selected != null:
		_push_undo("", "deleted a marker")
		session.remove_marker(_selected)
		_selected = null
		_timeline.selected = null
		_refresh_panel()
		_mark_dirty()


## Temporal capture: when the playhead crosses into a new _ECHO_INTERVAL slot,
## snapshot the current frame (quarter-res) into the ring and push the ring to
## both materials in AGE ORDER (u_echo0 = newest). GPU readback at ~3Hz is cheap
## enough for a demo; frames where no whisp/echo/snow/oracle/serpent/chimera
## layer is actually on screen (_temporal_active) skip all of it.
func _maybe_capture_echo() -> void:
	if _player == null or session == null:
		return
	var pos := _player.stream_position
	# The readback below (tex.get_image()) is a synchronous GPU->CPU stall. We only
	# ever want to pay it when it MATTERS and when it WON'T fight the user:
	#   - during playback (the effects need to advance), OR
	#   - once, when a scrub has SETTLED (playhead stationary since last frame) - so
	#     dragging the playhead stays responsive and the preview refreshes on release.
	# Never per-scrub-position mid-drag, which is what made clicking the timeline lag
	# and starved the audio. render_mode auto-plays, so export still captures every
	# slot deterministically.
	if not _playing:
		var settled := absf(pos - _prev_pos) < 0.0005
		_prev_pos = pos
		if not settled:
			return
	else:
		_prev_pos = pos
	if not _temporal_active:
		return
	var slot := int(pos / _ECHO_INTERVAL)
	if slot == _echo_slot:
		return
	var tex := _player.get_video_texture()
	if tex == null:
		return
	var img := tex.get_image()
	if img == null or img.is_empty():
		return
	_echo_slot = slot
	img.resize(480, 270, Image.INTERPOLATE_BILINEAR)
	_echo_ring[slot % 8] = ImageTexture.create_from_image(img)
	for age in 8:
		var t: Variant = _echo_ring[((slot - age) % 8 + 8) % 8]
		if t == null:
			t = _echo_ring[slot % 8]
		for mat in [_mat_main, _mat_inset]:
			mat.set_shader_parameter("u_echo%d" % age, t)
	_update_whisp_anchor(img)
	# The track readback is chimera's alone - skip it entirely unless a chimera layer
	# is actually rendering this frame (it usually isn't: the marker sits at one point
	# in the timeline). This is what my earlier change was paying for everywhere.
	if _chimera_active:
		_update_track_frame()


## The anchor uniform, glided per frame: lerp(prev EMA, latest EMA) by the
## playhead's fraction through the current capture slot. Deterministic (pure
## function of playback position + capture history) and continuous - the
## pattern drifts to each new lock instead of jumping there.
func _push_anchor() -> void:
	var f := clampf(fposmod(_player.stream_position, _ECHO_INTERVAL) / _ECHO_INTERVAL, 0.0, 1.0)
	var anchor := _anchor_prev.lerp(_anchor_ema, f)
	# Chimera's landmark frames, glided on the same fraction (position + size, both
	# faces). Pure function of playback position + capture history, so live and
	# export trace the identical lock.
	var anchor_scale := lerpf(_anchor_scale_prev, _anchor_scale_ema, f)
	var track_anchor := _track_anchor_prev.lerp(_track_anchor_ema, f)
	var track_scale := lerpf(_track_scale_prev, _track_scale_ema, f)
	for mat in [_mat_main, _mat_inset]:
		mat.set_shader_parameter("u_anchor", anchor)
		mat.set_shader_parameter("u_anchor_scale", anchor_scale)
		mat.set_shader_parameter("u_track_anchor", track_anchor)
		mat.set_shader_parameter("u_track_scale", track_scale)
		# Only push once impulses exist - a short array is silently dropped by
		# Godot (see the u_l_* comment above), so an empty/partial one is worse
		# than leaving the shader's own all-zero (inactive) uniform defaults.
		if _wave_amp.size() == _WAVE_SLOTS:
			mat.set_shader_parameter("u_wave_pos", _wave_pos)
			mat.set_shader_parameter("u_wave_time", _wave_time)
			mat.set_shader_parameter("u_wave_amp", _wave_amp)


## The landmark anchor (whisp's field origin, chimera's graft window): the
## first whisp-or-chimera marker's target-color mass centroid in the captured
## frame, EMA-smoothed so the lock glides to landmarks instead of jittering
## with noise. When the key color exists NOWHERE (flat lighting), the motion
## centroid anchors instead - see the fallback below. Fur no longer reads
## this - its strands root per-pixel on the keyed surface itself (see
## fur_root_mass / the fur branch of apply_layer in mask_split.gdshader).
func _update_whisp_anchor(img: Image) -> void:
	var hue := -1.0
	for m in session.markers:
		var e := int(m.get("effect_a", 0))
		# Fur joins whisp/chimera in driving the anchor: its Stickiness cue wants the
		# key colour's centroid (with the motion-centroid fallback for flat footage) to
		# track the surface the strands root on. Fur ignores u_anchor at Stickiness 0,
		# so this has no effect on the existing look.
		if e == 5 or e == MaskSession.EFFECT_CHIMERA or e == MaskSession.EFFECT_FUR:
			hue = float(m.get("hue_a", 0.0))
			break
	if hue < 0.0:
		return
	var tc := Color.from_hsv(hue, 1.0, 1.0)
	var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
	var tdir := Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized()
	img.resize(48, 27, Image.INTERPOLATE_BILINEAR)
	# Read the frame as one flat RGBA8 buffer instead of 1296 Image.get_pixel()
	# calls. This runs every _ECHO_INTERVAL during playback (once a temporal effect
	# is live), right after the synchronous GPU readback - the per-pixel Color
	# construction get_pixel does was a measurable slice of that periodic hitch. The
	# math is unchanged; only the pixel access is. (_face_frame does the same.)
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	var acc := Vector2.ZERO
	var acc2 := 0.0          # weighted sum of |pos|^2 - second moment, for the RMS radius (size)
	var wsum := 0.0
	var macc := Vector2.ZERO
	var macc2 := 0.0
	var msum := 0.0
	var have_prev := _wave_prev_lum.size() == 48 * 27
	if not have_prev:
		_wave_prev_lum.resize(48 * 27)
	var motion := 0.0
	for y in 27:
		for x in 48:
			var idx := y * 48 + x
			var base := idx * 4
			var r := float(data[base]) / 255.0
			var g := float(data[base + 1]) / 255.0
			var b := float(data[base + 2]) / 255.0
			var l := 0.299 * r + 0.587 * g + 0.114 * b
			var pos := Vector2((float(x) + 0.5) / 48.0, (float(y) + 0.5) / 27.0)
			var pr := maxf(0.0, (r - l) * tdir.x + (g - l) * tdir.y + (b - l) * tdir.z)
			acc += pos * pr
			acc2 += pos.length_squared() * pr
			wsum += pr
			if have_prev:
				var dm := absf(l - _wave_prev_lum[idx])
				motion += dm
				macc += pos * dm
				macc2 += pos.length_squared() * dm
				msum += dm
			_wave_prev_lum[idx] = l
	if wsum > 0.01:
		_anchor_prev = _anchor_ema
		_anchor_ema = _anchor_ema.lerp(acc / wsum, 0.15)
		# Size = RMS radius of the mass about its own centroid (Var = E[|p|^2] -
		# |E[p]|^2), the second half of the main face's landmark frame.
		_anchor_scale_prev = _anchor_scale_ema
		_anchor_scale_ema = lerpf(_anchor_scale_ema,
			clampf(sqrt(maxf(1e-4, acc2 / wsum - (acc / wsum).length_squared())), 0.05, 0.9), 0.15)
	elif msum > 0.3:
		# FLAT-LIGHTING FALLBACK (chimera's first test case): standard, flat
		# footage may carry the key color NOWHERE - then the landmark is
		# wherever the pixels MOVE. The motion centroid of a talking head IS
		# the head; slower EMA than the color lock because motion is noisier
		# frame to frame.
		_anchor_prev = _anchor_ema
		_anchor_ema = _anchor_ema.lerp(macc / msum, 0.1)
		_anchor_scale_prev = _anchor_scale_ema
		_anchor_scale_ema = lerpf(_anchor_scale_ema,
			clampf(sqrt(maxf(1e-4, macc2 / msum - (macc / msum).length_squared())), 0.05, 0.9), 0.1)
	if have_prev:
		_update_wave_impulses(motion / (48.0 * 27.0))


## Model a face as a landmark FRAME - centroid + isotropic size (RMS radius) - from
## the key-colour mass in an already-48x27 frame, with the motion centroid+spread
## as the flat-lighting fallback (the same thresholds _update_whisp_anchor uses).
## This is the "simple EMA over key thresholds" model chimera phase-locks to; it
## runs on the imported TRACK frame so the graft is normalised by the OTHER head's
## own frame before being re-fitted onto the main head. Returns
## {c, s, cur_lum, ok}; cur_lum is the caller's next previous-luminance grid.
func _face_frame(img: Image, tdir: Vector3, prev_lum: PackedFloat32Array) -> Dictionary:
	var acc := Vector2.ZERO
	var acc2 := 0.0
	var wsum := 0.0
	var macc := Vector2.ZERO
	var macc2 := 0.0
	var msum := 0.0
	var cur := PackedFloat32Array()
	cur.resize(48 * 27)
	var have_prev := prev_lum.size() == 48 * 27
	# Flat RGBA8 buffer read - see _update_whisp_anchor for why (same per-tick path).
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	for y in 27:
		for x in 48:
			var idx := y * 48 + x
			var base := idx * 4
			var r := float(data[base]) / 255.0
			var g := float(data[base + 1]) / 255.0
			var b := float(data[base + 2]) / 255.0
			var l := 0.299 * r + 0.587 * g + 0.114 * b
			var pos := Vector2((float(x) + 0.5) / 48.0, (float(y) + 0.5) / 27.0)
			var pr := maxf(0.0, (r - l) * tdir.x + (g - l) * tdir.y + (b - l) * tdir.z)
			acc += pos * pr
			acc2 += pos.length_squared() * pr
			wsum += pr
			if have_prev:
				var dm := absf(l - prev_lum[idx])
				macc += pos * dm
				macc2 += pos.length_squared() * dm
				msum += dm
			cur[idx] = l
	if wsum > 0.01:
		var c0 := acc / wsum
		return {"c": c0, "s": clampf(sqrt(maxf(1e-4, acc2 / wsum - c0.length_squared())), 0.05, 0.9),
			"cur_lum": cur, "ok": true}
	elif msum > 0.3:
		var c1 := macc / msum
		return {"c": c1, "s": clampf(sqrt(maxf(1e-4, macc2 / msum - c1.length_squared())), 0.05, 0.9),
			"cur_lum": cur, "ok": true}
	return {"c": Vector2(0.5, 0.5), "s": 0.28, "cur_lum": cur, "ok": false}


## Model the imported track face (chimera's graft source) each capture tick, so its
## frame - centroid + size - can normalise the graft before it's re-fitted onto the
## main head. Keyed by the chimera marker's own colour; nothing to do without one.
func _update_track_frame() -> void:
	if _track_runtime.is_empty():
		return
	var rt: Dictionary = _track_runtime[0]
	if not rt.has("player"):
		return
	var tp: VideoStreamPlayer = rt.player
	if tp == null:
		return
	var ttex := tp.get_video_texture()
	if ttex == null:
		return
	var timg := ttex.get_image()
	if timg == null or timg.is_empty():
		return
	var hue := -1.0
	for m in session.markers:
		if int(m.get("effect_a", 0)) == MaskSession.EFFECT_CHIMERA:
			hue = float(m.get("hue_a", 0.0))
			break
	if hue < 0.0:
		return
	var tc := Color.from_hsv(hue, 1.0, 1.0)
	var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
	var tdir := Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized()
	timg.resize(48, 27, Image.INTERPOLATE_BILINEAR)
	var fr := _face_frame(timg, tdir, _track_prev_lum)
	_track_prev_lum = fr.cur_lum
	if fr.ok:
		_track_anchor_prev = _track_anchor_ema
		_track_anchor_ema = _track_anchor_ema.lerp(fr.c, 0.15)
		_track_scale_prev = _track_scale_ema
		_track_scale_ema = lerpf(_track_scale_ema, float(fr.s), 0.15)


## Onset detection for the wave impulses: motion is this capture's average
## per-pixel luminance jolt (see caller). A steady baseline (_wave_motion_ema)
## and its own deviation (_wave_dev_ema) track each clip's ambient motion level
## adaptively - talking-head footage idles near-still, a real head turn spikes
## several deviations above it - so one fixed threshold doesn't have to guess
## right for every source video. Rate-limited (see _WAVE_COOLDOWN) so a shaky
## run of frames fires one wave, not a pile of overlapping ones.
func _update_wave_impulses(motion: float) -> void:
	var onset := motion - _wave_motion_ema
	var t: float = _player.stream_position
	if onset > _wave_dev_ema * 3.5 and t - _wave_last_time >= _WAVE_COOLDOWN:
		_wave_last_time = t
		if _wave_amp.size() != _WAVE_SLOTS:
			_wave_pos.resize(_WAVE_SLOTS)
			_wave_time.resize(_WAVE_SLOTS)
			_wave_amp.resize(_WAVE_SLOTS)
		_wave_pos[_wave_slot] = _anchor_ema
		_wave_time[_wave_slot] = t
		_wave_amp[_wave_slot] = clampf(onset / maxf(_wave_dev_ema * 6.0, 0.02), 0.35, 1.0)
		_wave_slot = (_wave_slot + 1) % _WAVE_SLOTS
	_wave_motion_ema = lerp(_wave_motion_ema, motion, 0.2)
	_wave_dev_ema = lerp(_wave_dev_ema, absf(onset), 0.2)


# --- undo/redo ---------------------------------------------------------------

## Snapshot markers + the primary clip's trim + every track onto the undo
## stack, called BEFORE a mutation - so the stack always holds "what it looked
## like before this happened". Pass a `key` for edits that repeat rapidly
## during one gesture (a slider drag, a marker or trim/track handle dragged
## along the timeline): a call whose key matches the in-flight gesture just
## extends the coalescing window instead of pushing again, so the whole
## gesture undoes in one Ctrl+Z. Leave `key` empty for one-shot actions
## (add/delete a marker, import/delete a track) - those always open a fresh
## boundary. Trim/track edits are exactly as accident-prone as marker edits -
## see the whole reason this project asked for undo in the first place - so
## they ride the same stack, not a separate one.
func _snapshot() -> Dictionary:
	return {
		"markers": session.markers.duplicate(true),
		"clip_in": session.clip_in, "clip_out": session.clip_out,
		"tracks": session.tracks.duplicate(true),
	}


func _restore_snapshot(snap: Dictionary) -> void:
	session.markers = snap.markers
	session.clip_in = snap.clip_in
	session.clip_out = snap.clip_out
	session.tracks = snap.tracks


## `desc` is what a Ctrl+Z right after this action would revert - shown live
## above the Ramp/Damp buttons (see _refresh_history_label) so undo is never a
## blind guess. Coalesced calls (matching `key`) keep the first desc: the whole
## gesture is one undo step, so it should read as the one action it is (e.g.
## "adjusted Contrast", not the field's very last no-op tick).
func _push_undo(key: String = "", desc: String = "") -> void:
	if key != "" and key == _undo_coalesce_key and _undo_coalesce_cooldown > 0.0:
		_undo_coalesce_cooldown = _UNDO_COALESCE_WINDOW
		return
	_undo_coalesce_key = key
	_undo_coalesce_cooldown = _UNDO_COALESCE_WINDOW
	_undo_stack.append(_snapshot())
	_undo_descs.append(desc)
	if _undo_stack.size() > _UNDO_LIMIT:
		_undo_stack.pop_front()
		_undo_descs.pop_front()
	_redo_stack.clear()
	_redo_descs.clear()
	_refresh_history_label()


func _undo() -> void:
	if _undo_stack.is_empty():
		return
	_redo_stack.append(_snapshot())
	_restore_snapshot(_undo_stack.pop_back())
	_redo_descs.append(_undo_descs.pop_back())
	_undo_coalesce_key = ""   # the next edit must open its own fresh boundary
	_after_history_restore()


func _redo() -> void:
	if _redo_stack.is_empty():
		return
	_undo_stack.append(_snapshot())
	_restore_snapshot(_redo_stack.pop_back())
	_undo_descs.append(_redo_descs.pop_back())
	_undo_coalesce_key = ""
	_after_history_restore()


## The live preview above the Ramp/Damp buttons - see feedback/0019: "if a user
## uses undo, they have a little preview of what they would be reverting."
## _undo_descs.back() is the description passed to the _push_undo() call that
## opened the CURRENT undo step, i.e. exactly the action Ctrl+Z would revert.
func _refresh_history_label() -> void:
	if _history_label == null:
		return
	if _undo_descs.is_empty():
		_history_label.text = "Undo: nothing yet"
		_history_label.tooltip_text = ""
	else:
		var desc: String = _undo_descs.back()
		_history_label.text = "Undo: " + desc
		_history_label.tooltip_text = "Ctrl+Z would revert: " + desc


## _selected points INTO the array a restore just replaced wholesale, so it's
## dangling - re-resolve it by time in the restored array (or drop the
## selection if that marker no longer exists there) before refreshing anything
## that reads it. Tracks went through the same wholesale replacement - the
## runtime players (see _track_runtime) need reconciling against whatever
## session.tracks now holds, same as _delete_track/_finish_track_import do.
func _after_history_restore() -> void:
	if _selected != null:
		var t: float = float(_selected.get("time", -1.0))
		_selected = null
		for m in session.markers:
			if absf(float(m.time) - t) < 0.0005:
				_selected = m
				break
	_select_generation += 1
	_timeline.selected = _selected
	_reconcile_track_runtime()
	_refresh_lanes()
	_refresh_marker_list()
	_refresh_panel()
	_refresh_history_label()
	_mark_dirty()


## After undo/redo swaps session.tracks wholesale, the live VideoStreamPlayers
## in _track_runtime (built incrementally by _build_track_view) no longer
## necessarily match it 1:1 - a track undo/redo added or removed can leave too
## many or too few. Rebuild runtime state to match: free anything past the
## restored count, (re)build anything missing. Existing entries are trusted
## as-is even if that track's trim/offset changed - _sync_tracks reseeks on
## the next frame regardless, same as any ordinary drag.
func _reconcile_track_runtime() -> void:
	if _composition_parent == null:
		return   # render_mode / no editor UI - tracks aren't interactive there anyway
	while _track_runtime.size() > session.tracks.size():
		var rt: Dictionary = _track_runtime.pop_back()
		if rt.has("player"):
			(rt.player as Node).queue_free()
		if rt.has("wrap"):
			(rt.wrap as Node).queue_free()
		if rt.has("audio") and rt.audio != null:
			(rt.audio as Node).queue_free()
	for i in session.tracks.size():
		if i >= _track_runtime.size():
			_build_track_view(i)


# --- auto-save --------------------------------------------------------------

func _mark_dirty() -> void:
	_dirty = true
	_autosave_cooldown = _AUTOSAVE_DELAY


func _save_session() -> void:
	if session == null or _session_path.is_empty():
		return
	# Once a restart has committed to quitting, _player's playback is no longer a
	# trustworthy live read - the engine's own shutdown sequence stops it before
	# _exit_tree runs, which reports stream_position back as 0. _restart_now already
	# took the definitive capture right before quitting; exit's own catch-all save
	# (below) must not re-derive playhead from a player that may already be torn
	# down, or it silently overwrites the correct saved value with 0 (this was why
	# a restart always landed back at the start of the timeline).
	if _player != null and not _restarting:
		session.playhead = _player.stream_position   # persist where the playhead is
	if _tview != null:
		session.timeline_zoom = _tview.zoom
		session.timeline_view_start = _tview.view_start
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))
	_dirty = false


## Reload the app to pick up code the assistant just edited, landing back on THIS
## session at the current playhead (both persisted, see _save_session). Standalone
## Godot can't hot-reload GDScript, so this is a clean restart-and-restore.
##
## But a restart mid-flight would corrupt any OTHER assistant run still writing files,
## so it never fires while runs are in progress: it hands the restart to the Assistant,
## which holds it until every agent has returned, THEN restarts (see Assistant.reload_when_idle).
##
## Same deference applies to an export in progress (_render_state != "idle", see
## _poll_render): quitting to restart kills the render/transcode subprocess it's waiting
## on, losing the render (feedback/0027). Checked first, and re-checked by _poll_render
## once the export finishes, since one can start after this was first requested.
func _reload_requested() -> void:
	if _render_state != "idle":
		_reload_after_export = true
		_set_status("⟳  Reload queued - restarting once the export in progress finishes")
		return
	var a := get_tree().get_first_node_in_group("assistant")
	if a != null and a.has_method("reload_when_idle") and bool(a.call("is_busy")):
		a.call("reload_when_idle", Callable(self, "_do_restart"))
		_set_status("⟳  Reload queued - restarting once assistant runs finish")
	else:
		_do_restart()


## The actual restart: relaunch this same executable straight back into --mask-edit on
## the current session (so it reopens here, at the saved playhead), preserving whatever
## engine args (--path etc.) it was launched with.
func _do_restart() -> void:
	_save_session()   # captures the playhead
	if _reload_check_pid > 0:
		return   # a check is already running - don't stack another
	# NEVER restart into code that doesn't compile: the assistant's edit might have a
	# syntax error, and relaunching into it would leave the app unable to open. Validate
	# headless first (same check as scripts/scratchpad.py compile); _process reads the
	# result and only then restarts, or reports the errors and stays put.
	var exe := OS.get_executable_path()
	var proj := ProjectSettings.globalize_path("res://")
	_reload_check_log = ProjectSettings.globalize_path("user://reload_compile_check.log")
	var script := "\"%s\" --headless --path \"%s\" --editor --quit > \"%s\" 2>&1" % [exe, proj, _reload_check_log]
	_reload_check_pid = OS.create_process("/bin/bash", ["-c", script])
	if _reload_check_pid <= 0:
		_set_status("⚠  Couldn't run the pre-reload compile check - NOT reloading (edits left as-is)")
		_reload_check_pid = -1
		return
	_set_status("⟳  Checking the edits compile before reloading…")


## The actual restart, run only once _do_restart's compile check comes back clean:
## relaunch this same executable straight back into --mask-edit on the current session
## (so it reopens here, at the saved playhead), preserving the engine args it had.
func _restart_now() -> void:
	# One last accurate capture right before quitting - the compile check the app just
	# ran can take a while, and the app stays fully interactive while it does, so the
	# playhead _do_restart captured when the check STARTED may already be stale. This
	# is also the last point _player is guaranteed to still report a live position -
	# see _save_session/_exit_tree for why nothing may re-derive it after this.
	_save_session()
	_restarting = true
	var engine_args := PackedStringArray()
	for a in OS.get_cmdline_args():
		if a == "--":
			break                     # everything before the user-args separator
		engine_args.append(a)
	engine_args.append("--")
	engine_args.append("--mask-edit")
	engine_args.append(_session_path)
	OS.set_restart_on_exit(true, engine_args)
	get_tree().quit()


## Poll the pre-reload compile check (see _do_restart). Clean -> restart; errors ->
## block the reload and surface them, so a broken assistant edit never bricks the app.
func _poll_reload_check() -> void:
	if _reload_check_pid <= 0 or OS.is_process_running(_reload_check_pid):
		return
	_reload_check_pid = -1
	var log := FileAccess.get_file_as_string(_reload_check_log) if FileAccess.file_exists(_reload_check_log) else ""
	if not _reload_check_log.is_empty():
		DirAccess.remove_absolute(_reload_check_log)
	var errs := []
	for line in log.split("\n"):
		for marker in ["SCRIPT ERROR", "Parse Error", "Compile Error", "Identifier not found", "Failed to load"]:
			if line.contains(marker):
				errs.append(line.strip_edges())
				break
	if errs.is_empty():
		_restart_now()
	else:
		_set_status("⚠  Reload blocked - the edits don't compile (%d error%s); app left running" % [
			errs.size(), "" if errs.size() == 1 else "s"])
		push_warning("ghost: reload blocked, compile errors:\n" + "\n".join(errs))


## Whatever the debounce hasn't flushed yet lands on disk when the editor goes away -
## closing the window mid-burst never loses the last edit. Always save (not only when
## dirty) so the current playhead is captured even after a pure play/scrub with no edit.
func _exit_tree() -> void:
	_save_session()
	# Join the audio loader if it's still running, or Godot warns about an orphaned
	# thread on close.
	if _audio_thread != null and _audio_thread.is_started():
		_audio_thread.wait_to_finish()
		_audio_thread = null
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _select_marker(m: Dictionary) -> void:
	_selected = m
	_select_generation += 1   # a new marker's edits must never coalesce with the last one's
	_timeline.selected = m
	_refresh_panel()


func _refresh_panel() -> void:
	var m: Dictionary = _selected if _selected != null else MaskSession.DEFAULTS
	_kind.select(int(m.get("kind", 0.0)))
	_color_a.color = Color.from_hsv(float(m.get("hue_a", 0.02)), 0.85, 0.9)
	_hue_a.set_value_no_signal(float(m.get("hue_a", 0.02)))
	_threshold.set_value_no_signal(float(m.get("threshold", 0.24)))
	_feather.set_value_no_signal(float(m.get("feather", 0.12)))
	_sat_floor.set_value_no_signal(float(m.get("sat_floor", 0.18)))
	_effect_a.select(int(m.get("effect_a", 0)))
	_intensity_a.set_value_no_signal(float(m.get("intensity_a", 1.0)))
	_marker_duration.set_value_no_signal(float(m.get("duration", 1.0)))
	_fx_x.set_value_no_signal(float(m.get("fx_x", 0.0)))
	_fx_y.set_value_no_signal(float(m.get("fx_y", 0.0)))
	_fx_scale.set_value_no_signal(float(m.get("fx_scale", 1.0)))
	_fx_density.set_value_no_signal(float(m.get("fx_density", 0.45)))
	_fx_contrast.set_value_no_signal(float(m.get("fx_contrast", 0.5)))
	_fx_speed.set_value_no_signal(float(m.get("fx_speed", 1.0)))
	_fx_lag.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_fx_smooth.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_gust.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_undul.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_coil.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_stick.set_value_no_signal(float(m.get("fx_stick", 0.0)))
	_resonance.set_value_no_signal(float(m.get("resonance", 0.0)))
	_update_effect_controls(int(m.get("effect_a", 0)))
	_refresh_marker_label()


## The control hierarchy in action (MaskSession.EFFECT_CONTROLS): show only the
## option groups the selected effect consumes. Threshold doubles as restore's
## reach - same stored field, relabeled so it says what it does here.
func _update_effect_controls(effect_id: int) -> void:
	var groups: Array = MaskSession.EFFECT_CONTROLS.get(effect_id, [])
	# clear fades out EVERYTHING earlier - it has no target color at all. snow
	# picks its foreground/background split automatically - it has no target
	# color either. arealight lights the whole frame, not a keyed color.
	# meta mirrors the whole workspace - it keys on nothing, so no colour picker.
	var has_color := effect_id != MaskSession.EFFECT_CLEAR and effect_id != MaskSession.EFFECT_SNOW \
		and effect_id != MaskSession.EFFECT_SERPENT and effect_id != MaskSession.EFFECT_AREALIGHT \
		and effect_id != MaskSession.EFFECT_META
	_grp_color.visible = has_color
	_show_field(_hue_a, has_color)   # the Hue row lives in _grp_options now, not under _grp_color
	_show_field(_threshold, groups.has("keying") or groups.has("reach"), _threshold_label)
	if groups.has("reach"):
		_threshold_label.text = "Reach"
		_threshold_label.tooltip_text = "How wide around the picked color this restore reaches"
	else:
		_threshold_label.text = "Threshold"
		_threshold_label.tooltip_text = "How far a pixel's hue may drift from the key color and still be masked"
	_threshold.tooltip_text = _threshold_label.tooltip_text
	_show_field(_feather, groups.has("keying"))
	_show_field(_sat_floor, groups.has("keying"))
	# Prune the individual pattern knobs this effect never reads ("only show
	# properties that can be used"). Effects absent from PATTERN_KNOBS show them
	# all (the default); listed ones show only their subset.
	var has_pattern := groups.has("pattern")
	var knobs: Array = MaskSession.PATTERN_KNOBS.get(effect_id, MaskSession.PATTERN_KNOBS_ALL) \
		if has_pattern else []
	_show_field(_fx_scale, has_pattern and knobs.has("scale"))
	_show_field(_fx_x, has_pattern and knobs.has("pan"), _fx_x_label)
	_show_field(_fx_y, has_pattern and knobs.has("pan"), _fx_y_label)
	_show_field(_fx_density, has_pattern and knobs.has("coverage"), _fx_density_label)
	_show_field(_fx_contrast, has_pattern and knobs.has("contrast"), _fx_contrast_label)
	_show_field(_fx_speed, has_pattern and knobs.has("velocity"))
	_show_field(_resonance, has_pattern and knobs.has("resonance"))
	_show_field(_fx_lag, groups.has("echo"), _fx_lag_label)
	_show_field(_fx_smooth, groups.has("echo"))
	_show_field(_gust, groups.has("snow"))
	_show_field(_undul, groups.has("fur"))
	_show_field(_coil, groups.has("fur"))
	_show_field(_stick, groups.has("fur"))
	var is_oracle := effect_id == MaskSession.EFFECT_ORACLE
	_fx_lag_label.text = "Lead (s)" if is_oracle else "Lag (s)"
	_fx_lag_label.tooltip_text = "How far ahead it leads" if is_oracle else "How the past is worn"
	_fx_lag.tooltip_text = _fx_lag_label.tooltip_text
	var is_snow := effect_id == MaskSession.EFFECT_SNOW
	var is_arealight := effect_id == MaskSession.EFFECT_AREALIGHT
	if is_snow:
		_fx_contrast_label.text = "Sensitivity"
		_fx_contrast_label.tooltip_text = "How far snow's fall reaches toward the subject"
	elif is_arealight:
		_fx_contrast_label.text = "Envelope"
		_fx_contrast_label.tooltip_text = "Where along the rig's mood this sits - warm, soft, " + \
			"single-source practical at 0, toward cold, hard, full-spectrum multi-source at 1"
	else:
		_fx_contrast_label.text = "Contrast"
		_fx_contrast_label.tooltip_text = "Edge hardness of the pattern - 0.5 is neutral"
	_fx_contrast.tooltip_text = _fx_contrast_label.tooltip_text
	_fx_x_label.text = "Wind X" if is_snow else "Pan X"
	_fx_x_label.tooltip_text = "Fall direction - horizontal component" \
		if is_snow else "Shifts the pattern horizontally over the frame"
	_fx_x.tooltip_text = _fx_x_label.tooltip_text
	_fx_y_label.text = "Wind Y" if is_snow else "Pan Y"
	_fx_y_label.tooltip_text = "Fall direction - vertical component" \
		if is_snow else "Shifts the pattern vertically over the frame"
	_fx_y.tooltip_text = _fx_y_label.tooltip_text
	var is_crystal := effect_id == MaskSession.EFFECT_CRYSTAL
	_fx_density_label.text = "Stickiness" if is_crystal else "Coverage"
	_fx_density_label.tooltip_text = "Pull toward the tracked face's edges" \
		if is_crystal else "How much of the keyed region the pattern consumes - 0 untouched, 1 fully devoured"
	_fx_density.tooltip_text = _fx_density_label.tooltip_text
	# "Strength" means something different for the two subtractive effects - the
	# ambiguity the feedback flagged (an unlabeled, unexplained "Intensity").
	if effect_id == MaskSession.EFFECT_RESTORE:
		_intensity_label.tooltip_text = "How completely this restore fades out earlier layers on this color"
	elif effect_id == MaskSession.EFFECT_CLEAR:
		_intensity_label.tooltip_text = "How completely this clears every earlier layer"
	else:
		_intensity_label.tooltip_text = "How strongly this layer's effect applies"
	_intensity_a.tooltip_text = _intensity_label.tooltip_text
	_apply_sort()   # visibility/labels just changed - re-rank the now-current set (see _apply_sort)


func _refresh_marker_label() -> void:
	if _selected == null:
		_marker_label.text = "nothing selected - editing plants a ramp here"
	else:
		var t := float(_selected.time)
		var kind_name: String = MaskSession.MARKER_KINDS[int(_selected.get("kind", 0.0))]
		_marker_label.text = "%s @ %s  (%d total)" % \
			[kind_name.capitalize(), MaskTimeline.format_time(t), session.markers.size()]
	_refresh_marker_list()


## The sequential ramp/damp list, pinned to the panel's bottom. Rebuilt wholesale -
## cheap at the marker counts a session actually has, and simpler than diffing.
## Piggybacks on _refresh_marker_label's call sites (add/delete/select/drag all
## already call it) rather than needing its own scattered call sites.
func _refresh_marker_list() -> void:
	if _marker_list == null:
		return
	for c in _marker_list.get_children():
		c.queue_free()
	for m in session.markers:
		var kind_name: String = MaskSession.MARKER_KINDS[int(m.get("kind", 0.0))]
		var b := Button.new()
		b.focus_mode = Control.FOCUS_NONE
		b.alignment = HORIZONTAL_ALIGNMENT_LEFT
		var eff_name: String = MaskSession.MASK_EFFECTS[int(m.get("effect_a", 0))]
		var view_tag := "  · raw ⟲" if int(m.get("view_mode", 2.0)) == 2 else ""
		b.text = "%s   %s · %s%s" % [MaskTimeline.format_time(float(m.time)),
			kind_name.capitalize(), eff_name, view_tag]
		if _selected != null and _selected == m:
			b.add_theme_color_override("font_color", Color(1.0, 0.85, 0.5))
		b.pressed.connect(func(): _select_marker(m))
		_marker_list.add_child(b)


func _on_scrub(t: float) -> void:
	_player.stream_position = t
	_audio.seek(t)


func _play(on: bool) -> void:
	_playing = on
	if not _player.is_playing():
		_player.play()
	if not _audio.playing:
		_audio.play(_player.stream_position)
	_player.paused = not on
	_audio.stream_paused = not on
	_audio_holding = false   # any pending catch-up hold is moot once playback state changes
	# THE VIDEO IS THE MASTER CLOCK. The two players don't pause at the same
	# instant (video stops on a decoded-frame boundary, audio on a mix chunk),
	# so every pause/resume cycle - spacebar, the feedback console's freeze -
	# banked a little offset, and nothing ever corrected it. Snap audio to the
	# video on every resume; _process keeps them corrected from there.
	if on:
		_audio.seek(_player.stream_position)
	# Track audios are driven entirely by _sync_tracks (windowed play/seek/pause), so
	# nothing to do for them here.


## Worker-thread body: the blocking WAV read (see _ready_with_session). Returns the
## stream; the main thread attaches it in _process the frame the thread finishes.
func _load_wav_threaded(path: String) -> AudioStreamWAV:
	return AudioStreamWAV.load_from_file(path)


## Attach the threaded main-audio load the frame it's ready, started at the live
## position so it's in sync from its first sample. No-op until the thread finishes.
func _poll_audio_thread() -> void:
	if _audio_thread == null or _audio_thread.is_alive():
		return
	var stream = _audio_thread.wait_to_finish()
	_audio_thread = null
	if stream != null and _audio != null:
		_audio.stream = stream
		_apply_main_volume()
	# The autostart was waiting for exactly this: begin playback now, synced at the
	# current (start) position, so the intro plays with its audio from the first
	# sample instead of skipping. _play(true) seeks the audio to the video's position,
	# so it stays in sync. If the load FAILED (stream null) start anyway - a silent
	# editor beats one frozen forever waiting on audio that will never arrive.
	if _autostart_pending:
		_autostart_pending = false
		_play(true)
		return
	if stream != null and _audio != null and _playing:
		_audio.play(_player.stream_position if _player != null else 0.0)
		_audio.stream_paused = false


## The main clip's 0/1 audio toggle -> the AudioStreamPlayer's level. -80 dB reads as
## silence; 0 dB is unity. Per-frame the fade overrides this (see _apply_main_fade);
## this covers the moment the toggle is flipped and the initial attach.
func _apply_main_volume() -> void:
	if _audio != null:
		_audio.volume_db = _track_level_db(session.main_volume, 1.0)


## The clip-fade fraction (0 at a faded edge, 1 across the flat middle) for a position
## `local` seconds into a clip of length `span`, given fade_in / fade_out durations.
func _clip_fade_gain(local: float, span: float, fi: float, fo: float) -> float:
	var g := 1.0
	if fi > 0.001 and local < fi:
		g = clampf(local / fi, 0.0, 1.0)
	if fo > 0.001 and local > span - fo:
		g = minf(g, clampf((span - local) / fo, 0.0, 1.0))
	return clampf(g, 0.0, 1.0)


## Exponential audio taper for the pull-rope knob's raw 0..1 reading: pushes the quiet
## end down hard (a real dead zone near the anchor, not just a shallower version of
## "audible") while leaving the top of the pull comparatively spacious, so fine-tuning
## a loud level doesn't blow past it - feedback/0025: "1% volume should be nearly
## inaudible, but isn't"; "the middle volume growth seems very fast". _TAPER_K controls
## how hard the low end is suppressed; the curve is 0 at v=0 and 1 at v=1 regardless.
const _TAPER_K := 5.0
func _volume_taper(v: float) -> float:
	return (exp(_TAPER_K * v) - 1.0) / (exp(_TAPER_K) - 1.0)


## Audio gain in dB for a clip: the pull-rope volume `v` (0..1), exponentially tapered
## (see _volume_taper), times the fade envelope `g` (0..1) gives the linear gain
## fraction. This used to run through lerpf(-40, 0, l) - a shallow floor that left a
## fade idling around a still-audible -40dB for nearly the whole marker-to-marker span,
## then jumped a discontinuous 40dB to silence in the last fraction of a percent
## (feedback/0024: "barely reduces until the very tail end"). An equal-power curve (the
## standard cinematic/DAW fade shape) into a real dB conversion spreads the perceived
## loudness change smoothly across the whole span instead, with no cliff at the end -
## and its own natural flattening near l=1 gives the fine control near the top of the
## pull that feedback/0025 asked for, on top of the taper above.
func _track_level_db(v: float, g: float) -> float:
	var l := _volume_taper(clampf(v, 0.0, 1.0)) * clampf(g, 0.0, 1.0)
	if l < 0.0005:
		return -80.0
	return clampf(linear_to_db(sin(l * PI * 0.5)), -80.0, 0.0)


## The main clip's own fade, applied every frame: the whole composite (video) dims via
## _composition_parent.modulate and the main audio ramps in dB - the same envelope,
## coupled. _composition_parent.modulate alone only reaches the RAW (unshaded) view
## though - mask_split.gdshader's fragment() samples TEXTURE directly rather than
## starting from the built-in COLOR, so the CanvasItem's modulate never reached the
## shaded fx overlay, and with any fx layer active the picture stayed full-opacity no
## matter what the envelope said (feedback/0022). u_fade is the shader's own copy of
## the same `g`, so the fade holds whether or not fx is on screen. Deterministic off
## the playhead, so live and export match.
func _apply_main_fade() -> void:
	if _player == null:
		return
	var cin := session.clip_in
	var cout := session.effective_clip_out()
	var t := _player.stream_position
	# Past the main clip's own kept range, the composite's alpha follows whichever
	# continuation track (see MaskSession.continuation_track_at) actually owns `t` now -
	# its OWN fade_in/fade_out, not main's - since the picture showing there is that
	# track's own independent frame (see _apply_frame_state/_cont_view). No owning
	# track (a gap, or past all content) holds at full opacity, same as before
	# continuation tracks respected their own fade at all.
	var g := 1.0
	if t < cout:
		g = _clip_fade_gain(t - cin, cout - cin, session.main_fade_in, session.main_fade_out)
	else:
		var cont_idx := session.continuation_track_at(t)
		if cont_idx != -1:
			var tr: Dictionary = session.tracks[cont_idx]
			var offset := float(tr.get("offset", 0.0))
			var span := float(tr.get("clip_out", 0.0)) - float(tr.get("clip_in", 0.0))
			g = _clip_fade_gain(t - offset, span, float(tr.get("fade_in", 0.0)), float(tr.get("fade_out", 0.0)))
	if _composition_parent != null:
		_composition_parent.modulate.a = g
	_mat_main.set_shader_parameter("u_fade", g)
	if _audio != null:
		# Audio cuts off exactly at cout, full stop - unlike the picture (see
		# _apply_frame_state), this is NOT extended by a continuation track's window.
		# A continuation track (_split_main's tail, same video_path) already plays its
		# OWN independent audio via _sync_tracks' taudio the moment it's active - gating
		# this on main_visible_at too meant the main clip's audio kept playing right
		# alongside it, doubled with the track's own copy of the same source audio
		# (feedback/0013). Audio ownership passes to the track the moment one exists
		# there - which, after the track's own in-point is re-trimmed independently of
		# cout, can be BEFORE cout (session.track_owns_audio_at) rather than exactly at
		# it. Without this check that brief overlap played main's audio and the track's
		# own copy of the same source at once, right at the handoff (feedback/0014).
		var audible := 0.0 if (t >= cout or session.track_owns_audio_at(t)) else g
		_audio.volume_db = _track_level_db(session.main_volume, audible)


## Master-timeline seconds a dragged clip should snap its start/end to: 0, the playhead,
## the primary clip's end, and every OTHER clip's start and end. exclude_i is the lane
## doing the dragging (-1 = the primary), so a clip never snaps to itself.
func _snap_targets_for(exclude_i: int) -> Array:
	var targets := [0.0, session.effective_clip_out()]
	if _player != null:
		targets.append(_player.stream_position)
	for j in session.tracks.size():
		if j == exclude_i:
			continue
		var t: Dictionary = session.tracks[j]
		var o := float(t.get("offset", 0.0))
		var span := float(t.get("clip_out", 0.0)) - float(t.get("clip_in", 0.0))
		targets.append(o)
		targets.append(o + span)
	return targets


## THE keyboard map (mirrored by the help overlay - keep the two in sync):
## Space play/pause · V cycle view · P hold-to-peek · T import track ·
## Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y undo/redo · F1 help · Esc close help.
## These aren't shortcuts FOR buttons any more - they're the only way; the
## toolbar collapsed to the single Help button. Only live once a clip is
## actually loaded (_player exists); main.gd defers Space entirely while this
## editor is open (see main.gd's KEY_SPACE handling), so this doesn't fight
## Director.next() for the key. echo excluded on undo/redo so a held key
## doesn't spam repeats; a real accident deserves a deliberate press each
## time it's undone.
## Plain _input (not _unhandled_input) so cursor motion resets the idle timer
## even while it's over a panel/button - GUI controls can eat mouse motion
## before it would ever reach _unhandled_input, and hovering the toolbar
## while playing should un-hide the cursor same as moving it over the video.
func _input(event: InputEvent) -> void:
	if render_mode or not event is InputEventMouseMotion:
		return
	_cursor_idle_t = 0.0
	if Input.mouse_mode == Input.MOUSE_MODE_HIDDEN:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _unhandled_input(event: InputEvent) -> void:
	if render_mode or _player == null:
		return
	if not event is InputEventKey or event.echo:
		return
	# Hold-to-peek wants the RELEASE too - everything below this wants presses
	# only. Hold beats the old toggle button for its actual job ("let me just
	# look at raw for a second"): letting go always puts the effects back.
	if event.keycode == KEY_P:
		_peek_raw = event.pressed
		get_viewport().set_input_as_handled()
		return
	if not event.pressed:
		return
	if event.keycode == KEY_SPACE:
		_play(not _playing)
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_V:
		_cycle_view_mode()
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_T:
		_prompt_import_track()
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_F1:
		_toggle_help()
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_F5:
		_reload_requested()   # restart-and-restore, deferred until assistant runs finish
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_ESCAPE and _help_panel != null and _help_panel.visible:
		# Only claim Escape while help is open - otherwise it stays main.gd's
		# quit key. Consuming it here is what keeps closing the overlay from
		# ALSO quitting ghost.
		_help_panel.visible = false
		get_viewport().set_input_as_handled()
	elif event.ctrl_pressed and event.keycode == KEY_Z:
		if event.shift_pressed:
			_redo()
		else:
			_undo()
		get_viewport().set_input_as_handled()
	elif event.ctrl_pressed and event.keycode == KEY_Y:
		_redo()
		get_viewport().set_input_as_handled()


func _toggle_help() -> void:
	if _help_panel != null:
		_help_panel.visible = not _help_panel.visible


## The keyboard map, centered over the video (F1 / the panel's one Help
## button). This is where the buttons went: everything the old toolbar did is
## a key now, and this overlay is how anyone who never learned the shortcuts
## finds them. Keep in sync with _unhandled_input.
func _build_help_overlay() -> void:
	_help_panel = PanelContainer.new()
	_help_panel.visible = false
	_help_panel.z_index = 40
	var hm := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		hm.add_theme_constant_override("margin_" + side, 20)
	_help_panel.add_child(hm)
	var hv := VBoxContainer.new()
	hv.add_theme_constant_override("separation", 7)
	hm.add_child(hv)
	var ht := Label.new()
	ht.text = "Keyboard"
	ht.add_theme_font_size_override("font_size", 18)
	hv.add_child(ht)
	for pair in [
		["Space", "play / pause"],
		["V", "cycle view - raw → PiP (raw) → PiP (fx) → both → full fx"],
		["P (hold)", "peek raw footage - display only, edits nothing"],
		["T", "import a second video track (picture-in-picture)"],
		["Ctrl+Z", "undo"],
		["Ctrl+Shift+Z", "redo (also Ctrl+Y)"],
		["`", "feedback console"],
		["F11", "fullscreen"],
		["F1", "this help (Esc closes it)"],
	]:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 14)
		var kl := Label.new()
		kl.text = pair[0]
		kl.custom_minimum_size = Vector2(130, 0)
		kl.add_theme_color_override("font_color", Color(1.0, 0.85, 0.5))
		row.add_child(kl)
		var dl := Label.new()
		dl.text = pair[1]
		dl.add_theme_color_override("font_color", Color(0.78, 0.84, 0.94))
		row.add_child(dl)
		hv.add_child(row)
	hv.add_child(HSeparator.new())
	for note in [
		"Timeline - click/drag scrubs · drag markers to move them · Ctrl+scroll zooms at the cursor, plain scroll pans when zoomed · drag clip/track edges to trim",
		"Editing - touching any knob plants a ramp at the playhead if nothing is selected · sliders are drag-only, the wheel always scrolls the panel",
	]:
		var nl := Label.new()
		nl.text = note
		nl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		nl.custom_minimum_size = Vector2(560, 0)
		nl.add_theme_font_size_override("font_size", 12)
		nl.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
		hv.add_child(nl)
	# Centered over the VIDEO side, not the whole window (the wrapper starts
	# at PANEL_W so the box doesn't sit half-under the left panel). The
	# wrapper ignores the mouse; the box itself still catches its own clicks.
	var wrap := CenterContainer.new()
	wrap.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	wrap.offset_left = PANEL_W
	wrap.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(wrap)
	wrap.add_child(_help_panel)


# --- view mode: main (raw/fx) x inset (hidden/raw/fx) --------------------------
# A per-marker field (MaskSession.VIEW_MODES), not just an editing preference - see
# class doc. There is deliberately no standalone "current mode" variable: the single
# source of truth is always session.at_time(playhead), read fresh every frame in
# _process (live AND render_mode alike) and applied via _apply_frame_state - which
# consumes the CONTINUOUS per-layer amounts, so mode changes fade across their
# marker's window instead of popping.

## The DISPLAY/cycle order - the "evolution": raw, then the inset appears (still
## raw), then the inset gets the effect, then the main screen joins it, then
## full-frame fx alone. VIEW_MODES' own order is append-only storage layout (see
## MaskSession), so the narrative order lives here.
const VIEW_CYCLE := [2, 3, 0, 5, 4, 1]   # raw -> pip_raw -> pip -> fx+raw-pip -> masked_pip -> masked

## The marker GOVERNING time `t` - the same last-wins search at_time() uses
## internally to resolve its `cur`, exposed here so callers that need the
## actual marker OBJECT (not at_time()'s resolved copy) can find it without
## re-deriving the search. Null before the first marker.
func _governing_marker(t: float) -> Variant:
	if session == null:
		return null
	var cur = null
	for m in session.markers:
		if m.time <= t:
			cur = m
	return cur


## The toggle button edits the marker at the playhead exactly like every other panel
## control (see _edit) - cycling is relative to whatever's ACTIVE right now, not to
## some separately-tracked button state, so it can never drift from the timeline.
## That means retargeting _selected to the playhead's own governing marker FIRST:
## _selected may still be pointing at whatever the marker list was last clicked on,
## which never moves the playhead - editing through the stale selection silently
## changes an unrelated marker elsewhere on the timeline and nothing visibly happens.
func _cycle_view_mode() -> void:
	# The cycle is the base looks, then per imported track a pair of stops showing
	# THAT track raw in the PiP: once with a RAW main (pip_raw, 3), once with an FX
	# main (masked_pip_raw, 5) - so you can see the fx composite beside the raw source.
	# Each stop is [view_mode, pip_track]; the list grows/shrinks with the track count.
	var stops := []
	for vm in VIEW_CYCLE:
		stops.append([vm, 0])
	for k in range(1, session.tracks.size() + 1):
		stops.append([3, k])
		stops.append([5, k])
	var cur_vm := 2
	var cur_pt := 0
	if session != null and _player != null:
		var resolved: Dictionary = session.at_time(_player.stream_position)
		cur_vm = int(resolved.get("view_mode", 2.0))
		cur_pt = int(resolved.get("pip_track", 0.0))
	var idx := -1
	for j in stops.size():
		if int(stops[j][0]) == cur_vm and int(stops[j][1]) == cur_pt:
			idx = j
			break
	var nxt: Array = stops[(idx + 1) % stops.size()] if idx >= 0 else stops[0]
	if session != null and _player != null:
		var t: float = _player.stream_position
		_selected = _governing_marker(t)
	_edit("view_mode", float(int(nxt[0])))   # plants/selects a marker if needed; applied next frame
	if _selected != null:
		_selected["pip_track"] = float(int(nxt[1]))
		_mark_dirty()


## Pushes one resolved timeline state (see MaskSession.at_time) into the layers:
## channel params to both materials, each layer's intensities scaled by its own
## PRESENCE amount - so "how present is this layer" is a continuous, blendable
## quantity and a mode transition is a fade, not a toggle. The inset's border
## fades with it (modulate), and fully-absent layers are hidden entirely so they
## cost nothing.
func _apply_frame_state(p: Dictionary) -> void:
	var main_amt := clampf(float(p.get("main_fx", 0.0)), 0.0, 1.0)
	var inset_show := clampf(float(p.get("inset_show", 0.0)), 0.0, 1.0)
	var inset_fx := clampf(float(p.get("inset_fx", 0.0)), 0.0, 1.0)
	if _peek_raw:
		main_amt = 0.0
		inset_show = 0.0
		inset_fx = 0.0
	_last_inset_show = inset_show
	# Which source fills the PiP this frame. Fall back to the main clip if the stored
	# index points past the current track count (e.g. a track was deleted).
	var pt := int(p.get("pip_track", 0.0))
	_pip_track = 0 if (_peek_raw or pt < 0 or pt > session.tracks.size()) else pt
	var t: float = _player.stream_position if _player != null else 0.0
	var env := _env_at(t)
	var layers: Array = p.get("layers", [])
	# Does a chimera layer actually render this frame? _maybe_capture_echo reads this
	# to decide whether the (expensive) track readback is worth doing right now.
	# _temporal_active is the same idea one level up: gates the echo/whisp capture
	# itself on whatever's actually on screen, not on the session's marker list.
	_chimera_active = false
	_temporal_active = false
	_meta_amount = 0.0
	for l in layers:
		var le := int(l.get("effect_a", 0))
		if le == MaskSession.EFFECT_CHIMERA:
			_chimera_active = true
		if le == 5 or le == 7 or le == MaskSession.EFFECT_SNOW or le == MaskSession.EFFECT_ORACLE \
				or le == MaskSession.EFFECT_SERPENT or le == MaskSession.EFFECT_CHIMERA:
			_temporal_active = true
		# The META mirror's strength - the same env x intensity the shader gets as
		# this layer's weight. Drives whether the (expensive) workspace readback runs
		# at all this frame, and how far the render-mode editor chrome has revealed.
		if le == MaskSession.EFFECT_META:
			_meta_amount = maxf(_meta_amount,
				clampf(float(l.get("env", 0.0)) * float(l.get("intensity_a", 0.0)), 0.0, 1.0))
	# Build the layer arrays ONCE; only the weights differ per surface (each
	# material multiplies its own presence in). Arrays are pushed at FULL declared
	# length - a short uniform array is silently dropped (flame.gdshader lesson).
	var n: int = mini(layers.size(), MaskSession.MAX_LAYERS)
	var hues := PackedFloat32Array()
	var effects := PackedInt32Array()
	var base_w := PackedFloat32Array()
	var offs := PackedVector2Array()
	var scales := PackedFloat32Array()
	var densities := PackedFloat32Array()
	var contrasts := PackedFloat32Array()
	var glows := PackedFloat32Array()
	var speeds := PackedFloat32Array()
	var smooths := PackedFloat32Array()   # raw fx_smooth - snow's Gust; echo bakes its own use into echo_w below
	var tdirs := PackedVector3Array()
	var echo_w := PackedFloat32Array()
	var echo_lag := PackedInt32Array()
	var lagf := PackedFloat32Array()   # raw fx_lag - fur's Coil knob (echo's use is baked into echo_w/echo_lag)
	var sticks := PackedFloat32Array()   # raw fx_stick - fur's Stickiness (0 = today's free coat)
	var slot_frac := fposmod((_player.stream_position if _player != null else 0.0) / _ECHO_INTERVAL, 1.0)
	for i in MaskSession.MAX_LAYERS:
		if i < n:
			var l: Dictionary = layers[i]
			var res := float(l.get("resonance", 0.0))
			hues.append(float(l.get("hue_a", 0.0)))
			effects.append(int(l.get("effect_a", 0)))
			base_w.append(float(l.get("env", 0.0)) * float(l.get("intensity_a", 0.0)))
			offs.append(Vector2(float(l.get("fx_x", 0.0)), float(l.get("fx_y", 0.0))))
			scales.append(float(l.get("fx_scale", 1.0)))
			# Resonance folds in CPU-side: the audio envelope swings coverage around
			# its nominal (loud opens the field, quiet closes it) and pulses the rim.
			densities.append(clampf(float(l.get("fx_density", 0.45)) + 0.5 * res * (env - 0.35), 0.0, 1.0))
			contrasts.append(float(l.get("fx_contrast", 0.5)))
			glows.append(1.0 + res * env * 1.3)
			speeds.append(maxf(0.05, float(l.get("fx_speed", 1.0))))
			smooths.append(clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0))
			# The echo's temporal kernel: weights over the 8 ring ages, centered
			# on the layer's lag. Age of ring index k is (k + slot_frac) slots -
			# continuous in playback time, so a spread kernel (Smoothing > 0)
			# glides through the ring with no steps; Smoothing ~ 0 collapses to
			# the nearest single frame - the held-frame stutter, now at an
			# adjustable distance. Pure function of position + fields: live and
			# export blend identically.
			var lag_slots := clampf(float(l.get("fx_lag", 0.35)) / _ECHO_INTERVAL, 0.0, 7.0)
			var smooth_amt := clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0)
			echo_lag.append(clampi(int(round(lag_slots)), 0, 7))
			lagf.append(float(l.get("fx_lag", 0.35)))
			sticks.append(clampf(float(l.get("fx_stick", 0.0)), 0.0, 1.0))
			var w := PackedFloat32Array()
			var wsum := 0.0
			for k in 8:
				var wv: float
				if smooth_amt < 0.02:
					wv = 1.0 if k == clampi(int(round(lag_slots - slot_frac)), 0, 7) else 0.0
				else:
					wv = exp(-absf(float(k) + slot_frac - lag_slots) / (smooth_amt * 2.5))
				w.append(wv)
				wsum += wv
			for k in 8:
				echo_w.append(w[k] / maxf(wsum, 0.0001))
			# The target hue's normalized chroma direction, for erase's
			# projection-subtraction (see the shader: erase is subtraction,
			# not classification - no gates, no boundary rings).
			var tc := Color.from_hsv(float(l.get("hue_a", 0.0)), 1.0, 1.0)
			var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
			tdirs.append(Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized())
		else:
			hues.append(0.0)
			effects.append(0)
			base_w.append(0.0)
			offs.append(Vector2.ZERO)
			scales.append(1.0)
			densities.append(0.0)
			contrasts.append(0.5)
			speeds.append(1.0)
			smooths.append(0.0)
			echo_lag.append(0)
			lagf.append(0.0)
			sticks.append(0.0)
			for k in 8:
				echo_w.append(1.0 if k == 0 else 0.0)
			glows.append(1.0)
			tdirs.append(Vector3(1, 0, 0))
	# Which source actually has valid picture at `t`: the main clip's own kept range,
	# or - once that's ended - whichever continuation track (see
	# MaskSession.continuation_track_at) has picked it up. Each continuation track
	# renders through its OWN player (see _sync_tracks), never by borrowing _player's -
	# see continuation_track_at's doc for why that used to be a fragile invariant.
	var main_active := t < session.effective_clip_out()
	var cont_idx := -1 if main_active else session.continuation_track_at(t)
	var cont_tex: Texture2D = null
	if cont_idx != -1 and cont_idx < _track_runtime.size() and _track_runtime[cont_idx].has("player"):
		var cp: VideoStreamPlayer = _track_runtime[cont_idx].player
		if cp != null and cp.get_video_texture() != null and cp.get_video_texture().get_height() > 0:
			cont_tex = cp.get_video_texture()
	_cont_view.visible = cont_tex != null
	if cont_tex != null:
		_cont_view.texture = cont_tex
	for pair in [[_mat_main, main_amt], [_mat_inset, inset_fx]]:
		var mat: ShaderMaterial = pair[0]
		var amt: float = pair[1]
		mat.set_shader_parameter("u_threshold", p.threshold)
		mat.set_shader_parameter("u_feather", p.feather)
		mat.set_shader_parameter("u_sat_floor", p.sat_floor)
		# The wisp field's clock is the CLIP's own playback position, never
		# wall-time - live and export step the same clock, so a session
		# reproduces its exact wisps frame-for-frame (flame.gdshader discipline).
		mat.set_shader_parameter("u_time", t)
		var tex := (_player.get_video_texture() if _player != null else null) if main_active else cont_tex
		if tex != null and tex.get_height() > 0:
			mat.set_shader_parameter("u_aspect", float(tex.get_width()) / float(tex.get_height()))
		var ws := PackedFloat32Array()
		for i in MaskSession.MAX_LAYERS:
			ws.append(base_w[i] * amt)
		mat.set_shader_parameter("u_l_count", n)
		mat.set_shader_parameter("u_l_hue", hues)
		mat.set_shader_parameter("u_l_effect", effects)
		mat.set_shader_parameter("u_l_w", ws)
		mat.set_shader_parameter("u_l_off", offs)
		mat.set_shader_parameter("u_l_scale", scales)
		mat.set_shader_parameter("u_l_dens", densities)
		mat.set_shader_parameter("u_l_con", contrasts)
		mat.set_shader_parameter("u_l_glow", glows)
		mat.set_shader_parameter("u_l_speed", speeds)
		mat.set_shader_parameter("u_l_smooth", smooths)
		mat.set_shader_parameter("u_l_ew", echo_w)
		mat.set_shader_parameter("u_l_elag", echo_lag)
		mat.set_shader_parameter("u_l_lagf", lagf)
		mat.set_shader_parameter("u_l_stick", sticks)
		mat.set_shader_parameter("u_l_tdir", tdirs)
		# Chimera's graft source: the first track's live frame. The explicit
		# flag matters - the sampler's default-black fallback must never read
		# as footage.
		var track_tex: Texture2D = null
		if _track_runtime.size() > 0 and _track_runtime[0].has("player"):
			var tp: VideoStreamPlayer = _track_runtime[0].player
			if tp != null and tp.get_video_texture() != null \
					and tp.get_video_texture().get_height() > 0:
				track_tex = tp.get_video_texture()
		mat.set_shader_parameter("u_track_on", 1 if track_tex != null else 0)
		if track_tex != null:
			mat.set_shader_parameter("u_track", track_tex)
	# The active source's own raw frame - and everything sourced from it (the fx
	# overlay, its own PiP inset) - only while the timeline actually claims this
	# instant (feedback/0009: past the main track's own trim, with no track
	# continuing it here, video.ogv kept rendering anyway).
	_player.visible = main_active
	_fx_overlay.visible = main_amt > 0.001 and (main_active or cont_tex != null)
	# Main clip's PiP only when it's the selected source; _sync_tracks re-confirms.
	_mask_wrap.visible = inset_show > 0.001 and _pip_track == 0 and main_active
	_mask_wrap.modulate.a = inset_show
	if _view_label != null:
		if _pip_track > 0:
			var main_lbl := "fx" if main_amt > 0.5 else "raw"
			_view_label.text = "🎞  main %s · Track %d (raw)" % [main_lbl, _pip_track]
		else:
			match MaskSession.VIEW_MODES[clampi(int(p.get("view_mode", 2.0)), 0, MaskSession.VIEW_MODES.size() - 1)]:
				"raw":            _view_label.text = "🎬  Raw"
				"pip_raw":        _view_label.text = "🖼  PiP (raw)"
				"pip":            _view_label.text = "🖼  PiP (fx)"
				"masked_pip_raw": _view_label.text = "🎭  Full (fx) · PiP (raw)"
				"masked_pip":     _view_label.text = "🎭  Both (fx)"
				"masked":         _view_label.text = "🎭  Full (fx)"


# --- per-frame: push the timeline's blended params into the shader ---------------

func _process(_dt: float) -> void:
	match _prep_state:
		"prepping_video":
			if OS.is_process_running(_prep_video_pid):
				_set_status("⏳  Preparing clip (video)…  %d%%" % _read_prep_pct())
				return
			_start_prep_audio()
			return
		"prepping_audio":
			if OS.is_process_running(_prep_audio_pid):
				return
			_clear_lock(_pending.dir)
			_prep_state = "idle"
			_finish_session(_pending.source, _pending.video, _pending.audio)
			return
		"waiting_external":
			var abs_video := ProjectSettings.globalize_path(_pending.video)
			var abs_audio := ProjectSettings.globalize_path(_pending.audio)
			if FileAccess.file_exists(abs_video) and FileAccess.file_exists(abs_audio):
				_prep_state = "idle"
				_finish_session(_pending.source, _pending.video, _pending.audio)
				return
			if _prep_looks_live(_pending.dir, _pending.video):
				return   # still going elsewhere - keep waiting
			# Nothing's touched it in a while and it never produced both files - the
			# other writer died mid-prep. Run it ourselves rather than waiting forever.
			_prep_state = "idle"
			_prep(_pending.source, _pending.dir, _pending.video, _pending.audio)
			return
		_:
			pass
	if _waveform_pid > 0 and not OS.is_process_running(_waveform_pid):
		_waveform_pid = -1
		var abs_wave := ProjectSettings.globalize_path(_waveform_path)
		if FileAccess.file_exists(abs_wave):
			_load_waveform(abs_wave)
	if _import_pid > 0 and not OS.is_process_running(_import_pid):
		_import_pid = -1
		_finish_track_import()
	_poll_audio_thread()   # cheap bool check; attaches the main audio once its load finishes
	if not _track_audio_jobs.is_empty():
		_poll_track_audio()
	if _lanes_col != null:
		var visible_lanes := 0
		for c in _lanes_col.get_children():
			if c is Control and (c as Control).visible:
				visible_lanes += 1
		_apply_lane_reserved(visible_lanes)
	if _reload_check_pid > 0:
		_poll_reload_check()   # gates the reload on a clean headless compile
	if session == null or _player == null:
		return
	# THE EXPORT CLOCK. In render mode the movie is driven by accumulated fixed-fps time
	# (_render_t), NOT the video/audio stream positions - those can drift a little, stall,
	# or (when a source is shorter than the session) end early, and binding the export to
	# them truncated the movie to the shorter stream and let effects/video slide out of
	# alignment. render_pos is where the timeline is at this recorded frame; the decoded
	# video normally free-runs within ~2 frames of it (measured), so it's only re-synced
	# on real divergence, never seeked every frame (which would thrash the OGV decoder).
	var render_pos := _player.stream_position
	if render_mode:
		_render_t += _dt
		render_pos = _render_t   # export runs the whole timeline from 0 (see _build_render_view)
		if _player.is_playing() and absf(_player.stream_position - render_pos) > 0.2:
			_player.stream_position = render_pos
	# Restore the persisted playhead once, now that the player has had a frame to start
	# (a same-frame seek in _ready_with_session is ignored). Retried until it takes or a
	# short budget lapses, then the seek + audio sync catch up from there.
	if _pending_restore >= 0.0:
		var target := clampf(_pending_restore, session.clip_in, session.effective_clip_out())
		_player.stream_position = target
		if _audio != null:
			_audio.seek(target)
		if absf(_player.stream_position - target) < 1.0 or _pending_restore_tries > 20:
			_pending_restore = -1.0
		_pending_restore_tries += 1
	# Same resolve, live or exported: whatever the timeline says at this instant is
	# what's shown - render_mode doesn't special-case a fixed "always masked" look. Live
	# reads the (just-restored) video position; export reads the deterministic movie clock.
	_apply_frame_state(session.at_time(render_pos if render_mode else _player.stream_position))
	_apply_main_fade()   # dim the whole composite + main audio at the clip's fade edges
	# The fx overlay re-draws whichever raw source is actually active this frame -
	# _player while the main clip's own kept range covers it, else _cont_view's
	# texture (a continuation track's own independent decode, see _apply_frame_state) -
	# each only while actually on screen; "raw" mode skips both (and the shader
	# passes they'd otherwise cost) entirely.
	if _fx_overlay != null and _fx_overlay.visible:
		_fx_overlay.texture = _player.get_video_texture() if _player.visible else _cont_view.texture
	if _pip_view != null and _mask_wrap.visible:
		_pip_view.texture = _player.get_video_texture()
	_maybe_capture_echo()
	# META: while a meta layer is live, capture the editor's own frame for the mirror
	# and (in export) lerp the editor chrome into view. Both are gated on _meta_amount
	# so the expensive readback only ever runs during an actual meta section.
	if _meta_amount > 0.001:
		_capture_workspace()
	if render_mode:
		_apply_meta_chrome(_meta_amount)
	_push_anchor()
	# Standing A/V drift correction (see _play: video is the master clock).
	# 0.15s tolerance sits above audio mix-chunk granularity so this never
	# chatters. Video ahead of audio: seek audio forward (a silent skip -
	# no artifact). Audio ahead of video: HOLD audio in place (pause, no
	# seek) until video's decode catches back up, instead of seeking it
	# backward - a backward seek replays audio just heard, audible as an
	# echo/glitch (feedback/0012, which is why the backward correction was
	# dropped entirely). But dropping it left the OTHER direction fully
	# uncorrected: _maybe_capture_echo's synchronous GPU readback (whisp/
	# echo/chimera/snow/oracle/serpent) stalls this thread every
	# _ECHO_INTERVAL and freezes _player.stream_position for the stall's
	# duration while _audio keeps flowing on its own thread, so audio comes
	# out ahead after every single capture - uncorrected, that drift only
	# ever grows over a session (feedback/0025). Holding (not seeking) closes
	# that gap without ever replaying already-heard audio.
	# The hold's own exit check has to run OUTSIDE the "_audio.playing" gate:
	# setting stream_paused = true immediately flips .playing to false (that's
	# just what a paused AudioStreamPlayer reports), so gating the exit check
	# on .playing meant a hold could engage but never release - audio stayed
	# silent, permanently, until a manual pause/play cycle called _play()'s own
	# "if not _audio.playing: _audio.play(...)" restart (feedback/0027).
	if _playing and _audio_holding:
		var hold_drift := _player.stream_position - _audio.get_playback_position()
		if hold_drift >= -0.02:
			_audio.stream_paused = false
			_audio_holding = false
	elif _playing and _audio.playing and not _audio.stream_paused:
		var av_drift := _player.stream_position - _audio.get_playback_position()
		if av_drift > 0.15:
			_audio.seek(_player.stream_position)
		elif av_drift < -0.15:
			_audio.stream_paused = true
			_audio_holding = true
	_sync_tracks()
	# A trimmed clip's OUT point is a hard wall for playback (both live preview
	# and the export relaunch - export additionally needs a QUIT, not just a
	# pause, since Movie Maker keeps recording for as long as the process runs).
	# clip_in is not enforced here on purpose: scrubbing earlier to look at
	# trimmed-away footage while editing is fine, only PLAYBACK (and export) are
	# bounded to the kept range. Bounded by content_end(), not clip_out directly -
	# _split_main trims clip_out but appends the trimmed tail as a track at that
	# same offset, so the show must keep running (the master clock IS the main
	# clip's own decode position) until that track's own span is done too.
	var content_stop := session.content_end()
	if render_mode:
		# The export ends on the MOVIE clock reaching the session's own content length -
		# never on a source stream ending. Binding it to the streams truncated the movie
		# to whichever of the audio/video was shorter than the session (the 13:37 -> 13:00
		# report) and, since a stalled/ended source froze the position the effects keyed
		# off, slid the whole show out of alignment. Past a source's own end the timeline
		# simply shows raw for that region, but the movie still runs its full length with
		# the full audio. content_end() already folds in clip_out + any continuation track;
		# the export runs [0, content_end()] - the whole timeline, matching the editor.
		if _render_t >= content_stop - 0.001:
			get_tree().quit()
			return
	elif content_stop < session.duration and _player.stream_position >= content_stop and _playing:
		_play(false)
	if _undo_coalesce_cooldown > 0.0:
		_undo_coalesce_cooldown -= _dt
	if render_mode:
		return
	# Cross a marker during playback and it selects itself - the panel and timeline
	# follow the playhead, so scrubbing through a whole session no longer means
	# clicking every tiny flag by hand. Only while actually playing; a paused scrub
	# or a deliberate click in the marker list is left alone.
	if _playing:
		var governing: Variant = _governing_marker(_player.stream_position)
		if governing != null and governing != _selected:
			_select_marker(governing)
	# Auto-hide the cursor once it's been still for a beat during playback -
	# _input() above resets the timer and un-hides on any motion, and pausing
	# (or a mouse click, which also fires motion-adjacent hover) restores it
	# immediately rather than leaving an editor with a phantom-hidden pointer.
	if _playing:
		_cursor_idle_t += _dt
		if _cursor_idle_t >= _CURSOR_HIDE_DELAY and Input.mouse_mode == Input.MOUSE_MODE_VISIBLE:
			Input.mouse_mode = Input.MOUSE_MODE_HIDDEN
	elif Input.mouse_mode == Input.MOUSE_MODE_HIDDEN:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	if _time_label != null:
		_time_label.text = "%s / %s" % [
			MaskTimeline.format_time(_player.stream_position), MaskTimeline.format_time(session.duration)]
	# Auto-save: any edit marks the session dirty (see _mark_dirty); it lands on
	# disk shortly after the LAST change in a burst - a slider drag saves once,
	# not once per pixel of mouse travel.
	if _dirty:
		_autosave_cooldown -= _dt
		if _autosave_cooldown <= 0.0:
			_save_session()
	_poll_render()


# --- shared bottom-right notification: prep progress AND export progress use the
# --- same label, same position - one status line, whatever phase produced it -----

func _build_status_label() -> void:
	_status = Label.new()
	_status.name = "MaskStatus"
	_status.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_status.offset_left = -548
	_status.offset_top = -64
	_status.offset_right = -116
	_status.offset_bottom = -28
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_status.visible = false
	# Built before the chrome/timeline exist (see the doc comment on the _status
	# var) so prep messages show pre-session; once the chrome IS built, MaskTimeline's
	# near-opaque background (mask_timeline.gd's _draw) is added afterward and, being
	# a later sibling, painted over this label - silently swallowing every status this
	# label ever shows for the rest of the session (export progress included; feedback
	# 0026). z_index keeps it on top regardless of what gets added later.
	_status.z_index = 5
	add_child(_status)


# --- export: relaunch in Movie Maker mode (--mask-render), then ffmpeg mux ------

func _build_export_ui() -> void:
	_export_btn = Button.new()
	_export_btn.focus_mode = Control.FOCUS_NONE
	_export_btn.text = "⤓"                    # icon-only - matches assistant.gd's chat-bubble toggle
	_export_btn.tooltip_text = "Render this mask session to a video file (in the background)"
	_export_btn.custom_minimum_size = Vector2(40, 40)
	_export_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# Same 40x40 box, same row (-28/-68), as assistant.gd's toggle, right of this one -
	# see that file's _TOGGLE_SIZE/_TOGGLE_ROW_BOTTOM doc for why the numbers match.
	_export_btn.offset_left = -112
	_export_btn.offset_top = -68
	_export_btn.offset_right = -72
	_export_btn.offset_bottom = -28
	_export_btn.pressed.connect(_on_export_pressed)
	add_child(_export_btn)

	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.use_native_dialog = true
	_dialog.title = "Export mask video"
	_dialog.filters = PackedStringArray(["*.mp4 ; Video (MP4, H.264 + AAC)"])
	_dialog.current_file = "ghost_mask.mp4"
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_dialog.current_dir = downloads
	_dialog.size = Vector2i(800, 560)
	_dialog.file_selected.connect(_on_export_path)
	add_child(_dialog)


func _on_export_pressed() -> void:
	_dialog.popup_centered()


func _on_export_path(out_path: String) -> void:
	_out = out_path if out_path.get_extension().to_lower() == "mp4" else out_path + ".mp4"
	_avi = _out.get_basename() + ".render.avi"
	if FileAccess.file_exists(_avi):
		DirAccess.remove_absolute(_avi)   # a stale scratch AVI from an interrupted prior export
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))   # the relaunch reads THIS file
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	var args := PackedStringArray([
		"--path", project, "--write-movie", _avi, "--fixed-fps", "25",
		"--", "--mask-render", _session_path])
	_render_pid = OS.create_process(exe, args)
	if _render_pid > 0:
		_render_state = "rendering"
		_set_status("⏺  Rendering…")
	else:
		_set_status("⚠  Could not start the render process")


func _poll_render() -> void:
	match _render_state:
		"rendering":
			if OS.is_process_running(_render_pid):
				return
			if _file_size(_avi) > 65536:
				_repair_avi_sizes(_avi)   # Godot's 32-bit AVI sizes wrap past 4 GiB; fix before transcode
				_start_transcode()
			else:
				_set_status("⚠  Render produced no file (see console)")
				_render_state = "idle"
		"transcoding":
			if OS.is_process_running(_transcode_pid):
				return
			# Always clear the scratch AVI (the transcode's own `&& rm` usually already
			# did, but not if it failed or was interrupted) - never leave an intermediate
			# behind. remove_absolute is a harmless no-op when it's already gone.
			if _file_size(_out) > 4096:
				DirAccess.remove_absolute(_avi)
				_set_status("✓  Saved  " + _out)
			else:
				DirAccess.remove_absolute(_avi)
				_set_status("⚠  Transcode failed (see console)")
			_render_state = "idle"
	# A reload asked for while this export was in flight (see _reload_requested) is
	# held here until the export just went idle, then re-asked - re-checking the
	# assistant's own busy state fresh rather than assuming it's still the same.
	if _render_state == "idle" and _reload_after_export:
		_reload_after_export = false
		_reload_requested()


func _start_transcode() -> void:
	_set_status("⏳  Finalizing…")
	# `-fflags +genpts` re-derives timestamps so a damaged/wrapped AVI index (see
	# _repair_avi_sizes) is bypassed instead of trusted - without it a >4 GiB render
	# transcodes to a broken file or fails outright, leaving the raw .render.avi behind.
	# `-pix_fmt yuv420p` keeps the MP4 playable everywhere (VLC/QuickTime/browsers).
	# Run through bash so the scratch AVI is deleted BY THE TRANSCODE ITSELF the moment
	# it succeeds (`&& rm`), not by a _poll_render tick that never comes if the editor is
	# closed while ffmpeg (a child that outlives it) is still finalizing - which is how
	# the orphaned .render.avi got left "alongside the final version". Paths are passed as
	# $1/$2, never interpolated, so spaces/quotes in the export path are safe.
	var script := "ffmpeg -y -loglevel error -fflags +genpts -i \"$1\" " \
		+ "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k \"$2\" " \
		+ "&& rm -f \"$1\""
	_transcode_pid = OS.create_process("/bin/bash", PackedStringArray(["-c", script, "bash", _avi, _out]))
	_render_state = "transcoding"


# Godot's AVI writer keeps 32-bit RIFF/LIST size fields, and a full-resolution mask
# render crosses 4 GiB in a few minutes - past that the written sizes WRAP (mod 2^32)
# and the container lies about where the frame data ends, even though every 00db/01wb
# chunk after it is written correctly to EOF. Demuxers that trust those fields (and
# players, whose seeks hit the equally-wrapped idx1 offsets) stall, repeat frames, or
# break time indexing (this is why VLC's scrub bar goes wrong on the raw AVI). The
# repair is two words: RIFF size and the movi LIST size become 0 - "size unknown, read
# to end of file" - turning any demux into a clean sequential walk of the intact chunks.
# No-op for files under 4 GiB (their sizes are already correct). Mirrors exporter.gd.
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
			print("ghost mask export: repaired wrapped >4GiB AVI sizes in ", path.get_file())
			break
		if csize <= 0 or tag.is_empty():
			break
		pos += 8 + csize + (csize & 1)
	f.close()


func _file_size(path: String) -> int:
	if not FileAccess.file_exists(path):
		return 0
	var fa := FileAccess.open(path, FileAccess.READ)
	return fa.get_length() if fa != null else 0


func _set_status(text: String) -> void:
	if _status == null:     # render_mode never builds it - status there is stdout only
		print("ghost mask: ", text)
		return
	_status.text = text
	_status.visible = true
