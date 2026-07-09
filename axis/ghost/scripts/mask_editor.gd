extends CanvasLayer
class_name MaskEditor

## MaskEditor - the mask-mode authoring surface.
##
## Load a clip, key two colors apart (per-pixel hue classification, see
## shaders/mask_split.gdshader), place MARKERS where the split/effect should
## change, scrub the timeline, export. See scripts/mask_session.gd for the data
## model - a session's markers are fixed-schema scalar vectors, not free-form
## params, and every one is either a RAMP (eases in before its anchor) or a DECAY
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

## Set by main.gd before open_source() for the --mask-render relaunch: skip the
## editing panel, autoplay from t=0, quit when the audio finishes.
var render_mode := false

var session: MaskSession = null
var _session_path := ""       # res://-relative or absolute; wherever it was loaded from

var _player: VideoStreamPlayer     # always the RAW decode - never carries the shader
var _audio: AudioStreamPlayer
# One material PER LAYER: the main overlay and the inset can be mid-transition at
# different presences (e.g. fx-inset -> both: the inset holds full while the main
# overlay fades in), and a layer's presence multiplies into its own intensities -
# impossible with one shared material.
var _mat_main := ShaderMaterial.new()
var _mat_inset := ShaderMaterial.new()
var _playing := false
var _cursor_idle_t := 0.0
const _CURSOR_HIDE_DELAY := 1.5   # seconds of stillness during playback before the mouse cursor hides

var _fx_overlay: TextureRect       # full-frame fx layer - _player's texture, shaded
var _pip_view: TextureRect         # the inset's content - shaded or raw per view mode
var _mask_wrap: PanelContainer     # the inset's border/placement box (holds _pip_view)
var _view_btn: Button
var _peek_btn: Button
var _peek_raw := false     # DISPLAY-ONLY raw override; never touches session data

var _timeline: MaskTimeline
var _tview: TimelineView          # shared pixel<->time mapping - see timeline_view.gd
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
var _import_pending := {}   # {source, dir, video} mid-transcode
var _selected: Variant = null   # the marker Dictionary currently shown in the panel

var _color_a: ColorPickerButton
var _threshold: HSlider
var _threshold_label: Label
var _grp_color: VBoxContainer       # the control hierarchy's option groups -
var _grp_threshold: VBoxContainer   # shown per selected effect, see
var _grp_keymisc: VBoxContainer     # MaskSession.EFFECT_CONTROLS
var _grp_pattern: VBoxContainer
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
var _fx_smooth: HSlider
var _grp_echo: VBoxContainer
var _echo_header: Label             # swaps text when oracle borrows the group
var _grp_snow: VBoxContainer
var _gust: HSlider   # snow's own Gust slider - a second, independent view onto fx_smooth (see _grp_echo)
var _grp_fur: VBoxContainer
var _undul: HSlider  # fur's Undulation - fur's view onto fx_smooth (same stored-field reuse as _gust)
var _coil: HSlider   # fur's Coil - fur's view onto fx_lag (pushed raw as u_l_lagf; echo bakes its lag into u_l_ew)
var _resonance: HSlider
var _effect_a: OptionButton
var _intensity_a: HSlider
var _kind: OptionButton     # ramp / decay - see MaskSession.MARKER_KINDS
var _marker_duration: HSlider
var _marker_label: Label
var _time_label: Label
var _marker_list: VBoxContainer   # sequential ramp/decay list, pinned to the panel's bottom

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
# Only runs at all while the session actually contains a whisp/echo marker.
var _echo_ring: Array = [null, null, null, null, null, null, null, null]   # ImageTexture, slot-indexed
var _echo_slot := -1
# The whisp anchor is double-buffered: _anchor_ema is the EMA at the LATEST
# capture, _anchor_prev the one before. The uniform pushed per frame lerps
# between them by position-within-slot (see _push_anchor) - pushing the EMA
# directly stepped the whole pattern once per capture, a visible jump amplified
# by pattern zoom ("jittery, it resets, it jumps"). Position-keyed, so live and
# export trace the identical glide.
var _anchor_prev := Vector2(0.5, 0.5)
var _anchor_ema := Vector2(0.5, 0.5)
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


func _probe_duration(path: String) -> float:
	var out := []
	OS.execute("ffprobe", ["-v", "error", "-show_entries", "format=duration",
		"-of", "csv=p=0", path], out)
	if out.size() > 0:
		var s := String(out[0]).strip_edges()
		if s.is_valid_float():
			return float(s)
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
	# Plain load() has no loader for a raw .wav outside the import pipeline (unlike
	# VideoStreamTheora, which does resolve on an absolute path); AudioStreamWAV's
	# static loader is the runtime-safe path for both cases.
	_audio.stream = AudioStreamWAV.load_from_file(ProjectSettings.globalize_path(session.audio_path))
	add_child(_audio)

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
	_play(true)


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


## Three stacked layers in `parent`'s full rect, shared by both the live editor and
## the render_mode export so they composite identically:
##   _player      raw video, always visible underneath everything
##   _fx_overlay  full-frame shaded copy (its own material) - the MAIN fx layer
##   _mask_wrap   the bordered inset holding _pip_view (its own material)
## Which layers show, and how strongly, comes from the per-frame AMOUNTS
## (MaskSession.mode_amounts, blended through ramp/decay windows by at_time) -
## applied every frame in _apply_frame_state. A layer's presence multiplies into
## its own material's intensities, which is why each has its own material: the
## inset can hold full fx while the main overlay is still fading in.
func _build_video_composition(parent: Control) -> void:
	_composition_parent = parent   # secondary tracks' PiP views land here too - see _build_track_view
	parent.add_child(_player)
	_player.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

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
## not built for a second independent source yet) picture-in-picture box,
## positioned/sized from the track's own x/y/w/h (normalized 0..1, top-left +
## size) - draggable in a later pass; a sensible default corner for now. Drawn
## on TOP of the primary composition (added after it), below nothing - PiP
## always rides over the main picture.
func _build_track_view(i: int) -> void:
	var track: Dictionary = session.tracks[i]
	var player := VideoStreamPlayer.new()
	player.stream = load(ProjectSettings.globalize_path(String(track.video_path)))
	player.expand = true
	_composition_parent.add_child(player)

	var wrap := Control.new()
	wrap.anchor_left = float(track.get("x", 0.68))
	wrap.anchor_top = float(track.get("y", 0.04))
	wrap.anchor_right = float(track.get("x", 0.68)) + float(track.get("w", 0.28))
	wrap.anchor_bottom = float(track.get("y", 0.04)) + float(track.get("h", 0.28))
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

	while _track_runtime.size() <= i:
		_track_runtime.append({})
	_track_runtime[i] = {"player": player, "view": view, "wrap": wrap, "active": false}
	player.paused = true
	player.play()   # must be playing before .stream_position can be set (see _sync_tracks)


# --- multi-track: trim lanes, import, playback sync ----------------------------

const _LANE_H := 26.0

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
	primary.label = "Clip"
	primary.color = Color(0.55, 0.75, 1.0)
	primary.movable = false
	primary.full_duration = session.duration
	primary.get_in = func(): return session.clip_in
	primary.set_in = func(v): session.clip_in = v
	primary.get_out = func(): return session.effective_clip_out()
	primary.set_out = func(v): session.clip_out = v
	primary.drag_started.connect(func(): _push_undo())
	primary.drag_ended.connect(func(): _tview.refresh(session))
	primary.changed.connect(_mark_dirty)
	_lanes_col.add_child(primary)

	for i in session.tracks.size():
		var lane := TrackLane.new()
		lane.tview = _tview
		lane.label = String(session.tracks[i].get("video_path", "")).get_file()
		lane.color = Color(0.6, 0.95, 0.7)
		lane.movable = true
		lane.full_duration = float(session.tracks[i].get("duration", 0.0))
		lane.get_in = _track_getter(i, "clip_in")
		lane.set_in = _track_setter(i, "clip_in")
		lane.get_out = _track_getter(i, "clip_out")
		lane.set_out = _track_setter(i, "clip_out")
		lane.get_offset = _track_getter(i, "offset")
		lane.set_offset = _track_setter(i, "offset")
		lane.drag_started.connect(func(): _push_undo())
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

	var count := 1 + session.tracks.size()
	_lanes_col.offset_top = -90 - count * _LANE_H
	_lanes_col.offset_bottom = -90
	_tview.refresh(session)


func _delete_track(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	_push_undo()
	if i < _track_runtime.size():
		var rt: Dictionary = _track_runtime[i]
		if rt.has("player"):
			(rt.player as Node).queue_free()
		if rt.has("wrap"):
			(rt.wrap as Node).queue_free()
		_track_runtime.remove_at(i)
	session.tracks.remove_at(i)
	_refresh_lanes()
	_mark_dirty()


func _prompt_import_track() -> void:
	_import_dialog = FileDialog.new()
	_import_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_import_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_import_dialog.use_native_dialog = true
	_import_dialog.title = "Import a second track (picture-in-picture)"
	_import_dialog.filters = PackedStringArray(["*.mp4,*.mov,*.mkv,*.webm ; Video"])
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_import_dialog.current_dir = downloads
	_import_dialog.size = Vector2i(800, 560)
	_import_dialog.file_selected.connect(_start_track_import)
	add_child(_import_dialog)
	_import_dialog.popup_centered()


## Same one-time ffmpeg->theora transcode _prep() does for the primary clip,
## minus the audio extraction step (v1 tracks are silent - see the class doc's
## scope note) - PID-polled in _process(), never blocking, matching the
## project's one established pattern for external subprocesses.
func _start_track_import(source: String) -> void:
	var slug := _slugify(source)
	var dir := _session_path.get_base_dir()
	var video := dir + "/track_%s_%d.ogv" % [slug, session.tracks.size()]
	_import_pending = {"source": source, "video": video}
	_set_status("⏳  Importing track…")
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", source, "-an",
		"-c:v", "libtheora", "-q:v", "6", "-g", "25",
		ProjectSettings.globalize_path(video)])
	_import_pid = OS.create_process("ffmpeg", args)
	if _import_pid <= 0:
		_set_status("⚠  Could not start ffmpeg for track import")
		_import_pending = {}


func _finish_track_import() -> void:
	var abs_video := ProjectSettings.globalize_path(String(_import_pending.video))
	var dur := _probe_duration(abs_video)
	if dur <= 0.0:
		_set_status("⚠  Track import failed (could not read the transcoded video)")
		_import_pending = {}
		return
	_push_undo()
	session.tracks.append({
		"video_path": _import_pending.video, "duration": dur,
		"clip_in": 0.0, "clip_out": dur, "offset": 0.0,
		"x": 0.68, "y": 0.04, "w": 0.28, "h": 0.28,
	})
	_build_track_view(session.tracks.size() - 1)
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
		if inside:
			wrap.visible = true
			if not bool(rt.active) or absf(tplayer.stream_position - local_t) > 0.2:
				tplayer.stream_position = local_t
				rt.active = true
			tplayer.paused = not _playing
		else:
			wrap.visible = false
			tplayer.paused = true
			rt.active = false


func _build_render_view() -> void:
	var full := Control.new()
	full.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(full)
	_build_video_composition(full)
	for i in session.tracks.size():
		_build_track_view(i)
	# Trimmed exports start AT clip_in, never at 0 - a cut clip renders only its
	# kept range. Movie Maker records every processed frame, so the seek has to
	# land before the very first one, not the first _process() tick.
	_player.play()
	_player.stream_position = session.clip_in
	_apply_frame_state(session.at_time(session.clip_in))
	_audio.finished.connect(func(): get_tree().quit())


func _build_editor_ui() -> void:
	var video_area := AspectRatioContainer.new()
	video_area.set_anchors_preset(Control.PRESET_FULL_RECT)
	video_area.offset_left = PANEL_W
	video_area.ratio = 16.0 / 9.0
	add_child(video_area)

	# A plain Control fills the AspectRatioContainer's one centered/letterboxed slot.
	var inner := Control.new()
	inner.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	video_area.add_child(inner)
	_build_video_composition(inner)
	for i in session.tracks.size():
		_build_track_view(i)

	_tview = TimelineView.new()

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
	_timeline.marker_drag_started.connect(func(_m): _push_undo())
	_timeline.marker_moved.connect(func(_m):
		_refresh_marker_label()
		_mark_dirty())
	add_child(_timeline)

	# The trim/track lane stack - the primary clip's own trim block, plus one
	# lane per imported track - sits directly above the marker strip, sharing
	# its ruler via the same _tview (see _refresh_lanes for why its height is
	# computed and set explicitly rather than left to container auto-sizing).
	_lanes_col = VBoxContainer.new()
	_lanes_col.add_theme_constant_override("separation", 2)
	_lanes_col.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_lanes_col.offset_left = PANEL_W
	add_child(_lanes_col)
	_refresh_lanes()

	_build_panel()
	_build_export_ui()
	_build_feedback()
	_refresh_panel()
	_apply_frame_state(session.at_time(_player.stream_position))


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
	add_child(panel)

	# Two independently-scrolling regions, stacked: the controls above (which can
	# get tall - color pickers, a dozen sliders) scroll in whatever space is left,
	# and the sequential ramp/decay list is pinned to the bottom with its own fixed-
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

	var title := Label.new()
	title.text = "Mask Lab"
	title.add_theme_font_size_override("font_size", 22)
	col.add_child(title)

	var play_row := HBoxContainer.new()
	col.add_child(play_row)
	var play_btn := Button.new()
	play_btn.text = "▶ / ⏸"
	play_btn.focus_mode = Control.FOCUS_NONE
	play_btn.pressed.connect(func(): _play(not _playing))
	play_row.add_child(play_btn)

	# View toggle: cycles pip (raw + a masked inset) -> masked (full look) -> raw
	# (the default - just the source video, nothing masked/effected until you
	# explicitly switch or place a marker) -> back to pip. See VIEW_MODES /
	# _apply_frame_state / MaskSession.DEFAULTS.
	_view_btn = Button.new()
	_view_btn.focus_mode = Control.FOCUS_NONE
	_view_btn.custom_minimum_size = Vector2(160, 0)
	_view_btn.pressed.connect(_cycle_view_mode)
	play_row.add_child(_view_btn)

	# Peek: a display-only raw view for inspecting footage / picking colors. The
	# view button EDITS the marker at the playhead (that's its job - view modes
	# are sequenced cinematically), which made it a trap for quick checks: "let
	# me just look at raw for a second" silently rewrote the current marker's
	# view. Peek renders raw without touching any data.
	_peek_btn = Button.new()
	_peek_btn.text = "👁 Peek"
	_peek_btn.tooltip_text = "Show raw footage while held ON - display only, edits nothing"
	_peek_btn.toggle_mode = true
	_peek_btn.focus_mode = Control.FOCUS_NONE
	_peek_btn.toggled.connect(func(on): _peek_raw = on)
	play_row.add_child(_peek_btn)

	# Undo/redo: buttons mirror Ctrl+Z / Ctrl+Shift+Z (see _unhandled_input) for
	# anyone who doesn't reach for the shortcut - pressing on an empty stack is
	# just a harmless no-op, not worth wiring up disabled-state tracking for.
	# OWN row, not crammed into play_row - that row's three buttons (one of them
	# a forced 160px-wide _view_btn) already claim nearly the full panel width;
	# two more pushed the row's natural size well past PANEL_W and right over
	# the timeline, wide enough that its markers stopped being clickable.
	var hist_row := HBoxContainer.new()
	col.add_child(hist_row)
	var undo_btn := Button.new()
	undo_btn.text = "↶ Undo"
	undo_btn.tooltip_text = "Ctrl+Z"
	undo_btn.focus_mode = Control.FOCUS_NONE
	undo_btn.pressed.connect(_undo)
	hist_row.add_child(undo_btn)
	var redo_btn := Button.new()
	redo_btn.text = "↷ Redo"
	redo_btn.tooltip_text = "Ctrl+Shift+Z"
	redo_btn.focus_mode = Control.FOCUS_NONE
	redo_btn.pressed.connect(_redo)
	hist_row.add_child(redo_btn)

	# Own row again (see the comment above hist_row - this panel is only
	# PANEL_W wide, and rows fill up fast).
	var track_row := HBoxContainer.new()
	col.add_child(track_row)
	var import_btn := Button.new()
	import_btn.text = "⬆ Import track"
	import_btn.tooltip_text = "Add a second video as a picture-in-picture overlay"
	import_btn.focus_mode = Control.FOCUS_NONE
	import_btn.pressed.connect(_prompt_import_track)
	track_row.add_child(import_btn)

	_time_label = Label.new()
	_time_label.add_theme_font_size_override("font_size", 15)
	_time_label.add_theme_color_override("font_color", Color(0.85, 0.9, 1.0))
	col.add_child(_time_label)

	# One channel by default (a channel = one target color + one effect + one
	# strength); the second is opt-in behind a toggle. Channels are independent
	# layers in the shader, not competing "sides" - so there's no swap control any
	# more either (swapping meant something when every pixel was forced to one side
	# or the other; independent channels just re-pick their own colors).
	_grp_color = VBoxContainer.new()
	_grp_color.add_theme_constant_override("separation", 8)
	col.add_child(_grp_color)
	_grp_color.add_child(HSeparator.new())
	_grp_color.add_child(_label("Key color - what this channel targets"))
	_color_a = ColorPickerButton.new()
	_color_a.focus_mode = Control.FOCUS_NONE
	_color_a.custom_minimum_size = Vector2(0, 40)
	_color_a.edit_alpha = false
	_color_a.color_changed.connect(func(c): _edit("hue_a", c.h))
	_grp_color.add_child(_color_a)
	col.add_child(_label("Effect"))
	_effect_a = _effect_menu(col, func(id): _edit("effect_a", float(id)))
	_intensity_a = _slider(col, "Intensity", 0.0, 1.0, func(v): _edit("intensity_a", v))

	# Option GROUPS, shown per the selected effect's needs (the control hierarchy,
	# MaskSession.EFFECT_CONTROLS): a slider that does nothing for the current
	# effect is not on screen for it. Erase shows none of these (projection is
	# gate-free); restore shows only the threshold, relabeled as its reach;
	# the volumetrics show everything. See _update_effect_controls.
	_grp_threshold = VBoxContainer.new()
	_grp_threshold.add_theme_constant_override("separation", 8)
	col.add_child(_grp_threshold)
	_grp_threshold.add_child(HSeparator.new())
	_threshold_label = _label("Threshold")
	_grp_threshold.add_child(_threshold_label)
	_threshold = HSlider.new()
	_threshold.focus_mode = Control.FOCUS_NONE
	_threshold.scrollable = false   # wheel scrolls the panel, never edits (see _slider)
	_threshold.min_value = 0.0
	_threshold.max_value = 1.0
	_threshold.step = 0.005
	_threshold.value_changed.connect(func(v): _edit("threshold", v))
	_grp_threshold.add_child(_threshold)

	_grp_keymisc = VBoxContainer.new()
	_grp_keymisc.add_theme_constant_override("separation", 8)
	col.add_child(_grp_keymisc)
	_feather = _slider(_grp_keymisc, "Feather", 0.0, 0.5, func(v): _edit("feather", v))
	_sat_floor = _slider(_grp_keymisc, "Min colorfulness", 0.0, 1.0, func(v): _edit("sat_floor", v))

	# The wisp field's placement - pan/zoom the pattern over the frame (keyframe a
	# tendril onto an eye), and dial its coverage from one wisp to an engulfing.
	# All continuous marker fields, so they blend through ramps/decays.
	_grp_pattern = VBoxContainer.new()
	_grp_pattern.add_theme_constant_override("separation", 8)
	col.add_child(_grp_pattern)
	_grp_pattern.add_child(HSeparator.new())
	_grp_pattern.add_child(_label("Pattern - field placement"))
	_fx_x_label = _label("Pan X")
	_grp_pattern.add_child(_fx_x_label)
	_fx_x = HSlider.new()
	_fx_x.focus_mode = Control.FOCUS_NONE
	_fx_x.scrollable = false
	_fx_x.min_value = -2.0
	_fx_x.max_value = 2.0
	_fx_x.step = 0.01
	_fx_x.value_changed.connect(func(v): _edit("fx_x", v))
	_grp_pattern.add_child(_fx_x)
	_fx_y_label = _label("Pan Y")
	_grp_pattern.add_child(_fx_y_label)
	_fx_y = HSlider.new()
	_fx_y.focus_mode = Control.FOCUS_NONE
	_fx_y.scrollable = false
	_fx_y.min_value = -2.0
	_fx_y.max_value = 2.0
	_fx_y.step = 0.01
	_fx_y.value_changed.connect(func(v): _edit("fx_y", v))
	_grp_pattern.add_child(_fx_y)
	_fx_scale = _slider(_grp_pattern, "Scale", 0.1, 8.0, func(v): _edit("fx_scale", v))
	_fx_scale.exp_edit = true
	_fx_density_label = _label("Coverage")
	_grp_pattern.add_child(_fx_density_label)
	_fx_density = HSlider.new()
	_fx_density.focus_mode = Control.FOCUS_NONE
	_fx_density.scrollable = false
	_fx_density.min_value = 0.0
	_fx_density.max_value = 1.0
	_fx_density.step = 0.005
	_fx_density.value_changed.connect(func(v): _edit("fx_density", v))
	_grp_pattern.add_child(_fx_density)
	_fx_contrast_label = _label("Contrast")
	_grp_pattern.add_child(_fx_contrast_label)
	_fx_contrast = HSlider.new()
	_fx_contrast.focus_mode = Control.FOCUS_NONE
	_fx_contrast.scrollable = false
	_fx_contrast.min_value = 0.0
	_fx_contrast.max_value = 1.0
	_fx_contrast.step = 0.005
	_fx_contrast.value_changed.connect(func(v): _edit("fx_contrast", v))
	_grp_pattern.add_child(_fx_contrast)
	_fx_speed = _slider(_grp_pattern, "Velocity", 0.1, 4.0, func(v): _edit("fx_speed", v))
	_fx_speed.exp_edit = true

	_grp_echo = VBoxContainer.new()
	_grp_echo.add_theme_constant_override("separation", 8)
	col.add_child(_grp_echo)
	_grp_echo.add_child(HSeparator.new())
	_echo_header = _label("Echo - how the past is worn")
	_grp_echo.add_child(_echo_header)
	_fx_lag = _slider(_grp_echo, "Lag (s)", 0.05, 2.4, func(v): _edit("fx_lag", v))
	_fx_lag.exp_edit = true
	_fx_smooth = _slider(_grp_echo, "Smoothing - stutter → smear", 0.0, 1.0, func(v): _edit("fx_smooth", v))

	# Snow's own view onto fx_smooth - a separate widget from echo's Smoothing
	# above (same stored field, different meaning; the two groups never show
	# together, see _update_effect_controls, so there's no risk of them
	# fighting over what the slider looks like).
	_grp_snow = VBoxContainer.new()
	_grp_snow.add_theme_constant_override("separation", 8)
	col.add_child(_grp_snow)
	_grp_snow.add_child(HSeparator.new())
	_grp_snow.add_child(_label("Weather - how the fall drifts"))
	_gust = _slider(_grp_snow, "Gust - 0 = steady drift, 1 = chaotic gusts", 0.0, 1.0, func(v): _edit("fx_smooth", v))

	# Fur's tendril dynamics - fur-only views onto fx_smooth/fx_lag, the same
	# stored-field reuse as snow's Gust above (the groups never show together,
	# see _update_effect_controls).
	_grp_fur = VBoxContainer.new()
	_grp_fur.add_theme_constant_override("separation", 8)
	col.add_child(_grp_fur)
	_grp_fur.add_child(HSeparator.new())
	_grp_fur.add_child(_label("Tendrils - how the strands move"))
	_undul = _slider(_grp_fur, "Undulation - traveling waves along each strand", 0.0, 1.0, func(v): _edit("fx_smooth", v))
	_coil = _slider(_grp_fur, "Coil - eddies and spiral curl", 0.0, 1.0, func(v): _edit("fx_lag", v))

	_resonance = _slider(_grp_pattern, "Resonance (audio drive)", 0.0, 1.0, func(v): _edit("resonance", v))

	col.add_child(HSeparator.new())
	# Every marker is a ramp or a decay - there is no plain/neutral marker (see
	# MaskSession class doc). Both transition TO this marker's values; the kind is
	# which side of the anchor the transition occupies: a ramp eases in BEFORE,
	# complete at the anchor; a decay begins AT the anchor and accumulates after.
	col.add_child(_label("Kind - which way this marker's change runs"))
	_kind = OptionButton.new()
	_kind.focus_mode = Control.FOCUS_NONE
	for i in MaskSession.MARKER_KINDS.size():
		_kind.add_item(MaskSession.MARKER_KINDS[i].capitalize(), i)
	_kind.item_selected.connect(func(id): _edit("kind", float(id)))
	col.add_child(_kind)
	# Exponential response: fine-grained fractions of a second on the left, whole
	# minutes on the right - one slider covers a subtle 0.2s blend and a transition
	# spanning the entire clip. (exp_edit needs a strictly positive min.)
	_marker_duration = _slider(col, "Ramp/decay span (s)", 0.05, maxf(8.0, session.duration),
		func(v): _edit("duration", v))
	_marker_duration.exp_edit = true
	_marker_duration.step = 0.01

	# --- create/delete + the sequential list, pinned to the panel's bottom with its
	# --- own scroll - the whole "manage markers" workflow stays visible together,
	# --- rather than the create buttons living up in the scrolling edit area where
	# --- reaching them means scrolling past everything else first.
	outer.add_child(HSeparator.new())
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

	var mrow := HBoxContainer.new()
	list_col.add_child(mrow)
	var ramp_btn := Button.new()
	ramp_btn.text = "+ Ramp"
	ramp_btn.tooltip_text = "Eases IN before the playhead, arriving here complete"
	ramp_btn.focus_mode = Control.FOCUS_NONE
	ramp_btn.pressed.connect(func(): _add_marker_at_playhead(0))
	mrow.add_child(ramp_btn)
	var decay_btn := Button.new()
	decay_btn.text = "+ Decay"
	decay_btn.tooltip_text = "Begins here and accumulates over the span that follows"
	decay_btn.focus_mode = Control.FOCUS_NONE
	decay_btn.pressed.connect(func(): _add_marker_at_playhead(1))
	mrow.add_child(decay_btn)
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


func _label(text: String) -> Label:
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
	return l


func _slider(col: VBoxContainer, text: String, lo: float, hi: float, cb: Callable) -> HSlider:
	col.add_child(_label(text))
	var s := HSlider.new()
	s.focus_mode = Control.FOCUS_NONE
	# The wheel belongs to the panel's ScrollContainer, never to a slider you
	# happen to pass over on the way down - the panel is long enough now that
	# scrolling it drags random knobs (and silently edits the marker under
	# them). Drag-only.
	s.scrollable = false
	s.min_value = lo
	s.max_value = hi
	s.step = (hi - lo) / 200.0
	s.value_changed.connect(cb)
	col.add_child(s)
	return s


func _effect_menu(col: VBoxContainer, cb: Callable) -> OptionButton:
	var ob := OptionButton.new()
	ob.focus_mode = Control.FOCUS_NONE
	for i in MaskSession.MASK_EFFECTS.size():
		ob.add_item(MaskSession.MASK_EFFECTS[i], i)
	ob.item_selected.connect(cb)
	col.add_child(ob)
	return ob


# --- marker editing -----------------------------------------------------------

## Every panel edit targets the selected marker; if none is selected yet, planting
## one at the current playhead is the edit's first move (a knob you touch becomes a
## marker - no separate "create" step needed for the common case). Defaults to a
## ramp when auto-created this way; press +Decay explicitly for the other kind.
func _edit(field: String, value: float) -> void:
	var m: Variant = _selected
	if m == null:
		_push_undo()   # about to create a marker - always its own boundary
		m = session.add_marker(_player.stream_position if _player != null else 0.0)
		_selected = m
		_select_generation += 1
	else:
		_push_undo("marker:%d:%s" % [_select_generation, field])
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
	_push_undo()
	_selected = session.add_marker(_player.stream_position if _player != null else 0.0, kind_id)
	_select_generation += 1
	_timeline.selected = _selected
	_refresh_panel()
	_mark_dirty()


func _delete_selected() -> void:
	if _selected != null:
		_push_undo()
		session.remove_marker(_selected)
		_selected = null
		_timeline.selected = null
		_refresh_panel()
		_mark_dirty()


## Temporal capture: when the playhead crosses into a new _ECHO_INTERVAL slot,
## snapshot the current frame (quarter-res) into the ring and push the ring to
## both materials in AGE ORDER (u_echo0 = newest). GPU readback at ~3Hz is cheap
## enough for a demo; sessions without whisp/echo markers skip all of it.
func _maybe_capture_echo() -> void:
	if _player == null or session == null or not _session_uses_temporal():
		return
	var slot := int(_player.stream_position / _ECHO_INTERVAL)
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


## The anchor uniform, glided per frame: lerp(prev EMA, latest EMA) by the
## playhead's fraction through the current capture slot. Deterministic (pure
## function of playback position + capture history) and continuous - the
## pattern drifts to each new lock instead of jumping there.
func _push_anchor() -> void:
	var f := clampf(fposmod(_player.stream_position, _ECHO_INTERVAL) / _ECHO_INTERVAL, 0.0, 1.0)
	var anchor := _anchor_prev.lerp(_anchor_ema, f)
	for mat in [_mat_main, _mat_inset]:
		mat.set_shader_parameter("u_anchor", anchor)
		# Only push once impulses exist - a short array is silently dropped by
		# Godot (see the u_l_* comment above), so an empty/partial one is worse
		# than leaving the shader's own all-zero (inactive) uniform defaults.
		if _wave_amp.size() == _WAVE_SLOTS:
			mat.set_shader_parameter("u_wave_pos", _wave_pos)
			mat.set_shader_parameter("u_wave_time", _wave_time)
			mat.set_shader_parameter("u_wave_amp", _wave_amp)


func _session_uses_temporal() -> bool:
	for m in session.markers:
		var e := int(m.get("effect_a", 0))
		if e == 5 or e == 7 or e == MaskSession.EFFECT_SNOW or e == MaskSession.EFFECT_ORACLE:
			return true   # whisp (anchor) / echo / snow (motion probe) / oracle (delayed world)
	return false


## The whisp anchor: the first whisp marker's target-color mass centroid in
## the captured frame, EMA-smoothed (alpha 0.3 per capture ≈ a short
## multi-window average) so the lock glides to landmarks instead of jittering
## with noise. Fur no longer reads this - its strands root per-pixel on the
## keyed surface itself (see fur_root_mass / the fur branch of apply_layer
## in mask_split.gdshader).
func _update_whisp_anchor(img: Image) -> void:
	var hue := -1.0
	for m in session.markers:
		var e := int(m.get("effect_a", 0))
		if e == 5:
			hue = float(m.get("hue_a", 0.0))
			break
	if hue < 0.0:
		return
	var tc := Color.from_hsv(hue, 1.0, 1.0)
	var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
	var tdir := Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized()
	img.resize(48, 27, Image.INTERPOLATE_BILINEAR)
	var acc := Vector2.ZERO
	var wsum := 0.0
	var have_prev := _wave_prev_lum.size() == 48 * 27
	if not have_prev:
		_wave_prev_lum.resize(48 * 27)
	var motion := 0.0
	for y in 27:
		for x in 48:
			var c := img.get_pixel(x, y)
			var l := 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			var pr := maxf(0.0, (c.r - l) * tdir.x + (c.g - l) * tdir.y + (c.b - l) * tdir.z)
			acc += Vector2((float(x) + 0.5) / 48.0, (float(y) + 0.5) / 27.0) * pr
			wsum += pr
			var idx := y * 48 + x
			if have_prev:
				motion += absf(l - _wave_prev_lum[idx])
			_wave_prev_lum[idx] = l
	if wsum > 0.01:
		_anchor_prev = _anchor_ema
		_anchor_ema = _anchor_ema.lerp(acc / wsum, 0.15)
	if have_prev:
		_update_wave_impulses(motion / (48.0 * 27.0))


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


func _push_undo(key: String = "") -> void:
	if key != "" and key == _undo_coalesce_key and _undo_coalesce_cooldown > 0.0:
		_undo_coalesce_cooldown = _UNDO_COALESCE_WINDOW
		return
	_undo_coalesce_key = key
	_undo_coalesce_cooldown = _UNDO_COALESCE_WINDOW
	_undo_stack.append(_snapshot())
	if _undo_stack.size() > _UNDO_LIMIT:
		_undo_stack.pop_front()
	_redo_stack.clear()


func _undo() -> void:
	if _undo_stack.is_empty():
		return
	_redo_stack.append(_snapshot())
	_restore_snapshot(_undo_stack.pop_back())
	_undo_coalesce_key = ""   # the next edit must open its own fresh boundary
	_after_history_restore()


func _redo() -> void:
	if _redo_stack.is_empty():
		return
	_undo_stack.append(_snapshot())
	_restore_snapshot(_redo_stack.pop_back())
	_undo_coalesce_key = ""
	_after_history_restore()


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
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))
	_dirty = false


## Whatever the debounce hasn't flushed yet lands on disk when the editor goes
## away - closing the window mid-burst never loses the last edit.
func _exit_tree() -> void:
	if _dirty:
		_save_session()
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
	# color either.
	_grp_color.visible = effect_id != MaskSession.EFFECT_CLEAR and effect_id != MaskSession.EFFECT_SNOW
	_grp_threshold.visible = groups.has("keying") or groups.has("reach")
	_threshold_label.text = "Reach - hue range this restore covers" if groups.has("reach") else "Threshold"
	_grp_keymisc.visible = groups.has("keying")
	_grp_pattern.visible = groups.has("pattern")
	_grp_echo.visible = groups.has("echo")
	_grp_snow.visible = groups.has("snow")
	_grp_fur.visible = groups.has("fur")
	_echo_header.text = "Oracle - how far ahead it leads" \
		if effect_id == MaskSession.EFFECT_ORACLE else "Echo - how the past is worn"
	_fx_contrast_label.text = "Sensitivity - snow's reach toward the subject" \
		if effect_id == MaskSession.EFFECT_SNOW else "Contrast"
	var is_snow := effect_id == MaskSession.EFFECT_SNOW
	_fx_x_label.text = "Wind X" if is_snow else "Pan X"
	_fx_y_label.text = "Wind Y" if is_snow else "Pan Y"
	_fx_density_label.text = "Stickiness - pull toward the face's edges" \
		if effect_id == MaskSession.EFFECT_CRYSTAL else "Coverage"


func _refresh_marker_label() -> void:
	if _selected == null:
		_marker_label.text = "nothing selected - editing plants a ramp here"
	else:
		var t := float(_selected.time)
		var kind_name: String = MaskSession.MARKER_KINDS[int(_selected.get("kind", 0.0))]
		_marker_label.text = "%s @ %s  (%d total)" % \
			[kind_name.capitalize(), MaskTimeline.format_time(t), session.markers.size()]
	_refresh_marker_list()


## The sequential ramp/decay list, pinned to the panel's bottom. Rebuilt wholesale -
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
	# THE VIDEO IS THE MASTER CLOCK. The two players don't pause at the same
	# instant (video stops on a decoded-frame boundary, audio on a mix chunk),
	# so every pause/resume cycle - spacebar, the feedback console's freeze -
	# banked a little offset, and nothing ever corrected it. Snap audio to the
	# video on every resume; _process keeps them corrected from there.
	if on:
		_audio.seek(_player.stream_position)


## Space toggles play/pause - the same action the panel's ▶/⏸ button does. Only
## live once a clip is actually loaded (_player exists); main.gd defers to this
## instance for Space entirely while it's open (see main.gd's KEY_SPACE handling),
## so this doesn't need to fight Director.next() for the key.
## Ctrl+Z / Ctrl+Shift+Z (and Ctrl+Y, the other common redo binding) drive the
## undo/redo stack - see _push_undo. echo excluded so a held key doesn't spam
## repeats; a real accident deserves a deliberate press each time it's undone.
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
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_SPACE:
			_play(not _playing)
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
const VIEW_CYCLE := [2, 3, 0, 4, 1]   # raw -> pip_raw -> pip -> masked_pip -> masked

## The toggle button edits the marker at the playhead exactly like every other panel
## control (see _edit) - cycling is relative to whatever's ACTIVE right now, not to
## some separately-tracked button state, so it can never drift from the timeline.
func _cycle_view_mode() -> void:
	var cur := 2
	if session != null and _player != null:
		cur = int(session.at_time(_player.stream_position).get("view_mode", 2.0))
	var i := VIEW_CYCLE.find(cur)
	var next_id: int = VIEW_CYCLE[(i + 1) % VIEW_CYCLE.size()] if i >= 0 else VIEW_CYCLE[0]
	_edit("view_mode", float(next_id))   # _process applies it next frame


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
	var t: float = _player.stream_position if _player != null else 0.0
	var env := _env_at(t)
	var layers: Array = p.get("layers", [])
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
			for k in 8:
				echo_w.append(1.0 if k == 0 else 0.0)
			glows.append(1.0)
			tdirs.append(Vector3(1, 0, 0))
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
		var tex := _player.get_video_texture() if _player != null else null
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
		mat.set_shader_parameter("u_l_tdir", tdirs)
	_fx_overlay.visible = main_amt > 0.001
	_mask_wrap.visible = inset_show > 0.001
	_mask_wrap.modulate.a = inset_show
	if _view_btn != null:
		match MaskSession.VIEW_MODES[clampi(int(p.get("view_mode", 2.0)), 0, MaskSession.VIEW_MODES.size() - 1)]:
			"raw":        _view_btn.text = "🎬  Raw"
			"pip_raw":    _view_btn.text = "🖼  PiP (raw)"
			"pip":        _view_btn.text = "🖼  PiP (fx)"
			"masked_pip": _view_btn.text = "🎭  Both (fx)"
			"masked":     _view_btn.text = "🎭  Full (fx)"


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
	if session == null or _player == null:
		return
	# Same call, live or exported: whatever the timeline says at this instant is
	# what's shown - render_mode doesn't special-case a fixed "always masked" look.
	_apply_frame_state(session.at_time(_player.stream_position))
	# Only one video ever decodes (_player); the fx overlay and the inset just
	# re-draw that same frame, each only when actually on screen - "raw" mode skips
	# both (and the shader passes they'd otherwise cost) entirely.
	if _fx_overlay != null and _fx_overlay.visible:
		_fx_overlay.texture = _player.get_video_texture()
	if _pip_view != null and _mask_wrap.visible:
		_pip_view.texture = _player.get_video_texture()
	_maybe_capture_echo()
	_push_anchor()
	# Standing A/V drift correction (see _play: video is the master clock).
	# 0.15s tolerance sits above audio mix-chunk granularity so this never
	# chatters; beyond it, snap audio back to the video.
	if _playing and _audio.playing and not _audio.stream_paused:
		if absf(_audio.get_playback_position() - _player.stream_position) > 0.15:
			_audio.seek(_player.stream_position)
	_sync_tracks()
	# A trimmed clip's OUT point is a hard wall for playback (both live preview
	# and the export relaunch - export additionally needs a QUIT, not just a
	# pause, since Movie Maker keeps recording for as long as the process runs).
	# clip_in is not enforced here on purpose: scrubbing earlier to look at
	# trimmed-away footage while editing is fine, only PLAYBACK (and export) are
	# bounded to the kept range.
	if session.clip_out > 0.0 and _player.stream_position >= session.clip_out:
		if render_mode:
			get_tree().quit()
		elif _playing:
			_play(false)
	if _undo_coalesce_cooldown > 0.0:
		_undo_coalesce_cooldown -= _dt
	if render_mode:
		return
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
	_status.offset_left = -560
	_status.offset_top = -64
	_status.offset_right = -238
	_status.offset_bottom = -28
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_status.visible = false
	add_child(_status)


# --- export: relaunch in Movie Maker mode (--mask-render), then ffmpeg mux ------

func _build_export_ui() -> void:
	_export_btn = Button.new()
	_export_btn.focus_mode = Control.FOCUS_NONE
	_export_btn.text = "⤓  Export video"
	_export_btn.tooltip_text = "Render this mask session to a video file (in the background)"
	_export_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_export_btn.offset_left = -210
	_export_btn.offset_top = -72
	_export_btn.offset_right = -28
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
				_start_transcode()
			else:
				_set_status("⚠  Render produced no file (see console)")
				_render_state = "idle"
		"transcoding":
			if OS.is_process_running(_transcode_pid):
				return
			if _file_size(_out) > 4096:
				DirAccess.remove_absolute(_avi)
				_set_status("✓  Saved  " + _out)
			else:
				_set_status("⚠  Transcode failed; raw file kept: " + _avi)
			_render_state = "idle"


func _start_transcode() -> void:
	_set_status("⏳  Finalizing…")
	_transcode_pid = OS.create_process("ffmpeg", PackedStringArray([
		"-y", "-loglevel", "error", "-i", _avi,
		"-c:v", "libx264", "-preset", "medium", "-crf", "18",
		"-c:a", "aac", "-b:a", "192k", _out]))
	_render_state = "transcoding"


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
