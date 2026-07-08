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

var _fx_overlay: TextureRect       # full-frame fx layer - _player's texture, shaded
var _pip_view: TextureRect         # the inset's content - shaded or raw per view mode
var _mask_wrap: PanelContainer     # the inset's border/placement box (holds _pip_view)
var _view_btn: Button

var _timeline: MaskTimeline
var _selected: Variant = null   # the marker Dictionary currently shown in the panel

var _color_a: ColorPickerButton
var _color_b: ColorPickerButton
var _threshold: HSlider
var _feather: HSlider
var _sat_floor: HSlider
var _effect_a: OptionButton
var _effect_b: OptionButton
var _intensity_a: HSlider
var _intensity_b: HSlider
var _chan2: VBoxContainer     # second channel's controls - hidden until opted into
var _chan2_toggle: Button
var _kind: OptionButton     # ramp / decay - see MaskSession.MARKER_KINDS
var _marker_duration: HSlider
var _marker_label: Label
var _time_label: Label
var _marker_list: VBoxContainer   # sequential ramp/decay list, pinned to the panel's bottom

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

var _waveform_pid := -1
var _waveform_path := ""         # set once we know it; polled in _process until it exists

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
		_timeline.waveform_texture = _load_png(abs_wave)
		return
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", ProjectSettings.globalize_path(session.audio_path),
		"-filter_complex", "showwavespic=s=4096x160:colors=white:scale=sqrt",
		"-frames:v", "1", abs_wave])
	_waveform_pid = OS.create_process("ffmpeg", args)


## Plain load() has no loader for a raw .png outside the import pipeline (the same
## lesson as AudioStreamWAV.load_from_file elsewhere in this file) - Image's own
## static loader is the runtime-safe path.
static func _load_png(path: String) -> Texture2D:
	var img := Image.load_from_file(path)
	return ImageTexture.create_from_image(img) if img != null else null


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


func _build_render_view() -> void:
	var full := Control.new()
	full.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(full)
	_build_video_composition(full)
	# Set the correct state for frame 0 - Movie Maker records every processed frame,
	# so leaving this to the first _process() tick would bake one wrong frame in.
	_apply_frame_state(session.at_time(0.0))
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

	_timeline = MaskTimeline.new()
	_timeline.session = session
	_timeline.player = _player
	_timeline.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_timeline.offset_left = PANEL_W
	_timeline.offset_bottom = -90
	_timeline.offset_top = -90
	_timeline.scrubbed.connect(_on_scrub)
	_timeline.marker_picked.connect(_select_marker)
	_timeline.marker_moved.connect(func(_m):
		_refresh_marker_label()
		_mark_dirty())
	add_child(_timeline)

	_build_panel()
	_build_export_ui()
	_refresh_panel()
	_apply_frame_state(session.at_time(_player.stream_position))


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

	_time_label = Label.new()
	_time_label.add_theme_font_size_override("font_size", 15)
	_time_label.add_theme_color_override("font_color", Color(0.85, 0.9, 1.0))
	col.add_child(_time_label)

	# One channel by default (a channel = one target color + one effect + one
	# strength); the second is opt-in behind a toggle. Channels are independent
	# layers in the shader, not competing "sides" - so there's no swap control any
	# more either (swapping meant something when every pixel was forced to one side
	# or the other; independent channels just re-pick their own colors).
	col.add_child(HSeparator.new())
	col.add_child(_label("Key color - what this channel targets"))
	_color_a = ColorPickerButton.new()
	_color_a.custom_minimum_size = Vector2(0, 40)
	_color_a.edit_alpha = false
	_color_a.color_changed.connect(func(c): _edit("hue_a", c.h))
	col.add_child(_color_a)
	col.add_child(_label("Effect"))
	_effect_a = _effect_menu(col, func(id): _edit("effect_a", float(id)))
	_intensity_a = _slider(col, "Intensity", 0.0, 1.0, func(v): _edit("intensity_a", v))

	_chan2_toggle = Button.new()
	_chan2_toggle.text = "+ second channel"
	_chan2_toggle.focus_mode = Control.FOCUS_NONE
	_chan2_toggle.pressed.connect(_toggle_chan2)
	col.add_child(_chan2_toggle)

	_chan2 = VBoxContainer.new()
	_chan2.add_theme_constant_override("separation", 8)
	_chan2.visible = false
	col.add_child(_chan2)
	_chan2.add_child(_label("Key color - channel 2"))
	_color_b = ColorPickerButton.new()
	_color_b.custom_minimum_size = Vector2(0, 40)
	_color_b.edit_alpha = false
	_color_b.color_changed.connect(func(c): _edit("hue_b", c.h))
	_chan2.add_child(_color_b)
	_chan2.add_child(_label("Effect"))
	_effect_b = _effect_menu(_chan2, func(id): _edit("effect_b", float(id)))
	_intensity_b = _slider(_chan2, "Intensity", 0.0, 1.0, func(v): _edit("intensity_b", v))

	col.add_child(HSeparator.new())
	_threshold = _slider(col, "Threshold", 0.0, 1.0, func(v): _edit("threshold", v))
	_feather = _slider(col, "Feather", 0.0, 0.5, func(v): _edit("feather", v))
	_sat_floor = _slider(col, "Min saturation", 0.0, 1.0, func(v): _edit("sat_floor", v))

	col.add_child(HSeparator.new())
	# Every marker is a ramp or a decay - there is no plain/neutral marker (see
	# MaskSession class doc). Both transition TO this marker's values; the kind is
	# which side of the anchor the transition occupies: a ramp eases in BEFORE,
	# complete at the anchor; a decay begins AT the anchor and accumulates after.
	col.add_child(_label("Kind - which way this marker's change runs"))
	_kind = OptionButton.new()
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
	s.min_value = lo
	s.max_value = hi
	s.step = (hi - lo) / 200.0
	s.value_changed.connect(cb)
	col.add_child(s)
	return s


func _effect_menu(col: VBoxContainer, cb: Callable) -> OptionButton:
	var ob := OptionButton.new()
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
		m = session.add_marker(_player.stream_position if _player != null else 0.0)
		_selected = m
	m[field] = value
	_timeline.selected = _selected
	_refresh_marker_label()
	_mark_dirty()


func _add_marker_at_playhead(kind_id: int) -> void:
	_selected = session.add_marker(_player.stream_position if _player != null else 0.0, kind_id)
	_timeline.selected = _selected
	_refresh_panel()
	_mark_dirty()


func _delete_selected() -> void:
	if _selected != null:
		session.remove_marker(_selected)
		_selected = null
		_timeline.selected = null
		_refresh_panel()
		_mark_dirty()


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


func _select_marker(m: Dictionary) -> void:
	_selected = m
	_timeline.selected = m
	_refresh_panel()


func _refresh_panel() -> void:
	var m: Dictionary = _selected if _selected != null else MaskSession.DEFAULTS
	_kind.select(int(m.get("kind", 0.0)))
	_color_a.color = Color.from_hsv(float(m.get("hue_a", 0.02)), 0.85, 0.9)
	_color_b.color = Color.from_hsv(float(m.get("hue_b", 0.58)), 0.85, 0.9)
	_threshold.set_value_no_signal(float(m.get("threshold", 0.24)))
	_feather.set_value_no_signal(float(m.get("feather", 0.12)))
	_sat_floor.set_value_no_signal(float(m.get("sat_floor", 0.18)))
	_effect_a.select(int(m.get("effect_a", 0)))
	_effect_b.select(int(m.get("effect_b", 0)))
	_intensity_a.set_value_no_signal(float(m.get("intensity_a", 1.0)))
	_intensity_b.set_value_no_signal(float(m.get("intensity_b", 0.0)))
	_marker_duration.set_value_no_signal(float(m.get("duration", 1.0)))
	# A marker actually USING channel 2 reveals its controls - otherwise leave the
	# user's own show/hide choice alone.
	if float(m.get("intensity_b", 0.0)) > 0.0 and not _chan2.visible:
		_toggle_chan2()
	_refresh_marker_label()


func _toggle_chan2() -> void:
	_chan2.visible = not _chan2.visible
	_chan2_toggle.text = "- hide second channel" if _chan2.visible else "+ second channel"


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
		b.text = "%s   %s" % [MaskTimeline.format_time(float(m.time)), kind_name.capitalize()]
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


## Space toggles play/pause - the same action the panel's ▶/⏸ button does. Only
## live once a clip is actually loaded (_player exists); main.gd defers to this
## instance for Space entirely while it's open (see main.gd's KEY_SPACE handling),
## so this doesn't need to fight Director.next() for the key.
func _unhandled_input(event: InputEvent) -> void:
	if render_mode or _player == null:
		return
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_SPACE:
		_play(not _playing)
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
	for mat in [_mat_main, _mat_inset]:
		mat.set_shader_parameter("u_hue_a", p.hue_a)
		mat.set_shader_parameter("u_hue_b", p.hue_b)
		mat.set_shader_parameter("u_threshold", p.threshold)
		mat.set_shader_parameter("u_feather", p.feather)
		mat.set_shader_parameter("u_sat_floor", p.sat_floor)
		mat.set_shader_parameter("u_effect_a", int(p.effect_a))
		mat.set_shader_parameter("u_effect_b", int(p.effect_b))
	_mat_main.set_shader_parameter("u_intensity_a", float(p.intensity_a) * main_amt)
	_mat_main.set_shader_parameter("u_intensity_b", float(p.intensity_b) * main_amt)
	_mat_inset.set_shader_parameter("u_intensity_a", float(p.intensity_a) * inset_fx)
	_mat_inset.set_shader_parameter("u_intensity_b", float(p.intensity_b) * inset_fx)
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
		if _timeline != null and FileAccess.file_exists(abs_wave):
			_timeline.waveform_texture = _load_png(abs_wave)
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
	if render_mode:
		return
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
