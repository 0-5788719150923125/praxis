extends SceneTree

## Does mask mode handle a clip that ISN'T 16:9 - a phone video shot in portrait,
## 1080x1920 - end to end?
##   godot --headless --path . --script res://tests/mask_portrait_check.gd
##
## Everything downstream of the picture is shaped by ONE pair of numbers,
## MaskEditor._src_size (probed from the clip, confirmed from the decoded texture).
## This probe boots a real MaskEditor on a synthetic portrait session and checks the
## three places that pair has to reach, because each of them used to be a hardcoded
## 16:9 constant and each failed differently:
##
##   1. the editor's video slot   - _video_area.ratio. Was 16.0/9.0, so a portrait
##      clip was stretched sideways to fill a wide box instead of being pillarboxed
##      (black bars left and right) inside it.
##   2. the exported resolution   - _render_size() / the override.cfg it writes.
##      Movie Maker locks its size at engine startup, so the EDITOR has to name the
##      size before the render process launches. Was never named at all, so every
##      export came out 1920x1080 whatever went in.
##   3. the meta mirror's pane    - _shrink_into_video_pane, which rebuilds the live
##      editor's pane geometry by hand for the export.
##
## SWITCHES (the portrait clip, what the fix is for) and HOLDS (a 16:9 clip, which
## must come out exactly as it always did) - the second half is the safety net, same
## discipline as the pronunciation gates.
##
## Fixtures are two tiny ffmpeg-generated clips under user://; regenerated only when
## missing. Nothing is written into the project tree.

const DIR := "user://mask_shape_check"
const PORTRAIT := Vector2i(360, 640)     # 9:16, the same shape as a 1080x1920 phone clip
const LANDSCAPE := Vector2i(640, 360)    # the control
const CLIP_SECONDS := 3

var _fails: PackedStringArray = []


func _initialize() -> void:
	# _write_render_override targets res://override.cfg, the same file
	# tests/run_boot_probe.sh uses to hijack the main scene. Never clobber one.
	if FileAccess.file_exists(ProjectSettings.globalize_path("res://override.cfg")):
		print("mask_portrait_check: override.cfg already exists in the project root - ",
			"inspect and remove it first. Refusing to run.")
		quit(2)
		return
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(DIR))
	for size in [PORTRAIT, LANDSCAPE]:
		if not _ensure_fixture(size):
			print("mask_portrait_check: could not build the %dx%d fixture (is ffmpeg on PATH?)"
				% [size.x, size.y])
			quit(2)
			return

	await _check(PORTRAIT, "SWITCH portrait")
	await _check(LANDSCAPE, "HOLD landscape")

	print("")
	if _fails.is_empty():
		print("mask_portrait_check: PASS - the clip's own frame size reaches the editor slot, ",
			"the export resolution and the meta pane.")
		quit(0)
	else:
		for f in _fails:
			print("mask_portrait_check: FAIL - ", f)
		quit(1)


## Boot a real MaskEditor on `size`'s session and check everything the picture's
## shape has to reach.
func _check(size: Vector2i, label: String) -> void:
	var aspect := float(size.x) / float(size.y)
	var editor: Node = load("res://scripts/mask_editor.gd").new()
	root.add_child(editor)
	editor.open_source(_session_path(size))
	# A few frames: the session opens synchronously, but _sync_source_size only
	# adopts the decoder's own size once the first frame has actually decoded, and
	# the container needs a layout pass before its rect means anything.
	for i in 30:
		await process_frame

	print("")
	print("--- %s (%dx%d, aspect %.4f) ---" % [label, size.x, size.y, aspect])

	# 1. THE SOURCE SIZE ITSELF.
	var src: Vector2i = editor._src_size
	print("  _src_size            %dx%d  (confirmed by the decoder: %s)"
		% [src.x, src.y, editor._src_size_confirmed])
	_expect(src == size, "%s: _src_size is %dx%d, expected %dx%d"
		% [label, src.x, src.y, size.x, size.y])

	# 2. THE EDITOR'S VIDEO SLOT. The container's ratio is the clip's, and the slot
	# it hands its child is genuinely narrower than the space available - which IS
	# the black bars either side, measured rather than assumed.
	var area: AspectRatioContainer = editor._video_area
	if area == null:
		_expect(false, "%s: no _video_area was built" % label)
	else:
		print("  _video_area.ratio    %.4f" % area.ratio)
		_expect(absf(area.ratio - aspect) < 0.001,
			"%s: _video_area.ratio is %.4f, expected the clip's %.4f"
			% [label, area.ratio, aspect])
		var avail := area.size
		var pane: Vector2 = (editor._composition_parent as Control).size
		print("  available %.0fx%.0f -> pane %.0fx%.0f  (bars: %.0f px side, %.0f px top/bottom)"
			% [avail.x, avail.y, pane.x, pane.y,
			   (avail.x - pane.x) * 0.5, (avail.y - pane.y) * 0.5])
		_expect(pane.x > 1.0 and pane.y > 1.0, "%s: the video pane has no size" % label)
		if pane.y > 1.0:
			_expect(absf(pane.x / pane.y - aspect) < 0.01,
				"%s: the pane is %.3f wide-to-tall, expected the clip's %.3f - the picture is being stretched"
				% [label, pane.x / pane.y, aspect])
		_expect(pane.x <= avail.x + 0.5 and pane.y <= avail.y + 0.5,
			"%s: the pane (%.0fx%.0f) overflows the space it was given (%.0fx%.0f)"
			% [label, pane.x, pane.y, avail.x, avail.y])
		if size.y > size.x:
			_expect(avail.x - pane.x > 1.0,
				"%s: a portrait clip should be PILLARBOXED - bars left and right - but the pane fills the width"
				% label)

	# 2b. THE PANEL IS WHOLE. Every row in the control list is a (label, control)
	# pair built by _build_panel and re-read by _refresh_panel; a control that is
	# declared and SYNCED but never built is null, and the sync line then throws on
	# every panel refresh - at open, on every selection, and once per frame from
	# _process. That shipped: a slider was cut along with the block around it while
	# its declaration and its sync line stayed, and nothing here noticed, because
	# a GDScript runtime error prints and carries on rather than failing a check.
	# This is the cheap structural assertion that catches the whole class.
	var opts: Array = editor._options
	print("  panel rows          %d" % opts.size())
	_expect(opts.size() > 20, "%s: only %d panel rows were registered" % [label, opts.size()])
	for row in opts:
		_expect(row.control != null,
			"%s: a registered panel row has no control - it is declared and synced "
			% label + "but never built, so every panel refresh throws")
		_expect(row.label != null,
			"%s: a registered panel row has no label - _apply_sort moves the two "
			% label + "together and will desynchronise the whole list")

	# 2c. NO LABEL HAS BEEN SQUEEZED INTO A COLUMN OF LETTERS. Every label in this
	# panel word-wraps - deliberately, because an unwrapped one's natural width
	# becomes the whole column's minimum and pushes the panel over the timeline -
	# and the price is that a wrapping label's MINIMUM width is about one
	# character. Put it in an HBox beside anything that expands and the container
	# hands the slack to the other child and squeezes the label to that minimum:
	# "In order" rendered 1px wide and 157px tall, eight lines of one letter, and
	# took the whole row with it. Nothing about that is visible from the code.
	# A short label wrapping to more than two lines is the signature.
	for node in _walk(editor):
		if not (node is Label):
			continue
		var lbl := node as Label
		if lbl.text.length() > 24 or lbl.text.is_empty() or not lbl.is_visible_in_tree():
			continue     # long ones may legitimately wrap
		_expect(lbl.get_line_count() <= 2,
			"%s: the label \"%s\" wrapped to %d lines in %.0fx%.0f - it has been "
			% [label, lbl.text, lbl.get_line_count(), lbl.size.x, lbl.size.y]
			+ "squeezed to its minimum width by something expanding beside it")

	# 3. THE EXPORT RESOLUTION. Both the number the editor computes and what it
	# actually writes into override.cfg for the render process to boot with.
	var rsz: Vector2i = editor._render_size()
	print("  _render_size()       %dx%d" % [rsz.x, rsz.y])
	_expect(rsz == size, "%s: the export would record %dx%d, expected the source's %dx%d"
		% [label, rsz.x, rsz.y, size.x, size.y])
	_expect(rsz.x % 2 == 0 and rsz.y % 2 == 0,
		"%s: %dx%d is odd - yuv420p cannot encode it" % [label, rsz.x, rsz.y])
	editor._write_render_override(rsz)
	var cfg := _read_override()
	editor._clear_render_override()
	print("  override.cfg         viewport %sx%s, stretch %s"
		% [cfg.get("window/size/viewport_width", "?"),
		   cfg.get("window/size/viewport_height", "?"),
		   cfg.get("window/stretch/mode", "?")])
	_expect(String(cfg.get("window/size/viewport_width", "")) == str(rsz.x)
		and String(cfg.get("window/size/viewport_height", "")) == str(rsz.y),
		"%s: override.cfg does not carry the render size" % label)
	_expect(String(cfg.get("window/stretch/mode", "")) == "\"viewport\"",
		"%s: override.cfg must set viewport stretch, or the movie records the WINDOW" % label)
	_expect(not FileAccess.file_exists(ProjectSettings.globalize_path("res://override.cfg")),
		"%s: override.cfg was left behind" % label)

	# 4. THE META MIRROR'S PANE, which rebuilds the live editor's geometry by hand
	# for the export. Same shape as the real pane, or the mirror shows a squashed
	# copy of the workspace.
	var img := Image.create_empty(240, 240, false, Image.FORMAT_RGBA8)
	var pane_img: Image = editor._shrink_into_video_pane(img)
	print("  meta mirror canvas   %dx%d" % [pane_img.get_width(), pane_img.get_height()])
	_expect(pane_img.get_width() == 240 and pane_img.get_height() == 240,
		"%s: the meta mirror must return a canvas the size of the capture" % label)

	editor.queue_free()
	await process_frame
	await process_frame


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


# --- fixtures -------------------------------------------------------------------

func _session_path(size: Vector2i) -> String:
	return "%s/%dx%d.json" % [DIR, size.x, size.y]


## A clip of `size`, its audio, and the session that pairs them - built once and
## cached. testsrc is deliberately a picture with structure in it, so a stretched
## frame would be visible if anyone ever looks at one of these by hand.
func _ensure_fixture(size: Vector2i) -> bool:
	var base := "%s/%dx%d" % [DIR, size.x, size.y]
	var video := base + ".ogv"
	var audio := base + ".wav"
	var abs_video := ProjectSettings.globalize_path(video)
	var abs_audio := ProjectSettings.globalize_path(audio)
	if not FileAccess.file_exists(abs_video):
		var out := []
		OS.execute("ffmpeg", ["-y", "-loglevel", "error",
			"-f", "lavfi", "-i", "testsrc=size=%dx%d:rate=25:duration=%d" % [size.x, size.y, CLIP_SECONDS],
			"-c:v", "libtheora", "-q:v", "6", "-g", "25", "-f", "ogg", abs_video], out)
		if not FileAccess.file_exists(abs_video):
			return false
	if not FileAccess.file_exists(abs_audio):
		var out2 := []
		OS.execute("ffmpeg", ["-y", "-loglevel", "error",
			"-f", "lavfi", "-i", "sine=frequency=220:duration=%d" % CLIP_SECONDS,
			"-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", "-f", "wav", abs_audio], out2)
		# The .ogg sidecar is the path _ready_with_session prefers; without it the
		# editor loads the raw WAV on a worker thread and holds playback for it.
		var out3 := []
		OS.execute("ffmpeg", ["-y", "-loglevel", "error", "-i", abs_audio,
			"-c:a", "libvorbis", "-q:a", "5", "-f", "ogg",
			abs_audio.get_basename() + ".ogg"], out3)
	var session := MaskSession.new()
	session.video_path = video
	session.audio_path = audio
	session.duration = float(CLIP_SECONDS)
	return session.save(ProjectSettings.globalize_path(_session_path(size)))


## override.cfg is a plain ini; read it back as key -> raw value so the check reads
## the file the render process would actually boot from, not our own arguments.
func _read_override() -> Dictionary:
	var out := {}
	var f := FileAccess.open(ProjectSettings.globalize_path("res://override.cfg"), FileAccess.READ)
	if f == null:
		return out
	for line in f.get_as_text().split("\n"):
		var s := line.strip_edges()
		var eq := s.find("=")
		if eq > 0 and not s.begins_with("["):
			out[s.substr(0, eq)] = s.substr(eq + 1)
	f.close()
	return out


## Every node under the editor, so the label sweep can reach the ones nested in
## rows and scroll containers rather than only the panel's direct children.
func _walk(n: Node) -> Array:
	var out := [n]
	for c in n.get_children():
		out.append_array(_walk(c))
	return out
