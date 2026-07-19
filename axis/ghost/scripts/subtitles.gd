extends CanvasLayer
class_name Subtitles

## Subtitles - the karaoke overlay, session-owned rather than editor-owned.
##
## Draws the current sentence at the bottom in the SOURCE spelling (caps and
## punctuation intact - the phoneme normalization never reaches the reader's
## eyes), wrapping onto up to three lines before it ever shrinks the font.
##
## The tracker is a **narrator's eye**, not a metronome: an eased cursor that
## chases the true playback position with momentum - it ramps up when the
## words run ahead, drifts when it is close, and comes to REST at pauses and
## hesitations (the target holds at a word boundary, so the eye settles there
## and waits for the voice) - weaving around the exact timing the way a
## storyteller's finger weaves over a page. Highlight hue rides the live
## harmonic signature.
##
## Timing comes from a **sidecar JSON** written next to a voice take by
## [SynthEditor] (`take_N.wav` + `take_N.json`), so the same overlay works
## everywhere the take plays: the live synthesis session, a plain `--audio`
## boot, and the export render, where no editor exists.

var words: Array = []               # [{text, t0, t1, sentence}] - may still be GROWING
                                    # (a live VoiceStream shares its array by reference)
var loop_length := 0.0              # >0 once a streamed take loops: wrap time by this
var time_base := 0.0                # playback time when the current content started
var _cursor := 0.0                  # the narrator's eye: global word progress, eased
var _hue_sm := 0.6
var _overlay: Control


## The sidecar path for an audio file, or "" if none exists.
static func sidecar_for(audio_path: String) -> String:
	if audio_path.is_empty():
		return ""
	var side := audio_path.get_basename() + ".json"
	return side if FileAccess.file_exists(side) else ""


## Load a sidecar written by the synth editor. Returns false on malformed data.
func load_sidecar(path: String) -> bool:
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not (parsed is Dictionary) or not (parsed.get("words") is Array):
		push_warning("ghost: subtitle sidecar unreadable: " + path)
		return false
	words = parsed.words
	return true


func _ready() -> void:
	layer = 9
	_overlay = Overlay.new()
	_overlay.owner_node = self
	_overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
	_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_overlay)


func _now() -> float:
	var t: float = Spectrum.current.time - time_base
	if loop_length > 0.0:
		t = fmod(maxf(t, 0.0), loop_length)
	return t


## The true position as global word progress: word index + fraction within it.
## Between words (a pause, a breath, a hesitation) the target HOLDS at the
## previous word's end - which is what lets the eased eye rest there.
func _target(t: float) -> float:
	var last_end := 0.0
	for k in words.size():
		var w: Dictionary = words[k]
		if t < float(w.t0):
			return float(k)          # in the gap before word k: rest at its door
		if t < float(w.t1):
			return float(k) + (t - float(w.t0)) / maxf(0.001, float(w.t1) - float(w.t0))
		last_end = float(k) + 1.0
	return last_end


func _process(delta: float) -> void:
	if words.is_empty():
		return
	var target := _target(_now())
	var gap := target - _cursor
	if absf(gap) > 3.0:
		_cursor = target             # a loop seam or a restart: snap, don't chase
	else:
		# the weave: momentum grows with how far behind the eye is, so it
		# ramps to catch a run of quick words and slows as it closes in
		var rate := 3.0 + 7.0 * clampf(absf(gap) - 0.15, 0.0, 1.5)
		_cursor = lerpf(_cursor, target, 1.0 - exp(-rate * delta))
	_overlay.queue_redraw()


class Overlay:
	extends Control
	var owner_node: Subtitles

	const BASE_FS := 30
	const MIN_FS := 20
	const MAX_LINES := 3

	func _draw() -> void:
		if owner_node == null or owner_node.words.is_empty():
			return
		var t: float = owner_node._now()
		var line_words: Array = _current_sentence(t)
		if line_words.is_empty():
			return
		var font := get_theme_default_font()
		var vp := get_viewport_rect().size
		var max_w := vp.x * 0.92
		# wrap first, shrink only as a last resort
		var fs := BASE_FS
		var lines := _wrap(line_words, font, fs, max_w)
		while lines.size() > MAX_LINES and fs > MIN_FS:
			fs -= 2
			lines = _wrap(line_words, font, fs, max_w)
		var lh := float(fs) + 12.0
		var hue := _harmonic_hue()
		var bright := Color.from_hsv(hue, 0.55, 1.0)
		var spoken := Color.from_hsv(hue, 0.25, 0.55)
		var waiting := Color(1, 1, 1, 0.28)
		var gap := 14.0
		var cursor: float = owner_node._cursor
		var y: float = vp.y - 70.0 - (lines.size() - 1) * lh
		for row in lines:
			var total := -gap
			for item in row:
				total += item.w + gap
			var x: float = (vp.x - total) * 0.5
			for item in row:
				var w: Dictionary = item.word
				var lit: float = clampf(cursor - float(item.idx), 0.0, 1.0)
				var text: String = w.text
				var pos := Vector2(x, y)
				if lit >= 1.0:
					draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, spoken)
				elif lit <= 0.0:
					draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, waiting)
				else:
					var chars := int(ceil(lit * text.length()))
					draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, waiting)
					draw_string(font, pos, text.substr(0, chars),
						HORIZONTAL_ALIGNMENT_LEFT, -1, fs, bright)
					draw_rect(Rect2(pos.x, pos.y + 8.0, item.w * lit, 3.0), bright)
				x += item.w + gap
			y += lh

	## The current sentence as [{idx (global), word, w (pixel width)}] items.
	func _current_sentence(t: float) -> Array:
		var all: Array = owner_node.words
		var si := -1
		for w in all:
			if t >= float(w.t0) - 0.4 and t < float(w.t1) + 0.4:
				si = int(w.sentence)
				break
		if si < 0:
			for w in all:
				if t < float(w.t0):
					si = int(w.sentence)
					break
		if si < 0:
			return []
		var out: Array = []
		for k in all.size():
			if int(all[k].sentence) == si:
				out.append({"idx": k, "word": all[k], "w": 0.0})
		return out

	## Greedy wrap into rows that fit max_w at the given font size; also fills
	## each item's pixel width.
	func _wrap(items: Array, font: Font, fs: int, max_w: float) -> Array:
		var gap := 14.0
		var lines: Array = []
		var row: Array = []
		var used := -gap
		for item in items:
			item.w = font.get_string_size(item.word.text,
				HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
			if used + gap + item.w > max_w and not row.is_empty():
				lines.append(row)
				row = []
				used = -gap
			row.append(item)
			used += gap + item.w
		if not row.is_empty():
			lines.append(row)
		return lines

	func _harmonic_hue() -> float:
		var sig := Spectrum.harmonic_signature()
		if sig.size() >= 12:
			var best := 0
			for i in 12:
				if sig[i] > sig[best]:
					best = i
			var target := float(best) / 12.0
			var d := fposmod(target - owner_node._hue_sm + 0.5, 1.0) - 0.5
			owner_node._hue_sm = fposmod(owner_node._hue_sm + d * 0.03, 1.0)
		return owner_node._hue_sm
