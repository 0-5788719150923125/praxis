extends CanvasLayer
class_name Subtitles

## Subtitles - the karaoke overlay, session-owned rather than editor-owned.
##
## Draws the current sentence at the bottom of the frame: spoken words settle
## dim, the current word fills bright letter-by-letter against its exact
## synthesis timing, upcoming words wait faint, and the highlight hue rides the
## live harmonic signature (wrap-aware EMA, so it drifts with the harmony
## rather than flickering). Timing comes from a **sidecar JSON** written next to
## a voice take by [SynthEditor] (`take_N.wav` + `take_N.json`), so the same
## overlay works everywhere the take plays: the live synthesis session, a plain
## `--audio take.wav` boot, and - the part that matters for the product - the
## **export render**, where no editor exists. main attaches one to every session
## whose audio has a sidecar; music without one simply gets no overlay.

var words: Array = []               # [{text, t0, t1, sentence}] - may still be GROWING
                                    # (a live VoiceStream shares its array by reference)
var loop_length := 0.0              # >0 once a streamed take loops: wrap time by this
var time_base := 0.0                # playback time when the current content started
                                    # (a stream restarted in place rebases here)
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


func _process(_delta: float) -> void:
	if not words.is_empty():
		_overlay.queue_redraw()


class Overlay:
	extends Control
	var owner_node: Subtitles

	func _draw() -> void:
		if owner_node == null or owner_node.words.is_empty():
			return
		var t: float = Spectrum.current.time - owner_node.time_base
		if t < 0.0:
			return
		if owner_node.loop_length > 0.0:
			t = fmod(t, owner_node.loop_length)
		var line: Array = _current_sentence(t)
		if line.is_empty():
			return
		var font := get_theme_default_font()
		var fs := 30
		var gap := 14.0
		var widths: Array = []
		var total := -gap
		for w in line:
			var wd: float = font.get_string_size(w.text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
			widths.append(wd)
			total += wd + gap
		var vp := get_viewport_rect().size
		while total > vp.x * 0.92 and fs > 14:
			fs -= 2
			total = -gap
			for k in line.size():
				widths[k] = font.get_string_size(line[k].text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
				total += widths[k] + gap
		var x: float = (vp.x - total) * 0.5
		var y: float = vp.y - 70.0
		var hue := _harmonic_hue()
		var bright := Color.from_hsv(hue, 0.55, 1.0)
		var spoken := Color.from_hsv(hue, 0.25, 0.55)
		var waiting := Color(1, 1, 1, 0.28)
		for k in line.size():
			var w: Dictionary = line[k]
			var pos := Vector2(x, y)
			var t0 := float(w.t0)
			var t1 := float(w.t1)
			if t >= t1:
				draw_string(font, pos, w.text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, spoken)
			elif t < t0:
				draw_string(font, pos, w.text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, waiting)
			else:
				# the fill: bright prefix by elapsed fraction of the word's own span
				var frac: float = (t - t0) / maxf(0.001, t1 - t0)
				var chars := int(ceil(frac * String(w.text).length()))
				draw_string(font, pos, w.text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, waiting)
				draw_string(font, pos, String(w.text).substr(0, chars), HORIZONTAL_ALIGNMENT_LEFT, -1, fs, bright)
				draw_rect(Rect2(pos.x, y + 8.0, widths[k] * frac, 3.0), bright)
			x += widths[k] + gap

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
		return all.filter(func(w): return int(w.sentence) == si)

	## Dominant chroma bin of the live harmonic signature -> a hue, EMA-smoothed
	## on the hue circle.
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
