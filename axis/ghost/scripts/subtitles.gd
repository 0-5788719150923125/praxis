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
	# The colour is a GRADIENT keyed on the CHARACTER index across the whole
	# sentence, not the word - so a band of hue drifts through the text spanning
	# several words at once, and the reader can watch it flow rather than catch
	# it word by word. HUE_SPAN sets how tight the band is (how much the hue
	# turns from one glyph to the next); HUE_DRIFT sets how fast the whole band
	# slides forward over time. LINGER is the payload of the request: a spoken
	# glyph does NOT snap back to its resting dim the instant the voice leaves
	# it - it holds its vivid gradient hue and cools over this many characters
	# behind the cursor, so the colour stays long enough to rest the eye on and
	# read what the change meant.
	const HUE_SPAN := 0.011          # hue turned per character (band tightness)
	const HUE_DRIFT := 0.02          # hue slid per second (the band flows)
	const LINGER := 24.0             # characters a spoken glyph stays lit behind the cursor
	# A SECOND channel: SATURATION ebbs and flows in slow bands along the text, at
	# a tighter, differently-timed rhythm than the hue (two incommensurate waves so
	# the pattern never quite repeats). It pulls the colour down toward a grounded,
	# near-grey calm in the valleys and lets it burn full in the peaks - so the line
	# is not a solid rainbow but stable regions with colour activity between them.
	const SAT_SPAN := 0.17           # saturation band spatial frequency (a valley ~every 37 glyphs)
	const SAT_DRIFT := 0.09          # the bands drift per second (their own rhythm)
	const SAT_FLOOR := 0.14          # how far the grounded valleys desaturate (0 = grey)

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
		# the harmonic hue is now the BASE the gradient rides from, not the one
		# colour of the whole line - each glyph turns off it by its position and
		# by time (see _glyph_color)
		var base_hue := _harmonic_hue()
		var now := owner_node._now()
		var gap := 14.0
		# the cursor as a CHARACTER position within this sentence, so the lit
		# front and the lingering trail are both measured in glyphs, not words
		var ccur: float = _char_cursor(line_words)
		var y: float = vp.y - 70.0 - (lines.size() - 1) * lh
		# THE PLATE: scenes range from black voids to white-hot fields, so
		# colour alone can never keep text legible. Each line gets a rounded
		# dark plate sized to its own width (a full-width band would read as
		# broadcast furniture and cover the show), and every glyph is drawn
		# once in near-black underneath - the plate carries most of the
		# contrast, the shadow catches the edges over bright content.
		for row in lines:
			var total := -gap
			for item in row:
				total += item.w + gap
			var pad := 12.0
			var plate := Rect2((vp.x - total) * 0.5 - pad, y - float(fs) - 4.0,
				total + pad * 2.0, lh + 2.0)
			draw_rect(plate, Color(0.04, 0.04, 0.05, 0.72), true)
			y += lh
		y = vp.y - 70.0 - (lines.size() - 1) * lh
		# the pen advances glyph by glyph so the gradient can turn WITHIN a word,
		# and so a spoken glyph keeps its own lingering colour independent of its
		# neighbours - the whole reason to key on characters instead of words
		for row in lines:
			var total := -gap
			for item in row:
				total += item.w + gap
			var x: float = (vp.x - total) * 0.5
			for item in row:
				var text: String = item.word.text
				var ci: int = int(item.cstart)   # this word's first char, sentence-local
				for ch in text.length():
					var glyph := text.substr(ch, 1)
					var cw := font.get_string_size(glyph, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
					var pos := Vector2(x, y)
					var col := _glyph_color(base_hue, ci + ch, ccur, now)
					# the shadow, under every state - the edge that survives a
					# bright frame bleeding past the plate
					draw_string(font, pos + Vector2(1.5, 1.5), glyph,
						HORIZONTAL_ALIGNMENT_LEFT, -1, fs, Color(0, 0, 0, 0.85))
					draw_string(font, pos, glyph, HORIZONTAL_ALIGNMENT_LEFT, -1, fs, col)
					x += cw
				x += gap
			y += lh

	## A single glyph's colour: a hue that drifts by position AND time (the band
	## flowing through the sentence), and a brightness that tells the reading
	## state - a muted preview ahead of the voice, a vivid flare as it is spoken,
	## then a slow cool over LINGER characters behind so the colour stays to be
	## looked at rather than snapping dim the instant the word ends.
	func _glyph_color(base_hue: float, ci: int, ccur: float, t: float) -> Color:
		var hue := fposmod(base_hue + float(ci) * HUE_SPAN - t * HUE_DRIFT, 1.0)
		# the saturation band at this glyph: two incommensurate waves -> organic,
		# non-repeating valleys (grounded) and peaks (colourful). Scales the state's
		# own saturation from a near-grey floor up to full.
		var s1 := sin(float(ci) * SAT_SPAN - t * SAT_DRIFT)
		var s2 := sin(float(ci) * SAT_SPAN * 1.73 + t * SAT_DRIFT * 0.5)
		var sat_env := clampf(0.5 + 0.35 * s1 + 0.15 * s2, 0.0, 1.0)
		var sm := lerpf(SAT_FLOOR, 1.0, sat_env)        # saturation multiplier
		var d := ccur - float(ci)                       # >0 spoken (behind), <=0 waiting (ahead)
		if d <= 0.0:
			# ahead of the voice: dim but present, faintly tinted so the coming
			# colour is previewed rather than a wall of grey
			return Color.from_hsv(hue, 0.22 * sm, 0.6, 0.92)
		# spoken: full flare at the front, cooling to a resting tint over LINGER
		var glow := clampf(1.0 - (d - 1.0) / LINGER, 0.0, 1.0)
		var rest := Color.from_hsv(hue, 0.34 * sm, 0.7)
		var vivid := Color.from_hsv(hue, 0.9 * sm, 1.0)
		return rest.lerp(vivid, glow)

	## The cursor expressed as a CHARACTER position within the current sentence:
	## the word-level eye (owner._cursor) resolved through the per-word character
	## offsets (item.cstart) the layout already carries. Past the last word it
	## reads as the full length so the whole sentence lingers together.
	func _char_cursor(items: Array) -> float:
		if items.is_empty():
			return 0.0
		var cw := int(floor(owner_node._cursor))
		var frac: float = owner_node._cursor - float(cw)
		for it in items:
			if int(it.idx) == cw:
				return float(it.cstart) + frac * float(String(it.word.text).length())
		var last: Dictionary = items[items.size() - 1]
		if cw < int(items[0].idx):
			return 0.0
		return float(last.cstart) + float(String(last.word.text).length()) + 1.0

	## The current sentence as [{idx (global), word, w (pixel width), cstart}]
	## items. cstart is the word's first-character index WITHIN the sentence
	## (words separated by one gap character), so colour can be keyed on the
	## continuous character sequence rather than reset at every word.
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
		var cstart := 0
		for k in all.size():
			if int(all[k].sentence) == si:
				out.append({"idx": k, "word": all[k], "w": 0.0, "cstart": cstart})
				cstart += String(all[k].text).length() + 1   # +1 for the inter-word gap
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
