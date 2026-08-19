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

## THE LINE ONLY EXISTS WHILE THERE IS SOMETHING TO READ.
##
## The sentence to show used to be chosen as "the current one, or else the first one not
## yet spoken", with no bound on how far ahead that was - so through a five second intro,
## before a word had been said, the opening line sat on screen the whole time waiting for
## a voice. The same at the end: nothing brought it off, so it hung through the outro.
##
## [constant LEAD] is how long before its first word a line may appear, and [constant
## HANG] how long after its last it stays. Both are longer than an ordinary gap between
## sentences, deliberately: at a normal seam the next line's window opens before the
## previous one's closes, so the overlay never blinks between sentences. It only leaves
## when the speaking genuinely stops. A pause INSIDE a sentence never brings it off at all,
## however long - see [method span_at], which is the single rule both the plate's ease and
## the drawn text obey.
const LEAD := 0.8
const HANG := 0.7
## Seconds to ease in and out over. The whole overlay eases like everything else here -
## a line that snapped on at full brightness would be the only hard cut in the frame.
const FADE := 0.3

var words: Array = []               # [{text, t0, t1, sentence}] - may still be GROWING
                                    # (a live VoiceStream shares its array by reference)
var loop_length := 0.0              # >0 once a streamed take loops: wrap time by this
var time_base := 0.0                # playback time when the current content started
## 0..1, eased. Multiplies every alpha the overlay draws, plate included.
var presence := 0.0
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


## Sentence SPANS: `[{si, lo, hi}]` in time order, where lo/hi are the first word's start and
## the last word's end - so a span covers a sentence's INTERNAL pauses as well as its words.
## Cached, because the list is walked every frame and can run to thousands of words; rebuilt
## when it changes, which it does word by word while a take streams (a live [VoiceStream] shares
## this array by reference, and [GenerativeEditor] clears it in place between takes).
var _spans: Array = []
var _spans_n := -1
var _spans_tail := -1.0


func _spans_now() -> Array:
	var n := words.size()
	var tail := float(words[n - 1].t1) if n > 0 else -1.0
	if n == _spans_n and is_equal_approx(tail, _spans_tail):
		return _spans
	_spans = []
	var at := {}                       # sentence id -> its position in _spans
	for w in words:
		var si := int(w.sentence)
		if at.has(si):
			var e: Dictionary = _spans[int(at[si])]
			e.lo = minf(float(e.lo), float(w.t0))
			e.hi = maxf(float(e.hi), float(w.t1))
		else:
			at[si] = _spans.size()
			_spans.append({"si": si, "lo": float(w.t0), "hi": float(w.t1)})
	_spans_n = n
	_spans_tail = tail
	return _spans


## THE ONE RULE for what belongs on screen at [param t]: the position in [method _spans_now] of
## the sentence to show, or -1 for nothing.
##
## It is one rule now because TWO of them was the bug. The presence ease asked "is t inside this
## SENTENCE's span, +/- LEAD/HANG"; the draw asked "is t within 0.4 s of a WORD, or within LEAD
## of the next one" - and those disagree for any pause longer than 1.2 s INSIDE a sentence.
## Reported from a ch19 render, at a colon: "at exactly the same place where the colon is, the
## subtitles briefly disappeared, then reappeared... the SAME subtitles flickered". A colon is
## sent to the voice as a colon (see generative_editor._build_chunks), the model answers it with
## a clause pause, and a pause_scale over 1 stretches it past that 1.2 s - so `presence` held at
## 1.0 while the draw had no sentence to draw and returned early, taking the plate with it.
## Measured before the fix: a 1.6 s internal pause went dark 0.45 s in, for a third of a second.
##
## The order of preference is what makes one rule serve both:
##   1. t inside a span - internal pauses included, which is the fix;
##   2. else the NEXT sentence, if it starts within LEAD (it takes over at a seam, because it is
##      the one about to be spoken);
##   3. else the PREVIOUS sentence, if it ended within HANG (a line lingers rather than snapping
##      off, including in a long gap between two sentences);
##   4. else nothing - the speaking has genuinely stopped.
func span_at(t: float) -> int:
	var spans := _spans_now()
	for i in spans.size():
		var e: Dictionary = spans[i]
		if t >= float(e.lo) and t < float(e.hi):
			return i
	var nxt := -1
	var prv := -1
	for i in spans.size():
		if t < float(spans[i].lo):
			nxt = i
			break
		prv = i
	if nxt >= 0 and float(spans[nxt].lo) - t <= LEAD:
		return nxt
	if prv >= 0 and t - float(spans[prv].hi) <= HANG:
		return prv
	return -1


## Should anything be on screen at time [param t], and how strongly? 1 while a sentence is
## selected by [method span_at], 0 otherwise - the same question the draw asks, so the plate and
## the text can never disagree about whether there is anything to read.
func _presence_target(t: float) -> float:
	return 1.0 if span_at(t) >= 0 else 0.0


func _process(delta: float) -> void:
	if words.is_empty():
		presence = 0.0
		return
	var now := _now()
	presence = lerpf(presence, _presence_target(now), 1.0 - exp(-delta / maxf(0.01, FADE)))
	# Snap the tail of the ease to zero. An exponential never actually arrives, and a
	# plate at alpha 0.003 is still a plate - over a dark scene it is invisible, over a
	# bright one it is a faint grey bar sitting at the bottom of the frame for the whole
	# silence, which is the complaint in miniature.
	if presence < 0.004:
		presence = 0.0
	# The CURSOR keeps tracking through all of this, deliberately. It is eased over
	# several seconds, so freezing it while the line is hidden would leave it stale when
	# the next sentence arrives and it would visibly race to catch up on screen.
	var target := _target(now)
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
	## The frame height every hard-coded pixel below is expressed against.
	const REF_H := 1080.0
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
		# Nothing is being read: draw nothing at all, plate included. This is the
		# whole point of `presence` - an empty line still drew its dark plate, so
		# through the intro there was a black bar across the bottom of the frame
		# waiting for words.
		var vis: float = owner_node.presence
		if vis <= 0.01:
			return
		var t: float = owner_node._now()
		var line_words: Array = _current_sentence(t)
		if line_words.is_empty():
			return
		var font := get_theme_default_font()
		var vp := get_viewport_rect().size
		var max_w := vp.x * 0.92
		# EVERY PIXEL NUMBER BELOW IS RELATIVE TO A 1080-TALL FRAME. The export renders at a
		# multiple of the delivered size and lets ffmpeg resolve it back down (that supersample is
		# ghost's only antialiasing - see exporter.QUALITIES), so a subtitle sized in literal
		# pixels came out 1/1.5 as large in the file as it looks in the viewer. Sizing off the
		# viewport instead makes the type occupy the same FRACTION of the frame at any render
		# resolution, which is what "the same size" actually means.
		var k := vp.y / float(REF_H)
		# wrap first, shrink only as a last resort
		var fs := int(round(BASE_FS * k))
		var min_fs := int(round(MIN_FS * k))
		var lines := _wrap(line_words, font, fs, max_w)
		while lines.size() > MAX_LINES and fs > min_fs:
			fs -= maxi(1, int(round(2.0 * k)))
			lines = _wrap(line_words, font, fs, max_w)
		var lh := float(fs) + 12.0 * k
		# the harmonic hue is now the BASE the gradient rides from, not the one
		# colour of the whole line - each glyph turns off it by its position and
		# by time (see _glyph_color)
		var base_hue := _harmonic_hue()
		var now := owner_node._now()
		var gap := 14.0 * k
		# the cursor as a CHARACTER position within this sentence, so the lit
		# front and the lingering trail are both measured in glyphs, not words
		var ccur: float = _char_cursor(line_words)
		var y: float = vp.y - 70.0 * k - (lines.size() - 1) * lh
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
			var pad := 12.0 * k
			var plate := Rect2((vp.x - total) * 0.5 - pad, y - float(fs) - 4.0 * k,
				total + pad * 2.0, lh + 2.0 * k)
			draw_rect(plate, Color(0.04, 0.04, 0.05, 0.72 * vis), true)
			y += lh
		y = vp.y - 70.0 * k - (lines.size() - 1) * lh
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
					col.a *= vis
					# the shadow, under every state - the edge that survives a
					# bright frame bleeding past the plate
					draw_string(font, pos + Vector2(1.5 * k, 1.5 * k), glyph,
						HORIZONTAL_ALIGNMENT_LEFT, -1, fs, Color(0, 0, 0, 0.85 * vis))
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
		# ONE RULE, shared with the presence ease - see Subtitles.span_at for why this is not
		# its own window any more. It used to be "the sentence with a word within 0.4 s, else
		# the first one starting within LEAD", which had a dead zone in the middle of any pause
		# longer than 1.2 s and blinked the line off inside a sentence at a colon.
		var pos: int = owner_node.span_at(t)
		if pos < 0:
			return []
		var si := int(owner_node._spans_now()[pos].si)
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
