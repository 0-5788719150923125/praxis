extends Node

## Gate for EMPHASIS surviving the pipeline - italic and bold reaching the reader's eyes.
##
## Markdown emphasis used to be deleted outright by [TextNorm._strip_markdown], for a good
## reason: a marker is not silent to a phonemizer, it is a word, and eSpeak reads
## `*I will never hurt you*` as "ASTERISK I will never hurt you ASTERISK". That fixed the
## mouth and broke the page - the subtitle under an emphasised line came out as flat prose
## with nothing anywhere to say the writing had leaned on those words. Reported as
## "ghost strips quotes and italics/bold in the subtitles"; the quotes had already been
## fixed by the time it was measured, the emphasis had not.
##
## THE TWO HALVES ARE BOTH REQUIRED, and they pull against each other:
##   the SPOKEN text must be byte-identical to what the old outright-delete produced -
##   nothing about the reading may change, and a sentinel that leaked into a dictionary
##   key or a phoneme list would change it silently;
##   the SHOWN text must carry the level, with the markers themselves gone.
## So every case below asserts both: the level on each word, and that
## [method TextNorm.normalize] still hands back plain speakable words.
##
## The last block is about the RENDERER rather than the text: an emphasis level nothing
## draws differently is the same bug wearing a data structure. The faces are synthesized
## from the theme font by [FontVariation], so the gate measures them - a bold face must
## actually be wider than the plain one, or `variation_embolden` is not doing anything.
##
## Runs inside a real boot, because the last block reaches the overlay itself and
## [Subtitles] reads the Spectrum autoload:
##   tests/run_boot_probe.sh tests/subtitle_emphasis_check.gd 60

const TextNorm_ := preload("res://scripts/text_norm.gd")
const Phonemes_ := preload("res://scripts/phonemes.gd")

# [source, [[display, level], ...]] - one entry per word the parse should produce,
# flattened across sentences. A level is 0 plain, 1 italic, 2 bold, 3 both.
const CASES := [
	["He whispered, *I will never hurt you*, and left.",
		[["He", 0], ["whispered,", 0], ["I", 1], ["will", 1], ["never", 1], ["hurt", 1],
		 ["you,", 1], ["and", 0], ["left.", 0]]],
	["That was **absolutely** the point.",
		[["That", 0], ["was", 0], ["absolutely", 2], ["the", 0], ["point.", 0]]],
	["She said _no one_ was coming.",
		[["She", 0], ["said", 0], ["no", 1], ["one", 1], ["was", 0], ["coming.", 0]]],
	["***Everything***, then.",
		[["Everything,", 3], ["then.", 0]]],
	# nesting: the inner pair closes against the inner opener, and the sentinels being
	# toggles is what lets the word between carry both levels at once
	["**bold with *italic* inside** after.",
		[["bold", 2], ["with", 2], ["italic", 3], ["inside", 2], ["after.", 0]]],
	# THE WRAPPERS STAY. A quote is not markup - the reported bug named quotes first, and
	# an emphasis marker sitting inside one must not take it with it.
	["\"*Now.*\"", [["\"Now.\"", 1]]],
	# ...and emphasis must not defeat the passes that run after it. `Mr.` is matched by an
	# abbreviation table that strips wrappers off a token, and `2009` by a numeral pass
	# whose rewritten run carries its own source spelling.
	["*Mr. Smith* arrived.", [["Mr.", 1], ["Smith", 1], ["arrived.", 0]]],
	# things that are NOT emphasis, and must survive whole
	["snake_case and __init__ stay whole.",
		[["snake_case", 0], ["and", 0], ["__init__", 0], ["stay", 0], ["whole.", 0]]],
	["A lone _private and a bare * asterisk.",
		[["A", 0], ["lone", 0], ["_private", 0], ["and", 0], ["a", 0], ["bare", 0],
		 ["asterisk.", 0]]],
]

# What the mouth gets, which may not change by one character. Each pair is
# [source, exactly what normalize must return].
const SPOKEN := [
	["He whispered, *I will never hurt you*, and left.",
		"He whispered, I will never hurt you, and left."],
	["That was **absolutely** the point.", "That was absolutely the point."],
	["She said _no one_ was coming.", "She said no one was coming."],
	["\"*Now.*\"", "\"Now.\""],
	["*Mr. Smith* arrived.", "mister Smith arrived."],
	["In *2009* it ended.", "In two thousand nine it ended."],
	["snake_case and __init__ stay whole.", "snake_case and __init__ stay whole."],
	# a bullet list: the marker is followed by a space, so it is not emphasis and is
	# dropped as it always was
	["* a bullet\n* another", "a bullet\n another"],
]

var _fails: Array = []


func _ready() -> void:
	_check_levels()
	_check_spoken()
	_check_no_sentinels()
	_check_faces()
	if _fails.is_empty():
		print("subtitle_emphasis_check: ALL OK (%d level cases, %d spoken cases)"
			% [CASES.size(), SPOKEN.size()])
	else:
		for f in _fails:
			printerr("subtitle_emphasis_check: " + String(f))
		printerr("subtitle_emphasis_check: %d FAILURE(S)" % _fails.size())
	get_tree().quit(0 if _fails.is_empty() else 1)


func _check_levels() -> void:
	for c in CASES:
		var src := String(c[0])
		var want: Array = c[1]
		var got: Array = []
		for sent in Phonemes_.parse(src):
			for w in sent:
				var shown := String((w as Dictionary).get("display", ""))
				if shown.is_empty():
					continue          # a continuation of a rewritten run draws nothing
				got.append([shown.strip_edges(), int((w as Dictionary).get("emph", 0))])
		if got.size() != want.size():
			_fails.append("%s: %d words, expected %d - got %s" % [src, got.size(),
				want.size(), str(got)])
			continue
		for i in want.size():
			var g: Array = got[i]
			var e: Array = want[i]
			if String(g[0]) != String(e[0]) or int(g[1]) != int(e[1]):
				_fails.append("%s: word %d is %s/%d, expected %s/%d"
					% [src, i, g[0], int(g[1]), e[0], int(e[1])])


func _check_spoken() -> void:
	for c in SPOKEN:
		var got := TextNorm_.normalize(String(c[0]))
		if got != String(c[1]):
			_fails.append("spoken text changed: %s -> %s, expected %s"
				% [c[0], got, c[1]])


## NO SENTINEL MAY REACH A PHONEME. This is the failure mode that would be silent: a
## private-use character in a dictionary key is a miss, not an error, and the word would
## simply come out mispronounced by the letter rules with nothing logged.
func _check_no_sentinels() -> void:
	var src := "**Absolutely** *not*, said *Mr. Smith* in *2009*."
	for sent in Phonemes_.parse(src):
		for w in sent:
			var d: Dictionary = w
			for field in ["text", "display"]:
				var v := String(d.get(field, ""))
				if v.contains(TextNorm_.EMPH_ITALIC) or v.contains(TextNorm_.EMPH_BOLD):
					_fails.append("a sentinel survived into %s: %s" % [field, v])
			for p in (d.get("phones", []) as Array):
				var ph := String(p)
				if ph.contains(TextNorm_.EMPH_ITALIC) or ph.contains(TextNorm_.EMPH_BOLD):
					_fails.append("a sentinel survived into a phoneme: %s" % ph)


## The faces the OVERLAY draws with - asked of the overlay itself, not of a FontVariation
## built here, which would only prove that Godot works.
##
## A level nothing draws differently is the same bug wearing a data structure, so this
## MEASURES THE GLYPH rather than inspecting the properties. It has to go through
## [TextServer] to do it: `get_string_size` reads advances, and neither variation moves an
## advance much - a shear moves none at all - so a width check would sail straight past a
## transform that never took. The rendered outline of `H` is the thing that changed.
func _check_faces() -> void:
	var ov := Subtitles.Overlay.new()
	var base := ThemeDB.fallback_font
	if base == null:
		_fails.append("no fallback font - cannot measure the emphasis faces")
		ov.free()
		return
	var box := {}
	for lvl in [0, 1, 2, 3]:
		var face: Font = ov._face(base, lvl)
		if lvl > 0 and face == base:
			_fails.append("level %d got the plain face back" % lvl)
			continue
		box[lvl] = _glyph_box(face, "H")
	print("subtitle_emphasis_check: glyph H - plain %v, italic %v, bold %v, both %v"
		% [box.get(0, Vector2.ZERO), box.get(1, Vector2.ZERO),
			box.get(2, Vector2.ZERO), box.get(3, Vector2.ZERO)])
	var plain: Vector2 = box.get(0, Vector2.ZERO)
	# the slant leans the glyph, so its box gets WIDER while its height is untouched
	if float((box.get(1, Vector2.ZERO) as Vector2).x) <= plain.x:
		_fails.append("the slant did nothing: italic H is %.0f px wide, plain %.0f"
			% [(box.get(1, Vector2.ZERO) as Vector2).x, plain.x])
	# the embolden thickens the outline, so the box grows in BOTH directions
	var bold: Vector2 = box.get(2, Vector2.ZERO)
	if bold.x <= plain.x or bold.y <= plain.y:
		_fails.append("emboldening did nothing: bold H is %v, plain %v" % [bold, plain])
	# and the two are independent - level 3 must be leaned AND thickened
	var both: Vector2 = box.get(3, Vector2.ZERO)
	if both.x <= bold.x or both.y <= plain.y:
		_fails.append("level 3 is not both: %v against italic %v and bold %v"
			% [both, box.get(1, Vector2.ZERO), bold])
	ov.free()


## The rendered bounding box of one character in `face`, in pixels at 30 pt.
func _glyph_box(face: Font, ch: String) -> Vector2:
	var ts := TextServerManager.get_primary_interface()
	var rids := face.get_rids()
	if rids.is_empty():
		return Vector2.ZERO
	var rid: RID = rids[0]
	var gid: int = ts.font_get_glyph_index(rid, 30, ch.unicode_at(0), 0)
	return ts.font_get_glyph_size(rid, Vector2i(30, 0), gid)
