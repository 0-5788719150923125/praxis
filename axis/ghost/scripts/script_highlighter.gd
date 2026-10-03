extends SyntaxHighlighter
class_name ScriptHighlighter

## ScriptHighlighter - colors a script by the marks the panel reading it understands.
##
## Driven by [ScriptMarks] patterns rather than rules of its own, so what is colored and
## what the palette offers are the same list. Only the marks of one MODE are colored: in
## the Synthesis panel a `<!-- hesitation -->` is just a note, and is shown as one.
##
## THE WHOLE TEXT IS SCANNED, NOT EACH LINE. Picture descriptions run over several lines and
## a speaker cue is only a cue on a line of its own, neither of which a line can see from
## inside itself. The scan is redone when the text's version moves, and every line's cached
## colors are thrown away with it - [SyntaxHighlighter] only forgets the lines that were
## edited, which leaves the rest of an opened or closed comment showing its old color.

var mode := "generative"

var _rules: Array = []          # [{re: RegEx, color: Color}] in registry (priority) order
var _version := -1
var _lines := {}                # line -> {column: {"color": Color}}


func _init(for_mode := "generative") -> void:
	mode = for_mode
	for k in ScriptMarks.for_mode(mode):
		var re := RegEx.new()
		if re.compile(String(ScriptMarks.REGISTRY[k]["pattern"])) != OK:
			push_warning("ScriptHighlighter: the pattern for '%s' does not compile" % k)
			continue
		_rules.append({"re": re, "color": ScriptMarks.color_of(k)})


## Forget every line's colors. Wired to the text edit's own change signals in
## [method _update_cache], so nothing outside has to remember to call it.
func invalidate() -> void:
	_version = -1
	clear_highlighting_cache()
	var te := get_text_edit()
	if te != null:
		te.queue_redraw()


func _update_cache() -> void:
	var te := get_text_edit()
	if te == null:
		return
	for sig in [te.text_changed, te.text_set]:
		if not (sig as Signal).is_connected(invalidate):
			(sig as Signal).connect(invalidate)
	_version = -1


func _get_line_syntax_highlighting(line: int) -> Dictionary:
	var te := get_text_edit()
	if te == null:
		return {}
	if _version < 0 or te.get_version() != _version:
		_scan(te)
	return _lines.get(line, {})


## Color every character, then cut the result into the per-line column maps TextEdit asks
## for. A line that starts inside a span gets an entry at column 0, or it would draw the
## continuation of a multi-line comment in the default color.
func _scan(te: TextEdit) -> void:
	_version = te.get_version()
	_lines = {}
	var text := te.text
	var n := text.length()
	var owner := PackedInt32Array()
	owner.resize(n)
	owner.fill(-1)
	for ri in _rules.size():
		for m in (_rules[ri]["re"] as RegEx).search_all(text):
			var a := m.get_start()
			var b := m.get_end()
			var free := true
			for i in range(a, b):
				if owner[i] != -1:
					free = false
					break
			if not free:
				continue
			for i in range(a, b):
				owner[i] = ri
	var plain := te.get_theme_color("font_color")
	var line := 0
	var col := 0
	var cur := -2
	var row := {}
	for i in n:
		var o := owner[i]
		if col == 0 or o != cur:
			row[col] = {"color": plain if o == -1 else _rules[o]["color"]}
			cur = o
		if text[i] == "\n":
			_lines[line] = row
			row = {}
			line += 1
			col = 0
			cur = -2
		else:
			col += 1
	_lines[line] = row
