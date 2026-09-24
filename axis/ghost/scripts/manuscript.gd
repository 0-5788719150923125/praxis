extends RefCounted
class_name Manuscript

## Manuscript - what a chapter file SAYS about itself, beyond the words to be read.
##
## A chapter carries three kinds of authoring mark, all as HTML comments so the book build
## never prints them:
##
##     <!-- speaker: Emily White -->     a line of its own; hands the text after it to a voice
##     <!-- hesitation -->               a longer rest here, for effect (anywhere, even mid-line)
##     <!-- hesitation: 2.5 -->          ...the same, for exactly this many seconds
##     <!-- image: a wolf in a coat -->  an illustration, described; placed where it sits
##
## ONE PARSER, SEVERAL READERS. The Generative panel asks it for the speakers (its tabs) and
## the hesitations (its rests); the book medium asks it for the page blocks; the illustration
## library asks it for the image prompts. Each used to be a regex in whichever file needed it,
## and three copies of "what is a speaker cue" is how a cue gets read by one and missed by
## another - a tab with no voice, or a voice with no tab.
##
## SPEAKERS ARE NAMES, NOT NUMBERS. `<!-- speaker: 2 -->` still parses - "2" is simply a name -
## but a reader of the prose cannot tell who speaker 2 is, and the author writes the prose.

## A cue owns its line, for the reason the Generative panel gives: prose that mentions a
## speaker must never be mistaken for one, and the failure is silent.
const SPEAKER := "^\\s*(?:<!--\\s*speaker\\s*:\\s*(.+?)\\s*-->|\\[\\s*speaker\\s*:\\s*([^\\]]+?)\\s*\\])\\s*$"
## A hesitation may sit anywhere - "She comes all the way around, <!-- hesitation --> and stops".
## The optional number is seconds.
const HESITATION := "<!--\\s*hesitation\\s*(?::\\s*([0-9]*\\.?[0-9]+)\\s*(?:s|secs?|seconds)?)?\\s*-->|\\[\\s*hesitation\\s*(?::\\s*([0-9]*\\.?[0-9]+)\\s*(?:s|secs?|seconds)?)?\\s*\\]"
## An image. The optional `(full)`, `(inline)`, `(left)` or `(right)` pins its placement;
## without one it is decided from where it sits (see [method _auto_placement]).
const IMAGE := "<!--\\s*image\\s*(?:\\(\\s*(full|inline|left|right)\\s*\\))?\\s*:\\s*([\\s\\S]*?)\\s*-->"
## Any other comment is an authoring note.
const COMMENT := "<!--[\\s\\S]*?-->"
## Text before the first cue belongs to this voice. A chapter that opens on a cue never has one.
const NARRATOR := "Narrator"

static var _re := {}


static func _rx(pattern: String) -> RegEx:
	if not _re.has(pattern):
		var r := RegEx.new()
		r.compile(pattern)
		_re[pattern] = r
	return _re[pattern]


## The body without its YAML frontmatter. Only the FIRST block, and only at the very top
## (blank lines before it allowed), so a `---` rule mid-chapter is left alone - the same rule
## the Generative panel has always used.
static func strip_frontmatter(body: String) -> String:
	var lines := body.split("\n")
	var i := 0
	while i < lines.size() and String(lines[i]).strip_edges().is_empty():
		i += 1
	if i >= lines.size() or String(lines[i]).strip_edges() != "---":
		return body
	for j in range(i + 1, lines.size()):
		if String(lines[j]).strip_edges() == "---":
			return "\n".join(lines.slice(j + 1))
	return body          # unterminated: not frontmatter, the whole thing is text


## The speaker a line hands over to, or "" when the line is not a cue.
static func speaker_of_line(line: String) -> String:
	var m := _rx(SPEAKER).search(line)
	if m == null:
		return ""
	var name := m.get_string(1) if not m.get_string(1).is_empty() else m.get_string(2)
	return name.strip_edges()


## Every speaker the script uses, in order of first appearance. [constant NARRATOR] leads the
## list when there is anything to read before the first cue - and only then, so a chapter that
## opens on `<!-- speaker: Ryan -->` has no empty narrator tab.
static func speakers(body: String) -> PackedStringArray:
	var out := PackedStringArray()
	var text_before := false
	var seen_cue := false
	for line in strip_frontmatter(body).split("\n"):
		var who := speaker_of_line(String(line))
		if who.is_empty():
			if not seen_cue and not _is_blank_or_note(String(line)):
				text_before = true
			continue
		if not seen_cue and text_before:
			out.append(NARRATOR)
		seen_cue = true
		if not out.has(who):
			out.append(who)
	if not seen_cue:
		out.append(NARRATOR)
	return out


## Nothing to read on this line: blank, or nothing but comments.
static func _is_blank_or_note(line: String) -> bool:
	return _rx(COMMENT).sub(line, "", true).strip_edges().is_empty()


## The script cut into passages by speaker: `[{speaker, text}]`, in order, with the cues
## consumed. Text before the first cue is the narrator's. Comments are NOT stripped here -
## a hesitation is a comment and its position matters to the caller.
static func passages(body: String) -> Array:
	var out: Array = []
	var who := NARRATOR
	var buf := PackedStringArray()
	for line in strip_frontmatter(body).split("\n"):
		var next := speaker_of_line(String(line))
		if next.is_empty():
			buf.append(String(line))
			continue
		if next == who:
			continue
		out.append({"speaker": who, "text": "\n".join(buf)})
		buf = PackedStringArray()
		who = next
	out.append({"speaker": who, "text": "\n".join(buf)})
	return out


## Every hesitation in [param text]: `[{at, len, seconds}]`, ascending, where `seconds` is
## the author's own figure or -1 for "the panel's default".
static func hesitations(text: String) -> Array:
	var out: Array = []
	for m in _rx(HESITATION).search_all(text):
		var v := m.get_string(1) if not m.get_string(1).is_empty() else m.get_string(2)
		out.append({"at": m.get_start(), "len": m.get_end() - m.get_start(),
			"seconds": float(v) if not v.is_empty() else -1.0})
	return out


## A stable key for an image description: whitespace-folded and hashed, so re-wrapping a long
## comment in an editor does not turn it into a different picture.
static func image_key(prompt: String) -> String:
	var folded := " ".join(prompt.strip_edges().split(" ", false)).replace("\n", " ")
	folded = " ".join(folded.split(" ", false))
	return folded.sha256_text().substr(0, 16)


## Every image in the script, in reading order: `[{prompt, key, placement, side, ordinal}]`.
## `placement` is "full" (a page of its own) or "inline" (text wraps around it); `side` is
## "left" or "right" - for an inline picture the edge it floats against, for a full one the
## page of the spread it would rather sit on.
static func images(body: String) -> Array:
	var out: Array = []
	for b in blocks(body):
		if String((b as Dictionary)["kind"]) == "image":
			out.append(b)
	return out


## The chapter as a sequence of page BLOCKS, for a medium that typesets it:
##
##     {kind: "heading", level, text}
##     {kind: "para", text, speaker}      - inline markdown kept (*italic*, **bold**);
##                                          hesitation marks kept (the page ignores them,
##                                          they are here so offsets line up with speech)
##     {kind: "image", prompt, key, placement, side, ordinal}
##     {kind: "rule"}
##
## Speaker cues are consumed (the page does not print them) but every paragraph records whose
## voice reads it. Authoring notes other than hesitations are removed.
static func blocks(body: String) -> Array:
	var out: Array = []
	var who := NARRATOR
	var para: Array = []
	var text := strip_frontmatter(body)
	# Images first, as whole units: a description may run over several lines, and a line
	# scanner would cut it in half. Each is replaced by a placeholder line of its own.
	var imgs: Array = []
	var img_re := _rx(IMAGE)
	var cursor := 0
	var flat := ""
	for m in img_re.search_all(text):
		flat += text.substr(cursor, m.get_start() - cursor)
		flat += "\n\uE0F0IMG%d\uE0F0\n" % imgs.size()
		imgs.append({"prompt": m.get_string(2).strip_edges(), "pin": m.get_string(1)})
		cursor = m.get_end()
	flat += text.substr(cursor)

	for raw in flat.split("\n"):
		var line := String(raw)
		var cue := speaker_of_line(line)
		if not cue.is_empty():
			_flush(out, para, who)
			who = cue
			continue
		var s := line.strip_edges()
		if s.begins_with("\uE0F0IMG") and s.ends_with("\uE0F0"):
			_flush(out, para, who)
			var n := int(s.substr(4, s.length() - 5))
			var im: Dictionary = imgs[n]
			out.append({"kind": "image", "prompt": String(im["prompt"]),
				"key": image_key(String(im["prompt"])), "pin": String(im["pin"]),
				"ordinal": n})
			continue
		if s.is_empty():
			_flush(out, para, who)          # a blank line ends a paragraph
			continue
		# Notes go; hesitations stay, so a paragraph's text still lines up with what is
		# spoken. A line holding nothing but a note neither adds text nor ends a paragraph.
		var kept := _strip_notes(line)
		if kept.strip_edges().is_empty():
			continue
		if s == "---" or s == "***" or s == "* * *":
			_flush(out, para, who)
			out.append({"kind": "rule"})
			continue
		if s.begins_with("#"):
			_flush(out, para, who)
			var level := 0
			while level < s.length() and s[level] == "#":
				level += 1
			out.append({"kind": "heading", "level": level, "text": s.substr(level).strip_edges()})
			continue
		para.append(kept.strip_edges())
	_flush(out, para, who)
	_place_images(out)
	return out


## End the paragraph being gathered. A paragraph of nothing but a hesitation is a beat
## between paragraphs - it is heard, and there is nothing on it to print.
static func _flush(out: Array, para: Array, who: String) -> void:
	if para.is_empty():
		return
	var t := " ".join(PackedStringArray(para)).strip_edges()
	para.clear()
	if not _rx(HESITATION).sub(t, "", true).strip_edges().is_empty():
		out.append({"kind": "para", "text": t, "speaker": who})


## Remove every comment that is not a hesitation.
static func _strip_notes(line: String) -> String:
	var res := ""
	var at := 0
	for m in _rx(COMMENT).search_all(line):
		res += line.substr(at, m.get_start() - at)
		if _rx(HESITATION).search(m.get_string()) != null:
			res += m.get_string()
		at = m.get_end()
	return res + line.substr(at)


## WHERE A PICTURE GOES, decided by where the author put it.
##
## FULL PAGE when it opens something: the chapter's first block, or an image followed (past
## any other images) by a SCENE LINE - a short paragraph set entirely in italics, which is how
## this manuscript announces a change of time or place ("*Nine-thirty. Orientation, session
## four.*"). That is an establishing shot, and a book gives an establishing shot a page.
## INLINE everywhere else: a picture in the middle of a scene belongs beside the lines it
## illustrates, with the text wrapping round it. Inline pictures alternate edges, starting on
## the right, so two in a row do not stack down one margin. A pin in the marker overrides all
## of this.
static func _place_images(blocks_out: Array) -> void:
	var float_right := true
	var full_right := true
	var seen_text := false
	for i in blocks_out.size():
		var b: Dictionary = blocks_out[i]
		var kind := String(b["kind"])
		if kind == "para" or kind == "heading":
			seen_text = true
			continue
		if kind != "image":
			continue
		var pin := String(b.get("pin", ""))
		var full := false
		if pin == "full":
			full = true
		elif pin.is_empty():
			full = not seen_text or _opens_scene(blocks_out, i)
		b["placement"] = "full" if full else "inline"
		if full:
			b["side"] = "right" if full_right else "left"
			full_right = not full_right
		elif pin == "left" or pin == "right":
			b["side"] = pin
			float_right = pin == "left"
		else:
			b["side"] = "right" if float_right else "left"
			float_right = not float_right
		b.erase("pin")


static func _opens_scene(blocks_out: Array, i: int) -> bool:
	for j in range(i + 1, blocks_out.size()):
		var b: Dictionary = blocks_out[j]
		var kind := String(b["kind"])
		if kind == "image":
			continue
		if kind == "heading":
			return true
		if kind != "para":
			return false
		return is_scene_line(String(b["text"]))
	return false


## A paragraph that is one short italic run: `*Seven-thirty in the morning.*`
static func is_scene_line(text: String) -> bool:
	var t := _rx(HESITATION).sub(text, "", true).strip_edges()
	if t.length() < 3 or t.length() > 160:
		return false
	var single := t.begins_with("*") and t.ends_with("*") and not t.begins_with("**") \
		and t.count("*") == 2
	var under := t.begins_with("_") and t.ends_with("_") and t.count("_") == 2
	return single or under
