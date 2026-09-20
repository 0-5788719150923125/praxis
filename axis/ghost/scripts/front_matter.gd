extends RefCounted
class_name FrontMatter

## FrontMatter - YAML frontmatter on a Markdown document, read and (carefully) written.
##
## WHAT IT IS FOR. A reading's script lives in a real file the author is editing in their
## own editor, not in ghost's text box (see [DocSource]). Two things then have to pass
## between the document and the panel:
##
##   THE BODY, which is everything after the frontmatter. Frontmatter is metadata for
##   whatever else reads the document - a title, a date, tags - and reading it aloud would
##   be nonsense, so [method split] cuts it off before a word reaches the voice.
##
##   THE VOICE, which ghost writes back INTO the frontmatter under one key of its own.
##   Before this, every dial - the reader, the tone, the room, the whole cast of a
##   multi-speaker chapter - lived in ghost's own settings file, which holds exactly one
##   of them; opening a second document silently inherited the first one's voice, and
##   returning to the first had lost it. A document that carries its own voice is a
##   document that sounds the same tomorrow.
##
## THE WRITE IS THE DANGEROUS HALF, because the file is the author's work and ghost is not
## its editor. Four rules, all enforced in [method write_block]:
##
##   1. NEVER FROM A CACHE. The file is re-read from disk immediately before the edit, so
##      what is written back is the document as it is now, not as it was when it was
##      opened. The author is expected to be typing in it off-screen; that is the point.
##   2. ONE KEY, TEXTUALLY. The replacement is line surgery on our own top-level key -
##      nothing else in the frontmatter is re-serialized, so a comment, a quoting style,
##      a key order or a construct this file cannot even parse survives untouched.
##   3. VERIFY BEFORE COMMITTING. [method _verify] re-splits the proposed text and
##      requires the body to be byte-identical and every foreign frontmatter line to be
##      unchanged. A mismatch aborts with an error rather than writing.
##   4. ATOMIC. The new text goes to a sibling temp file, is read back, and only then is
##      renamed over the original - so an interrupted write cannot leave a half-file.
##
## Nothing here touches the scene tree, so [code]tests/doc_source_check.gd[/code] can hold
## all of it to account headless.

const MiniYaml_ := preload("res://scripts/yaml.gd")

## The one top-level frontmatter key ghost owns. Everything it writes lives under this and
## nothing outside it is ever touched.
const KEY := "ghost"
## Suffix for the temp file rule 4 renames from. Beside the original, because a rename is
## only atomic within one filesystem.
const TEMP_SUFFIX := ".ghost-tmp"


## Cut [param raw] into its frontmatter and its body.
##
## STRICT ON PURPOSE. A document has frontmatter only when its very first line is exactly
## `---` and a later line is exactly `---` or `...`; an unterminated opening fence is a
## document with no frontmatter, not a document that is all frontmatter. The looser rule -
## "find a `---` anywhere" - would eat the prose above a horizontal rule.
##
## Returns [code]{has, head, body, open, close, nl, bom}[/code]: [code]head[/code] is the
## frontmatter WITHOUT its fences, [code]body[/code] everything after the closing one,
## [code]open[/code]/[code]close[/code] the fence line indices (-1 when there is none),
## [code]nl[/code] the document's line ending, [code]open_fence[/code]/[code]fence[/code]
## the two fence lines VERBATIM (`...` closes frontmatter too, and neither is ours to
## rewrite) and [code]bom[/code] whether it opened with a byte-order mark.
static func split(raw: String) -> Dictionary:
	var nl := "\r\n" if raw.contains("\r\n") else "\n"
	var bom := raw.begins_with("﻿")
	var text := raw.substr(1) if bom else raw
	var lines := _lines(text)
	var out := {"has": false, "head": "", "body": text, "open": -1, "close": -1,
		"nl": nl, "bom": bom, "open_fence": "---", "fence": "---"}
	if lines.is_empty() or lines[0].strip_edges() != "---":
		return out
	for i in range(1, lines.size()):
		var t: String = lines[i].strip_edges()
		if t == "---" or t == "...":
			out.has = true
			out.open = 0
			out.close = i
			out.open_fence = String(lines[0])
			out.fence = String(lines[i])
			out.head = nl.join(lines.slice(1, i))
			out.body = nl.join(lines.slice(i + 1))
			return out
	return out


## Everything the document says about the voice: the value of our own key, or an empty
## dictionary when there is none.
##
## ONE KEY, TEXTUALLY - the reading rule is the mirror of rule 2, and for the same reason.
## [MiniYaml] reads a documented subset and rejects the rest loudly, which is right for a
## storyboard ghost owns and wrong for a document it is a guest in: `summary: |` is ordinary
## frontmatter and a block scalar is outside the subset, so parsing the WHOLE head would
## mean a voice could never be restored from a perfectly normal document. Only our own key's
## lines are cut out and parsed, so nothing anywhere else in the frontmatter can reach this.
##
## What is left to fail is our own block being malformed - hand-edited into something the
## parser refuses. That is reported rather than raised, so the panel can say why no voice
## was restored and the reading goes ahead regardless.
##
## Returns [code]{ok, data, error}[/code].
static func read_block(raw: String, key := KEY) -> Dictionary:
	var parts := split(raw)
	if not parts.has or String(parts.head).strip_edges().is_empty():
		return {"ok": true, "data": {}, "error": ""}
	var head := _lines(String(parts.head))
	var span := _span_of(head, key)
	if span.is_empty():
		return {"ok": true, "data": {}, "error": ""}
	var res := MiniYaml_.parse("\n".join(head.slice(int(span[0]), int(span[1]))))
	if not res.ok:
		return {"ok": false, "data": {}, "error": String(res.error)}
	var data: Variant = res.data
	if not (data is Dictionary):
		return {"ok": false, "data": {}, "error": "`%s:` is not a mapping" % key}
	var v: Variant = (data as Dictionary).get(key, null)
	if v == null:
		return {"ok": true, "data": {}, "error": ""}
	if not (v is Dictionary):
		return {"ok": false, "data": {}, "error": "`%s:` in the frontmatter is not a mapping" % key}
	return {"ok": true, "data": v as Dictionary, "error": ""}


## The document with [param value] stored under [param key], and NOTHING else different.
##
## Pure: it takes text and returns text, so the gate can check the guarantee on documents
## that were never on a disk. [method write_block] is the half that touches files.
##
## Three cases, in the order they are tried: our key is already there and its block is
## replaced in place; the document has frontmatter without our key and the block is
## appended to it; the document has no frontmatter at all and a fresh one is opened above
## the body.
static func put_block(raw: String, value: Variant, key := KEY) -> String:
	var parts := split(raw)
	var nl := String(parts.nl)
	var block := _block_lines(value, key)
	var prefix := "﻿" if parts.bom else ""
	if not parts.has:
		# A fresh frontmatter, and a blank line under it so the first line of the
		# document is not welded to the closing fence.
		var body := String(parts.body)
		var lead: Array = ["---"] + block + ["---"]
		if not body.is_empty():
			lead.append("")
		return prefix + nl.join(lead) + (nl + body if not body.is_empty() else nl)
	var head := _lines(String(parts.head))
	var span := _span_of(head, key)
	if span.is_empty():
		# Appended rather than inserted: an author's own keys stay where they were put,
		# and ghost's block is recognisably the machine-written one at the bottom.
		head.append_array(block)
	else:
		var tail: Array = head.slice(int(span[1]))
		head = head.slice(0, int(span[0]))
		head.append_array(block)
		head.append_array(tail)
	var lines := _lines(String(parts.body))
	var out: Array = [String(parts.open_fence)] + head + [String(parts.fence)]
	out.append_array(lines)
	return prefix + nl.join(out)


## Store [param value] in the frontmatter of the file at [param path], on disk.
##
## The four rules at the top of this file, in order. Returns "" on success and a
## human-readable reason on failure - and on failure the file has not been touched.
static func write_block(path: String, value: Variant, key := KEY) -> String:
	if path.is_empty():
		return "no document is open"
	if not FileAccess.file_exists(path):
		return "there is no file at %s" % path
	# RULE 1: from the disk, now. Never a copy the panel has been holding.
	var fh := FileAccess.open(path, FileAccess.READ)
	if fh == null:
		return "could not read %s (error %d)" % [path, FileAccess.get_open_error()]
	var before := fh.get_as_text()
	fh.close()
	# RULES 2 and 3: line surgery on one key, then prove nothing else moved.
	var after := put_block(before, value, key)
	if after == before:
		return ""
	var bad := _verify(before, after, key)
	if not bad.is_empty():
		return bad
	# RULE 4: write beside, read back, rename over.
	var tmp := path + TEMP_SUFFIX
	var out := FileAccess.open(tmp, FileAccess.WRITE)
	if out == null:
		return "could not write beside %s (error %d)" % [path, FileAccess.get_open_error()]
	out.store_string(after)
	out.close()
	var check := FileAccess.open(tmp, FileAccess.READ)
	var wrote := check.get_as_text() if check != null else ""
	if check != null:
		check.close()
	if wrote != after:
		DirAccess.remove_absolute(tmp)
		return "the temporary file did not read back as written - %s is untouched" % path.get_file()
	var err := DirAccess.rename_absolute(tmp, path)
	if err != OK:
		DirAccess.remove_absolute(tmp)
		return "could not replace %s (error %d)" % [path.get_file(), err]
	return ""


## RULE 3, on its own so the gate can call it: what [param after] destroys, if anything.
##
## The body must be identical BYTE FOR BYTE - not trimmed, not normalized - and every
## frontmatter line outside our key's span must survive in order. Anything else returns
## the reason, and the reason is what the panel shows instead of writing.
static func _verify(before: String, after: String, key: String) -> String:
	var a := split(before)
	var b := split(after)
	if not b.has:
		return "the edit did not produce frontmatter - refusing to write"
	if String(a.body) != String(b.body):
		return "the edit would have changed the document body - refusing to write"
	if a.has and (String(a.fence) != String(b.fence) \
			or String(a.open_fence) != String(b.open_fence)):
		return "the edit would have changed a frontmatter fence - refusing to write"
	var old_head := _lines(String(a.head)) if a.has else []
	var new_head := _lines(String(b.head))
	var kept_old := _without(old_head, _span_of(old_head, key))
	var kept_new := _without(new_head, _span_of(new_head, key))
	if kept_old != kept_new:
		return "the edit would have changed another frontmatter key - refusing to write"
	return ""


## The half-open line range [code][from, to)[/code] that [param key]'s block occupies in
## [param head], or [] when the key is not there.
##
## THE END OF A BLOCK is the next line at column zero that is not blank - a sibling key.
## Blank lines and comments trailing our block belong to whatever follows it, so the span
## is then pulled back to the last line that actually carries content; without that, a
## blank line an author left between two keys would be eaten by every save.
static func _span_of(head: Array, key: String) -> Array:
	var start := -1
	for i in head.size():
		var line := String(head[i])
		if line.begins_with(" ") or line.begins_with("\t") or line.strip_edges().is_empty():
			continue
		if start >= 0:
			var last := i
			while last > start and String(head[last - 1]).strip_edges().is_empty():
				last -= 1
			return [start, last]
		if _key_of(line) == key:
			start = i
	if start < 0:
		return []
	var end := head.size()
	while end > start + 1 and String(head[end - 1]).strip_edges().is_empty():
		end -= 1
	return [start, end]


## The key a top-level frontmatter line declares, or "" when it declares none. Quoted
## keys are unwrapped, because `"ghost":` and `ghost:` are the same key.
static func _key_of(line: String) -> String:
	var colon := line.find(":")
	if colon < 0:
		return ""
	var k := line.substr(0, colon).strip_edges()
	if k.length() >= 2 and ((k.begins_with("\"") and k.ends_with("\"")) \
			or (k.begins_with("'") and k.ends_with("'"))):
		k = k.substr(1, k.length() - 2)
	return k


static func _without(head: Array, span: Array) -> Array:
	if span.is_empty():
		return head.duplicate()
	var out := head.slice(0, int(span[0]))
	out.append_array(head.slice(int(span[1])))
	return out


## [param value] under [param key], as frontmatter lines. Never a trailing blank.
static func _block_lines(value: Variant, key: String) -> Array:
	var text := MiniYaml_.emit({key: value})
	var lines := _lines(text)
	while not lines.is_empty() and String(lines[-1]).strip_edges().is_empty():
		lines.remove_at(lines.size() - 1)
	return lines


## Split on newlines with the line ending taken back off, so CRLF and LF documents are
## the same problem. Rejoining uses the [code]nl[/code] [method split] reported.
static func _lines(text: String) -> Array:
	var out: Array = []
	for line in text.split("\n"):
		out.append(String(line).trim_suffix("\r"))
	return out
