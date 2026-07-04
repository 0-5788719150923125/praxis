extends RefCounted
class_name MiniYaml

## MiniYaml - an in-house parser for the YAML *subset* storyboards are written in.
##
## Godot has no built-in YAML, and an external addon is a heavy dependency for what a
## storyboard needs - so this is a small, strict parser for a documented subset. The
## rule it lives by: anything outside the subset is REJECTED loudly with a line number,
## never silently misparsed (a wrong parse is worse than no parse).
##
## Supported:
##   - `#` comments (full-line and trailing, outside quotes)
##   - block maps (`key: value`) and block lists (`- item`), nested by indentation
##     (spaces only - a tab in the indent is an error)
##   - single-line flow collections `[a, b, [c]]` and `{k: v, k2: v2}`, nestable
##   - scalars: int, float, `true`/`false`, `null`/`~`, bare strings, `'single'` and
##     `"double"` quoted strings (`\\`, `\"`, `\n`, `\t` escapes in double quotes)
##   - a list item may open an inline map (`- {id: x}`) or a block map (`- id: x`
##     with further keys aligned beneath)
##
## Rejected (with a line-numbered error): anchors/aliases (`&`, `*`), tags (`!`),
## multi-document (`---`), directives (`%`), block scalars (`|`, `>`), merge keys
## (`<<`), flow collections spanning lines, duplicate keys, tabs in indentation.
##
## `parse(text)` returns `{ok: bool, data: Variant, error: String}`. Numbers parse as
## int when integral, float otherwise (JSON parses everything as float; every consumer
## casts its numeric fields, so the two formats behave identically downstream).

static func parse(text: String) -> Dictionary:
	var p := _Parser.new()
	var data: Variant = p.parse_document(text)
	if p.err != "":
		return {"ok": false, "data": null, "error": p.err}
	return {"ok": true, "data": data, "error": ""}


class _Parser:
	var lines: Array = []        # {n: line number, indent: int, text: String} - non-blank only
	var i := 0                   # cursor into lines
	var err := ""

	func fail(n: int, msg: String) -> void:
		if err == "":            # keep the FIRST error; later ones are cascade noise
			err = "line %d: %s" % [n, msg]

	func parse_document(text: String) -> Variant:
		var raw := text.split("\n")
		for k in raw.size():
			var n := k + 1
			var line := _strip_comment(raw[k])
			var j := 0
			while j < line.length() and line[j] == " ":
				j += 1
			if j < line.length() and line[j] == "\t":
				fail(n, "tab in indentation (use spaces)")
				return null
			var body := line.substr(j).strip_edges(false, true)
			if body.is_empty():
				continue
			if body.begins_with("---") or body.begins_with("..."):
				fail(n, "multi-document markers are not supported")
				return null
			if body.begins_with("%"):
				fail(n, "directives are not supported")
				return null
			lines.append({"n": n, "indent": j, "text": body})
		if lines.is_empty():
			return {}
		var root: Variant = parse_block(int(lines[0].indent))
		if err == "" and i < lines.size():
			fail(int(lines[i].n), "unexpected content (indentation mismatch?)")
		return root

	# Cut a trailing `#` comment, respecting quotes. A `#` starts a comment only at the
	# start of the line or after whitespace (so `a#b` stays a bare scalar, per YAML).
	func _strip_comment(line: String) -> String:
		var q := ""
		var k := 0
		while k < line.length():
			var c := line[k]
			if q == "":
				if c == "'" or c == "\"":
					q = c
				elif c == "#" and (k == 0 or line[k - 1] == " " or line[k - 1] == "\t"):
					return line.substr(0, k)
			elif q == "'":
				if c == "'":
					q = ""
			else:
				if c == "\\":
					k += 1
				elif c == "\"":
					q = ""
			k += 1
		return line

	func parse_block(indent: int) -> Variant:
		var t := String(lines[i].text)
		if t == "-" or t.begins_with("- "):
			return parse_list(indent)
		return parse_map(indent)

	func parse_list(indent: int) -> Variant:
		var out := []
		while err == "" and i < lines.size():
			var ln: Dictionary = lines[i]
			if int(ln.indent) < indent:
				break
			if int(ln.indent) > indent:
				fail(int(ln.n), "bad indentation")
				break
			var t := String(ln.text)
			if not (t == "-" or t.begins_with("- ")):
				break                                   # back to the enclosing map
			if t == "-":
				i += 1
				if i < lines.size() and int(lines[i].indent) > indent:
					out.append(parse_block(int(lines[i].indent)))
				else:
					out.append(null)
				continue
			# Inline content after the dash. It may be a scalar/flow value, or open a
			# block map / nested list whose siblings align under it.
			var after := t.substr(1)
			var s := 0
			while s < after.length() and after[s] == " ":
				s += 1
			var content := after.substr(s)
			var content_indent := indent + 1 + s
			if content.begins_with("- ") or content == "-" or _find_key_colon(content) >= 0:
				lines[i] = {"n": ln.n, "indent": content_indent, "text": content}
				out.append(parse_block(content_indent))
			else:
				out.append(parse_value(content, int(ln.n)))
				i += 1
		return out

	func parse_map(indent: int) -> Variant:
		var out := {}
		while err == "" and i < lines.size():
			var ln: Dictionary = lines[i]
			if int(ln.indent) < indent:
				break
			if int(ln.indent) > indent:
				fail(int(ln.n), "bad indentation")
				break
			var t := String(ln.text)
			if t == "-" or t.begins_with("- "):
				fail(int(ln.n), "unexpected list item inside a map")
				break
			var ci := _find_key_colon(t)
			if ci < 0:
				fail(int(ln.n), "expected 'key: value'")
				break
			var key := _unquote_key(t.substr(0, ci).strip_edges(), int(ln.n))
			if key == "<<":
				fail(int(ln.n), "merge keys are not supported")
				break
			if out.has(key):
				fail(int(ln.n), "duplicate key '%s'" % key)
				break
			var rest := t.substr(ci + 1).strip_edges()
			i += 1
			if rest.is_empty():
				if i < lines.size() and int(lines[i].indent) > indent:
					out[key] = parse_block(int(lines[i].indent))
				elif i < lines.size() and int(lines[i].indent) == indent \
						and (String(lines[i].text) == "-" or String(lines[i].text).begins_with("- ")):
					out[key] = parse_list(indent)   # a list may sit at the key's own indent
				else:
					out[key] = null
			else:
				out[key] = parse_value(rest, int(ln.n))
		return out

	# Index of the colon that separates a block-map key from its value: at bracket
	# depth 0, outside quotes, followed by a space or the end of the line. -1 = none.
	func _find_key_colon(s: String) -> int:
		var q := ""
		var depth := 0
		var k := 0
		while k < s.length():
			var c := s[k]
			if q == "":
				if c == "'" or c == "\"":
					q = c
				elif c == "[" or c == "{":
					depth += 1
				elif c == "]" or c == "}":
					depth -= 1
				elif c == ":" and depth == 0 and (k + 1 >= s.length() or s[k + 1] == " "):
					return k
			elif q == "'":
				if c == "'":
					q = ""
			else:
				if c == "\\":
					k += 1
				elif c == "\"":
					q = ""
			k += 1
		return -1

	func _unquote_key(raw: String, n: int) -> String:
		if raw.length() >= 2 and (raw[0] == "'" or raw[0] == "\""):
			var pos := [0]
			var v: Variant = _quoted(raw, pos, n)
			return String(v)
		return raw

	# A value on one line: a flow collection, or a scalar.
	func parse_value(s: String, n: int) -> Variant:
		if s.begins_with("[") or s.begins_with("{"):
			return parse_flow(s, n)
		if s.begins_with("'") or s.begins_with("\""):
			var pos := [0]
			var v: Variant = _quoted(s, pos, n)
			if err == "":
				_skip_ws(s, pos)
				if pos[0] < s.length():
					fail(n, "trailing characters after quoted string")
			return v
		return parse_scalar(s, n)

	func parse_flow(s: String, n: int) -> Variant:
		var pos := [0]
		var v: Variant = _flow_value(s, pos, n)
		if err == "":
			_skip_ws(s, pos)
			if pos[0] < s.length():
				fail(n, "trailing characters after flow value (flow must fit on one line)")
		return v

	func _skip_ws(s: String, pos: Array) -> void:
		while pos[0] < s.length() and (s[pos[0]] == " " or s[pos[0]] == "\t"):
			pos[0] += 1

	func _flow_value(s: String, pos: Array, n: int) -> Variant:
		_skip_ws(s, pos)
		if pos[0] >= s.length():
			fail(n, "unterminated flow value (flow must fit on one line)")
			return null
		var c := s[pos[0]]
		if c == "[":
			return _flow_list(s, pos, n)
		if c == "{":
			return _flow_map(s, pos, n)
		if c == "'" or c == "\"":
			return _quoted(s, pos, n)
		var start: int = pos[0]
		while pos[0] < s.length() and s[pos[0]] != "," and s[pos[0]] != "]" and s[pos[0]] != "}":
			pos[0] += 1
		return parse_scalar(s.substr(start, pos[0] - start).strip_edges(), n)

	func _flow_list(s: String, pos: Array, n: int) -> Variant:
		pos[0] += 1
		var out := []
		_skip_ws(s, pos)
		if pos[0] < s.length() and s[pos[0]] == "]":
			pos[0] += 1
			return out
		while err == "":
			out.append(_flow_value(s, pos, n))
			_skip_ws(s, pos)
			if pos[0] >= s.length():
				fail(n, "unterminated '[' (flow must fit on one line)")
				return out
			if s[pos[0]] == ",":
				pos[0] += 1
				continue
			if s[pos[0]] == "]":
				pos[0] += 1
				return out
			fail(n, "expected ',' or ']' in flow list")
		return out

	func _flow_map(s: String, pos: Array, n: int) -> Variant:
		pos[0] += 1
		var out := {}
		_skip_ws(s, pos)
		if pos[0] < s.length() and s[pos[0]] == "}":
			pos[0] += 1
			return out
		while err == "":
			_skip_ws(s, pos)
			var key: String
			if pos[0] < s.length() and (s[pos[0]] == "'" or s[pos[0]] == "\""):
				key = String(_quoted(s, pos, n))
			else:
				var st: int = pos[0]
				while pos[0] < s.length() and s[pos[0]] != ":" and s[pos[0]] != "}" and s[pos[0]] != ",":
					pos[0] += 1
				key = s.substr(st, pos[0] - st).strip_edges()
			_skip_ws(s, pos)
			if pos[0] >= s.length() or s[pos[0]] != ":":
				fail(n, "expected ':' in flow map")
				return out
			pos[0] += 1
			if out.has(key):
				fail(n, "duplicate key '%s'" % key)
				return out
			out[key] = _flow_value(s, pos, n)
			_skip_ws(s, pos)
			if pos[0] >= s.length():
				fail(n, "unterminated '{' (flow must fit on one line)")
				return out
			if s[pos[0]] == ",":
				pos[0] += 1
				continue
			if s[pos[0]] == "}":
				pos[0] += 1
				return out
			fail(n, "expected ',' or '}' in flow map")
		return out

	# A quoted string starting at pos[0]; advances past the closing quote.
	func _quoted(s: String, pos: Array, n: int) -> Variant:
		var q := s[pos[0]]
		pos[0] += 1
		var out := ""
		while pos[0] < s.length():
			var c := s[pos[0]]
			if q == "'":
				if c == "'":
					if pos[0] + 1 < s.length() and s[pos[0] + 1] == "'":
						out += "'"              # '' is an escaped quote inside single quotes
						pos[0] += 2
						continue
					pos[0] += 1
					return out
				out += c
			else:
				if c == "\\":
					pos[0] += 1
					if pos[0] >= s.length():
						break
					var e := s[pos[0]]
					match e:
						"n": out += "\n"
						"t": out += "\t"
						"\"": out += "\""
						"\\": out += "\\"
						_: out += e
				elif c == "\"":
					pos[0] += 1
					return out
				else:
					out += c
			pos[0] += 1
		fail(n, "unterminated quoted string")
		return out

	# A bare scalar: typed if it reads as one, else a plain string.
	func parse_scalar(t: String, n: int) -> Variant:
		if t.is_empty():
			return null
		var c0 := t[0]
		if c0 == "&" or c0 == "*":
			fail(n, "anchors/aliases are not supported")
			return null
		if c0 == "!":
			fail(n, "tags are not supported")
			return null
		if c0 == "|" or c0 == ">":
			fail(n, "block scalars are not supported")
			return null
		match t:
			"true", "True":
				return true
			"false", "False":
				return false
			"null", "Null", "~":
				return null
		if t.is_valid_int():
			return t.to_int()
		if t.is_valid_float():
			return t.to_float()
		return t
