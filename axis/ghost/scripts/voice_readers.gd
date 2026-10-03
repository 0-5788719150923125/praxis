extends RefCounted
class_name VoiceReaders

## VoiceReaders - who a multi-speaker voice's speaker NUMBER actually is.
##
## Piper's libritts voices hold hundreds of readers (904 in en_US-libritts-high) addressed by
## an index, and the index says nothing: the only way to find a voice was to step through them
## and listen. The model's own config names each index by its LibriTTS reader id
## (`speaker_id_map`: "p3615" -> 10), and LibriTTS-P (LINE, CC BY 4.0) has annotated those
## readers - perceived gender and a description of the voice ("soft, clear, calm, reassuring").
## `data/libritts_speakers.json` is that annotation, keyed by reader id (see its .LICENSE).
##
## Read only from files: the model config the voice host downloaded, and the data file. A voice
## that is not installed yet, or has no map, simply has no descriptions.

const DATA := "res://data/libritts_speakers.json"

static var _readers := {}          # reader id -> {g, d}
static var _maps := {}             # voice id -> PackedStringArray: speaker index -> reader id


static func _load_readers() -> void:
	if not _readers.is_empty():
		return
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(DATA))
	if parsed is Dictionary:
		_readers = (parsed as Dictionary).get("readers", {})


static func _map_of(voice_id: String) -> PackedStringArray:
	if _maps.has(voice_id):
		return _maps[voice_id]
	var out := PackedStringArray()
	var cfg := Deps.data_dir().path_join("voices/piper/%s.onnx.json" % voice_id)
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(cfg)) \
		if FileAccess.file_exists(cfg) else null
	if parsed is Dictionary:
		var m: Dictionary = (parsed as Dictionary).get("speaker_id_map", {})
		out.resize(m.size())
		for key in m:
			var i := int(m[key])
			if i >= 0 and i < out.size():
				out[i] = String(key).trim_prefix("p")
	_maps[voice_id] = out
	return out


## Speaker [param index] of [param voice_id]: `{reader, g, d}` (g "F", "M" or ""), or empty
## when nothing is known about it.
static func describe(voice_id: String, index: int) -> Dictionary:
	_load_readers()
	var m := _map_of(voice_id)
	if index < 0 or index >= m.size() or m[index].is_empty():
		return {}
	var r: Dictionary = _readers.get(m[index], {})
	return {"reader": m[index], "g": String(r.get("g", "")), "d": String(r.get("d", ""))}


## Every speaker of [param voice_id], in index order: `[{i, reader, g, d}]`.
static func all(voice_id: String) -> Array:
	_load_readers()
	var out: Array = []
	var m := _map_of(voice_id)
	for i in m.size():
		var r: Dictionary = _readers.get(m[i], {})
		out.append({"i": i, "reader": m[i], "g": String(r.get("g", "")), "d": String(r.get("d", ""))})
	return out


## One line for a speaker: a gender sign and the start of its description.
static func line(d: Dictionary, words := 6) -> String:
	if d.is_empty():
		return ""
	var sign: String = {"F": "♀ female", "M": "♂ male"}.get(String(d["g"]), "unannotated")
	var desc := String(d["d"]).split(", ")
	var short := ", ".join(desc.slice(0, words)) + ("…" if desc.size() > words else "")
	return "%s  ·  %s" % [sign, short] if not short.is_empty() else sign
