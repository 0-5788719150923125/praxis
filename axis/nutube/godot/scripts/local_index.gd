extends Node

## On-device index and recommender (autoloaded as `LocalIndex`).
##
## This is the core idea of nuTube: instead of asking a remote service what to
## watch next, we keep a small index on the device and rank it with simple,
## inspectable local algorithms (keyword overlap now; embedding similarity and
## watch-history feedback later). Nothing here talks to a network yet.
##
## The index is intentionally generic - an item is just a dictionary with a
## `title`, some `tags`, and an opaque `source`/`id`. YouTube is the first
## source; the plan is to add others behind the same shape and let this file
## stay the single ranking brain across all of them.

# Where the index is persisted between launches. Thumbnails are not stored
# inline (a Texture isn't JSON-serializable); each item keeps a `thumb_path`
# pointing at a cached image file written by the source.
const SAVE_PATH := "user://index.json"

# In-memory store of indexed items. Sources (see `youtube.gd`) populate this
# via `upsert`. Growing it from a real crawler is the obvious next step.
var _items: Array[Dictionary] = []


func _ready() -> void:
	load_index()


## Add (or replace) an item in the index. `item` must carry at least an `id`.
func upsert(item: Dictionary) -> void:
	var id := str(item.get("id", ""))
	if id == "":
		return
	for i in _items.size():
		if str(_items[i].get("id", "")) == id:
			_items[i] = item
			return
	_items.append(item)


## True if `id` is absent, or present but missing its thumbnail. Sources use
## this to skip the network entirely when a video is already fully cached.
func needs_fetch(id: String) -> bool:
	for item in _items:
		if str(item.get("id", "")) == id:
			return item.get("thumbnail") == null
	return true


## Rank the index against `query` and return the top `limit` items, each with a
## human-readable `reason`. An empty query returns a recent/default slice so the
## feed is never blank. The scorer is deliberately trivial for now - keyword
## overlap over title + tags - and is the main thing to improve.
func recommend(query: String, limit: int = 10) -> Array:
	if _items.is_empty():
		return []
	var terms := _tokenize(query)
	var scored: Array = []
	for item in _items:
		var score := _score(item, terms)
		scored.append({"item": item, "score": score})
	scored.sort_custom(func(a, b): return a["score"] > b["score"])

	var out: Array = []
	for entry in scored.slice(0, limit):
		var item: Dictionary = entry["item"].duplicate()
		item["reason"] = _reason(item, terms, entry["score"])
		out.append(item)
	return out


func _score(item: Dictionary, terms: PackedStringArray) -> float:
	if terms.is_empty():
		return 0.0
	var haystack := _tokenize(
		str(item.get("title", "")) + " " + " ".join(item.get("tags", []))
	)
	var overlap := 0
	for term in terms:
		if haystack.has(term):
			overlap += 1
	return float(overlap) / float(terms.size())


func _reason(item: Dictionary, terms: PackedStringArray, score: float) -> String:
	if terms.is_empty():
		return "From your local index"
	if score <= 0.0:
		return "Loosely related"
	return "Matches your search by tag overlap"


func _tokenize(text: String) -> PackedStringArray:
	var cleaned := text.to_lower()
	for ch in [",", ".", "!", "?", ":", ";", "(", ")", "\"", "'"]:
		cleaned = cleaned.replace(ch, " ")
	var tokens: PackedStringArray = []
	for token in cleaned.split(" ", false):
		if token.length() > 1:
			tokens.append(token)
	return tokens


## Persist the index to disk. Textures and transient fields are dropped; the
## cached `thumb_path` is enough to rebuild the thumbnail on the next load.
func save_index() -> void:
	var serializable: Array = []
	for item in _items:
		var copy := item.duplicate()
		copy.erase("thumbnail")
		copy.erase("reason")
		serializable.append(copy)
	var file := FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(serializable))


## Load the index from disk and rehydrate each thumbnail from its cached file.
func load_index() -> void:
	if not FileAccess.file_exists(SAVE_PATH):
		return
	var file := FileAccess.open(SAVE_PATH, FileAccess.READ)
	if not file:
		return
	var data = JSON.parse_string(file.get_as_text())
	if not (data is Array):
		return
	for entry in data:
		if not (entry is Dictionary):
			continue
		var item: Dictionary = entry
		var thumb_path := str(item.get("thumb_path", ""))
		if thumb_path != "" and FileAccess.file_exists(thumb_path):
			var image := Image.new()
			if image.load(thumb_path) == OK:
				item["thumbnail"] = ImageTexture.create_from_image(image)
		_items.append(item)
