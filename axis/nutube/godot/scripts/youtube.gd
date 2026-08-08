class_name YouTubeSource
extends Node

## Resolves YouTube videos into feed items - no API key required.
##
## Two public, key-free endpoints do the work:
##  - oEmbed (`/oembed`) returns the title and author as JSON.
##  - the thumbnail CDN (`img.youtube.com/vi/<id>/hqdefault.jpg`) returns a JPG.
##
## Why HTTPClient instead of the simpler HTTPRequest node: Godot's HTTPRequest
## resolves a host to a single address and, on a network with no working IPv6
## route, blocks ~12s per connection waiting for the IPv6 connect to time out
## before falling back to IPv4 (it has no Happy-Eyeballs racing like curl/
## browsers do). We sidestep that by resolving the IPv4 address ourselves and
## connecting straight to it, with a TLS common-name override so the
## certificate still validates against the real hostname. The two requests for
## a video also run concurrently. Net effect: a cold fetch drops from ~30s to
## well under a second. Warm launches fetch nothing (see LocalIndex caching).
##
## Each resolved video is emitted as an `item` dictionary in the same shape
## LocalIndex stores, so the recommender stays source-agnostic. Fetching the
## actual video stream is deliberately out of scope.

signal item_resolved(item: Dictionary)

# Cached thumbnails live here so a warm launch never touches the network.
const CACHE_DIR := "user://cache"
# Abort any single request that stalls past this, so a dead connection or a
# missing IPv4 route can't wedge a resolve forever.
const REQUEST_TIMEOUT_MS := 12000


## Kick off resolution for a watch URL. `item_resolved` fires once, after both
## the metadata and thumbnail requests have settled (success or failure).
func resolve(url: String) -> void:
	var id := id_from_url(url)
	if id == "":
		push_warning("nuTube: could not parse a video id from: %s" % url)
		return
	var item := {
		"id": id,
		"source": "youtube",
		"url": "https://youtu.be/%s" % id,
		"title": "",
		"author": "",
		"tags": [],
	}
	_resolve(item)


## Extract the 11-char video id from the common YouTube URL shapes.
func id_from_url(url: String) -> String:
	var u := url.strip_edges()
	if u.contains("watch?v="):
		u = u.get_slice("watch?v=", 1)
	elif u.contains("youtu.be/"):
		u = u.get_slice("youtu.be/", 1)
	elif u.contains("/shorts/"):
		u = u.get_slice("/shorts/", 1)
	# Drop any trailing query string, fragment, or path segment.
	for sep in ["?", "&", "#", "/"]:
		u = u.get_slice(sep, 0)
	return u


func thumbnail_url(id: String) -> String:
	return "https://img.youtube.com/vi/%s/hqdefault.jpg" % id


func _resolve(item: Dictionary) -> void:
	# Launch both fetches as concurrent coroutines (no await here), then wait
	# for both to drop the shared counter before emitting.
	var pending := {"open": 2}
	_load_metadata(item, pending)
	_load_thumbnail(item, pending)
	var deadline := Time.get_ticks_msec() + REQUEST_TIMEOUT_MS + 2000
	while pending["open"] > 0 and Time.get_ticks_msec() < deadline:
		await get_tree().process_frame
	item_resolved.emit(item)


func _load_metadata(item: Dictionary, pending: Dictionary) -> void:
	var endpoint := "/oembed?url=%s&format=json" % item["url"].uri_encode()
	var response = await _fetch("www.youtube.com", endpoint)
	if response["code"] == 200:
		var data = JSON.parse_string(response["body"].get_string_from_utf8())
		if data is Dictionary:
			item["title"] = data.get("title", item["title"])
			item["author"] = data.get("author_name", "")
	pending["open"] -= 1


func _load_thumbnail(item: Dictionary, pending: Dictionary) -> void:
	var response = await _fetch("img.youtube.com", "/vi/%s/hqdefault.jpg" % item["id"])
	if response["code"] == 200:
		var image := Image.new()
		if image.load_jpg_from_buffer(response["body"]) == OK:
			item["thumbnail"] = ImageTexture.create_from_image(image)
			item["thumb_path"] = _cache_thumbnail(item["id"], response["body"])
	pending["open"] -= 1


## GET `https://<host><path>` over IPv4, validating TLS against `host`.
## Returns `{ "code": int, "body": PackedByteArray }`; code 0 means it failed.
func _fetch(host: String, path: String) -> Dictionary:
	var result := {"code": 0, "body": PackedByteArray()}
	var ipv4 := IP.resolve_hostname(host, IP.TYPE_IPV4)
	var client := HTTPClient.new()
	var tls := TLSOptions.client(null, host)
	if client.connect_to_host(ipv4 if ipv4 != "" else host, 443, tls) != OK:
		return result

	var deadline := Time.get_ticks_msec() + REQUEST_TIMEOUT_MS
	while client.get_status() in [HTTPClient.STATUS_CONNECTING, HTTPClient.STATUS_RESOLVING]:
		client.poll()
		if Time.get_ticks_msec() > deadline:
			client.close()
			return result
		await get_tree().process_frame
	if client.get_status() != HTTPClient.STATUS_CONNECTED:
		client.close()
		return result

	var headers := ["Host: " + host, "User-Agent: nuTube/0.1 (Godot)"]
	if client.request(HTTPClient.METHOD_GET, path, headers) != OK:
		client.close()
		return result
	while client.get_status() == HTTPClient.STATUS_REQUESTING:
		client.poll()
		if Time.get_ticks_msec() > deadline:
			client.close()
			return result
		await get_tree().process_frame

	result["code"] = client.get_response_code()
	var body := PackedByteArray()
	while client.get_status() == HTTPClient.STATUS_BODY:
		client.poll()
		var chunk := client.read_response_body_chunk()
		if chunk.size() > 0:
			body.append_array(chunk)
		elif Time.get_ticks_msec() > deadline:
			break
		else:
			await get_tree().process_frame
	client.close()
	result["body"] = body
	return result


## Write the raw JPG bytes to the cache and return the path (empty on failure).
func _cache_thumbnail(id: String, bytes: PackedByteArray) -> String:
	if not DirAccess.dir_exists_absolute(CACHE_DIR):
		DirAccess.make_dir_recursive_absolute(CACHE_DIR)
	var path := "%s/thumb_%s.jpg" % [CACHE_DIR, id]
	var file := FileAccess.open(path, FileAccess.WRITE)
	if not file:
		return ""
	file.store_buffer(bytes)
	file.close()
	return path
