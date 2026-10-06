extends SceneTree

## youtube_flow_check - the YouTube sign-in and upload end to end, against a stand-in for Google on
## 127.0.0.1: no network, no quota, no account. It speaks the protocol as Google documents it - the
## installed-app sign-in with PKCE, the token endpoint, and resumable uploads (308 Resume
## Incomplete with `Range`, the `bytes */N` status check, 404 for a lapsed session).
##
##   godot --headless --path . --script res://tests/youtube_flow_check.gd
##
## - SIGN-IN: the "browser" (YouTube.open_url) reads the consent URL ghost built and answers at its
##   redirect - first a favicon and a stale tab carrying the wrong state, both turned away while the
##   wait goes on, then the code. The stand-in only trades the code for the right client, the same
##   redirect and a verifier whose S256 is the challenge; the tokens are kept owner-only.
## - UPLOAD: a file of six chunks, against a stand-in that drops one chunk (503), keeps half of
##   another and refuses a stale token (401): YouTube ends up with exactly the file's bytes - the
##   stand-in rejects any chunk that does not start where it stopped, so nothing is sent twice - the
##   metadata is unlisted / not for kids / synthetic / no paid promotion / People & Blogs, the session
##   names exactly the parts it sets, the result is recorded and nothing is
##   left pending.
## - RESUME: an upload stopped after two chunks, as a quit would stop it, waits in pending.json and a
##   fresh node finishes it from where the stand-in says it stopped; a session YouTube has forgotten
##   (404) starts over in a new one.
## - A LAPSED SIGN-IN: a revoked refresh token is forgotten and read as signed out.
## - DECLINED: the consent screen's `access_denied` ends the sign-in, saying so.
## - SIGN-OUT: the refresh token is revoked at Google and forgotten here; the client file stays.
## - THE THUMBNAIL (2026-10-06, the user: "submit a screenshot from the 'intro' section of the
##   video"): ffmpeg takes a 1280x720 frame of a real video at the moment asked; queued with an
##   upload, it is set once the video is up; a channel YouTube will not let set one (403) still has
##   its video, and is told why.
## - THE EXPORT MENU'S SIGN-IN (2026-10-06, the user: after Google's "Access blocked" for an account
##   that was not a test user, "I am not prompted for credentials, nor am I redirected through the
##   OAuth flow"): ticking the box with no sign-in opens the browser and a dialog at once; Google's
##   dead-end block leaves the dialog waiting and naming it; "Open the page again" finishes the same
##   sign-in once the account is allowed; closing the dialog cancels; a refusal shows the hint and
##   "Try again"; a newer sign-in replaces one still waiting without the old one reporting over it.

const YouTube := preload("res://scripts/youtube.gd")
const ROOT := "user://youtube_flow_check"
const CHUNK := 256 * 1024

var _fails := 0
var _fake: FakeGoogle
var _yt: YouTube
var _mode := "accept"           # how the browser answers: accept | deny | block (Google's dead end)
var _opens := 0                 # times the consent page was opened
var _saw := {}                  # the consent URL's query, as the browser read it
var _turned_away: Array = []    # statuses of the knocks that were not the answer
var _page := 0                  # status of the page the answer got


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _run() -> void:
	# a backstop: a hung exchange fails the gate rather than holding it forever
	create_timer(120.0).timeout.connect(func() -> void:
		print("youtube_flow_check: TIMED OUT")
		quit(1))
	var keep := {"root": YouTube.root, "upload": YouTube.upload_url, "revoke": YouTube.revoke_url,
		"thumbnail": YouTube.thumbnail_url,
		"chunk": YouTube.chunk, "backoff": YouTube.backoff_unit, "open": YouTube.open_url}
	YouTube.root = ROOT
	_wipe(ROOT)
	_wipe(ROOT + "_files")
	_fake = FakeGoogle.new()
	root.add_child(_fake)
	var base := "http://127.0.0.1:%d" % _fake.start()
	YouTube.upload_url = base + "/upload/youtube/v3/videos"
	YouTube.revoke_url = base + "/revoke"
	YouTube.thumbnail_url = base + "/thumbnails/set"
	YouTube.chunk = CHUNK
	YouTube.backoff_unit = 0.01
	YouTube.open_url = _browser
	var client := _file("client.json", JSON.stringify({"installed": {"client_id": _fake.client_id,
		"project_id": "flow-check", "auth_uri": base + "/auth", "token_uri": base + "/token",
		"client_secret": _fake.client_secret, "redirect_uris": ["http://localhost"]}}))
	_ok(YouTube.import_client(client) == "", "the stand-in's client imports")
	_yt = YouTube.new()
	root.add_child(_yt)
	for check in [_sign_in, _upload, _resume, _lapsed_session, _thumbnail, _frame, _lapsed_sign_in, _declined,
			_sign_out, _menu_sign_in]:
		if not bool(await check.call()):
			_ok(false, "a check stopped part way (script error?)")
	_wipe(ROOT)
	_wipe(ROOT + "_files")
	YouTube.root = keep["root"]
	YouTube.upload_url = keep["upload"]
	YouTube.revoke_url = keep["revoke"]
	YouTube.thumbnail_url = keep["thumbnail"]
	YouTube.chunk = keep["chunk"]
	YouTube.backoff_unit = keep["backoff"]
	YouTube.open_url = keep["open"]
	print("youtube_flow_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	quit(0 if _fails == 0 else 1)


func _wipe(path: String) -> void:
	var dir := ProjectSettings.globalize_path(path)
	if not DirAccess.dir_exists_absolute(dir):
		return
	for d in DirAccess.get_directories_at(dir):
		_wipe(path.path_join(d))
	for f in DirAccess.get_files_at(dir):
		DirAccess.remove_absolute(dir.path_join(f))
	DirAccess.remove_absolute(dir)


func _file(name: String, text: String) -> String:
	var path := ProjectSettings.globalize_path(ROOT + "_files").path_join(name)
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()
	return path


## A "video" of [param size] random bytes: chunks that went to the wrong place could not still match.
func _video(name: String, size: int) -> String:
	var path := ProjectSettings.globalize_path(ROOT + "_files").path_join(name)
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_buffer(Crypto.new().generate_random_bytes(size))
	f.close()
	return path


# --- the browser ---------------------------------------------------------------------------------

## What the browser does with the consent URL: Google's page would sign in and send it back to the
## redirect. A coroutine - it goes on knocking while the sign-in waits.
func _browser(url: String) -> void:
	_opens += 1
	_saw = YouTube.query_of(url)
	_fake.challenge = str(_saw.get("code_challenge", ""))
	_fake.redirect = str(_saw.get("redirect_uri", ""))
	_turned_away = []
	_page = 0
	var back := str(_saw.get("redirect_uri", ""))
	var state := str(_saw.get("state", ""))
	await process_frame
	var r: Dictionary
	if _mode == "block":
		return                       # "Access blocked": Google's page never sends the browser back
	if _mode == "deny":
		r = await _fetch(back + "/?state=%s&error=access_denied" % state)
		_page = int(r["status"])
		return
	r = await _fetch(back + "/favicon.ico")
	_turned_away.append(int(r["status"]))
	r = await _fetch(back + "/?state=stale&code=old-code")
	_turned_away.append(int(r["status"]))
	r = await _fetch(back + "/?state=%s&code=%s&scope=%s" % [state, _fake.code.uri_encode(), YouTube.SCOPE.uri_encode()])
	_page = int(r["status"])


## The browser's last page, once it has read it: the sign-in can return a frame before that.
func _page_read() -> int:
	var t0 := Time.get_ticks_msec()
	while _page == 0 and Time.get_ticks_msec() - t0 < 3000:
		await process_frame
	return _page


## A plain GET, read to the end: `{status, body}`.
func _fetch(url: String) -> Dictionary:
	var hostport := url.get_slice("://", 1).get_slice("/", 0)
	var path := url.get_slice("://", 1).substr(hostport.length())
	var c := StreamPeerTCP.new()
	c.connect_to_host("127.0.0.1", int(hostport.get_slice(":", 1)))
	var t0 := Time.get_ticks_msec()
	while c.get_status() == StreamPeerTCP.STATUS_CONNECTING and Time.get_ticks_msec() - t0 < 3000:
		c.poll()
		await process_frame
	c.put_data(("GET %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n" % [path, hostport]).to_utf8_buffer())
	var got := PackedByteArray()
	while Time.get_ticks_msec() - t0 < 5000:
		c.poll()
		if c.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			break
		var n := c.get_available_bytes()
		if n > 0:
			got.append_array(c.get_partial_data(n)[1] as PackedByteArray)
		await process_frame
	var text := got.get_string_from_utf8()
	return {"status": int(text.get_slice(" ", 1)) if text.begins_with("HTTP/") else 0,
		"body": text.get_slice("\r\n\r\n", 1)}


# --- the checks ----------------------------------------------------------------------------------

func _sign_in() -> bool:
	_mode = "accept"
	var err: String = await _yt.sign_in()
	_ok(err == "", "the sign-in completes through the browser (%s)" % err)
	_ok(_turned_away == [404, 400], "a favicon and a stale tab are turned away while the wait goes on (%s)" % str(_turned_away))
	_ok(await _page_read() == 200, "the browser is told the sign-in worked")
	_ok(_saw.get("scope") == YouTube.SCOPE and _saw.get("code_challenge_method") == "S256"
		and _saw.get("access_type") == "offline" and _saw.get("prompt") == "consent"
		and str(_saw.get("redirect_uri", "")).begins_with("http://127.0.0.1:"),
		"the consent URL asks for uploads only, with PKCE, offline, at a loopback redirect")
	_ok(_fake.exchanged == 1, "the code was traded once - for the right client, redirect and verifier")
	_ok(YouTube.signed_in(), "the sign-in is kept")
	var kept := ProjectSettings.globalize_path(ROOT.path_join("token.json"))
	_ok(OS.get_name() == "Windows" or FileAccess.get_unix_permissions(kept) == YouTube.OWNER_FILE,
		"token.json is owner read/write only")
	var before := _fake.refreshes
	var tok: Dictionary = await _yt.access_token()
	_ok(not str(tok.get("token", "")).is_empty() and _fake.refreshes == before,
		"a live access token is handed out without asking Google again")
	return true


func _upload() -> bool:
	var file := _video("one.mp4", CHUNK * 5 + 12345)
	_fake.faults = {1: "503", 2: "half", 4: "401"}
	_fake.chunks = 0
	var refreshes := _fake.refreshes
	var body: Dictionary = YouTube.video_body({"title": "A <title>", "description": "Hello.\n\n0:00 Intro",
		"tags": ["tarot", "Tarot", "pick a card"]})
	var rec := ProjectSettings.globalize_path(ROOT + "_files/episode/youtube.json")
	var res: Dictionary = await _yt.upload(file, body, rec)
	_ok(not res.has("error"), "the upload finishes through a dropped chunk, a half-kept one and a stale token (%s)"
		% res.get("error", ""))
	var s: Dictionary = _fake.last()
	_ok(s.get("data", PackedByteArray()) == FileAccess.get_file_as_bytes(file), "YouTube holds exactly the file's bytes")
	_ok(_fake.status_checks >= 1, "after the dropped chunk it asked YouTube where the upload stood")
	_ok(_fake.refreshes > refreshes, "the stale token was renewed part way")
	var meta: Dictionary = s.get("meta", {})
	_ok(meta.get("status") == {"privacyStatus": "unlisted", "selfDeclaredMadeForKids": false, "containsSyntheticMedia": true}
		and meta["snippet"]["categoryId"] == "22" and meta.get("paidProductPlacementDetails") == {"hasPaidProductPlacement": false},
		"it went up unlisted, not for kids, synthetic, no paid promotion, People & Blogs")
	var named := Array(str(s.get("parts", "")).split(","))
	named.sort()
	_ok(named == ["paidProductPlacementDetails", "snippet", "status"],
		"the session names exactly the parts the body sets, in any order (%s)" % s.get("parts"))
	_ok(meta["snippet"]["title"] == "A ‹title›" and meta["snippet"]["tags"] == ["tarot", "pick a card"]
		and meta["snippet"]["description"] == "Hello.\n\n0:00 Intro", "with the title, description and tags fitted")
	_ok(int(s.get("declared", 0)) == FileAccess.get_file_as_bytes(file).size() and s.get("type") == "video/mp4",
		"the session was opened for the file's size and type")
	_ok(res.get("url") == "https://youtu.be/%s" % s.get("id") and res.get("privacy") == "unlisted"
		and res.get("channel") == "Stand-in Channel", "the result names the video, its privacy and the channel")
	var ups: Array = YouTube.uploads_in(rec)
	_ok(ups.size() == 1 and ups[0]["url"] == res.get("url") and ups[0]["file"] == file, "the upload is recorded for the mode")
	_ok(YouTube.pending().is_empty(), "nothing is left pending")
	_ok(is_equal_approx(_yt.progress, 1.0), "progress ends at 100%")
	return true


func _resume() -> bool:
	var file := _video("two.mp4", CHUNK * 4 + 777)
	_fake.faults = {}
	_fake.chunks = 0
	_fake.on_chunk = func(n: int) -> void:
		if n == 2:
			_yt.stop()               # as a quit would, part way through
	var first: Dictionary = await _yt.upload(file, YouTube.video_body({"title": "Resumed"}), "")
	_fake.on_chunk = Callable()
	_ok(first.has("error"), "a stopped upload says it stopped (%s)" % first.get("error", ""))
	var p: Dictionary = YouTube.pending()
	_ok(not str(p.get("session", "")).is_empty() and p.get("file") == file, "it waits in pending.json, its session kept")
	_ok((_fake.last()["data"] as PackedByteArray).size() == CHUNK * 2, "two chunks had gone up")
	var fresh := YouTube.new()
	root.add_child(fresh)
	var checks := _fake.status_checks
	var res: Dictionary = await fresh.resume()
	_ok(not res.has("error"), "a fresh node resumes it to the end (%s)" % res.get("error", ""))
	_ok(_fake.status_checks == checks + 1, "it asked YouTube where it stood before sending anything")
	_ok(_fake.chunks == 5, "and sent only the three chunks left (%d in all)" % _fake.chunks)
	_ok(_fake.last()["data"] == FileAccess.get_file_as_bytes(file), "YouTube holds exactly the file's bytes")
	_ok(YouTube.pending().is_empty(), "nothing is left pending")
	fresh.queue_free()
	return true


func _lapsed_session() -> bool:
	var file := _video("three.mp4", CHUNK + 5)
	var q: Dictionary = YouTube.queue(file, YouTube.video_body({"title": "Lapsed"}), "")
	q["session"] = "http://127.0.0.1:%d/session/forgotten" % _fake.port
	YouTube._write(YouTube._path("pending.json"), JSON.stringify(q), true)
	var opened := _fake.sessions.size()
	var res: Dictionary = await _yt.resume()
	_ok(not res.has("error"), "a session YouTube has forgotten starts over (%s)" % res.get("error", ""))
	_ok(_fake.sessions.size() == opened + 1 and _fake.last()["data"] == FileAccess.get_file_as_bytes(file),
		"in a new session, with the whole file")
	return true


func _lapsed_sign_in() -> bool:
	YouTube._expire()
	_fake.refresh_ok = false
	var tok: Dictionary = await _yt.access_token()
	_fake.refresh_ok = true
	_ok(tok.has("signed_out") and str(tok.get("error", "")).contains("sign in again"),
		"a revoked refresh token reads as signed out (%s)" % tok.get("error", ""))
	_ok(not YouTube.signed_in(), "and is forgotten")
	return true


func _declined() -> bool:
	_mode = "deny"
	var err: String = await _yt.sign_in()
	_ok(err.contains("declined"), "a declined consent screen ends the sign-in, saying so (%s)" % err)
	_ok(await _page_read() == 200 and not YouTube.signed_in(), "the browser hears it too, and nothing is kept")
	return true


func _sign_out() -> bool:
	_mode = "accept"
	var err: String = await _yt.sign_in()
	_ok(err == "" and YouTube.signed_in(), "signed in again (%s)" % err)
	await _yt.sign_out()
	_ok(not YouTube.signed_in(), "signing out forgets the sign-in")
	_ok(_fake.revoked.has("rt-1"), "and revokes it at Google")
	_ok(YouTube.has_client(), "the client file stays imported")
	return true



func _frames(n: int) -> void:
	for i in n:
		await process_frame


func _until(cond: Callable, ms := 5000) -> bool:
	var t0 := Time.get_ticks_msec()
	while not bool(cond.call()) and Time.get_ticks_msec() - t0 < ms:
		await process_frame
	return bool(cond.call())


## The exporter's own sign-in, through its menu and dialog. Built by hand and its children moved into
## the tree: its _ready would clear a live render's override.cfg.
func _menu_sign_in() -> bool:
	YouTube.forget_token()
	var ex = load("res://scripts/exporter.gd").new()
	ex._build_ui()
	var holder := Node.new()
	root.add_child(holder)
	for c in ex.get_children():
		ex.remove_child(c)
		holder.add_child(c)
	ex.upload_provider = func(_take: String) -> Dictionary: return {"title": "An Episode"}
	ex._refresh_upload_items()

	_mode = "block"
	_opens = 0
	ex._toggle_upload()
	await _frames(10)
	_ok(ex._sign == "signing_in" and ex._sign_dialog.visible and not ex._upload and _opens == 1,
		"ticking the box with no sign-in opens the browser and a dialog at once, the box not ticked yet")
	_ok(ex._sign_dialog.dialog_text.contains("Access blocked") and ex._sign_dialog.dialog_text.contains("test user"),
		"while Google's page is a dead end, the dialog names the test-user block")
	_mode = "accept"
	ex._on_sign_action("again")
	_ok(await _until(func() -> bool: return ex._sign == "ok"), "the account allowed, the page opened again finishes the SAME sign-in")
	_ok(_opens == 2 and YouTube.signed_in() and not ex._sign_dialog.visible and ex._upload,
		"signed in: the dialog closes and the box is ticked")
	_ok(ex._quality_menu.get_item_index(ex.CLIENT_ID) >= 0, "the menu offers a different client file once one is kept")

	YouTube.forget_token()
	_mode = "block"
	ex._upload = false
	ex._toggle_upload()
	await _frames(5)
	ex._sign_dialog.hide()
	ex._on_sign_close()
	await _frames(5)
	_ok(ex._sign == "failed" and ex._sign_why.contains("cancelled") and ex._yt.phase == "" and not ex._upload,
		"closing the dialog cancels the sign-in, and the box stays clear")

	_mode = "deny"
	ex._toggle_upload()
	_ok(await _until(func() -> bool: return ex._sign == "failed"), "a refused consent fails the sign-in")
	_ok(ex._sign_dialog.visible and ex._sign_dialog.dialog_text.contains("access_denied")
		and ex._sign_dialog.dialog_text.contains("test user") and ex._sign_again.text == "Try again",
		"the dialog says so, with the test-user hint, and offers to try again")
	_mode = "accept"
	ex._on_sign_action("again")
	_ok(await _until(func() -> bool: return ex._sign == "ok") and ex._upload and YouTube.signed_in(),
		"trying again signs in and ticks the box")

	YouTube.forget_token()
	_mode = "block"
	ex._check_sign_in()
	await _frames(5)
	var first: String = ex._yt.sign_in_url
	_mode = "accept"
	ex._check_sign_in()
	_ok(await _until(func() -> bool: return ex._sign == "ok"), "a newer sign-in replaces one still waiting, and finishes")
	await _frames(5)
	_ok(ex._sign == "ok" and not first.is_empty() and ex._yt.sign_in_url.is_empty() and ex._yt.phase == "",
		"the replaced one reports nothing over it")
	holder.queue_free()
	ex.free()
	return true


func _thumbnail() -> bool:
	var file := _video("four.mp4", CHUNK + 99)
	var jpg := _video("four.thumbnail.jpg", 4096)
	_fake.thumb_forbidden = false
	YouTube.queue(file, YouTube.video_body({"title": "With A Thumbnail"}), "", jpg)
	var res: Dictionary = await _yt.resume()
	var id := str(res.get("id", ""))
	_ok(not res.has("error") and res.get("thumbnail") == true and str(res.get("thumbnail_error", "x")).is_empty(),
		"a queued thumbnail is set once the video is up (%s)" % res.get("thumbnail_error", res.get("error", "")))
	_ok(_fake.thumbs.get(id, PackedByteArray()) == FileAccess.get_file_as_bytes(jpg), "YouTube got exactly the image, for that video")
	var file2 := _video("five.mp4", CHUNK + 7)
	_fake.thumb_forbidden = true
	YouTube.queue(file2, YouTube.video_body({"title": "No Thumbnail Allowed"}), "", jpg)
	var res2: Dictionary = await _yt.resume()
	_fake.thumb_forbidden = false
	_ok(not res2.has("error") and str(res2.get("url", "")).begins_with("https://youtu.be/") and res2.get("thumbnail") == false
		and str(res2.get("thumbnail_error", "")).contains("verified channel"),
		"a channel that may not set thumbnails still has its video, and is told why (%s)" % res2.get("thumbnail_error", ""))
	return true


## A real frame of a real video, taken by the exporter's own code - where ffmpeg is installed.
func _frame() -> bool:
	if not Deps.has("ffmpeg"):
		print("  skip  no ffmpeg here: the frame is not taken")
		return true
	var mp4 := ProjectSettings.globalize_path(ROOT + "_files/clip.mp4")
	var out: Array = []
	OS.execute(Deps.resolve("ffmpeg"), ["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=25",
		"-pix_fmt", "yuv420p", mp4], out)
	var ex = load("res://scripts/exporter.gd").new()
	ex._build_ui()
	var jpg: String = await ex._take_thumbnail(mp4, 1.5)
	var img := Image.load_from_file(jpg) if not jpg.is_empty() else null
	_ok(img != null and img.get_width() == 1280 and img.get_height() == 720 and jpg.ends_with("clip.thumbnail.jpg"),
		"the exporter takes a 1280x720 frame of the saved video, beside it (%s)" % jpg)
	_ok(await ex._take_thumbnail(mp4 + ".missing.mp4", 1.0) == "", "no video, no thumbnail - and no hang")
	ex.free()
	return true

# --- the stand-in for Google ---------------------------------------------------------------------

class FakeGoogle extends Node:
	var client_id := "flow-check.apps.googleusercontent.com"
	var client_secret := "flow-check-secret"
	var code := "4/0AQ-flow+check"
	var challenge := ""        # from the consent URL the browser opened
	var redirect := ""
	var refresh_ok := true
	var exchanged := 0
	var refreshes := 0
	var status_checks := 0
	var revoked: Array = []
	var thumbs := {}           # video id -> the image bytes set as its thumbnail
	var thumb_forbidden := false
	var sessions := {}         # id -> {meta, declared, type, data, id}
	var order: Array = []      # session ids, oldest first
	var faults := {}           # chunk number -> "503" | "half" | "401"
	var chunks := 0            # chunks received
	var on_chunk := Callable()
	var port := 0
	var _tcp := TCPServer.new()
	var _conns: Array = []
	var _live := {}            # access tokens it honors
	var _issued := 0

	func start() -> int:
		_tcp.listen(0, "127.0.0.1")
		port = _tcp.get_local_port()
		return port

	func last() -> Dictionary:
		return sessions[order[order.size() - 1]] if not order.is_empty() else {}

	func _process(_dt: float) -> void:
		while _tcp.is_connection_available():
			_conns.append({"peer": _tcp.take_connection(), "buf": PackedByteArray()})
		for c in _conns.duplicate():
			var conn: Dictionary = c
			var peer: StreamPeerTCP = conn["peer"]
			peer.poll()
			if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
				_conns.erase(conn)
				continue
			var n := peer.get_available_bytes()
			if n > 0:
				var buf: PackedByteArray = conn["buf"]
				buf.append_array(peer.get_partial_data(n)[1] as PackedByteArray)
				conn["buf"] = buf
			var req := AgentTools.parse_request(conn["buf"])
			if req.is_empty():
				continue
			_conns.erase(conn)
			var out := _route(req)
			peer.put_data(AgentTools.http_response(int(out["status"]), str(out.get("body", "")), out.get("headers", {})))
			peer.disconnect_from_host()

	func _route(req: Dictionary) -> Dictionary:
		var path := str(req.get("path", ""))
		var method := str(req.get("method", ""))
		var headers: Dictionary = req.get("headers", {})
		var body: PackedByteArray = req.get("body", PackedByteArray())
		if method == "POST" and path == "/token":
			return _token(YouTube.query_of("/?" + body.get_string_from_utf8()))
		if method == "POST" and path == "/revoke":
			revoked.append(str(YouTube.query_of("/?" + body.get_string_from_utf8()).get("token", "")))
			return {"status": 200}
		if not _live.has(str(headers.get("authorization", "")).trim_prefix("Bearer ")):
			return _json(401, {"error": {"code": 401, "message": "Invalid Credentials", "errors": [{"reason": "authError"}]}})
		if method == "POST" and path.begins_with("/upload/youtube/v3/videos?"):
			var q := YouTube.query_of(path)
			if q.get("uploadType") != "resumable":
				return _json(400, {"error": {"code": 400, "message": "bad query"}})
			var j := JSON.new()
			if j.parse(body.get_string_from_utf8()) != OK:
				return _json(400, {"error": {"code": 400, "message": "bad metadata"}})
			var id := "vid-%d" % (sessions.size() + 1)
			sessions[id] = {"id": id, "meta": j.data, "parts": str(q.get("part", "")), "declared": int(str(headers.get("x-upload-content-length", "0"))),
				"type": str(headers.get("x-upload-content-type", "")), "data": PackedByteArray()}
			order.append(id)
			return {"status": 200, "headers": {"Location": "http://127.0.0.1:%d/session/%s" % [port, id]}}
		if method == "POST" and path.begins_with("/thumbnails/set?"):
			var tq := YouTube.query_of(path)
			if thumb_forbidden:
				return _json(403, {"error": {"code": 403, "message": "The authenticated user doesn't have permissions to upload and set custom video thumbnails.",
					"errors": [{"reason": "forbidden"}]}})
			if tq.get("uploadType") != "media" or not str(headers.get("content-type", "")).begins_with("image/") or body.is_empty():
				return _json(400, {"error": {"code": 400, "message": "bad thumbnail request"}})
			thumbs[str(tq.get("videoId", ""))] = body
			return _json(200, {"kind": "youtube#thumbnailSetResponse", "items": [{"default": {"url": "x"}}]})
		if method == "PUT" and path.begins_with("/session/"):
			return _chunk(path.trim_prefix("/session/"), str(headers.get("content-range", "")), body)
		return _json(404, {"error": {"code": 404, "message": "no such thing"}})

	func _token(f: Dictionary) -> Dictionary:
		if f.get("client_id") != client_id or f.get("client_secret") != client_secret:
			return _json(401, {"error": "invalid_client", "error_description": "The OAuth client was not found."})
		match str(f.get("grant_type", "")):
			"authorization_code":
				if f.get("code") != code or f.get("redirect_uri") != redirect \
						or YouTube.challenge(str(f.get("code_verifier", ""))) != challenge:
					return _json(400, {"error": "invalid_grant", "error_description": "Bad Request"})
				exchanged += 1
				return _json(200, {"access_token": _issue(), "expires_in": 3599, "refresh_token": "rt-1",
					"scope": YouTube.SCOPE, "token_type": "Bearer"})
			"refresh_token":
				if f.get("refresh_token") != "rt-1" or not refresh_ok:
					return _json(400, {"error": "invalid_grant", "error_description": "Token has been expired or revoked."})
				refreshes += 1
				return _json(200, {"access_token": _issue(), "expires_in": 3599, "scope": YouTube.SCOPE, "token_type": "Bearer"})
		return _json(400, {"error": "unsupported_grant_type"})

	func _chunk(id: String, range_text: String, body: PackedByteArray) -> Dictionary:
		if not sessions.has(id):
			return _json(404, {"error": {"code": 404, "message": "session not found"}})
		var s: Dictionary = sessions[id]
		var data: PackedByteArray = s["data"]
		var total := int(range_text.get_slice("/", 1))
		if range_text.begins_with("bytes */"):
			status_checks += 1
			return _done(s) if data.size() == total else _incomplete(data.size())
		chunks += 1
		var n := chunks
		if on_chunk.is_valid():
			on_chunk.call(n)
		match str(faults.get(n, "")):
			"503":
				return _json(503, {"error": {"code": 503, "message": "Backend Error", "errors": [{"reason": "backendError"}]}})
			"401":
				_live.clear()
				return _json(401, {"error": {"code": 401, "message": "Invalid Credentials", "errors": [{"reason": "authError"}]}})
		var start := int(range_text.trim_prefix("bytes ").get_slice("-", 0))
		if start != data.size():
			return _json(400, {"error": {"code": 400, "message": "chunk at %d, but %d bytes are held" % [start, data.size()]}})
		var keep := body.slice(0, body.size() >> 1) if str(faults.get(n, "")) == "half" else body
		data.append_array(keep)
		s["data"] = data
		return _done(s) if data.size() == total else _incomplete(data.size())

	func _done(s: Dictionary) -> Dictionary:
		var meta: Dictionary = s["meta"]
		return _json(201, {"kind": "youtube#video", "id": s["id"],
			"snippet": {"title": meta["snippet"]["title"], "channelTitle": "Stand-in Channel"},
			"status": {"privacyStatus": meta["status"]["privacyStatus"], "uploadStatus": "uploaded"}})

	func _incomplete(held: int) -> Dictionary:
		return {"status": 308, "headers": {"Range": "bytes=0-%d" % (held - 1)} if held > 0 else {}}

	func _issue() -> String:
		_issued += 1
		var t := "at-%d" % _issued
		_live[t] = true
		return t

	func _json(status: int, value: Variant) -> Dictionary:
		return {"status": status, "body": JSON.stringify(value), "headers": {"Content-Type": "application/json"}}
