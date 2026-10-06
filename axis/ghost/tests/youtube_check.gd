extends SceneTree

## youtube_check - the YouTube upload's pieces that hold without a network: the client file, the
## sign-in's parts, what a video says, the records, the tarot episode's upload notes and the
## export menu's items.
##
##   godot --headless --path . --script res://tests/youtube_check.gd
##
## - THE CLIENT FILE: a "Desktop app" client is read; a Web client, a file with no secret, one that
##   is not JSON and one whose endpoints are not Google's are refused, each saying why. Imported, it
##   is kept owner read/write only; re-importing the same client keeps the sign-in, another ends it,
##   and a bad file leaves the kept one alone.
## - THE SIGN-IN'S PARTS: PKCE against RFC 7636's own example, the consent URL (upload scope only,
##   S256, offline, consent, the loopback redirect - all encoded), the redirect's query decoded.
## - THE WIRE: the next offset from a 308's `Range`, headers in any case, Google's error shapes in
##   words.
## - WHAT A VIDEO SAYS: `<`/`>` made safe, the title one line of at most 100 characters cut at a
##   word, the description at most 5000 bytes, tags split as a document writes them, each once,
##   counted as YouTube counts them and fitted to 500; unlisted, not for kids, synthetic, no paid
##   promotion, People & Blogs.
## - RECORDS: an upload recorded and read back; a queued upload pending, owner-only; a missing file
##   refused.
## - THE TAROT EPISODE: its upload notes from the plan, with a chapter per card timed from a take's
##   word timings (an hour on, `h:mm:ss`), and `upload.md` with the show's tags first.
## - THE EXPORT MENU: "Upload to YouTube" only when the mode describes something, unticked, "again"
##   once uploaded; a resume while an upload waits on a file that exists; a sign-out while signed in.
## - THE TAG FIELD (tag_field.gd, 2026-10-06, the user: "show all of the tags, with the ability to
##   click an 'x' to remove them on each one... comma-delimited splits the tags into the button with
##   the x"): a chip per tag, filled without a change signal; a comma, Enter or a paste of "a, b, c"
##   makes chips and leaves the rest typed; × and Backspace in the empty box remove; each tag once
##   whatever its case; fixed tags first with no ×; a dimmed chip says why; read-only has no ×; a long
##   tag cannot widen the field.
## - THE TAROT PANEL'S FIELDS: the title and description filled from the plan and edits written back
##   into it (an emptied title is not; nothing is written without an edit); THE TAGS ARE THE SHOW'S
##   (the user: "I want global tags, not per-episode ones"), the chips editing its `tags:` field; the
##   byline into its `byline:`; nothing per episode in the document's block but the picked episode's
##   title, one value; the upload described from all that, with the title screen's thumbnail moment,
##   for the episode an export rendered even after another is picked.
## - A REAL SHOW FILE: the tags and the byline land as top-level lines above ghost's block - the
##   chapter format - the block holds the picked episode's title and no record per episode, and the
##   rest of the file is untouched.
## - THE THUMBNAIL'S MOMENT and ffmpeg's arguments for taking it.

## By path, not by class name: a new class is unknown to `--script` runs until the editor rescans.
const YouTube := preload("res://scripts/youtube.gd")
const ROOT := "user://youtube_check"
const TAROT := "user://youtube_check_tarot"

var _fails := 0


func _init() -> void:
	# deferred: the exporter needs the autoloads, which are only up once the tree is
	_run.call_deferred()


func _run() -> void:
	# A GATE NEVER WRITES THE SETTINGS OF WHOEVER RUNS IT: a --script run is not read-only by itself,
	# and the panel built below binds its controls to them
	var settings := root.get_node_or_null("Settings")
	if settings != null:
		settings.set("_read_only", true)
	var keep_root: String = YouTube.root
	var keep_tarot := TarotEpisode.root
	YouTube.root = ROOT
	TarotEpisode.root = TAROT
	_wipe(ROOT)
	_wipe(TAROT)
	_wipe(ROOT + "_files")
	for check in [_client, _import, _pkce, _auth_url, _query, _wire, _fit, _body, _records, _episode_notes, _menu,
			_tag_field, _panel_fields, _doc_file]:
		if not bool(check.call()):
			_ok(false, "a check stopped part way (script error?)")
	_wipe(ROOT)
	_wipe(TAROT)
	_wipe(ROOT + "_files")
	YouTube.root = keep_root
	TarotEpisode.root = keep_tarot
	print("youtube_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	quit(0 if _fails == 0 else 1)


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _wipe(path: String) -> void:
	var dir := ProjectSettings.globalize_path(path)
	if not DirAccess.dir_exists_absolute(dir):
		return
	for d in DirAccess.get_directories_at(dir):
		_wipe(path.path_join(d))
	for f in DirAccess.get_files_at(dir):
		DirAccess.remove_absolute(dir.path_join(f))
	DirAccess.remove_absolute(dir)


## A file under the scratch folder, written whole; its absolute path.
func _file(name: String, text: String) -> String:
	var path := ProjectSettings.globalize_path(ROOT + "_files").path_join(name)
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()
	return path


func _client_json(id := "123-abc.apps.googleusercontent.com", kind := "installed", extra := {}) -> String:
	var c := {"client_id": id, "project_id": "ghost-check", "auth_uri": "https://accounts.google.com/o/oauth2/auth",
		"token_uri": "https://oauth2.googleapis.com/token",
		"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
		"client_secret": "GOCSPX-not-a-real-secret", "redirect_uris": ["http://localhost"]}
	c.merge(extra, true)
	return JSON.stringify({kind: c})


func _owner_only(path: String) -> bool:
	return OS.get_name() == "Windows" or FileAccess.get_unix_permissions(path) == YouTube.OWNER_FILE


func _client() -> bool:
	var c: Dictionary = YouTube.parse_client(_client_json())
	_ok(not c.has("error") and c["client_id"] == "123-abc.apps.googleusercontent.com"
		and c["client_secret"] == "GOCSPX-not-a-real-secret" and c["project_id"] == "ghost-check"
		and c["token_uri"] == "https://oauth2.googleapis.com/token", "a Desktop app client file is read")
	var web: Dictionary = YouTube.parse_client(_client_json("x", "web"))
	_ok(String(web.get("error", "")).contains("Desktop app"), "a Web application client is refused, naming the kind wanted")
	var bare: Dictionary = YouTube.parse_client(_client_json("x", "installed", {"client_secret": ""}))
	_ok(String(bare.get("error", "")).contains("client_secret"), "a client file with no secret is refused")
	_ok(YouTube.parse_client("not json at all").has("error") and YouTube.parse_client("").has("error")
		and YouTube.parse_client("[1, 2]").has("error"), "a file that is not a client is refused")
	var evil: Dictionary = YouTube.parse_client(_client_json("x", "installed", {"token_uri": "https://evil.example/token"}))
	_ok(String(evil.get("error", "")).contains("token_uri"), "a token endpoint that is not Google's is refused")
	var lookalike: Dictionary = YouTube.parse_client(_client_json("x", "installed",
		{"token_uri": "https://oauth2.googleapis.com.evil.example/token"}))
	_ok(lookalike.has("error"), "a host that only begins like Google's is refused")
	var plain: Dictionary = YouTube.parse_client(_client_json("x", "installed", {"token_uri": "http://oauth2.googleapis.com/token"}))
	_ok(plain.has("error"), "Google over plain http is refused")
	var local: Dictionary = YouTube.parse_client(_client_json("x", "installed", {"token_uri": "http://127.0.0.1:9/token"}))
	_ok(not local.has("error"), "this machine over http is allowed (a gate standing in for Google)")
	var defaults: Dictionary = YouTube.parse_client(JSON.stringify({"installed": {"client_id": "a", "client_secret": "b"}}))
	_ok(defaults.get("auth_uri") == "https://accounts.google.com/o/oauth2/auth"
		and defaults.get("token_uri") == "https://oauth2.googleapis.com/token", "missing endpoints default to Google's")
	return true


func _import() -> bool:
	_ok(not YouTube.has_client() and String(YouTube.client().get("error", "")).contains("imported"),
		"no client before one is imported")
	_ok(YouTube.import_client(_file("nope/missing.json", "")) != "", "an empty file is refused")
	_ok(YouTube.import_client(ProjectSettings.globalize_path(ROOT + "_files/absent.json")) == "that file does not exist",
		"a missing file is refused")
	var good := _file("client_secret_123.json", _client_json())
	_ok(YouTube.import_client(good) == "", "a good client file imports")
	var kept := ProjectSettings.globalize_path(ROOT.path_join("client.json"))
	_ok(YouTube.has_client() and YouTube.client()["client_id"] == "123-abc.apps.googleusercontent.com",
		"the imported client is what is read from then on")
	_ok(_owner_only(kept), "the kept client is owner read/write only (%o)" % FileAccess.get_unix_permissions(kept))
	_ok(OS.get_name() == "Windows" or FileAccess.get_unix_permissions(ProjectSettings.globalize_path(ROOT)) == YouTube.OWNER_DIR,
		"its folder is owner-only too")
	_ok(not FileAccess.get_file_as_string(kept).contains("redirect_uris"), "only what the flow needs is kept")
	YouTube._write(YouTube._path("token.json"), JSON.stringify({"refresh_token": "rt"}), true)
	_ok(YouTube.import_client(good) == "" and YouTube.signed_in(), "re-importing the same client keeps the sign-in")
	_ok(YouTube.import_client(_file("web.json", _client_json("x", "web"))) != ""
		and YouTube.client()["client_id"] == "123-abc.apps.googleusercontent.com" and YouTube.signed_in(),
		"a bad file leaves the kept client and sign-in alone")
	_ok(YouTube.import_client(_file("other.json", _client_json("456-def.apps.googleusercontent.com"))) == ""
		and not YouTube.signed_in(), "another client ends the sign-in made with the old one")
	YouTube.forget_client()
	_ok(not YouTube.has_client() and not YouTube.signed_in(), "forgetting the client forgets the sign-in with it")
	return true


func _pkce() -> bool:
	_ok(YouTube.challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
		"the S256 challenge matches RFC 7636's example")
	var v: String = YouTube.verifier()
	var allowed := RegEx.create_from_string("^[A-Za-z0-9._~-]{43,128}$")
	_ok(allowed.search(v) != null, "a verifier is 43-128 characters of the unreserved alphabet (%d)" % v.length())
	_ok(YouTube.verifier() != v, "each verifier is fresh")
	_ok(YouTube.base64url(PackedByteArray([251, 255, 254])) == "-__-", "base64url swaps + and / and drops the padding")
	return true


func _auth_url() -> bool:
	var c: Dictionary = YouTube.parse_client(_client_json())
	var url: String = YouTube.auth_url(c, "http://127.0.0.1:5555", "CHALLENGE", "st4te")
	_ok(url.begins_with("https://accounts.google.com/o/oauth2/auth?"), "the consent URL is the client's auth_uri")
	var q: Dictionary = YouTube.query_of(url)
	_ok(q.get("client_id") == "123-abc.apps.googleusercontent.com" and q.get("response_type") == "code"
		and q.get("redirect_uri") == "http://127.0.0.1:5555" and q.get("state") == "st4te"
		and q.get("code_challenge") == "CHALLENGE" and q.get("code_challenge_method") == "S256",
		"it carries the client, the loopback redirect, the state and the S256 challenge")
	_ok(q.get("scope") == YouTube.SCOPE and YouTube.SCOPE == "https://www.googleapis.com/auth/youtube.upload",
		"it asks for the upload scope and nothing else")
	_ok(q.get("access_type") == "offline" and q.get("prompt") == "consent", "offline access, the consent screen shown")
	_ok(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A5555")
		and url.contains("scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutube.upload&"),
		"every value is encoded")
	return true


func _query() -> bool:
	var q: Dictionary = YouTube.query_of("/?state=abc&code=4%2F0AQlEd8x-yz_W&scope=a+b%20c&empty=&flag")
	_ok(q.get("state") == "abc" and q.get("code") == "4/0AQlEd8x-yz_W", "a redirect's code is decoded")
	_ok(q.get("scope") == "a b c" and q.get("empty") == "" and q.has("flag"), "+ and %20 are spaces; empty and bare keys kept")
	_ok(YouTube.query_of("/favicon.ico").is_empty(), "a path with no query has none")
	var f: String = YouTube.form({"a b": "c&d=e", "token": "1/x+y"})
	_ok(f == "a%20b=c%26d%3De&token=1%2Fx%2By", "a form body encodes keys and values")
	_ok(YouTube.query_of("?" + f) == {"a b": "c&d=e", "token": "1/x+y"}, "and decodes back")
	return true


func _wire() -> bool:
	_ok(YouTube.next_offset(PackedStringArray(["Content-Length: 0", "Range: bytes=0-262143"])) == 262144,
		"a 308's Range says where to carry on")
	_ok(YouTube.next_offset(PackedStringArray(["range: bytes=0-0"])) == 1, "header names in any case")
	_ok(YouTube.next_offset(PackedStringArray(["Content-Length: 0"])) == 0, "no Range: nothing is held yet")
	_ok(YouTube.header(PackedStringArray(["LOCATION: https://x/y?upload_id=1"]), "location") == "https://x/y?upload_id=1",
		"a header is found whatever its case")
	var quota := JSON.stringify({"error": {"code": 403, "message": "The request cannot be completed because you have exceeded your quota.",
		"errors": [{"reason": "quotaExceeded", "domain": "youtube.quota"}]}}).to_utf8_buffer()
	_ok(String(YouTube.error_text(403, quota)).contains("quota is used up"), "a spent quota is said plainly")
	var odd := JSON.stringify({"error": {"code": 400, "message": "Bad thing", "errors": [{"reason": "weird"}]}}).to_utf8_buffer()
	_ok(YouTube.error_text(400, odd) == "YouTube said: Bad thing (HTTP 400, weird)", "any other API error keeps its message and reason")
	var oauth := JSON.stringify({"error": "invalid_grant", "error_description": "Token has been expired or revoked."}).to_utf8_buffer()
	_ok(YouTube.error_text(400, oauth) == "Google said: invalid_grant - Token has been expired or revoked."
		and YouTube.oauth_error(oauth) == "invalid_grant", "the token endpoint's errors are read")
	_ok(YouTube.error_text(502, "<html>".to_utf8_buffer()) == "YouTube answered HTTP 502" and YouTube.oauth_error(quota) == "",
		"a reply that is not JSON still says its status")
	return true


func _fit() -> bool:
	_ok(YouTube.clean("a <b> c\u0007\r\n\td") == "a ‹b› c\n\td", "< and > are made safe; control characters but breaks and tabs go")
	_ok(YouTube.fit_title("  Why <you> keep\nwaking   up  ") == "Why ‹you› keep waking up", "a title is one clean line")
	var words := "word ".repeat(30)
	var t: String = YouTube.fit_title(words)
	_ok(t.length() <= 100 and t.ends_with("word") and not t.ends_with(" "), "a long title is cut at a word, within 100 (%d)" % t.length())
	var unbroken := "x".repeat(150)
	_ok(YouTube.fit_title(unbroken).length() == 100, "a title with no space is cut at 100")
	_ok(YouTube.fit_title("Ünïcödé · 🜁 tarot") == "Ünïcödé · 🜁 tarot", "a title keeps what it says")
	var para := ("é".repeat(40) + "\n").repeat(80)
	var d: String = YouTube.fit_description(para)
	_ok(d.to_utf8_buffer().size() <= 5000 and d.ends_with("é") and para.begins_with(d), "a long description fits 5000 bytes, cut at a line (%d bytes)" % d.to_utf8_buffer().size())
	_ok(YouTube.fit_description("short\n\n0:00 Intro") == "short\n\n0:00 Intro", "a short description is untouched")
	_ok(Array(YouTube.split_tags("\"Oregon, Nexpo, Down the Rabbit Hole, Hilbert's hotel\"")) == ["Oregon", "Nexpo", "Down the Rabbit Hole", "Hilbert's hotel"],
		"tags split as a document's tags: line holds them")
	_ok(Array(YouTube.split_tags("[tarot, 'pick a card', \"\", ]")) == ["tarot", "pick a card"], "a YAML list too; empties dropped")
	_ok(YouTube.split_tags("").is_empty(), "no tags from nothing")
	_ok(YouTube.tags_length(PackedStringArray(["ab", "c d"])) == 2 + 1 + 3 + 2, "length counts the commas and the quotes round a spaced tag")
	var fitted: PackedStringArray = YouTube.fit_tags(["Tarot", "tarot", " pick   a card ", "a, b", "say \"hi\"", "<3", ""])
	_ok(Array(fitted) == ["Tarot", "pick a card", "a b", "say hi", "‹3"], "tags are cleaned, each once whatever its case (%s)" % str(fitted))
	var many: Array = []
	for i in 60:
		many.append("tag number %02d" % i)
	many.append("tiny")
	var fit: PackedStringArray = YouTube.fit_tags(many)
	_ok(YouTube.tags_length(fit) <= 500 and fit.size() > 20 and fit[fit.size() - 1] == "tiny",
		"tags fill 500 characters, and a short one after the room ran out still fits (%d)" % YouTube.tags_length(fit))
	return true


func _body() -> bool:
	var b: Dictionary = YouTube.video_body({"title": "A <title>", "description": "Hello\n\n0:00 Intro",
		"tags": PackedStringArray(["tarot", "TAROT", "pick a card"])})
	_ok(b["snippet"]["title"] == "A ‹title›" and b["snippet"]["description"] == "Hello\n\n0:00 Intro"
		and b["snippet"]["tags"] == ["tarot", "pick a card"] and b["snippet"]["categoryId"] == "22",
		"the snippet is fitted, the category People & Blogs")
	_ok(b.get("paidProductPlacementDetails") == {"hasPaidProductPlacement": false},
		"no paid promotion is declared, not left unanswered")
	_ok(b["status"] == {"privacyStatus": "unlisted", "selfDeclaredMadeForKids": false, "containsSyntheticMedia": true},
		"unlisted, not made for kids, disclosed as synthetic")
	_ok(YouTube.video_body({}, "Episode 7")["snippet"]["title"] == "Episode 7"
		and YouTube.video_body({})["snippet"]["title"] == "Untitled", "a video always has a title")
	_ok(YouTube.video_body({"tags": ["a", "b"]})["snippet"]["tags"] == ["a", "b"], "tags as an Array too")
	return true


func _records() -> bool:
	var rec := ProjectSettings.globalize_path(ROOT + "_files/ep/youtube.json")
	_ok(YouTube.uploads_in(rec).is_empty() and YouTube.uploads_in("").is_empty(), "no record, no uploads")
	_ok(YouTube.record_upload(rec, {"id": "a", "url": "https://youtu.be/a"}) == ""
		and YouTube.record_upload(rec, {"id": "b", "url": "https://youtu.be/b"}) == "", "uploads are recorded")
	var ups: Array = YouTube.uploads_in(rec)
	_ok(ups.size() == 2 and ups[0]["id"] == "a" and ups[1]["id"] == "b", "and read back oldest first")
	_ok(YouTube.queue(ProjectSettings.globalize_path(ROOT + "_files/none.mp4"), {}).has("error")
		and YouTube.pending().is_empty(), "a missing video is not queued")
	var video := _file("video.mp4", "0123456789")
	var p: Dictionary = YouTube.queue(video, {"snippet": {"title": "T"}}, rec)
	var back: Dictionary = YouTube.pending()
	_ok(not p.has("error") and back.get("file") == video and int(back.get("size", 0)) == 10
		and back.get("record") == rec and back.get("session") == "" and back["body"]["snippet"]["title"] == "T",
		"a queued upload waits in pending.json, with no session yet")
	_ok(_owner_only(YouTube._path("pending.json")), "pending.json is owner read/write only (it will hold the session)")
	YouTube.forget_pending()
	_ok(YouTube.pending().is_empty(), "and is forgotten")
	return true


## An episode on disk: a plan, a draw of two cards, a script with their marks, and a take whose
## words put the spread past the hour.
func _episode_notes() -> bool:
	var ep := TarotEpisode.open("youtube-check", 7)
	DirAccess.make_dir_recursive_absolute(ep.dir)
	ep.write_json("plan", {"episode_title": "Why You Keep Waking Up At 4AM",
		"description": "Hello, my loves.\n\nThis one is for you.", "premise": "unused",
		"tags": ["tarot", "4am", "timeless reading"],
		"spread": {"name": "Two", "positions": [{"name": "Past", "asks": ""}, {"name": "Future", "asks": ""}]}})
	ep.write_json("draw", {"cards": [{"name": "The Tower", "reversed": true}, {"name": "The Star", "reversed": false}]})
	ep.write_text("script", "<!-- tarot: shuffle -->\n\nHello my loves welcome back.\n\n<!-- tarot: draw 1 -->\n\n"
		+ "The Tower is here.\n\n<!-- tarot: draw 2 -->\n\nThen the Star.\n\n<!-- tarot: spread -->\n\nThat is the spread.\n")
	var words: Array = []
	var at := {0: 1.0, 5: 70.0, 9: 130.0, 12: 3700.0}
	var t := 1.0
	var i := 0
	for w in "Hello my loves welcome back The Tower is here Then the Star That is the spread".split(" "):
		t = float(at.get(i, t))
		words.append({"text": w, "t0": t, "t1": t + 0.4})
		t += 0.5
		i += 1
	var take := ProjectSettings.globalize_path(ROOT + "_files/take_1.wav")
	var side := FileAccess.open(take.get_basename() + ".json", FileAccess.WRITE)
	side.store_string(JSON.stringify({"words": words}))
	side.close()

	var bare: Dictionary = ep.upload_notes()
	_ok(bare["title"] == "Why You Keep Waking Up At 4AM" and bare["description"] == "Hello, my loves.\n\nThis one is for you."
		and Array(bare["tags"]) == ["tarot", "4am", "timeless reading"] and (bare["chapters"] as PackedStringArray).is_empty(),
		"the upload notes are the plan's title, description and tags; no take, no chapters")
	var n: Dictionary = ep.upload_notes(take)
	var ch: PackedStringArray = n["chapters"]
	_ok(ch.size() == 4 and ch[0] == "0:00 Intro", "a chapter per card and the spread, from 0:00 (%s)" % str(ch))
	if ch.size() == 4:
		_ok(ch[1].ends_with(" Past - The Tower (reversed)") and ch[1].begins_with("1:0"), "a card's chapter names its position and how it fell (%s)" % ch[1])
		_ok(ch[2].ends_with(" Future - The Star") and ch[2].begins_with("2:0"), "in the order drawn (%s)" % ch[2])
		_ok(RegEx.create_from_string("^1:0\\d:\\d\\d The spread$").search(ch[3]) != null, "an hour on it reads h:mm:ss (%s)" % ch[3])
	_ok(TarotEpisode.chapter_clock(0) == "0:00" and TarotEpisode.chapter_clock(65) == "1:05"
		and TarotEpisode.chapter_clock(3725) == "1:02:05", "chapter clocks read as YouTube reads them")
	ep.write_json("plan", {"episode_title": "T", "description": "", "premise": "The premise."})
	_ok(ep.upload_notes()["description"] == "The premise.", "no description: the premise stands in")
	ep.write_json("plan", {"episode_title": "Why You Keep Waking Up At 4AM", "description": "Hello.",
		"tags": ["tarot", "Truthful Tarot", "4am"],
		"spread": {"name": "Two", "positions": [{"name": "Past", "asks": ""}, {"name": "Future", "asks": ""}]}})
	_ok(ep.write_upload_notes(take, PackedStringArray(["Truthful Tarot", "satire"])) == "", "upload.md is written")
	var md := ep.read_text("upload")
	_ok(md.begins_with("# Why You Keep Waking Up At 4AM\n\nHello.\n\nChapters\n0:00 Intro\n"), "upload.md: title, description, chapters")
	_ok(md.contains("\nTags: Truthful Tarot, satire\n"), "its tags are the show's, as they go up")
	_ok(ep.write_upload_notes(ProjectSettings.globalize_path(ROOT + "_files/no_take.wav")) != "", "no take's sidecar, no notes")
	_ok(ep.uploads().is_empty(), "an episode never uploaded has no uploads")
	YouTube.record_upload(ep.file_of("youtube"), {"id": "v1", "url": "https://youtu.be/v1"})
	_ok(ep.uploads().size() == 1 and ep.file_of("youtube").ends_with("/youtube.json"), "its uploads are recorded in its folder")
	return true


func _menu() -> bool:
	var ex = load("res://scripts/exporter.gd").new()
	ex._build_ui()            # not added to the tree: its _ready would clear a live render's override.cfg
	var menu: PopupMenu = ex._quality_menu
	ex.upload_provider = Callable()
	ex._refresh_upload_items()
	_ok(menu.get_item_index(ex.UPLOAD_ID) < 0, "no upload offered when the mode describes nothing")
	ex.upload_provider = func(_take: String) -> Dictionary: return {}
	ex._refresh_upload_items()
	_ok(menu.get_item_index(ex.UPLOAD_ID) < 0, "nor when it describes nothing yet (no plan)")
	var rec := ProjectSettings.globalize_path(ROOT + "_files/menu/youtube.json")
	ex.upload_provider = func(_take: String) -> Dictionary: return {"title": "An <Episode>", "record": rec}
	ex._refresh_upload_items()
	var at: int = menu.get_item_index(ex.UPLOAD_ID)
	_ok(at >= 0 and menu.is_item_checkable(at) and not menu.is_item_checked(at)
		and menu.get_item_text(at) == "Upload to YouTube (unlisted)", "a mode with something to upload offers the box, unticked")
	_ok(menu.get_item_tooltip(at).contains("An ‹Episode›") and menu.get_item_tooltip(at).contains("client file"),
		"its tooltip names the video and, with no client yet, the file it will ask for")
	ex._upload = true
	ex._refresh_upload_items()
	_ok(menu.is_item_checked(menu.get_item_index(ex.UPLOAD_ID)), "the box shows the choice it holds")
	YouTube.record_upload(rec, {"id": "z", "url": "https://youtu.be/z", "privacy": "unlisted", "at": "2026-10-06T01:02:03"})
	ex._refresh_upload_items()
	at = menu.get_item_index(ex.UPLOAD_ID)
	_ok(menu.get_item_text(at) == "Upload to YouTube again (unlisted)" and menu.get_item_tooltip(at).contains("https://youtu.be/z"),
		"once uploaded, the box says so and where")
	_ok(menu.get_item_index(ex.RESUME_ID) < 0 and menu.get_item_index(ex.SIGN_OUT_ID) < 0, "no resume or sign-out with nothing waiting")
	_ok(menu.get_item_index(ex.CLIENT_ID) < 0, "no other client file offered before one is kept")
	var video := _file("menu.mp4", "abc")
	YouTube.queue(video, {}, "")
	ex._refresh_upload_items()
	at = menu.get_item_index(ex.RESUME_ID)
	_ok(at >= 0 and menu.get_item_text(at).ends_with("menu.mp4"), "a waiting upload can be resumed")
	DirAccess.remove_absolute(video)
	ex._refresh_upload_items()
	_ok(menu.get_item_index(ex.RESUME_ID) < 0, "not once its file is gone")
	YouTube.forget_pending()
	YouTube._write(YouTube._path("token.json"), JSON.stringify({"refresh_token": "rt"}), true)
	ex._refresh_upload_items()
	_ok(menu.get_item_index(ex.SIGN_OUT_ID) >= 0, "a kept sign-in can be signed out of")
	_ok(menu.get_item_tooltip(menu.get_item_index(ex.UPLOAD_ID)).contains("client file"),
		"with no client the tooltip still asks for one first")
	YouTube.import_client(_file("menu_client.json", _client_json()))
	YouTube._write(YouTube._path("token.json"), JSON.stringify({"refresh_token": "rt"}), true)
	ex._upload = false
	ex._refresh_upload_items()
	ex._toggle_upload()
	_ok(ex._upload and menu.is_item_checked(menu.get_item_index(ex.UPLOAD_ID)), "with a client kept, the box simply ticks")
	_ok(menu.get_item_tooltip(menu.get_item_index(ex.UPLOAD_ID)).contains("signed in"), "and says it is signed in")
	_ok(menu.get_item_index(ex.CLIENT_ID) >= 0, "a kept client can be replaced from the menu")
	ex.free()
	return true


## Built by hand, as panel_fit_check builds it: its _ready would start the voice host.
func _panel_fields() -> bool:
	# loaded here, not named: the panel's scripts need the autoloads, which are only up by now
	var ed = load("res://scripts/tarot_editor.gd").new()
	ed._build_panel()
	var ep := TarotEpisode.open("panel-check", 21)
	ep.write_json("plan", {"episode_title": "The Producer's Title", "description": "The producer's words.",
		"tags": ["tarot", "4am"], "premise": "p", "spread": {"positions": [{"name": "Past"}]}})
	ed._knobs["show"] = "panel-check"
	ed._knobs["seed"] = 21
	ed._open_episode()
	_ok(ed._yt_title.text == "The Producer's Title" and ed._yt_desc.text == "The producer's words."
		and ed._yt_title.editable, "the title and description are the plan's")
	_ok(ed._yt_tags.get_tags().is_empty() and ed._yt_note.text.contains("No tags yet"),
		"the tags are the show's, not the episode's: none until the show has some")
	var before := ep.read_text("plan")
	ed._flush_upload_fields()
	_ok(ep.read_text("plan") == before, "nothing is written without an edit")
	ed._yt_title.text = "  My Own Title  "
	ed._yt_desc.text = "Mine."
	ed._upload_edited()
	_ok(ed._yt_dirty > 0.0, "an edit waits a moment before it is written")
	ed._flush_upload_fields(true)
	var plan: Dictionary = ep.read_json("plan")
	_ok(plan["episode_title"] == "My Own Title" and plan["description"] == "Mine." and plan["tags"] == ["tarot", "4am"]
		and plan["premise"] == "p" and (plan["spread"] as Dictionary).has("positions"),
		"an edit is written into the plan, the rest of it kept")
	_ok(ed.export_name() == "My Own Title", "and names the export")
	ed._yt_title.text = "   "
	ed._upload_edited()
	ed._flush_upload_fields(true)
	_ok(String((ep.read_json("plan") as Dictionary)["episode_title"]) == "My Own Title", "an emptied title is not written")
	# THE SHOW'S TAGS, typed as chips: the document's own `tags:` field
	ed._yt_tags._input.text = "Truthful Tarot, satire, "
	ed._yt_tags._on_typed(ed._yt_tags._input.text)
	ed._yt_tags._input.text = "pick a card"
	ed._yt_tags.commit()
	_ok(ed._doc.field("tags") == "Truthful Tarot, satire, pick a card", "the chips write the show's tags line (%s)" % ed._doc.field("tags"))
	ed._yt_tags.remove("satire")
	_ok(ed._doc.field("tags") == "Truthful Tarot, pick a card", "× takes a tag out of the show's line")
	var meta: Dictionary = ed.upload_meta("")
	_ok(meta.get("title") == "My Own Title" and meta.get("description") == "Mine."
		and meta.get("tags") == ["Truthful Tarot", "pick a card"] and String(meta.get("record", "")).ends_with("/21/youtube.json"),
		"the upload: the episode's title and description, the show's tags - not the episode's")
	_ok(is_equal_approx(float(meta.get("thumbnail_at", -1.0)), ed.thumbnail_moment(ed._take_intro(""))),
		"and the title screen's moment for its thumbnail")
	# THE BYLINE: the document's `byline:`, and on the title screen's document
	ed._byline.text = "with Pen & Ink"
	ed._save_byline()
	_ok(ed._doc.field("byline") == "with Pen & Ink" and ed.book_document()["byline"] == "with Pen & Ink",
		"the byline is the show's `byline:`, handed to the title screen")
	# NOTHING PER EPISODE in the block but the picked episode's title, one value
	var doc: Dictionary = ed._doc_capture()
	_ok(not doc.has("youtube") and doc.get("episode_title") == "My Own Title",
		"the block keeps no record per episode - only the picked episode's title (%s)" % str(doc.get("episode_title")))
	var other := TarotEpisode.open("panel-check", 22)
	other.write_json("plan", {"episode_title": "Another Episode"})
	var take := ProjectSettings.globalize_path(ROOT + "_files/take_9.wav")
	ed._taken = {"take": take, "episode": ep}
	ed._knobs["seed"] = 22
	ed._open_episode()
	_ok(ed._yt_title.text == "Another Episode" and ed._doc_capture().get("episode_title") == "Another Episode",
		"another episode picked: its title, and it overwrites the block's one title")
	_ok(Array(ed._yt_tags.get_tags()) == ["Truthful Tarot", "pick a card"], "the show's tags stay with the show")
	_ok(ed.upload_meta(take).get("title") == "My Own Title" and ed.upload_meta("").get("title") == "Another Episode",
		"an export's upload describes the episode it rendered, not the one picked since")
	var empty := TarotEpisode.open("panel-check", 23)
	ed._knobs["seed"] = 23
	ed._open_episode()
	_ok(not ed._yt_title.editable and ed.upload_meta("").is_empty() and ed._yt_note.text.contains("plan first"),
		"an episode with no plan has nothing to upload and says why")
	_ok(not DirAccess.dir_exists_absolute(empty.dir), "and nothing was written for it")
	_ok(ed.thumbnail_moment(9.0) == 3.6 and ed.thumbnail_moment(3.0) == 1.5 and ed.thumbnail_moment(0.5) == 1.5,
		"the thumbnail is taken with the name up and the table still out of focus")
	var args: PackedStringArray = load("res://scripts/exporter.gd").thumbnail_args("/v/a b.mp4", "/v/a b.thumbnail.jpg", 3.6)
	_ok(Array(args) == ["-y", "-nostdin", "-loglevel", "error", "-ss", "3.600", "-i", "/v/a b.mp4", "-frames:v", "1",
		"-vf", "scale=1280:720:flags=lanczos", "-q:v", "2", "/v/a b.thumbnail.jpg"], "one frame, at the moment, 1280x720")
	ed.free()
	return true


func _tag_field() -> bool:
	var f = load("res://scripts/tag_field.gd").new()
	var changes := [0]
	f.changed.connect(func() -> void: changes[0] += 1)
	var chips := func() -> Array:
		return f.get_children().filter(func(c: Node) -> bool: return c is PanelContainer)
	f.set_tags(PackedStringArray(["tarot", "Tarot", "  4am "]))
	_ok(Array(f.get_tags()) == ["tarot", "4am"] and changes[0] == 0 and (chips.call() as Array).size() == 2,
		"filled: a chip per tag, each once, trimmed - and no change signal")
	_ok(f.get_child(f.get_child_count() - 1) == f._input, "the typing box comes after the chips")
	f._input.text = "pick a card, timeless"
	f._on_typed(f._input.text)
	_ok(Array(f._tags) == ["tarot", "4am", "pick a card"] and f._input.text == "timeless" and changes[0] == 1,
		"a comma makes a chip of what came before it, and the rest stays typed")
	_ok(Array(f.get_tags()).back() == "timeless", "what is still typed counts as a tag")
	f.commit()
	_ok(Array(f._tags).back() == "timeless" and f._input.text.is_empty(), "Enter (or leaving the box) makes it a chip")
	f._input.text = "x, \"y\", , TAROT, z,"
	f._on_typed(f._input.text)
	_ok(Array(f._tags).slice(-3) == ["x", "y", "z"] and f._input.text.is_empty(),
		"a pasted list splits into chips, quotes and empties dropped, a tag already there not doubled")
	var before: int = changes[0]
	var row: Control = (chips.call() as Array)[1]
	var x: Button = row.find_children("*", "Button", true, false)[0]
	x.pressed.emit()
	_ok(not f._has(f._tags, "4am") and changes[0] == before + 1, "× removes its tag")
	var k := InputEventKey.new()
	k.keycode = KEY_BACKSPACE
	k.pressed = true
	f._on_key(k)
	_ok(not f._has(f._tags, "z"), "Backspace in the empty box removes the last chip")
	f.set_fixed(PackedStringArray(["Truthful Tarot"]), "The show's own tag.")
	var first: Control = (chips.call() as Array)[0]
	_ok(first.find_children("*", "Button", true, false).is_empty() and first.tooltip_text.contains("show's own"),
		"a fixed tag comes first, with no ×, saying where it is kept")
	f.mark({"X": "it is already there"})
	var marked: Control = null
	for c in chips.call():
		if ((c as Control).find_children("*", "Label", true, false)[0] as Label).text == "x":
			marked = c
	_ok(marked != null and marked.modulate.a < 1.0 and marked.tooltip_text.contains("it is already there"),
		"a chip that will not go up is dimmed, saying why")
	f.editable = false
	var buttons := 0
	for c in chips.call():
		buttons += (c as Control).find_children("*", "Button", true, false).size()
	_ok(buttons == 0 and not f._input.editable, "read-only: no × and no typing")
	f.editable = true
	f.set_tags(PackedStringArray(["a very long tag ".repeat(12).strip_edges()]))
	var long_chip: Control = (chips.call() as Array).back()
	_ok(long_chip.get_combined_minimum_size().x <= f.MAX_CHIP + 40.0,
		"a long tag is cut with an ellipsis instead of widening the field (%.0f px)" % long_chip.get_combined_minimum_size().x)
	f.free()
	return true


## A real show file: opened, its tags and byline edited, written by the document's own writer, read back.
func _doc_file() -> bool:
	var ed = load("res://scripts/tarot_editor.gd").new()
	ed._build_panel()
	var ep := TarotEpisode.open("doc-check", 31)
	ep.write_json("plan", {"episode_title": "Episode Thirty-One", "description": "The producer's words.",
		"tags": ["episode", "only"], "spread": {"positions": [{"name": "Past"}]}})
	var body := "# The brief\n\nWhat the show is.\n"
	var path := _file("show.md", "---\ntitle: Doc Check Show\nghost:\n  tarot:\n    show: doc-check\n    seed: 31\n---\n" + body)
	ed._doc._on_picked(path)
	_ok(ed._doc.is_sync() and int(ed._knobs["seed"]) == 31 and ed._yt_title.text == "Episode Thirty-One",
		"the show file opens on its episode")
	ed._yt_tags._input.text = "tarot, pick a card,"
	ed._yt_tags._on_typed(ed._yt_tags._input.text)
	ed._byline.text = "with Pen & Ink"
	ed._save_byline()
	_ok(ed._doc.save(), "the panel's block is written into the show file")
	var raw := FileAccess.get_file_as_string(path)
	var head := raw.substr(0, raw.find("\n---\n", 4))
	_ok(head.begins_with("---\ntitle: Doc Check Show\n") and head.contains("\ntags: tarot, pick a card\n")
		and head.contains("\nbyline: with Pen & Ink\n") and head.find("tags:") < head.find("ghost:")
		and head.find("byline:") < head.find("ghost:"),
		"the tags and the byline are the file's own lines, above ghost's block, as a chapter keeps them:\n%s" % head)
	var data: Variant = FrontMatter.read_block(raw).get("data")
	var tarot: Dictionary = (data as Dictionary).get("tarot", {}) if data is Dictionary else {}
	_ok(tarot.get("episode_title") == "Episode Thirty-One" and not tarot.has("youtube"),
		"the block holds the picked episode's title and no record per episode")
	_ok(raw.ends_with("---\n" + body) and BookLayout.field_of(raw, "tags") == "tarot, pick a card",
		"the body is untouched, and the tags read back as the chapters' do")
	ed.free()
	return true
