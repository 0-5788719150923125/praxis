extends RefCounted
class_name Provision

## Provision - what ghost installs for itself, and the one door every feature asks through.
##
## ghost needs FFmpeg for video, uv to build Python environments, a Python for those to run on, and
## per-feature packages (the neural voice, face tracking, URL import, page capture). None of it is
## the user's to install. ghost downloads each piece from its upstream release, checks it against
## the published SHA-256, keeps it under the user data directory and keeps it current.
##
## This file is the static half: where things live, which release fits this machine, how an archive
## is unpacked, and whether an environment is the one its requirements describe. It runs anywhere,
## which is how tests/provision_check.gd covers it with no network. The live half is the
## `Provisioner` autoload (provisioner.gd), which downloads, runs uv and reports progress. Features
## never name the autoload: they call [method ensure] here, which reaches it while the app runs and
## otherwise answers from what is on disk.
##
## The table of what exists is [Deps]: `Deps.FETCHED` for downloaded programs, `Deps.MANAGED` for
## Python and the environments.

## The Python every environment runs on: the newest stable minor that every environment's wheels
## support. uv installs its newest patch, and updates move the environments to newer patches in place
## (a uv virtualenv points at the minor version, not the patch).
const PYTHON := "3.14"

## Every machine ghost knows how to provision, as `<os>-<arch>`.
const PLATFORMS := ["linux-x86_64", "linux-arm64", "macos-arm64", "macos-x86_64",
	"windows-x86_64", "windows-arm64"]

## The wheel platform tags each machine runs, as patterns, for programs fetched from PyPI. Patterns
## rather than exact tags because a project raises its manylinux or macOS floor between releases.
const WHEEL_TAGS := {
	"linux-x86_64": "^manylinux.*_x86_64$",
	"linux-arm64": "^manylinux.*_aarch64$",
	"macos-arm64": "^macosx_\\d+_\\d+_(arm64|universal2)$",
	"macos-x86_64": "^macosx_\\d+_\\d+_(x86_64|universal2)$",
	"windows-x86_64": "^win_amd64$",
	"windows-arm64": "^win_arm64$",
}

const RIEDL := "https://ffmpeg.martin-riedl.de"

## Written into an environment once it is installed and its imports are checked.
const STAMP := ".ghost-env.json"

## Where everything lives. "" means user://; tests point it at a scratch directory.
static var base_override := ""


# --- the front door ----------------------------------------------------------

## The Provisioner autoload, or null in a process without one (a `--script` run).
static func agent() -> Node:
	var tree := Engine.get_main_loop() as SceneTree
	return tree.root.get_node_or_null("Provisioner") if tree != null else null


## Is `key` installed and usable now? If not, the Provisioner starts installing it (when this
## process may install anything) and this answers false; ask again on a later frame. Safe to call
## every frame, and from any thread.
static func ensure(key: String) -> bool:
	var a := agent()
	if a == null:
		return is_installed(key)
	if OS.get_thread_caller_id() != OS.get_main_thread_id():
		a.call_deferred("ensure", key)
		return is_installed(key)
	return bool(a.call("ensure", key))


## Whether this process can install anything at all. False in a render, a test probe or a headless
## tool, where a missing dependency stays missing.
static func can_install() -> bool:
	var a := agent()
	return a != null and bool(a.call("enabled"))


## What `key` is doing, for a status line: `ready`, `running`, and while running `action`
## ("install" or "update"), `phase` (words), `fraction` (0..1, or -1 when unknown), `done`/`total`
## (bytes, when a download is the step), `started` (ticks msec); `error` after a failure.
static func state(key: String) -> Dictionary:
	var a := agent()
	if a == null:
		return {"ready": is_installed(key), "running": false}
	return a.call("state", key)


## Every key with a job running, in start order.
static func active() -> PackedStringArray:
	var a := agent()
	return a.call("active") if a != null else PackedStringArray()


## Bring `key` to its newest version now, if it is installed. The yt-dlp retry uses it.
static func update(key: String) -> void:
	var a := agent()
	if a != null and OS.get_thread_caller_id() == OS.get_main_thread_id():
		a.call("update", key)


## Keep `key` from being updated while `who` uses it. Pair with [method release].
static func hold(key: String, who: String) -> void:
	var a := agent()
	if a != null:
		a.call("hold", key, who)


static func release(key: String, who: String) -> void:
	var a := agent()
	if a != null:
		a.call("release", key, who)


## A download something else performs (the voice host fetching a model), shown with ghost's own.
static func report(key: String, phase: String, fraction := -1.0, done := 0, total := 0) -> void:
	var a := agent()
	if a != null:
		a.call("report", key, phase, fraction, done, total)


static func report_done(key: String, ok: bool, message: String) -> void:
	var a := agent()
	if a != null:
		a.call("report_done", key, ok, message)


## One line describing `key` right now, for a status label: what it is doing and how far along,
## why it failed, or "" once it is ready.
static func describe(key: String, s: Dictionary = {}) -> String:
	if s.is_empty():
		s = state(key)
	var name := String(Deps.entry(key).get("name", key))
	if bool(s.get("running", false)):
		var line := "%s: %s" % [name, String(s.get("phase", "working"))]
		var f := float(s.get("fraction", -1.0))
		var total := int(s.get("total", 0))
		if total > 0:
			line += " · %s / %s" % [mb(int(s.get("done", 0))), mb(total)]
		elif f >= 0.0:
			line += " · %d%%" % int(f * 100.0)
		var secs := (Time.get_ticks_msec() - int(s.get("started", Time.get_ticks_msec()))) / 1000
		if secs >= 3:
			line += " · %d:%02d" % [secs / 60, secs % 60]
		return line
	if bool(s.get("ready", false)):
		return ""
	var why := unsupported(key)
	if not why.is_empty():
		return "%s is not available on this machine: %s" % [name, why]
	var err := String(s.get("error", ""))
	if not err.is_empty():
		return "%s could not be installed: %s" % [name, err]
	return "%s is not installed yet" % name


## The sentence a feature shows when `key` is what it is missing: what ghost is doing about it.
static func hint(key: String) -> String:
	var name := String(Deps.entry(key).get("name", key))
	var why := unsupported(key)
	if not why.is_empty():
		return "%s is not available on this machine: %s." % [name, why]
	var s := state(key)
	if bool(s.get("ready", false)):
		return ""
	if bool(s.get("running", false)):
		return "Ghost Notes is installing %s - %s." % [name, describe(key, s)]
	var err := String(s.get("error", ""))
	if not err.is_empty():
		return "%s could not be installed (%s). Retry it from the Environment panel on the home screen." \
			% [name, err]
	if can_install():
		return "Ghost Notes is about to install %s by itself." % name
	return "Ghost Notes installs %s by itself when it is launched normally (or run it with --provision)." % name


static func mb(n: int) -> String:
	return "%.1f MB" % (float(n) / 1048576.0)


# --- this machine -------------------------------------------------------------

static func platform_key() -> String:
	return "%s-%s" % [Deps.platform(), arch()]


static func arch() -> String:
	var a := Engine.get_architecture_name()
	return "arm64" if a in ["arm64", "aarch64", "armv8"] else a


## Why `key` cannot be installed here, "" when it can.
static func unsupported(key: String) -> String:
	var plat := platform_key()
	var row := Deps.entry(key)
	var kind := int(row.get("kind", Deps.KIND_TOOL))
	if kind == Deps.KIND_TOOL:
		return ""
	var why: Dictionary = row.get("unsupported", {})
	if why.has(plat):
		return String(why[plat])
	if not PLATFORMS.has(plat):
		return "Ghost Notes has no builds for %s" % plat
	if kind == Deps.KIND_FETCHED and source(row).is_empty():
		return "there is no build for %s" % plat
	return ""


## Where a fetched program comes from on this machine: its `sources` entry, {} when there is none.
static func source(row: Dictionary, plat := "") -> Dictionary:
	var sources: Dictionary = row.get("sources", {})
	var key := plat if not plat.is_empty() else platform_key()
	return sources.get(key, sources.get("*", {}))


## The HTTPS proxy the environment names, as [host, port], or []. Godot's HTTPRequest does not read
## HTTPS_PROXY by itself, and uv (which does) should reach the network the same way ghost does.
static func https_proxy() -> Array:
	var raw := OS.get_environment("HTTPS_PROXY")
	if raw.is_empty():
		raw = OS.get_environment("https_proxy")
	if raw.is_empty():
		return []
	var hostport := raw.get_slice("://", 1) if raw.contains("://") else raw
	hostport = hostport.get_slice("@", 1) if hostport.contains("@") else hostport
	hostport = hostport.get_slice("/", 0)
	var port := hostport.get_slice(":", 1).to_int() if hostport.contains(":") else 80
	return [hostport.get_slice(":", 0), port]


# --- where things live --------------------------------------------------------

static func base() -> String:
	return base_override if not base_override.is_empty() else OS.get_user_data_dir()


static func tools_dir() -> String:
	return base().path_join("tools")


static func partial_dir() -> String:
	return tools_dir().path_join(".partial")


static func uv_dir() -> String:
	return base().path_join("uv")


static func python_dir() -> String:
	return uv_dir().path_join("python")


static func cache_dir() -> String:
	return uv_dir().path_join("cache")


static func models_dir() -> String:
	return base().path_join("models")


## A job's log. Kept, so a failure can be read after the fact (the panel names the file).
static func log_path(key: String, step := "") -> String:
	return base().path_join("logs/provision").path_join(
		key + ("" if step.is_empty() else "_" + step) + ".log")


## The flags every uv call carries: ghost's own cache, and no uv.toml of anybody's. uv is also
## handed paths for everything else it touches, never environment variables, because those would
## reach every other program ghost starts (a dispatched assistant run included).
static func uv_args() -> PackedStringArray:
	return PackedStringArray(["--cache-dir", cache_dir(), "--no-config"])


static func uv() -> String:
	return tool_path("uv", "uv")


# --- fetched programs ---------------------------------------------------------
#
# Each fetched program lives in tools/<key>/<version>/, and tools/<key>/current.json names the
# version in use. A new version is unpacked beside the old one and current.json is switched once it
# is whole, so a program that is running is never replaced under itself; the older directory is
# removed the next time nothing can be using it.

static func _manifest_path(key: String) -> String:
	return tools_dir().path_join(key).path_join("current.json")


## What is installed for `key`: {version, dir, files: {program: file name}, source, installed_at,
## checked_at}, or {} when nothing is.
static func installed(key: String) -> Dictionary:
	return read_json(_manifest_path(key))


## A JSON object from a file, {} when the file is missing, empty or not one. Quietly: a missing
## stamp or manifest is the ordinary state of a first run, not an error to print.
static func read_json(path: String) -> Dictionary:
	var text := FileAccess.get_file_as_string(path) if FileAccess.file_exists(path) else ""
	if text.strip_edges().is_empty():
		return {}
	var json := JSON.new()
	if json.parse(text) != OK:
		return {}
	return json.data if json.data is Dictionary else {}


static func write_installed(key: String, data: Dictionary) -> bool:
	var path := _manifest_path(key)
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var tmp := path + ".tmp"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	if f == null:
		return false
	f.store_string(JSON.stringify(data, "\t"))
	f.close()
	if FileAccess.file_exists(path):
		DirAccess.remove_absolute(path)
	return DirAccess.rename_absolute(tmp, path) == OK


## The absolute path of `prog` from ghost's own copy of `key`, "" when there is none.
static func tool_path(key: String, prog: String) -> String:
	var have := installed(key)
	var files: Dictionary = have.get("files", {})
	if not files.has(prog):
		return ""
	var p := tools_dir().path_join(key).path_join(String(have.get("dir", ""))).path_join(String(files[prog]))
	return p if FileAccess.file_exists(p) else ""


## Is every program `key` provides there?
static func tool_ok(key: String) -> bool:
	var row := Deps.entry(key)
	var provides: Array = row.get("provides", [])
	if provides.is_empty():
		return false
	for p in provides:
		if tool_path(key, String(p)).is_empty():
			return false
	return true


## Remove every version of `key` but the one in use, and what a killed session left
## half-unpacked. A staging directory younger than a day may be another ghost window's unpack in
## progress, so it is left alone; so is anything that cannot be removed (a program in it is still
## running), until next time.
static func prune(key: String) -> void:
	var dir := tools_dir().path_join(key)
	var keep := String(installed(key).get("dir", ""))
	var d := DirAccess.open(dir)
	if d == null:
		return
	for sub in d.get_directories():
		if sub == keep:
			continue
		if sub.ends_with(".staging") and not stale(dir.path_join(sub)):
			continue
		remove_tree(dir.path_join(sub))


## Older than a day: left behind by a session that is gone, not one still working on it.
static func stale(path: String) -> bool:
	var at := FileAccess.get_modified_time(path)
	return at > 0 and int(Time.get_unix_time_from_system()) - at > 86400


## Rename `from` to `to`, retrying for a moment: on Windows a virus scanner opens a program the
## instant it is written, and a rename while it looks is refused. File IO and sleeps only - call it
## from a worker thread.
static func move(from: String, to: String) -> Error:
	var err := ERR_CANT_CREATE
	for attempt in 10:
		err = DirAccess.rename_absolute(from, to)
		if err == OK:
			return OK
		OS.delay_msec(200)
	return err


## Delete `path`, file or directory, recursively. A symbolic link is removed as a link and never
## followed, so a link to somewhere else cannot take that somewhere with it.
static func remove_tree(path: String) -> void:
	var parent := DirAccess.open(path.get_base_dir())
	if parent != null and parent.is_link(path):
		DirAccess.remove_absolute(path)
		return
	if not DirAccess.dir_exists_absolute(path):
		if FileAccess.file_exists(path):
			DirAccess.remove_absolute(path)
		return
	var d := DirAccess.open(path)
	if d == null:
		return
	d.include_hidden = true
	for f in d.get_files():
		DirAccess.remove_absolute(path.path_join(f))
	for sub in d.get_directories():
		remove_tree(path.path_join(sub))
	DirAccess.remove_absolute(path)


# --- releases -----------------------------------------------------------------

## The wheel of a PyPI project (the `pypi.org/pypi/<name>/json` document) that runs on `plat`:
## {version, files: [{url, sha256, size, name}]}, or {error}. A wheel is a zip, which is why uv is
## taken from PyPI: [ZIPReader] opens it, where uv's own release archives are tar.gz on two systems.
static func pick_pypi(doc: Dictionary, plat: String) -> Dictionary:
	var info: Dictionary = doc.get("info", {})
	var version := String(info.get("version", ""))
	var pattern := String(WHEEL_TAGS.get(plat, ""))
	var project := String(info.get("name", "the package"))
	if version.is_empty():
		return {"error": "PyPI did not say which %s is newest" % project}
	if pattern.is_empty():
		return {"error": "no %s build for %s" % [project, plat]}
	var re := RegEx.create_from_string(pattern)
	for f in doc.get("urls", []):
		var file: Dictionary = f
		var name := String(file.get("filename", ""))
		if not name.ends_with(".whl") or bool(file.get("yanked", false)):
			continue
		var parts := name.get_basename().split("-")
		for tag in String(parts[parts.size() - 1]).split("."):
			if re.search(tag) != null:
				var digests: Dictionary = file.get("digests", {})
				return {"version": version, "files": [{"url": String(file.get("url", "")),
					"sha256": String(digests.get("sha256", "")).to_lower(),
					"size": int(file.get("size", 0)), "name": name}]}
	return {"error": "%s %s has no build for %s" % [project, version, plat]}


## The asset of a GitHub release (the `/releases/latest` document) whose name matches `pattern`,
## with the SHA-256 GitHub records for it: {version, files: [...]}, or {error}. When the pattern
## captures a version and several assets match, the highest wins (BtbN publishes a build per FFmpeg
## branch); without a capture the release tag is the version.
static func pick_github(doc: Dictionary, pattern: String) -> Dictionary:
	var re := RegEx.create_from_string(pattern)
	var best := {}
	for a in doc.get("assets", []):
		var asset: Dictionary = a
		var name := String(asset.get("name", ""))
		var m := re.search(name)
		if m == null:
			continue
		var digest := String(asset.get("digest", ""))
		if not digest.begins_with("sha256:"):
			continue
		var v := m.get_string(1) if m.get_group_count() >= 1 else ""
		if v.is_empty():
			v = String(doc.get("tag_name", ""))
		if best.is_empty() or Deps.version_lt(String(best["version"]), v):
			best = {"version": v, "files": [{"url": String(asset.get("browser_download_url", "")),
				"sha256": digest.substr(7).to_lower(), "size": int(asset.get("size", 0)),
				"name": name}]}
	if best.is_empty():
		return {"error": "the newest %s release has no matching build"
			% String(doc.get("html_url", "release")).get_slice("/releases", 0).get_file()}
	return best


## Martin Riedl's "newest release" link for one program. It redirects to the versioned build, and
## that redirect is how the version is learned without downloading anything.
static func riedl_url(os_name: String, arch_name: String, prog: String) -> String:
	return "%s/redirect/latest/%s/%s/release/%s.zip" % [RIEDL, os_name, arch_name, prog]


## Where a Riedl redirect points - /download/<os>/<arch>/<build>_<version>/<program>.zip - as
## {version, url}, or {error}.
static func parse_riedl(location: String) -> Dictionary:
	var re := RegEx.create_from_string(
		"^(?:https?://[^/]+)?(/download/[^/]+/[^/]+/\\d+_([^/]+)/[^/]+\\.zip)$")
	var m := re.search(location.strip_edges())
	if m == null:
		return {"error": "unexpected FFmpeg download location '%s'" % location}
	return {"version": m.get_string(2), "url": RIEDL + m.get_string(1)}


## The first SHA-256 in a checksum file's text ("<hash>  <name>"), "" when there is none.
static func parse_sha256(text: String) -> String:
	var m := RegEx.create_from_string("\\b([0-9a-fA-F]{64})\\b").search(text)
	return m.get_string(1).to_lower() if m != null else ""


# --- unpacking ----------------------------------------------------------------

## Unpack `programs` from the zip `archives` into the directory `dest`, which appears whole or not at
## all: everything goes into a staging directory that is renamed into place once every program is
## there. A program is found by its file name anywhere in an archive (`ffmpeg.exe` sits in `bin/`
## in one build and at the root of another), with `.exe` when `windows`. Returns
## {ok, files: {program: file name}} or {ok: false, error}. File IO only, so it may run on a worker.
static func unpack(archives: PackedStringArray, programs: PackedStringArray, dest: String,
		windows: bool) -> Dictionary:
	var staging := dest + ".staging"
	remove_tree(staging)
	if DirAccess.make_dir_recursive_absolute(staging) != OK:
		return {"ok": false, "error": "could not create %s" % staging}
	var files := {}
	for archive in archives:
		var zr := ZIPReader.new()
		if zr.open(archive) != OK:
			remove_tree(staging)
			return {"ok": false, "error": "the download is not a readable zip"}
		for member in zr.get_files():
			var name := String(member).get_file()
			if windows and not name.ends_with(".exe"):
				continue
			var prog := name.trim_suffix(".exe") if windows else name
			if not programs.has(prog) or files.has(prog):
				continue
			var out_path := staging.path_join(name)
			var out := FileAccess.open(out_path, FileAccess.WRITE)
			if out == null:
				zr.close()
				remove_tree(staging)
				return {"ok": false, "error": "could not write %s" % out_path}
			out.store_buffer(zr.read_file(member))
			out.close()
			if not windows:
				FileAccess.set_unix_permissions(out_path, FileAccess.UNIX_READ_OWNER
					| FileAccess.UNIX_WRITE_OWNER | FileAccess.UNIX_EXECUTE_OWNER
					| FileAccess.UNIX_READ_GROUP | FileAccess.UNIX_EXECUTE_GROUP
					| FileAccess.UNIX_READ_OTHER | FileAccess.UNIX_EXECUTE_OTHER)
			files[prog] = name
		zr.close()
	for p in programs:
		if not files.has(p):
			remove_tree(staging)
			return {"ok": false, "error": "the download has no %s in it" % p}
	remove_tree(dest)
	if move(staging, dest) != OK:
		remove_tree(staging)
		return {"ok": false, "error": "could not move the download into place"}
	return {"ok": true, "files": files}


# --- Python and the environments ----------------------------------------------

## ghost's own interpreter for [constant PYTHON], "" before uv has installed it. Taken through uv's
## minor-version link (`cpython-3.14-<platform>`), which uv repoints at each newer patch it
## installs, so the environments built on it move with it.
static func python_exe() -> String:
	var d := DirAccess.open(python_dir())
	if d == null:
		return ""
	var link := "cpython-%s-" % PYTHON
	var patch := RegEx.create_from_string("^cpython-%s\\.(\\d+)-" % PYTHON.replace(".", "\\."))
	var newest := ""
	var newest_n := -1
	for sub in d.get_directories():
		if sub.begins_with(link):
			var exe := _python_in(python_dir().path_join(sub))
			if not exe.is_empty():
				return exe
		var m := patch.search(sub)
		if m != null and m.get_string(1).to_int() > newest_n:
			newest_n = m.get_string(1).to_int()
			newest = sub
	return _python_in(python_dir().path_join(newest)) if not newest.is_empty() else ""


static func _python_in(dir: String) -> String:
	var names := ["python.exe"] if Deps.platform() == "windows" \
		else ["bin/python" + PYTHON, "bin/python3"]
	for rel in names:
		var p := dir.path_join(rel)
		if FileAccess.file_exists(p):
			return p
	return ""


## Is the virtualenv at `venv` one ghost's uv made on ghost's own Python? Anything else - missing,
## made by `python -m venv`, or on another interpreter - is rebuilt before it is used.
static func uv_built(venv: String) -> bool:
	var cfg := FileAccess.get_file_as_string(venv.path_join("pyvenv.cfg"))
	if cfg.is_empty():
		return false
	var home := ""
	var by_uv := false
	for line in cfg.split("\n"):
		var kv := line.split("=", true, 1)
		if kv.size() != 2:
			continue
		match kv[0].strip_edges():
			"uv":
				by_uv = true
			"home":
				home = kv[1].strip_edges()
	return by_uv and _inside(home, python_dir())


static func _inside(path: String, root: String) -> bool:
	var a := path.replace("\\", "/")
	var b := root.replace("\\", "/")
	if Deps.platform() == "windows":
		a = a.to_lower()
		b = b.to_lower()
	return not b.is_empty() and a.begins_with(b)


## What an environment is meant to contain, as one hash: its requirements as written (the file's
## text, or the inline package list), its extra steps and files, and the Python it runs on. When it
## changes, the environment is brought up to date before it is used again.
static func env_digest(row: Dictionary) -> String:
	var text := PYTHON + "\n"
	var req := String(row.get("requirements", ""))
	if not req.is_empty():
		text += FileAccess.get_file_as_string(req) + "\n"
	for p in row.get("packages", []):
		text += String(p) + "\n"
	for c in row.get("post", []):
		text += " ".join(PackedStringArray((c as Dictionary).get("args", []))) + "\n"
	for f in row.get("files", []):
		text += String((f as Dictionary).get("sha256", "")) + "\n"
	return text.sha256_text()


## Is the environment `row` describes installed, checked and current with its requirements?
static func env_ready(row: Dictionary) -> bool:
	var venv := ProjectSettings.globalize_path(String(row.get("path", "")))
	if venv.is_empty() or not FileAccess.file_exists(Deps.venv_bin(venv, "python")):
		return false
	if not uv_built(venv):
		return false
	var stamp := read_json(venv.path_join(STAMP))
	if String(stamp.get("digest", "")) != env_digest(row):
		return false
	for f in row.get("files", []):
		if not FileAccess.file_exists(file_path(String((f as Dictionary).get("key", "")))):
			return false
	return true


static func write_stamp(row: Dictionary) -> void:
	var venv := ProjectSettings.globalize_path(String(row.get("path", "")))
	var f := FileAccess.open(venv.path_join(STAMP), FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify({"digest": env_digest(row),
			"installed_at": int(Time.get_unix_time_from_system())}, "\t"))
		f.close()


## Where a data file an environment needs lands, by its key in that environment's `files`.
static func file_path(file_key: String) -> String:
	for row in Deps.MANAGED:
		for f in (row as Dictionary).get("files", []):
			if String((f as Dictionary).get("key", "")) == file_key:
				return models_dir().path_join(String(f["file"]))
	return ""


## Is this key's thing installed on disk? No side effects, no network.
static func is_installed(key: String) -> bool:
	var row := Deps.entry(key)
	match int(row.get("kind", -1)):
		Deps.KIND_FETCHED:
			return tool_ok(key)
		Deps.KIND_MANAGED:
			if key == "python":
				return not python_exe().is_empty()
			if row.has("requirements") or row.has("packages"):
				return env_ready(row)
	return false
