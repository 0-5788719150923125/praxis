extends Node

## Provisioner (autoload) - installs, updates and reports on everything ghost fetches for itself.
##
## One job per dependency. FFmpeg and uv are downloaded from their upstream releases, checked
## against the published SHA-256 and unpacked into a versioned directory (see [Provision]). Python
## is installed by uv. Each feature's environment is a uv virtualenv built from its requirements,
## then checked by importing what it promises. Jobs are coroutines on the main thread, because every
## child process has to start there (see subprocess.gd); checksums and unpacking run on worker
## threads so no frame waits on a disk.
##
## WHEN THINGS HAPPEN. An interactive launch installs the core (uv, FFmpeg, Python) in the
## background and, at most once a day and only when the home screen's "keep up to date" box is
## ticked, updates everything already installed. A feature's environment is built the first time
## [method Provision.ensure] asks for it. A render, the offline analyzer, a test probe and every
## headless tool never start a job; they read what is on disk. `--provision` is the exception, a
## headless run that installs on purpose.
##
## Progress is the point of the shape: [signal changed] fires as a job moves, and [method state]
## says what it is doing in words, with a fraction whenever one is known. The Environment panel on
## the home screen and the badge at the top of every mode are both drawn from it.

signal changed(key: String)
signal finished(key: String, ok: bool, message: String)

## Installed at every interactive launch: small, and needed by most of ghost.
const BOOT := ["uv", "ffmpeg", "python"]
const UPDATE_EVERY_S := 86400
## A failed install is not retried by a feature asking again before this; the panel's retry is.
const RETRY_AFTER_MS := 60000
## A download that receives nothing for this long is abandoned.
const STALL_S := 90.0
## A job that has shown no sign of life for this long has died inside its coroutine (a script
## error ends one silently), and is closed as failed so nothing waits on it forever.
const DEAD_AFTER_MS := 300000
const POLL_S := 0.25

var _jobs := {}           # key -> {action, phase, fraction, done, total, started, beat}
var _fail := {}           # key -> {message, at, permanent}
var _holds := {}          # key -> {who: true}
var _memo := {}           # key -> installed, between jobs
var _enabled := false
var _updating := false
var _cli := false
var _tick := 0.0


func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	_enabled = _may_run(args)
	if not _enabled:
		set_process(false)
		return
	_prune()
	if not args.has("--provision"):
		# A beat after boot, so the first frames of the home screen are not spent starting jobs.
		get_tree().create_timer(1.0).timeout.connect(_on_boot)


## Only an interactive, writable launch installs anything by itself.
func _may_run(args: PackedStringArray) -> bool:
	if args.has("--provision"):
		return true
	for flag in ["--export", "--bake-file", "--bake-song", "--mask-render", "--deps"]:
		if args.has(flag):
			return false
	var st := get_node_or_null("/root/Settings")
	if st != null and bool(st.is_read_only()):
		return false
	return DisplayServer.get_name() != "headless"


func _on_boot() -> void:
	var fresh := false
	for key in BOOT:
		fresh = fresh or not is_ready(key)
		ensure(key)
	# A first install is already the newest of everything: count it as today's check.
	if fresh:
		_mark_checked()
	elif _update_due():
		check_updates()


func _process(dt: float) -> void:
	_tick += dt
	if _tick < 1.0:
		return
	_tick = 0.0
	var now := Time.get_ticks_msec()
	for key in _jobs.keys():
		if now - int(_jobs[key].get("beat", now)) > DEAD_AFTER_MS:
			_end(key, false, "the install stopped responding")


# --- what features ask ---------------------------------------------------------

func enabled() -> bool:
	return _enabled


## See [method Provision.ensure].
func ensure(key: String) -> bool:
	if is_ready(key):
		return true
	if not _enabled or _jobs.has(key) or not _installable(key):
		return false
	var why := Provision.unsupported(key)
	if not why.is_empty():
		_fail[key] = {"message": why, "at": Time.get_ticks_msec(), "permanent": true}
		return false
	var f: Dictionary = _fail.get(key, {})
	if not f.is_empty() and (bool(f.get("permanent", false))
			or Time.get_ticks_msec() - int(f.get("at", 0)) < RETRY_AFTER_MS):
		return false
	_start(key, "install")
	return false


## Try again now, whatever the last failure was.
func retry(key: String) -> void:
	_fail.erase(key)
	_memo.erase(key)
	ensure(key)


func is_ready(key: String) -> bool:
	# An environment being installed into is not usable until the job ends. A program being
	# updated is: the new version unpacks beside the one in use.
	if _jobs.has(key) and _env_key(key):
		return false
	if not _memo.has(key):
		_memo[key] = Provision.is_installed(key)
	return bool(_memo[key])


func state(key: String) -> Dictionary:
	var out := {"ready": is_ready(key), "running": _jobs.has(key)}
	if _jobs.has(key):
		out.merge(_jobs[key])
	if _fail.has(key):
		out["error"] = String(_fail[key].get("message", ""))
	return out


func active() -> PackedStringArray:
	return PackedStringArray(_jobs.keys())


## Forget what is known about the disk, for the panel's rescan.
func forget() -> void:
	_memo.clear()


func update(key: String) -> void:
	if _enabled and not _jobs.has(key) and _installable(key) and Provision.is_installed(key):
		_start(key, "update")


func hold(key: String, who: String) -> void:
	if not _holds.has(key):
		_holds[key] = {}
	_holds[key][who] = true


func release(key: String, who: String) -> void:
	if _holds.has(key):
		(_holds[key] as Dictionary).erase(who)
		if (_holds[key] as Dictionary).is_empty():
			_holds.erase(key)


func report(key: String, phase: String, fraction := -1.0, done := 0, total := 0) -> void:
	if not _jobs.has(key):
		_begin(key, "install", phase)
	_step(key, phase, fraction, done, total)


func report_done(key: String, ok: bool, message: String) -> void:
	if _jobs.has(key):
		_end(key, ok, message)


# --- updates --------------------------------------------------------------------

## Unix time of the last update pass, 0 if there has never been one.
func last_checked() -> int:
	return int(Provision.read_json(_updates_path()).get("checked_at", 0))


func checking() -> bool:
	return _updating


## Bring everything installed to its newest version now: the core first, then each environment
## nothing is using. One at a time, so an update never competes with itself for the network.
func check_updates() -> void:
	if _enabled and not _updating:
		_update_all()


func _update_all() -> void:
	_updating = true
	_mark_checked()
	changed.emit("")
	var keys := PackedStringArray(BOOT)
	for row in Deps.MANAGED:
		if _env_key(String(row["key"])):
			keys.append(String(row["key"]))
	for key in keys:
		while _jobs.has(key):
			await finished
		if not Provision.is_installed(key) or _holds.has(key):
			continue
		_start(key, "update")
		while _jobs.has(key):
			await finished
	_updating = false
	changed.emit("")


func _mark_checked() -> void:
	DirAccess.make_dir_recursive_absolute(Provision.tools_dir())
	var f := FileAccess.open(_updates_path(), FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify({"checked_at": int(Time.get_unix_time_from_system())}))
		f.close()


func _update_due() -> bool:
	var st := get_node_or_null("/root/Settings")
	if st != null and not bool(st.read("deps", "auto_update", true)):
		return false
	return int(Time.get_unix_time_from_system()) - last_checked() >= UPDATE_EVERY_S


func _updates_path() -> String:
	return Provision.tools_dir().path_join("updates.json")


# --- `--provision` -------------------------------------------------------------

## `--provision [all|update]`: install the core (with `all`, every environment too), or update
## everything installed, printing each step. True when everything asked for is ready.
func run_cli(scope: String) -> bool:
	_enabled = true
	_cli = true
	set_process(true)
	if scope == "update":
		await _update_all()
		return _fail.is_empty()
	var keys := PackedStringArray(BOOT)
	if scope == "all":
		for row in Deps.MANAGED:
			if _env_key(String(row["key"])):
				keys.append(String(row["key"]))
	var ok := true
	for key in keys:
		var why := Provision.unsupported(key)
		if not why.is_empty():
			print("  %s: not available on this machine (%s)" % [key, why])
			continue
		var got: bool = await _need(key)
		ok = ok and got
	return ok


# --- jobs ----------------------------------------------------------------------

func _installable(key: String) -> bool:
	var row := Deps.entry(key)
	return int(row.get("kind", -1)) == Deps.KIND_FETCHED or key == "python" or _env_key(key)


func _env_key(key: String) -> bool:
	var row := Deps.entry(key)
	return int(row.get("kind", -1)) == Deps.KIND_MANAGED \
		and (row.has("requirements") or row.has("packages"))


func _start(key: String, action: String) -> void:
	if int(Deps.entry(key).get("kind", -1)) == Deps.KIND_FETCHED:
		_tool_job(key, action)
	elif key == "python":
		_python_job(action)
	elif _env_key(key):
		_env_job(key, action)


## Wait until `key` is ready, installing it if it is not. False if it cannot be had.
func _need(key: String) -> bool:
	if is_ready(key):
		return true
	ensure(key)
	while _jobs.has(key):
		await finished
	return is_ready(key)


func _begin(key: String, action: String, phase: String) -> void:
	var now := Time.get_ticks_msec()
	_jobs[key] = {"action": action, "phase": phase, "fraction": -1.0, "done": 0, "total": 0,
		"started": now, "beat": now}
	_memo.erase(key)
	_say(key, phase)
	changed.emit(key)


func _step(key: String, phase: String, fraction := -1.0, done := 0, total := 0) -> void:
	var job: Dictionary = _jobs.get(key, {})
	if job.is_empty():
		return
	job["beat"] = Time.get_ticks_msec()
	if phase == String(job.get("phase", "")) and absf(fraction - float(job.get("fraction", -1.0))) < 0.004 \
			and done == int(job.get("done", 0)):
		return
	if phase != String(job.get("phase", "")):
		_say(key, phase)
	job["phase"] = phase
	job["fraction"] = fraction
	job["done"] = done
	job["total"] = total
	changed.emit(key)


func _beat(key: String) -> void:
	if _jobs.has(key):
		_jobs[key]["beat"] = Time.get_ticks_msec()


func _end(key: String, ok: bool, message: String) -> bool:
	_jobs.erase(key)
	_memo.erase(key)
	if ok:
		_fail.erase(key)
	else:
		_fail[key] = {"message": message, "at": Time.get_ticks_msec()}
	Deps.forget_all()
	print("ghost provision: %s %s - %s" % [key, "ok" if ok else "FAILED", message])
	finished.emit(key, ok, message)
	changed.emit(key)
	return ok


func _say(key: String, phase: String) -> void:
	if _cli:
		print("  %s: %s" % [key, phase])


# --- a fetched program ---------------------------------------------------------

func _tool_job(key: String, action: String) -> void:
	var row := Deps.entry(key)
	var name := String(row.get("name", key))
	var src := Provision.source(row)
	_begin(key, action, "looking for the newest release" if action == "install"
		else "checking for a newer release")
	var rel: Dictionary = await _release(row, src)
	if rel.has("error"):
		_end(key, false, String(rel["error"]))
		return
	var version := String(rel["version"])
	var have := Provision.installed(key)
	if Provision.tool_ok(key) and String(have.get("version", "")) == version:
		have["checked_at"] = int(Time.get_unix_time_from_system())
		Provision.write_installed(key, have)
		_end(key, true, "%s %s is the newest" % [name, version])
		return
	DirAccess.make_dir_recursive_absolute(Provision.partial_dir())
	var files: Array = rel["files"]
	var parts := PackedStringArray()
	for i in files.size():
		var file: Dictionary = files[i]
		var part := Provision.partial_dir().path_join("%s-%s-%d-%d.zip"
			% [key, version.validate_filename(), i, OS.get_process_id()])
		var label := "downloading %s" % version
		if files.size() > 1:
			label += " (%d of %d)" % [i + 1, files.size()]
		var why: String = await _download(key, String(file["url"]), part, label, int(file.get("size", 0)))
		if why.is_empty():
			_step(key, "verifying the download")
			var sum: Variant = await _on_worker(func() -> String: return FileAccess.get_sha256(part))
			if String(sum) != String(file["sha256"]):
				why = "the download does not match its published checksum"
		if not why.is_empty():
			parts.append(part)
			_discard(parts)
			_end(key, false, why)
			return
		parts.append(part)
	_step(key, "unpacking")
	var dir := version.validate_filename()
	if String(have.get("dir", "")) == dir:
		dir += "-%d" % int(Time.get_unix_time_from_system())   # never unpack over the copy in use
	var dest := Provision.tools_dir().path_join(key).path_join(dir)
	var provides := PackedStringArray(row.get("provides", []))
	var windows := Deps.platform() == "windows"
	var res: Variant = await _on_worker(func() -> Dictionary:
		return Provision.unpack(parts, provides, dest, windows))
	_discard(parts)
	var out: Dictionary = res if res is Dictionary else {}
	if not bool(out.get("ok", false)):
		_end(key, false, String(out.get("error", "unpacking failed")))
		return
	var now := int(Time.get_unix_time_from_system())
	Provision.write_installed(key, {"version": version, "dir": dir, "files": out["files"],
		"source": String(src.get("via", "")), "installed_at": now, "checked_at": now})
	Provision.prune(key)
	_end(key, true, "%s %s installed" % [name, version])


## Which release is newest, and what to download for this machine: {version, files: [{url,
## sha256, size}]}, or {error}.
func _release(row: Dictionary, src: Dictionary) -> Dictionary:
	match String(src.get("via", "")):
		"pypi":
			var r: Dictionary = await _ask("https://pypi.org/pypi/%s/json" % String(src["package"]))
			if r.has("error"):
				return r
			return Provision.pick_pypi(r.get("json", {}), Provision.platform_key())
		"github":
			var r: Dictionary = await _ask("https://api.github.com/repos/%s/releases/latest"
				% String(src["repo"]), true, ["Accept: application/vnd.github+json"])
			if r.has("error"):
				return r
			return Provision.pick_github(r.get("json", {}), String(src["asset"]))
		"riedl":
			var version := ""
			var files := []
			for prog in row.get("provides", []):
				var r: Dictionary = await _ask(Provision.riedl_url(String(src["os"]),
					String(src["arch"]), String(prog)), false)
				if not r.has("location"):
					return r if r.has("error") else {"error": "the FFmpeg build server gave no release"}
				var at := Provision.parse_riedl(String(r["location"]))
				if at.has("error"):
					return at
				if not version.is_empty() and version != String(at["version"]):
					return {"error": "the newest ffmpeg and ffprobe builds are different versions"}
				version = String(at["version"])
				var sums: Dictionary = await _ask(String(at["url"]) + ".sha256")
				if sums.has("error"):
					return sums
				var sha := Provision.parse_sha256(String(sums.get("text", "")))
				if sha.is_empty():
					return {"error": "no checksum is published for %s" % String(at["url"]).get_file()}
				files.append({"url": String(at["url"]), "sha256": sha, "size": 0})
			return {"version": version, "files": files}
	return {"error": "there is no build for %s" % Provision.platform_key()}


# --- Python --------------------------------------------------------------------

func _python_job(action: String) -> void:
	var key := "python"
	_begin(key, action, ("installing CPython %s" if action == "install"
		else "checking CPython %s for a newer patch") % Provision.PYTHON)
	if not await _need("uv"):
		_end(key, false, "uv is not available")
		return
	var args := PackedStringArray(["python", "install", Provision.PYTHON,
		"--install-dir", Provision.python_dir(), "--no-bin", "--no-registry"])
	if action == "update":
		args.append("--upgrade")
	args.append_array(Provision.uv_args())
	var log_file := Provision.log_path(key)
	var code: int = await _run_logged(key, Provision.uv(), args, log_file, Callable())
	if code != 0 or Provision.python_exe().is_empty():
		_end(key, false, "uv could not install Python: " + _error_line(log_file))
		return
	var said := _last_matching(log_file, "Installed Python ")
	_end(key, true, said if not said.is_empty() else "Python %s is current" % Provision.PYTHON)


# --- an environment ------------------------------------------------------------

func _env_job(key: String, action: String) -> void:
	var row := Deps.entry(key)
	var name := String(row.get("name", key))
	var venv := ProjectSettings.globalize_path(String(row["path"]))
	_begin(key, action, "updating" if action == "update" else "installing")
	if not await _need("uv"):
		_end(key, false, "uv is not available")
		return
	if not await _need("python"):
		_end(key, false, "Python is not available")
		return
	var uv := Provision.uv()
	var log_file := Provision.log_path(key)
	if not Provision.uv_built(venv):
		_step(key, "rebuilding it on Ghost Notes' own Python" if DirAccess.dir_exists_absolute(venv)
			else "creating the environment")
		var mk := PackedStringArray(["venv", venv, "--clear", "--python", Provision.python_exe(),
			"--no-python-downloads"])
		mk.append_array(Provision.uv_args())
		var made: int = await _run_logged(key, uv, mk, log_file, Callable())
		if made != 0 or not FileAccess.file_exists(Deps.venv_bin(venv, "python")):
			_end(key, false, "could not create the environment: " + _error_line(log_file))
			return
	var args := PackedStringArray(["pip", "install", "--python", venv, "--no-build",
		"--no-python-downloads"])
	if action == "update":
		args.append("--upgrade")
	var req := String(row.get("requirements", ""))
	if not req.is_empty():
		args.append_array(PackedStringArray(["-r", ProjectSettings.globalize_path(req)]))
	for p in row.get("packages", []):
		args.append(String(p))
	args.append_array(Provision.uv_args())
	_step(key, "resolving packages")
	var code: int = await _run_logged(key, uv, args, log_file, _watch_pip.bind(key))
	if code != 0:
		_end(key, false, "the packages could not be installed: " + _error_line(log_file))
		return
	var changes := _pip_changes(log_file)
	var py := Deps.venv_bin(venv, "python")
	for step in row.get("post", []):
		var post: Dictionary = step
		var post_log := Provision.log_path(key, "post")
		_step(key, String(post.get("label", "finishing")))
		var ran: int = await _run_logged(key, py, PackedStringArray(post.get("args", [])), post_log,
			_watch_post.bind(key, String(post.get("label", "finishing"))))
		if ran != 0:
			_end(key, false, "%s failed: %s" % [String(post.get("label", "a setup step")),
				_error_line(post_log)])
			return
	for file in row.get("files", []):
		var why: String = await _fetch_file(key, file)
		if not why.is_empty():
			_end(key, false, why)
			return
	_step(key, "checking the installation")
	var check_log := Provision.log_path(key, "check")
	var imported: int = await _run_logged(key, py, PackedStringArray(["-c",
		"import " + String(row.get("imports", "sys"))]), check_log, Callable())
	if imported != 0:
		_end(key, false, "it installed, but does not import: " + _error_line(check_log))
		return
	Provision.write_stamp(row)
	_end(key, true, "%s %s" % [name, changes if not changes.is_empty() else "is current"])


## A data file an environment needs (a tracker's model), fetched by ghost rather than by Python, so
## it has a progress bar and the same checksum rule as everything else. "" or why not.
func _fetch_file(key: String, file: Dictionary) -> String:
	var dest := Provision.file_path(String(file.get("key", "")))
	if FileAccess.file_exists(dest):
		return ""
	DirAccess.make_dir_recursive_absolute(Provision.partial_dir())
	DirAccess.make_dir_recursive_absolute(dest.get_base_dir())
	var part := Provision.partial_dir().path_join("%s-%d" % [String(file["file"]), OS.get_process_id()])
	var why: String = await _download(key, String(file["url"]), part,
		"downloading the %s" % String(file.get("name", "model")), int(file.get("size", 0)))
	if why.is_empty():
		var sum: Variant = await _on_worker(func() -> String: return FileAccess.get_sha256(part))
		if String(sum) != String(file.get("sha256", "")):
			why = "the %s does not match its checksum" % String(file.get("name", "model"))
		else:
			var moved: Variant = await _on_worker(func() -> int: return Provision.move(part, dest))
			if int(moved) != OK:
				why = "could not move the %s into place" % String(file.get("name", "model"))
	DirAccess.remove_absolute(part)
	return why


# --- progress from logs ---------------------------------------------------------

## uv's progress when it writes to a pipe: "Resolved N packages", " Downloaded <name>" for the big
## ones, "Prepared", then "Installed". No byte counts, so the bar is indeterminate and the words do
## the work.
func _watch_pip(text: String, key: String) -> void:
	var resolved := -1
	var latest := ""
	var installing := false
	for line in text.split("\n", false):
		var s := line.strip_edges()
		if s.begins_with("Resolved "):
			resolved = s.get_slice(" ", 1).to_int()
		elif s.begins_with("Downloaded "):
			latest = s.substr(11).strip_edges()
		elif s.begins_with("Prepared ") or s.begins_with("Uninstalled "):
			installing = true
		elif s.begins_with("Audited "):
			_step(key, "already current")
			return
	if installing:
		_step(key, "installing packages")
	elif resolved >= 0:
		_step(key, "downloading %d packages" % resolved + ("" if latest.is_empty()
			else " (%s done)" % latest))


## Playwright's browser download: "Downloading Chromium <version> ..." and "| 40% of 173.9 MiB".
func _watch_post(text: String, key: String, label: String) -> void:
	var what := label
	var pct := -1.0
	var re := RegEx.create_from_string("(\\d+)% of ([\\d.]+ ?[KMG]i?B)")
	for line in text.split("\n", false):
		var s := line.strip_edges()
		if s.begins_with("Downloading "):
			what = "downloading " + s.substr(12).get_slice(" from ", 0).get_slice(" (", 0)
		var m := re.search(s)
		if m != null:
			pct = m.get_string(1).to_float() / 100.0
	_step(key, what, pct)


## What an install changed, from uv's " + name==version" lines: "updated yt-dlp 2026.8.19", or
## "installed 12 packages" on a first install.
func _pip_changes(log_file: String) -> String:
	var added := PackedStringArray()
	var removed := {}
	for line in FileAccess.get_file_as_string(log_file).split("\n", false):
		var s := line.strip_edges()
		if s.begins_with("+ "):
			added.append(s.substr(2))
		elif s.begins_with("- "):
			removed[s.substr(2).get_slice("==", 0)] = true
	if added.is_empty():
		return ""
	var upgraded := PackedStringArray()
	for a in added:
		if removed.has(a.get_slice("==", 0)):
			upgraded.append(a.replace("==", " "))
	if not upgraded.is_empty():
		return "updated " + ", ".join(upgraded)
	return "installed %d packages" % added.size()


## The line of a failed step worth showing: uv's `error:`, Python's exception, or the last line.
func _error_line(log_file: String) -> String:
	var lines := FileAccess.get_file_as_string(log_file).strip_edges().split("\n", false)
	for i in lines.size():
		var s := lines[i].strip_edges()
		if s.begins_with("error:") or s.begins_with("×") or s.ends_with("Error") \
				or s.contains("Error: ") or s.begins_with("Caused by"):
			var out := s
			if i + 1 < lines.size():
				out += " " + lines[i + 1].strip_edges()
			return out.substr(0, 300)
	if lines.is_empty():
		return "no output (see %s)" % log_file
	return lines[lines.size() - 1].strip_edges().substr(0, 300)


func _last_matching(log_file: String, prefix: String) -> String:
	var found := ""
	for line in FileAccess.get_file_as_string(log_file).split("\n", false):
		if line.strip_edges().begins_with(prefix):
			found = line.strip_edges()
	return found


# --- the network ----------------------------------------------------------------

func _http() -> HTTPRequest:
	var h := HTTPRequest.new()
	h.use_threads = true
	var proxy := Provision.https_proxy()
	if not proxy.is_empty():
		h.set_https_proxy(String(proxy[0]), int(proxy[1]))
		h.set_http_proxy(String(proxy[0]), int(proxy[1]))
	add_child(h)
	return h


func _headers(extra: Array = []) -> PackedStringArray:
	var out := PackedStringArray(["User-Agent: ghost (Godot %s)" % Engine.get_version_info().get("string", "4")])
	for e in extra:
		out.append(String(e))
	return out


## GET `url`: {code, text, json} or {error}. With `follow` false a redirect is not followed and
## comes back as {code, location}, which is how a "latest" link names the release it stands for.
func _ask(url: String, follow := true, extra: Array = []) -> Dictionary:
	var host := url.get_slice("/", 2)
	var h := _http()
	h.max_redirects = 8 if follow else 0
	h.timeout = 30.0
	if h.request(url, _headers(extra)) != OK:
		h.queue_free()
		return {"error": "could not reach %s" % host}
	var r: Array = await h.request_completed
	h.queue_free()
	var result := int(r[0])
	var code := int(r[1])
	if not follow and code >= 300 and code < 400:
		for line in r[2]:
			if String(line).to_lower().begins_with("location:"):
				return {"code": code, "location": String(line).substr(9).strip_edges()}
	if result != HTTPRequest.RESULT_SUCCESS:
		return {"error": "%s could not be reached (%s)" % [host, _result_text(result)]}
	if code != 200:
		return {"error": "%s answered HTTP %d" % [host, code]}
	var text := (r[3] as PackedByteArray).get_string_from_utf8()
	var out := {"code": code, "text": text}
	if text.begins_with("{"):
		var json := JSON.new()
		if json.parse(text) == OK and json.data is Dictionary:
			out["json"] = json.data
	return out


## Stream `url` into `dest`, reporting progress on `key`. "" or why not.
func _download(key: String, url: String, dest: String, label: String, size_hint := 0) -> String:
	var host := url.get_slice("/", 2)
	var h := _http()
	h.download_file = dest
	h.download_chunk_size = 262144
	h.max_redirects = 8
	if h.request(url, _headers()) != OK:
		h.queue_free()
		return "could not reach %s" % host
	var box := [false, 0, 0]
	var on_done := func(result: int, code: int, _h: PackedStringArray, _b: PackedByteArray) -> void:
		box[0] = true
		box[1] = result
		box[2] = code
	h.request_completed.connect(on_done, CONNECT_ONE_SHOT)
	var last := -1
	var still := 0.0
	while not box[0]:
		await get_tree().create_timer(POLL_S).timeout
		var got := h.get_downloaded_bytes()
		var total := h.get_body_size()
		if total <= 0:
			total = size_hint
		if got == last:
			still += POLL_S
			if still > STALL_S:
				h.cancel_request()
				h.queue_free()
				DirAccess.remove_absolute(dest)
				return "the download from %s stalled" % host
		else:
			still = 0.0
			last = got
		_step(key, label, float(got) / float(total) if total > 0 else -1.0, got, maxi(total, 0))
	h.queue_free()
	if int(box[1]) != HTTPRequest.RESULT_SUCCESS or int(box[2]) != 200:
		DirAccess.remove_absolute(dest)
		if int(box[1]) != HTTPRequest.RESULT_SUCCESS:
			return "the download from %s failed (%s)" % [host, _result_text(int(box[1]))]
		return "%s answered HTTP %d" % [host, int(box[2])]
	return ""


func _result_text(result: int) -> String:
	match result:
		HTTPRequest.RESULT_CANT_CONNECT, HTTPRequest.RESULT_CANT_RESOLVE:
			return "no connection"
		HTTPRequest.RESULT_TLS_HANDSHAKE_ERROR:
			return "a secure connection could not be made"
		HTTPRequest.RESULT_TIMEOUT:
			return "it timed out"
		HTTPRequest.RESULT_CONNECTION_ERROR:
			return "the connection dropped"
		HTTPRequest.RESULT_DOWNLOAD_FILE_CANT_OPEN, HTTPRequest.RESULT_DOWNLOAD_FILE_WRITE_ERROR:
			return "it could not be written to disk"
	return "error %d" % result


func _discard(paths: PackedStringArray) -> void:
	for p in paths:
		DirAccess.remove_absolute(p)


# --- helpers --------------------------------------------------------------------

## Run `fn` on a worker thread and wait for its result without holding up a frame.
func _on_worker(fn: Callable) -> Variant:
	var box := [null]
	var id := WorkerThreadPool.add_task(func() -> void: box[0] = fn.call())
	while not WorkerThreadPool.is_task_completed(id):
		await get_tree().process_frame
	WorkerThreadPool.wait_for_task_completion(id)
	return box[0]


## Run a program with its output in `log_file`, letting `watch` read the log as it grows. Returns
## the exit code, -1 when it could not start.
func _run_logged(key: String, prog: String, args: PackedStringArray, log_file: String,
		watch: Callable) -> int:
	if prog.is_empty():
		return -1
	DirAccess.make_dir_recursive_absolute(log_file.get_base_dir())
	var pid := Subprocess.start_logged(prog, args, log_file, "provision " + key)
	if pid <= 0:
		return -1
	var seen := -1
	while Subprocess.alive(pid):
		await get_tree().create_timer(POLL_S).timeout
		_beat(key)
		if watch.is_valid():
			var text := FileAccess.get_file_as_string(log_file)
			if text.length() != seen:
				seen = text.length()
				watch.call(text)
	if watch.is_valid():
		watch.call(FileAccess.get_file_as_string(log_file))
	return OS.get_process_exit_code(pid)


## Clear what a killed session left behind: half-downloaded files, and versions nothing uses.
## A fresh partial download may be another ghost window's, in progress, so only stale ones go.
func _prune() -> void:
	var d := DirAccess.open(Provision.partial_dir())
	if d != null:
		for f in d.get_files():
			if Provision.stale(Provision.partial_dir().path_join(f)):
				DirAccess.remove_absolute(Provision.partial_dir().path_join(f))
	for row in Deps.FETCHED:
		Provision.prune(String(row["key"]))
