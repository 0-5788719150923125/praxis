extends RefCounted
class_name PageCapture

## PageCapture - a REAL web page, captured as one tall picture, for the tablet medium.
##
## A `<!-- url: -->` the chapter writes nothing under is the real page at that address. It is
## shown as a capture of the page: taken ONCE, on request, and kept in the [Illustrations]
## library as that page's picture - so a render shows exactly what the live session showed, the
## versions/import/delete machinery applies to it, and a site that refuses a headless browser
## can be replaced by a screenshot imported from disk.
##
## The capture is Playwright's Chromium, driven by `capture_host/capture.py` in ghost's OWN venv
## (`user://capture_venv`, the yt-dlp discipline: nothing is installed system-wide or into the
## author's environment). It is pinned to the Playwright the Praxis checkout uses, so the browser
## is the one already in the shared `~/.cache/ms-playwright` and normally nothing is downloaded.
## Set up the first time a page is captured; a session that never captures installs nothing.
##
## SUCCESS IS THE ARTIFACT, never an exit code: a step is judged by what it left behind (the
## venv's python, then the PNG), and a capture that produced no PNG reports its log's last lines.

const VENV := "user://capture_venv"
const DIR := "user://captures"
const SCRIPT := "res://capture_host/capture.py"
const SETUP := "res://capture_host/setup.py"
## The tablet's portrait width, and a screenful; the capture runs to the page's own length.
const WIDTH := 1200
const HEIGHT := 1600
const MAX_HEIGHT := 7000
const USER_AGENT := "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
const STEP_TIMEOUT_S := 600
## Written into the venv once a capture has worked, so setup is not repeated.
const READY := ".ready"

static var _queue: Array = []        # [{key, url}]
static var _cur := {}                # the request being worked on
static var _step := ""               # "" | "venv" | "install" | "capture"
static var _pid := -1
static var _started := 0
static var _errors := {}


## Ask for [param url] as [param key]'s picture. "" when queued, else why not.
static func request(key: String, url: String) -> String:
	if busy(key):
		return ""
	if _python().is_empty() and not _ready():
		return "Python 3 is not installed - %s" % Deps.hint("python")
	_errors.erase(key)
	_queue.append({"key": key, "url": url})
	return ""


static func busy(key: String) -> bool:
	if String(_cur.get("key", "")) == key:
		return true
	for q in _queue:
		if String((q as Dictionary)["key"]) == key:
			return true
	return false


static func pending() -> int:
	return _queue.size() + (0 if _cur.is_empty() else 1)


static func error_of(key: String) -> String:
	return String(_errors.get(key, ""))


## What the capture is doing, for the panel's status line.
static func doing() -> String:
	match _step:
		"venv":
			return "setting up ghost's capture environment (one-time)"
		"install":
			return "installing Playwright into it (one-time)"
		"capture":
			return "capturing %s" % String(_cur.get("url", ""))
	return ""


## Advance the work: start what is next, notice what ended. Returns the captures that landed,
## `[{key, url, file}]`, for [Illustrations] to take into the library.
static func pump() -> Array:
	var landed: Array = []
	if _pid > 0:
		if Subprocess.alive(_pid):
			if int(Time.get_unix_time_from_system()) - _started > STEP_TIMEOUT_S:
				Subprocess.stop(_pid)
				_fail("timed out while %s" % doing())
			return landed
		Subprocess.forget(_pid)
		_pid = -1
		match _step:
			"venv":
				if not FileAccess.file_exists(Deps.venv_bin(_abs(VENV), "python")):
					_fail("could not create the capture environment - see %s" % _log("venv"))
					return landed
				_run("install")
				return landed
			"install":
				_run("capture")
				return landed
			"capture":
				var out := _abs(DIR.path_join("%s.png" % String(_cur["key"])))
				if FileAccess.file_exists(out):
					FileAccess.open(_abs(VENV).path_join(READY), FileAccess.WRITE)
					landed.append({"key": _cur["key"], "url": _cur["url"], "file": out})
					_cur = {}
					_step = ""
				else:
					_fail(_tail(_log("capture")))
				return landed
	if _cur.is_empty() and not _queue.is_empty():
		_cur = _queue.pop_front()
		if _ready():
			_run("capture")
		elif FileAccess.file_exists(Deps.venv_bin(_abs(VENV), "python")):
			_run("install")
		else:
			_run("venv")
	return landed


static func _run(step: String) -> void:
	_step = step
	_started = int(Time.get_unix_time_from_system())
	DirAccess.make_dir_recursive_absolute(_abs(DIR))
	var args := PackedStringArray()
	var prog := ""
	match step:
		"venv":
			prog = _python()
			args = PackedStringArray(["-m", "venv", _abs(VENV)])
		"install":
			# the package, then its browser - a no-op when the shared cache already has it
			prog = Deps.venv_bin(_abs(VENV), "python")
			args = PackedStringArray([_abs(SETUP)])
		"capture":
			var key := String(_cur["key"])
			var out := _abs(DIR.path_join("%s.png" % key))
			DirAccess.remove_absolute(out)
			var spec := _abs(DIR.path_join("%s.json" % key))
			var f := FileAccess.open(spec, FileAccess.WRITE)
			f.store_string(JSON.stringify({"url": String(_cur["url"]), "out": out, "width": WIDTH,
				"height": HEIGHT, "max_height": MAX_HEIGHT, "user_agent": USER_AGENT}))
			f.close()
			prog = Deps.venv_bin(_abs(VENV), "python")
			args = PackedStringArray([_abs(SCRIPT), spec])
	print("ghost capture: %s" % doing())
	_pid = Subprocess.start_logged(prog, args, _log(step), "page capture")
	if _pid <= 0:
		_fail("could not start %s" % prog.get_file())


static func _fail(why: String) -> void:
	if not _cur.is_empty():
		_errors[String(_cur["key"])] = why
		push_warning("ghost capture: %s failed - %s" % [String(_cur.get("url", "")), why])
	_cur = {}
	_step = ""
	_pid = -1


static func _ready() -> bool:
	return FileAccess.file_exists(_abs(VENV).path_join(READY)) \
		and FileAccess.file_exists(Deps.venv_bin(_abs(VENV), "python"))


static func _python() -> String:
	return Deps.resolve_any(["python", "python3", "py"] if OS.get_name() == "Windows"
		else ["python3", "python"])


static func _abs(p: String) -> String:
	return ProjectSettings.globalize_path(p)


static func _log(step: String) -> String:
	return _abs(DIR.path_join("%s.log" % step))


## The last few lines of a log: the reason, when a step left nothing behind.
static func _tail(path: String) -> String:
	var text := FileAccess.get_file_as_string(path).strip_edges()
	if text.is_empty():
		return "nothing was captured (no log)"
	var lines := text.split("\n")
	return " / ".join(lines.slice(maxi(0, lines.size() - 3)))
