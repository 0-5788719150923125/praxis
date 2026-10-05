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
## The capture is Playwright's Chromium, driven by `capture_host/capture.py` in ghost's OWN
## environment (`capture_venv`, see [Provision]: built by uv on ghost's own Python, with the browser
## downloaded into Playwright's cache as its last step). Built the first time a page is captured;
## a session that never captures installs nothing.
##
## SUCCESS IS THE ARTIFACT, never an exit code: a capture is judged by the PNG it left behind, and
## one that produced no PNG reports its log's last lines.

const ENV := "capture_venv"
const VENV := "user://capture_venv"
const DIR := "user://captures"
const SCRIPT := "res://capture_host/capture.py"
## The tablet's portrait width, and a screenful; the capture runs to the page's own length.
const WIDTH := 1200
const HEIGHT := 1600
const MAX_HEIGHT := 7000
const USER_AGENT := "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
const STEP_TIMEOUT_S := 600

static var _queue: Array = []        # [{key, url}]
static var _cur := {}                # the request being worked on
static var _step := ""               # "" | "provisioning" | "capture"
static var _pid := -1
static var _started := 0
static var _errors := {}


## Ask for [param url] as [param key]'s picture. "" when queued, else why not.
static func request(key: String, url: String) -> String:
	if busy(key):
		return ""
	var why := Provision.unsupported(ENV)
	if not why.is_empty():
		return "page capture is not available on this machine: " + why
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
		"provisioning":
			return "preparing page capture - " + Provision.describe(ENV)
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
		var out := _abs(DIR.path_join("%s.png" % String(_cur["key"])))
		if FileAccess.file_exists(out):
			landed.append({"key": _cur["key"], "url": _cur["url"], "file": out})
			Provision.release(ENV, "page capture")
			_cur = {}
			_step = ""
		else:
			_fail(_tail(_log("capture")))
		return landed
	if _step == "provisioning":
		_wait_for_env()
		return landed
	if _cur.is_empty() and not _queue.is_empty():
		_cur = _queue.pop_front()
		_step = "provisioning"
		_wait_for_env()
	return landed


## Capture once the environment is ready; fail with the install's own reason if it cannot be.
static func _wait_for_env() -> void:
	if Provision.ensure(ENV):
		_run("capture")
		return
	var s := Provision.state(ENV)
	if not bool(s.get("running", false)) and s.has("error"):
		_fail("the capture environment could not be installed: %s" % s["error"])
	elif not bool(s.get("running", false)) and not Provision.can_install():
		_fail(Provision.hint(ENV))


static func _run(step: String) -> void:
	_step = step
	_started = int(Time.get_unix_time_from_system())
	DirAccess.make_dir_recursive_absolute(_abs(DIR))
	var key := String(_cur["key"])
	var out := _abs(DIR.path_join("%s.png" % key))
	DirAccess.remove_absolute(out)
	var spec := _abs(DIR.path_join("%s.json" % key))
	var f := FileAccess.open(spec, FileAccess.WRITE)
	f.store_string(JSON.stringify({"url": String(_cur["url"]), "out": out, "width": WIDTH,
		"height": HEIGHT, "max_height": MAX_HEIGHT, "user_agent": USER_AGENT}))
	f.close()
	var prog := Deps.venv_bin(_abs(VENV), "python")
	Provision.hold(ENV, "page capture")
	print("ghost capture: %s" % doing())
	_pid = Subprocess.start_logged(prog, PackedStringArray([_abs(SCRIPT), spec]), _log(step),
		"page capture")
	if _pid <= 0:
		_fail("could not start %s" % prog.get_file())


static func _fail(why: String) -> void:
	Provision.release(ENV, "page capture")
	if not _cur.is_empty():
		_errors[String(_cur["key"])] = why
		push_warning("ghost capture: %s failed - %s" % [String(_cur.get("url", "")), why])
	_cur = {}
	_step = ""
	_pid = -1


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
