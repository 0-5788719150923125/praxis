extends Node
class_name VoiceHost

## VoiceHost - Godot's end of the neural voice subprocess (see VOICE_PLAN.md).
##
## Owns a Python process running `voice_host/host.py` and talks to it in
## newline-delimited JSON over stdio. Adding a model is a file in
## `voice_host/backends/` and a registry entry; nothing here changes. That is the
## swappability requirement, and it is why this is a subprocess rather than a
## GDExtension wrapping ONNX Runtime - a native binding would mean per-platform
## binaries and a rebuild per model, and ghost has no export presets for any
## platform yet.
##
## The venv follows the pattern `mask_editor.gd` already uses for yt-dlp:
##   1. python3 -m venv user://voice_venv          (once, first ever use)
##   2. <venv>/bin/pip install -r requirements.txt (once; also the repair step)
##   3. <venv>/bin/python voice_host/host.py       (stays warm for the session)
## A dedicated venv means ghost owns its own onnxruntime and can upgrade it
## without touching anything else on the machine.
##
## Nothing here blocks the frame. Bootstrap is polled exactly like the yt-dlp
## import; synthesis is a request whose reply arrives on a later frame. The
## caller gets signals, not return values.

signal host_ready(backends: PackedStringArray)  # not `ready`: Node already has one
signal failed(stage: String, message: String)
signal progress(stage: String, message: String)
signal synthesized(request_id: int, result: Dictionary)

const VENV_DIR := "user://voice_venv"
const HOST_REL := "voice_host/host.py"
const POLL_SEC := 0.25

var _state := "idle"                 # idle | venv | deps | starting | up | dead
var _pid := -1
var _stdio: FileAccess               # the host's stdin/stdout pair
var _pending := {}                   # request id -> metadata
var _next_id := 1
var _poll := 0.0
var _boot_pid := -1
var _repaired := false               # deps reinstalled once before giving up
var _rx := ""                        # partial line buffer
var _stderr: FileAccess              # the host's diagnostics
var _erx := ""


func _ready() -> void:
	set_process(false)


## Bring the host up, bootstrapping the venv if this is the first ever use.
## Emits `ready` or `failed`; safe to call again after a failure.
func start() -> void:
	if _state in ["up", "starting"]:
		return
	if not _python_available():
		failed.emit("python", "python3 was not found on PATH")
		return
	if _venv_python().is_empty():
		_begin_venv()
	elif not _deps_present():
		_begin_deps()
	else:
		_spawn_host()
	set_process(true)


func stop() -> void:
	if _state == "up":
		_send({"op": "shutdown"})
	if _pid > 0:
		OS.kill(_pid)
		_pid = -1
	_state = "idle"
	set_process(false)


func _exit_tree() -> void:
	stop()


## Synthesize. Returns the request id immediately; the audio arrives later on
## `synthesized`. `out_path` is where the WAV lands - the host writes the file
## and returns its path rather than shipping megabytes back through the pipe.
func request(text: String, voice: String, out_path: String,
		params: Dictionary = {}, phonemes: Variant = null) -> int:
	var id := _next_id
	_next_id += 1
	var req := {"id": id, "op": "synthesize", "text": text, "voice": voice,
		"out": ProjectSettings.globalize_path(out_path), "params": params}
	if phonemes != null:
		req["phonemes"] = phonemes
	_pending[id] = {"voice": voice}
	_send(req)
	return id


func capabilities() -> int:
	var id := _next_id
	_next_id += 1
	_pending[id] = {"op": "capabilities"}
	_send({"id": id, "op": "capabilities"})
	return id


func list_voices() -> int:
	var id := _next_id
	_next_id += 1
	_pending[id] = {"op": "voices"}
	_send({"id": id, "op": "voices"})
	return id


func is_up() -> bool:
	return _state == "up"


# --- bootstrap ---------------------------------------------------------------


static func _which(prog: String) -> String:
	# a GUI-launched Godot does not inherit a shell PATH, so ask explicitly -
	# the same resolution mask_editor.gd and assistant.gd already use
	var out := []
	if OS.execute("which", [prog], out) == 0 and out.size() > 0:
		var p := String(out[0]).strip_edges().split("\n")[0].strip_edges()
		if not p.is_empty():
			return p
	return ""


func _python_available() -> bool:
	return not _which("python3").is_empty()


func _venv_python() -> String:
	var p := ProjectSettings.globalize_path(VENV_DIR).path_join("bin").path_join("python")
	return p if FileAccess.file_exists(p) else ""


func _host_script() -> String:
	return ProjectSettings.globalize_path("res://" + HOST_REL)


## Import EVERY module the host actually needs, not just the first two.
##
## This checked only onnxruntime and numpy, so a venv created before the
## phonemizer was added looked healthy forever and the deps step never re-ran -
## reported as "eSpeak phonemizer unavailable". Any future requirements.txt
## addition must be listed here too, or the same silent staleness returns.
const REQUIRED_IMPORTS := "import onnxruntime, numpy, espeakng_loader, phonemizer"


func _deps_present() -> bool:
	var py := _venv_python()
	if py.is_empty():
		return false
	var out := []
	return OS.execute(py, ["-c", REQUIRED_IMPORTS], out) == 0


func _begin_venv() -> void:
	_state = "venv"
	progress.emit("venv", "Creating the voice environment…")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(VENV_DIR))
	_boot_pid = OS.create_process(_which("python3"),
		["-m", "venv", ProjectSettings.globalize_path(VENV_DIR)])
	if _boot_pid <= 0:
		_fail("venv", "could not start python3 -m venv")


func _begin_deps() -> void:
	_state = "deps"
	progress.emit("deps", "Installing voice dependencies (a minute or so)…")
	var req := ProjectSettings.globalize_path("res://voice_host/requirements.txt")
	_boot_pid = OS.create_process(_venv_python(),
		["-m", "pip", "install", "--upgrade", "-r", req])
	if _boot_pid <= 0:
		_fail("deps", "could not start pip")


func _spawn_host() -> void:
	_state = "starting"
	var py := _venv_python()
	# blocking=false gives a FileAccess over the child's stdio pair
	var info := OS.execute_with_pipe(py, ["-u", _host_script()])
	if info.is_empty():
		_fail("host", "could not start the voice host process")
		return
	_pid = int(info.get("pid", -1))
	_stdio = info.get("stdio")
	# The child's stderr is a SEPARATE pipe. Leaving it unread loses every
	# diagnostic the host prints AND risks the child blocking once the pipe
	# fills, which presents as "the voice host exited unexpectedly" with no
	# explanation anywhere. Drained every frame and echoed with print(), so it
	# lands in the terminal ghost was launched from - copy-pasteable, unlike the
	# in-game console.
	_stderr = info.get("stderr")
	_rx = ""
	_erx = ""


func _fail(stage: String, msg: String) -> void:
	print("ghost/voice: FAILED at %s - %s" % [stage, msg])
	_state = "dead"
	set_process(false)
	failed.emit(stage, msg)


# --- pump --------------------------------------------------------------------


func _process(delta: float) -> void:
	_poll += delta
	if _poll < POLL_SEC and _state in ["venv", "deps"]:
		return
	_poll = 0.0

	match _state:
		"venv":
			if _boot_pid > 0 and not OS.is_process_running(_boot_pid):
				_boot_pid = -1
				if _venv_python().is_empty():
					_fail("venv", "the environment was not created")
				else:
					_begin_deps()
		"deps":
			if _boot_pid > 0 and not OS.is_process_running(_boot_pid):
				_boot_pid = -1
				if _deps_present():
					_spawn_host()
				elif not _repaired:
					# one automatic repair pass, mirroring the yt-dlp retry:
					# a half-installed venv is the common failure, not a rare one
					_repaired = true
					_begin_deps()
				else:
					_fail("deps", "voice dependencies could not be installed - "
						+ "see the >_ log for pip's output")
		"starting", "up":
			_drain()
			_drain_stderr()
			if _pid > 0 and not OS.is_process_running(_pid):
				_drain_stderr()          # whatever it managed to say on the way out
				_fail("host", "the voice host exited unexpectedly - see the "
					+ "ghost/voice lines above in the terminal")


## Read whatever the host has written and dispatch complete lines. Partial lines
## are held: a JSON object split across two reads is normal on a pipe.
func _drain() -> void:
	if _stdio == null:
		return
	# read whatever is buffered; get_as_text() has no skip-cr parameter in 4.x
	if _stdio.get_length() > _stdio.get_position():
		_rx += _stdio.get_as_text()
	while true:
		var nl := _rx.find("\n")
		if nl < 0:
			break
		var line := _rx.substr(0, nl).strip_edges()
		_rx = _rx.substr(nl + 1)
		if not line.is_empty():
			_handle(line)


## Echo the host's stderr to the terminal, line by line. Godot's print() goes to
## the launching shell's stdout, which is the one place the user can select and
## copy text from.
func _drain_stderr() -> void:
	if _stderr == null:
		return
	if _stderr.get_length() > _stderr.get_position():
		_erx += _stderr.get_as_text()
	while true:
		var nl := _erx.find("\n")
		if nl < 0:
			break
		var line := _erx.substr(0, nl)
		_erx = _erx.substr(nl + 1)
		if not line.strip_edges().is_empty():
			print("ghost/voice: " + line)


func _handle(line: String) -> void:
	var parsed: Variant = JSON.parse_string(line)
	if typeof(parsed) != TYPE_DICTIONARY:
		push_warning("ghost/voice: unparseable host line: " + line.substr(0, 120))
		return
	var msg: Dictionary = parsed

	if msg.has("event"):
		match String(msg.event):
			"ready":
				_state = "up"
				host_ready.emit(PackedStringArray(msg.get("backends", [])))
			"backend_unavailable":
				# reportable, not fatal: the other backends still work
				progress.emit("backend", "%s unavailable: %s"
					% [msg.get("backend", "?"), msg.get("error", "")])
		return

	var id := int(msg.get("id", -1))
	var meta: Dictionary = _pending.get(id, {})
	_pending.erase(id)
	if not bool(msg.get("ok", false)):
		var err := String(msg.get("error", "unknown error"))
		print("ghost/voice: request %d failed: %s" % [id, err])
		failed.emit("synthesize", err)
		return
	synthesized.emit(id, msg)


func _send(payload: Dictionary) -> void:
	if _stdio == null:
		push_warning("ghost/voice: host is not running")
		return
	_stdio.store_string(JSON.stringify(payload) + "\n")
	_stdio.flush()
