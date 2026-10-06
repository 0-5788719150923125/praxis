extends Node
class_name VoiceHost

## VoiceHost - Godot's end of the neural voice subprocess (see VOICE_PLAN.md).
##
## Owns a Python process running `voice_host/host.py` and talks to it in
## newline-delimited JSON over stdio. Adding a model is a file in
## `voice_host/backends/` and a registry entry; nothing here changes. That is the
## swappability requirement, and it is why this is a subprocess rather than a
## GDExtension wrapping ONNX Runtime - a native binding would mean per-platform
## binaries and a rebuild per model.
##
## The environment is ghost's own (`voice_venv`, see [Provision]): built by uv on
## ghost's own Python the first time the voice is wanted, kept current with the rest,
## and reported on the home screen. This waits for it, saying what the install is
## doing, then starts `voice_host/host.py` with that environment's Python and keeps it
## warm for the session.
##
## Nothing here blocks the frame. The wait is polled; synthesis is a request whose
## reply arrives on a later frame. The caller gets signals, not return values.

signal host_ready(backends: PackedStringArray)  # not `ready`: Node already has one
signal failed(stage: String, message: String)
signal progress(stage: String, message: String)
signal synthesized(request_id: int, result: Dictionary)

const ENV := "voice_venv"
const VENV_DIR := "user://voice_venv"
const HOST_REL := "voice_host/host.py"
const POLL_SEC := 0.25

var _state := "idle"                 # idle | provisioning | starting | up | dead
var _pid := -1
var _stdio: FileAccess               # the host's stdin/stdout pair
var _pending := {}                   # request id -> metadata
var _next_id := 1
var _poll := 0.0
var _said := ""                      # the last progress line, so each is emitted once
var _rx := ""                        # partial line buffer
var _stderr: FileAccess              # the host's diagnostics
var _erx := ""


func _ready() -> void:
	set_process(false)


## Bring the host up, waiting for its environment if this is the first use.
## Emits `host_ready` or `failed`; safe to call again after a failure.
func start() -> void:
	if _state in ["up", "starting", "provisioning"]:
		return
	var why := Provision.unsupported(ENV)
	if not why.is_empty():
		failed.emit("deps", "the neural voice is not available on this machine: " + why)
		return
	_state = "provisioning"
	_said = ""
	set_process(true)
	_wait_for_env()


func stop() -> void:
	# Only ask a host that is actually there. On the way out `Boot` has already reaped the
	# registry by the time this runs from `_exit_tree`, so the pipe's far end is gone and the
	# polite shutdown is a write into a dead process (see the note in subprocess.gd, which is
	# where the visible half of this bug was).
	if _state == "up" and Subprocess.alive(_pid):
		_send({"op": "shutdown"})
	if _pid > 0:
		Subprocess.stop(_pid)
		_pid = -1
	Provision.release(ENV, "voice host")
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


# --- the environment -----------------------------------------------------------


## The environment's OWN interpreter - `bin/python` on Unix, `Scripts\python.exe` on
## Windows, which is why this asks [Deps] rather than joining "bin" itself.
func _venv_python() -> String:
	var p := Deps.venv_bin(VENV_DIR, "python")
	return p if FileAccess.file_exists(p) else ""


func _host_script() -> String:
	return ProjectSettings.globalize_path("res://" + HOST_REL)


## Start the host once the environment is ready; until then, say what its install is
## doing (every change, once), and fail with the install's own reason if it stops.
func _wait_for_env() -> void:
	if Provision.ensure(ENV):
		_spawn_host()
		return
	var s := Provision.state(ENV)
	if not bool(s.get("running", false)):
		if s.has("error"):
			_fail("deps", "the voice environment could not be installed: %s" % s["error"])
			return
		if not Provision.can_install():
			_fail("deps", Provision.hint(ENV))
			return
	var line := "Preparing the voice - " + Provision.describe(ENV, s)
	if line != _said:
		_said = line
		progress.emit("deps", line)


func _spawn_host() -> void:
	_state = "starting"
	Provision.hold(ENV, "voice host")
	var py := _venv_python()
	# blocking=false gives a FileAccess over the child's stdio pair. Through [Subprocess] so
	# the host dies with ghost: it is a python process holding a voice model in memory, and a
	# detached one survives the app invisibly, still holding the model and the venv.
	var info := Subprocess.start_with_pipe(py, ["-u", _host_script()], "voice host")
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
	Provision.release(ENV, "voice host")
	_state = "dead"
	set_process(false)
	failed.emit(stage, msg)


# --- pump --------------------------------------------------------------------


func _process(delta: float) -> void:
	_poll += delta
	if _poll < POLL_SEC and _state == "provisioning":
		return
	_poll = 0.0

	match _state:
		"provisioning":
			_wait_for_env()
		"starting", "up":
			_drain()
			_drain_stderr()
			if _pid > 0 and not Subprocess.alive(_pid):
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
			"download":
				_on_download(msg)
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


## The host fetching something it needs (a voice model, the tagger's data), shown beside
## ghost's own installs so the wait on a first Play is never a blank one.
func _on_download(msg: Dictionary) -> void:
	var what := String(msg.get("name", "a voice"))
	if bool(msg.get("finished", false)):
		Provision.report_done("voices", bool(msg.get("ok", true)),
			String(msg.get("error", "downloaded " + what)))
		return
	var done := int(msg.get("done", 0))
	var total := int(msg.get("total", 0))
	Provision.report("voices", "downloading " + what,
		float(done) / float(total) if total > 0 else -1.0, done, total)


func _send(payload: Dictionary) -> void:
	if _stdio == null:
		push_warning("ghost/voice: host is not running")
		return
	_stdio.store_string(JSON.stringify(payload) + "\n")
	_stdio.flush()
