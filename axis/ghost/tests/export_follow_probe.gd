extends Node

## NOT a gate (it needs ffmpeg, util-linux, xvfb-run and a GPU). The export's SLIDING WINDOW,
## through the exporter's own state machine (`_tick_render`, `_tick_transcode`) against a REAL
## Movie Maker render - a small project written into --dir, a field of moving disks and a tone,
## launched the way the exporter launches one ([method Exporter.virtual_display]).
##
##   whole    - the scratch's NAME leaves the folder while the render is still writing, what it
##              ever held on disk stays small, every frame and the audio reach the MP4, and
##              nothing is left: no scratch, no descriptor on it, no render, no Xvfb
##   stopped  - closing ghost mid-render takes the render's Godot and its Xvfb with it (and the
##              control: the launch this replaced leaves that Godot rendering)
##   lost     - the encoder dying after a release abandons the export and stops the render
##   early    - the encoder dying before any release falls back to encoding the whole AVI after
##
##   tests/run_boot_probe.sh tests/export_follow_probe.gd 600 --dir <scratch on a real disk>
##
## Refuses to run during an export: the exporter's code it drives clears the project's
## override.cfg and rewrites the transcode progress file, which a running export owns.

const FPS := 30

var _dir := ""
var _exe := ""
var _fails := 0
var _peak := 0
var _gone_at := -1
var _sample_t := 0
var _log_t := 0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size() - 1:
		if args[i] == "--dir":
			_dir = args[i + 1]
	if _dir.is_empty() or OS.get_name() != "Linux":
		print("export_follow_probe: needs --dir <a directory on a real disk>, on Linux")
		get_tree().quit(2)
		return
	if FileAccess.file_exists("res://override.cfg"):
		print("export_follow_probe: an export is running (override.cfg exists) - run this once it is done")
		get_tree().quit(2)
		return
	DirAccess.make_dir_recursive_absolute(_dir)
	_exe = OS.get_executable_path()
	_write_writer()
	await _whole()
	await _stopped()
	await _control()
	await _lost()
	await _early()
	print("export_follow_probe: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	get_tree().quit(1 if _fails > 0 else 0)


# --- the runs ----------------------------------------------------------------------------------

func _whole() -> void:
	print("export_follow_probe: [whole]")
	var frames := 900
	var ex: Exporter = _exporter("whole", frames)
	var tree: Dictionary = await _launch(ex, frames)
	var gone: bool = await _drive(ex, func() -> bool: return ex._state == "done", 300.0)
	_check(gone, "the export finished")
	_check(str(ex._status.text).begins_with("✓"), "it reports saved: %s" % ex._status.text)
	_check(_gone_at > 0, "the scratch left the folder while the render was writing (at %.1f MB)" % (_gone_at / 1e6))
	var total := _file_len(ex._out)
	_check(_peak > 0 and _peak < (64 << 20), "the most it held on disk: %.1f MB" % (_peak / 1e6))
	var counts := _stream_frames(ex._out)
	_check(absi(int(counts.get("video", 0)) - frames) <= 2, "video frames %s of %d" % [counts.get("video", 0), frames])
	_check(int(counts.get("audio", 0)) > 0, "audio frames %s" % counts.get("audio", 0))
	await _nothing_left(ex, tree)
	print("   (MP4 %.1f MB)" % (total / 1e6))
	ex.free()


func _stopped() -> void:
	print("export_follow_probe: [stopped]")
	var ex: Exporter = _exporter("stopped", 3000)
	var tree: Dictionary = await _launch(ex, 3000)
	await _drive(ex, func() -> bool: return ex._unlinked, 60.0)
	_check(ex._unlinked, "the scratch left the folder before the stop")
	tree = _tree(ex._render_pid)         # Xvfb and Godot are up by now
	ex.notification(NOTIFICATION_WM_CLOSE_REQUEST)
	await _nothing_left(ex, tree)
	ex.free()


## THE CONTROL for the pact: the launch this replaced, stopped the same way, leaves its Godot
## rendering. Cleaned up by pid afterwards.
func _control() -> void:
	print("export_follow_probe: [control - the old launch]")
	var avi := _dir.path_join("control.render.avi")
	DirAccess.remove_absolute(avi)
	var xvfb := Deps.resolve("xvfb-run")
	var argv := PackedStringArray(["-a", "-s", "-screen 0 960x540x24", _exe])
	argv.append_array(_writer_args(avi, 3000))
	var pid := _start_writer(xvfb, argv, "probe control")
	await _until(func() -> bool: return _godot_of(_tree(pid)) > 0 and _file_len(avi) > 65536, 60.0)
	var tree := _tree(pid)
	var godot := _godot_of(tree)
	Subprocess.stop(pid)
	await get_tree().create_timer(2.0).timeout
	_check(godot > 0 and _alive(tree, godot), "control: stopping xvfb-run alone leaves its Godot rendering")
	for p in tree.keys():
		OS.kill(int(p))
	await get_tree().create_timer(1.0).timeout
	DirAccess.remove_absolute(avi)


func _lost() -> void:
	print("export_follow_probe: [lost - the encoder dies after a release]")
	var ex: Exporter = _exporter("lost", 3000)
	var tree: Dictionary = await _launch(ex, 3000)
	await _drive(ex, func() -> bool: return ex._unlinked, 60.0)
	tree = _tree(ex._render_pid)
	Subprocess.stop(ex._transcode_pid)
	await _drive(ex, func() -> bool: return ex._state == "done", 20.0)
	_check(str(ex._status.text).begins_with("⚠  Encoding stopped"), "abandoned: %s" % ex._status.text)
	await _nothing_left(ex, tree)
	ex.free()


func _early() -> void:
	print("export_follow_probe: [early - the encoder dies before any release]")
	var frames := 300
	var ex: Exporter = _exporter("early", frames)
	var tree: Dictionary = await _launch(ex, frames)
	await _drive(ex, func() -> bool: return ex._live_encode, 60.0)
	Subprocess.stop(ex._transcode_pid)
	await _drive(ex, func() -> bool: return not ex._live_encode, 10.0)
	_check(not ex._released and FileAccess.file_exists(ex._avi), "nothing released and the AVI keeps its name")
	await _drive(ex, func() -> bool: return ex._state == "done", 300.0)
	_check(str(ex._status.text).begins_with("✓"), "encoded after the render: %s" % ex._status.text)
	var counts := _stream_frames(ex._out)
	_check(absi(int(counts.get("video", 0)) - frames) <= 2, "video frames %s of %d" % [counts.get("video", 0), frames])
	await _nothing_left(ex, tree)
	ex.free()


# --- driving the exporter ----------------------------------------------------------------------

func _exporter(tag: String, frames: int) -> Exporter:
	var ex: Exporter = Exporter.new()
	ex._status = Label.new()
	ex.add_child(ex._status)
	ex._avi = _dir.path_join("%s.render.avi" % tag)
	ex._out = _dir.path_join("%s.mp4" % tag)
	ex._song_dur = float(frames) / FPS
	ex._quality = {"label": "probe", "w": 1280, "h": 720, "fps": FPS, "tag": "probe", "ss": 1.0}
	DirAccess.remove_absolute(ex._out)
	_peak = 0
	_gone_at = -1
	return ex


## Start the render as _start_render does - fresh scratch, virtual display, pact - and return
## its process tree once Godot is up.
func _launch(ex: Exporter, frames: int) -> Dictionary:
	ex._begin_scratch()
	var argv := Exporter.virtual_display(_exe, _writer_args(ex._avi, frames))
	if argv.is_empty():
		_check(false, "xvfb-run is installed")
		return {}
	ex._render_pid = _start_writer(argv[0], argv.slice(1), "probe render")
	ex._state = "rendering"
	ex._stall_t = 0.0
	ex._stall_frac = -1.0
	ex._stall_size = 0
	await _until(func() -> bool: return _godot_of(_tree(ex._render_pid)) > 0, 30.0)
	return _tree(ex._render_pid)


## Tick the exporter as its _process does until [param done], sampling the scratch on the way.
func _drive(ex: Exporter, done: Callable, limit_s: float) -> bool:
	var t0 := Time.get_ticks_msec()
	while not done.call():
		if Time.get_ticks_msec() - t0 > limit_s * 1000.0:
			return false
		await get_tree().process_frame
		var dt := get_process_delta_time()
		match ex._state:
			"rendering":
				ex._tick_render(dt)
			"transcoding":
				ex._tick_transcode()
		_sample(ex)
	return true


func _sample(ex: Exporter) -> void:
	if Time.get_ticks_msec() - _sample_t < 300:
		return
	_sample_t = Time.get_ticks_msec()
	if ex._state == "rendering" and _gone_at < 0 and ex._live_encode \
			and not FileAccess.file_exists(ex._avi) and Subprocess.alive(ex._render_pid):
		_gone_at = ex._scratch_len()
	var on_disk := -1
	if not ex._held.is_empty():
		on_disk = _blocks(ex._held, true)
	elif FileAccess.file_exists(ex._avi):
		on_disk = _blocks(ex._avi, false)
	_peak = maxi(_peak, on_disk)
	if ex._state == "rendering" and ex._live_encode and Time.get_ticks_msec() - _log_t >= 3000:
		_log_t = Time.get_ticks_msec()
		print("   written %6.1f MB   encoder at %6.1f MB   on disk %5.1f MB   in the folder: %s" % [
			ex._scratch_len() / 1e6, ex._encoder_read_pos() / 1e6, on_disk / 1e6,
			"yes" if FileAccess.file_exists(ex._avi) else "no"])


## After an export ends, however it ended: no scratch in the folder, no descriptor of ghost's on
## it, and every process of the render gone - Godot at once, Xvfb within its linger.
func _nothing_left(ex: Exporter, tree: Dictionary) -> void:
	var want := ProjectSettings.globalize_path(ex._avi)
	await _until(func() -> bool: return not _any_alive(tree), Exporter.XVFB_LINGER + 8.0)
	await _until(func() -> bool: return not Subprocess.alive(ex._transcode_pid), 90.0)
	var godot := _godot_of(tree)
	_check(godot > 0 and not _alive(tree, godot), "the render's Godot is gone")
	_check(not _any_alive(tree), "its whole tree is gone (%s)" % ", ".join(PackedStringArray(tree.values())))
	_check(not FileAccess.file_exists(ex._avi), "no scratch in the folder")
	var held := 0
	var dir := "/proc/%d/fd" % OS.get_process_id()
	var d := DirAccess.open(dir)
	for fd in DirAccess.get_files_at(dir):
		var link := d.read_link(dir.path_join(fd))
		if link == want or link == want + " (deleted)":
			held += 1
	_check(held == 0 and ex._hold == null, "ghost holds nothing of it")
	_check(not Subprocess.alive(ex._transcode_pid), "the encoder is gone")


# --- the writer --------------------------------------------------------------------------------

func _writer_dir() -> String:
	return _dir.path_join("writer")


func _writer_args(avi: String, frames: int) -> PackedStringArray:
	return PackedStringArray(["--path", _writer_dir(), "--write-movie", avi, "--fixed-fps", str(FPS),
		"--quit-after", str(frames)])


## Start a writer with its user:// (Godot's shader caches) inside --dir, not the author's data.
func _start_writer(path: String, argv: PackedStringArray, tag: String) -> int:
	var was := OS.get_environment("XDG_DATA_HOME")
	OS.set_environment("XDG_DATA_HOME", _dir.path_join("data"))
	var pid := Subprocess.start(path, argv, tag)
	if was.is_empty():
		OS.unset_environment("XDG_DATA_HOME")
	else:
		OS.set_environment("XDG_DATA_HOME", was)
	return pid


## The render's stand-in: a project Movie Maker records exactly as it records ghost's.
func _write_writer() -> void:
	var w := _writer_dir()
	DirAccess.make_dir_recursive_absolute(w)
	_put(w.path_join("project.godot"), "config_version=5\n\n[application]\n\nconfig/name=\"export_probe_writer\"\n"
		+ "run/main_scene=\"res://writer.tscn\"\n\n[debug]\n\nfile_logging/enable_file_logging.pc=false\n\n"
		+ "[display]\n\nwindow/size/viewport_width=1280\nwindow/size/viewport_height=720\n"
		+ "window/size/window_width_override=480\nwindow/size/window_height_override=270\n"
		+ "window/stretch/mode=\"viewport\"\n\n[editor]\n\nmovie_writer/video_quality=1.0\n")
	_put(w.path_join("writer.tscn"), "[gd_scene load_steps=2 format=3]\n\n"
		+ "[ext_resource type=\"Script\" path=\"res://writer.gd\" id=\"1\"]\n\n"
		+ "[node name=\"Writer\" type=\"Node2D\"]\nscript = ExtResource(\"1\")\n")
	_put(w.path_join("writer.gd"), """extends Node2D

func _ready() -> void:
	var rate := 44100
	var data := PackedByteArray()
	data.resize(rate * 2)
	for i in rate:
		data.encode_s16(i * 2, int(sin(TAU * 220.0 * i / rate) * 6000.0))
	var wav := AudioStreamWAV.new()
	wav.format = AudioStreamWAV.FORMAT_16_BITS
	wav.mix_rate = rate
	wav.data = data
	wav.loop_mode = AudioStreamWAV.LOOP_FORWARD
	wav.loop_end = rate
	var p := AudioStreamPlayer.new()
	p.stream = wav
	add_child(p)
	p.play()

func _process(_dt: float) -> void:
	OS.delay_msec(25)   # a render's pace: slower than the encoder, as a real one is
	queue_redraw()

func _draw() -> void:
	var t := Engine.get_frames_drawn() / 30.0
	var r := get_viewport_rect().size
	for i in 600:
		var p := Vector2(fposmod(i * 97.3 + t * (40.0 + i % 7 * 25.0), r.x),
			fposmod(i * 61.7 + t * (30.0 + i % 5 * 20.0), r.y))
		draw_circle(p, 4.0 + (i % 11) * 2.5, Color.from_hsv(fmod(i * 0.618, 1.0), 0.7, 0.4 + 0.5 * fmod(i * 0.37, 1.0)))
""")


func _put(path: String, text: String) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()


# --- processes and files -----------------------------------------------------------------------

## pid -> cmdline for [param pid] and everything under it.
func _tree(pid: int) -> Dictionary:
	var out := {}
	var todo: Array[int] = [pid]
	while not todo.is_empty():
		var p: int = todo.pop_back()
		var cmd := _cmdline(p)
		if cmd.is_empty():
			continue
		out[p] = cmd
		var f := FileAccess.open("/proc/%d/task/%d/children" % [p, p], FileAccess.READ)
		if f == null:
			continue
		for c in f.get_buffer(4096).get_string_from_utf8().split(" ", false):
			todo.append(int(c))
	return out


func _godot_of(tree: Dictionary) -> int:
	for p in tree.keys():
		var cmd: String = tree[p]
		if cmd.contains("--write-movie") and not cmd.contains("xvfb-run") and not cmd.contains("setpriv"):
			return int(p)
	return -1


func _cmdline(pid: int) -> String:
	var f := FileAccess.open("/proc/%d/cmdline" % pid, FileAccess.READ)
	if f == null:
		return ""
	# the arguments are NUL-separated, and a NUL ends a decoded string
	var b := f.get_buffer(8192)
	for i in b.size():
		if b[i] == 0:
			b[i] = 32
	return b.get_string_from_utf8().strip_edges()


## Alive and still the same program - a zombie has no cmdline, and a reused pid has another.
func _alive(tree: Dictionary, pid: int) -> bool:
	return _cmdline(pid) == str(tree.get(pid, ""))


func _any_alive(tree: Dictionary) -> bool:
	for p in tree.keys():
		if _alive(tree, int(p)):
			return true
	return false


func _until(done: Callable, limit_s: float) -> bool:
	var t0 := Time.get_ticks_msec()
	while not done.call():
		if Time.get_ticks_msec() - t0 > limit_s * 1000.0:
			return false
		await get_tree().create_timer(0.1).timeout
	return true


func _blocks(path: String, follow: bool) -> int:
	var out: Array = []
	var args := PackedStringArray(["-L", "-c", "%b", path]) if follow else PackedStringArray(["-c", "%b", path])
	OS.execute("stat", args, out)
	return int(String(out[0]).strip_edges()) * 512 if not out.is_empty() else -1


func _file_len(path: String) -> int:
	var f := FileAccess.open(path, FileAccess.READ)
	return f.get_length() if f != null else 0


func _stream_frames(path: String) -> Dictionary:
	var out: Array = []
	OS.execute(Deps.resolve("ffprobe"), ["-v", "error", "-count_frames", "-show_entries",
		"stream=codec_type,nb_read_frames", "-of", "csv=p=0", path], out)
	var counts := {}
	for line in String(out[0] if not out.is_empty() else "").split("\n", false):
		var parts := line.strip_edges().split(",")
		if parts.size() == 2:
			counts[parts[0]] = int(parts[1])
	return counts


func _check(ok: bool, what: String) -> void:
	if not ok:
		_fails += 1
	print("   %s %s" % ["ok  " if ok else "FAIL", what])
