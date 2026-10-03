extends Node

## NOT a gate (it needs ffmpeg, util-linux and a rendered AVI). The export's SLIDING WINDOW,
## through the exporter's own code: a finished Movie Maker AVI is fed into the scratch path a
## piece at a time, at roughly render speed, while the exporter encodes it as it grows
## (`_start_transcode(true)`) and releases what it has read (`_punch_behind_encoder`). Reports
## the most the scratch file ever held on disk, and checks every frame and the audio came out.
##
##   tests/run_boot_probe.sh tests/export_follow_probe.gd 300 --src <rendered.avi> --dir <scratch>

var _src := ""
var _dir := ""


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size() - 1:
		match args[i]:
			"--src": _src = args[i + 1]
			"--dir": _dir = args[i + 1]
	var ex = preload("res://scripts/exporter.gd").new()
	ex._avi = _dir.path_join("feed.avi")
	ex._out = _dir.path_join("feed.mp4")
	ex._song_dur = 20.0
	ex._punched = ex.PUNCH_KEEP
	DirAccess.remove_absolute(ex._avi)
	DirAccess.remove_absolute(ex._out)
	var data := FileAccess.get_file_as_bytes(_src)
	# as a LIVE render has it: Movie Maker leaves the RIFF and movi sizes 0 until it finishes
	data.encode_u32(4, 0)
	var mv := 12
	while mv < 65536:
		mv = data.find(0x4C, mv)          # 'L' of LIST
		if mv < 0:
			break
		if data.slice(mv, mv + 4).get_string_from_ascii() == "LIST" and data.slice(mv + 8, mv + 12).get_string_from_ascii() == "movi":
			data.encode_u32(mv + 4, 0)
			break
		mv += 1
	var w := FileAccess.open(ex._avi, FileAccess.WRITE)
	var at := 0
	var piece := 1 << 18
	var peak := 0
	var t_punch := 0.0
	while at < data.size():
		w.store_buffer(data.slice(at, mini(at + piece, data.size())))
		w.flush()
		at += piece
		if not ex._live_encode and at > 65536:
			ex._start_transcode(true)
			print("export_follow_probe: live encode started = %s" % ex._live_encode)
			await get_tree().create_timer(1.5).timeout
			var d := "/proc/%d/fd" % ex._transcode_pid
			var da := DirAccess.open(d)
			print("export_follow_probe: alive=%s fd dir=%s" % [Subprocess.alive(ex._transcode_pid), da != null])
			if da != null:
				for fd in DirAccess.get_files_at(d):
					print("   fd %s -> %s" % [fd, da.read_link(d.path_join(fd))])
			var cmd: Array = []
			OS.execute("cat", ["/proc/%d/cmdline" % ex._transcode_pid], cmd)
			print("   cmdline: %s" % String(cmd[0]).replace(char(0), " ") if not cmd.is_empty() else "")
		await get_tree().create_timer(0.12).timeout
		t_punch += 0.12
		if t_punch >= 2.0:
			t_punch = 0.0
			print("export_follow_probe: written %.1f MB, encoder read pos %d" % [float(at) / 1e6, ex._encoder_read_pos()])
			ex._punch_behind_encoder()
		peak = maxi(peak, _on_disk(ex._avi))
	w.close()
	print("export_follow_probe: fed %.1f MB; most ever on disk %.1f MB (released up to %.1f MB)" % [
		float(data.size()) / 1e6, float(peak) / 1e6, float(ex._punched) / 1e6])
	var t0 := Time.get_ticks_msec()
	while DirAccess.dir_exists_absolute("/proc/%d" % ex._transcode_pid) and Time.get_ticks_msec() - t0 < 120000:
		await get_tree().create_timer(2.0).timeout
		print("export_follow_probe: waiting, read pos %d" % ex._encoder_read_pos())
	var fdl := "/proc/%d/fd" % ex._transcode_pid
	if DirAccess.dir_exists_absolute(fdl):
		var da := DirAccess.open(fdl)
		for fd in DirAccess.get_files_at(fdl):
			print("   still open: fd %s -> %s" % [fd, da.read_link(fdl.path_join(fd))])
	var out: Array = []
	OS.execute("ffprobe", ["-v", "error", "-count_frames", "-show_entries",
		"stream=codec_type,nb_read_frames", "-of", "csv=p=0", ex._out], out)
	print("export_follow_probe: output streams: %s" % String(out[0]).strip_edges().replace("\n", " | "))
	ex.free()
	get_tree().quit()


static func _on_disk(path: String) -> int:
	var out: Array = []
	OS.execute("stat", ["-c", "%b", path], out)
	return int(String(out[0]).strip_edges()) * 512 if not out.is_empty() else 0
