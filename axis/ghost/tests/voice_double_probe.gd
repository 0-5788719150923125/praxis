extends Node

## NOT a gate (it needs the voice host). Records what a live reading actually PLAYS: the real
## Generative panel and voice host, the user's own settings, a reading started at a chosen
## sentence through the scrub path, and the master bus captured by an AudioEffectRecord. Written
## to answer "am I hearing two voices?" against the raw chunk audio the host wrote.
##
##   tests/run_boot_probe.sh tests/voice_double_probe.gd 300 --find "What our kings" --secs 30 --out <wav>

var _find := "What our kings"
var _secs := 30.0
var _out := "user://voice_double.wav"


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size() - 1:
		match args[i]:
			"--find": _find = args[i + 1]
			"--secs": _secs = float(args[i + 1])
			"--out": _out = args[i + 1]
	var ed: GenerativeEditor = GenerativeEditor.new()
	ed.begin_stream = func(fp: int, sr: int, _w: Array) -> AudioStreamGeneratorPlayback:
		print("voice_double_probe: begin_stream (sr %d)" % sr)
		Director.detach()
		Spectrum.stop()
		return Spectrum.begin_stream(fp, sr)
	ed.end_stream = func() -> void:
		print("voice_double_probe: end_stream")
		Spectrum.stop()
	add_child(ed)
	var t0 := Time.get_ticks_msec()
	while not (ed._host != null and ed._host.is_up() and ed._voices.selected >= 0):
		if Time.get_ticks_msec() - t0 > 180000:
			print("voice_double_probe: the voice host never came up")
			get_tree().quit(2)
			return
		await get_tree().create_timer(0.5).timeout
	var body := ed._doc.pull().strip_edges()
	var chunks: Array = ed._build_chunks(body)
	var k := -1
	var words := _find.to_lower().split(" ")
	for i in chunks.size():
		var said := PackedStringArray()
		for w in (chunks[i] as Dictionary)["words"]:
			said.append(String((w as Dictionary)["text"]).to_lower())
		if " ".join(said).contains(" ".join(words)):
			k = i
			break
	print("voice_double_probe: %d chunks; '%s' is chunk %d" % [chunks.size(), _find, k])
	var rec := AudioEffectRecord.new()
	AudioServer.add_bus_effect(0, rec)
	rec.set_recording_active(true)
	ed._speak_from(maxi(k, 0))
	await get_tree().create_timer(_secs).timeout
	rec.set_recording_active(false)
	var got := rec.get_recording()
	print("voice_double_probe: drained chunks %s, epoch %d" % [str(ed._chunk_at.keys()), ed._epoch])
	if got == null:
		print("voice_double_probe: nothing recorded")
	else:
		got.save_to_wav(_out)
		print("voice_double_probe: recorded %.1f s -> %s (mix %d Hz)" % [got.get_length(), ProjectSettings.globalize_path(_out), got.mix_rate])
	get_tree().quit()
