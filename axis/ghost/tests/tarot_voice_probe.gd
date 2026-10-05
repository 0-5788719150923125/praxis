extends Node

## NOT a gate. The tarot mode end to end in a real boot: the Tarot panel opens a show, the real
## voice reads an episode already made, and the table follows the voice's real word timings -
## then, optionally, the panel renders the export take and its sidecar is checked.
##
##   GHOST_PROBE_GPU=1 GHOST_PROBE_MUTE=1 tests/run_boot_probe.sh tests/tarot_voice_probe.gd 600 \
##       --spec ../../../rift/tarot/truthful-tarot.md --seed 1 --out /tmp/tv/f --play 80 --export 1
##
## Read-only like every probe: nothing is written to the show's document or the author's
## settings. Prints what the voice planned, where the table placed each action against the real
## timings, and photographs the table at `--times` (show seconds).

const W := 1280
const H := 720

var _ed: TarotEditor
var _stage: SubViewport
var _medium: Medium
var _subs: Subtitles
var _out := "user://tarot_voice"
var _times: Array = [3.0, 30.0]
var _play := 60.0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var spec := ""
	var seed := 1
	var export := false
	for i in args.size():
		if i + 1 >= args.size():
			break
		match args[i]:
			"--spec": spec = args[i + 1]
			"--seed": seed = int(args[i + 1])
			"--out": _out = args[i + 1]
			"--play": _play = float(args[i + 1])
			"--export": export = args[i + 1] == "1"
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))
	if not spec.is_empty() and not FileAccess.file_exists(spec):
		print("tarot_voice_probe: FAILED - no show document at %s" % spec)
		get_tree().quit(2)
		return
	_ed = TarotEditor.new()
	_ed.begin_stream = _begin
	_ed.end_stream = func() -> void: pass
	add_child(_ed)
	_ok(Director.resolved_medium() == "tarot", "the tarot mode pins its medium (%s)" % Director.resolved_medium())
	if not spec.is_empty():
		_ed._doc._on_picked(ProjectSettings.globalize_path(spec) if spec.begins_with("res://") else spec)
	_ed._knobs["seed"] = seed
	_ed._open_episode()
	print("tarot_voice_probe: show '%s' (%s), episode #%d complete=%s" % [_ed._show_title(),
		_ed._show_key(), seed, str(_ed._episode.complete())])
	var t0 := Time.get_ticks_msec()
	while _ed._voice_meta.is_empty() and Time.get_ticks_msec() - t0 < 180000:
		await get_tree().process_frame
	if _ed._voice_meta.is_empty():
		print("tarot_voice_probe: FAILED - the voice host never listed its voices")
		get_tree().quit(2)
		return
	var cast := _ed._cast_dict()
	print("tarot_voice_probe: voices up after %.1fs; cast %s" % [float(Time.get_ticks_msec() - t0) / 1000.0, str(cast)])
	_ok(_ed._has_reading(), "the episode has a reading")
	_ed._on_speak()
	var holds := 0
	for c in _ed._chunks:
		holds += ((c as Dictionary).get("holds", []) as Array).size()
	print("tarot_voice_probe: planned %d chunks, %d rests" % [_ed._chunks.size(), holds])
	var acts: Array = TarotScript.parse(_ed._episode.script())["actions"]
	_ok(holds == acts.size(), "every action has its rest in the voice (%d rests, %d actions)" % [holds, acts.size()])
	var shots := _times.duplicate()
	shots.sort()
	var start := Time.get_ticks_msec()
	while float(Time.get_ticks_msec() - start) / 1000.0 < _play + 30.0:
		await get_tree().process_frame
		if _subs == null:
			continue
		var now := _subs.now()
		if not shots.is_empty() and now >= float(shots[0]):
			var want := float(shots.pop_front())
			var img := _stage.get_texture().get_image()
			var path := "%s_t%05.1f.png" % [_out, want]
			img.save_png(path)
			print("tarot_voice_probe: t=%.1f -> %s" % [now, path])
		if now >= _play:
			break
	if _medium != null:
		var tm := _medium as TarotMedium
		print("tarot_voice_probe: the table followed %d of %d voice words; %d action(s) placed" % [
			tm._follow.known_last() + 1, (tm._parse.get("spoken", PackedStringArray()) as PackedStringArray).size(),
			tm._sched.size()])
		for e in tm._sched:
			print("   %-8s card %d  at %.2fs  speed %.2f" % [String((e["a"] as Dictionary)["kind"]),
				int((e["a"] as Dictionary)["card"]), float(e["t0"]), float(e["s"])])
		_ok(tm._sched.size() >= 2, "the shuffle and the first draw were placed against the real voice")
	if export:
		_ed._stop_speaking()
		var t1 := Time.get_ticks_msec()
		var take: String = await _ed.export_take()
		print("tarot_voice_probe: export take %s in %.0fs" % [take, float(Time.get_ticks_msec() - t1) / 1000.0])
		_ok(not take.is_empty() and FileAccess.file_exists(take), "the export rendered a take")
		var side := FileAccess.get_file_as_string(take.get_basename() + ".json")
		var j := JSON.new()
		if j.parse(side) == OK and j.data is Dictionary:
			var book: Dictionary = (j.data as Dictionary).get("book", {})
			var tarot: Dictionary = book.get("tarot", {})
			_ok(not tarot.is_empty() and (tarot.get("cards", []) as Array).size() == _ed._episode.card_count(),
				"the sidecar carries the episode for the render's table")
			_ok(TarotScript.is_tarot(String(book.get("source", ""))), "the sidecar carries the reading's script")
			var words: Array = (j.data as Dictionary).get("words", [])
			print("tarot_voice_probe: sidecar - %d words, %.1fs to the last" % [words.size(),
				float((words.back() as Dictionary)["t1"]) if not words.is_empty() else 0.0])
		else:
			_ok(false, "the sidecar is readable JSON")
	print("tarot_voice_probe: %s" % ("ALL OK" if _fails == 0 else "%d FAILURE(S)" % _fails))
	Director.detach()
	Spectrum.stop()
	get_tree().quit(1 if _fails > 0 else 0)


var _fails := 0

func _ok(cond: bool, what: String) -> void:
	print("  %s %s" % ["ok  " if cond else "FAIL", what])
	if not cond:
		_fails += 1


## main's _begin_generative_stream, as a probe can host it: the stage, the pinned medium, and the
## subtitle overlay the medium follows.
func _begin(fp: int, sr: int, words: Array) -> AudioStreamGeneratorPlayback:
	Director.detach()
	Spectrum.stop()
	var pb: AudioStreamGeneratorPlayback = Spectrum.begin_stream(fp, sr)
	if _stage == null:
		_stage = SubViewport.new()
		_stage.size = Vector2i(W, H)
		_stage.own_world_3d = true
		_stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		add_child(_stage)
		_medium = Medium.make(Director.resolved_medium())
		_medium.mount(_stage)
	Director.attach(_stage, _medium)
	if _subs != null:
		_subs.queue_free()
	_subs = preload("res://scripts/subtitles.gd").new()
	_subs.words = words
	_subs.document = _ed.book_document()
	add_child(_subs)
	_medium.bind_captions(_subs)
	_ed.subtitles = _subs
	print("tarot_voice_probe: the stream opened (%d Hz), medium '%s'" % [sr, _medium.key])
	return pb
