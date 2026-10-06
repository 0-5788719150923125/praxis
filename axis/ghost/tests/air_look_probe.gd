extends Node

## NOT a gate. The table's AIR ([Effects]) over a real episode, photographed: a COPY of the episode
## (the author's own is never touched) with effects laid into its table, read with a synthetic voice
## (tarot_look_probe's), and frames written at the times asked for - or around every moment a burst
## can mark, with `--moments 1`.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/air_look_probe.gd 400 \
##       --show truthful-tarot --seed 352029 --spec <effects.json> [--times 12,30] [--moments 1] [--out /tmp/air/a]
##
## `--spec` is a JSON list of effects, or a table with `effects` - laid over the episode's own table.
## `--clip A,B` writes every frame from A to B seconds, for judging motion:
##   ffmpeg -framerate 30 -i <out>_c%04d.png -pix_fmt yuv420p air.mp4

const W := 1280
const H := 720
const DT := 1.0 / 30.0
const ROOT := "user://air_look"


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var show := "truthful-tarot"
	var seed := 0
	var spec_path := ""
	var out := "user://air_look/look"
	var times: Array = []
	var moments := false
	var clip := Vector2(-1.0, -1.0)
	var show_motes := false
	for i in args.size() - 1:
		match args[i]:
			"--show": show = args[i + 1]
			"--seed": seed = int(args[i + 1])
			"--spec": spec_path = args[i + 1]
			"--out": out = args[i + 1]
			"--moments": moments = args[i + 1] == "1"
			"--motes": show_motes = args[i + 1] == "1"
			"--times":
				for x in String(args[i + 1]).split(","):
					times.append(float(x))
			"--clip":
				var ab := String(args[i + 1]).split(",")
				clip = Vector2(float(ab[0]), float(ab[1]))
	var source := TarotEpisode.open(show, seed)
	TarotEpisode.root = ROOT
	var ep := TarotEpisode.open(show, seed)
	DirAccess.make_dir_recursive_absolute(ep.dir)
	for f in DirAccess.get_files_at(source.dir):
		DirAccess.copy_absolute(source.dir.path_join(f), ep.dir.path_join(f))
	var table: Variant = ep.read_json("table")
	var spec: Variant = JSON.parse_string(FileAccess.get_file_as_string(spec_path)) if not spec_path.is_empty() else []
	var fx: Variant = (spec as Dictionary).get("effects", []) if spec is Dictionary else spec
	var t_dict: Dictionary = table if table is Dictionary else {"things": []}
	t_dict["effects"] = fx if fx is Array else []
	ep.write_json("table", t_dict)
	var script := ep.script()
	if script.is_empty():
		print("air_look_probe: %s #%d has no script" % [show, seed])
		get_tree().quit(2)
		return
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.own_world_3d = true
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	Director.detach()
	var medium: Medium = Medium.make("tarot")
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)
	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	var parse := TarotScript.parse(script)
	subs.words = preload("res://tests/tarot_look_probe.gd").timeline(parse, 0.33, Director.intro_hold)
	subs.document = {"source": script, "title": "Truthful Tarot", "tarot": ep.document()}
	add_child(subs)
	medium.bind_captions(subs)
	var tm := medium as TarotMedium
	var end := float((subs.words.back() as Dictionary)["t1"]) + 4.0
	# one step, so the table and its air are built and the schedule placed
	var t := 0.0
	Spectrum.virtual_clock = 0.0
	Spectrum.current.time = t
	subs._process(DT)
	medium.advance(Spectrum.current, DT, 1.0)
	print("air_look_probe: %d effects built; fog %s; %.0f s of reading" % [(t_dict["effects"] as Array).size(),
		str(tm._env.volumetric_fog_enabled), end])
	if moments:
		var mo: Dictionary = tm._air_moments()
		for name in mo:
			for m in mo[name]:
				var t0 := float((m as Dictionary)["t"])
				var dur := float((m as Dictionary).get("dur", 0.0))
				print("air_look_probe: moment %-9s at %6.1f s for %.2f s" % [name, t0, dur])
				for d in [0.12, 0.45, 0.9, 1.5]:
					times.append(t0 + d + (dur * 0.5 if dur > 0.5 else 0.0))
	if clip.y > clip.x:
		var tc := clip.x
		while tc <= clip.y:
			times.append(tc)
			tc += DT
	times.sort()
	var frame := 0
	var step := 0
	for want in times:
		while t < float(want):
			t += DT
			Spectrum.virtual_clock = t
			Spectrum.current.time = t
			subs._process(DT)
			medium.advance(Spectrum.current, DT, 1.0)
			step += 1
			if step % 20 == 0 or float(want) - t < 0.2:
				await get_tree().process_frame
		for _i in 2:
			await get_tree().process_frame
		var img := stage.get_texture().get_image()
		if show_motes and tm._air != null:
			for pop in tm._air.motes:
				for m in (pop as Dictionary)["each"]:
					var at: Dictionary = Effects.mote_at(pop, m, float(want))
					var scr: Variant = Effects._screen(tm._air.stage, at["pos"])
					print("air_look_probe:   mote at %s bright %.2f presence %.2f screen %s hidden %s" % [
						str(at["pos"]), float(at["bright"]), float(at["presence"]), str(scr), str(Effects._hidden(tm._air.stage, at["pos"]))])
		var path := "%s_t%05.1f.png" % [out, float(want)]
		if clip.y > clip.x and float(want) >= clip.x and float(want) <= clip.y + 1e-4:
			path = "%s_c%04d.png" % [out, frame]
			frame += 1
		img.save_png(path)
		if clip.y <= clip.x or frame <= 1:
			print("air_look_probe: t=%.2f -> %s" % [want, path])
	Spectrum.virtual_clock = -1.0
	Director.hold(false)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(0)
