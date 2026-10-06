extends Node

## NOT a gate. What the set dresser SEES through its tools ([SetDresserTools]) - with no agent: a
## table description is put on an episode's table a few things at a time, one thing is looked at
## from four sides, and the table is set; every picture is written out, beside the words each tool
## answered with.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/set_dresser_look_probe.gd 300 \
##       --show truthful-tarot --seed 26551 [--spec <table.json>] [--air <effects.json>] [--out /tmp/sd/look]
##
## The description is the episode's own `table.json` unless `--spec` names another; `--air` puts a
## list of effects (or a table's `effects`) beside it, and every one of them is watched. Last, the
## name's color is chosen (`--ink`, else the description's own `title`, else cream) and the OPENING is
## shown under `--name`/`--byline`. The episode is only read: the tools work in a folder of their own
## (`user://set_dresser_look`), never its jobs.

var _out := "user://set_dresser_look/look"


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var show := "truthful-tarot"
	var seed := 0
	var spec_path := ""
	var air_path := ""
	var ink := ""
	var name := "Truthful Tarot"
	var byline := ""
	for i in args.size() - 1:
		match args[i]:
			"--air": air_path = args[i + 1]
			"--ink": ink = args[i + 1]
			"--name": name = args[i + 1]
			"--byline": byline = args[i + 1]
			"--show": show = args[i + 1]
			"--seed": seed = int(args[i + 1])
			"--spec": spec_path = args[i + 1]
			"--out": _out = args[i + 1]
	var ep := TarotEpisode.open(show, seed)
	if spec_path.is_empty():
		spec_path = ep.file_of("table")
	var table: Variant = JSON.parse_string(FileAccess.get_file_as_string(spec_path))
	var plan: Variant = ep.read_json("plan")
	if not (table is Dictionary) or not (plan is Dictionary):
		print("set_dresser_look_probe: no table at %s, or no plan for %s #%d" % [spec_path, show, seed])
		get_tree().quit(2)
		return
	var dir := ProjectSettings.globalize_path("user://set_dresser_look")
	DirAccess.make_dir_recursive_absolute(_out.get_base_dir() if _out.is_absolute_path() else ProjectSettings.globalize_path(_out).get_base_dir())
	var tools := SetDresserTools.new(ep, plan as Dictionary, dir, name, byline)
	print("set_dresser_look_probe: %s #%d, %d things; sees %s, stands %s" % [show, seed,
		((table as Dictionary).get("things", []) as Array).size(), TablePreview.can_see(), TablePreview.can_set()])
	var things: Array = (table as Dictionary).get("things", [])
	var t0 := Time.get_ticks_msec()
	# a few at a time, as the set dresser is told to put them
	var half := ceili(things.size() / 2.0)
	await _call(tools, "put", {"things": things.slice(0, half), "materials": (table as Dictionary).get("materials", {}),
		"idea": String((table as Dictionary).get("idea", ""))}, "put1")
	await _call(tools, "put", {"things": things.slice(half)}, "put2")
	var air: Variant = JSON.parse_string(FileAccess.get_file_as_string(air_path)) if not air_path.is_empty() else null
	var effects: Array = ((air as Dictionary).get("effects", []) if air is Dictionary else air) if air != null else []
	if not effects.is_empty():
		await _call(tools, "put", {"effects": effects}, "air")
		for e in effects:
			await _call(tools, "watch", {"name": String((e as Dictionary).get("name", ""))}, "watch_" + String((e as Dictionary).get("name", "")).replace(" ", "_"))
	if not things.is_empty():
		await _call(tools, "look", {"name": String((things[0] as Dictionary).get("name", ""))}, "look")
	await _call(tools, "set", {}, "set")
	if ink.is_empty():
		var own: Variant = (table as Dictionary).get("title", {})
		ink = String((own as Dictionary).get("color", TarotTable.TITLE_INK)) if own is Dictionary else TarotTable.TITLE_INK
	await _call(tools, "title", {"color": ink, "why": "the probe's"}, "title")
	print("set_dresser_look_probe: done in %.1f s" % [(Time.get_ticks_msec() - t0) / 1000.0])
	tools.release()
	get_tree().quit(0)


func _call(tools: SetDresserTools, name: String, args: Dictionary, tag: String) -> void:
	var t0 := Time.get_ticks_msec()
	var r: Dictionary = await tools.call_tool(name, args)
	print("\n== %s (%d ms)%s\n%s" % [name, Time.get_ticks_msec() - t0, "  ERROR" if bool(r.get("error", false)) else "", String(r.get("text", ""))])
	var i := 0
	for img in r.get("images", []):
		var path := "%s_%s%s.png" % [_out, tag, "" if i == 0 else str(i)]
		(img as Image).save_png(path)
		var flat := _uniform(img as Image)
		print("   picture: %s (%dx%d)%s" % [path, (img as Image).get_width(), (img as Image).get_height(), "  UNIFORM" if flat else ""])
		i += 1


## Whether a picture is one color (a render that drew nothing).
static func _uniform(img: Image) -> bool:
	var first := img.get_pixel(0, 0)
	for y in range(0, img.get_height(), 37):
		for x in range(0, img.get_width(), 41):
			if img.get_pixel(x, y).is_equal_approx(first) == false:
				return false
	return true
