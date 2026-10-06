extends Node

## NOT A GATE - it spends quota. Sets one episode's table with a REAL writer working through its
## tools ([SetDresserTools]), the way the Tarot panel's "Set the table again" does - on a COPY of the
## episode (its plan, shuffle and pictures) in a show folder of its own, so the author's episodes are
## never touched - and prints what the set dresser did, call by call.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/set_dresser_run_probe.gd 1600 \
##       --spec ../../../rift/tarot/truthful-tarot.md --seed 352029 [--model haiku]
##
## Its pictures and its log are in the copy's `jobs/table/` (`look_NN.jpg`, `tools.jsonl`), and the
## table it handed in in `table.json` beside them; `set_dresser_look_probe --show <copy's show>` looks
## at it afterwards.

const ROOT := "user://set_dresser_run"
const COPIED := ["plan.json", "draw.json", "surface.png", "backdrop.png", "backdrop.json", "back.png"]


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var spec_path := _arg(args, "--spec")
	var seed := int(_arg(args, "--seed", "0"))
	var model := _arg(args, "--model", "haiku")
	if not FileAccess.file_exists(spec_path):
		print("set_dresser_run_probe: --spec <show.md> is required")
		get_tree().quit(2)
		return
	var raw := FileAccess.get_file_as_string(spec_path)
	var block: Dictionary = FrontMatter.read_block(raw)["data"]
	var knobs: Dictionary = block.get("tarot", {}) if block.get("tarot") is Dictionary else {}
	var title := BookLayout.field_of(raw, "title")
	var show := String(knobs.get("show", "")) if not String(knobs.get("show", "")).is_empty() else TarotEpisode.slug(title)
	var source := TarotEpisode.open(show, seed)
	TarotEpisode.root = ROOT
	var ep := TarotEpisode.open(show, seed)
	# the copy is made again each run: a table left from the last would be a step already made
	DirAccess.make_dir_recursive_absolute(ep.dir)
	DirAccess.remove_absolute(ep.file_of("table"))
	for f in COPIED:
		if FileAccess.file_exists(source.dir.path_join(String(f))):
			DirAccess.copy_absolute(source.dir.path_join(String(f)), ep.dir.path_join(String(f)))
	if not ep.has("plan") or not ep.has("image:surface"):
		print("set_dresser_run_probe: %s #%d has no plan or no cloth to set a table on" % [show, seed])
		get_tree().quit(2)
		return
	AgentJobs.allow_for_tool()
	var body := Manuscript.strip_frontmatter(raw)
	var prod := TarotProducer.new(ep, {"title": title, "brief": TarotDeck.strip(body), "deck": TarotDeck.of(body),
		"writer": "claude", "writer_model": model})
	print("set_dresser_run_probe: %s #%d (a copy, %s) with %s" % [show, seed, ep.dir, model])
	var t0 := Time.get_ticks_msec()
	prod.start(["table"])
	var seen := 0
	var log := ep.job_dir("table").path_join("tools.jsonl")
	while (prod.running or prod.busy()) and Time.get_ticks_msec() - t0 < 1590000:
		AgentJobs.pump()
		prod.tick()
		var lines := FileAccess.get_file_as_string(log).strip_edges().split("\n", false)
		while seen < lines.size():
			var e: Variant = JSON.parse_string(String(lines[seen]))
			if e is Dictionary:
				var d: Dictionary = e
				print("  %6.1fs  %-7s%s %s%s" % [float(Time.get_ticks_msec() - t0) / 1000.0, String(d["tool"]),
					" ERROR" if bool(d.get("error", false)) else "", String(d["text"]).get_slice("\n", 0).substr(0, 110),
					("  [%s]" % ", ".join(PackedStringArray(d.get("images", [])))) if not (d.get("images", []) as Array).is_empty() else ""])
			seen += 1
		await get_tree().process_frame
	var table: Variant = ep.read_json("table")
	print("set_dresser_run_probe: %s after %.0f s - %s" % ["SET" if table is Dictionary else "NOT SET",
		float(Time.get_ticks_msec() - t0) / 1000.0, prod.error_of("table") if prod.error_of("table") != "" else "no error"])
	if table is Dictionary:
		for th in (table as Dictionary).get("things", []):
			print("  - %s (%s)" % [String((th as Dictionary).get("name", "")), String((th as Dictionary).get("place", ""))])
	_usage(ep.job_dir("table").path_join("reply.jsonl"))
	get_tree().quit(0 if table is Dictionary else 1)


## The run's turns, time and tokens, from its stream's result.
static func _usage(path: String) -> void:
	for line in FileAccess.get_file_as_string(path).split("\n"):
		var d: Variant = JSON.parse_string(String(line)) if String(line).begins_with("{") else null
		if d is Dictionary and String((d as Dictionary).get("type", "")) == "result":
			var u: Dictionary = (d as Dictionary).get("usage", {})
			print("  %d turns, %.0f s, $%.3f; input %d + cache %d/%d, output %d" % [int(d.get("num_turns", 0)),
				float(d.get("duration_ms", 0)) / 1000.0, float(d.get("total_cost_usd", 0.0)), int(u.get("input_tokens", 0)),
				int(u.get("cache_creation_input_tokens", 0)), int(u.get("cache_read_input_tokens", 0)), int(u.get("output_tokens", 0))])


static func _arg(args: PackedStringArray, flag: String, dflt := "") -> String:
	var i := args.find(flag)
	return args[i + 1] if i >= 0 and i + 1 < args.size() else dflt
