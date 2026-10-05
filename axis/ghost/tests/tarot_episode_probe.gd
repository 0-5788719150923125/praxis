extends SceneTree

## Make (or finish) one tarot episode without the app: the same [TarotProducer], [AgentJobs] and
## episode directory the Tarot panel uses, driven from the command line.
##
##   godot --headless --path . --script res://tests/tarot_episode_probe.gd -- \
##       --spec ../../../rift/tarot/truthful-tarot.md --seed 2 [--only plan,draw] [--redo say:3]
##
## NOT A GATE: it spends real quota, writes into the author's own episodes
## (user://tarot/<show>/<seed>/) and takes minutes. It exists to make an episode headlessly - for
## a look probe to render, or to compare seeds - and it prints every step as it lands.
## `--only` stops once the named steps exist; `--redo` deletes a step (and what was made from it)
## first, as the panel's redo menu does.

func _init() -> void:
	var args := OS.get_cmdline_user_args()
	var spec_path := _arg(args, "--spec")
	if spec_path.is_empty() or not FileAccess.file_exists(spec_path):
		printerr("tarot_episode_probe: --spec <show.md> is required (got '%s')" % spec_path)
		quit(2)
		return
	var raw := FileAccess.get_file_as_string(spec_path)
	var block: Dictionary = FrontMatter.read_block(raw)["data"]
	var knobs: Dictionary = block.get("tarot", {}) if block.get("tarot") is Dictionary else {}
	var title := BookLayout.field_of(raw, "title")
	var show := String(knobs.get("show", ""))
	if show.is_empty():
		show = TarotEpisode.slug(title)
	# `--seed new` draws one the way the panel's New episode does
	var seed_arg := _arg(args, "--seed", str(knobs.get("seed", 1)))
	var seed := TarotDeck.true_seed() if seed_arg == "new" else int(seed_arg)
	AgentJobs.allow_for_tool()
	var ep := TarotEpisode.open(show, seed)
	var redo := _arg(args, "--redo")
	if not redo.is_empty():
		print("tarot_episode_probe: redo %s - removed %s" % [redo, str(ep.invalidate(redo))])
	var only := _arg(args, "--only")
	var want: Array = Array(only.split(",", false)) if not only.is_empty() else []
	var body := Manuscript.strip_frontmatter(raw)
	var prod := TarotProducer.new(ep, {"title": title, "brief": TarotDeck.strip(body), "deck": TarotDeck.of(body),
		"cards": knobs.get("cards", [3, 6]), "reversals": bool(knobs.get("reversals", true)),
		"jumpers": bool(knobs.get("jumpers", true)), "writer": String(knobs.get("writer", "claude")),
		"painter": String(knobs.get("painter", "codex"))})
	print("tarot_episode_probe: %s #%d -> %s" % [show, seed, ep.dir])
	var seen := {}
	var t0 := Time.get_ticks_msec()
	prod.start()
	while true:
		AgentJobs.pump()
		prod.tick()
		for s in ep.steps():
			var st := prod.state_of(String(s))
			if seen.get(s, "") != st:
				seen[s] = st
				var why := prod.error_of(String(s))
				print("  %6.1fs  %-16s %s%s" % [float(Time.get_ticks_msec() - t0) / 1000.0, s, st,
					("  - " + why) if not why.is_empty() else ""])
		if not want.is_empty():
			var all := true
			for w in want:
				all = all and ep.has(String(w))
			if all:
				prod.stop()
				break
		if not prod.running and not prod.busy():
			break
		OS.delay_msec(400)
	print("tarot_episode_probe: %s after %.0fs" % ["complete" if ep.complete() else "stopped",
		float(Time.get_ticks_msec() - t0) / 1000.0])
	quit(0 if ep.complete() or not want.is_empty() else 1)


func _arg(args: PackedStringArray, flag: String, dflt := "") -> String:
	var i := args.find(flag)
	return args[i + 1] if i >= 0 and i + 1 < args.size() else dflt
