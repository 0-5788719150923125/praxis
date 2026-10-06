extends SceneTree

## The set dresser's tools ([SetDresserTools]) and how the producer works them - with no agent, no
## model and no renderer.
##
##   godot --headless --path . --script res://tests/set_dresser_tools_check.gd
##
## - A RUN STARTS CLEAN: what a last run handed in, logged or photographed is gone; its prompt stays.
## - PUT says what the builder makes of a thing: its real size and flames, parts it cannot build and
##   why, materials it does not know, more parts or deeper groups than a thing may have, and a thing
##   taller than its place can show. A thing put again under its name REPLACES it. The tally counts
##   lit things against the look's ask.
## - REMOVE, LOOK and SET answer by name and say what is on the table when a name is wrong; with no
##   renderer, LOOK and SET say there is no picture and take none.
## - SUBMIT refuses an empty table, writes the draft exactly as written, and - where a table can be
##   stood - sends the set dresser to look first, once.
## - THE PRODUCER gives a writer that takes tools the toolset (a URL, a long timeout, the working
##   prompt) and one that does not the one-reply prompt; it lands a handed-in table whatever the run's
##   last words, says so when a run with tools handed nothing in, and closes the tools when it stops.
## - CLAUDE WITH TOOLS loads no settings instead of `--safe-mode` (which drops every MCP server) and is
##   pointed at the toolset by a config file; without tools its argv is as it was.
## - THE AIR: effects put beside things are reported - what was built, and every kind, place, look or
##   moment the builder does not know, a size kept within its look's range - replaced and removed by
##   name, handed in with the table, and watched by name (with no renderer, said so).

const ROOT := "user://set_dresser_check"

var _fails := 0


## A toolset whose table can be stood (to reach the submit rule) but which takes no pictures.
class Standing:
	extends SetDresserTools

	func _can_stand() -> bool:
		return true

	func _can_see() -> bool:
		return false


## A producer that records the job it would have submitted instead of starting it.
class Spy:
	extends TarotProducer

	var sent := {}

	func _submit_text(step: String, p: Dictionary, tier: String, extra: Dictionary = {}) -> void:
		sent = {"step": step, "prompt": p, "tier": tier, "extra": extra}


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails += 1
		print("  FAIL: " + what)


func _run() -> void:
	AgentJobs.allow_for_tool()
	for check in [_fresh_start, _tools_listed, _put_reports, _remove_look_set, _submit, _producer, _prompts, _claude_argv, _air]:
		var done: Variant = await (check as Callable).call()
		_ok(done == true, "%s stopped part way (a script error - see above)" % (check as Callable).get_method())
	print("set_dresser_tools_check: %s (%d failure%s)" % ["ALL OK" if _fails == 0 else "FAILED", _fails, "" if _fails == 1 else "s"])
	quit(1 if _fails > 0 else 0)


func _episode() -> TarotEpisode:
	TarotEpisode.root = ROOT
	var ep := TarotEpisode.open("check-show", 5150)
	if DirAccess.dir_exists_absolute(ep.dir):
		for f in DirAccess.get_files_at(ep.dir):
			DirAccess.remove_absolute(ep.dir.path_join(f))
	ep.write_json("plan", {"episode_title": "A Table Set On Purpose", "audience": "everyone", "topic": "tables",
		"premise": "the premise", "reader_mood": "calm", "spread": {"name": "Three", "positions": [{"name": "One"},
		{"name": "Two"}, {"name": "Three"}]}, "look": TarotTable.sanitize_look({"deck_name": "The Test Deck", "candles": 2})})
	var cards: Array = []
	for c in TarotDeck.shuffled(TarotDeck.standard(), ep.seed, true).slice(0, 3):
		cards.append(c)
	ep.write_json("draw", {"seed": ep.seed, "cards": cards})
	return ep


func _tools(ep: TarotEpisode) -> SetDresserTools:
	return SetDresserTools.new(ep, ep.read_json("plan"), ep.job_dir("table"))


func _candle(name: String, h: float, place := "back left") -> Dictionary:
	return {"name": name, "why": "light", "place": place, "group": "c", "parts": [{"shape": "lathe",
		"profile": [[0, 0], [2, 0], [2, h], [0, h]], "material": "beeswax", "wick": true}]}


func _fresh_start() -> bool:
	var ep := _episode()
	var jd := ep.job_dir("table")
	DirAccess.make_dir_recursive_absolute(jd)
	for f in [SetDresserTools.SUBMITTED, "tools.jsonl", "look_01.jpg", "prompt.txt"]:
		TextGen.put(jd.path_join(String(f)), "old")
	var t := _tools(ep)
	_ok(not FileAccess.file_exists(jd.path_join(SetDresserTools.SUBMITTED)) and not FileAccess.file_exists(jd.path_join("tools.jsonl"))
		and not FileAccess.file_exists(jd.path_join("look_01.jpg")), "a new run kept the last run's table, log or pictures")
	_ok(FileAccess.file_exists(jd.path_join("prompt.txt")), "a new run took more than the last run's own output")
	t.release()
	return true


func _tools_listed() -> bool:
	var t := _tools(_episode())
	var names: Array = []
	var shaped := true
	for d in t.list_tools():
		names.append(String((d as Dictionary)["name"]))
		shaped = shaped and String(((d as Dictionary)["inputSchema"] as Dictionary).get("type", "")) == "object" \
			and not String((d as Dictionary).get("description", "")).is_empty()
	_ok(names == ["put", "remove", "look", "set", "watch", "submit"], "the set dresser's tools are %s" % str(names))
	_ok(shaped, "a tool has no description or no object schema")
	var r: Dictionary = await t.call_tool("paint", {})
	_ok(bool(r.get("error", false)) and String(r["text"]).contains("put, remove, look, set, watch and submit"),
		"an unknown tool did not name the real ones: %s" % r)
	t.release()
	return true


func _put_reports() -> bool:
	var ep := _episode()
	var t := _tools(ep)
	var r: Dictionary = await t.call_tool("put", {"things": [{"name": "a hat", "place": "left", "parts": [{"shape": "hat"}]}]})
	var text := String(r["text"])
	_ok(text.contains("a hat") and text.contains("NOTHING BUILT") and text.contains("\"hat\" is not a shape"),
		"a thing with no buildable part was not reported as such:\n%s" % text)
	r = await t.call_tool("put", {"things": [_candle("a pillar", 9.0)]})
	text = String(r["text"])
	_ok(text.contains("\"beeswax\" is not one of the table's materials"), "an unknown material was not reported:\n%s" % text)
	_ok(text.contains("4 x 9 x 4 cm") and text.contains("1 flame"), "a candle's size and flame were not reported:\n%s" % text)
	_ok(text.contains("No pictures in this run"), "a run with no renderer did not say it took no picture:\n%s" % text)
	_ok((r.get("images", []) as Array).is_empty(), "a run with no renderer returned a picture")
	r = await t.call_tool("put", {"materials": {"beeswax": {"kind": "wax", "color": "#e8d9a8"}, "odd": {"kind": "cheese", "color": "pale", "play": "sparkle"}}})
	text = String(r["text"])
	_ok(text.contains("\"cheese\" is not a kind") and text.contains("its color is not #rrggbb") and text.contains("\"sparkle\" is not a play"),
		"a material's faults were not reported:\n%s" % text)
	r = await t.call_tool("put", {"things": [_candle("a pillar", 12.0), _candle("a taper", 7.0, "back right")]})
	text = String(r["text"])
	_ok(not text.contains("not one of the table's materials"), "a material put after its thing was still unknown:\n%s" % text)
	_ok((t.draft()["things"] as Array).size() == 3 and text.contains("the table now holds 3") and text.contains("4 x 12 x 4 cm"),
		"a thing put again under its name was not replaced (%d things):\n%s" % [(t.draft()["things"] as Array).size(), text])
	_ok(text.contains("2 lit things (2 asked for)"), "the tally does not count lit things against the ask:\n%s" % text)
	var room := int(TarotTable.headroom(ep.seed)["back"])
	r = await t.call_tool("put", {"things": [{"name": "a tall jar", "place": "back", "parts": [{"shape": "box",
		"size": [6, room + 10, 6], "material": "beeswax"}]}]})
	_ok(String(r["text"]).contains("\"back\" shows only %d cm" % room), "a thing taller than its place was not reported:\n%s" % r["text"])
	var many: Array = []
	for i in 20:
		many.append({"shape": "ball", "radius": 1, "at": [i * 2, 0, 0], "material": "beeswax"})
	var deep := {"parts": [{"parts": [{"parts": [{"parts": [{"shape": "ball", "radius": 1}]}]}]}]}
	r = await t.call_tool("put", {"things": [{"name": "beads", "parts": many}, {"name": "nest", "parts": [deep, {"shape": "ball", "radius": 2}]}]})
	text = String(r["text"])
	_ok(text.contains("past the %d parts" % Props.MAX_PARTS), "more parts than a thing may have were not reported:\n%s" % text)
	r = await t.call_tool("put", {"things": [{"name": "a stub", "parts": [{"shape": "lathe", "profile": [[1, 1]]},
		{"shape": "tube", "path": [[0, 0, 0]]}]}]})
	_ok(String(r["text"]).contains("a lathe's profile needs two") and String(r["text"]).contains("a tube's path needs two"),
		"a lathe or a tube too short to build was not reported:\n%s" % r["text"])
	_ok(text.contains("more than %d groups deep" % Props.MAX_DEPTH), "a group nested too deep was not reported:\n%s" % text)
	var copies := {"name": "tea lights", "parts": [{"shape": "lathe", "profile": [[0, 0], [2, 0], [2, 1.5], [0, 1.5]], "material": "beeswax",
		"wick": true, "copies": {"ring": {"count": 3, "radius": 4}}}]}
	r = await t.call_tool("put", {"things": [copies]})
	_ok(String(r["text"]).contains("3 flames") and String(r["text"]).contains("3 lit things"), "copied candles were not counted:\n%s" % r["text"])
	r = await t.call_tool("put", {"things": "a lamp"})
	_ok(bool(r.get("error", false)), "a put of nothing usable was not refused")
	t.release()
	return true


func _remove_look_set() -> bool:
	var t := _tools(_episode())
	await t.call_tool("put", {"things": [_candle("a pillar", 9.0), _candle("a taper", 14.0)], "materials": {"beeswax": {"kind": "wax", "color": "#e8d9a8"}}})
	var r: Dictionary = await t.call_tool("remove", {"names": ["a lamp"]})
	_ok(bool(r.get("error", false)) and String(r["text"]).contains("On it: a pillar, a taper"), "removing a thing not there was not refused with what is: %s" % r)
	r = await t.call_tool("remove", {"names": ["a taper"]})
	_ok(not bool(r.get("error", false)) and (t.draft()["things"] as Array).size() == 1, "removing a thing did not take it off: %s" % r)
	r = await t.call_tool("look", {"name": "a lamp"})
	_ok(bool(r.get("error", false)) and String(r["text"]).contains("On it: a pillar"), "looking at a thing not there was not refused with what is: %s" % r)
	r = await t.call_tool("look", {"name": "a pillar"})
	_ok(not bool(r.get("error", false)) and String(r["text"]).contains("a pillar") and (r.get("images", []) as Array).is_empty(),
		"looking with no renderer did not describe the thing without a picture: %s" % r)
	r = await t.call_tool("set", {})
	_ok(not bool(r.get("error", false)) and String(r["text"]).contains("cannot be stood in this run") and (r.get("images", []) as Array).is_empty(),
		"setting with no renderer did not say so: %s" % r)
	t.release()
	return true


func _submit() -> bool:
	var ep := _episode()
	var t := _tools(ep)
	var r: Dictionary = await t.call_tool("submit", {})
	_ok(bool(r.get("error", false)) and not t.submitted, "an empty table was handed in")
	await t.call_tool("put", {"idea": "a calm table", "things": [_candle("a pillar", 9.0)], "materials": {"beeswax": {"kind": "wax", "color": "#e8d9a8"}}})
	r = await t.call_tool("submit", {})
	var path := ep.job_dir("table").path_join(SetDresserTools.SUBMITTED)
	var j := JSON.new()
	var same := FileAccess.file_exists(path) and j.parse(FileAccess.get_file_as_string(path)) == OK \
		and JSON.stringify(j.data) == JSON.stringify(JSON.parse_string(JSON.stringify(t.draft())))
	_ok(not bool(r.get("error", false)) and t.submitted and same and String(r["text"]).begins_with("Handed in: 1 thing"),
		"a table was not handed in exactly as drafted: %s" % r)
	t.release()
	# WHERE A TABLE CAN BE STOOD, the set dresser looks before it hands in - asked once, then trusted
	var s := Standing.new(ep, ep.read_json("plan"), ep.job_dir("table"))
	await s.call_tool("put", {"things": [_candle("a pillar", 9.0)], "materials": {"beeswax": {"kind": "wax", "color": "#e8d9a8"}}})
	r = await s.call_tool("submit", {})
	_ok(bool(r.get("error", false)) and String(r["text"]).contains("call set") and not s.submitted, "a table never seen was handed in without a word")
	r = await s.call_tool("submit", {})
	_ok(not bool(r.get("error", false)) and s.submitted, "a second submit, unchanged, was refused: %s" % r)
	s.release()
	return true


func _producer() -> bool:
	var ep := _episode()
	TextGen.put(ep.file_of("image:surface"), "png")
	# WITH TOOLS: the job gets the toolset's URL and a long timeout, and the working prompt
	var spy := Spy.new(ep, {"title": "Test Tarot", "brief": "A brief.", "writer": "claude"})
	spy._make_table()
	var extra: Dictionary = spy.sent.get("extra", {})
	var prompt := String((spy.sent.get("prompt", {}) as Dictionary).get("prompt", ""))
	var url := String(extra.get("tools_url", ""))
	_ok(url.begins_with("http://127.0.0.1:") and int(extra.get("timeout", 0)) == SetDresserTools.TIMEOUT,
		"a writer that takes tools was not given them: %s" % extra)
	_ok(prompt.contains("HOW YOU WORK") and prompt.contains("submit") and not prompt.contains("Reply with ONLY a JSON object"),
		"a writer given tools was not told how to work with them")
	_ok(AgentTools.calls(url) == 0 and spy._tools.has("table"), "the toolset was not opened for the step")
	spy.stop()
	_ok(AgentTools.calls(url) == -1 and spy._tools.is_empty(), "stopping the producer left the tools answering")
	# WITHOUT: the one-reply prompt, no tools
	var plain := Spy.new(ep, {"title": "Test Tarot", "brief": "A brief.", "writer": "bedrock"})
	plain._make_table()
	_ok((plain.sent.get("extra", {}) as Dictionary).is_empty() and plain._tools.is_empty()
		and String((plain.sent["prompt"] as Dictionary)["prompt"]).contains("Reply with ONLY a JSON object"),
		"a writer that takes no tools was not asked for the table in one reply")
	# A HANDED-IN TABLE LANDS whatever the run's last words, even a run that ended badly
	var prod := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief."})
	var given := SetDresserTools.new(ep, ep.read_json("plan"), ep.job_dir("table"))
	prod._tools["table"] = {"url": AgentTools.open(ep.job_dir("table"), given), "set": given}
	TextGen.put(ep.job_dir("table").path_join(SetDresserTools.SUBMITTED), TarotPrompts.SET_EXAMPLE)
	prod._land("table", {"ok": false, "error": "timed out after 1500 s"})
	_ok(ep.has("table") and String(((ep.read_json("table") as Dictionary)["things"][0] as Dictionary)["name"]) == "a boxwood chess pawn",
		"a table handed in with a tool did not land")
	# ...but a run WITHOUT tools is answered by its own reply, never by a table some earlier run left
	ep.invalidate("table")
	TextGen.put(ep.job_dir("table").path_join(SetDresserTools.SUBMITTED), TarotPrompts.SET_EXAMPLE)
	var plain_land := TarotProducer.new(ep, {"title": "Test Tarot", "brief": "A brief."})
	plain_land._tries["table"] = TarotProducer.RETRIES + 1
	plain_land._land("table", {"ok": true, "text": "no table here"})
	_ok(not ep.has("table") and plain_land.error_of("table") == "the table was not JSON",
		"a run without tools landed a table an earlier run had handed in")
	plain._make_table()
	_ok(not FileAccess.file_exists(ep.job_dir("table").path_join(SetDresserTools.SUBMITTED)),
		"a new run of the table step kept an earlier run's handed-in table")
	# ...and a run with tools that handed nothing in says so
	DirAccess.remove_absolute(ep.job_dir("table").path_join(SetDresserTools.SUBMITTED))
	ep.invalidate("table")
	var tools := SetDresserTools.new(ep, ep.read_json("plan"), ep.job_dir("table"))
	prod._tools["table"] = {"url": AgentTools.open(ep.job_dir("table"), tools), "set": tools}
	prod._tries["table"] = TarotProducer.RETRIES + 1
	prod._land("table", {"ok": true, "text": "Done."})
	_ok(prod.error_of("table") == "the set dresser stopped without handing its table in" and prod._tools.is_empty(),
		"a run with tools that handed nothing in failed as '%s'" % prod.error_of("table"))
	return true


func _prompts() -> bool:
	var ep := _episode()
	var head := TarotTable.headroom(ep.seed)
	var size := TarotPrompts.table_size(ep.seed)
	_ok(TarotPrompts.set_dresser("T", "B", ep.read_json("plan"), ep.seed, head, [], true, 0, ["sea mist (fog)"])["prompt"].contains("EARLIER EPISODES' AIR was this. Give this table its own, or none: sea mist (fog)"),
		"the set dresser is not told what air earlier tables had")
	for looks in [0, SetDresserTools.LOOKS]:
		var p := TarotPrompts.set_dresser("Test Tarot", "A brief.", ep.read_json("plan"), ep.seed, head, [], true, looks)
		var text := String(p["prompt"])
		var tag := "with tools" if looks > 0 else "in one reply"
		_ok(text.contains("cannot make a BODY") and text.contains("it has no body"), "the set dresser %s is not told it builds no bodies" % tag)
		_ok(text.contains("Exactly 2 lit things -") and text.contains("and %d to %d other things" % [size.x, size.y]),
			"the set dresser %s is not told what the table holds" % tag)
		for k in Props.SHAPES:
			_ok(text.contains("- %s:" % k), "the set dresser %s is not told the shape %s" % [tag, k])
		for z in TarotTable.ZONES:
			_ok(text.contains("\"%s\": %s; up to %d cm" % [z, String((TarotTable.ZONES[z] as Dictionary)["about"]), int(head[z])]),
				"the set dresser %s is not told the zone %s" % [tag, z])
		_ok(text.contains(TarotPrompts.SET_EXAMPLE), "the set dresser %s is not shown the format" % tag)
		_ok(text.contains("THE AIR:"), "the set dresser %s is not told about the air" % tag)
		for k in Effects.KINDS.keys() + Effects.MOTES.keys() + Effects.BURSTS.keys():
			_ok(text.contains("- %s:" % k), "the set dresser %s is not told the effect word %s" % [tag, k])
		for k in TarotTable.AIR.keys() + TarotTable.MOMENTS.keys():
			_ok(text.contains("- \"%s\":" % k), "the set dresser %s is not told the place or moment %s" % [tag, k])
		_ok(text.contains("HOW YOU WORK") == (looks > 0) and text.contains("Reply with ONLY a JSON object") == (looks == 0),
			"the set dresser %s is told the other way of working" % tag)
		if looks > 0:
			_ok(text.contains("against the %d you have" % looks), "the set dresser is not told how many pictures it has")
	return true


func _claude_argv() -> bool:
	var dir := ProjectSettings.globalize_path(ROOT).path_join("claude")
	DirAccess.make_dir_recursive_absolute(dir)
	var job := {"dir": dir, "tier": "best", "tools_url": "http://127.0.0.1:4242/mcp/" + "ab".repeat(16)}
	var tools := TextGen.Claude.tool_args(job)
	var args := TextGen.Claude.argv(job, dir.path_join("system.txt"), tools)
	var j := JSON.new()
	var cfg_ok := j.parse(FileAccess.get_file_as_string(dir.path_join("mcp.json"))) == OK \
		and String(j.data["mcpServers"]["ghost"]["url"]) == String(job["tools_url"]) and String(j.data["mcpServers"]["ghost"]["type"]) == "http"
	_ok(cfg_ok, "the tools' config does not name the job's URL")
	var at := args.find("--setting-sources")
	_ok(at >= 0 and args[at + 1] == "" and not args.has("--safe-mode"), "a writer with tools still runs in safe mode (which drops ghost's tools): %s" % str(args))
	_ok(args.has("--strict-mcp-config") and args[args.find("--mcp-config") + 1] == dir.path_join("mcp.json")
		and args[args.find("--allowedTools") + 1] == "mcp__ghost" and args[args.find("--tools") + 1] == "",
		"a writer with tools is not pointed at ghost's alone: %s" % str(args))
	for a in args:
		_ok(not String(a).contains("/mcp/"), "the tools' URL is in argv")
	var none := TextGen.Claude.argv({"dir": dir, "tier": "best"}, dir.path_join("system.txt"), TextGen.Claude.tool_args({"dir": dir}))
	_ok(none.has("--safe-mode") and not none.has("--setting-sources") and not none.has("--mcp-config") and none[none.find("--tools") + 1] == "",
		"a writer without tools is not run as before: %s" % str(none))
	_ok(TextGen.make("claude").takes_tools() and not TextGen.make("codex").takes_tools() and not TextGen.make("bedrock").takes_tools(),
		"the writers that take tools are not Claude alone")
	return true


func _air() -> bool:
	var ep := _episode()
	var t := _tools(ep)
	var r: Dictionary = await t.call_tool("put", {"effects": [
		{"name": "sea mist", "kind": "fog", "where": "beyond the table", "density": 0.5},
		{"name": "wisps", "kind": "motes", "look": "pixie", "where": "beyond the table", "count": 5, "size": 40},
		{"name": "sparks", "kind": "burst", "look": "sparks", "on": "jumper"},
		{"name": "rain", "kind": "weather"},
		{"name": "lost", "kind": "fog", "where": "the ceiling"},
		{"name": "odd", "kind": "burst", "look": "confetti", "on": "the end"},
		{"name": "flies", "kind": "motes", "look": "fly", "where": "over the cloth", "count": 3, "light": true}]})
	var text := String(r["text"])
	_ok(text.contains("sea mist: fog beyond the table") and text.contains("wisps: 5 pixies beyond the table") and text.contains("sparks: sparks on jumper"),
		"effects put were not reported as built:\n%s" % text)
	_ok(text.contains("a pixie is 1.5 to 8 mm"), "a mote's size outside its look's range was not reported:\n%s" % text)
	_ok(text.contains("rain: NOT BUILT") and text.contains("\"weather\" is not a kind of effect"), "an unknown kind was not reported:\n%s" % text)
	_ok(text.contains("lost: NOT BUILT") and text.contains("\"the ceiling\" is not a place in the air"), "an unknown place was not reported:\n%s" % text)
	_ok(text.contains("\"confetti\" is not a look of burst") and text.contains("\"the end\" is not a moment"), "an unknown look or moment was not reported:\n%s" % text)
	_ok(text.contains("flies: 3 flies over the cloth") and not text.contains("flies over the cloth, 5 mm, lit") and text.contains("a fly is a dark speck and carries no light"),
		"a fly was not reported unlit, or not counted as flies:\n%s" % text)
	r = await t.call_tool("put", {"effects": [{"name": "sea mist", "kind": "fog", "where": "low on the cloth", "density": 0.2}]})
	var fogs := 0
	for e in t.draft()["effects"]:
		fogs += 1 if String((e as Dictionary)["name"]) == "sea mist" else 0
	_ok(fogs == 1 and String(r["text"]).contains("sea mist: fog low on the cloth"), "an effect put again under its name was not replaced")
	r = await t.call_tool("remove", {"names": ["rain", "lost", "odd", "flies"]})
	_ok((t.draft()["effects"] as Array).size() == 3, "effects were not removed by name: %s" % str(t.draft()["effects"]))
	r = await t.call_tool("watch", {"name": "snow"})
	_ok(bool(r.get("error", false)) and String(r["text"]).contains("In it: sea mist, wisps, sparks"), "watching an effect not there was not refused with what is: %s" % r)
	r = await t.call_tool("watch", {"name": "sparks"})
	_ok(not bool(r.get("error", false)) and String(r["text"]).contains("cannot be watched") and (r.get("images", []) as Array).is_empty(),
		"watching with no renderer did not say so: %s" % r)
	await t.call_tool("put", {"things": [_candle("a pillar", 9.0)], "materials": {"beeswax": {"kind": "wax", "color": "#e8d9a8"}}})
	r = await t.call_tool("submit", {})
	var handed: Variant = JSON.parse_string(FileAccess.get_file_as_string(ep.job_dir("table").path_join(SetDresserTools.SUBMITTED)))
	_ok(handed is Dictionary and ((handed as Dictionary).get("effects", []) as Array).size() == 3, "the air was not handed in with the table")
	_ok((TarotTable.sanitize_table(handed, TarotTable.sanitize_look({}))["effects"] as Array).size() == 3, "the air handed in does not build")
	t.release()
	return true

