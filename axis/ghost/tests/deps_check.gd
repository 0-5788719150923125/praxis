extends SceneTree

## Is the dependency table still true, and does resolution still behave?
##
## The panel this backs exists to make a new machine's first run explainable, and
## it can fail in two ways that are invisible on the author's own machine - which
## has everything installed and runs Linux only.
##
##   THE TABLE GOES STALE. Someone adds a call site that shells out to a new
##   program and does not add a row for it, so the Environment panel reports "all
##   present" on a machine that is missing the thing that just broke. So this gate
##   READS THE SOURCE: every bare program name handed to [Subprocess] or
##   [method Deps.execute] anywhere in `scripts/` must appear in [constant
##   Deps.TOOLS]. That is the same drift discipline `docs.py` applies to scenes and
##   CLI flags, for the same reason - a map nobody can forget to update.
##
##   THE HINTS GO WRONG. `install` is keyed by platform, and the author cannot run
##   Windows or macOS, so a row that ships with only a Linux hint is a row that
##   tells a Mac user nothing at the exact moment they need telling. Every row must
##   carry a hint for every platform it claims to apply to.
##
## Plus the resolver's own invariants, which are cheap and which nothing else
## covers: a miss stays a miss, an absolute path passes through, and `venv_bin`
## puts things where this platform actually puts them.
##
##   godot --headless --path axis/ghost --script tests/deps_check.gd

## Programs every platform provides as part of itself. They are excluded from the
## drift scan because a row telling someone to install `sleep` would be absurd -
## and because a machine without them has bigger problems than ghost.
const ASSUMED := ["sleep", "true", "false", "sh", "bash", "cmd", "env"]

var _fails: Array = []


func _init() -> void:
	_table_shape()
	_platform_hints()
	_source_drift()
	_resolver()
	_reporting()
	if _fails.is_empty():
		print("deps_check: ALL OK")
		quit(0)
	else:
		print("deps_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
		quit(1)


func _check(ok: bool, msg: String) -> void:
	print(("   ok   " if ok else "   FAIL ") + msg)
	if not ok:
		_fails.append(msg)


# --- the table ---------------------------------------------------------------

func _table_shape() -> void:
	print("-- table shape")
	var keys := {}
	for t in Deps.TOOLS:
		var key := String(t.get("key", ""))
		_check(not key.is_empty(), "every tool row has a key")
		_check(not keys.has(key), "key '%s' is unique" % key)
		keys[key] = true
		_check(not String(t.get("name", "")).is_empty(), "%s has a display name" % key)
		# The description IS the panel's tooltip and the report's explanation. A row
		# without one is a row that tells the reader a name and nothing else.
		_check(String(t.get("used_for", "")).length() > 30,
			"%s says what it is used for" % key)
		var has_probe := not Array(t.get("bins", [])).is_empty() \
			or not String(t.get("check", "")).is_empty()
		_check(has_probe, "%s is actually probeable (bins or check)" % key)
	for m in Deps.MANAGED:
		var key := String(m.get("key", ""))
		_check(not keys.has(key), "managed key '%s' does not collide with a tool" % key)
		keys[key] = true
		_check(String(m.get("used_for", "")).length() > 30,
			"%s says what it is used for" % key)
		_check(not String(m.get("check", "")).is_empty()
			or not String(m.get("path", "")).is_empty(),
			"%s knows where it lives" % key)


## THE ONE THE AUTHOR CANNOT CATCH BY RUNNING IT. ghost is developed on Linux; a
## row whose `install` map only has a "linux" key is silently useless to everyone
## else, and there is no way to notice that from here except by asserting it.
func _platform_hints() -> void:
	print("-- install hints cover every platform a row applies to")
	for t in Deps.TOOLS:
		var key := String(t.get("key", ""))
		var only: Array = t.get("platforms", [])
		var wanted: Array = only if not only.is_empty() else ["linux", "macos", "windows"]
		var install: Dictionary = t.get("install", {})
		for plat in wanted:
			_check(not String(install.get(plat, "")).strip_edges().is_empty(),
				"%s has an install hint for %s" % [key, plat])
		for plat in install.keys():
			_check(wanted.has(String(plat)),
				"%s's '%s' hint is for a platform it claims" % [key, plat])
		_check(String(t.get("site", "")).begins_with("http"),
			"%s links somewhere to download it" % key)


## Read every script and collect the bare program names actually spawned. A name
## with a slash in it is a path (a venv binary, `OS.get_executable_path()`) and is
## nobody's dependency to declare; a name behind a variable cannot be seen from
## here and is out of scope by construction - which is fine, because the point is
## to catch the LITERAL that someone added without a table row.
func _source_drift() -> void:
	print("-- every spawned program name has a table row")
	var declared := {}
	for t in Deps.TOOLS:
		for b in t.get("bins", []):
			declared[String(b)] = true
		for b in t.get("bins_windows", []):
			declared[String(b)] = true
	for a in ASSUMED:
		declared[a] = true

	var re := RegEx.new()
	re.compile("(?:Subprocess\\.start(?:_with_pipe|_detached)?|Deps\\.execute|OS\\.execute)"
		+ "\\(\\s*\"([^\"/\\\\]+)\"")
	var seen := {}
	for path in _scripts():
		var src := FileAccess.get_file_as_string(path)
		if src.is_empty():
			continue
		for m in re.search_all(src):
			var prog := m.get_string(1)
			if prog.is_empty() or prog.begins_with("-"):
				continue
			if not seen.has(prog):
				seen[prog] = []
			seen[prog].append(path.get_file())
	_check(not seen.is_empty(), "the scan found spawn sites at all (the regex still matches)")
	for prog in seen.keys():
		_check(declared.has(prog), "'%s' (spawned in %s) has a Deps.TOOLS row"
			% [prog, ", ".join(PackedStringArray(seen[prog]))])


func _scripts() -> PackedStringArray:
	var out := PackedStringArray()
	for base in ["res://scripts", "res://scripts/scenes"]:
		var d := DirAccess.open(base)
		if d == null:
			continue
		for f in d.get_files():
			if f.ends_with(".gd") and f != "deps.gd":
				out.append(base + "/" + f)
	return out


# --- the resolver ------------------------------------------------------------

func _resolver() -> void:
	print("-- resolution")
	var nope := "ghost-no-such-program-xyzzy"
	_check(Deps.resolve(nope).is_empty(), "a program that does not exist resolves to \"\"")
	_check(not Deps.has(nope), "and has() agrees")
	# Cached, so the second ask does not re-walk the whole search path. Same answer
	# is all that can be asserted from here, but a regression that broke the cache
	# key would change it.
	_check(Deps.resolve(nope).is_empty(), "and stays \"\" from the cache")
	_check(Deps.resolve("").is_empty(), "an empty name resolves to \"\"")

	var me := OS.get_executable_path()
	_check(Deps.resolve(me) == me, "an absolute path that exists passes straight through")
	_check(Deps.resolve(me + ".nope").is_empty(), "an absolute path that does not, does not")

	var dirs := Deps.search_dirs()
	_check(dirs.size() > 0, "there are directories to search")
	# PATH first: the user's own statement of intent must outrank our guesses, or a
	# deliberately-chosen ffmpeg loses to whatever is in /usr/bin.
	var path_env := OS.get_environment("PATH")
	if not path_env.is_empty():
		var first := String(path_env.split(":" if OS.get_name() != "Windows" else ";", false)[0])
		_check(dirs[0] == first, "PATH's first entry is searched first (%s)" % first)

	var vb := Deps.venv_bin("user://probe_venv", "python")
	if OS.get_name() == "Windows":
		_check(vb.contains("Scripts"), "a venv binary lands in Scripts\\ on Windows")
	else:
		_check(vb.ends_with("bin/python"), "a venv binary lands in bin/ elsewhere")
	_check(not vb.begins_with("user://"), "and comes back globalized, ready for the OS")

	_check(Deps.data_dir().to_lower().contains("ghost"),
		"the data directory is ghost's own (%s)" % Deps.data_dir())


# --- the report --------------------------------------------------------------

func _reporting() -> void:
	print("-- reporting")
	var rows := Deps.report()
	_check(rows.size() >= Deps.MANAGED.size(), "the report has rows")
	for r in rows:
		_check(r.has("found") and r.has("name"), "%s row is shaped" % String(r.get("key", "?")))
		# Nothing on a platform's exclusion list should be reported there at all -
		# a red `setpriv` on a Mac is a bug report waiting to happen.
		var only: Array = r.get("platforms", [])
		_check(only.is_empty() or only.has(Deps._platform()),
			"%s applies here" % String(r.get("key", "?")))

	var text := Deps.format_report(rows)
	for r in rows:
		_check(text.contains(String(r.get("name", "?"))),
			"the text report mentions %s" % String(r.get("name", "?")))

	# verdict() is what the panel's header and `--deps`'s exit code both read, so
	# it has to agree with the rows rather than be computed twice.
	var missing_feature := false
	for r in rows:
		if int(r.get("tier", Deps.TIER_EXTRA)) == Deps.TIER_FEATURE and not bool(r.get("found", false)):
			missing_feature = true
	_check(Deps.verdict(rows).is_empty() != missing_feature,
		"the verdict agrees with the rows")

	var snap := Deps.snapshot()
	_check(snap.has("host") and snap.has("tools"), "the feedback snapshot is shaped")
	_check(JSON.stringify(snap).length() > 0, "and it survives JSON encoding")
