extends Node

## settings_check - that a setting written through the app's own API is actually on disk,
## including when the process is KILLED rather than quit.
##
## Run: tests/run_boot_probe.sh tests/settings_check.gd 90
##
## THE COMPLAINT: "Medium and some of the other options are not serializing correctly.
## This happens a lot." Every previous mechanism here saved on a debounce and flushed at
## exit, which covers a clean quit and nothing else - and the failure is silent, so the only
## way it ever surfaced was someone noticing a setting had reverted.
##
## Three things are checked, and the third is the one that used to be untrue:
##   1. a write reaches the file at all, through the ordinary debounce;
##   2. a value written and immediately re-read comes back, before any flush;
##   3. a write survives with NO clean shutdown - Settings.MAX_DIRTY_MS bounds how long a
##      change may sit unwritten, so a kill can cost seconds and not a session.
## And that a render subprocess CANNOT write, which is the other half of settings going
## missing: two ghosts against one file, last one out wins.

var _fails: Array = []
var _restore := {}


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	# A probe is read-only by default (see Settings._ready) precisely so gates cannot edit
	# the config of whoever runs them. This gate is the exception, because persistence is
	# the thing under test - and it still puts back everything it touches.
	Settings.allow_writes_for_test()
	# Everything this touches is put back at the end - it is the user's real config.
	for k in ["pacing", "flourish", "camera", "medium"]:
		_restore[k] = Settings.read("director", k, null)

	# 1. read-back before any disk write
	Settings.write("director", "pacing", 3.14)
	if not is_equal_approx(float(Settings.read("director", "pacing", 0.0)), 3.14):
		_fail("a value written is not readable until it is flushed")

	# 2. the ordinary debounce reaches the disk
	Settings.flush()
	if not is_equal_approx(_on_disk("pacing", 0.0), 3.14):
		_fail("flush() did not reach the file")

	# 3. THE KILL CASE. Write, never flush, and wait out MAX_DIRTY_MS: the periodic write
	# must land on its own. Before Settings this was simply lost - every mechanism waited
	# for quiet or for exit, and a drag that never goes quiet in a process that is killed
	# reaches neither.
	Settings.write("director", "flourish", 2.5)
	var t0 := Time.get_ticks_msec()
	while Time.get_ticks_msec() - t0 < Settings.MAX_DIRTY_MS + 600:
		await get_tree().process_frame
	if not is_equal_approx(_on_disk("flourish", 0.0), 2.5):
		_fail("a change sat unwritten past MAX_DIRTY_MS - a kill would lose it")
	else:
		print("settings_check: an unflushed change reached disk within %d ms" % (Settings.MAX_DIRTY_MS + 600))

	# 4. every setting the Director owns round-trips through its own setter
	Director.set_medium("comic")
	Settings.flush()
	if String(_on_disk_str("medium", "")) != "comic":
		_fail("Director.set_medium did not persist")
	Director.set_pacing(2.25)
	Settings.flush()
	if not is_equal_approx(_on_disk("pacing", 0.0), 2.25):
		_fail("Director.set_pacing did not persist")
	Director.set_camera(1.75)
	Settings.flush()
	if not is_equal_approx(_on_disk("camera", 0.0), 1.75):
		_fail("Director.set_camera did not persist")

	# 5. A CONTAINER SETTING IS NOT SHARED WITH THE CONFIG, in either direction.
	#
	# Reported as "I keep restarting and losing my settings. Sometimes filters are toggled off,
	# sometimes their values are reset, it's not consistent between restarts" - and the
	# inconsistency is the tell. A Dictionary is a REFERENCE in GDScript and ConfigFile keeps
	# the reference it is given, so a live dictionary the app goes on mutating IS the config's
	# copy; `write`'s no-op guard then compares it with itself, finds no change, and never marks
	# the file dirty. The setting reaches the disk only when some OTHER write happens to carry
	# the file there in the same breath, which is why it survived some restarts and not others.
	#
	# Everything below fails on that build and passes on this one, which is what makes it a
	# check rather than a demonstration.
	_restore["filters"] = Settings.read("director", "filters", null)
	var live := {"monochrome": 1.0}
	Settings.write("director", "filters", live)
	Settings.flush()
	# (a) the config does not alias what it was handed
	live["static"] = 0.4
	var stored: Variant = Settings.read("director", "filters", {})
	if (stored as Dictionary).has("static"):
		_fail("the config aliases the dictionary it was given - a caller mutating its own "
			+ "copy is editing the settings through a side door")
	# (b) ...so writing the mutated one is seen as a CHANGE and reaches the disk
	Settings.write("director", "filters", live)
	Settings.flush()
	var on_disk: Dictionary = _on_disk_dict("filters")
	if not on_disk.has("static"):
		_fail("a mutated dictionary written again never reached the disk - the no-op guard "
			+ "is comparing the stored value with itself")
	elif not is_equal_approx(float(on_disk.get("monochrome", 0.0)), 1.0):
		_fail("the dictionary that reached the disk lost a key it had: %s" % str(on_disk))
	else:
		print("settings_check: a filter set survives being mutated and rewritten (%s)"
			% str(on_disk))
	# (c) nor does READ hand out the config's own object
	var got: Dictionary = Settings.read("director", "filters", {})
	got["bloom"] = 0.9
	if (Settings.read("director", "filters", {}) as Dictionary).has("bloom"):
		_fail("read() handed back the config's own dictionary - mutating what you read "
			+ "edits the settings")
	# (d) and the guard still works, or a drag writes the file every frame
	var before_dirty := Settings._dirty
	Settings.flush()
	Settings.write("director", "filters", Settings.read("director", "filters", {}))
	if Settings._dirty:
		_fail("writing an unchanged dictionary marked the config dirty - the no-op guard is "
			+ "gone and a slider drag will hit the disk every frame")
	# (e) NESTED containers are copied too: a voice slot list is an Array of Dictionaries.
	var rows := [{"pace": 1.0}]
	Settings.write("generative", "__probe_slots", rows)
	(rows[0] as Dictionary)["pace"] = 2.0
	var back: Array = Settings.read("generative", "__probe_slots", [])
	if not is_equal_approx(float((back[0] as Dictionary).get("pace", 0.0)), 1.0):
		_fail("the copy was shallow - the rows inside an array are still shared")
	Settings._cfg.erase_section_key("generative", "__probe_slots")

	# 6. nothing but Settings owns the file
	var writers := _writers()
	if writers > 0:
		_fail("%d script(s) still open the config themselves" % writers)

	for k in _restore:
		if _restore[k] != null:
			Settings.write("director", k, _restore[k])
	Settings.flush()
	print("settings_check: restored %s" % _restore)

	if _fails.is_empty():
		print("settings_check: ALL OK")
	else:
		for f in _fails:
			print("settings_check: FAILED - %s" % f)
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(_fails.size())


func _on_disk_dict(key: String) -> Dictionary:
	var cfg := ConfigFile.new()
	if cfg.load(Settings.PATH) != OK:
		return {}
	var v: Variant = cfg.get_value("director", key, {})
	return v as Dictionary if v is Dictionary else {}


## Read the file FRESH off the disk - never Settings' own copy, which would pass whether
## anything was written or not.
func _on_disk(key: String, dflt: float) -> float:
	var cfg := ConfigFile.new()
	if cfg.load(Settings.PATH) != OK:
		return dflt
	return float(cfg.get_value("director", key, dflt))


func _on_disk_str(key: String, dflt: String) -> String:
	var cfg := ConfigFile.new()
	if cfg.load(Settings.PATH) != OK:
		return dflt
	return String(cfg.get_value("director", key, dflt))


## How many scripts still open the config themselves. The same rule docs.py enforces, kept
## here too so a run of the gates catches it without regenerating the documentation.
func _writers() -> int:
	var n := 0
	for f in _gd_files("res://scripts"):
		if f.get_file() == "settings.gd" or f.get_file().contains("mask"):
			continue
		var t := FileAccess.get_file_as_string(f)
		# The PATH must be matched as a string LITERAL, quotes and all - half these files
		# name the file in a doc comment explaining who owns it, and a check that cannot
		# tell prose from code fails on its own documentation.
		if t.contains("ConfigFile.new()") or t.contains("\"user://ghost.cfg\""):
			print("settings_check:   direct writer: %s" % f)
			n += 1
	return n


func _gd_files(dir: String) -> Array:
	var out: Array = []
	var d := DirAccess.open(dir)
	if d == null:
		return out
	for f in d.get_files():
		if f.ends_with(".gd"):
			out.append(dir + "/" + f)
	for sub in d.get_directories():
		out.append_array(_gd_files(dir + "/" + sub))
	return out


func _fail(msg: String) -> void:
	_fails.append(msg)
