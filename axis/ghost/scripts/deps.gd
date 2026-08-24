extends RefCounted
class_name Deps

## Deps - every external program ghost needs, in one place, with one promise:
## RESOLUTION AND REPORTING ARE THE SAME CODE.
##
## WHY THIS EXISTS. ghost shells out constantly - ffmpeg for every Masking clip,
## ffprobe for its frame counts, python3 for the voice host and the clown's face
## tracker and the YouTube import - and until this file each of those call sites
## resolved its own binary with its own copy of `OS.execute("which", ...)`. Four
## copies, all Linux-only (`which` is not a Windows program), all silent when they
## missed: the failure surfaced as "the clip won't open" or "voice host exited"
## with nothing pointing at the actual cause, which was that the machine had no
## ffmpeg on it.
##
## So there are two halves and they must not drift apart:
##
##   [method resolve] is what actually launches things. [Subprocess] runs every
##   child through it, so a bare "ffmpeg" handed to `Subprocess.start` becomes an
##   absolute path before the kernel ever sees it.
##
##   [method report] is what the home screen lists. It asks the SAME resolver, so
##   the panel cannot claim a dependency is present while the launch path fails to
##   find it. That is the whole point of putting them in one file - a status panel
##   with its own detection logic is a panel that lies eventually.
##
## THE PATH PROBLEM, which is the real bug under all of this. A GUI-launched app
## does not inherit an interactive shell's PATH. On macOS a double-clicked app gets
## `/usr/bin:/bin:/usr/sbin:/sbin` and nothing else, so a Homebrew ffmpeg in
## `/opt/homebrew/bin` is invisible - the single most likely way this project fails
## on a Mac. On Linux the same happens to `~/.local/bin` and Flatpak/snap exports.
## [method search_dirs] therefore scans PATH *plus* the places each platform
## actually installs things, and the scan is a filesystem lookup rather than a
## `which` subprocess: it costs microseconds, works identically on all three
## platforms, and needs no external program to find external programs.
##
## WINDOWS gets two extra rules. Executables need an extension, so every candidate
## is tried against `PATHEXT` (`.exe`, `.cmd`, `.bat`, ...) - a `.cmd` shim is how
## npm-installed tools appear. And a virtualenv puts its programs in `Scripts\`,
## not `bin/`, which is what [method venv_bin] is for; three call sites used to
## hard-code `bin` and would each have needed the same fix.
##
## ADDING A DEPENDENCY is one entry in [constant TOOLS]. The home-screen panel, the
## `--deps` report, the feedback record's environment block and the install hints
## are all rendered off that table, so nothing else has to be touched.

## Severity, and it is about CONSEQUENCE, not about how much we like the program.
enum { TIER_FEATURE, TIER_EXTRA }

## Where a dependency's state comes from: something on the machine, or something
## ghost installs for itself on first use.
enum { KIND_TOOL, KIND_MANAGED }

## THE TABLE. `bins` are candidate program names, first hit wins (so "a JavaScript
## runtime" is one row, not two). `check` names a bespoke probe handled in
## [method _probe_tool]; everything else is probed by running `version_args` and
## reading a version out of the first line.
##
## `install` is keyed by platform because a hint that names the wrong package
## manager is worse than no hint - it sends the reader off to install something
## that does not exist. The author develops on Linux only, so the macOS and Windows
## strings are the ones most likely to rot: they are commands anyone can verify
## against the upstream page named in `site`.
const TOOLS := [
	{
		"key": "ffmpeg",
		"name": "FFmpeg",
		"bins": ["ffmpeg"],
		"version_args": ["-version"],
		"tier": TIER_FEATURE,
		"used_for": "Masking (clip prep, waveforms, thumbnails), video export and "
			+ "transcode, and decoding the audio formats the engine has no loader for "
			+ "(FLAC, and MP3 in some builds).",
		"install": {
			"linux": "sudo pacman -S ffmpeg   ·   sudo apt install ffmpeg   ·   sudo dnf install ffmpeg",
			"macos": "brew install ffmpeg",
			"windows": "winget install Gyan.FFmpeg   (then reopen your terminal so PATH updates)",
		},
		"site": "https://ffmpeg.org/download.html",
	},
	{
		"key": "ffprobe",
		"name": "FFprobe",
		"bins": ["ffprobe"],
		"version_args": ["-version"],
		"tier": TIER_FEATURE,
		"used_for": "Masking reads a clip's frame count, duration and stream layout "
			+ "with it. Ships inside the same FFmpeg download - if this is missing and "
			+ "FFmpeg is not, you have a stripped build.",
		"install": {
			"linux": "sudo pacman -S ffmpeg   ·   sudo apt install ffmpeg   ·   sudo dnf install ffmpeg",
			"macos": "brew install ffmpeg",
			"windows": "winget install Gyan.FFmpeg   (then reopen your terminal so PATH updates)",
		},
		"site": "https://ffmpeg.org/download.html",
	},
	{
		"key": "python",
		"name": "Python 3",
		"bins": ["python3", "python"],
		# `py` is the Windows launcher, and it is often the ONLY thing on PATH after a
		# Microsoft Store install.
		"bins_windows": ["python", "python3", "py"],
		"version_args": ["--version"],
		"min_version": "3.9",
		"tier": TIER_FEATURE,
		"used_for": "Generative voice (the neural TTS host), the clown effect's face "
			+ "tracking, and YouTube/URL clip import. ghost builds its OWN virtualenvs "
			+ "for these - nothing is ever installed system-wide or into your own venv.",
		"install": {
			"linux": "sudo pacman -S python   ·   sudo apt install python3 python3-venv   ·   sudo dnf install python3",
			"macos": "brew install python   (or: xcode-select --install)",
			"windows": "winget install Python.Python.3.13   ·   tick \"Add python.exe to PATH\" if you use the installer",
		},
		"site": "https://www.python.org/downloads/",
	},
	{
		"key": "venv",
		"name": "Python venv + pip",
		"check": "venv",
		"tier": TIER_FEATURE,
		"used_for": "How ghost installs its own dependencies without touching yours: "
			+ "every optional feature gets a private virtualenv under the user data "
			+ "directory. Debian and Ubuntu ship these as SEPARATE packages from Python "
			+ "itself, which is the usual reason this row is red while the one above is green.",
		"install": {
			"linux": "sudo apt install python3-venv python3-pip   (already included on Arch and Fedora)",
			"macos": "included with Python 3",
			"windows": "included with Python 3",
		},
		"site": "https://docs.python.org/3/library/venv.html",
	},
	{
		"key": "jsruntime",
		"name": "JS runtime (Deno/Node)",
		"bins": ["deno", "node"],
		"version_args": ["--version"],
		"tier": TIER_EXTRA,
		"used_for": "YouTube's nsig challenge, which yt-dlp solves in a JavaScript "
			+ "runtime. Without one, URL imports still finish - at YouTube's punitive "
			+ "fallback throttle, around 40 KB/s.",
		"install": {
			"linux": "sudo pacman -S deno   ·   sudo apt install nodejs   ·   curl -fsSL https://deno.land/install.sh | sh",
			"macos": "brew install deno   ·   brew install node",
			"windows": "winget install DenoLand.Deno   ·   winget install OpenJS.NodeJS",
		},
		"site": "https://deno.com/",
	},
	{
		"key": "setpriv",
		"name": "setpriv (util-linux)",
		"bins": ["setpriv"],
		"version_args": ["--version"],
		"tier": TIER_EXTRA,
		"platforms": ["linux"],
		"used_for": "Binds every background program (ffmpeg, the voice host, a render) "
			+ "to ghost at the kernel level, so they die with it even when ghost is "
			+ "killed outright. Without it they are only cleaned up on a clean quit - "
			+ "see subprocess.gd.",
		"install": {
			"linux": "sudo pacman -S util-linux   ·   sudo apt install util-linux   (needs 2.33+ for --pdeathsig)",
		},
		"site": "https://github.com/util-linux/util-linux",
	},
	{
		"key": "claude",
		"name": "Claude Code CLI",
		"bins": ["claude"],
		"version_args": ["--version"],
		"tier": TIER_EXTRA,
		"used_for": "The Assistant dropdown on this screen. With it selected, a note "
			+ "left in the ` feedback console is dispatched to Claude Code as a one-shot "
			+ "fix against this checkout. Leave the dropdown Off and this is never touched.",
		"install": {
			"linux": "npm install -g @anthropic-ai/claude-code",
			"macos": "npm install -g @anthropic-ai/claude-code",
			"windows": "npm install -g @anthropic-ai/claude-code",
		},
		"site": "https://claude.com/claude-code",
	},
	{
		"key": "xvfb",
		"name": "xvfb-run",
		"bins": ["xvfb-run"],
		"no_version": true,
		"tier": TIER_EXTRA,
		"platforms": ["linux"],
		"dev": true,
		"used_for": "Development only: the pixel-readback gates in tests/ need a real "
			+ "GPU context and therefore a display, and tests/run_quiet.sh gives them a "
			+ "virtual one so no window ever appears.",
		"install": {
			"linux": "sudo pacman -S xorg-server-xvfb   ·   sudo apt install xvfb",
		},
		"site": "https://www.x.org/",
	},
]

## What ghost installs FOR ITSELF, lazily, the first time a feature is used. These
## are listed because "nothing is installed yet" and "the install failed" look
## identical from the home screen otherwise - and because a user who wants the
## download to happen on their own terms can see exactly what will appear where.
const MANAGED := [
	{
		"key": "voice_venv",
		"name": "Voice environment",
		"kind": KIND_MANAGED,
		"path": "user://voice_venv",
		"marker": "python",
		"size": "~90 MB",
		"used_for": "onnxruntime, numpy and the eSpeak phonemizer for the Generative "
			+ "voice. Created the first time you open Generative.",
	},
	{
		"key": "voices",
		"name": "Voice models (Piper)",
		"kind": KIND_MANAGED,
		"check": "voices",
		"size": "~60 MB each",
		"used_for": "The neural voices themselves, fetched per voice from Hugging Face "
			+ "the first time one is selected. Data, not code - the models are MIT-tagged "
			+ "and no GPL Piper code is installed (see voice_host/requirements.txt).",
	},
	{
		"key": "ytdlp_venv",
		"name": "Download environment",
		"kind": KIND_MANAGED,
		"path": "user://ytdlp_venv",
		"marker": "yt-dlp",
		"size": "~30 MB",
		"used_for": "yt-dlp, for importing a clip straight from a URL. Created the "
			+ "first time you paste one into the source field above.",
	},
	{
		"key": "face_venv",
		"name": "Face-tracking environment",
		"kind": KIND_MANAGED,
		"path": "user://face_venv",
		"marker": "python",
		"size": "~250 MB",
		"used_for": "MediaPipe, for the Masking clown effect's 478-point face landmarks. "
			+ "Created the first time a clown layer plays; a session that never uses the "
			+ "clown installs none of it.",
	},
]


# --- resolution --------------------------------------------------------------
#
# THE HOT PATH. `Subprocess` calls this for every child it starts, so it has to be
# cheap and it has to be thread-safe: mask_editor and voice_stream both start work
# off the main thread.

static var _resolved := {}          # program name -> absolute path, "" for a known miss
static var _dirs := PackedStringArray()
static var _lock := Mutex.new()


## The absolute path of `prog`, or "" if this machine does not have it. A name that
## already looks like a path is verified and returned as-is, so a caller holding a
## venv binary can pass it through the same door.
static func resolve(prog: String) -> String:
	if prog.is_empty():
		return ""
	if prog.contains("/") or prog.contains("\\"):
		return prog if FileAccess.file_exists(prog) else ""
	_lock.lock()
	var hit: bool = _resolved.has(prog)
	var cached: String = String(_resolved.get(prog, ""))
	_lock.unlock()
	if hit:
		return cached
	var found := _scan_for(prog)
	if found.is_empty():
		found = _ask_the_shell(prog)
	_lock.lock()
	_resolved[prog] = found
	_lock.unlock()
	return found


## The first of `names` this machine has, as an absolute path ("" if none). This is
## how "a JavaScript runtime" is asked for: `Deps.resolve_any(["deno", "node"])`.
static func resolve_any(names: Array) -> String:
	for n in names:
		var p := resolve(String(n))
		if not p.is_empty():
			return p
	return ""


## Is `prog` available at all? Sugar for the many `if not X: complain` call sites.
static func has(prog: String) -> bool:
	return not resolve(prog).is_empty()


## Forget every resolution and re-scan. For the panel's Rescan button: a user who
## installs ffmpeg while the home screen is open should not have to restart.
static func forget_all() -> void:
	_lock.lock()
	_resolved.clear()
	_dirs = PackedStringArray()
	_lock.unlock()


## `OS.execute` with the program resolved first. Returns -1 when it is not installed,
## which is what `OS.execute` returns for an unlaunchable program anyway - so an
## existing call site keeps its error handling when it switches over.
static func execute(prog: String, args: Array, output: Array = [],
		read_stderr := false) -> int:
	var bin := resolve(prog)
	if bin.is_empty():
		return -1
	return OS.execute(bin, args, output, read_stderr)


## Every directory worth looking in: PATH first (it is the user's own statement of
## intent), then the platform's usual install locations, which is where a
## GUI-launched app finds the things its PATH never mentioned.
static func search_dirs() -> PackedStringArray:
	_lock.lock()
	var have := not _dirs.is_empty()
	var out := _dirs
	_lock.unlock()
	if have:
		return out

	var dirs := PackedStringArray()
	var sep := ";" if _is_windows() else ":"
	for d in OS.get_environment("PATH").split(sep, false):
		var s := String(d).strip_edges()
		if not s.is_empty() and not dirs.has(s):
			dirs.append(s)

	var home := _home()
	var extra := PackedStringArray()
	match _platform():
		"macos":
			# Apple Silicon Homebrew, Intel Homebrew, MacPorts - none of which are on a
			# double-clicked app's PATH, which is `/usr/bin:/bin:/usr/sbin:/sbin`.
			extra = PackedStringArray([
				"/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin",
				"/opt/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin",
				home.path_join(".local/bin"), home.path_join("bin"),
				home.path_join("homebrew/bin"),
			])
		"windows":
			var local := OS.get_environment("LOCALAPPDATA")
			var progf := OS.get_environment("ProgramFiles")
			var progf86 := OS.get_environment("ProgramFiles(x86)")
			extra = PackedStringArray([
				local.path_join("Microsoft/WindowsApps"),
				local.path_join("Programs/Python/Launcher"),
				home.path_join("scoop/shims"),                  # scoop
				"C:/ProgramData/chocolatey/bin",                # chocolatey
				local.path_join("Microsoft/WinGet/Links"),      # winget shims
				progf.path_join("ffmpeg/bin"), "C:/ffmpeg/bin",
				progf.path_join("nodejs"), progf86.path_join("nodejs"),
				local.path_join("Programs/nodejs"),
				home.path_join(".deno/bin"),
				"C:/Windows/System32", "C:/Windows",
			])
			# Windows installs Python into a VERSIONED directory and only optionally
			# puts it on PATH, so enumerate whatever is actually there.
			for base in [local.path_join("Programs/Python"), "C:/"]:
				for sub in _dir_children(base):
					if sub.begins_with("Python"):
						extra.append(base.path_join(sub))
						extra.append(base.path_join(sub).path_join("Scripts"))
		_:
			extra = PackedStringArray([
				"/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin",
				home.path_join(".local/bin"), home.path_join("bin"),
				"/snap/bin",                                     # snap
				"/var/lib/flatpak/exports/bin",                  # flatpak, system
				home.path_join(".local/share/flatpak/exports/bin"),
				home.path_join(".nix-profile/bin"), "/nix/var/nix/profiles/default/bin",
				home.path_join(".cargo/bin"), home.path_join(".deno/bin"),
				"/opt/bin",
			])
	for d in extra:
		var s := String(d).strip_edges()
		if not s.is_empty() and not dirs.has(s):
			dirs.append(s)

	_lock.lock()
	_dirs = dirs
	_lock.unlock()
	return dirs


## A program inside one of ghost's own virtualenvs. Windows puts them in `Scripts\`
## with an `.exe`; everywhere else it is `bin/` with no suffix. `venv` may be a
## `user://` path or an absolute one.
static func venv_bin(venv: String, tool_name: String) -> String:
	var root := ProjectSettings.globalize_path(venv) if venv.begins_with("user://") \
		or venv.begins_with("res://") else venv
	if _is_windows():
		var base := root.path_join("Scripts").path_join(tool_name)
		for ext in [".exe", ".cmd", ".bat", ""]:
			if FileAccess.file_exists(base + ext):
				return base + ext
		return base + ".exe"     # the path it WOULD have, so `file_exists` reads false
	return root.path_join("bin").path_join(tool_name)


## Where the platform wants an application to keep its data. Mirrors `user://`, and
## exists because the Python side has to agree with the Godot side about where the
## voice models live without importing any of this.
static func data_dir() -> String:
	match _platform():
		"windows":
			var local := OS.get_environment("LOCALAPPDATA")
			return (local if not local.is_empty() else _home().path_join("AppData/Local")).path_join("ghost")
		"macos":
			return _home().path_join("Library/Application Support/ghost")
		_:
			var xdg := OS.get_environment("XDG_DATA_HOME")
			return (xdg if not xdg.is_empty() else _home().path_join(".local/share")).path_join("ghost")


# --- reporting ---------------------------------------------------------------

## Every dependency's current state, in table order: the tools first, then ghost's
## own managed environments. Each entry carries what the panel and the text report
## both need - `found`, `path`, `version`, `note`, plus the table's own fields.
##
## This SPAWNS PROCESSES (one `--version` per installed tool), so it is slow enough
## to notice on a frame - a few hundred milliseconds cold. The panel runs it on a
## thread; `--deps` runs it inline because there is no frame to protect.
static func report(include_dev := true) -> Array:
	var out: Array = []
	for t in TOOLS:
		if not _on_this_platform(t):
			continue
		if bool(t.get("dev", false)) and not include_dev:
			continue
		out.append(_probe_tool(t))
	for m in MANAGED:
		out.append(_probe_managed(m))
	return out


## A one-line verdict over `rows`: "" when nothing is wrong, otherwise what is
## missing and how much it costs.
static func verdict(rows: Array) -> String:
	var missing: PackedStringArray = []
	for r in rows:
		if int(r.get("tier", TIER_EXTRA)) == TIER_FEATURE and not bool(r.get("found", false)):
			missing.append(String(r.get("name", "?")))
	if missing.is_empty():
		return ""
	# The header this feeds is one line beside two buttons. Past two names it says how
	# many rather than how long - the list itself is right underneath.
	if missing.size() > 2:
		return "missing %d" % missing.size()
	return "missing: " + ", ".join(missing)


## The whole thing as text - for `--deps`, for the `>_` log, for the clipboard
## button, and for pasting into a bug report. Deliberately plain ASCII apart from
## the status glyphs: it gets copied into terminals and issue trackers.
static func format_report(rows: Array = []) -> String:
	if rows.is_empty():
		rows = report()
	var lines: PackedStringArray = []
	lines.append("ghost - environment report")
	lines.append(describe_host())
	lines.append("")
	var in_managed := false
	for r in rows:
		if int(r.get("kind", KIND_TOOL)) == KIND_MANAGED and not in_managed:
			in_managed = true
			lines.append("")
			lines.append("ghost's own (installed on first use, never system-wide):")
		var glyph := "[ok]" if bool(r.get("found", false)) else \
			("[--]" if int(r.get("tier", TIER_EXTRA)) == TIER_EXTRA else "[!!]")
		var detail := String(r.get("version", ""))
		if detail.is_empty():
			detail = String(r.get("note", ""))
			# The download size is trimmed out of the panel's narrow column but there
			# is room for it here, and it is the first thing anyone wants to know.
			if int(r.get("kind", KIND_TOOL)) == KIND_MANAGED and not String(r.get("size", "")).is_empty():
				detail += " · " + String(r.get("size", ""))
		lines.append("  %s %-26s %s" % [glyph, String(r.get("name", "?")), detail])
		var p := String(r.get("path", ""))
		if not p.is_empty():
			lines.append("       %s" % p)
	var problems: Array = []
	for r in rows:
		if bool(r.get("found", false)) or int(r.get("kind", KIND_TOOL)) == KIND_MANAGED:
			continue
		problems.append(r)
	if not problems.is_empty():
		lines.append("")
		lines.append("Not found:")
		for r in problems:
			lines.append("")
			lines.append("  %s - %s" % [String(r.get("name", "?")),
				"needed for a feature" if int(r.get("tier", TIER_EXTRA)) == TIER_FEATURE
				else "optional"])
			lines.append("    " + String(r.get("used_for", "")))
			var hint := install_hint(r)
			if not hint.is_empty():
				lines.append("    install: " + hint)
			var site := String(r.get("site", ""))
			if not site.is_empty():
				lines.append("    " + site)
	return "\n".join(lines)


## The machine, in one line. Part of every report and of the feedback record,
## because "works here" and "not there" is usually this line.
static func describe_host() -> String:
	var v := Engine.get_version_info()
	var line := "%s %s · Godot %s · %s" % [
		OS.get_name(), OS.get_distribution_name(), String(v.get("string", "?")),
		String(ProjectSettings.get_setting("rendering/renderer/rendering_method", "?"))]
	# No rendering device under `--headless` (the dummy driver), so this is absent in
	# every gate and present in every real run.
	if RenderingServer.get_rendering_device() != null:
		line += " · " + RenderingServer.get_video_adapter_name()
	return line


## A compact machine-readable block for the feedback record. A dispatched fix that
## can see "this user has no ffmpeg" does not have to guess at a Masking bug report
## that is really a missing-dependency report.
static func snapshot() -> Dictionary:
	var rows := report()
	var tools := {}
	for r in rows:
		tools[String(r.get("key", "?"))] = {
			"found": bool(r.get("found", false)),
			"version": String(r.get("version", "")),
			"path": String(r.get("path", "")),
		}
	return {"host": describe_host(), "tools": tools, "missing": verdict(rows)}


## The install line for THIS platform, "" if the table has none for it.
static func install_hint(entry: Dictionary) -> String:
	var m: Dictionary = entry.get("install", {})
	return String(m.get(_platform(), ""))


## A one-sentence "here is how to get it" for a table key, for the error message a
## mode shows when it discovers the gap itself. The point is that a failure deep
## inside a mode says the same thing the home screen would have said.
static func hint(key: String) -> String:
	for t in TOOLS:
		if String(t.get("key", "")) != key:
			continue
		var line := install_hint(t)
		if line.is_empty():
			line = String(t.get("site", ""))
		return "Install it with:  %s" % line if not line.is_empty() else ""
	return ""


## The table entry for a key, {} if there is none. For a call site that wants the
## description or the site rather than the install line.
static func entry(key: String) -> Dictionary:
	for t in TOOLS:
		if String(t.get("key", "")) == key:
			return t
	return {}


# --- probes ------------------------------------------------------------------

static func _probe_tool(t: Dictionary) -> Dictionary:
	var r := t.duplicate(true)
	r["kind"] = KIND_TOOL
	r["found"] = false
	r["path"] = ""
	r["version"] = ""
	r["note"] = ""

	if String(t.get("check", "")) == "venv":
		return _probe_venv(r)

	var names: Array = t.get("bins_windows", t.get("bins", [])) if _is_windows() \
		else t.get("bins", [])
	for n in names:
		var p := resolve(String(n))
		if p.is_empty():
			continue
		r["found"] = true
		r["path"] = p
		r["bin"] = String(n)
		# `no_version` is for programs that have no version flag at all (xvfb-run
		# answers `--help` with its usage). Present is the whole answer there.
		if not bool(t.get("no_version", false)):
			r["version"] = _version_of(p, t.get("version_args", ["--version"]))
		break
	if not r["found"]:
		r["note"] = "not found"
		return r
	if String(r["version"]).is_empty() and bool(t.get("no_version", false)):
		r["version"] = "installed"

	# The Microsoft Store stub: `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe`
	# exists, is on PATH, and does nothing but open the Store. It resolves and then
	# every venv creation fails, which is unreadable without saying so here.
	if _is_windows() and String(r["path"]).contains("WindowsApps") and String(r["version"]).is_empty():
		r["found"] = false
		r["note"] = "Microsoft Store stub - not a real install"
		return r

	var minv := String(t.get("min_version", ""))
	if not minv.is_empty() and not String(r["version"]).is_empty() \
			and _version_lt(String(r["version"]), minv):
		r["note"] = "older than %s - some features will not install" % minv
	return r


## Python's `venv` and `pip` are separate Debian/Ubuntu packages, and their absence
## presents as ghost failing to bootstrap ANY of its own environments. Asking the
## interpreter directly is the only honest check - a `pip` binary on PATH says
## nothing about whether `python3 -m venv` can produce one.
static func _probe_venv(r: Dictionary) -> Dictionary:
	var py := resolve_any(["python", "python3", "py"] if _is_windows() else ["python3", "python"])
	if py.is_empty():
		r["note"] = "needs Python 3 first"
		return r
	var out: Array = []
	if OS.execute(py, ["-c", "import venv, ensurepip"], out, true) == 0:
		r["found"] = true
		r["path"] = py
		r["version"] = "available"
	else:
		r["note"] = "python3 -m venv is not usable"
	return r


static func _probe_managed(m: Dictionary) -> Dictionary:
	var r := m.duplicate(true)
	r["kind"] = KIND_MANAGED
	r["tier"] = TIER_EXTRA
	r["found"] = false
	r["version"] = ""
	r["path"] = ""
	if String(m.get("check", "")) == "voices":
		var dir := data_dir().path_join("voices").path_join("piper")
		r["path"] = dir
		var n := 0
		for f in _dir_files(dir):
			if f.ends_with(".onnx"):
				n += 1
		r["found"] = n > 0
		r["version"] = "%d installed" % n if n > 0 else ""
		r["note"] = "on first use"
		return r
	var root := ProjectSettings.globalize_path(String(m.get("path", "")))
	r["path"] = root
	r["found"] = FileAccess.file_exists(venv_bin(root, String(m.get("marker", "python"))))
	r["version"] = "ready" if r["found"] else ""
	# Short, because it shares a narrow column with version strings. The size and the
	# path are in the detail pane, which has the room.
	r["note"] = "" if r["found"] else "on first use"
	return r


## Run `bin` with `args` and pull a version out of what it says. Reads stderr too:
## some tools print their version there, and a tool that fails to run at all should
## surface as "no version" rather than as a hang or a stray console window.
static func _version_of(bin: String, args: Array) -> String:
	var out: Array = []
	if OS.execute(bin, args, out, true) != 0 or out.is_empty():
		return ""
	var first := String(out[0]).strip_edges().split("\n")[0].strip_edges()
	var re := RegEx.new()
	re.compile("\\d+(?:\\.\\d+)+")
	var m := re.search(first)
	if m != null:
		return m.get_string()
	return first.substr(0, 40)


# --- primitives --------------------------------------------------------------

## The filesystem half of resolution: every candidate name (with every Windows
## extension) against every search directory. No subprocess, so this is the part
## that is allowed to run on the hot path.
static func _scan_for(prog: String) -> String:
	var names := PackedStringArray([prog])
	if _is_windows() and prog.get_extension().is_empty():
		names = PackedStringArray()
		var pathext := OS.get_environment("PATHEXT")
		if pathext.is_empty():
			pathext = ".COM;.EXE;.BAT;.CMD"
		for ext in pathext.split(";", false):
			names.append(prog + String(ext).strip_edges().to_lower())
		names.append(prog)
	for d in search_dirs():
		for n in names:
			var cand := String(d).path_join(n)
			if FileAccess.file_exists(cand):
				return cand
	return ""


## The fallback, and only ever reached on a miss: ask the OS's own lookup, which
## can still know something the scan does not (a PATH entry behind a symlinked
## directory the scan normalised differently, a shim registered by an app store).
static func _ask_the_shell(prog: String) -> String:
	var out: Array = []
	var finder := "where" if _is_windows() else "which"
	if OS.execute(finder, [prog], out) != 0 or out.is_empty():
		return ""
	var p := String(out[0]).strip_edges().split("\n")[0].strip_edges()
	return p if FileAccess.file_exists(p) else ""


static func _platform() -> String:
	match OS.get_name():
		"Windows", "UWP": return "windows"
		"macOS": return "macos"
		_: return "linux"


static func _is_windows() -> bool:
	return _platform() == "windows"


static func _home() -> String:
	var h := OS.get_environment("HOME")
	if h.is_empty():
		h = OS.get_environment("USERPROFILE")
	return h


## Does this row apply here at all? A `setpriv` row on a Mac is noise, not a warning.
static func _on_this_platform(t: Dictionary) -> bool:
	var only: Array = t.get("platforms", [])
	return only.is_empty() or only.has(_platform())


static func _dir_children(path: String) -> PackedStringArray:
	var d := DirAccess.open(path)
	return d.get_directories() if d != null else PackedStringArray()


static func _dir_files(path: String) -> PackedStringArray:
	var d := DirAccess.open(path)
	return d.get_files() if d != null else PackedStringArray()


## "3.9" < "3.10", which is the whole reason this is not a string compare.
static func _version_lt(a: String, b: String) -> bool:
	var pa := a.split(".")
	var pb := b.split(".")
	for i in maxi(pa.size(), pb.size()):
		var x := int(pa[i]) if i < pa.size() else 0
		var y := int(pb[i]) if i < pb.size() else 0
		if x != y:
			return x < y
	return false
