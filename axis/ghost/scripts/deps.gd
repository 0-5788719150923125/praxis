extends RefCounted
class_name Deps

## Deps - every external program and environment ghost uses, in one place, with one promise:
## RESOLUTION AND REPORTING ARE THE SAME CODE.
##
## ghost shells out constantly - ffmpeg for every Masking clip, ffprobe for its frame counts, a
## Python environment for the voice host, the clown's face tracker, the URL import and the tablet's
## page capture. There are two halves and they must not drift apart:
##
##   [method resolve] is what actually launches things. [Subprocess] runs every child through it,
##   so a bare "ffmpeg" handed to `Subprocess.start` becomes an absolute path before the kernel
##   ever sees it.
##
##   [method report] is what the home screen lists. It asks the SAME resolver, so the panel cannot
##   claim a dependency is present while the launch path fails to find it - a status panel with its
##   own detection logic is a panel that lies eventually.
##
## THREE KINDS OF ROW. [constant FETCHED] are programs ghost downloads and keeps current itself
## (FFmpeg, uv), and [constant MANAGED] is what ghost builds with them: its own Python and one
## environment per feature. Both are installed by the `Provisioner` autoload - see [Provision] -
## so nothing in them is the user's to install. [constant TOOLS] are the few programs that stay the
## machine's own: Linux system utilities, and the AI CLIs, which keep their own logins.
##
## THE PATH PROBLEM, for the programs found on the machine. A GUI-launched app does not inherit an
## interactive shell's PATH. On macOS a double-clicked app gets `/usr/bin:/bin:/usr/sbin:/sbin` and
## nothing else, so a Homebrew program in `/opt/homebrew/bin` is invisible; on Linux the same
## happens to `~/.local/bin` and Flatpak/snap exports. [method search_dirs] therefore scans PATH
## *plus* the places each platform actually installs things, as a filesystem lookup rather than a
## `which` subprocess: it costs microseconds, works identically on all three platforms, and needs no
## external program to find external programs.
##
## WINDOWS gets two extra rules. Executables need an extension, so every candidate is tried against
## `PATHEXT` (`.exe`, `.cmd`, `.bat`, ...) - a `.cmd` shim is how npm-installed tools appear. And a
## virtualenv puts its programs in `Scripts\`, not `bin/`, which is what [method venv_bin] is for.
##
## ADDING A DEPENDENCY is one row in one of the three tables. The home-screen panel, the `--deps`
## report, the feedback record's environment block and the hints are all rendered off them.

## Severity, and it is about CONSEQUENCE, not about how much we like the program.
enum { TIER_FEATURE, TIER_EXTRA }

## Where a row's state comes from: the machine, a program ghost builds from what it fetched, or a
## program ghost downloads itself.
enum { KIND_TOOL, KIND_MANAGED, KIND_FETCHED }

## Programs ghost downloads from their upstream releases, checks against the published SHA-256 and
## keeps current. `provides` are the program names it answers [method resolve] for; `sources` says
## where each machine's build comes from (`*` is every machine), in a form [Provision] reads.
const FETCHED := [
	{
		"key": "ffmpeg",
		"name": "FFmpeg",
		"kind": KIND_FETCHED,
		"provides": ["ffmpeg", "ffprobe"],
		"version_args": ["-version"],
		"tier": TIER_FEATURE,
		"size": "~70 MB (~115 MB on Windows)",
		"used_for": "Masking (clip prep, waveforms, thumbnails), video export and transcode, film "
			+ "panels, and decoding the audio formats the engine has no loader for (FLAC, and MP3 "
			+ "in some builds). ghost downloads the newest release itself: Martin Riedl's static "
			+ "builds on Linux and macOS, Gyan Doshi's on Windows, BtbN's on Windows on ARM.",
		"sources": {
			"linux-x86_64": {"via": "riedl", "os": "linux", "arch": "amd64"},
			"linux-arm64": {"via": "riedl", "os": "linux", "arch": "arm64"},
			"macos-arm64": {"via": "riedl", "os": "macos", "arch": "arm64"},
			"macos-x86_64": {"via": "riedl", "os": "macos", "arch": "amd64"},
			"windows-x86_64": {"via": "github", "repo": "GyanD/codexffmpeg",
				"asset": "^ffmpeg-[0-9.]+-essentials_build\\.zip$"},
			"windows-arm64": {"via": "github", "repo": "BtbN/FFmpeg-Builds",
				"asset": "^ffmpeg-n([0-9.]+)-latest-winarm64-gpl-[0-9.]+\\.zip$"},
		},
		"site": "https://ffmpeg.org/download.html",
	},
	{
		"key": "uv",
		"name": "uv",
		"kind": KIND_FETCHED,
		"provides": ["uv"],
		"version_args": ["--version"],
		"tier": TIER_FEATURE,
		"size": "~20 MB",
		"used_for": "Builds ghost's Python environments: installs Python itself, then each "
			+ "feature's packages, from its own cache. The newest release, from PyPI.",
		"sources": {"*": {"via": "pypi", "package": "uv"}},
		"site": "https://docs.astral.sh/uv/",
	},
]

## What ghost builds for itself with uv. `python` is installed at launch; each environment the
## first time its feature is used, from `requirements` (a file) or `packages` (inline), then checked
## by importing `imports`. `post` runs with the environment's Python after the packages land, and
## `files` are data downloaded beside it. `unsupported` names the machines its wheels do not exist
## for, so they are told why instead of watching an install fail.
const MANAGED := [
	{
		"key": "python",
		"name": "Python",
		"kind": KIND_MANAGED,
		"tier": TIER_FEATURE,
		"size": "~35 MB",
		"used_for": "The interpreter every environment below runs on: CPython's newest patch of "
			+ "the version ghost targets, installed by uv into ghost's own directory - never onto "
			+ "PATH or into the system.",
	},
	{
		"key": "voice_venv",
		"name": "Voice environment",
		"kind": KIND_MANAGED,
		"path": "user://voice_venv",
		"requirements": "res://voice_host/requirements.txt",
		"imports": "onnxruntime, numpy, espeakng_loader, phonemizer, nltk",
		"size": "~320 MB",
		"unsupported": {"macos-x86_64": "onnxruntime publishes no build for Intel Macs"},
		"used_for": "onnxruntime, numpy and the eSpeak phonemizer for the Generative voice. "
			+ "Built the first time you open Generative.",
	},
	{
		"key": "ytdlp_venv",
		"name": "Download environment",
		"kind": KIND_MANAGED,
		"path": "user://ytdlp_venv",
		"packages": ["yt-dlp[default]>=2026.8.19", "deno>=2.9.7"],
		"imports": "yt_dlp, yt_dlp_ejs",
		"size": "~140 MB",
		"unsupported": {"windows-arm64": "yt-dlp's brotli dependency has no Windows-on-ARM build"},
		"used_for": "yt-dlp, for importing a clip straight from a URL, with Deno to answer "
			+ "YouTube's download challenges (without one, imports crawl at YouTube's punitive "
			+ "throttle). Built the first time you paste a URL into the source field above.",
	},
	{
		"key": "capture_venv",
		"name": "Page-capture environment",
		"kind": KIND_MANAGED,
		"path": "user://capture_venv",
		"requirements": "res://capture_host/requirements.txt",
		"imports": "playwright",
		"post": [{"label": "downloading Chromium", "args": ["-m", "playwright", "install", "chromium"]}],
		"size": "~150 MB, and Chromium ~170 MB",
		"used_for": "Playwright and its Chromium, for the tablet medium's REAL pages: a url the "
			+ "chapter writes nothing under is captured from the web once, as a picture. Built the "
			+ "first time you press Capture on one.",
	},
	{
		"key": "face_venv",
		"name": "Body/face-tracking environment",
		"kind": KIND_MANAGED,
		"path": "user://face_venv",
		"requirements": "res://face_host/requirements.txt",
		"imports": "mediapipe, cv2, numpy",
		"files": [
			{"key": "face_model", "name": "face model", "file": "face_landmarker.task",
				"url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
				"sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
				"size": 3758596},
			{"key": "pose_model", "name": "pose model", "file": "pose_landmarker_full.task",
				"url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
				"sha256": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
				"size": 9398198},
		],
		"size": "~520 MB",
		"unsupported": {
			"macos-x86_64": "mediapipe publishes no build for Intel Macs",
			"windows-arm64": "opencv-contrib-python has no Windows-on-ARM build",
		},
		"used_for": "MediaPipe and its two models, for the Masking clown effect's 478-point face "
			+ "landmarks and the umbra effect's person silhouette and 33 body landmarks. ONE "
			+ "environment for both. Built the first time a clown or umbra layer plays.",
	},
	{
		"key": "voices",
		"name": "Voice models (Piper)",
		"kind": KIND_MANAGED,
		"check": "voices",
		"size": "~60 MB each",
		"used_for": "The neural voices themselves, fetched per voice from Hugging Face the first "
			+ "time one is selected. Data, not code - the models are MIT-tagged and no GPL Piper "
			+ "code is installed (see voice_host/requirements.txt).",
	},
]

## Programs that stay the machine's own. `bins` are candidate names, first hit wins; `install` is
## keyed by platform, because a hint that names the wrong package manager is worse than none.
const TOOLS := [
	{
		"key": "setpriv",
		"name": "setpriv (util-linux)",
		"bins": ["setpriv"],
		"version_args": ["--version"],
		"tier": TIER_EXTRA,
		"platforms": ["linux"],
		"used_for": "Binds every background program (ffmpeg, the voice host, a render) to ghost at "
			+ "the kernel level, so they die with it even when ghost is killed outright. Without it "
			+ "they are only cleaned up on a clean quit - see subprocess.gd. Part of util-linux, "
			+ "which every distribution installs.",
		"install": {
			"linux": "sudo pacman -S util-linux   ·   sudo apt install util-linux   (needs 2.33+ for --pdeathsig)",
		},
		"site": "https://github.com/util-linux/util-linux",
	},
	{
		"key": "fallocate",
		"name": "fallocate (util-linux)",
		"bins": ["fallocate"],
		"no_version": true,
		"tier": TIER_FEATURE,
		"platforms": ["linux"],
		"used_for": "Keeping a video export's scratch file SMALL: the render's AVI is encoded as it "
			+ "is written, and the part already encoded is released from disk as it goes (a "
			+ "punched hole), so the intermediate never grows past a few seconds of video. Without "
			+ "it the scratch file grows to the whole film before it is deleted.",
		"install": {
			"linux": "part of util-linux - installed on every mainstream distribution",
		},
		"site": "https://github.com/util-linux/util-linux",
	},
	{
		"key": "xvfb",
		"name": "xvfb-run",
		"bins": ["xvfb-run"],
		"no_version": true,
		"tier": TIER_FEATURE,
		"platforms": ["linux"],
		"used_for": "Giving a video export a display of its own, so the recording cannot be frozen "
			+ "by the desktop: Godot stops rendering whenever the compositor stops drawing its "
			+ "window, and the movie writer then re-captures the last frame while the audio keeps "
			+ "advancing. Also what the pixel-readback gates in tests/ use (tests/run_quiet.sh), so "
			+ "no window ever appears for those. An X server, so it stays the system's to install.",
		"install": {
			"linux": "sudo pacman -S xorg-server-xvfb   ·   sudo apt install xvfb   ·   sudo dnf install xorg-x11-server-Xvfb",
		},
		"site": "https://www.x.org/",
	},
	{
		"key": "claude",
		"name": "Claude Code CLI",
		"bins": ["claude"],
		"version_args": ["--version"],
		"tier": TIER_EXTRA,
		"used_for": "A writer for the tarot mode (its plan, booklet and reading), and the Assistant "
			+ "dropdown on this screen: with it selected, a note left in the ` feedback console is "
			+ "dispatched to Claude Code as a one-shot fix against this checkout. It keeps its own "
			+ "login, so it stays yours to install, and it runs only when asked.",
		"install": {
			"linux": "curl -fsSL https://claude.ai/install.sh | bash",
			"macos": "curl -fsSL https://claude.ai/install.sh | bash",
			"windows": "irm https://claude.ai/install.ps1 | iex   (PowerShell)   ·   winget install Anthropic.ClaudeCode",
		},
		"site": "https://code.claude.com/docs/en/setup",
	},
	{
		"key": "codex",
		"name": "OpenAI Codex CLI",
		"bins": ["codex"],
		"version_args": ["--version"],
		"tier": TIER_EXTRA,
		"used_for": "The tarot mode's painter (and a writer it can pick), the pictures the book and "
			+ "notebook media print - both through Codex's built-in image generation - and the "
			+ "Assistant dropdown's Codex option (a ` feedback note dispatched to Codex as a "
			+ "one-shot fix against this checkout). It keeps its own login, so it stays yours to "
			+ "install, and it runs only when asked.",
		"install": {
			"linux": "curl -fsSL https://chatgpt.com/codex/install.sh | sh",
			"macos": "curl -fsSL https://chatgpt.com/codex/install.sh | sh",
			"windows": "npm install -g @openai/codex   (needs Node.js)",
		},
		"site": "https://developers.openai.com/codex/cli",
	},
]


# --- resolution --------------------------------------------------------------
#
# THE HOT PATH. `Subprocess` calls this for every child it starts, so it has to be cheap and it has
# to be thread-safe: mask_editor and voice_stream both start work off the main thread.

static var _resolved := {}          # program name -> absolute path, "" for a known miss
static var _dirs := PackedStringArray()
static var _lock := Mutex.new()


## The absolute path of `prog`, or "" if this machine does not have it. ghost's own copy of a
## fetched program wins over one on the machine: it is the build the features were written against
## (a distribution's FFmpeg can lack libtheora or libx264), and it is the one kept current. Until it
## has downloaded, a copy on the machine stands in. A name that already looks like a path is
## verified and returned as-is, so a caller holding a venv binary can pass it through the same door.
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
	var found := ""
	var key := fetched_key(prog)
	if not key.is_empty():
		found = Provision.tool_path(key, prog)
	if found.is_empty():
		found = _scan_for(prog)
	if found.is_empty():
		found = _ask_the_shell(prog)
	_lock.lock()
	_resolved[prog] = found
	_lock.unlock()
	return found


## Is `prog` available at all? Sugar for the many `if not X: complain` call sites.
static func has(prog: String) -> bool:
	return not resolve(prog).is_empty()


## The [constant FETCHED] row that provides `prog`, "" when ghost does not fetch it.
static func fetched_key(prog: String) -> String:
	for t in FETCHED:
		if (t["provides"] as Array).has(prog):
			return String(t["key"])
	return ""


## Forget every resolution and re-scan. For the panel's Rescan button, and for the Provisioner once
## a download lands: a program installed while ghost is open should not need a restart.
static func forget_all() -> void:
	_lock.lock()
	_resolved.clear()
	_dirs = PackedStringArray()
	_lock.unlock()


## `OS.execute` with the program resolved first. Returns -1 when it is not installed, which is what
## `OS.execute` returns for an unlaunchable program anyway - so an existing call site keeps its error
## handling when it switches over.
static func execute(prog: String, args: Array, output: Array = [],
		read_stderr := false) -> int:
	var bin := resolve(prog)
	if bin.is_empty():
		return -1
	return OS.execute(bin, args, output, read_stderr)


## Every directory worth looking in: PATH first (it is the user's own statement of intent), then the
## platform's usual install locations, which is where a GUI-launched app finds the things its PATH
## never mentioned.
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
				home.path_join(".local/bin"),                   # Claude Code's native installer
				home.path_join("scoop/shims"),                  # scoop
				"C:/ProgramData/chocolatey/bin",                # chocolatey
				local.path_join("Microsoft/WinGet/Links"),      # winget shims
				progf.path_join("ffmpeg/bin"), "C:/ffmpeg/bin",
				progf.path_join("nodejs"), progf86.path_join("nodejs"),
				local.path_join("Programs/nodejs"),
				OS.get_environment("APPDATA").path_join("npm"), # npm -g shims
				home.path_join(".deno/bin"),
				"C:/Windows/System32", "C:/Windows",
			])
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


## A program inside one of ghost's own virtualenvs. Windows puts them in `Scripts\` with an `.exe`;
## everywhere else it is `bin/` with no suffix. `venv` may be a `user://` path or an absolute one.
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


## Where the platform wants an application to keep its data. The Python side has to agree with the
## Godot side about where the voice models live without importing any of this.
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

## The row for `key`, from whichever table holds it; {} if none does.
static func entry(key: String) -> Dictionary:
	for table in [FETCHED, MANAGED, TOOLS]:
		for t in table:
			if String(t.get("key", "")) == key:
				return t
	return {}


## Every dependency's current state on disk, in table order: what ghost fetches, what it builds,
## then the machine's own. Each entry carries what the panel and the text report both need -
## `found`, `path`, `version`, `note`, plus the table's own fields. What a job is doing right now is
## [method Provision.state]'s to say; this is what is installed.
##
## This SPAWNS PROCESSES (one `--version` per installed program), so it is slow enough to notice on
## a frame - a few hundred milliseconds cold. The panel runs it on a thread; `--deps` runs it inline
## because there is no frame to protect.
static func report(include_dev := true) -> Array:
	var out: Array = []
	for t in FETCHED:
		out.append(_probe_fetched(t))
	for m in MANAGED:
		out.append(_probe_managed(m))
	for t in TOOLS:
		if not _on_this_platform(t):
			continue
		if bool(t.get("dev", false)) and not include_dev:
			continue
		out.append(_probe_tool(t))
	return out


## A one-line verdict over `rows`: "" when nothing a feature needs is absent, otherwise what is
## missing. Environments are not counted - they are built when their feature first asks.
static func verdict(rows: Array) -> String:
	var missing: PackedStringArray = []
	for r in rows:
		if int(r.get("tier", TIER_EXTRA)) != TIER_FEATURE or bool(r.get("found", false)):
			continue
		if int(r.get("kind", KIND_TOOL)) == KIND_MANAGED and String(r.get("key", "")) != "python":
			continue
		missing.append(String(r.get("name", "?")))
	if missing.is_empty():
		return ""
	# The header this feeds is one line beside two buttons. Past two names it says how many rather
	# than how long - the list itself is right underneath.
	if missing.size() > 2:
		return "missing %d" % missing.size()
	return "missing: " + ", ".join(missing)


## The whole thing as text - for `--deps`, for the `>_` log, for the clipboard button, and for
## pasting into a bug report. Plain ASCII apart from the status glyphs: it gets copied into
## terminals and issue trackers.
static func format_report(rows: Array = []) -> String:
	if rows.is_empty():
		rows = report()
	var lines: PackedStringArray = []
	lines.append("ghost - environment report")
	lines.append(describe_host())
	var group := -1
	for r in rows:
		var g := _group_of(r)
		if g != group:
			group = g
			lines.append("")
			lines.append(["ghost's own, downloaded and kept current:",
				"ghost's own, built the first time a feature needs it:",
				"From this machine:"][g])
		var glyph := "[ok]" if bool(r.get("found", false)) else \
			("[!!]" if int(r.get("tier", TIER_EXTRA)) == TIER_FEATURE and g != 1 else "[--]")
		var detail := String(r.get("version", ""))
		if detail.is_empty():
			detail = String(r.get("note", ""))
			if g != 2 and not String(r.get("size", "")).is_empty():
				detail += " · " + String(r.get("size", ""))
		lines.append("  %s %-30s %s" % [glyph, String(r.get("name", "?")), detail])
		var p := String(r.get("path", ""))
		if not p.is_empty():
			lines.append("       %s" % p)
	var problems: Array = []
	for r in rows:
		if not bool(r.get("found", false)) and _group_of(r) != 1:
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
			if _group_of(r) == 0:
				lines.append("    ghost downloads this itself at an ordinary launch, or now with: "
					+ "godot --headless --path axis/ghost -- --provision")
				continue
			var hint := install_hint(r)
			if not hint.is_empty():
				lines.append("    install: " + hint)
			var site := String(r.get("site", ""))
			if not site.is_empty():
				lines.append("    " + site)
	return "\n".join(lines)


## Which of the three groups a row is listed under: 0 fetched (with Python, which is installed at
## launch like them), 1 built on first use, 2 the machine's own.
static func _group_of(r: Dictionary) -> int:
	match int(r.get("kind", KIND_TOOL)):
		KIND_FETCHED:
			return 0
		KIND_MANAGED:
			return 0 if String(r.get("key", "")) == "python" else 1
	return 2


static func group_of(r: Dictionary) -> int:
	return _group_of(r)


## The machine, in one line. Part of every report and of the feedback record, because "works here"
## and "not there" is usually this line.
static func describe_host() -> String:
	var v := Engine.get_version_info()
	var line := "%s %s · %s · Godot %s · %s" % [
		OS.get_name(), OS.get_distribution_name(), Engine.get_architecture_name(),
		String(v.get("string", "?")),
		String(ProjectSettings.get_setting("rendering/renderer/rendering_method", "?"))]
	# No rendering device under `--headless` (the dummy driver), so this is absent in every gate and
	# present in every real run.
	if RenderingServer.get_rendering_device() != null:
		line += " · " + RenderingServer.get_video_adapter_name()
	return line


## A compact machine-readable block for the feedback record. A dispatched fix that can see "this
## user has no ffmpeg" does not have to guess at a Masking bug report that is really a
## missing-dependency report.
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
static func install_hint(entry_row: Dictionary) -> String:
	var m: Dictionary = entry_row.get("install", {})
	return String(m.get(_platform(), ""))


## A one-sentence "here is what happens about it" for a table key, for the error message a mode
## shows when it discovers the gap itself - so a failure deep inside a mode says the same thing the
## home screen would have said. For what ghost installs, that is what ghost is doing about it.
static func hint(key: String) -> String:
	var t := entry(key)
	if t.is_empty():
		return ""
	if int(t.get("kind", KIND_TOOL)) != KIND_TOOL:
		return Provision.hint(key)
	var line := install_hint(t)
	if line.is_empty():
		line = String(t.get("site", ""))
	return "Install it with:  %s" % line if not line.is_empty() else ""


# --- probes ------------------------------------------------------------------

static func _probe_tool(t: Dictionary) -> Dictionary:
	var r := t.duplicate(true)
	r["kind"] = KIND_TOOL
	r["found"] = false
	r["path"] = ""
	r["version"] = ""
	r["note"] = ""
	for n in t.get("bins", []):
		var p := resolve(String(n))
		if p.is_empty():
			continue
		r["found"] = true
		r["path"] = p
		r["bin"] = String(n)
		# `no_version` is for programs that have no version flag at all (xvfb-run answers `--help`
		# with its usage). Present is the whole answer there.
		if not bool(t.get("no_version", false)):
			r["version"] = _version_of(p, t.get("version_args", ["--version"]))
		break
	if not r["found"]:
		r["note"] = "not found"
		return r
	if String(r["version"]).is_empty() and bool(t.get("no_version", false)):
		r["version"] = "installed"
	return r


## ghost's own copy when it has one; until then a copy on the machine stands in, and says so.
static func _probe_fetched(t: Dictionary) -> Dictionary:
	var r := t.duplicate(true)
	var key := String(t["key"])
	var first := String((t["provides"] as Array)[0])
	r["found"] = false
	r["path"] = ""
	r["version"] = ""
	r["note"] = ""
	r["own"] = false
	var why := Provision.unsupported(key)
	if Provision.tool_ok(key):
		r["found"] = true
		r["own"] = true
		r["path"] = Provision.tool_path(key, first)
		r["version"] = String(Provision.installed(key).get("version", ""))
		return r
	var local := _scan_for(first)
	if not local.is_empty():
		r["found"] = true
		r["path"] = local
		r["version"] = _version_of(local, t.get("version_args", ["--version"]))
		r["note"] = "this machine's copy until ghost's own downloads"
	else:
		r["note"] = why if not why.is_empty() else "downloads at launch"
	return r


static func _probe_managed(m: Dictionary) -> Dictionary:
	var r := m.duplicate(true)
	r["kind"] = KIND_MANAGED
	r["tier"] = int(m.get("tier", TIER_EXTRA))
	r["found"] = false
	r["version"] = ""
	r["path"] = ""
	var key := String(m.get("key", ""))
	var why := Provision.unsupported(key)
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
	if key == "python":
		var exe := Provision.python_exe()
		r["found"] = not exe.is_empty()
		r["path"] = exe
		r["version"] = _version_of(exe, ["--version"]) if not exe.is_empty() else ""
		r["note"] = "" if r["found"] else "downloads at launch"
		return r
	var root := ProjectSettings.globalize_path(String(m.get("path", "")))
	r["path"] = root
	r["found"] = Provision.env_ready(m)
	r["version"] = "ready" if r["found"] else ""
	# Short, because it shares a narrow column with version strings. The size and the path are in
	# the detail pane, which has the room.
	if not why.is_empty():
		r["note"] = "not available here"
	elif not r["found"] and FileAccess.file_exists(venv_bin(root, "python")):
		r["note"] = "rebuilt on next use"
	elif not r["found"]:
		r["note"] = "on first use"
	return r


## Run `bin` with `args` and pull a version out of what it says. Reads stderr too: some tools print
## their version there, and a tool that fails to run at all should surface as "no version" rather
## than as a hang or a stray console window.
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

## The filesystem half of resolution: every candidate name (with every Windows extension) against
## every search directory. No subprocess, so this is the part that is allowed to run on the hot path.
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


## The fallback, and only ever reached on a miss: ask the OS's own lookup, which can still know
## something the scan does not (a PATH entry behind a symlinked directory the scan normalized
## differently, a shim registered by an app store).
static func _ask_the_shell(prog: String) -> String:
	var out: Array = []
	var finder := "where" if _is_windows() else "which"
	if OS.execute(finder, [prog], out) != 0 or out.is_empty():
		return ""
	var p := String(out[0]).strip_edges().split("\n")[0].strip_edges()
	return p if FileAccess.file_exists(p) else ""


## "linux", "macos" or "windows".
static func platform() -> String:
	return _platform()


static func _platform() -> String:
	match OS.get_name():
		"Windows", "UWP": return "windows"
		"macOS": return "macos"
		_: return "linux"


static func _is_windows() -> bool:
	return _platform() == "windows"


## The user's home directory - `HOME`, or `USERPROFILE` on Windows, which has no `HOME`.
static func home() -> String:
	return _home()


static func _home() -> String:
	var h := OS.get_environment("HOME")
	if h.is_empty():
		h = OS.get_environment("USERPROFILE")
	return h


## Does this row apply here at all? A `setpriv` row on a Mac is noise, not a warning.
static func _on_this_platform(t: Dictionary) -> bool:
	var only: Array = t.get("platforms", [])
	return only.is_empty() or only.has(_platform())


static func _dir_files(path: String) -> PackedStringArray:
	var d := DirAccess.open(path)
	return d.get_files() if d != null else PackedStringArray()


## "3.9" < "3.10", which is the whole reason this is not a string compare.
static func version_lt(a: String, b: String) -> bool:
	var pa := a.split(".")
	var pb := b.split(".")
	for i in maxi(pa.size(), pb.size()):
		var x := int(pa[i]) if i < pa.size() else 0
		var y := int(pb[i]) if i < pb.size() else 0
		if x != y:
			return x < y
	return false
