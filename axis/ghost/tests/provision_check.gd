extends SceneTree

## Is ghost's own installer right about what to fetch, where it lands, and when an environment has
## to be rebuilt?
##
## The Provisioner's jobs are thin over [Provision]'s static half, and every way that half goes
## wrong is silent: a wheel picked for the wrong machine installs a binary that cannot run, a
## mis-parsed release reports a version that does not exist, an unpack that keeps a decoy file
## ships `ffmpeg.html` as ffmpeg, and an environment that is not recognized as stale keeps
## importing packages its requirements do not name. So each is checked against a fixture with a
## known answer, and the checks that guard a rule are two-sided: the wrong input must fail.
##
## No network, and nothing outside a scratch folder: release documents are canned, archives are
## built here with [ZIPPacker], and [member Provision.base_override] points every path at
## user://provision_check, which is removed at the end.
##
##   godot --headless --path axis/ghost --script tests/provision_check.gd

var _fails: Array = []
var _scratch := ""


func _init() -> void:
	_scratch = OS.get_user_data_dir().path_join("provision_check")
	Provision.remove_tree(_scratch)
	DirAccess.make_dir_recursive_absolute(_scratch)
	Provision.base_override = _scratch.path_join("data")
	_platforms()
	_pypi()
	_github()
	_riedl()
	_unpack()
	_installed()
	_resolver()
	_python()
	_envs()
	_table()
	_words()
	Provision.base_override = ""
	Deps.forget_all()
	Provision.remove_tree(_scratch)
	if _fails.is_empty():
		print("provision_check: ALL OK")
		quit(0)
	else:
		print("provision_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
		quit(1)


func _check(ok: bool, msg: String) -> void:
	print(("   ok   " if ok else "   FAIL ") + msg)
	if not ok:
		_fails.append(msg)


# --- this machine ----------------------------------------------------------------

func _platforms() -> void:
	print("-- platforms")
	var key := Provision.platform_key()
	_check(RegEx.create_from_string("^(linux|macos|windows)-[a-z0-9_]+$").search(key) != null,
		"this machine is named as <os>-<arch> (%s)" % key)
	for plat in Provision.PLATFORMS:
		_check(Provision.WHEEL_TAGS.has(plat), "%s has a wheel tag pattern" % plat)


# --- releases --------------------------------------------------------------------

const PYPI_UV := {
	"info": {"name": "uv", "version": "0.12.23"},
	"urls": [
		{"filename": "uv-0.12.23-py3-none-musllinux_1_1_x86_64.whl", "url": "U-musl",
			"digests": {"sha256": "11"}, "size": 1},
		{"filename": "uv-0.12.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
			"url": "U-linux", "digests": {"sha256": "AB"}, "size": 2},
		{"filename": "uv-0.12.23-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
			"url": "U-linux-arm", "digests": {"sha256": "22"}, "size": 3},
		{"filename": "uv-0.12.23-py3-none-macosx_11_0_arm64.whl", "url": "U-mac-arm",
			"digests": {"sha256": "33"}, "size": 4},
		{"filename": "uv-0.12.23-py3-none-macosx_10_12_x86_64.whl", "url": "U-mac-x86",
			"digests": {"sha256": "44"}, "size": 5},
		{"filename": "uv-0.12.23-py3-none-win_amd64.whl", "url": "U-win", "yanked": true,
			"digests": {"sha256": "55"}, "size": 6},
		{"filename": "uv-0.12.23-py3-none-win_arm64.whl", "url": "U-win-arm",
			"digests": {"sha256": "66"}, "size": 7},
		{"filename": "uv-0.12.23.tar.gz", "url": "U-sdist", "digests": {"sha256": "77"}, "size": 8},
	],
}


func _pypi() -> void:
	print("-- PyPI: the wheel for each machine")
	var want := {"linux-x86_64": "U-linux", "linux-arm64": "U-linux-arm",
		"macos-arm64": "U-mac-arm", "macos-x86_64": "U-mac-x86", "windows-arm64": "U-win-arm"}
	for plat in want:
		var got := Provision.pick_pypi(PYPI_UV, plat)
		var files: Array = got.get("files", [])
		_check(not files.is_empty() and String(files[0]["url"]) == want[plat],
			"%s gets %s (got %s)" % [plat, want[plat], files[0]["url"] if not files.is_empty() else got])
	var linux := Provision.pick_pypi(PYPI_UV, "linux-x86_64")
	_check(String(linux.get("version", "")) == "0.12.23", "the version is the release's")
	_check(String(linux["files"][0]["sha256"]) == "ab", "and its checksum is lower-cased")
	# A glibc machine must not get the musl build that happens to be listed first.
	_check(String(linux["files"][0]["url"]) != "U-musl", "a musllinux wheel is not a manylinux one")
	# The only win_amd64 wheel is yanked: that is no build, not a build.
	_check(Provision.pick_pypi(PYPI_UV, "windows-x86_64").has("error"),
		"a yanked wheel is never picked")
	_check(Provision.pick_pypi(PYPI_UV, "plan9-mips").has("error"), "an unknown machine gets an error")
	_check(Provision.pick_pypi({"info": {}, "urls": []}, "linux-x86_64").has("error"),
		"a document without a version gets an error")


const GITHUB_BTBN := {
	"tag_name": "latest",
	"html_url": "https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest",
	"assets": [
		{"name": "ffmpeg-master-latest-winarm64-gpl.zip", "digest": "sha256:aa",
			"browser_download_url": "B-master", "size": 1},
		{"name": "ffmpeg-n8.1-latest-winarm64-gpl-8.1.zip", "digest": "sha256:bb",
			"browser_download_url": "B-81", "size": 2},
		{"name": "ffmpeg-n9.0-latest-winarm64-gpl-9.0.zip", "digest": "sha256:CC",
			"browser_download_url": "B-90", "size": 3},
		{"name": "ffmpeg-n9.0-latest-winarm64-lgpl-9.0.zip", "digest": "sha256:dd",
			"browser_download_url": "B-90-lgpl", "size": 4},
		{"name": "ffmpeg-n9.1-latest-winarm64-gpl-9.1.zip", "browser_download_url": "B-91-nodigest",
			"size": 5},
	],
}

const GITHUB_GYAN := {
	"tag_name": "9.0.2",
	"assets": [
		{"name": "ffmpeg-9.0.2-full_build.zip", "digest": "sha256:ee",
			"browser_download_url": "G-full", "size": 1},
		{"name": "ffmpeg-9.0.2-essentials_build.7z", "digest": "sha256:ff",
			"browser_download_url": "G-7z", "size": 2},
		{"name": "ffmpeg-9.0.2-essentials_build.zip", "digest": "sha256:0a",
			"browser_download_url": "G-ess", "size": 3},
	],
}


func _github() -> void:
	print("-- GitHub releases")
	var ffmpeg := Deps.entry("ffmpeg")
	var arm := Provision.pick_github(GITHUB_BTBN, String(Provision.source(ffmpeg, "windows-arm64")["asset"]))
	_check(String(arm.get("version", "")) == "9.0", "BtbN: the highest FFmpeg branch wins (9.0, got %s)"
		% arm.get("version", arm))
	_check(String(arm["files"][0]["url"]) == "B-90", "and it is the GPL build, not the LGPL one")
	_check(String(arm["files"][0]["sha256"]) == "cc", "with GitHub's digest, prefix removed and lower-cased")
	var x64 := Provision.pick_github(GITHUB_GYAN, String(Provision.source(ffmpeg, "windows-x86_64")["asset"]))
	_check(String(x64.get("version", "")) == "9.0.2", "Gyan: the release tag is the version")
	_check(String(x64["files"][0]["url"]) == "G-ess", "and the essentials zip is the asset (not .7z, not full)")
	_check(Provision.pick_github(GITHUB_GYAN, "^nothing-matches$").has("error"),
		"a release with no matching asset is an error")


func _riedl() -> void:
	print("-- Martin Riedl's builds")
	var at := Provision.parse_riedl("/download/linux/amd64/1789931100_9.0.2/ffmpeg.zip")
	_check(String(at.get("version", "")) == "9.0.2", "the version is read from the redirect")
	_check(String(at.get("url", "")) == Provision.RIEDL + "/download/linux/amd64/1789931100_9.0.2/ffmpeg.zip",
		"and the download is that path on the build server")
	var absolute := Provision.parse_riedl(Provision.RIEDL + "/download/macos/arm64/1789931890_9.0.2/ffprobe.zip")
	_check(String(absolute.get("version", "")) == "9.0.2", "an absolute location parses the same")
	_check(Provision.parse_riedl("/download/linux/amd64/ffmpeg.zip").has("error"),
		"a location without a build and version is an error")
	_check(Provision.riedl_url("linux", "arm64", "ffprobe").ends_with("/redirect/latest/linux/arm64/release/ffprobe.zip"),
		"the release link asks for the release channel")
	var sha := "FA8ECF4ABBD290D98F7D188B8649CC6B391AE209A98452BE955A15AAB1909D7F"
	_check(Provision.parse_sha256(sha + "  ffmpeg.zip\n") == sha.to_lower(), "a checksum file reads as its hash")
	_check(Provision.parse_sha256("<html>404</html>").is_empty(), "a page without a hash reads as none")


# --- unpacking -------------------------------------------------------------------

func _zip(path: String, members: Dictionary) -> void:
	var zp := ZIPPacker.new()
	zp.open(path)
	for name in members:
		zp.start_file(String(name))
		zp.write_file(String(members[name]).to_utf8_buffer())
		zp.close_file()
	zp.close()


func _unpack() -> void:
	print("-- unpacking")
	var dir := _scratch.path_join("zips")
	DirAccess.make_dir_recursive_absolute(dir)
	# Windows: both programs in bin/, beside a third and a decoy that shares a name.
	var win := dir.path_join("win.zip")
	_zip(win, {"pkg/bin/ffmpeg.exe": "FFMPEG", "pkg/bin/ffprobe.exe": "FFPROBE",
		"pkg/bin/ffplay.exe": "FFPLAY", "pkg/doc/ffmpeg.html": "DOC"})
	var dest := dir.path_join("out-win")
	var res := Provision.unpack(PackedStringArray([win]), PackedStringArray(["ffmpeg", "ffprobe"]),
		dest, true)
	_check(bool(res.get("ok", false)), "a Windows build unpacks (%s)" % res.get("error", ""))
	_check(FileAccess.get_file_as_string(dest.path_join("ffmpeg.exe")) == "FFMPEG",
		"ffmpeg.exe is the program, not the page beside it")
	_check(not FileAccess.file_exists(dest.path_join("ffplay.exe")), "a program nobody asked for stays behind")
	_check(not DirAccess.dir_exists_absolute(dest + ".staging"), "no staging directory is left")

	# Linux and macOS: one zip per program, each at its root.
	var a := dir.path_join("ffmpeg.zip")
	var b := dir.path_join("ffprobe.zip")
	_zip(a, {"ffmpeg": "ELF-FFMPEG"})
	_zip(b, {"ffprobe": "ELF-FFPROBE"})
	var unix := dir.path_join("out-unix")
	res = Provision.unpack(PackedStringArray([a, b]), PackedStringArray(["ffmpeg", "ffprobe"]), unix, false)
	_check(bool(res.get("ok", false)) and (res["files"] as Dictionary).size() == 2,
		"two archives make one install")
	if OS.get_name() != "Windows":
		var perms := FileAccess.get_unix_permissions(unix.path_join("ffprobe"))
		_check(perms & FileAccess.UNIX_EXECUTE_OWNER != 0, "an unpacked program is executable")

	# A wheel: the program sits under <name>.data/scripts/, beside a package of the same name.
	var whl := dir.path_join("uv.whl")
	_zip(whl, {"uv/__init__.py": "PY", "uv-0.12.23.data/scripts/uv": "UV-BIN",
		"uv-0.12.23.data/scripts/uvx": "UVX"})
	var uv := dir.path_join("out-uv")
	res = Provision.unpack(PackedStringArray([whl]), PackedStringArray(["uv"]), uv, false)
	_check(FileAccess.get_file_as_string(uv.path_join("uv")) == "UV-BIN", "a wheel's script is the program")

	# Missing a program: nothing lands at all, and the staging is cleared.
	var short := dir.path_join("out-short")
	res = Provision.unpack(PackedStringArray([a]), PackedStringArray(["ffmpeg", "ffprobe"]), short, false)
	_check(not bool(res.get("ok", true)), "a download missing a program is refused")
	_check(not DirAccess.dir_exists_absolute(short) and not DirAccess.dir_exists_absolute(short + ".staging"),
		"and leaves nothing behind")
	res = Provision.unpack(PackedStringArray([dir.path_join("not-there.zip")]),
		PackedStringArray(["ffmpeg"]), dir.path_join("out-none"), false)
	_check(not bool(res.get("ok", true)), "a file that is not a zip is refused")


# --- what is installed -----------------------------------------------------------

func _fake_tool(key: String, version: String, programs: Array) -> void:
	var dir := Provision.tools_dir().path_join(key).path_join(version)
	DirAccess.make_dir_recursive_absolute(dir)
	var files := {}
	for p in programs:
		var name := String(p) + (".exe" if OS.get_name() == "Windows" else "")
		var f := FileAccess.open(dir.path_join(name), FileAccess.WRITE)
		f.store_string("#!/bin/sh\n")
		f.close()
		if OS.get_name() != "Windows":
			FileAccess.set_unix_permissions(dir.path_join(name), FileAccess.UNIX_READ_OWNER
				| FileAccess.UNIX_WRITE_OWNER | FileAccess.UNIX_EXECUTE_OWNER)
		files[String(p)] = name
	Provision.write_installed(key, {"version": version, "dir": version, "files": files})


func _installed() -> void:
	print("-- what is installed")
	_check(Provision.installed("ffmpeg").is_empty(), "nothing is installed in a fresh folder")
	_check(not Provision.tool_ok("ffmpeg"), "so FFmpeg is not ok")
	_fake_tool("ffmpeg", "9.0.2", ["ffmpeg", "ffprobe"])
	_check(String(Provision.installed("ffmpeg").get("version", "")) == "9.0.2", "the manifest reads back")
	_check(Provision.tool_ok("ffmpeg"), "FFmpeg is ok with both programs")
	DirAccess.remove_absolute(Provision.tool_path("ffmpeg", "ffprobe"))
	_check(not Provision.tool_ok("ffmpeg"), "and is not with one of them gone")
	_fake_tool("ffmpeg", "9.0.2", ["ffmpeg", "ffprobe"])

	var old := Provision.tools_dir().path_join("ffmpeg").path_join("8.1.3")
	DirAccess.make_dir_recursive_absolute(old)
	Provision.prune("ffmpeg")
	_check(not DirAccess.dir_exists_absolute(old), "prune removes a version nothing uses")
	_check(Provision.tool_ok("ffmpeg"), "and keeps the one in use")

	# remove_tree must take a link as a link: following it would delete what it points at.
	if OS.get_name() != "Windows":
		var outside := _scratch.path_join("outside")
		DirAccess.make_dir_recursive_absolute(outside)
		var keep := FileAccess.open(outside.path_join("keep.txt"), FileAccess.WRITE)
		keep.store_string("keep")
		keep.close()
		var tree := _scratch.path_join("tree")
		DirAccess.make_dir_recursive_absolute(tree)
		DirAccess.open(tree).create_link(outside, tree.path_join("link"))
		Provision.remove_tree(tree)
		_check(not DirAccess.dir_exists_absolute(tree), "remove_tree removes the tree")
		_check(FileAccess.file_exists(outside.path_join("keep.txt")),
			"but not what a link inside it points at")


func _resolver() -> void:
	print("-- resolution prefers ghost's own copy")
	Deps.forget_all()
	var own := Provision.tool_path("ffmpeg", "ffmpeg")
	_check(not own.is_empty() and Deps.resolve("ffmpeg") == own,
		"ffmpeg resolves to ghost's own copy (%s)" % Deps.resolve("ffmpeg"))
	_check(Deps.fetched_key("ffprobe") == "ffmpeg", "ffprobe is FFmpeg's")
	_check(Deps.fetched_key("setpriv").is_empty(), "setpriv is nobody's to fetch")
	DirAccess.remove_absolute(Provision.tools_dir().path_join("ffmpeg/current.json"))
	Deps.forget_all()
	_check(Deps.resolve("ffmpeg") != own, "and without it, falls back to the machine's (%s)"
		% Deps.resolve("ffmpeg"))
	Deps.forget_all()


# --- Python and the environments -------------------------------------------------

func _touch(path: String, text := "") -> void:
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()


func _python() -> void:
	print("-- ghost's own Python")
	if OS.get_name() == "Windows":
		print("   (layout checks are for Linux and macOS)")
		return
	_check(Provision.python_exe().is_empty(), "no Python before uv installs one")
	var patch := Provision.python_dir().path_join("cpython-%s.9-linux-x86_64-gnu" % Provision.PYTHON)
	_touch(patch.path_join("bin/python" + Provision.PYTHON))
	_check(Provision.python_exe().begins_with(patch), "a patch directory alone is found")
	var minor := Provision.python_dir().path_join("cpython-%s-linux-x86_64-gnu" % Provision.PYTHON)
	_touch(minor.path_join("bin/python" + Provision.PYTHON))
	_check(Provision.python_exe().begins_with(minor),
		"the minor-version link wins, so environments follow patch upgrades")
	var other := Provision.python_dir().path_join("cpython-3.13-linux-x86_64-gnu")
	_touch(other.path_join("bin/python3.13"))
	_check(not Provision.python_exe().begins_with(other), "another minor is never taken")


func _envs() -> void:
	print("-- when an environment must be rebuilt")
	var venv := _scratch.path_join("venv")
	var row := {"key": "check_env", "path": venv, "packages": ["example>=1"], "imports": "sys"}
	_touch(Deps.venv_bin(venv, "python"))
	var home := Provision.python_dir().path_join("cpython-%s-linux-x86_64-gnu/bin" % Provision.PYTHON)
	_touch(venv.path_join("pyvenv.cfg"), "home = %s\nimplementation = CPython\nuv = 0.12.23\n" % home)
	_check(Provision.uv_built(venv), "a uv environment on ghost's Python is recognized")
	_check(not Provision.env_ready(row), "but without a stamp it is not ready")
	Provision.write_stamp(row)
	_check(Provision.env_ready(row), "stamped, it is")
	var changed := row.duplicate(true)
	changed["packages"] = ["example>=2"]
	_check(not Provision.env_ready(changed), "a changed requirement makes it stale")
	var with_file := row.duplicate(true)
	with_file["files"] = [{"key": "check_model", "file": "check.task", "sha256": "00"}]
	_check(not Provision.env_ready(with_file), "a data file it needs and does not have makes it stale")

	_touch(venv.path_join("pyvenv.cfg"), "home = /usr/bin\nimplementation = CPython\nversion_info = 3.14\n")
	_check(not Provision.uv_built(venv), "an environment made by `python -m venv` is not uv's")
	_check(not Provision.env_ready(row), "so it is rebuilt, stamp or not")
	_touch(venv.path_join("pyvenv.cfg"), "home = /usr/bin\nuv = 0.12.23\n")
	_check(not Provision.uv_built(venv), "and neither is one on the system's Python")


# --- the table -------------------------------------------------------------------

func _table() -> void:
	print("-- the tables say enough to install from")
	for t in Deps.FETCHED:
		var key := String(t["key"])
		_check(not (t.get("provides", []) as Array).is_empty(), "%s names what it provides" % key)
		for plat in Provision.PLATFORMS:
			var why: Dictionary = t.get("unsupported", {})
			var src := Provision.source(t, plat)
			_check(why.has(plat) or ["pypi", "github", "riedl"].has(String(src.get("via", ""))),
				"%s has a source for %s" % [key, plat])
	for m in Deps.MANAGED:
		var key := String(m["key"])
		var why: Dictionary = m.get("unsupported", {})
		for plat in why:
			_check(Provision.PLATFORMS.has(plat), "%s's exception %s is a real machine" % [key, plat])
		if not (m.has("requirements") or m.has("packages")):
			continue
		_check(String(m.get("path", "")).begins_with("user://"), "%s lives under user://" % key)
		_check(not String(m.get("imports", "")).is_empty(), "%s says what to import to check it" % key)
		var req := String(m.get("requirements", ""))
		_check(req.is_empty() or FileAccess.file_exists(req), "%s's requirements file exists" % key)
		for p in m.get("packages", []):
			# argv on Windows is quoted, never escaped: a package spec may not carry a quote.
			_check(not String(p).contains("\""), "%s's package %s has no quote in it" % [key, p])
		for f in m.get("files", []):
			var file: Dictionary = f
			_check(String(file.get("url", "")).begins_with("https://"), "%s fetches over https" % file.get("key", ""))
			_check(RegEx.create_from_string("^[0-9a-f]{64}$").search(String(file.get("sha256", ""))) != null,
				"%s has a pinned checksum" % file.get("key", ""))


# --- what features say -----------------------------------------------------------

func _words() -> void:
	print("-- what a feature tells the user")
	var running := {"running": true, "phase": "downloading 9.0.2", "fraction": 0.5,
		"done": 50 * 1048576, "total": 100 * 1048576, "started": Time.get_ticks_msec()}
	var line := Provision.describe("ffmpeg", running)
	_check(line.contains("FFmpeg") and line.contains("downloading 9.0.2") and line.contains("50.0 MB / 100.0 MB"),
		"a download says what, which version and how much (%s)" % line)
	var failed := {"running": false, "ready": false, "error": "no connection"}
	_check(Provision.describe("ffmpeg", failed).contains("no connection"), "a failure says why")
	_check(Provision.describe("ffmpeg", {"ready": true}).is_empty(), "a ready thing says nothing")
