extends RefCounted
class_name Subprocess

## Subprocess - every external program ghost starts, in one place, with one promise:
## A CHILD NEVER OUTLIVES THE APP THAT STARTED IT.
##
## WHY THIS EXISTS. `OS.create_process` on Unix is fully detached: the child gets no
## link back to its parent, so closing (or killing) ghost leaves it running. That is not
## hypothetical - it was reported from the export path, where the render window and the
## app window were both shut and `ffmpeg` carried on transcoding to disk with nothing on
## screen to show it, no way to stop it, and no "godot" in `ps` to explain it. The same
## hole exists for every other program the app starts: the bake process, the Movie Maker
## render itself, the mask editor's prep/waveform/import passes, the voice host.
##
## `mask_editor` already killed its OWN pids from `_exit_tree`, and that pattern is the
## reason this file exists rather than another copy of it: a per-owner list is a list
## somebody must remember to extend, and the exporter - the one that got reported - never
## had one at all. Going through `start()` is what makes a new call site covered by
## default instead of covered if remembered.
##
## TWO INDEPENDENT MECHANISMS, because they fail in different ways:
##
##   1. THE REGISTRY (everywhere). Every pid started here is remembered until it exits or
##      is stopped, and [method reap_all] kills whatever is left. `Boot` calls it when the
##      app closes, so a graceful quit takes every child with it even if the owning mode
##      forgot to clean up. This covers the ordinary case and it is portable.
##
##   2. THE DEATH PACT (Linux, when available). A registry cannot help if the app is
##      SIGKILLed - no shutdown code runs at all. That is not an exotic case here: ghost's
##      OWN quit ends in `OS.kill(OS.get_process_id())` (see main._shutdown, a workaround
##      for an audio-driver teardown crash), so every ordinary close is a SIGKILL and no
##      autoload `_exit_tree` and no engine child-reaping ever runs. `setpriv --pdeathsig KILL` sets the
##      child's PR_SET_PDEATHSIG before exec'ing the real program, so the KERNEL kills it
##      the moment the parent dies, however the parent died. Verified both ways: with the
##      pact the child is gone within a second of `kill -9` on the Godot process; without
##      it the same child is still running. `setpriv` execs, so the pid we get back IS the
##      real program's - callers keep polling it and reading its progress file exactly as
##      before.
##
## THE PACT IS PER-THREAD, which is the one caveat worth knowing: PR_SET_PDEATHSIG fires
## when the parent THREAD exits, not the parent process, so a child must be started from a
## thread that lives as long as the app. Every caller here starts from the main thread (UI
## callbacks and `_process`), which is exactly that thread. Do not call `start()` from a
## WorkerThreadPool task without thinking about this; use [method start_detached] there.
##
## DELIBERATE EXCEPTIONS use [method start_detached] and say why at the call site. There is
## one: `assistant.gd` dispatches `claude -p` runs that write straight to the working tree
## and are MEANT to finish after the editor closes.
##
## NO SHELL IS ASSUMED. `/bin/bash` does not exist on Windows, and seven call sites used to
## reach for it just to redirect output. Two portable forms replace it:
##   [method start_logged] - stdout+stderr into one log file, pumped by this process. Pure
##     Godot, so it behaves identically on every platform and Linux exercises the same path
##     Windows runs.
##   [method start_redirected] - a working directory, stdin from a file, stdout and stderr to
##     separate files, for runs that must not hold a pipe to this process (the assistant's
##     detached runs) or must see stdin closed (`codex exec` waits on an open one). `/bin/sh`
##     on Linux and macOS, PowerShell on Windows.
## FREE TEXT NEVER GOES IN ARGV. Godot's Windows command-line quoting wraps an argument in
## quotes without escaping the quotes inside it, so a prompt containing `"` arrives split.
## Prompts travel through [method start_redirected]'s stdin file instead.

## pid -> a short label, kept only so the shutdown reap can say what it killed.
static var _tracked := {}
## Resolved once: "" not looked up yet, "-" no pact available, otherwise the setpriv path.
static var _pact := ""
## Programs already reported missing, so a per-frame caller complains once, not 60x/s.
static var _warned := {}
## pid -> {threads, exited_ms}: the reader threads behind a [method start_logged] child.
## [method alive] keeps answering true until they have drained the pipes, so a caller that
## reads the log the moment the child is gone reads all of it.
static var _pumps := {}
## Reader threads no longer tied to a live pid, joined once they finish.
static var _stray: Array[Thread] = []
## How long a finished child's readers may keep draining before [method alive] stops
## waiting for them - a grandchild that inherited the pipe can hold it open indefinitely.
const _DRAIN_MS := 2000

## The Windows half of [method start_redirected]. Fixed text, written once per run into
## user://; everything variable arrives in a JSON spec file, so nothing the caller passes is
## ever parsed as PowerShell. Arguments are re-quoted with the CommandLineToArgvW rules
## (backslashes before a quote doubled, the quote escaped), which Godot's own quoting skips.
## Output files are opened unbuffered and share-readable, so a caller tailing them sees
## each event as it is written.
const _PS_LAUNCHER := """param([string]$SpecPath)
$ErrorActionPreference = 'Stop'
$s = [IO.File]::ReadAllText($SpecPath, [Text.Encoding]::UTF8) | ConvertFrom-Json
Remove-Item -LiteralPath $SpecPath -ErrorAction SilentlyContinue
function Format-Arg([string]$a) {
	if ($a.Length -gt 0 -and $a -notmatch '[\\s"]') { return $a }
	$sb = New-Object System.Text.StringBuilder
	[void]$sb.Append([char]'"')
	$bs = 0
	foreach ($c in $a.ToCharArray()) {
		if ($c -eq [char]'\\') { $bs++; continue }
		if ($c -eq [char]'"') { [void]$sb.Append([char]'\\', 2 * $bs + 1) }
		elseif ($bs -gt 0) { [void]$sb.Append([char]'\\', $bs) }
		[void]$sb.Append($c)
		$bs = 0
	}
	[void]$sb.Append([char]'\\', 2 * $bs)
	[void]$sb.Append([char]'"')
	return $sb.ToString()
}
function Open-Sink([string]$p) {
	return New-Object System.IO.FileStream($p, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite, 1)
}
$out = Open-Sink $s.out
$err = Open-Sink $s.err
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $s.program
$psi.Arguments = (@($s.args) | ForEach-Object { Format-Arg ([string]$_) }) -join ' '
if ($s.cwd) { $psi.WorkingDirectory = $s.cwd }
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
try { $p = [System.Diagnostics.Process]::Start($psi) } catch {
	$b = [Text.Encoding]::UTF8.GetBytes("ghost launcher: " + $_.Exception.Message + "`n")
	$err.Write($b, 0, $b.Length); $out.Close(); $err.Close(); exit 127
}
$to = $p.StandardOutput.BaseStream.CopyToAsync($out)
$te = $p.StandardError.BaseStream.CopyToAsync($err)
if ($s.stdin) {
	$in = [IO.File]::OpenRead($s.stdin)
	try { $in.CopyTo($p.StandardInput.BaseStream) } catch {}
	$in.Close()
}
try { $p.StandardInput.Close() } catch {}
$p.WaitForExit()
try { [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($to, $te)) } catch {}
$out.Close(); $err.Close()
exit $p.ExitCode
"""

## The Unix half: `cd`, then exec with the redirects, so the pid the caller holds IS the
## program's (and a `setpriv` pact on the shell carries over the exec). Every value is a
## positional parameter; nothing is interpolated into the script.
const _SH_LAUNCHER := "cd \"$1\" || exit 1; in=\"$2\"; out=\"$3\"; err=\"$4\"; shift 4; " \
	+ "exec \"$@\" < \"$in\" > \"$out\" 2> \"$err\""


## Start `path` with `args`, bound to this process, and return its pid (<= 0 on failure).
## `tag` is a human label for the shutdown log; it defaults to the program's own name.
static func start(path: String, args: PackedStringArray, tag := "") -> int:
	var prog := _program(path)
	if prog.is_empty():
		return -1
	var bin := _pact_bin()
	var pid := -1
	if bin == "-":
		pid = OS.create_process(prog, args)
	else:
		var full := PackedStringArray(["--pdeathsig", "KILL", "--", prog])
		full.append_array(args)
		pid = OS.create_process(bin, full)
	if pid > 0:
		_tracked[pid] = tag if tag != "" else path.get_file()
	return pid


## Start `path` with `args` and keep a pipe to its stdio - the `OS.execute_with_pipe` form,
## bound and registered exactly like [method start]. Returns that call's Dictionary, empty on
## failure. `setpriv` execs the real program, so the returned pid and the pipe are the real
## program's; nothing about the caller's protocol changes.
static func start_with_pipe(path: String, args: PackedStringArray, tag := "") -> Dictionary:
	var prog := _program(path)
	if prog.is_empty():
		return {}
	var bin := _pact_bin()
	var info := {}
	if bin == "-":
		info = OS.execute_with_pipe(prog, args)
	else:
		var full := PackedStringArray(["--pdeathsig", "KILL", "--", prog])
		full.append_array(args)
		info = OS.execute_with_pipe(bin, full)
	var pid := int(info.get("pid", -1))
	if pid > 0:
		_tracked[pid] = tag if tag != "" else path.get_file()
	return info


## Start a program that is SUPPOSED to outlive the app. Not tracked, not bound; the call
## site must justify itself in a comment, because this is the behaviour every reported
## orphan came from.
static func start_detached(path: String, args: PackedStringArray) -> int:
	var prog := _program(path)
	return OS.create_process(prog, args) if not prog.is_empty() else -1


## Start `path` with `args`, stdout and stderr both written to `log_path` (truncated first),
## bound and registered like [method start]. Returns the pid, <= 0 on failure.
##
## One reader thread per stream, because a single blocking reader deadlocks the moment the
## child fills the pipe it is not reading. Both append to the log under one lock and flush
## per chunk, so a caller tailing the file each frame (yt-dlp's progress line) sees it live.
## The child's stdin is a pipe this process never writes: fine for pip, yt-dlp, python and a
## headless Godot, wrong for anything that reads stdin (use [method start_redirected]).
static func start_logged(path: String, args: PackedStringArray, log_path: String, tag := "") -> int:
	_join_finished_pumps()
	var info := start_with_pipe(path, args, tag)
	if info.is_empty():
		return -1
	# A log that cannot be opened still gets drained - an unread pipe stalls the child.
	var sink := FileAccess.open(log_path, FileAccess.WRITE)
	var lock := Mutex.new()
	var threads: Array[Thread] = []
	for key in ["stdio", "stderr"]:
		var pipe: FileAccess = info.get(key)
		if pipe == null:
			continue
		var t := Thread.new()
		t.start(_pump.bind(pipe, sink, lock))
		threads.append(t)
	var pid := int(info.get("pid", -1))
	_pumps[pid] = {"threads": threads, "exited_ms": 0}
	return pid


## Start `path` with `args` with its stdio wired to files. `io` keys, all optional:
##   cwd   - working directory ("" keeps this process's)
##   stdin - a file whose bytes are the child's whole stdin ("" = empty, closed at once)
##   out, err - where stdout and stderr go; must be two different files
## Registered and bound like [method start]; with `detached` true it is neither, exactly as
## [method start_detached]. Returns the pid, <= 0 on failure.
##
## On Linux and macOS the pid is the program's own (`sh` execs it). On Windows it is the
## PowerShell launcher's, which lives exactly as long as the program and is stopped with
## its whole tree by [method terminate].
static func start_redirected(path: String, args: PackedStringArray, io: Dictionary,
		tag := "", detached := false) -> int:
	var prog := _program(path)
	if prog.is_empty():
		return -1
	var out := String(io.get("out", ""))
	var err := String(io.get("err", ""))
	if out.is_empty() or err.is_empty() or out == err:
		push_error("Subprocess.start_redirected: `out` and `err` must be two different files")
		return -1
	var shell := ""
	var full := PackedStringArray()
	if _is_windows():
		shell = _powershell()
		var spec := _write_ps_spec({"program": prog, "args": Array(args),
			"cwd": String(io.get("cwd", "")), "stdin": String(io.get("stdin", "")),
			"out": out, "err": err})
		if shell.is_empty() or spec.is_empty():
			return -1
		full = PackedStringArray(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
			"-File", _ps_launcher_path(), spec])
	else:
		shell = "/bin/sh"
		var cwd := String(io.get("cwd", ""))
		var stdin := String(io.get("stdin", ""))
		full = PackedStringArray(["-c", _SH_LAUNCHER, "sh",
			cwd if not cwd.is_empty() else ".",
			stdin if not stdin.is_empty() else "/dev/null", out, err, prog])
		full.append_array(args)
	if detached:
		return start_detached(shell, full)
	return start(shell, full, tag if tag != "" else path.get_file())


## Kill `pid` - and on Windows its whole process tree, since a [method start_redirected]
## child there sits under a PowerShell launcher and killing the launcher alone would orphan
## it. Not registry-checked: [method stop] is the guarded form, this is for pids the caller
## owns outright (a detached assistant run).
static func terminate(pid: int) -> void:
	if pid <= 0:
		return
	if _is_windows():
		var tk := OS.get_environment("SystemRoot").path_join("System32/taskkill.exe")
		if FileAccess.file_exists(tk) and OS.execute(tk, ["/PID", str(pid), "/T", "/F"]) == 0:
			return
	OS.kill(pid)


## EVERY child goes through [Deps] first, so a bare "ffmpeg" becomes an absolute path
## before the kernel sees it. That is not a convenience: a GUI-launched app does not
## inherit a shell's PATH, so on macOS a Homebrew ffmpeg is invisible to a bare name
## and the child simply never starts. Returns "" - and says why, once - when the
## program is not installed at all, which used to present as an unexplained pid of -1
## somewhere far from the cause.
static func _program(path: String) -> String:
	var bin := Deps.resolve(path)
	if bin.is_empty() and not _warned.has(path):
		_warned[path] = true
		push_warning("ghost: '%s' is not installed (or not on PATH) - "
			% path + "see the Environment panel on the home screen")
		printerr("ghost: cannot start '%s' - not found on this machine" % path)
	return bin


## THE REGISTRY IS THE AUTHORITY ON WHETHER A PID IS OURS TO ASK ABOUT, and both calls below
## check it BEFORE they touch the OS. That is not tidiness, it is the fix for an error printed
## on every single close:
##
##     ghost: stopped 1 background process(es) on exit: voice host
##     ERROR: The process 1199699 does not exist or is not a child of the calling process.
##        at: _check_pid_is_running (drivers/unix/os_unix.cpp:863)
##        [0] stop (res://scripts/subprocess.gd) [1] stop (voice_host.gd) [2] _exit_tree
##
## Read the order: `Boot` reaps the registry, and THEN the tree tears down and the owning
## autoload's `_exit_tree` stops its own pid a second time. `stop()` asked
## `OS.is_process_running` first, and that call `waitpid()`s a child Godot has already
## reaped - ECHILD, which the Unix driver reports as an error. It is exactly the race
## [method reap_all] documents refusing to run into, and it was still in these two.
##
## So: a pid in `_tracked` is one we started and have not stopped, reaped or seen exit, and it
## is the only kind either call will hand to the kernel. Anything else - never ours, already
## reaped, already exited, a stale variable a mode's shutdown list still holds - answers from
## the registry alone and costs nothing. `kill()` on a tracked child that has since exited on
## its own stays harmless (it is a signal, not a wait), which is why `stop` needs no liveness
## question at all.

## Is this child of ours still running? Forgets it once it is not, so the registry stays tight
## and a shutdown reap can never signal a pid that has been recycled since. Use this in the
## polling loops that used to call `OS.is_process_running` directly. A pid we did not start (or
## have already forgotten) is not "alive" as far as this registry is concerned - including a
## `start_detached` child, which is deliberately nobody's to track.
static func alive(pid: int) -> bool:
	if pid <= 0 or not _tracked.has(pid):
		return false
	if OS.is_process_running(pid) or _draining(pid):
		return true
	_tracked.erase(pid)
	return false


## Kill one child now and forget it. Returns whether it actually signalled anything, so a
## second stop - after a reap, or on a pid a mode's shutdown list still holds from last time -
## is visibly a no-op rather than an error in the log.
static func stop(pid: int) -> bool:
	if pid <= 0 or not _tracked.has(pid):
		return false
	terminate(pid)
	_tracked.erase(pid)
	_release_pumps(pid)
	return true


## Forget a pid WITHOUT killing it - for a child deliberately handed off. Rare.
static func forget(pid: int) -> void:
	_tracked.erase(pid)


## Kill every child still registered. Called from `Boot` when the app closes; safe to call
## more than once. Deliberately does NOT call `OS.is_process_running` first: at shutdown
## Godot is reaping its own children on another path, and a same-frame
## `is_process_running` on the same pid races that reap into ECHILD - the crash-on-close
## `assistant.gd` documents. `kill()` on an already-dead child is harmless.
static func reap_all() -> int:
	var n := 0
	for pid in _tracked.keys():
		if int(pid) > 0:
			terminate(int(pid))
			n += 1
	if n > 0:
		print("ghost: stopped %d background process(es) on exit: %s"
			% [n, ", ".join(PackedStringArray(_tracked.values()))])
	_tracked.clear()
	for pid in _pumps.keys():
		_release_pumps(int(pid))
	_join_finished_pumps()
	return n


## How many children are currently registered - for status lines and gates.
static func tracked() -> int:
	return _tracked.size()


# One stream of a [method start_logged] child, copied into the shared log until EOF. A
# blocking read returns whatever is available, and nothing only at EOF - or, rarely, when
# a signal interrupts it, which is why one empty read is not believed until it repeats.
static func _pump(pipe: FileAccess, sink: FileAccess, lock: Mutex) -> void:
	var empty := 0
	while empty < 3:
		var chunk := pipe.get_buffer(4096)
		if chunk.is_empty():
			empty += 1
			OS.delay_msec(20)
			continue
		empty = 0
		if sink != null:
			lock.lock()
			sink.store_buffer(chunk)
			sink.flush()
			lock.unlock()


# Is an exited child's output still on its way into the log? Gives up after _DRAIN_MS.
static func _draining(pid: int) -> bool:
	if not _pumps.has(pid):
		return false
	var rec: Dictionary = _pumps[pid]
	var busy := false
	for t in rec.threads:
		busy = busy or (t as Thread).is_alive()
	if busy:
		var now := Time.get_ticks_msec()
		if int(rec.exited_ms) == 0:
			rec.exited_ms = now
		if now - int(rec.exited_ms) < _DRAIN_MS:
			return true
	_release_pumps(pid)
	return false


static func _release_pumps(pid: int) -> void:
	if _pumps.has(pid):
		_stray.append_array(_pumps[pid].threads)
		_pumps.erase(pid)
	_join_finished_pumps()


static func _join_finished_pumps() -> void:
	for t in _stray.duplicate():
		if not t.is_alive():
			t.wait_to_finish()
			_stray.erase(t)


static func _is_windows() -> bool:
	return OS.get_name() == "Windows"


# By absolute path under SystemRoot first: it ships with every Windows since 7, and a
# stripped PATH should not be what decides whether the assistant can start.
static func _powershell() -> String:
	var p := OS.get_environment("SystemRoot").path_join(
		"System32/WindowsPowerShell/v1.0/powershell.exe")
	if FileAccess.file_exists(p):
		return p
	var bin := Deps.resolve("powershell")
	if bin.is_empty():
		push_warning("ghost: PowerShell not found - programs that need their output "
			+ "redirected (the assistant, illustrations) cannot start")
	return bin


static func _ps_launcher_path() -> String:
	var path := OS.get_user_data_dir().path_join("launch/redirect.ps1")
	if FileAccess.get_file_as_string(path) != _PS_LAUNCHER:
		DirAccess.make_dir_recursive_absolute(path.get_base_dir())
		var f := FileAccess.open(path, FileAccess.WRITE)
		if f != null:
			f.store_string(_PS_LAUNCHER)
			f.close()
	return path


# One spec per launch; the launcher deletes it once read.
static func _write_ps_spec(spec: Dictionary) -> String:
	var dir := OS.get_user_data_dir().path_join("launch")
	DirAccess.make_dir_recursive_absolute(dir)
	var path := dir.path_join("%d_%d.json" % [Time.get_ticks_usec(), randi()])
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		return ""
	f.store_string(JSON.stringify(spec))
	f.close()
	return path


# The `setpriv` binary if it can bind a child to us, "-" if it cannot. Resolved once per
# run. PROVEN, not assumed: `--pdeathsig` arrived in util-linux 2.33, and an older setpriv
# rejects the flag - which would mean every child failing to start at all rather than
# merely failing to be bound. So the flag is exercised on `true` before anything real
# rides on it.
static func _pact_bin() -> String:
	if _pact != "":
		return _pact
	_pact = "-"
	if OS.get_name() != "Linux":
		return _pact          # PR_SET_PDEATHSIG is a Linux facility; elsewhere the registry stands alone
	var bin := Deps.resolve("setpriv")
	if bin.is_empty():
		push_warning("ghost: setpriv not found - background programs will only be stopped on a "
			+ "clean quit, not if ghost is killed outright")
		return _pact
	if OS.execute(bin, ["--pdeathsig", "KILL", "--", "true"]) == 0:
		_pact = bin
	return _pact
