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

## pid -> a short label, kept only so the shutdown reap can say what it killed.
static var _tracked := {}
## Resolved once: "" not looked up yet, "-" no pact available, otherwise the setpriv path.
static var _pact := ""


## Start `path` with `args`, bound to this process, and return its pid (<= 0 on failure).
## `tag` is a human label for the shutdown log; it defaults to the program's own name.
static func start(path: String, args: PackedStringArray, tag := "") -> int:
	var bin := _pact_bin()
	var pid := -1
	if bin == "-":
		pid = OS.create_process(path, args)
	else:
		var full := PackedStringArray(["--pdeathsig", "KILL", "--", path])
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
	var bin := _pact_bin()
	var info := {}
	if bin == "-":
		info = OS.execute_with_pipe(path, args)
	else:
		var full := PackedStringArray(["--pdeathsig", "KILL", "--", path])
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
	return OS.create_process(path, args)


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
	if OS.is_process_running(pid):
		return true
	_tracked.erase(pid)
	return false


## Kill one child now and forget it. Returns whether it actually signalled anything, so a
## second stop - after a reap, or on a pid a mode's shutdown list still holds from last time -
## is visibly a no-op rather than an error in the log.
static func stop(pid: int) -> bool:
	if pid <= 0 or not _tracked.has(pid):
		return false
	OS.kill(pid)
	_tracked.erase(pid)
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
			OS.kill(int(pid))
			n += 1
	if n > 0:
		print("ghost: stopped %d background process(es) on exit: %s"
			% [n, ", ".join(PackedStringArray(_tracked.values()))])
	_tracked.clear()
	return n


## How many children are currently registered - for status lines and gates.
static func tracked() -> int:
	return _tracked.size()


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
	var out: Array = []
	if OS.execute("which", ["setpriv"], out) != 0 or out.is_empty():
		push_warning("ghost: setpriv not found - background programs will only be stopped on a "
			+ "clean quit, not if ghost is killed outright")
		return _pact
	var bin := String(out[0]).strip_edges().split("\n")[0]
	if bin != "" and OS.execute(bin, ["--pdeathsig", "KILL", "--", "true"]) == 0:
		_pact = bin
	return _pact
