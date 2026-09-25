extends SceneTree

## Does each assistant backend launch the command it claims, and read its own stream?
##
## Both halves fail SILENTLY. A command built wrong is a dispatch that exits at once and
## shows as an error nobody connects to the argument list; a stream read with the wrong
## backend's rules is a run with no progress line that never reports a result. So:
##
##   THE COMMAND - the argv for a fresh run and for a resume, per backend, with the prompt
##   in none of them, and that Subprocess.start_redirected really delivers a hostile prompt
##   on stdin byte for byte. That last one is RUN, not inspected: `cat` stands in for the
##   CLI, and the captured output must be the prompt exactly. A `$(...)` that expanded would
##   show. It launches detached, exactly as the dispatch does.
##
##   THE STREAM - recorded output (tests/fixtures/assistant/; the codex files are real
##   `codex exec --json` runs from codex-cli 0.150.0, one a resume of the other and one a
##   failed turn) through progress, session id, final text and usage. Each backend's stream
##   is also read with the OTHER backend's rules and must NOT come out as a clean result,
##   which is the mix-up the registry exists to prevent.
##
##   godot --headless --path axis/ghost --script tests/assistant_backend_check.gd

const B := preload("res://scripts/assistant_backends.gd")
const FIX := "res://tests/fixtures/assistant/"

var _fails: Array = []


func _init() -> void:
	_commands()
	_launcher()
	_claude_stream()
	_codex_stream()
	_crossed()
	if _fails.is_empty():
		print("assistant_backend_check: ALL OK")
		quit(0)
	else:
		print("assistant_backend_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
		quit(1)


func _check(ok: bool, msg: String) -> void:
	print(("   ok   " if ok else "   FAIL ") + msg)
	if not ok:
		_fails.append(msg)


func _events(name: String) -> Array:
	return B.events_in(FileAccess.get_file_as_string(FIX + name))


func _commands() -> void:
	print("-- commands")
	var c := B.argv("claude_cli", "/bin/claude", "", "/repo")
	_check(c == PackedStringArray(["/bin/claude", "-p", "--model", "sonnet",
		"--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]),
		"claude fresh run: -p with no prompt argument (it reads stdin)")
	c = B.argv("claude_cli", "/bin/claude", "sid-1", "/repo")
	_check(c.has("--resume") and c[c.find("--resume") + 1] == "sid-1",
		"claude resume passes --resume <id>")
	var x := B.argv("codex_cli", "/bin/codex", "", "/repo")
	_check(x == PackedStringArray(["/bin/codex", "exec", "--json", "--skip-git-repo-check",
		"--dangerously-bypass-approvals-and-sandbox", "-C", "/repo", "-"]),
		"codex fresh run: exec --json, full access, -C repo, `-` for a stdin prompt")
	x = B.argv("codex_cli", "/bin/codex", "th-9", "/repo")
	_check(x.slice(0, 4) == PackedStringArray(["/bin/codex", "exec", "resume", "th-9"]),
		"codex resume is the `exec resume <id>` subcommand")
	_check(not x.has("-C"), "codex resume does not pass -C (the subcommand has no such flag)")
	_check(x.has("--json") and x[x.size() - 1] == "-", "codex resume still streams JSON, stdin prompt")
	_check(B.label("nope") == B.label(B.LEGACY) and B.dep("codex_cli") == "codex",
		"unknown key falls back to Claude; codex resolves through the codex Deps row")


func _launcher() -> void:
	print("-- launcher (Subprocess.start_redirected)")
	var dir := OS.get_user_data_dir().path_join("assistant_check")
	DirAccess.make_dir_recursive_absolute(dir)
	var out_p := dir.path_join("out.txt")
	var err_p := dir.path_join("err.txt")
	var in_p := dir.path_join("prompt.txt")
	var hostile := "a \"b\" 'c' $(echo INJECTED) `echo X` ; exit 7 \\ $HOME %PATH% & ^\nsecond line"
	var f := FileAccess.open(in_p, FileAccess.WRITE)
	f.store_string(hostile)
	f.close()
	_run("cat", [], {"stdin": in_p, "out": out_p, "err": err_p})
	_check(FileAccess.get_file_as_string(out_p) == hostile,
		"the prompt arrived on stdin, byte for byte")
	# The cd is real: a relative path in the program resolves against the repo root.
	_run("pwd", [], {"cwd": dir, "out": out_p, "err": err_p})
	_check(FileAccess.get_file_as_string(out_p).strip_edges() == dir,
		"the run starts in the repo root")
	# ...and with no stdin file, stdin is empty and closed, or `codex exec` sits waiting.
	var code := _run("cat", [], {"out": out_p, "err": err_p})
	_check(code == 0 and FileAccess.get_file_as_string(out_p).is_empty(),
		"no stdin file: cat returns at once, empty")
	# stdout and stderr land apart - the dispatch parses one and shows the other.
	_run("sh", ["-c", "echo to-out; echo to-err >&2"], {"out": out_p, "err": err_p})
	_check(FileAccess.get_file_as_string(out_p) == "to-out\n"
		and FileAccess.get_file_as_string(err_p) == "to-err\n", "stdout and stderr are separate")
	_check(Subprocess.start_redirected("cat", [], {"out": out_p, "err": out_p}) <= 0,
		"one file for both streams is refused, not silently interleaved")
	for p in [out_p, err_p, in_p]:
		DirAccess.remove_absolute(p)


## Launch detached, exactly as the dispatch does, and wait for it. The exit code is not
## observable from create_process, so this returns 0 once the child is gone and -1 if it
## never started or overran - the checks read its output instead.
func _run(prog: String, args: Array, io: Dictionary) -> int:
	var pid := Subprocess.start_redirected(prog, PackedStringArray(args), io, "", true)
	if pid < 0:
		return -1
	for i in 200:
		if not OS.is_process_running(pid):
			return 0
		OS.delay_msec(25)
	Subprocess.terminate(pid)
	return -1


func _claude_stream() -> void:
	print("-- claude stream")
	var ev := _events("claude_run.jsonl")
	_check(B.session_of("claude_cli", ev[0]) == "5b1c0e2a-1111-4222-8333-944455556666",
		"session id from the init event")
	_check(B.describe("claude_cli", ev[1]) == "Read  axis/ghost/CLAUDE.md", "tool use -> progress")
	_check(B.describe("claude_cli", ev[3]) == "Fixed the fade in comic.gd.", "text -> progress")
	_check(B.describe("claude_cli", ev[2]) == "", "tool results are not shown")
	var r: Dictionary = B.result("claude_cli", ev)
	_check(bool(r.ok) and String(r.response) == "Fixed the fade in comic.gd.", "final result read")
	_check(is_equal_approx(float(r.cost_usd), 0.4213) and String(r.usage) == "$0.421",
		"cost in dollars")


func _codex_stream() -> void:
	print("-- codex stream (recorded)")
	var ev := _events("codex_run.jsonl")
	var tid := "01a0cc10-5ce8-7e32-80fc-003396cd996e"
	_check(B.session_of("codex_cli", ev[0]) == tid, "thread id from thread.started")
	_check(B.session_of("codex_cli", ev[2]) == "", "no id from other events")
	_check(B.describe("codex_cli", ev[3]) == "shell  /bin/bash -lc 'echo hi'", "command -> progress")
	_check(B.describe("codex_cli", ev[1]) == "", "turn.started is not progress")
	var r: Dictionary = B.result("codex_cli", ev)
	_check(bool(r.ok) and String(r.response) == "ok", "final text is the LAST agent message")
	_check(String(r.session) == tid, "result carries the thread id")
	_check(String(r.usage).contains("27.2k in") and String(r.usage).contains("112 out"),
		"usage in tokens: " + String(r.usage))
	var res: Dictionary = B.result("codex_cli", _events("codex_resume.jsonl"))
	_check(bool(res.ok) and String(res.response) == "again" and String(res.session) == tid,
		"a resumed run continues the same thread")
	var bad: Dictionary = B.result("codex_cli", _events("codex_failed.jsonl"))
	_check(not bool(bad.ok) and String(bad.error).contains("not supported"),
		"a failed turn is not a result, and says why")
	_check(String(bad.session) != "", "...but keeps its thread id, so it can be resumed")
	var retry: Array = ev.duplicate()
	retry.insert(2, {"type": "error", "message": "Reconnecting... 1/5"})
	_check(bool(B.result("codex_cli", retry).ok), "a retry notice alone does not fail a completed turn")
	var raw := FileAccess.get_file_as_string(FIX + "codex_failed.jsonl")
	_check(B.recover_session("codex_cli", raw) == "01a0cc10-9c5d-70d2-a2c5-c6b16abf4a80",
		"id recoverable from raw text")


func _crossed() -> void:
	print("-- crossed (each stream read by the wrong rules)")
	_check(not bool(B.result("claude_cli", _events("codex_run.jsonl")).ok),
		"a codex stream is not a clean claude result")
	_check(not bool(B.result("codex_cli", _events("claude_run.jsonl")).ok),
		"a claude stream is not a clean codex result")
