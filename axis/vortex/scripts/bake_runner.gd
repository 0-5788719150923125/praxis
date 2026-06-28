extends SceneTree

## bake_runner - a headless one-shot that analyzes a song and writes its spectrum
## cache, then quits. The exporter launches this (with --headless, so NO window
## appears) before the video render, so the render can just load the cache and start
## drawing immediately instead of freezing while it bakes. Run as:
##   godot --headless --path . --script res://scripts/bake_runner.gd -- \
##         --bake-song <path> --bake-out <cache.spec>
##
## The analysis parameters here MUST match Spectrum's (BAND_COUNT / FREQ_MIN /
## FREQ_MAX / DB_FLOOR / BAKE_FPS) so the cache the render loads is the right shape.

const BANDS := 64
const FREQ_MIN := 30.0
const FREQ_MAX := 16000.0
const DB_FLOOR := 60.0
const FPS := 30


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	var song := ""
	var out := ""
	for i in args.size():
		if args[i] == "--bake-song" and i + 1 < args.size():
			song = args[i + 1]
		elif args[i] == "--bake-out" and i + 1 < args.size():
			out = args[i + 1]
	if song.is_empty() or out.is_empty():
		push_error("bake_runner: need --bake-song and --bake-out")
		quit(1)
		return

	var bake := load("res://scripts/bake.gd")
	var frames = bake.bake(song, FPS, BANDS, FREQ_MIN, FREQ_MAX, DB_FLOOR)
	if frames.is_empty():
		push_error("bake_runner: bake failed (is ffmpeg on PATH?)")
		quit(1)
		return
	bake.save_cache(out, frames, BANDS)
	print("bake_runner: wrote %s (%d frames)" % [out, frames.size()])
	quit(0)
