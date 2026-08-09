extends Node

## Scratch probe for tests/run_boot_probe.sh - runs inside a REAL boot, so every
## autoload is alive. Rewrite the body per investigation.
##
## KEEP THIS FILE PARSEABLE. tests/boot_probe.tscn references it, so Godot
## compiles it at project load - a syntax error here makes ghost itself fail to
## start, which is exactly what happened once. Leave it in this harmless state
## when you are done with it, and note that `ed` is an untyped Variant in most
## probes: annotate locals (`var n: int = ...`) or type inference will fail.

func _ready() -> void:
	print("PROBE: idle - nothing to check")
	get_tree().quit()
