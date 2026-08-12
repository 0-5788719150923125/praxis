extends Node

## Harmless placeholder. tests/run_boot_probe.sh COPIES a probe over this file, runs it
## inside a real boot via a temporary override.cfg, and restores this stub on every exit
## path. It lives inside the project, so whatever sits here is parsed whenever Godot loads
## the project - a probe left behind with a stale reference stops ghost booting at all.
##
## If you are reading this because ghost booted into something odd: check for a stray
## override.cfg in the project root and delete it.

func _ready() -> void:
	print("boot_probe: placeholder (no probe installed)")
	get_tree().quit()
