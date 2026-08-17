#!/usr/bin/env bash
# Run the gates that NEED A REAL RENDERER, with no window on screen.
#
#   tests/run_quiet.sh clown_drip_check clown_controls_check ...
#   tests/run_quiet.sh --all
#
# THE PROBLEM. `--headless` forces Godot's DUMMY rendering driver - `godot --help`
# spells it out: `"headless" ("dummy")` - and the dummy driver returns nothing
# from a viewport readback: `ERROR: Parameter "t" is null. at: texture_2d_get`.
# So every gate that measures PIXELS needs a real GPU context, which on Linux
# means a real window on a real display. Two things that look like fixes are not:
#
#   --position -5000,-5000    ignored under WAYLAND, because a Wayland client
#                             cannot place its own window. This was the whole
#                             reason the windows kept appearing in front of the
#                             author despite the flag being passed.
#   no_focus + a 64x64 window stops it taking the keyboard, but the window still
#                             maps. A tiny window that steals nothing is still a
#                             window that pops up dozens of times an hour.
#
# THE FIX is a real X display that is not on screen. `xvfb-run` gives Godot a
# virtual one; the NVIDIA driver still renders on the actual GPU (verified: the
# adapter reports as the real card, and a SubViewport readback returns real
# pixels, where the dummy driver returns null). Nothing about what the gates
# measure changes - they render into SubViewports, so the main window was never
# part of the measurement.
#
# Needs `xorg-server-xvfb`. Without it this falls back to a visible window and
# says so, rather than silently doing the annoying thing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

RUNNER=(godot)
if command -v xvfb-run >/dev/null 2>&1; then
	# -a picks a free display number; -s sets a screen big enough that a gate
	# requesting a window never has it clamped.
	RUNNER=(xvfb-run -a -s "-screen 0 1280x1024x24")
	RUNNER+=(godot)
else
	echo "run_quiet: xvfb-run not found - falling back to a VISIBLE window."
	echo "           install it with: sudo pacman -S xorg-server-xvfb"
fi

# Every gate that needs a real renderer. Keep in step with CLAUDE.md's gate list.
ALL=(clown_drip_check clown_anchor_check clown_controls_check clown_scale_check
	clown_coat_check clown_coverage_check repaint_check rain_check)

if [ "${1:-}" = "--all" ]; then
	set -- "${ALL[@]}"
fi
if [ "$#" -eq 0 ]; then
	echo "usage: tests/run_quiet.sh <gate> [gate ...]   |   tests/run_quiet.sh --all"
	echo "       tests/run_quiet.sh -- <script.gd> [args...]   for a one-off probe"
	exit 2
fi

# `-- <script>` runs one script with the rest passed through to it, which is how
# clown_look_probe.gd (a probe, not a gate - it takes --frame/--track/--out) gets
# the same quiet treatment.
if [ "$1" = "--" ]; then
	shift
	script="$1"
	shift
	"${RUNNER[@]}" --path . --script "$script" -- "$@"
	exit $?
fi

fails=0
for t in "$@"; do
	name="${t%.gd}"
	if [ ! -f "tests/${name}.gd" ]; then
		echo "run_quiet: no such gate: ${name}"
		fails=$((fails + 1))
		continue
	fi
	echo "=== ${name} ==="
	"${RUNNER[@]}" --path . --script "res://tests/${name}.gd"
	rc=$?
	[ "$rc" -ne 0 ] && fails=$((fails + 1))
	echo "exit=${rc}"
done

if [ "$fails" -eq 0 ]; then
	echo "run_quiet: every gate passed."
else
	echo "run_quiet: ${fails} gate(s) FAILED."
fi
exit "$fails"
