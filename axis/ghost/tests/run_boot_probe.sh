#!/usr/bin/env bash
# Run tests/boot_probe.gd inside a REAL boot, with autoloads alive.
#
# WHY THIS EXISTS
# ---------------
# `godot --headless --script X.gd` runs a SceneTree script with no autoloads, so
# anything touching Spectrum or Director fails to compile with "Identifier not
# found". That limitation is real, but the conclusion drawn from it - that such
# code cannot be tested headlessly - was wrong, and it cost a string of bugs
# that were shipped on a compile check and a careful read instead of a run.
#
# A real boot has autoloads. All that is needed is to point the app at a probe
# scene instead of main, and Godot takes that as a POSITIONAL ARGUMENT - a scene
# path after the options runs that scene, autoloads and all.
#
#   tests/run_boot_probe.sh <probe.gd> [timeout_seconds] [probe args ...]
#   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh <probe.gd>   # real renderer, no window
#
# It used to say so in override.cfg instead, and that file is project-wide state:
# the exporter writes one too, so this script had to refuse to run at all while a
# render was in flight, and a run killed uncleanly left ghost itself booting into
# a probe. An argument is local to the process. Nothing global is touched now and
# a probe runs perfectly happily alongside a render.
#
# The probe is COPIED into place and the harmless stub is restored afterwards.
# That matters: tests/boot_probe.tscn lives inside the project, so Godot parses
# tests/boot_probe.gd at load - and a probe left there with a syntax error stops
# ghost itself from starting (black screen, "Failed to load script"). Nothing
# broken is ever left behind now.
#
# Print what you want to see and call get_tree().quit(); the timeout is a
# backstop for a probe that hangs. Note that most probe locals need explicit
# types - objects loaded with load().new() are untyped Variants and inference
# will fail on them.

set -uo pipefail
cd "$(dirname "$0")/.."          # axis/ghost
PROBE_SRC="${1:-}"
TIMEOUT="${2:-90}"
PROBE="tests/boot_probe.gd"
STUB=$(mktemp)
cp "$PROBE" "$STUB"

# ARMED BEFORE ANYTHING IS COPIED IN, and that ordering is not cosmetic. The trap
# used to be installed after the probe had already been written over tests/
# boot_probe.gd, so an early exit - a missing probe, say - returned without
# restoring it and LEFT THE PROBE IN THE PROJECT. That is exactly the state this
# script's own header warns about, and it happened: a scratch probe sat in
# tests/boot_probe.gd as a modified tracked file, silently. Nothing may be copied
# over that file until the restore is guaranteed.
cleanup() {
  cp "$STUB" "$PROBE"
  rm -f "$STUB"
}
trap cleanup EXIT INT TERM

if [[ -n "$PROBE_SRC" ]]; then
  if [[ ! -f "$PROBE_SRC" ]]; then
    echo "run_boot_probe: no such probe: $PROBE_SRC" >&2
    exit 2
  fi
  cp "$PROBE_SRC" "$PROBE"
fi

# THE RENDERER. A probe that needs autoloads AND REAL PIXELS has nowhere else to
# go: `--headless` is the dummy driver, whose viewport readback returns nothing
# (run_quiet.sh's header spells that out), and `godot --script` has no autoloads at
# all. GHOST_PROBE_GPU=1 boots this same probe scene on the real GPU inside a
# VIRTUAL DISPLAY, so it still never puts a window on screen. Needs
# xorg-server-xvfb, exactly like run_quiet.sh.
RUNNER=(godot --headless)
if [[ "${GHOST_PROBE_GPU:-0}" != "0" ]]; then
  if command -v xvfb-run >/dev/null 2>&1; then
    RUNNER=(xvfb-run -a -s "-screen 0 1280x1024x24" godot)
  else
    echo "run_boot_probe: GHOST_PROBE_GPU set but xvfb-run not found -" \
         "falling back to a VISIBLE window." >&2
    RUNNER=(godot)
  fi
fi

# Anything after the timeout is handed to the PROBE as user args (read them with
# OS.get_cmdline_user_args), so a probe can take options the way clown_look_probe does.
timeout "$TIMEOUT" "${RUNNER[@]}" --path . res://tests/boot_probe.tscn -- "${@:3}" 2>&1
status=$?
if [[ $status -eq 124 ]]; then
  echo "run_boot_probe: TIMED OUT after ${TIMEOUT}s (probe never quit)" >&2
fi
exit $status
