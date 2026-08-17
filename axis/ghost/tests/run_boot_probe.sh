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
# scene instead of main, which override.cfg does. This script writes that file,
# runs, and ALWAYS removes it again - a stale override.cfg silently hijacks the
# whole project (it has pinned the resolution before now), so the cleanup runs
# on any exit path including Ctrl-C.
#
#   tests/run_boot_probe.sh <probe.gd> [timeout_seconds]
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
OVERRIDE="override.cfg"
PROBE_SRC="${1:-}"
TIMEOUT="${2:-90}"
PROBE="tests/boot_probe.gd"
STUB=$(mktemp)
cp "$PROBE" "$STUB"

# ARMED BEFORE ANYTHING IS COPIED IN, and that ordering is not cosmetic. The trap
# used to be installed after the probe had already been written over tests/
# boot_probe.gd, so the two checks below - a missing probe, and a pre-existing
# override.cfg - exited without restoring it and LEFT THE PROBE IN THE PROJECT.
# That is exactly the state this script's own header warns about, and it happened:
# a scratch probe sat in tests/boot_probe.gd as a modified tracked file, silently,
# because a stale override.cfg (the exporter writes one) made every later run
# refuse before it could clean up. Nothing may be copied over that file until the
# restore is guaranteed.
# ...and it removes the override only if THIS run wrote it. Arming the trap early
# means it now also fires on the refuse path, where the override.cfg in the way is
# somebody else's - a render in flight, most likely - and deleting that would be a
# worse failure than the one being refused.
MINE=0
cleanup() {
  [[ "$MINE" = 1 ]] && rm -f "$OVERRIDE"
  cp "$STUB" "$PROBE"
  rm -f "$STUB"
}
trap cleanup EXIT INT TERM

if [[ -n "$PROBE_SRC" ]]; then
  if [[ ! -f "$PROBE_SRC" ]]; then
    echo "run_boot_probe: no such probe: $PROBE_SRC" >&2
    exit 2
  fi
fi

if [[ -e "$OVERRIDE" ]]; then
  echo "run_boot_probe: $OVERRIDE already exists - refusing to clobber it." >&2
  echo "Inspect and remove it first; a stale one changes how ghost itself runs." >&2
  exit 2
fi

if [[ -n "$PROBE_SRC" ]]; then
  cp "$PROBE_SRC" "$PROBE"
fi

MINE=1
cat > "$OVERRIDE" <<'CFG'
; TEMPORARY - written by tests/run_boot_probe.sh, removed when it exits.
; If you are reading this in a working tree, the script was killed uncleanly:
; delete this file. While it exists, ghost boots the probe instead of main.
[application]
run/main_scene="res://tests/boot_probe.tscn"
CFG

timeout "$TIMEOUT" godot --headless --path . 2>&1
status=$?
if [[ $status -eq 124 ]]; then
  echo "run_boot_probe: TIMED OUT after ${TIMEOUT}s (probe never quit)" >&2
fi
exit $status
