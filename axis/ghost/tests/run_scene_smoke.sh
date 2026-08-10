#!/usr/bin/env bash
# Run the whole-catalogue smoke gate (tests/scene_smoke.gd) and report a TRUSTWORTHY
# exit status.
#
# WHY THIS WRAPPER EXISTS
# -----------------------
# The probe itself is straightforward - build every Director.SCENES entry at several
# seeds, update it, render it, free it - and it works: it reaches its verdict and
# prints it. What it cannot do is EXIT cleanly.
#
# Creating and freeing 162 scenes inside one process leaves Godot's shutdown unhappy:
# after the verdict is printed and the tree is quit, the engine dies during its own
# teardown with `FATAL: Index p_index = 1 is out of bounds (size() = 0)` and a signal
# 11. Measured, so that nobody re-derives it:
#   - it happens with drawing disabled entirely, so it is not the renderer;
#   - it happens with immediate free() and with deferred queue_free();
#   - it happens with six settle frames between the last free and the quit;
#   - a probe that boots and quits without touching a scene exits 0 every time;
#   - every one of the 43 scripts passes in isolation, and the full run prints ALL OK.
# So it is a shutdown artifact of the probe's workload, not a fault in any scene, and
# it is not something this project can fix from GDScript.
#
# The rule below is therefore the honest one, and it is deliberately NOT "ignore
# crashes":
#
#   The probe prints its verdict BEFORE quitting. If a verdict was printed, the
#   catalogue was fully exercised and the verdict is the result. If NO verdict was
#   printed, the probe died partway through - which is a real failure and is reported
#   as one, along with the last scene it announced.
#
# That keeps the gate able to catch a genuinely crashing scene (it would die before
# printing anything) while not failing the build on engine teardown noise.
#
#   tests/run_scene_smoke.sh [timeout_seconds]
#
# Note the runtime: metropolis, spires and terrain_city each build a full heightfield
# or swarm per instantiation and are seconds apiece, so the default budget is generous.

set -uo pipefail
cd "$(dirname "$0")/.."          # axis/ghost
TIMEOUT="${1:-600}"
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

tests/run_boot_probe.sh tests/scene_smoke.gd "$TIMEOUT" 2>&1 | tee "$LOG" \
  | grep -Ev '^\s*at:|GDScript backtrace|^\s*\[[0-9]+\]|handle_crash|Load address|^=+$|Dumping the backtrace|Engine version'

# A RUNTIME SCRIPT ERROR IS A FAILURE, even though the probe survives it. Godot prints
# "SCRIPT ERROR: Out of bounds ..." and carries on with a null result, so a scene can be
# quietly broken - wrong geometry, missing birds, a torn array - while every structural
# check passes and the probe reports ALL OK. That is precisely how the FrameForge race
# nearly shipped: three stray errors in six hundred rendered frames, under a PASS.
if grep -qE '^(SCRIPT )?ERROR: (Out of bounds|Invalid|Trying to|Condition)' "$LOG" \
   || grep -q '^SCRIPT ERROR:' "$LOG"; then
  echo "run_scene_smoke: FAIL - the run printed script errors:" >&2
  grep -E '^SCRIPT ERROR:|^ *at: ' "$LOG" | head -20 >&2
  exit 1
fi

if grep -q '^scene_smoke: ALL OK' "$LOG"; then
  echo "run_scene_smoke: PASS"
  exit 0
fi

if grep -q '^scene_smoke: [0-9]* FAILURE' "$LOG"; then
  echo "run_scene_smoke: FAIL - scenes reported failures (above)" >&2
  exit 1
fi

echo "run_scene_smoke: FAIL - the probe never reached a verdict." >&2
echo "  That means it died partway through, which is a REAL failure." >&2
last=$(grep -E '^  -> ' "$LOG" | tail -1)
if [[ -n "$last" ]]; then
  echo "  last scene announced:$last" >&2
else
  echo "  no scene was announced; it failed during boot." >&2
fi
exit 1
