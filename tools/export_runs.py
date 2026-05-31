#!/usr/bin/env python3
"""Export recent-run validation curves into LaTeX-ready data for the paper.

Reads the newest experiments under ``build/runs/`` and emits, into
``research/``:

- ``data/run_N.csv`` - ``step,val`` series, one per experiment (run 1 = current).
- ``variables.tex`` - generated macros the paper ``\\input``s: the chosen
  metric's name/label, per-run name/hash/final/steps, and a ready-to-drop
  ``\\paperValPlots`` body (the ``\\addplot`` lines + legend).

The y-metric is chosen per family, since validation metrics are not comparable
across families: byte-latent runs report ``val_bits_per_byte``, codec/CALM runs
report ``val_brierlm`` (they emit no loss/bpb), token runs report ``val_loss``.
``--metric auto`` picks the first of those the current run actually populates;
override with ``--metric <col>``.

Usage:
    python tools/export_runs.py                 # 4 newest experiments, auto metric
    python tools/export_runs.py --n 4
    python tools/export_runs.py --metric val_brierlm
    python tools/export_runs.py --json

Note: run DBs are opened read-only/immutable, so a live run's most recent
(uncheckpointed) rows may lag by a validation interval.
"""

import argparse
import csv
import glob
import json
import math
import os
import sqlite3
import sys
import time

# Comparable generation metrics, in tie-break priority order. bpb is byte-latent
# only (so it is family-consistent by construction) and preferred; brierlm is the
# CALM generation metric; val_loss is the token-vocab fallback. val_codec_bpb is
# deliberately excluded from auto - it measures codec fidelity, not generation
# ("judge with val_brierlm, not this") - and is reachable only via --metric.
METRIC_PRIORITY = ["val_bits_per_byte", "val_brierlm", "val_loss"]
# Auto never picks codec fidelity, but it is a last-resort fallback so the
# current experiment can always plot *something*, and is reachable via --metric.
ALL_METRICS = METRIC_PRIORITY + ["val_codec_bpb"]

# Display labels (kept local so the tool runs without importing praxis/torch).
METRIC_LABELS = {
    "val_loss": ("Validation Loss", "Validation loss"),
    "val_bits_per_byte": ("Bits per Byte", "Validation bits/byte"),
    "val_brierlm": ("BrierLM", "BrierLM score"),
    "val_codec_bpb": ("Codec Recon", "Codec bits/byte"),
}

# Command flags that are not experiment selectors.
DENY_FLAGS = {
    "reset", "dev", "no-compile", "compile", "profile-memory", "device",
    "max-steps", "batch-size", "debug", "seed", "host", "port",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, "build", "runs")
OUT_DIR = os.path.join(REPO_ROOT, "research")


def experiment_name(command: str, stems: set) -> str:
    """Resolve a run command to its experiment name (the --<stem> flag)."""
    toks = command.split()
    for t in toks:
        if t.startswith("--") and t[2:] in stems:
            return t[2:]
    for t in toks:  # fallback: first flag that isn't a known non-experiment one
        if t.startswith("--") and t[2:] not in DENY_FLAGS:
            return t[2:]
    return command.strip() or "unknown"


def experiment_stems() -> set:
    exp = glob.glob(os.path.join(REPO_ROOT, "experiments", "*.yml"))
    return {os.path.splitext(os.path.basename(p))[0] for p in exp}


def _clean(points, min_points):
    """Shared post-filter: drop an all-zero column, enforce min length."""
    if points and all(v == 0.0 for _, v in points):
        return []
    return points if len(points) >= min_points else []


def metric_series(run: dict, metric: str, min_points: int = 2):
    """Usable [(step, value)] for a metric, preferring the run's CSV mirror.

    The metrics logger writes ``metrics.csv`` alongside ``metrics.db`` as a
    standard part of logging, so the CSV is the fast, lock-free, server-free
    source. Fall back to the SQLite DB for older runs that predate the mirror.
    """
    return read_csv_series(run.get("csv", ""), metric, min_points) \
        or read_series(run.get("db", ""), metric, min_points)


def read_csv_series(csv_path: str, metric: str, min_points: int = 2):
    """Return [(step, value)] for a metric from a per-run metrics.csv."""
    if not csv_path or not os.path.exists(csv_path):
        return []
    out = []
    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            if metric not in (reader.fieldnames or []):
                return []
            for row in reader:
                sv, vv = row.get("step"), row.get(metric)
                if vv in (None, ""):
                    continue
                try:
                    v, s = float(vv), int(float(sv))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    out.append((s, v))
    except OSError:
        return []
    return _clean(out, min_points)


def read_series(db_path: str, metric: str, min_points: int = 2):
    """Return [(step, value)] for a metric, or [] if absent/empty. Read-only.

    ``min_points`` is the minimum usable points to count as a series; the
    current experiment is read with 1 (a just-started run plots as a mark),
    comparison runs with 2 (they need an actual curve).
    """
    if not db_path or not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    except sqlite3.Error:
        return []
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
        if metric not in cols:
            return []
        rows = conn.execute(
            f"SELECT step, {metric} FROM metrics "
            f"WHERE {metric} IS NOT NULL ORDER BY step"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = [(int(s), float(v)) for s, v in rows
           if v is not None and math.isfinite(v)]
    return _clean(out, min_points)


def discover_runs(stems: set):
    """Newest run per experiment, sorted by created (desc)."""
    runs = []
    for cfg_path in glob.glob(os.path.join(RUNS_DIR, "*", "config.json")):
        run_dir = os.path.dirname(cfg_path)
        try:
            cfg = json.load(open(cfg_path))
        except (OSError, ValueError):
            continue
        created = cfg.get("created") or ""
        if not created:  # fall back to file mtime as an ISO-ish sort key
            created = str(os.path.getmtime(cfg_path))
        runs.append({
            "dir": run_dir,
            "hash": cfg.get("truncated_hash", os.path.basename(run_dir)),
            "command": cfg.get("command", ""),
            "name": experiment_name(cfg.get("command", ""), stems),
            "created": created,
            "updated": cfg.get("last_updated", created),
            "db": os.path.join(run_dir, "metrics.db"),
            "csv": os.path.join(run_dir, "metrics.csv"),
        })
    runs.sort(key=lambda r: r["created"], reverse=True)
    newest_per_exp, seen = [], set()
    for r in runs:
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        newest_per_exp.append(r)
    return newest_per_exp


NUM_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven"]


def latex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def export_once(n: int, metric_arg: str):
    """Select runs, write data/run_*.csv + variables.tex. Returns a summary
    dict, or None (with a stderr reason) if nothing usable was found yet."""
    stems = experiment_stems()
    runs = discover_runs(stems)
    if not runs:
        print(f"No runs with config.json under {RUNS_DIR}", file=sys.stderr)
        return None

    def pick_metric(run):
        """The metric to anchor on for a run, or None if it has no usable data.
        Honors --metric; else family priority, then best-populated column."""
        if metric_arg and metric_arg != "auto":
            return metric_arg if metric_series(run, metric_arg, 1) else None
        m = next((x for x in METRIC_PRIORITY if metric_series(run, x, 1)), None)
        if m:
            return m
        best = max(ALL_METRICS, key=lambda x: len(metric_series(run, x, 1)),
                   default=None)
        return best if best and metric_series(run, best, 1) else None

    window = runs[: max(n * 3, 12)]
    current = runs[0]  # the truly-newest experiment, always named in the note

    # The chart leads with the newest experiment that actually has data - which
    # is the current experiment once it has logged a validation point, and the
    # most recent prior experiment until then. The metric is anchored to that
    # leader's own family, so we never drop the current family for an unrelated
    # column.
    anchor = metric = None
    for r in window:
        m = pick_metric(r)
        if m:
            anchor, metric = r, m
            break
    if not anchor:
        print("No recent experiment has usable validation data yet.",
              file=sys.stderr)
        return None

    # Leader first; then the most recent experiments reporting the SAME metric
    # with a real curve. The rest of the window is reported as skipped
    # (incompatible metric) rather than silently dropped.
    lead_series = metric_series(anchor, metric, min_points=1)
    selected = [dict(anchor, series=lead_series)]
    skipped = []
    for r in window:
        if r["hash"] == anchor["hash"]:
            continue
        if len(selected) >= n:
            break
        series = metric_series(r, metric)
        if series:
            selected.append(dict(r, series=series))
        else:
            skipped.append(r["name"])

    # One generated sentence describing where the current experiment stands.
    cur_name = latex_escape(current["name"])
    if anchor["hash"] != current["hash"]:
        current_note = (
            f"The current experiment ({cur_name}) has not logged usable "
            f"validation metrics yet; the chart leads with the most recent "
            f"experiment that has ({latex_escape(anchor['name'])}).")
    elif len(lead_series) < 2:
        current_note = (
            f"The current experiment ({cur_name}) has logged only one "
            f"validation point so far, so it appears as a single marker; its "
            f"curve fills in as training proceeds.")
    else:
        current_note = ""

    title, ylabel = METRIC_LABELS.get(metric, (metric, metric))

    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
    plot_lines, var_lines = [], []
    var_lines.append("% Generated by tools/export_runs.py - do not edit by hand.")
    var_lines.append(f"\\newcommand{{\\paperMetricKey}}{{{latex_escape(metric)}}}")
    var_lines.append(f"\\newcommand{{\\paperMetricName}}{{{latex_escape(title)}}}")
    var_lines.append(f"\\newcommand{{\\paperMetricYlabel}}{{{latex_escape(ylabel)}}}")
    var_lines.append(f"\\newcommand{{\\paperRunCount}}{{{len(selected)}}}")
    var_lines.append(f"\\newcommand{{\\paperUpdated}}{{{selected[0]['updated'][:10]}}}")
    var_lines.append(f"\\newcommand{{\\paperSkipped}}{{{latex_escape(', '.join(skipped))}}}")
    # Empty when the current experiment leads with a full curve; otherwise a
    # ready-to-print sentence on why the leader differs / shows as a marker.
    var_lines.append(f"\\newcommand{{\\paperCurrentNote}}{{{current_note}}}")

    summary = []
    for i, r in enumerate(selected):
        word = NUM_WORDS[i + 1]
        rel = f"data/run_{i+1}.csv"
        with open(os.path.join(OUT_DIR, rel), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "val"])
            w.writerows(r["series"])
        final_step, final_val = r["series"][-1]
        var_lines += [
            f"\\newcommand{{\\run{word}Name}}{{{latex_escape(r['name'])}}}",
            f"\\newcommand{{\\run{word}Hash}}{{{latex_escape(r['hash'])}}}",
            f"\\newcommand{{\\run{word}Final}}{{{final_val:.4g}}}",
            f"\\newcommand{{\\run{word}Steps}}{{{final_step}}}",
        ]
        # Current run: bold and marked so a single just-logged point is still
        # visible; comparison runs are plain lines.
        opts = "thick, mark=*, mark size=2pt" if i == 0 else "mark=none"
        plot_lines.append(
            f"\\addplot[{opts}] table[x=step,y=val,col sep=comma]{{{rel}}};")
        plot_lines.append(f"\\addlegendentry{{{latex_escape(r['name'])}}}")
        summary.append({"rank": i + 1, "name": r["name"], "hash": r["hash"],
                        "points": len(r["series"]), "final": final_val,
                        "final_step": final_step, "leader": i == 0})

    body = " ".join(plot_lines)
    var_lines.append(f"\\newcommand{{\\paperValPlots}}{{{body}}}")
    with open(os.path.join(OUT_DIR, "variables.tex"), "w") as fh:
        fh.write("\n".join(var_lines) + "\n")

    return {"metric": metric, "metric_name": title, "current": current["name"],
            "selected": summary, "skipped": skipped}


def report(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"current experiment: {result['current']}")
    print(f"metric: {result['metric']} ({result['metric_name']})")
    for s in result["selected"]:
        tag = " (leader)" if s["leader"] else ""
        print(f"  {s['rank']}. {s['name']:<12} {s['hash']}  "
              f"n={s['points']:<4} final={s['final']:.4g} @ step {s['final_step']}{tag}")
    if result["skipped"]:
        print(f"  skipped (incompatible metric): {', '.join(result['skipped'])}")
    print(f"wrote {OUT_DIR}/variables.tex and {len(result['selected'])} data/run_*.csv")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4, help="experiments to include (default 4)")
    ap.add_argument("--metric", default="auto",
                    help="metric column, or 'auto' (default) to pick per family")
    ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                    help="regenerate every N seconds (decoupled interval render); "
                         "Ctrl-C to stop")
    args = ap.parse_args()

    if args.watch > 0:
        print(f"[watch] regenerating every {args.watch:g}s (Ctrl-C to stop)")
        try:
            while True:
                result = export_once(args.n, args.metric)
                if result:
                    sel = ", ".join(s["name"] for s in result["selected"])
                    print(f"[watch] {result['metric']}: {sel}")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[watch] stopped")
            return 0

    result = export_once(args.n, args.metric)
    if not result:
        # "Nothing to render yet" is not a build failure - leave any prior
        # variables.tex in place and let the paper build on its fallbacks.
        return 0
    report(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
