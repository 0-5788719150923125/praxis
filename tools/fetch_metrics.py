#!/usr/bin/env python3
"""Fetch metrics from a running Praxis web app and print per-series summaries.

Usage:
    python tools/fetch_metrics.py                         # current run, scalars only
    python tools/fetch_metrics.py --dynamics              # also include /api/dynamics
    python tools/fetch_metrics.py --run 9df1d58be         # specific run hash
    python tools/fetch_metrics.py --filter 'halting|loss' # regex on series names
    python tools/fetch_metrics.py --json                  # machine-readable
    python tools/fetch_metrics.py --host 0.0.0.0:2100     # override server
"""

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "localhost:2100"
SPARK_GLYPHS = " ▁▂▃▄▅▆▇█"
SPARK_WIDTH = 16
SLOPE_TAIL_FRACTION = 0.2
EMPTY = "-"


def fetch_json(host: str, path: str, params: dict = None) -> dict:
    url = f"http://{host}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _is_num(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return not (math.isnan(v) or math.isinf(v))
    return False


def _slope(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / den


def summarize(steps, values):
    """Stats for a (steps, values) series. Inputs may carry None for gaps."""
    pairs = [(s, v) for s, v in zip(steps, values) if s is not None and _is_num(v)]
    missing = sum(1 for v in values if v is None)
    if not pairs:
        return {
            "n": 0,
            "first": None,
            "latest": None,
            "min": None,
            "max": None,
            "missing": missing,
            "slope_per_1k": None,
        }
    ss, vs = zip(*pairs)
    n = len(pairs)
    tail_n = max(2, math.ceil(n * SLOPE_TAIL_FRACTION))
    slope = _slope(ss[-tail_n:], vs[-tail_n:]) if n >= 2 else None
    return {
        "n": n,
        "first": vs[0],
        "latest": vs[-1],
        "min": min(vs),
        "max": max(vs),
        "missing": missing,
        "slope_per_1k": (slope * 1000.0) if slope is not None else None,
    }


def sparkline(values, width=SPARK_WIDTH):
    pts = [v for v in values if _is_num(v)][-width:]
    if not pts:
        return ""
    lo, hi = min(pts), max(pts)
    if hi == lo:
        return SPARK_GLYPHS[4] * len(pts)
    last = len(SPARK_GLYPHS) - 1
    return "".join(SPARK_GLYPHS[int(round((v - lo) / (hi - lo) * last))] for v in pts)


def fmt(v):
    if v is None:
        return EMPTY
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if not isinstance(v, float):
        return str(v)
    av = abs(v)
    if av != 0 and (av < 1e-3 or av >= 1e5):
        return f"{v:.3e}"
    return f"{v:.4g}"


def collect_series(payload, key, name_filter):
    """Yield (name, steps, values) from runs[0][key]. Skips fully-empty series."""
    if not payload.get("runs"):
        return
    series_dict = payload["runs"][0].get(key, {})
    steps = series_dict.get("steps", [])
    for name in sorted(series_dict.keys()):
        if name == "steps":
            continue
        if name_filter and not name_filter.search(name):
            continue
        values = series_dict[name]
        if not any(_is_num(v) for v in values):
            continue
        yield name, steps, values


def build_rows(series_iter):
    rows = []
    for name, steps, values in series_iter:
        s = summarize(steps, values)
        rows.append(
            [
                name,
                s["n"],
                fmt(s["latest"]),
                fmt(s["first"]),
                fmt(s["min"]),
                fmt(s["max"]),
                fmt(s["slope_per_1k"]),
                s["missing"],
                sparkline(values),
            ]
        )
    return rows


def render(rows, title):
    if not rows:
        return
    headers = [
        "metric",
        "n",
        "latest",
        "first",
        "min",
        "max",
        "slope/1k",
        "missing",
        "spark",
    ]
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cols) for i in range(len(headers))]

    def fmt_row(r):
        out = []
        for i, (c, w) in enumerate(zip(r, widths)):
            out.append(
                str(c).ljust(w) if i in (0, len(headers) - 1) else str(c).rjust(w)
            )
        return "  ".join(out)

    print()
    print(title)
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def _series_to_summary_dict(payload, key, name_re):
    out = {}
    for name, steps, values in collect_series(payload, key, name_re):
        s = summarize(steps, values)
        s["spark"] = sparkline(values)
        out[name] = s
    return out


def _print_recent_runs(host):
    try:
        rl = fetch_json(host, "/api/runs")
    except Exception:
        return
    recents = [r for r in rl.get("runs", []) if r.get("num_steps", 0) > 0][:5]
    if not recents:
        return
    print("recent runs with data:", file=sys.stderr)
    for r in recents:
        marker = "*" if r["is_current"] else " "
        print(f"  {marker} {r['hash']}  steps={r['num_steps']}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"server host:port (default: {DEFAULT_HOST})",
    )
    p.add_argument("--run", help="run hash (default: current)")
    p.add_argument("--dynamics", action="store_true", help="also fetch /api/dynamics")
    p.add_argument("--filter", help="regex on series names")
    p.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    args = p.parse_args()

    name_re = re.compile(args.filter) if args.filter else None
    params = {"limit": 1000}
    if args.run:
        params["runs"] = args.run

    try:
        metrics = fetch_json(args.host, "/api/metrics", params)
    except urllib.error.URLError as e:
        print(f"error: could not reach {args.host}: {e}", file=sys.stderr)
        sys.exit(2)

    if metrics.get("status") != "ok":
        msg = metrics.get("message", "")
        print(f"metrics: {metrics.get('status')} ({msg})", file=sys.stderr)
        _print_recent_runs(args.host)
        sys.exit(1)

    meta = metrics["runs"][0]["metadata"]
    run_hash = meta["model_hash"]
    last_step = meta["last_step"]

    dynamics = None
    if args.dynamics:
        try:
            dynamics = fetch_json(args.host, "/api/dynamics", params)
        except urllib.error.URLError as e:
            print(f"warning: dynamics fetch failed: {e}", file=sys.stderr)

    if args.as_json:
        out = {
            "run": run_hash,
            "last_step": last_step,
            "metrics": _series_to_summary_dict(metrics, "metrics", name_re),
        }
        if dynamics and dynamics.get("status") == "ok":
            out["dynamics"] = _series_to_summary_dict(dynamics, "dynamics", name_re)
        json.dump(out, sys.stdout, indent=2, default=str)
        print()
        return

    print(f"run {run_hash} @ step {last_step}  (host {args.host})")
    metric_rows = build_rows(collect_series(metrics, "metrics", name_re))
    render(metric_rows, f"scalars ({len(metric_rows)})")
    if dynamics and dynamics.get("status") == "ok":
        dyn_rows = build_rows(collect_series(dynamics, "dynamics", name_re))
        render(dyn_rows, f"dynamics ({len(dyn_rows)})")


if __name__ == "__main__":
    main()
