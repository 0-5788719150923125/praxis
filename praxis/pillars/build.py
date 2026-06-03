"""Build the living paper's generated inputs from current repository state.

The paper in ``research/`` ``\\input``s four generated files; this is the one
command that regenerates all of them, the research-side analogue of
``praxis/web/src/build.py``:

- ``variables.tex`` + ``data/run_*.csv`` - recent validation curves (:mod:`runs`)
- ``framing.tex`` - component-gated prose for the current run (:mod:`framing`)
- ``geometries.tex`` + ``figures/`` - Center PCA density figure (:mod:`geometries`)
- ``inlines.tex`` - single-value substitutions (:mod:`inlines`)

Run it, then build the PDF::

    python -m praxis.research.build
    (cd research && latexmk -pdf main.tex)

Each step is independent and best-effort: a step that finds nothing to render
(no runs yet, no crystal geometry) leaves the paper's fallbacks in place rather
than failing the build.
"""

import argparse
import sys

from praxis.research import framing, geometries, inlines, runs


def build_all(n=4, metric="auto", experiment=None, limit=4, scan=40, as_json=False):
    """Regenerate every paper input. Returns a dict of per-step summaries."""
    summary = {}

    result = runs.export_once(n, metric)
    if result:
        runs.report(result, as_json)
    else:
        print("runs: no usable validation data yet; kept existing variables.tex")
    summary["runs"] = result

    try:
        summary["framing"] = framing.export_framing(experiment)
        f = summary["framing"]
        print(f"framing: {', '.join(f['active']) or '(none)'} for '{f['experiment']}'")
    except (ValueError, FileNotFoundError) as e:
        print(f"framing: skipped ({e})")
        summary["framing"] = None

    summary["geometries"] = geometries.export_geometries(limit, scan)
    print(f"geometries: {summary['geometries']['count']} panel(s)")

    summary["inlines"] = inlines.export_inlines()
    res = summary["inlines"]["resolved"]
    print(f"inlines: {', '.join(f'{k}={v}' for k, v in res.items()) or '(none)'}")

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=4, help="run curves to include (default 4)")
    ap.add_argument("--metric", default="auto", help="metric column, or 'auto'")
    ap.add_argument("--experiment", help="experiment to frame (default: newest run)")
    ap.add_argument("--limit", type=int, default=4, help="max geometry panels (default 4)")
    ap.add_argument("--scan", type=int, default=40, help="runs to scan for geometries")
    ap.add_argument("--json", action="store_true", help="machine-readable run summary")
    args = ap.parse_args()

    build_all(args.n, args.metric, args.experiment, args.limit, args.scan, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
