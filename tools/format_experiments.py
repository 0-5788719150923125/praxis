#!/usr/bin/env python3
"""Rewrite the git-tracked experiment configs in the documented format.

The same pass ``./launch`` runs on startup, callable on its own so a
documentation change can be reviewed as a diff without starting a run.

    python tools/format_experiments.py            # rewrite what changed
    python tools/format_experiments.py --check    # report, change nothing

Only git-tracked files under experiments/ are considered. Everything else there
is gitignored, belongs to whoever wrote it, and is never touched.
"""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _praxis():
    """The annotated-config module, imported with this tool's flags hidden: ``praxis.cli``
    parses ``sys.argv`` on import, and ``--check`` abbreviates
    ``--checkpoint-every``."""
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        import praxis.cli.annotated_config as annotated

        return annotated
    finally:
        sys.argv = argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a file is not in the format, without writing",
    )
    ap.add_argument(
        "--experiments-dir",
        default=str(REPO_ROOT / "experiments"),
        help="directory to scan (default: experiments/)",
    )
    args = ap.parse_args()

    annotated = _praxis()
    parser = annotated.documentation_parser()
    paths = annotated.tracked_experiments(args.experiments_dir)
    if not paths:
        print("No git-tracked experiments found.")
        return 0

    if args.check:
        stale = []
        for path in paths:
            rendered = annotated.render_experiment_file(path, parser)
            if rendered is None:
                unknown = annotated.undocumentable_keys(
                    yaml.safe_load(path.read_text()) or {}, parser
                )
                print(f"skipped  {path} - nothing describes {', '.join(unknown)}")
                continue
            original, text = rendered
            print(f"{'stale' if text != original else 'ok':8} {path}")
            if text != original:
                stale.append(path)
        return 1 if stale else 0

    changed = annotated.format_tracked_experiments(parser, args.experiments_dir)
    for path in changed:
        print(f"Formatted {path}")
    print(f"{len(changed)} of {len(paths)} tracked experiments rewritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
