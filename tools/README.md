# tools

A shared toolbox of small CLI utilities, callable directly by both the human and the assistant (and, eventually, by Praxis models at inference time) so neither side has to relay numbers off charts or read/copy/paste data between contexts. Each tool is single-purpose and runs as `python tools/<name>.py`.

## Conventions

- **Naming**: verb-prefixed (`fetch_metrics.py`, `get_runs.py`, ...), not bare nouns. Tools read as actions; this also leaves room for them to land in a model-callable tool schema later without renaming.
- **Dependencies**: prefer stdlib. A tool should run from a clean checkout without forcing the user into the project venv or a heavy install.
- **Output**: human-readable by default, with a `--json` flag when the tool produces structured data the assistant might want to parse.
- **Scope**: small and focused. New capability = new file, not a new flag on an existing tool.

## Available tools

- `fetch_metrics.py` - hits the running web app's `/api/metrics` (and `/api/dynamics` with `--dynamics`) and prints a per-series summary table (n, latest, first, min/max, slope per 1k steps, missing count, unicode sparkline) for the current run or `--run <hash>`. Pass `--filter <regex>` to narrow series, `--json` for machine output, `--host host:port` to point at a non-default server.
- `export_runs.py` - reads the newest experiments under `build/runs/` directly (read-only, no server) and writes `research/variables.tex` (LaTeX macros) + `research/data/run_*.csv` so the paper plots recent validation curves as a living document. The current experiment (newest by `created`) always leads the chart and fixes the y-metric to whatever its own family reports (`val_bits_per_byte` / `val_brierlm` / `val_loss`; codec fidelity excluded from auto, reachable via `--metric`); only prior experiments reporting that same metric are drawn alongside, and the incompatible ones are listed as skipped. `--n <k>` (default 4), `--metric <col|auto>`, `--json`. Re-run before building the paper: `python tools/export_runs.py && (cd research && latexmk -pdf main.tex)`.
