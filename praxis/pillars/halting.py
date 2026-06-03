"""Halting-distribution figure: where the recurrent gear stops, per run.

The KL halting gate (praxis/halting/kl.py) turns the recurrent loop into a
sequence of non-differentiable decisions: at each interior loop boundary it
compares successive hidden states and stops early once the KL divergence between
them collapses (the field has stopped changing). This module reads a run's
logged halting metrics and plots the resulting distribution over the number of
turns r the model actually takes - the empirical shape of that decision.

Two series, mirroring the dashboard's Halting Distribution card:
- train: the random loop-count schedule the model is forced to learn under
  (log-normal Poisson), which teaches it to front-load computation.
- eval: where the KL gate actually stops, at inference.

Source: the ``halting/*`` keys in a run's metrics.db ``extra_metrics`` JSON.
Output: research/figures/halting.png + research/halting.tex (\\paperHaltingFigure).

Entry point: :func:`export_halting`, driven by :mod:`praxis.pillars.build`.
"""

import json
import os

from praxis.pillars.geometries import RESEARCH_DIR, FIG_DIR, runs_newest_first

OUT_TEX = os.path.join(RESEARCH_DIR, "halting.tex")


def _latest_halting(run_dir):
    """The newest logged ``halting/*`` metric dict for a run, or {}."""
    import sqlite3

    db = os.path.join(run_dir, "metrics.db")
    if not os.path.exists(db):
        return {}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT extra_metrics FROM metrics WHERE extra_metrics LIKE '%halting%' "
            "ORDER BY step DESC LIMIT 1"
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return {}
    if not row or not row[0]:
        return {}
    try:
        blob = json.loads(row[0])
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in blob.items() if k.startswith("halting/")}


def _find_halting_run():
    """(name, hash, metrics) of the newest run with halting data, or None."""
    for _, run_hash, name, run_dir in runs_newest_first():
        metrics = _latest_halting(run_dir)
        if metrics.get("halting/max_loops"):
            return name, run_hash, metrics
    return None


def _series(metrics, prefix, max_loops):
    """Counts [r=1..max_loops] for a train_/eval_ histogram, or None if empty."""
    counts = [
        float(metrics.get(f"halting/{prefix}_r_{r}", 0.0))
        for r in range(1, max_loops + 1)
    ]
    return counts if sum(counts) > 0 else None


def render_png(name, metrics):
    """Plot the train/eval halting distribution; return its figure path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    max_loops = int(metrics["halting/max_loops"])
    rs = list(range(1, max_loops + 1))
    train = _series(metrics, "train", max_loops)
    eval_ = _series(metrics, "eval", max_loops)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    x = np.arange(len(rs))
    bars = [(train, "schedule (train)", "#7a7a7a"), (eval_, "halts (eval)", "#d8541e")]
    present = [(c, lbl, col) for c, lbl, col in bars if c is not None]
    width = 0.8 / max(len(present), 1)
    for i, (counts, lbl, col) in enumerate(present):
        frac = [c / sum(counts) for c in counts]
        ax.bar(
            x + (i - (len(present) - 1) / 2) * width, frac, width, label=lbl, color=col
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("turns before halting (r)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    out = os.path.join(FIG_DIR, "halting.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(out, RESEARCH_DIR)


def figure_tex(path, name, metrics):
    """The ``\\paperHaltingFigure`` macro."""
    max_loops = int(metrics["halting/max_loops"])
    mean_kl = metrics.get("halting/eval_mean_kl")
    halt_rate = metrics.get("halting/halt_rate")
    bits = [f"the {max_loops}-turn recurrent loop of run {name}"]
    if halt_rate is not None:
        bits.append(f"early-halt rate {halt_rate * 100:.0f}\\%")
    if mean_kl is not None:
        bits.append(f"mean boundary KL {mean_kl:.3g}")
    caption = (
        "Halting distribution: the fraction of forward passes that stop after "
        f"r turns, for {', '.join(bits)}. \\emph{{schedule}} is the random "
        "loop-count the model trains under; \\emph{{halts}} is where the KL gate "
        "actually stops at inference. A mass pinned at the maximum (KL never "
        "collapsing) is the honest signature of a model that has not yet learned "
        "to converge early - the decision is real, not yet decisive."
    )
    return (
        "\\newcommand{\\paperHaltingFigure}{%\n"
        "\\begin{figure}[tbp]\n  \\centering\n  "
        f"\\includegraphics[width=0.62\\linewidth]{{{path}}}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:halting}\n"
        "\\end{figure}\n}\n"
    )


def export_halting() -> dict:
    """Render the halting figure for the newest run that has halting data.
    Writes an empty macro (paper builds without the figure) when none does."""
    found = _find_halting_run()
    if not found:
        with open(OUT_TEX, "w") as fh:
            fh.write(
                "% Generated by praxis/pillars/halting.py - no halting data found.\n"
                "\\newcommand{\\paperHaltingFigure}{}\n"
            )
        return {"run": None}

    name, run_hash, metrics = found
    path = render_png(name, metrics)
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by praxis/pillars/halting.py - do not edit by hand.\n")
        fh.write(figure_tex(path, name, metrics))
    return {
        "run": name,
        "hash": run_hash,
        "max_loops": int(metrics["halting/max_loops"]),
        "halt_rate": metrics.get("halting/halt_rate"),
        "eval_mean_kl": metrics.get("halting/eval_mean_kl"),
    }
