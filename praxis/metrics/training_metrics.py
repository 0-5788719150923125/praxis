"""Single source of truth for the Research-tab scalar training metrics.

Each entry declares one metric that the trainer can emit; the rest of
the stack (SQLite schema, ``MetricsLogger.log()`` column dispatch, the
``/api/metrics`` SELECT statements, the validation-row preservation
checks, and the frontend chart configs) all derive their column lists
from this registry. Adding a new training metric is a one-entry change
here plus an emit at the trainer; no JS, SQL, or backfill edits.

Schema mirrors :data:`praxis.metrics.descriptions` for the dashboard's
head-side metrics. Each entry's ``chart`` hint may carry:

* ``title``: chart title text
* ``y_label``: y-axis label
* ``y_scale``: ``"linear"`` (default) or ``"logarithmic"``
* ``order``: integer ordering within the Research tab
* ``is_validation``: bool. Validation rows are sparse (one point every
  N steps) so the LTTB downsampler force-preserves them and the
  frontend dedups consecutive-equal values that Lightning's
  ``callback_metrics`` persistence smears across training rows.
"""

from typing import Any, Dict


TRAINING_METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "loss": {
        "description": "Per-step training cross-entropy loss.",
        "chart": {
            "title": "Training Loss",
            "y_label": "Training Loss",
            "y_scale": "linear",
            "order": 10,
            "is_validation": False,
        },
    },
    "val_loss": {
        "description": (
            "Cross-entropy on the validation set, emitted every "
            "``val_check_interval`` steps."
        ),
        "chart": {
            "title": "Validation Loss",
            "y_label": "Validation Loss",
            "y_scale": "linear",
            "order": 20,
            "is_validation": True,
        },
    },
    "val_perplexity": {
        "description": "exp(val_loss). Token-vocab runs only.",
        "chart": {
            "title": "Perplexity",
            "y_label": "Perplexity",
            "y_scale": "linear",
            "order": 30,
            "is_validation": True,
        },
    },
    "val_brierlm": {
        "description": (
            "BrierLM score over a small validation batch - bounded "
            "proper scoring rule, less sensitive to outliers than NLL."
        ),
        "chart": {
            "title": "BrierLM",
            "y_label": "BrierLM",
            "y_scale": "linear",
            "order": 40,
            "is_validation": True,
        },
    },
    "val_bits_per_byte": {
        "description": (
            "val_loss / log(2). Byte-latent runs only - the comparable "
            "metric to BPB reported by the BLT paper."
        ),
        "chart": {
            "title": "Bits per Byte",
            "y_label": "Bits per Byte",
            "y_scale": "linear",
            "order": 50,
            "is_validation": True,
        },
    },
    "learning_rate": {
        "description": "Optimizer learning rate at each step.",
        "chart": {
            "title": "Learning Rate",
            "y_label": "Learning Rate",
            "y_scale": "linear",
            "order": 60,
            "is_validation": False,
        },
    },
    "num_tokens": {
        "description": "Cumulative number of training tokens seen so far.",
        "chart": {
            "title": "Tokens (Billions)",
            "y_label": "Tokens (B)",
            "y_scale": "linear",
            "order": 70,
            "is_validation": False,
            "type": "bar",
        },
    },
    "avg_step_time": {
        "description": "EMA of seconds per optimizer step.",
        "chart": {
            "title": "Average Step Time",
            "y_label": "Avg Step Time (s)",
            "y_scale": "linear",
            "order": 80,
            "is_validation": False,
        },
    },
    "softmax_collapse": {
        "description": (
            "Fraction of softmax distributions whose top probability "
            "exceeds 0.999. Rising = the model is overcommitting to "
            "single tokens."
        ),
        "chart": {
            "title": "Softmax Collapse",
            "y_label": "Softmax Collapse",
            "y_scale": "linear",
            "order": 90,
            "is_validation": False,
        },
    },
    # The following are persisted for record-keeping but don't currently
    # get their own Research-tab chart (no chart hint). They still flow
    # through the logger and API as named columns.
    "batch": {"description": "Current batch index."},
    "local_layers": {"description": "Number of layers on the local node."},
    "remote_layers": {"description": "Number of layers held on remote peers."},
}


def metric_names() -> list:
    """Ordered list of column names backing the registry."""
    return list(TRAINING_METRIC_REGISTRY.keys())


def validation_metric_names() -> list:
    """Metric keys whose ``chart.is_validation`` flag is true."""
    return [
        key
        for key, entry in TRAINING_METRIC_REGISTRY.items()
        if entry.get("chart", {}).get("is_validation")
    ]
