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
            "metric to BPB reported by the BLT paper. Not emitted for codec "
            "encoders (CALM); see val_codec_bpb and val_brierlm instead."
        ),
        "chart": {
            "title": "Bits per Byte",
            "y_label": "Bits per Byte",
            "y_scale": "linear",
            "order": 50,
            "is_validation": True,
        },
    },
    "val_codec_bpb": {
        "description": (
            "Codec reconstruction bits/byte for autoencoder encoders (CALM): "
            "teacher-forced encode-then-decode fidelity. This is NOT "
            "generation quality - it is near-zero for any working codec. "
            "Judge the model with val_brierlm, not this."
        ),
        "chart": {
            "title": "Codec Recon (bits/byte)",
            "y_label": "Codec bits/byte",
            "y_scale": "logarithmic",
            "order": 55,
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
    # Remote-expert pool (orchestration). Sampled cheaply at logging intervals
    # from a subset of the swarm's per-expert EMAs - not the main model's loss.
    # Only emitted when an expert pool is active (--orchestration-type).
    "swarm_loss": {
        "description": (
            "Mean training loss across a sampled subset of swarm experts "
            "(each tiny expert's own EMA on real batches). The swarm's "
            "population-level learning curve, distinct from the main model's."
        ),
        "chart": {
            "title": "Swarm Loss (sampled)",
            "y_label": "Swarm Loss",
            "y_scale": "linear",
            "order": 92,
            "is_validation": False,
        },
    },
    "swarm_loss_std": {
        "description": (
            "Std of expert EMA loss across the sample - how much the swarm "
            "disagrees. High = a diverse population; collapsing = consensus."
        ),
        "chart": {
            "title": "Swarm Loss Spread",
            "y_label": "Loss Std",
            "y_scale": "linear",
            "order": 93,
            "is_validation": False,
        },
    },
    "swarm_acc": {
        "description": "Mean next-token accuracy across the sampled swarm experts.",
        "chart": {
            "title": "Swarm Accuracy (sampled)",
            "y_label": "Accuracy",
            "y_scale": "linear",
            "order": 94,
            "is_validation": False,
        },
    },
    "swarm_experts": {
        "description": "Live expert count in the pool (grows as peers join).",
        "chart": {
            "title": "Swarm Experts",
            "y_label": "Experts",
            "y_scale": "linear",
            "order": 95,
            "is_validation": False,
            "type": "bar",
        },
    },
    # The following are persisted for record-keeping but don't currently
    # get their own Research-tab chart (no chart hint). They still flow
    # through the logger and API as named columns.
    "batch": {"description": "Current batch index."},
    "local_layers": {"description": "Number of layers on the local node."},
    "remote_layers": {"description": "Number of layers held on remote peers."},
    # Harmonic-weight RL controller (rl_type=harmonic_weight). Sparse: emitted
    # at each episode end, carried forward between episodes.
    "rl_reward": {
        "description": (
            "EMA-integrated return over the horizon: the loss improvement vs "
            "L_before, accumulated per post-edit step with a slow-decay EMA so "
            "a benefit that ramps in slowly still counts. Positive = the edit "
            "helped. Compare with rl_reward_instant (the one-step endpoint "
            "delta) to see delayed credit appear."
        ),
        "chart": {
            "title": "RL Reward",
            "y_label": "reward (return)",
            "y_scale": "linear",
            "order": 200,
            "is_validation": False,
        },
    },
    "rl_reward_instant": {
        "description": (
            "Endpoint delta L_before - L_after at the horizon's end - the old "
            "one-step reward, kept as a diagnostic. When it lags rl_reward, the "
            "edit's benefit is manifesting slowly (the reason for the EMA return)."
        ),
        "chart": {
            "title": "RL Reward (Endpoint)",
            "y_label": "Δloss (endpoint)",
            "y_scale": "linear",
            "order": 201,
            "is_validation": False,
        },
    },
    "rl_baseline": {
        "description": (
            "EMA reward baseline b; advantage = reward - b. Variance-reduction "
            "reference for the REINFORCE update."
        ),
        "chart": {
            "title": "RL Baseline",
            "y_label": "baseline",
            "y_scale": "linear",
            "order": 210,
            "is_validation": False,
        },
    },
    "rl_advantage": {
        "description": (
            "reward - baseline, the signed learning signal. Watch its scale "
            "and sign-flipping: wild swings are the credit-assignment problem."
        ),
        "chart": {
            "title": "RL Advantage",
            "y_label": "advantage",
            "y_scale": "linear",
            "order": 220,
            "is_validation": False,
        },
    },
    "rl_policy_loss": {
        "description": "REINFORCE objective (-log_prob*advantage - entropy bonus).",
        "chart": {
            "title": "RL Policy Loss",
            "y_label": "policy loss",
            "y_scale": "linear",
            "order": 230,
            "is_validation": False,
        },
    },
    "rl_entropy": {
        "description": (
            "Policy entropy. Collapsing toward 0 = exploration dying (the "
            "policy is committing); staying high = it hasn't learned to act."
        ),
        "chart": {
            "title": "RL Policy Entropy",
            "y_label": "entropy (nats)",
            "y_scale": "linear",
            "order": 240,
            "is_validation": False,
        },
    },
    "rl_log_std_mean": {
        "description": "Mean log-std of the Gaussian policy (exploration width).",
        "chart": {
            "title": "RL Policy log-std",
            "y_label": "mean log-std",
            "y_scale": "linear",
            "order": 250,
            "is_validation": False,
        },
    },
    "rl_action_alpha": {
        "description": "Last action: harmonic modulation depth applied to the row.",
        "chart": {
            "title": "RL Action: alpha",
            "y_label": "alpha",
            "y_scale": "linear",
            "order": 260,
            "is_validation": False,
        },
    },
    "rl_action_omega": {
        "description": "Last action: harmonic spatial frequency across the row.",
        "chart": {
            "title": "RL Action: omega",
            "y_label": "omega",
            "y_scale": "linear",
            "order": 270,
            "is_validation": False,
        },
    },
    "rl_action_phi": {
        "description": "Last action: harmonic phase offset.",
        "chart": {
            "title": "RL Action: phi",
            "y_label": "phi",
            "y_scale": "linear",
            "order": 280,
            "is_validation": False,
        },
    },
    "rl_edit_kept": {
        "description": (
            "Rolling fraction of edits kept (an EMA of the per-episode keep/roll-back "
            "decision) - how often the controller finds an edit that improves loss. "
            "Near 1: most proposals help; near 0: most are rolled back."
        ),
        "chart": {
            "title": "RL Edit Kept",
            "y_label": "keep rate",
            "y_scale": "linear",
            "order": 290,
            "is_validation": False,
        },
    },
    "rl_gate_frac": {
        "description": (
            "anchor_gate mode: fraction of the row's elements reset to the "
            "frozen anchor on the last edit (the gate density the policy chose)."
        ),
        "chart": {
            "title": "RL Gate Fraction",
            "y_label": "fraction reset",
            "y_scale": "linear",
            "order": 295,
            "is_validation": False,
        },
    },
}


# Composite / specialty Research-tab charts. Unlike the scalars above,
# these aren't single named columns: some are families of router-emitted
# keys matched by ``key_pattern`` (one series per expert/layer), and some
# come from a different endpoint (``source``). Declaring them here keeps
# the frontend free of hardcoded chart configs - it builds every chart
# from what these registries serve. Each entry's fields:
#
# * ``key``: logical id. For ``line`` charts this is the literal metric
#   name; for family charts it's just an identifier and ``key_pattern``
#   selects the underlying series.
# * ``type``: renderer the frontend dispatches on - ``line``, ``bar``,
#   ``sampling``, ``multi_expert_line``, or ``expert_routing_heatmap``.
# * ``title`` / ``y_label``: chart title and y-axis label.
# * ``source``: ``"metrics"`` (default) or ``"data_metrics"`` - which
#   endpoint the series come from.
# * ``key_pattern``: regex (string) matching a family of metric names.
# * ``stepped``: draw as a step plot (cumulative counts).
# * ``order``: ordering within the Research tab, after the scalars above.
COMPOSITE_METRIC_REGISTRY: list = [
    {
        "key": "sampling_weights",
        "type": "sampling",
        "title": "Task Sampling Weights",
        "y_label": "Sampling Weights",
        "source": "data_metrics",
        "order": 100,
    },
    {
        "key": "expert_routing_weights",
        "type": "expert_routing_heatmap",
        "title": "Expert Routing Weights (Convergence)",
        "y_label": "Routing Weight",
        "key_pattern": r"^layer_\d+_expert_\d+_routing_weight$",
        "stepped": True,
        "order": 110,
    },
    {
        "key": "expert_selection",
        "type": "multi_expert_line",
        "title": "Expert Selection (Actual k_experts Usage)",
        "y_label": "Selection Count",
        "key_pattern": r"^expert_selection/expert_\d+_count$",
        "stepped": True,
        "order": 120,
    },
    {
        "key": "routing/entropy",
        "type": "line",
        "title": "Routing Entropy (Balance)",
        "y_label": "Entropy",
        "order": 130,
    },
    {
        "key": "routing/concentration",
        "type": "line",
        "title": "Routing Concentration (Collapse)",
        "y_label": "Max Weight",
        "order": 140,
    },
    {
        "key": "routing/variance",
        "type": "line",
        "title": "Routing Variance (Stability)",
        "y_label": "Variance",
        "order": 150,
    },
    {
        "key": "routing/balance",
        "type": "line",
        "title": "Routing Balance",
        "y_label": "Balance",
        "order": 160,
    },
    {
        "key": "expert_importance",
        "type": "multi_expert_line",
        "title": "Expert Importance (Soft Routing Probabilities)",
        "y_label": "Importance",
        "key_pattern": r"^routing/expert_\d+_importance$",
        "order": 170,
    },
    {
        "key": "expert_load",
        "type": "multi_expert_line",
        "title": "Expert Load (Hard Routing Decisions)",
        "y_label": "Load",
        "key_pattern": r"^routing/expert_\d+_load$",
        "order": 180,
    },
    {
        "key": "routing/diversity_loss",
        "type": "line",
        "title": "Parameter Diversity Loss (Distance Router)",
        "y_label": "Diversity Loss",
        "order": 190,
    },
]


# Dynamics-tab chart families. These render gradient/halting/task-weight
# series logged to dynamics.db (and merged routing keys). Each is a family
# of per-layer / per-expert / per-bucket keys matched by ``key_pattern``;
# the frontend detects presence, extracts layer indices, and dispatches to
# a bespoke builder by ``type`` - all from this list, so the metric-name
# regexes no longer live in JS. Fields:
#
# * ``key`` / ``type``: identifier and the renderer the frontend selects.
# * ``title`` / ``subtitle``: card title and subtitle. The subtitle is a
#   fallback - a live ``metric_descriptions`` entry for ``key`` overrides it.
# * ``key_pattern``: regex (string) selecting the family's series.
# * ``layer_toggles``: series are per-layer and respond to the layer toggles.
# * ``legend``: render a scrollable legend under the chart.
# * ``order``: ordering within the Dynamics tab.
DYNAMICS_CHART_REGISTRY: list = [
    {
        "key": "layer_grad_norms",
        "type": "layer_grad_norms",
        "title": "Gradient Flow",
        "subtitle": "L2 norm of gradients per decoder layer",
        "key_pattern": r"^layer_\d+_grad_norm$",
        "layer_toggles": True,
        "legend": True,
        "order": 10,
        "caller": "LocalLayer",
        "caller_et_al": True,
    },
    {
        "key": "layer_update_ratio",
        "type": "layer_update_ratio",
        "title": "Update-to-Weight Ratio",
        "subtitle": "Relative update magnitude per layer (||grad|| &times; lr / ||weight||)",
        "key_pattern": r"^layer_\d+_update_ratio$",
        "layer_toggles": True,
        "legend": True,
        "order": 20,
        "caller": "LocalLayer",
        "caller_et_al": True,
    },
    {
        "key": "expert_grad_norms",
        "type": "expert_grad_norms",
        "title": "Gradient Norms per Expert",
        "subtitle": "L2 norm of gradients across all parameters",
        "key_pattern": r"^layer_\d+_expert_\d+_grad_norm$",
        "layer_toggles": True,
        "legend": True,
        "order": 30,
        "caller": "Router",
        "caller_et_al": True,
    },
    {
        "key": "expert_grad_vars",
        "type": "expert_grad_vars",
        "title": "Gradient Variance per Expert",
        "subtitle": "Variance of gradient values across all parameters",
        "key_pattern": r"^layer_\d+_expert_\d+_grad_var$",
        "layer_toggles": True,
        "legend": True,
        "order": 40,
        "caller": "Router",
        "caller_et_al": True,
    },
    {
        "key": "task_weights",
        "type": "task_weights",
        "title": "Task Loss Weights",
        "subtitle": "Per-task scalar multipliers applied to the loss.",
        "key_pattern": r"^task_weight_",
        "legend": True,
        "order": 50,
        "caller": "TaskLossWeighter",
    },
    {
        "key": "halting_hist",
        "type": "halting_hist",
        "title": "Halting Distribution",
        "subtitle": (
            "Loop counts used per forward pass. Training = random samples "
            "(log-normal Poisson); inference = where KL-halting actually fired."
        ),
        "key_pattern": r"^halting/(train|eval)_r_\d+$",
        # Rendered after the head-metric sections (manifest + snapshots).
        "order": 110,
        "caller": "Halting",
    },
]


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
