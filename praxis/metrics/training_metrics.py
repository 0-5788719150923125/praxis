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
    # Background web spider (praxis.spider). Counters mirrored from spider.db
    # by SpiderCallback at logging intervals. Only emitted when --spider is on.
    "spider_pages": {
        "description": (
            "Pages currently held in spider.db across all watched sites - the "
            "spider's grounded corpus, capped per site."
        ),
        "chart": {
            "title": "Spider Pages Held",
            "y_label": "Pages",
            "y_scale": "linear",
            "order": 96,
            "is_validation": False,
        },
    },
    "spider_new_pages": {
        "description": (
            "Cumulative never-before-seen pages fetched. The slope is the "
            "discovery rate; it flattens as watched sites are fully walked."
        ),
        "chart": {
            "title": "Spider Discoveries",
            "y_label": "New Pages",
            "y_scale": "linear",
            "order": 97,
            "is_validation": False,
        },
    },
    "spider_revisits": {
        "description": (
            "Cumulative re-fetches of known pages (content refreshes plus "
            "cheap 304s). Rises once the frontier dries up - the eventually-"
            "consistent watch phase."
        ),
        "chart": {
            "title": "Spider Revisits",
            "y_label": "Revisits",
            "y_scale": "linear",
            "order": 98,
            "is_validation": False,
        },
    },
    "spider_frontier": {
        "description": (
            "URLs queued for a first fetch across all sites. Growth means "
            "discovery is outpacing the one-fetch-per-tick budget."
        ),
        "chart": {
            "title": "Spider Frontier",
            "y_label": "Queued URLs",
            "y_scale": "linear",
            "order": 99,
            "is_validation": False,
        },
    },
    "spider_sites": {
        "description": (
            "Enabled watched sites. Grows when a widely-cited external site "
            "is promoted into a free watchlist slot."
        ),
        "chart": {
            "title": "Spider Watchlist",
            "y_label": "Sites",
            "y_scale": "linear",
            "order": 100,
            "is_validation": False,
            "type": "bar",
        },
    },
    "kb_size_mb": {
        "description": (
            "On-disk size of the knowledge base (FTS index + spider store, with "
            "write-ahead logs) in MB. Rises as the spider grounds more pages; "
            "a monitor for crawl storage growth."
        ),
        "chart": {
            "title": "Knowledge Base Size",
            "y_label": "MB",
            "y_scale": "linear",
            "order": 101,
            "is_validation": False,
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
    # Engagement-prediction policy (rl_type=engagement). The headline learning
    # signals: is the model anticipating its own answers, and is the homeostatic
    # energy climbing toward its setpoint?
    "engagement_energy": {
        "description": (
            "Homeostatic energy: fast-accumulating, satiating, wall-clock "
            "decaying (1h half-life). Climbs as the policy lands predicted "
            "answers and live interactions arrive; folded into the RL reward. "
            "Flat near 0 = not learning."
        ),
        "chart": {
            "title": "Engagement Energy",
            "y_label": "energy",
            "y_scale": "linear",
            "order": 300,
            "is_validation": False,
        },
    },
    "engagement_activation_rate": {
        "description": (
            "Fraction of examples where any predicted answer token is mentioned "
            "in the response ('answered at all') - the headline reward activation."
        ),
        "chart": {
            "title": "Engagement Activation Rate",
            "y_label": "activation rate",
            "y_scale": "linear",
            "order": 310,
            "is_validation": False,
        },
    },
    "engagement_recall": {
        "description": (
            "Graded recall |A_hat & R| / |A_hat| over the answer tokens - the "
            "smoother reward the policy gradient actually optimizes."
        ),
        "chart": {
            "title": "Engagement Recall",
            "y_label": "recall",
            "y_scale": "linear",
            "order": 320,
            "is_validation": False,
        },
    },
    "engagement_reward": {
        "description": (
            "Total REINFORCE reward: recall + homeostatic energy. Live Gymnasium "
            "interactions spike the energy term, then it decays back - a "
            "transient reward pulse on top of the dense recall signal."
        ),
        "chart": {
            "title": "Engagement Reward",
            "y_label": "reward",
            "y_scale": "linear",
            "order": 322,
            "is_validation": False,
        },
    },
    "engagement_reward_baseline": {
        "description": (
            "Slow EMA of the total reward (recall + energy) - the REINFORCE "
            "variance-reduction baseline. Advantages are reward minus this, so "
            "they stay zero-mean."
        ),
        "chart": {
            "title": "Engagement Reward Baseline",
            "y_label": "recall EMA",
            "y_scale": "linear",
            "order": 325,
            "is_validation": False,
        },
    },
    "engagement_advantage": {
        "description": (
            "reward - reward-EMA baseline, the signed REINFORCE signal. Zero-mean "
            "by construction: positive on better-than-recent predictions, negative "
            "on worse, balanced so it never systematically suppresses tokens."
        ),
        "chart": {
            "title": "Engagement Advantage",
            "y_label": "advantage",
            "y_scale": "linear",
            "order": 330,
            "is_validation": False,
        },
    },
    # Live (real-user) `Print` rewards drained from the web UI into training.
    "engagement_live_reward": {
        "description": (
            "Recall of the most recent live interaction: a real user answered a "
            "model-led question and this is how well the model predicted it."
        ),
        "chart": {
            "title": "Engagement Live Reward",
            "y_label": "recall",
            "y_scale": "linear",
            "order": 340,
            "is_validation": False,
        },
    },
    "engagement_live_count": {
        "description": (
            "Cumulative count of live `Print` interactions consumed by training - "
            "how much real-user signal the online channel has delivered."
        ),
        "chart": {
            "title": "Engagement Live Interactions",
            "y_label": "count",
            "y_scale": "linear",
            "order": 350,
            "is_validation": False,
        },
    },
    "engagement_live_energy": {
        "description": (
            "Homeostatic energy of the live channel (real-user activations only) - "
            "the slow online signal folded into the policy's baseline."
        ),
        "chart": {
            "title": "Engagement Live Energy",
            "y_label": "energy",
            "y_scale": "linear",
            "order": 360,
            "is_validation": False,
        },
    },
    # Joke task (rl_type=joke): same recall machinery as engagement, dense
    # grounding from well-rated jokes, live signal from human approval (Loop UI).
    "joke_energy": {
        "description": (
            "Homeostatic energy of the joke policy - climbs as the model "
            "reproduces well-rated jokes and earns live human approval."
        ),
        "chart": {
            "title": "Joke Energy",
            "y_label": "energy",
            "y_scale": "linear",
            "order": 400,
            "is_validation": False,
        },
    },
    "joke_recall": {
        "description": "Recall over joke tokens - how well the model reproduces the rated joke.",
        "chart": {
            "title": "Joke Recall",
            "y_label": "recall",
            "y_scale": "linear",
            "order": 410,
            "is_validation": False,
        },
    },
    "joke_reward": {
        "description": "Total joke REINFORCE reward: recall + homeostatic energy (live approvals spike it).",
        "chart": {
            "title": "Joke Reward",
            "y_label": "reward",
            "y_scale": "linear",
            "order": 412,
            "is_validation": False,
        },
    },
    "joke_reward_baseline": {
        "description": "Slow EMA of the total joke reward (recall + energy) - the zero-mean REINFORCE baseline.",
        "chart": {
            "title": "Joke Reward Baseline",
            "y_label": "recall EMA",
            "y_scale": "linear",
            "order": 415,
            "is_validation": False,
        },
    },
    "joke_advantage": {
        "description": "reward - reward-EMA baseline for the joke policy (zero-mean REINFORCE signal).",
        "chart": {
            "title": "Joke Advantage",
            "y_label": "advantage",
            "y_scale": "linear",
            "order": 420,
            "is_validation": False,
        },
    },
    "joke_live_reward": {
        "description": "Most recent live human approval of a model-generated joke (Loop UI).",
        "chart": {
            "title": "Joke Live Approval",
            "y_label": "approval",
            "y_scale": "linear",
            "order": 430,
            "is_validation": False,
        },
    },
    "joke_live_count": {
        "description": "Cumulative live joke approvals consumed by training.",
        "chart": {
            "title": "Joke Live Interactions",
            "y_label": "count",
            "y_scale": "linear",
            "order": 440,
            "is_validation": False,
        },
    },
    "joke_live_correction": {
        "description": (
            "Calibration error of the most recent live loop interaction: |user "
            "score - model's self-predicted want->need score| (0..2). The "
            "decisive calibration-mode metric - shrinking = the model is "
            "learning to predict its own reception."
        ),
        "chart": {
            "title": "Joke Live Correction",
            "y_label": "correction",
            "y_scale": "linear",
            "order": 445,
            "is_validation": False,
        },
    },
    "joke_live_energy": {
        "description": "Homeostatic energy of the live joke-approval channel.",
        "chart": {
            "title": "Joke Live Energy",
            "y_label": "energy",
            "y_scale": "linear",
            "order": 450,
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
        # Repo-level, not per-run: the framework's own git-churn evolution.
        # source "standalone" -> the card fetches its own data (/api/evolution),
        # the SAME computation the LaTeX figure renders. Always shown.
        "key": "evolution",
        "type": "evolution",
        "title": "Praxis Evolution (self-history)",
        "description": (
            "Per-subsystem line churn over Praxis's git history, faded by "
            "distance from HEAD. The framework's recency kernel turned on its "
            "own development - the same data the research-paper figure renders."
        ),
        "source": "standalone",
        "order": 90,
    },
    {
        # Spider link graph: the most-cited URLs and busiest referrer pages
        # from spider.db's refs table - the same counts that rank the crawl
        # frontier. source "standalone" -> the card fetches /api/spider.
        "key": "spider_citations",
        "type": "spider_citations",
        "title": "Spider Citations",
        "description": (
            "Top cited URLs (and top referrers) in the spider's link graph. "
            "Citation count is the frontier's ranking signal: well-referenced "
            "links are fetched first, one-off links sink."
        ),
        "source": "standalone",
        "order": 95,
    },
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
    # SMEAR routers log entropy/concentration/variance per *depth* (one key
    # per layer the shared router runs at, hence the layer_N_ prefix), unlike
    # Prismatic's single global routing/* scalars above. multi_expert_line
    # draws one line per matched key; these cards auto-hide until a SMEAR
    # router is active and emitting (availableMetrics gates on key presence).
    {
        "key": "smear_routing_entropy",
        "type": "multi_expert_line",
        "title": "SMEAR Routing Entropy (Balance)",
        "y_label": "Entropy",
        "description": (
            "Per-depth routing entropy for SMEAR routers (one line per layer). "
            "High = balanced soft-merge across experts; low = the router is "
            "collapsing onto a single expert."
        ),
        "key_pattern": r"^layer_\d+_routing_entropy$",
        "order": 200,
    },
    {
        "key": "smear_routing_concentration",
        "type": "multi_expert_line",
        "title": "SMEAR Routing Concentration (Collapse)",
        "y_label": "Max weight",
        "description": (
            "Per-depth maximum routing weight for SMEAR routers (one line per "
            "layer). 1.0 = collapsed onto one expert; 1/num_experts = uniform "
            "merge."
        ),
        "key_pattern": r"^layer_\d+_routing_concentration$",
        "order": 210,
    },
    {
        "key": "smear_routing_variance",
        "type": "multi_expert_line",
        "title": "SMEAR Load-Balance Variance (normalized)",
        "y_label": "Variance [0,1]",
        "description": (
            "Per-depth variance of the BATCH-MEAN routing weights, normalized to "
            "[0,1] (0 = balanced load across experts, 1 = collapsed onto one "
            "expert). This is LOAD BALANCE, not per-sequence specialization - it "
            "stays near 0 even when VEAR is sharply committing each sequence "
            "(different sequences pick different experts, averaging back to "
            "uniform). Watch Routing Specialization for that."
        ),
        "key_pattern": r"^layer_\d+_routing_variance$",
        "order": 220,
    },
    {
        "key": "smear_routing_specialization",
        "type": "multi_expert_line",
        "title": "SMEAR/VEAR Routing Specialization (per-sequence)",
        "y_label": "Specialization [0,1]",
        "description": (
            "Per-sequence routing commitment, computed BEFORE the batch-mean and "
            "rescaled to [0,1]: 0 = uniform routing, 1 = every sequence commits to "
            "a single expert. This is the gauge VEAR's sharpening + repulsion "
            "actually move; it rises as experts specialize even when load stays "
            "balanced (which the load-balance cards cannot show)."
        ),
        "key_pattern": r"^layer_\d+_routing_specialization$",
        "order": 230,
    },
    {
        "key": "smear_routing_peak",
        "type": "multi_expert_line",
        "title": "SMEAR/VEAR Routing Peak (mean per-sequence top weight)",
        "y_label": "Mean peak weight",
        "description": (
            "Mean per-sequence maximum routing weight (1/num_experts = uniform .. "
            "1.0 = each sequence fully committed). The raw per-sequence "
            "concentration behind Routing Specialization."
        ),
        "key_pattern": r"^layer_\d+_routing_peak$",
        "order": 240,
    },
    {
        "key": "depth_step",
        "type": "multi_expert_line",
        "title": "Depth Trajectory: Step Size (spectral-attractor probe)",
        "y_label": "Relative step",
        "description": (
            "Per recurrent-depth transition, the relative move of the hidden "
            "state's fingerprint (mean over batch+seq). The conjecture's signature "
            "(next/harmonic_memory_velocity.md): a fixed-point iteration should "
            "show steps shrinking toward zero (settling to a spectral attractor), "
            "ideally as a few discrete hops rather than a smooth drift. One line "
            "per depth transition."
        ),
        "key_pattern": r"^depth/step_d\d+$",
        "order": 250,
    },
    {
        "key": "depth/convergence_ratio",
        "type": "line",
        "title": "Depth Convergence Ratio (settling)",
        "y_label": "last step / first step",
        "description": (
            "Ratio of the last to the first depth-transition step. < 1 = the "
            "iteration settles toward a fixed point across depth (spectral "
            "attractor); ~ 1 = no convergence; > 1 = diverging. Should fall over "
            "training if the model learns to compute to a cluster within budget."
        ),
        "order": 251,
    },
    {
        "key": "depth/jump_concentration",
        "type": "line",
        "title": "Depth Jump Concentration (discrete vs smooth)",
        "y_label": "max step / mean step",
        "description": (
            "How concentrated the depth movement is: high = one large hop then "
            "settle (discrete jump between clusters, the conjectured mechanism); "
            "~ 1 = uniform movement (smooth drift). The static-FFT periodicity "
            "test was the wrong shape; this is the dynamics-over-depth version."
        ),
        "order": 252,
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
        "key": "seq_length_mix",
        "type": "seq_mix",
        "title": "Sequence Length Mix",
        "subtitle": (
            "Learned sampling probability over the sequence-length multipliers "
            "(constant token count). The adaptive curriculum samples more of the "
            "length the model is improving fastest on; only present when "
            "seq_curriculum=adaptive."
        ),
        "key_pattern": r"^seq_prob_x\d+$",
        "legend": True,
        "order": 55,
        "caller": "SequenceCurriculum",
    },
    {
        "key": "width_profile",
        "type": "width_profile",
        "title": "Width Profile",
        "subtitle": (
            "Active fraction of each block's inner width per recurrent depth - "
            "inflating early, decaying through the tail (latest step)."
        ),
        "key_pattern": r"^width/active_d\d+$",
        "order": 105,
        "caller": "MixtureOfWidths",
    },
    {
        "key": "width_evolution",
        "type": "width_evolution",
        "title": "Width Evolution",
        "subtitle": (
            "Per-depth active width over training (faint strata = the arch) with "
            "the realized mean actually used each forward (bold) - it wanders as "
            "halting samples how deep the loop runs."
        ),
        "key_pattern": r"^width/(active_d\d+|realized_mean)$",
        "legend": True,
        "order": 106,
        "caller": "MixtureOfWidths",
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
