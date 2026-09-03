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
* ``y_clip_percentile``: float. Bound the y axis at this percentile of
  the plotted values instead of at the maximum, so a startup transient
  cannot set the scale for the whole run. Outliers are not removed -
  the line runs off the top edge and the tooltip still reports the true
  value. Only for metrics whose tail is noise; omit wherever the
  extremes ARE the signal.
* ``smooth``: bool. Draw a rolling-median trend line with the raw
  series kept faint behind it. For per-step metrics too jagged to read
  at full length.
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
            # An untrained model's first few steps can log a loss in the
            # thousands - one run opened at 3484 and was under 60 within a
            # dozen steps. On a max-bounded axis those dozen points flatten
            # the other sixteen thousand into a single line at the bottom.
            # p99.5 keeps essentially the whole run in frame (that run's p99.5
            # is 27.4, and only 7 of 16720 points exceed 100); the transient
            # still draws, running off the top edge where it reads as an
            # excursion rather than as the axis.
            "y_clip_percentile": 99.5,
            # Per-step CE is genuinely noisy - in the settled region of that
            # same run, consecutive steps differ by 2.45 on average against a
            # mean of 7.56. The trend is unreadable underneath that.
            "smooth": True,
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
            "BrierLM on a small validation batch - a bounded proper score, less "
            "outlier-sensitive than NLL."
        ),
        "chart": {
            "title": "BrierLM",
            "y_label": "BrierLM",
            "y_scale": "linear",
            "order": 40,
            "is_validation": True,
        },
    },
    # Per-order Brier-n breakdown behind val_brierlm. Chartless columns (no
    # individual chart): they render together as the "BrierLM by n-gram Order"
    # composite below. Raw, scaled x100 to the aggregate's units; written by
    # BrierLMCallback on the same validation step as val_brierlm, so they ride
    # its preserved validation rows (no separate is_validation needed).
    "val_brier_1": {"description": "BrierLM order-1 (unigram) raw score, x100."},
    "val_brier_2": {"description": "BrierLM order-2 raw score, x100."},
    "val_brier_3": {"description": "BrierLM order-3 raw score, x100."},
    "val_brier_4": {"description": "BrierLM order-4 raw score, x100."},
    "val_bits_per_byte": {
        "description": (
            "val_loss / log(2). Byte-latent runs only; codec encoders report "
            "val_codec_bpb instead."
        ),
        "chart": {
            "title": "Bits per Byte",
            "y_label": "Bits per Byte",
            "y_scale": "linear",
            "order": 50,
            "is_validation": True,
        },
    },
    "val_byte_nll_bits": {
        "description": (
            "Plain per-byte cross-entropy of the emitted logits, in bits - measured, "
            "never optimized. Chance is 8.0. Absent under cut-CE."
        ),
        "chart": {
            "title": "Byte NLL (bits)",
            "y_label": "Bits per byte",
            "y_scale": "linear",
            "order": 51,
            "is_validation": True,
        },
    },
    "val_codec_bpb": {
        "description": (
            "Teacher-forced reconstruction bits/byte for codec encoders (CALM). "
            "Fidelity, not generation quality."
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
    "inference_time": {
        "description": (
            "EMA of seconds per served generation request. These run inside the "
            "training loop, so it is time not spent stepping."
        ),
        "chart": {
            "title": "Inference Time",
            "y_label": "Seconds / Request",
            "y_scale": "linear",
            "order": 81,
            "is_validation": False,
            # Request shapes differ by two orders of magnitude (a 3-byte
            # terminal tick against a 256-byte chat turn), so the raw series
            # is a comb. The trend is the readable part.
            "smooth": True,
        },
    },
    "inference_rate": {
        "description": (
            "EMA of tokens per second across served generation requests - separates a "
            "slow request from a merely long one."
        ),
        "chart": {
            "title": "Inference Rate",
            "y_label": "Tokens / Second",
            "y_scale": "linear",
            "order": 82,
            "is_validation": False,
            "smooth": True,
        },
    },
    "softmax_collapse": {
        "description": (
            "Fraction of softmax distributions with top probability above 0.999. "
            "Rising = overcommitting to single tokens."
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
            "Mean EMA training loss across a sample of swarm experts - the "
            "population's learning curve, not the main model's."
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
            "Spread of expert EMA loss across the sample. High = a diverse population; "
            "falling = consensus."
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
            "Cumulative never-before-seen pages fetched. The slope is the discovery "
            "rate."
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
            "Cumulative re-fetches of known pages, including cheap 304s. Rises once "
            "the frontier dries up."
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
            "URLs queued for a first fetch. Growth means discovery outpaces the "
            "per-tick fetch budget."
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
            "On-disk size of the knowledge base (FTS index + spider store) in MB."
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
            "EMA-integrated loss improvement over the post-edit horizon, so slow "
            "benefits still count. Positive = the edit helped."
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
            "One-step endpoint delta L_before - L_after. Lagging rl_reward means the "
            "benefit arrives slowly."
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
            "reward - baseline, the signed learning signal. Wild sign-flipping is the "
            "credit-assignment problem."
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
            "Policy entropy. Falling to 0 = exploration dying; staying high = the "
            "policy hasn't learned to act."
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
            "Rolling fraction of proposed edits kept rather than rolled back. Near 1: "
            "most help; near 0: most are reverted."
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
            "anchor_gate mode: fraction of the row reset to the frozen anchor on the "
            "last edit."
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
            "Homeostatic energy: satiating, 1h half-life decay. Climbs as predicted "
            "answers land; folded into the RL reward."
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
            "Fraction of examples where any predicted answer token appears in the "
            "response - 'answered at all'."
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
            "Graded recall over the predicted answer tokens - the smooth signal the "
            "policy gradient optimizes."
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
            "Total REINFORCE reward: recall + homeostatic energy. Live interactions "
            "spike the energy term, then it decays."
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
            "Slow EMA of the total reward - the REINFORCE variance-reduction baseline."
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
            "reward - baseline, the signed REINFORCE signal. Zero-mean by "
            "construction."
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
            "Recall on the most recent live interaction - how well the model predicted "
            "a real user's answer."
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
        "description": "Cumulative live `Print` interactions consumed by training.",
        "chart": {
            "title": "Engagement Live Interactions",
            "y_label": "count",
            "y_scale": "linear",
            "order": 350,
            "is_validation": False,
        },
    },
    "engagement_live_energy": {
        "description": "Homeostatic energy from real-user activations only.",
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
            "Homeostatic energy of the joke policy - climbs on well-rated jokes and "
            "live approval."
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
            "Calibration error of the last live interaction: |user score - model's "
            "self-predicted score| (0..2). Shrinking = better calibrated."
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
    # ── Information density at the rim (see praxis/metrics/density.py) ───────
    # Whole-sequence readout by position: from a single hidden state at each
    # position, how much of the entire window (at three resolutions) a linear
    # readout recovers, scored prequentially and reported as R² above a
    # shuffled-target null. All keys live in extra_metrics; nothing here adds
    # a schema column. The primary card is a heatmap because the paper's own
    # figure (fig:density) IS a shaded strip over position: x = position bucket
    # head..tip, y = band, shade = R². Eight-line profile cards were tried and
    # were unreadable.
    {
        "key": "readout_profile",
        "type": "expert_routing_heatmap",
        "title": "Whole-Sequence Readout by Position",
        "y_label": "R\u00b2 above chance",
        "description": (
            "Per position (x) and band (y: bag, coarse, mid), how much of the whole "
            "window a linear readout recovers from that one hidden state. R² above a "
            "shuffled null."
        ),
        # Group 1 = row label (rendered along x), group 2 = column index (y).
        "key_pattern": r"^readout_cell_([a-z0-9]+)_(\d)$",
        "row_label": "Position in window (head \u2192 tip)",
        "col_label": "Band (0 bag, 1 coarse, 2 mid)",
        "uniform_note": False,
        "order": 191,
    },
    {
        "key": "readout_rim_gap",
        "type": "multi_expert_line",
        "title": "Whole-Sequence Readout: Tip minus Head",
        "y_label": "R\u00b2(tip) - R\u00b2(head)",
        "description": (
            "Readout R² at the tip minus at the head, per band. Large and positive = "
            "only the tip has seen the window; closing = the head anticipates it."
        ),
        "key_pattern": r"^readout_(bag|coarse|mid)_rim_gap$",
        "legend": True,
        "order": 192,
    },
    {
        "key": "readout_depth_gain",
        "type": "multi_expert_line",
        "title": "Whole-Sequence Readout: Gain over Depth",
        "y_label": "R\u00b2(exit) - R\u00b2(entry)",
        "description": (
            "Readout R² at loop exit minus at loop entry, averaged over positions. "
            "Zero = the depth loop adds nothing a linear readout can see."
        ),
        "key_pattern": r"^readout_(bag|coarse|mid)_depth_gain$",
        "legend": True,
        "order": 193,
    },
    {
        # Repo-level, not per-run: the framework's own git-churn evolution.
        # source "standalone" -> the card fetches its own data (/api/evolution),
        # the SAME computation the LaTeX figure renders. Always shown.
        "key": "evolution",
        "type": "evolution",
        "title": "Praxis Evolution (self-history)",
        "description": (
            "Per-subsystem line churn over Praxis's git history, faded by distance "
            "from HEAD."
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
            "Top cited URLs and referrers in the spider's link graph. Citation count "
            "ranks the fetch frontier."
        ),
        "source": "standalone",
        "order": 95,
    },
    {
        # Per-order Brier-n curves behind val_brierlm, on one chart. Each order
        # is a smooth, rarely-zero diagnostic; the aggregate (val_brierlm) is
        # their floored geometric mean, so this shows which order collapses it.
        "key": "val_brier_orders",
        "type": "multi_expert_line",
        "title": "BrierLM by n-gram Order",
        "y_label": "Brier-n (x100)",
        "key_pattern": r"^val_brier_[1-4]$",
        "series_noun": "Order",
        "order": 45,
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
        # Row = decoder layer, column = expert. The capture groups are what the
        # heatmap renderer reads; without them it can form no grid.
        #
        # No live router emits this key shape any more - it survives so the
        # runs that DO carry it stay readable, and auto-hides everywhere else.
        "key_pattern": r"^layer_(\d+)_expert_(\d+)_routing_weight$",
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
    # Depth-aware routing (arc_smear / arc_vear). Auto-hides for the plain
    # routers, which emit neither key.
    {
        "key": "router_depth_bias",
        "type": "multi_expert_line",
        "title": "Router Depth Specialization",
        "y_label": "specialization / cosine",
        "description": (
            "Per-pass router bias under arc_smear/arc_vear: between-depth variance (0 "
            "= every pass routes alike) and mean pairwise cosine. Both start at 0."
        ),
        "key_pattern": r"^router_depth_(specialization|similarity)$",
        "order": 199,
    },
    {
        "key": "smear_routing_entropy",
        "type": "multi_expert_line",
        "title": "SMEAR Routing Entropy (Balance)",
        "y_label": "Entropy",
        "description": (
            "Entropy of the batch-mean routing weights, per layer. High = load spread "
            "across experts; low = the whole batch lands on one."
        ),
        "key_pattern": r"^layer_\d+_routing_entropy$",
        "order": 200,
    },
    {
        "key": "smear_routing_entropy_seq",
        "type": "multi_expert_line",
        "title": "SMEAR Routing Entropy (per-sequence)",
        "y_label": "Entropy",
        "description": (
            "Mean entropy of a single sequence's routing distribution. Equal to the "
            "balance card = every sequence routes alike; a gap = they differ."
        ),
        "key_pattern": r"^layer_\d+_routing_entropy_seq$",
        "order": 205,
    },
    {
        "key": "smear_routing_input_dependence",
        "type": "multi_expert_line",
        "title": "SMEAR/VEAR Routing Input Dependence",
        "y_label": "I(input; expert) / log N",
        "description": (
            "Normalized mutual information between input and expert choice. 0 = the "
            "router is a constant; 1 = each sequence picks a different expert."
        ),
        "key_pattern": r"^layer_\d+_routing_input_dependence$",
        "order": 235,
    },
    {
        "key": "smear_routing_merge_entropy",
        "type": "multi_expert_line",
        "title": "SMEAR/VEAR Merge Entropy (post-transform)",
        "y_label": "Entropy",
        "description": (
            "Entropy of the weights the parameter merge actually used. The gap from "
            "the balance card is what VEAR's p**4 sharpening does."
        ),
        "key_pattern": r"^layer_\d+_routing_merge_entropy$",
        "order": 245,
    },
    {
        "key": "smear_routing_concentration",
        "type": "multi_expert_line",
        "title": "SMEAR Routing Concentration (Collapse)",
        "y_label": "Max weight",
        "description": (
            "Per-layer maximum routing weight. 1.0 = collapsed onto one expert; "
            "1/num_experts = uniform merge."
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
            "Variance of the batch-mean routing weights, normalized (0 = balanced "
            "load, 1 = collapsed). Load balance, not per-sequence specialization."
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
            "Per-sequence routing commitment before the batch mean, rescaled to [0,1]: "
            "0 = uniform, 1 = every sequence commits to one expert."
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
            "Mean per-sequence top routing weight (1/num_experts = uniform, 1.0 = "
            "fully committed)."
        ),
        "key_pattern": r"^layer_\d+_routing_peak$",
        "order": 240,
    },
    # --- Modular SMEAR (praxis/routers/smear.py) ----------------------------
    # Three cards, not nine. SMEAR emitted nine chart families each keyed by
    # `layer_{depth}_`, which at depth 6 drew 54 lines - all of them the SAME
    # router sampled at different recurrent passes, since one router serves
    # every depth. These keys carry no depth prefix (each pass overwrites the
    # last, so what surfaces is the most recent), and the per-pass story is told
    # by Router Depth Specialization, which is the only place a genuinely
    # different object exists. Auto-hide for the other routers, which emit none
    # of these keys.
    {
        "key": "smear_coefficients",
        "type": "expert_routing_heatmap",
        "title": "SMEAR Merge Coefficients (per target module)",
        "y_label": "Merge weight",
        "description": (
            "One row per routed target module, one column per expert deviation: the "
            "weights the merge used. A row at 1/N declined to specialize; 1.0 "
            "committed."
        ),
        # Two capture groups, in the order the heatmap renderer reads them:
        # group 1 the ROW (a target module label), group 2 the COLUMN index.
        # A pattern without both cannot form a grid and draws nothing.
        "key_pattern": r"^smear_coeff_(.+)_(\d+)$",
        "row_label": "Target module",
        "col_label": "Deviation",
        "stepped": True,
        "order": 260,
    },
    {
        "key": "smear_target_dispersion",
        "type": "multi_expert_line",
        "title": "SMEAR Target Dispersion (did per-module granularity earn it?)",
        "y_label": "Mean pairwise row distance",
        "description": (
            "Mean pairwise L1 distance between modules' coefficient rows, normalized. "
            "0 = every module chose the same mixture, so per-module routing bought "
            "nothing."
        ),
        "key_pattern": r"^smear_target_dispersion$",
        "order": 261,
    },
    {
        "key": "smear_input_dependence",
        "type": "multi_expert_line",
        "title": "SMEAR Routing Input Dependence",
        "y_label": "I(input; expert) / log N",
        "description": (
            "Normalized mutual information between input and expert choice, averaged "
            "over targets (per-target max alongside). 0 = the router is a constant."
        ),
        "key_pattern": r"^smear_input_dependence(_max)?$",
        "order": 262,
    },
    {
        "key": "smear_expert_utilization",
        "type": "multi_expert_line",
        "title": "SMEAR Deviation Utilization",
        "y_label": "Fraction in use",
        "description": (
            "Fraction of deviations carrying over half their fair share of a target's "
            "coefficient. 1.0 = all used; 1/num_experts = one picked, the rest "
            "abandoned."
        ),
        "key_pattern": r"^smear_expert_utilization$",
        "order": 263,
    },
    {
        "key": "smear_selection",
        "type": "multi_expert_line",
        "title": "SMEAR Selection: Sharpness vs Diversity",
        "y_label": "Normalized",
        "description": (
            "Sharpness = how peaked one routing decision is. Diversity = whether "
            "decisions spread over different deviations. Sharp but not diverse means "
            "dead experts."
        ),
        "key_pattern": r"^smear_selection_(sharpness|diversity)$",
        "order": 264,
    },
    {
        "key": "smear_delta_scale",
        "type": "multi_expert_line",
        "title": "SMEAR Deviation Magnitude",
        "y_label": "||merged delta|| / ||base||",
        "description": (
            "How far the merged deviations move each target's weights, as a fraction "
            "of its norm. Rich coefficients over tiny deviations move nothing. 0 at "
            "init."
        ),
        "key_pattern": r"^smear_delta_scale_.+$",
        "order": 265,
    },
    {
        "key": "depth_step",
        "type": "multi_expert_line",
        "title": "Depth Trajectory: Step Size (spectral-attractor probe)",
        "y_label": "Relative step",
        "description": (
            "Relative move of the hidden state's fingerprint per depth transition, one "
            "line each. A settling iteration shrinks toward zero."
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
            "Last depth step over the first. < 1 = settling toward a fixed point; ~1 = "
            "no convergence; > 1 = diverging."
        ),
        "order": 251,
    },
    {
        "key": "depth/jump_concentration",
        "type": "line",
        "title": "Depth Jump Concentration (discrete vs smooth)",
        "y_label": "max step / mean step",
        "description": (
            "How concentrated depth movement is: high = one large hop then settle; ~1 "
            "= smooth drift."
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
            "Sampling probability over the sequence-length multipliers "
            "(constant attention cost). Under seq_curriculum=probe this is the "
            "posterior probability that each length is the best teacher, fit by "
            "attributing a held-out probe's improvement to the arm mixture - "
            "read the companion t-statistic card to tell a real preference from "
            "a leader the evidence does not support. Absent under "
            "seq_curriculum=fixed."
        ),
        "key_pattern": r"^seq_prob_x\d+$",
        "legend": True,
        "order": 55,
        "caller": "SequenceProbe",
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


# ── X axes the Research tab can plot every time-series card against ──────────
#
# Optimizer steps stopped being a comparable budget the day the batch governor
# landed: it varies the effective batch, so tokens-per-step swings by two orders
# of magnitude WITHIN a single run, never mind between two of them. Which axis
# is "right" depends on the question, and the three questions are different:
#
#   step      - how many optimizer updates did it take? The learning-dynamics
#               question. Still the honest axis for anything schedule-driven.
#   tokens    - how much data did it take? The sample-efficiency question.
#   wallclock - how long did it take? The one the governor is actually trying
#               to win. A governed run is EXPECTED to look worse per-token than
#               a small-batch run while beating it per-second, so reading only
#               the token axis will condemn the governor exactly where it works.
#
# ``source`` names a key on the per-run ``metrics`` payload from /api/metrics;
# every one of them is index-aligned with the others because the whole read
# path (SQL sampling, LTTB, _transform_metrics) moves whole ROWS, never columns.
# Adding a fourth axis (samples, FLOPs) is one entry here plus one emit.
X_AXIS_REGISTRY: list = [
    {
        "key": "step",
        "label": "Step",
        "axis_title": "Training Step",
        "source": "steps",
        # Whole numbers, so charts may pin ticks to integers. Token counts (in
        # billions) and elapsed hours are fractional and must not be rounded -
        # a 113M-token run reads 0.113 and would tick as a row of zeroes.
        "integral": True,
        "order": 10,
        "description": "Optimizer updates. Not proportional to data under a batch governor.",
    },
    {
        "key": "tokens",
        "label": "Tokens",
        "axis_title": "Tokens (Billions)",
        "source": "num_tokens",
        "order": 20,
        "description": (
            "Cumulative training tokens. Exact on validation rows, not carried "
            "forward."
        ),
    },
    {
        "key": "wallclock",
        "label": "Wall-clock",
        "axis_title": "Elapsed (hours)",
        "source": "elapsed_s",
        "unit_scale": 3600.0,
        "order": 30,
        "description": (
            "Time actually spent training, rebuilt from row timestamps with pauses "
            "discounted."
        ),
    },
]


def x_axis_names() -> list:
    """Keys of the registered Research-tab x axes."""
    return [axis["key"] for axis in X_AXIS_REGISTRY]


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
