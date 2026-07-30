"""Sequence-length curriculum by probe attribution.

The learning-progress bandit in :mod:`praxis.data.seq_curriculum` scores an arm
by the loss drop between two visits to it. That quantity measures how much the
WHOLE model improved in the interval, not what the arm contributed, so it
carries no information about arm quality: an arm with zero teaching value still
earns a full share, and because sampling an arm more often shortens the gap
between its visits (shrinking the measured drop), the mechanism is a negative
feedback loop on visit rate that drives the distribution to uniform. What the
"Sequence Length Mix" card shows under that mode is the jitter around uniform.

This module measures the thing the curriculum is supposed to optimize instead.
Every ``window`` optimizer steps a FIXED held-out probe is re-scored, and the
probe's improvement is regressed onto the arm mixture that produced it:

    delta_loss over the window  =  sum_m  beta_m * visits_m

``beta_m`` is then the measured value of one visit to arm ``m``, denominated in
held-out loss. It is causal (the regression only credits arms that were actually
sampled), it targets the objective rather than an arm's self-improvement, and it
costs one probe forward per arm per window instead of an extra forward per step.

Two details make it work rather than merely look principled:

* **Recursive least squares with forgetting.** The best length changes over a
  run, so the normal equations decay by ``forget`` each window - the fit tracks
  instead of averaging over all of history.
* **Posterior-of-best scoring, not a softmax.** The old controller divided by
  the spread BETWEEN arms, which collapses toward zero when no arm is better, so
  pure noise rendered as a confident distribution. A softmax over t-statistics
  is no better - it needs a temperature relating coefficient units to
  confidence, and at temperature 1 a t of 1.2 (no significant edge) still took
  68% of the mixture. Sampling each arm's coefficient from its own fitted
  posterior and counting argmax wins needs no such constant: it is near-uniform
  while the standard errors swamp the differences and sharpens on its own as the
  evidence accumulates, so the card reads as "nothing to exploit" when that is
  the truth.

Self-prediction gain (Graves et al.'s prediction gain: score the same batch
before and after the update) is the other causal option and is cheaper to
reason about, but it optimizes an arm's improvement on its OWN distribution, so
it chases whichever length is furthest from converged. Measured against a
transfer-structured objective it lands worse than uniform sampling; probe
attribution does not, which is why this is the mechanism here.

Cross-process note: like the sampler weights and the batch schedule, the
distribution is class-level state. The callback writes it, the data pipeline
reads it. Praxis forces the ``spawn`` start method, so the dataset iterates in
the same process and sees writes immediately.
"""

import math
import random
from typing import Dict, List, Optional


class SequenceProbe:
    """Shared distribution over sequence-length multipliers, fit from a probe."""

    # Fixed, model-agnostic constants (no per-experiment tuning).
    forget = 0.98  # per-window decay of the normal equations (~50-window memory)
    ridge = 1e-6  # keeps the fit defined before the mixture has varied
    explore = 0.1  # uniform floor: every arm keeps being sampled and re-fit
    # Windows before the fit is allowed to steer SAMPLING. Deliberately small:
    # the posterior scoring is self-calibrating (thin evidence renders as a
    # diffuse mix), so a long warm-up gate buys nothing and costs visibility -
    # at 8 the first reported mix landed 1152 optimizer steps into a run, which
    # reads as a missing dashboard card rather than a warming one. Reporting
    # starts a window earlier still; see ``metrics``.
    min_windows = 3
    posterior_samples = 512  # Thompson draws per refit; pure python, ~0.1ms

    _rng = random.Random(0)  # deterministic: the mix must survive a resume

    enabled: bool = False
    block_size: int = 0
    arms: List[int] = []
    shared_probs: Optional[Dict[int, float]] = None  # consumed by sample()

    _xtx: List[List[float]] = []
    _xty: List[float] = []
    _beta: List[float] = []
    _se: List[float] = []
    _tstat: List[float] = []
    _resid_var: float = 0.0
    _windows: int = 0

    @classmethod
    def enable(cls, block_size: int, tiers) -> None:
        """Arm the controller. ``arms`` = {1} plus every tier multiplier.

        Idempotent: re-arming with the same shape keeps the accumulated fit.
        Lightning restores callback state BEFORE ``on_train_start``, so a reset
        here would discard a resumed regression.
        """
        arms = sorted({1} | {int(m) for m, _ in tiers})
        if cls.enabled and cls.arms == arms and cls.block_size == int(block_size):
            return
        cls.enabled = True
        cls.block_size = int(block_size)
        cls.arms = arms
        n = len(cls.arms)
        cls._xtx = [[0.0] * n for _ in range(n)]
        cls._xty = [0.0] * n
        cls._beta = [0.0] * n
        cls._se = [float("inf")] * n
        cls._tstat = [0.0] * n
        cls._resid_var = 0.0
        cls._windows = 0
        cls.shared_probs = None  # stays None until the fit is trusted
        print(
            f"[SeqProbe] probe-attribution sequence curriculum on; "
            f"arms={cls.arms} (block_size={cls.block_size})"
        )

    @classmethod
    def reset(cls) -> None:
        cls.enabled = False
        cls.block_size = 0
        cls.arms = []
        cls.shared_probs = None
        cls._xtx = []
        cls._xty = []
        cls._beta = []
        cls._se = []
        cls._tstat = []
        cls._resid_var = 0.0
        cls._windows = 0

    @classmethod
    def observe_window(cls, visits: Dict[int, int], delta: float) -> None:
        """Fold in one window: its arm counts and the probe's improvement.

        ``delta`` is ``previous_probe_loss - current_probe_loss``, so positive
        means the model got better over the window.
        """
        if not cls.enabled or not cls.arms:
            return
        x = [float(visits.get(m, 0)) for m in cls.arms]
        if sum(x) <= 0:
            return

        n = len(cls.arms)
        f = cls.forget
        for i in range(n):
            cls._xty[i] = f * cls._xty[i] + x[i] * delta
            for j in range(n):
                cls._xtx[i][j] = f * cls._xtx[i][j] + x[i] * x[j]
        cls._windows += 1

        # Residual variance, needed for the standard errors. Tracked with the
        # same forgetting factor against the fit as it stood BEFORE this
        # window, so it measures prediction error rather than fit error.
        predicted = sum(b * xi for b, xi in zip(cls._beta, x))
        err = delta - predicted
        cls._resid_var = f * cls._resid_var + (1 - f) * err * err

        cls._solve()
        if cls._windows >= cls.min_windows:
            first = cls.shared_probs is None
            cls._recompute()
            if first and cls.shared_probs is not None:
                mix = "  ".join(
                    f"x{m}={cls.shared_probs[m]:.2f}" for m in cls.arms
                )
                print(
                    f"[SeqProbe] fit engaged after {cls._windows} windows; "
                    f"sampling mix {mix}"
                )

    @classmethod
    def _solve(cls) -> None:
        """Ridge least squares for beta, plus t-statistics from (XtX)^-1.

        The visit counts sum to roughly the same total every window (the window
        length times the accumulation factor), so the all-ones direction is at
        best weakly identified: what the fit determines is the DIFFERENCES
        between coefficients, which is all the scoring uses. The ridge term
        keeps the inverse defined in that direction instead of pretending it is
        informed, and a batch governor moving the accumulation factor actually
        helps here - it varies the total and breaks the collinearity.
        """
        n = len(cls.arms)
        # Solve and invert in one Gauss-Jordan pass on [XtX + ridge | I | Xty].
        aug = []
        for i in range(n):
            row = list(cls._xtx[i])
            row[i] += cls.ridge
            row += [1.0 if j == i else 0.0 for j in range(n)]
            row.append(cls._xty[i])
            aug.append(row)

        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-18:
                return  # singular: leave the previous fit in place
            aug[col], aug[pivot] = aug[pivot], aug[col]
            d = aug[col][col]
            aug[col] = [v / d for v in aug[col]]
            for r in range(n):
                if r != col and aug[r][col]:
                    factor = aug[r][col]
                    aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]

        cls._beta = [aug[i][2 * n] for i in range(n)]
        # se(beta_i) = sqrt(resid_var * (XtX)^-1_ii). Kept explicitly rather
        # than recovered from the t-statistic later: near beta = 0 the ratio
        # beta/t is numerically meaningless, and a posterior built from it
        # collapses to a false certainty about a coefficient that is really
        # just noise.
        cls._se = []
        cls._tstat = []
        for i in range(n):
            var = cls._resid_var * max(aug[i][n + i], 0.0)
            se = math.sqrt(var) if var > 0 else float("inf")
            cls._se.append(se)
            cls._tstat.append(
                cls._beta[i] / se if math.isfinite(se) and se > 1e-18 else 0.0
            )

    @classmethod
    def _recompute(cls) -> None:
        """Posterior probability that each arm is the best one (Thompson).

        Draw from the fitted normal posterior on the coefficients and count how
        often each arm comes out on top. This is the scoring that makes the
        controller honest, and it is why a softmax was rejected: a softmax needs
        a temperature relating "coefficient units" to "confidence", which no
        fixed constant can supply. Softmax over t-statistics at temperature 1
        handed 68% of the mixture to a t of 1.2 - an arm with no significant
        edge at all. The posterior needs no such constant: when the standard
        errors swamp the differences the argmax is near-uniform by construction,
        and it sharpens on its own exactly as the evidence accumulates.

        Note this scoring is already cost-normalized. A coefficient is the value
        of one VISIT - one optimizer step - so a length that buys fewer tokens
        per step earns a proportionally smaller coefficient without anything
        extra being divided out.
        """
        n = len(cls.arms)
        if n == 0 or len(cls._beta) != n or len(cls._se) != n:
            return
        ses = list(cls._se)
        finite = [s for s in ses if math.isfinite(s) and s > 0]
        if not finite:
            cls.shared_probs = {m: 1.0 / n for m in cls.arms}
            return
        # An arm with no usable error estimate is treated as maximally
        # uncertain, not as maximally good.
        fallback = max(finite)
        ses = [s if math.isfinite(s) else fallback for s in ses]

        wins = [0] * n
        for _ in range(cls.posterior_samples):
            best, best_val = 0, -float("inf")
            for i in range(n):
                draw = cls._rng.gauss(cls._beta[i], ses[i])
                if draw > best_val:
                    best, best_val = i, draw
            wins[best] += 1
        probs = [w / cls.posterior_samples for w in wins]
        u = 1.0 / n
        probs = [(1 - cls.explore) * p + cls.explore * u for p in probs]
        cls.shared_probs = dict(zip(cls.arms, probs))

    @classmethod
    def sample(cls, batch_size: int, tiers, rng) -> Optional[int]:
        """Pick a multiplier from the fitted distribution, restricted to tiers
        eligible for this row budget. Returns None until the fit is trusted so
        the caller falls back to the fixed-chance roll."""
        if not cls.enabled or cls.shared_probs is None:
            return None
        eligible = [m for m in cls.arms if batch_size >= m * m]
        if not eligible:
            return 1
        weights = [cls.shared_probs.get(m, 0.0) for m in eligible]
        total = sum(weights)
        if total <= 0:
            return 1
        r = rng.random() * total
        acc = 0.0
        for m, w in zip(eligible, weights):
            acc += w
            if r <= acc:
                return m
        return eligible[-1]

    @classmethod
    def metrics(cls) -> Dict[str, float]:
        """Per-arm fitted value, t-statistic, and sampling probability.

        Value and evidence are reported from the FIRST completed window, before
        the fit is trusted enough to steer sampling. That split is deliberate:
        gating the telemetry on the same threshold as the actuation made the
        dashboard cards appear only after ~1000 optimizer steps, which is
        indistinguishable from a card that does not exist. The probability keys
        stay absent until the mix is genuinely in force, so the mix card never
        shows a distribution nothing is sampling from.
        """
        if not cls.enabled or cls._windows < 1:
            return {}
        out: Dict[str, float] = {}
        for i, m in enumerate(cls.arms):
            out[f"seq_value_x{m}"] = float(cls._beta[i]) if i < len(cls._beta) else 0.0
            out[f"seq_tstat_x{m}"] = (
                float(cls._tstat[i]) if i < len(cls._tstat) else 0.0
            )
            if cls.shared_probs is not None:
                out[f"seq_prob_x{m}"] = float(cls.shared_probs.get(m, 0.0))
        return out


# ── Dashboard cards (folded into get_metric_descriptions when the probe
#    controller is armed - see praxis/metrics/descriptions.py) ─────────────

_SEQ_GROUP = "curriculum"

# Declared for multipliers up to 8 (the widest tier ladder); absent arms simply
# never emit their key.
SEQ_PROBE_METRIC_DESCRIPTIONS: Dict[str, dict] = {}
for _m, _order in ((1, 0), (2, 1), (4, 2), (8, 3)):
    SEQ_PROBE_METRIC_DESCRIPTIONS[f"seq_value_x{_m}"] = {
        "description": (
            f"Measured value of one x{_m} batch, in held-out probe loss: the "
            "regression coefficient from attributing the probe's improvement to "
            "the arm mixture that produced it. Positive = batches at this "
            "length reduce held-out loss. Only DIFFERENCES between arms are "
            "identifiable (the visit counts sum to the window), so read the "
            "spread, not the level."
        ),
        "chart": {
            "title": "Sequence Curriculum: Measured Arm Value",
            "y_label": "probe loss per batch",
            "y_scale": "linear",
            "group": _SEQ_GROUP,
            "group_order": 60,
            "order": _order,
            "series_group": "seq_value",
            "series_label": f"x{_m}",
        },
    }
    SEQ_PROBE_METRIC_DESCRIPTIONS[f"seq_tstat_x{_m}"] = {
        "description": (
            f"t-statistic on x{_m}'s value: coefficient over its own standard "
            "error. This is what the sampling distribution is a softmax of, so "
            "it is the honest read on whether there is anything to exploit - all "
            "arms near zero means the run has no measurable length preference "
            "and the mix should sit near uniform. Magnitudes above ~2 are the "
            "usual threshold for taking a difference seriously."
        ),
        "chart": {
            "title": "Sequence Curriculum: Evidence (t-statistic)",
            "y_label": "t",
            "y_scale": "linear",
            "group": _SEQ_GROUP,
            "group_order": 61,
            "order": _order,
            "series_group": "seq_tstat",
            "series_label": f"x{_m}",
        },
    }
