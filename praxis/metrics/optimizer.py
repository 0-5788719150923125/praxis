"""Optimizer telemetry for the dynamics dashboard.

We instrument gradients and component internals heavily but the optimizer
itself has been a black box. This module computes a default suite of
optimizer-state diagnostics (aggregated over all parameters) and declares
their dashboard cards. The :class:`DynamicsLogger` calls
:func:`extract_optimizer_dynamics` in ``on_before_optimizer_step`` (gradients
ready, state from the previous step) and logs the result on the dynamics route.

Metrics emit only when the optimizer carries the state they need, so the suite
adapts to the optimizer (the praxis default is Lion - sign momentum, no second
moment):

* always: ``opt_lr``, ``opt_grad_rms``.
* needs momentum (``exp_avg`` / ``momentum_buffer``): ``opt_momentum_rms``,
  ``opt_momentum_grad_cos`` - Lion, AdamW, SGD-momentum, ...
* needs a second moment (``exp_avg_sq``): ``opt_second_moment_rms``,
  ``opt_update_rms``, ``opt_update_weight_ratio`` - Adam-family only.
* schedule-free wrapper: ``opt_sf_spread`` (the bias-variance gap in weight
  space, reconstructing the average from ``y = m*x + (1-m)*z``) and
  ``opt_gate_mean`` (the gated/wave averaging gate).
"""

import torch
from pytorch_optimizer.optimizer import ScheduleFreeWrapper

EPS = 1e-8

# Cards for the dynamics dashboard (group "optimizer"). Registry-driven: adding
# an entry here + emitting the value is all the frontend needs.
OPTIMIZER_METRIC_DESCRIPTIONS = {
    "opt_lr": {
        "description": "Current scheduled learning rate (group 0).",
        "chart": {
            "title": "Learning Rate",
            "y_label": "lr",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "group_order": 10,
            "order": 10,
        },
    },
    "opt_grad_rms": {
        "description": (
            "RMS of the raw gradient over all parameters - the global magnitude the "
            "optimizer is working from."
        ),
        "chart": {
            "title": "Gradient RMS",
            "y_label": "||grad|| RMS",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 20,
        },
    },
    "opt_grad_norm": {
        "description": (
            "Max pre-clip global gradient L2 norm over the logging interval, measured "
            "every step. This is what gradient_clip_val compares against."
        ),
        "chart": {
            "title": "Gradient Norm (pre-clip, interval max)",
            "y_label": "||grad|| max",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 21,
        },
    },
    "opt_clip_rate": {
        "description": (
            "Fraction of steps whose pre-clip gradient norm exceeded "
            "gradient_clip_val. ~0 = the threshold is slack; persistently high = it "
            "may be too low."
        ),
        "chart": {
            "title": "Clip Rate",
            "y_label": "fraction clipped",
            "y_scale": "linear",
            "group": "optimizer",
            "order": 22,
        },
    },
    "opt_clip_intensity": {
        "description": (
            "Mean clip severity: the fraction of gradient magnitude clipping shaved "
            "off, 0 on steps that did not clip. 0.4 = 40% removed on average."
        ),
        "chart": {
            "title": "Clip Intensity",
            "y_label": "mean fraction removed",
            "y_scale": "linear",
            "group": "optimizer",
            "order": 23,
        },
    },
    "opt_momentum_rms": {
        "description": (
            "RMS of the optimizer's momentum buffer - the smoothed direction it is "
            "actually following."
        ),
        "chart": {
            "title": "Momentum RMS",
            "y_label": "||momentum|| RMS",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 30,
        },
    },
    "opt_momentum_grad_cos": {
        "description": (
            "Cosine between momentum and the current gradient. ~1 = consistent "
            "descent; near 0 or negative = turning or thrashing."
        ),
        "chart": {
            "title": "Momentum-Gradient Cosine",
            "y_label": "cosine",
            "y_scale": "linear",
            "group": "optimizer",
            "order": 40,
        },
    },
    "opt_update_rms": {
        "description": (
            "RMS of the implied Adam update lr*m/(sqrt(v)+eps). Needs a second moment; "
            "under a sign optimizer it is the reference step, not the one taken."
        ),
        "chart": {
            "title": "Update RMS",
            "y_label": "||update|| RMS",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 50,
        },
    },
    "opt_update_weight_ratio": {
        "description": (
            "Implied-update RMS / weight RMS, the relative step size. Healthy Adam "
            "training sits near 1e-3."
        ),
        "chart": {
            "title": "Update / Weight Ratio",
            "y_label": "ratio",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 55,
        },
    },
    "opt_second_moment_rms": {
        "description": (
            "sqrt(mean(v)), the running estimate of gradient scale. From Adam's "
            "exp_avg_sq or a factored estimate over any base optimizer."
        ),
        "chart": {
            "title": "Second-Moment RMS",
            "y_label": "sqrt(v) RMS",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 60,
        },
    },
    "opt_geo_share": {
        "description": (
            "Mean share of LionGeo's spectral arm - the orthogonalized, Muon-like "
            "geometry. The three arm shares sum to 1 at every matrix."
        ),
        "chart": {
            "title": "Geometry Mixture (share per norm)",
            "y_label": "share",
            "y_scale": "linear",
            "group": "optimizer",
            "order": 65,
            "series_group": "geo_mixture",
            "series_label": "spectral",
        },
    },
    "opt_geo_share_sign": {
        "description": (
            "Mean share of the sign arm: Lion's geometry, steepest descent under the "
            "elementwise-infinity norm."
        ),
        # No title/axis: rides opt_geo_share's chart via series_group.
        "chart": {
            "group": "optimizer",
            "order": 66,
            "series_group": "geo_mixture",
            "series_label": "sign",
        },
    },
    "opt_geo_share_frobenius": {
        "description": (
            "Mean share of the Frobenius arm: RMS-rescaled momentum, the only arm that "
            "does not whiten."
        ),
        # No title/axis: rides opt_geo_share's chart via series_group.
        "chart": {
            "group": "optimizer",
            "order": 67,
            "series_group": "geo_mixture",
            "series_label": "frobenius",
        },
    },
    "opt_geo_share_spread": {
        "description": (
            "Disagreement between matrices: max minus min arm share, averaged over "
            "arms. Near 0 = one global geometry would do; wide = they want different "
            "ones."
        ),
        "chart": {
            "title": "Geometry Share Spread",
            "y_label": "max - min",
            "y_scale": "linear",
            "group": "optimizer",
            "order": 68,
        },
    },
    "opt_sf_spread": {
        "description": (
            "Schedule-free spread ||z - x|| / ||x|| - how far the responsive iterate "
            "has diverged from the deployed average."
        ),
        "chart": {
            "title": "Schedule-Free Spread",
            "y_label": "||z-x||/||x||",
            "y_scale": "logarithmic",
            "group": "optimizer",
            "order": 70,
        },
    },
    "opt_gate_mean": {
        "description": (
            "Mean averaging gate of the gated/wave schedule-free variants - how much "
            "of the iterate is admitted into the average."
        ),
        "chart": {
            "title": "Averaging Gate (mean)",
            "y_label": "gate",
            "y_scale": "linear",
            "group": "optimizer",
            "order": 80,
        },
    },
}

# Momentum buffer is named differently across optimizers.
_MOMENTUM_KEYS = ("exp_avg", "momentum_buffer")


def _unwrap(optimizer):
    """Return (innermost base optimizer, gate-bearing wrapper or None).

    The wrapper is schedule-free (z/x averaging, drives the spread + gate
    metrics) or HalfLion (a frozen-anchor wave that only carries ``gate_mean``;
    the schedule-free spread path no-ops for it - no ``momentum``/``z``)."""
    base, sf, o = optimizer, None, optimizer
    for _ in range(8):
        if isinstance(o, ScheduleFreeWrapper) or hasattr(o, "set_wave"):
            sf = o
        base = o
        nxt = getattr(o, "optimizer", None)
        if nxt is None or nxt is o:
            break
        o = nxt
    return base, sf


def _second_moment_provider(optimizer):
    """A wrapper in the stack that reconstructs a factored second moment
    (``get_second_moment``), or None."""
    o, depth = optimizer, 0
    while o is not None and depth < 8:
        if hasattr(o, "get_second_moment"):
            return o
        o = getattr(o, "optimizer", None)
        depth += 1
    return None


def _geo_provider(optimizer):
    """An optimizer in the stack exposing per-matrix geometry shares
    (``get_smear_shares``), or None. LionGeo sits as a composite's primary, so
    each level checks the node and its ``primary`` before unwrapping."""
    o, depth = optimizer, 0
    while o is not None and depth < 8:
        for cand in (o, getattr(o, "primary", None)):
            if cand is not None and hasattr(cand, "get_smear_shares"):
                return cand
        o = getattr(o, "optimizer", None)
        depth += 1
    return None


def _momentum(state):
    for k in _MOMENTUM_KEYS:
        m = state.get(k)
        if m is not None:
            return m
    return None


@torch.no_grad()
def extract_optimizer_dynamics(optimizer) -> dict:
    """Aggregate optimizer-state diagnostics over all parameters. Each metric
    emits only when the needed state is present, so it adapts to the optimizer
    (Lion has momentum but no second moment; SGD may have neither)."""
    if optimizer is None:
        return {}
    base, sf = _unwrap(optimizer)
    moment = _second_moment_provider(optimizer)  # factored estimate, if tracked
    pgs = getattr(base, "param_groups", None) or getattr(optimizer, "param_groups", [])
    if not pgs:
        return {}

    out = {"opt_lr": float(pgs[0].get("lr", 0.0))}

    geo = _geo_provider(optimizer)
    if geo is not None:
        by_arm = (
            geo.get_geometry_shares()
            if hasattr(geo, "get_geometry_shares")
            else {"spectral": geo.get_smear_shares()}
        )
        by_arm = {name: s for name, s in by_arm.items() if s}
        for name, shares in by_arm.items():
            # The spectral arm keeps the bare key: it is the series the card
            # existed for before the mixture grew past two geometries.
            key = "opt_geo_share" if name == "spectral" else f"opt_geo_share_{name}"
            out[key] = sum(shares) / len(shares)
        if by_arm:
            # Between-matrix disagreement, averaged over arms. With two
            # (complementary) arms this is exactly the old max-min spectral spread.
            out["opt_geo_share_spread"] = sum(
                max(s) - min(s) for s in by_arm.values()
            ) / len(by_arm)

    m = float(getattr(sf, "momentum", 0.0)) if sf is not None else 0.0
    sf_ok = sf is not None and 0.0 < m < 1.0

    n = 0
    s_g2 = s_m12 = s_m1g = s_v = s_upd2 = s_w2 = 0.0
    s_zx = s_x = 0.0
    have_m = have_v = False

    for group in pgs:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            n += g.numel()
            s_g2 += float((g * g).sum())
            s_w2 += float((p.data * p.data).sum())

            st = base.state.get(p, {})
            m1 = _momentum(st)
            v = st.get("exp_avg_sq")
            if v is None and moment is not None:  # reconstruct a factored estimate
                v = moment.get_second_moment(p)
            if m1 is not None and m1.shape == g.shape:
                have_m = True
                s_m12 += float((m1 * m1).sum())
                s_m1g += float((m1 * g).sum())
                if v is not None and v.shape == g.shape:
                    have_v = True
                    s_v += float(v.sum())
                    upd = m1 / v.sqrt().add(EPS)
                    s_upd2 += float((upd * upd).sum())

            if sf_ok:
                z = sf.state.get(p, {}).get("z")
                if z is not None and z.shape == p.shape:
                    x = (p.data - (1.0 - m) * z) / m  # reconstruct the average
                    d = z - x
                    s_zx += float((d * d).sum())
                    s_x += float((x * x).sum())

    if n == 0:
        return out

    out["opt_grad_rms"] = (s_g2 / n) ** 0.5
    w_rms = (s_w2 / n) ** 0.5

    if have_m:
        out["opt_momentum_rms"] = (s_m12 / n) ** 0.5
        denom = (s_m12 * s_g2) ** 0.5
        out["opt_momentum_grad_cos"] = (s_m1g / denom) if denom > 0 else 0.0

    if have_v:
        out["opt_second_moment_rms"] = (s_v / n) ** 0.5
        upd_rms = out["opt_lr"] * (s_upd2 / n) ** 0.5
        out["opt_update_rms"] = upd_rms
        if w_rms > 0:
            out["opt_update_weight_ratio"] = upd_rms / w_rms

    if sf_ok and s_x > 0:
        out["opt_sf_spread"] = (s_zx / s_x) ** 0.5
    if sf is not None and getattr(sf, "gate_mean", None) is not None:
        out["opt_gate_mean"] = float(sf.gate_mean)

    return out
