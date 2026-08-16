"""LionGeo: one Lion momentum, N norm geometries, SMEAR-blended per matrix.

Lion and Muon are endpoints of one family: sign() is steepest descent under
the elementwise (vector-infinity) norm, and Newton-Schulz orthogonalization is
steepest descent under the spectral norm (the Lion-K / Schatten-p view). Read
through the Schatten duality - for ``G = U S V^T`` the dual map under Schatten-p
is ``U S^(q-1) V^T``, with q the conjugate exponent - that family has more than
two members, and every member is a normalization of the SAME momentum:

    c      = lerp(m, g, 1 - beta1)          Lion's lookahead momentum
    u_sign = sign(c)                        elementwise infinity norm, RMS 1
    u_spec = NS(c) * sqrt(max(rows, cols))  spectral norm, q=1 -> S^0, RMS ~ 1
    u_frob = c * sqrt(numel) / ||c||        Frobenius, q=2 -> S^1, RMS exactly 1
    u      = sum_i w_i u_i,  w = softmax(geo_logits)

The Frobenius arm is the one that does NOT whiten: sign discards magnitude
coordinatewise and Newton-Schulz discards the singular-value profile outright,
so with only those two, no setting of the mixture can simply follow the
momentum's own conditioning. It costs no extra compute and no extra state.
Every arm is RMS-matched, so a single Lion-scale lr drives the mixture and the
convex combination is always a bounded step.

The mixture logits adapt by HYPERGRADIENT descent (Baydin et al., 2018). The
realized loss sensitivity to logit i is <g_t, dp_t/dlogit_i>, and the softmax
Jacobian gives dp_t/dlogit_i proportional to w_i (u_i - u_bar) evaluated at the
previous step, where u_bar is the mixture that was actually applied. We keep
those deviations, take each one's cosine against the incoming gradient
(norm-free, so the rate stays a fixed model-agnostic constant), damp by
4 w_i (1 - w_i), then CENTRE the logits - softmax is shift-invariant, so
without centring the whole vector can drift into the clamp and pin the mixture
for a reason that has nothing to do with geometry - and clamp to +/- LOGIT_CLAMP.

The clamp is the mixture floor, and it is what keeps a badly-chosen arm
recoverable: it bounds the logits, so the softmax Jacobian never reaches zero
and a suppressed arm can always climb back. It is the same floored-mixture rule
as the memory bandit, the residual SMEAR and the mode-loss floor.

Its width follows from CENTRING, not from the arm count. Centred logits sum to
zero, so the widest reachable configuration puts one logit at +LOGIT_CLAMP and
spreads -LOGIT_CLAMP/(n-1) across the rest. At LOGIT_CLAMP = 2 that gives:

    2 arms -> [0.018, 0.982]      3 arms -> [0.024, 0.909]
    4 arms -> [0.023, 0.828]

The ceiling falls as arms are added (one arm can dominate less) while the floor
sits near 0.02 throughout. So adding arms does NOT widen the band - three arms
are slightly tighter than two.

What widened it was the move to a centred softmax. The previous single-logit
form, w = sigmoid(logit) with the same +/-2 clamp, reached only [0.119, 0.881],
because one logit carried the whole relative preference; centring splits the
clamp across both, so the RELATIVE logit now spans +/-4. That old bound was
genuinely binding - on the -f run, opt_geo_share_spread peaked at 0.7616 against
a band width of 0.881 - 0.119 = 0.7616 exactly, i.e. individual matrices sat
pinned at both ends of it. The relief came from the reparameterization; the
third geometry was along for the ride.

State per matrix: exp_avg (the shared momentum; named so the optimizer dynamics
suite reads it), geo_diffs (previous u_i - u_bar, stacked, half precision),
geo_logits. Against fp32 params that is 2.5x the parameter bytes for three arms,
where the two-arm version was 2x - Adam's footprint. The deviations are what buy
real credit assignment (which past choice improved the next gradient) rather than
a greedy "which normalization matches the current gradient", which would need no
state at all. Compute is one Newton-Schulz per matrix per step, the same as Muon,
and the extra arm measured at about +5% step time. No syncs in the step path; the
share accessors sync only when the metrics interval reads them.

Intended for interior >=2D matrices only (the MuonGeo split): embeddings, the
head, norms and biases route to a plain Lion secondary via CompositeOptimizer.
"""

import math

import torch
from torch.optim import Optimizer

from pytorch_optimizer.optimizer.muon import zero_power_via_newton_schulz_5

# Candidate geometries, in state order. Each arm costs one stored tensor per
# matrix, so this tuple is the expressivity/memory dial.
GEOMETRIES = ("sign", "spectral", "frobenius")

# Hypergradient nudge per step, applied to a cosine in [-1, 1] damped by the
# softmax Jacobian: on the order of a hundred consistently-aligned steps
# traverse the clamp range.
ADAPT_RATE = 0.05
# Logit clamp = the mixture floor; see the module docstring for the reachable
# share band, which depends on len(GEOMETRIES).
LOGIT_CLAMP = 2.0

# The stored deviations only ever feed a cosine - a direction test damped by
# ADAPT_RATE - so they are kept at half precision. Norms are still reduced in
# fp32. This is what keeps a third arm from doubling the optimizer's footprint.
DIFF_DTYPE = torch.bfloat16

EPS = 1e-12


def _direction_dtype(dtype: torch.dtype) -> torch.dtype:
    """Reduced width for optimizer state that only carries a DIRECTION.

    bf16 is the floor for the usual cases, but it cannot be the floor for
    EVERY case: a float64 run exists to ask whether anything it measures is
    precision-limited, and an 8-bit mantissa sitting inside its optimizer
    would quietly answer that question for it. So fp64 halves to fp32 - still
    off the memory bill, still far wider than the cosine needs. Parameters
    already at 16 bits keep their own dtype, as before.
    """
    if torch.finfo(dtype).bits <= 16:
        return dtype
    return torch.float32 if dtype is torch.float64 else DIFF_DTYPE


class LionGeo(Optimizer):
    """SMEAR blend of sign, Newton-Schulz and Frobenius updates over a shared
    Lion momentum, with hypergradient-adapted mixture weights."""

    def __init__(self, params, lr=3e-4, betas=(0.95, 0.98), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "LionGeo"

    @staticmethod
    def _arm_directions(c: torch.Tensor) -> torch.Tensor:
        """The candidate updates, stacked ``[len(GEOMETRIES), *c.shape]`` in
        GEOMETRIES order. All are RMS-matched to ~1. Built by name, so
        GEOMETRIES really is the dial: drop or reorder entries and the state,
        the mixture and the metrics all follow."""
        flat = c.reshape(c.size(0), -1)
        # The library orthogonalizes in bf16 by default, which is the standard
        # Muon recipe and stays that way for fp32/bf16 parameters. Same
        # argument as _direction_dtype for fp64: an 8-bit mantissa inside the
        # spectral arm would be a precision floor the run cannot see.
        ns_dtype = _direction_dtype(c.dtype)
        arms = {
            "sign": lambda: torch.sign(c),
            "spectral": lambda: zero_power_via_newton_schulz_5(flat, dtype=ns_dtype)
            .to(c.dtype)
            .reshape_as(c)
            .mul_(math.sqrt(max(flat.shape))),
            "frobenius": lambda: c * (math.sqrt(c.numel()) / c.norm().clamp_min(EPS)),
        }
        return torch.stack([arms[name]() for name in GEOMETRIES])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        n_arms = len(GEOMETRIES)
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            wd = float(group.get("weight_decay", 0.0) or 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                diff_dtype = _direction_dtype(p.dtype)
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["geo_logits"] = torch.zeros(n_arms, device=p.device)
                    # Zero deviations make the first step's cosines exactly 0,
                    # so the mixture starts uniform with no special-casing.
                    state["geo_diffs"] = torch.zeros(
                        (n_arms, *p.shape), dtype=diff_dtype, device=p.device
                    )
                m = state["exp_avg"]
                # Optimizer.load_state_dict casts every float state tensor to the
                # param's dtype, so a resume can silently re-type both buffers.
                # Restore them here rather than paying fp32 diffs (or a bf16
                # logit accumulator) for the rest of the run.
                logits = state["geo_logits"]
                if logits.dtype != torch.float32:
                    logits = state["geo_logits"] = logits.float()
                d_prev = state["geo_diffs"]
                if d_prev.dtype != diff_dtype:
                    d_prev = state["geo_diffs"] = d_prev.to(diff_dtype)

                # Hypergradient on the mixture: if an arm's previous deviation
                # from the applied update still correlates with the new
                # gradient, that arm was the better descent direction there -
                # raise its share (and vice versa).
                w_now = torch.softmax(logits, dim=0)
                gf = g.reshape(1, -1)
                df = d_prev.reshape(n_arms, -1)
                # Reduce the norms at fp32 or wider, never narrower than the
                # gradient itself: a hardcoded float32 cannot take an fp64
                # gradient without narrowing, and vector_norm refuses rather
                # than silently rounding.
                norm_dtype = torch.promote_types(gf.dtype, torch.float32)
                denom = (
                    torch.linalg.vector_norm(df, dim=1, dtype=norm_dtype)
                    * torch.linalg.vector_norm(gf, dtype=norm_dtype)
                ).clamp_min(EPS)
                # Elementwise rather than a matvec: this promotes the half
                # precision deviations to fp32, where a bf16 matmul would
                # accumulate in bf16 on CPU and in fp32 on CUDA - a ~2e-3
                # device-dependent split in the cosine, for no measured speedup.
                # Back to the logits' dtype: the accumulator stays fp32 whatever
                # the parameters are, so an fp64 cosine must not try to land in
                # it in place.
                cos = ((df * gf).sum(dim=1) / denom).to(logits.dtype)
                jac = 4.0 * w_now * (1.0 - w_now)
                logits.add_(ADAPT_RATE * jac * cos)
                # Centre before clamping: the clamp then bounds RELATIVE
                # preference, not the softmax's free common shift.
                logits.sub_(logits.mean()).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)

                c = m.lerp(g, 1.0 - beta1)
                u = self._arm_directions(c)
                w = torch.softmax(logits, dim=0).to(u.dtype)
                update = (u * w.view(-1, *([1] * c.dim()))).sum(dim=0)

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                # u is dead after the step; copy into the standing buffer rather
                # than reallocating the deviations every step.
                d_prev.copy_(u.sub_(update))
                m.lerp_(g, 1.0 - beta2)
        return loss

    @torch.no_grad()
    def get_geometry_shares(self) -> dict:
        """Per-matrix mixture weights as ``{geometry: [share per matrix]}``.
        Syncs to host; call from the metrics interval, never the step path."""
        out = {name: [] for name in GEOMETRIES}
        for group in self.param_groups:
            for p in group["params"]:
                logits = self.state.get(p, {}).get("geo_logits")
                if logits is None:
                    continue
                for name, share in zip(GEOMETRIES, torch.softmax(logits, dim=0)):
                    out[name].append(float(share))
        return out

    @torch.no_grad()
    def get_smear_shares(self):
        """Per-matrix spectral shares: the historical one-number view, kept so
        the opt_geo_share card stays continuous across the arm-count change."""
        return self.get_geometry_shares()["spectral"]
