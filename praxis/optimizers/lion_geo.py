"""LionGeo: one Lion momentum, two norm geometries, SMEAR-blended per matrix.

Lion and Muon are endpoints of one family: sign() is steepest descent under
the elementwise (vector-infinity) norm, and Newton-Schulz orthogonalization is
steepest descent under the spectral norm (the Lion-K / Schatten-p view, where
Schatten-p interpolates between them). Both normalizations are scale-invariant
in their input, so they can share ONE momentum buffer - Lion's cheap state -
and differ only in the geometry they impose on it:

    c      = lerp(m, g, 1 - beta1)          Lion's lookahead momentum
    u_sign = sign(c)                        RMS exactly 1
    u_spec = NS(c) * sqrt(max(rows, cols))  semi-orthogonal, RMS ~ 1
    u      = lerp(u_sign, u_spec, w),       w = sigmoid(geo_logit)

The branches are RMS-matched, so a single Lion-scale lr drives the blend and
the convex mix is always a bounded step.

The per-matrix mixture logit adapts by HYPERGRADIENT descent (Baydin et al.,
2018): the realized loss sensitivity to the logit is <g_t, dp_t/dlogit>, and
dp_t/dlogit is proportional to w(1-w) times the previous step's
(u_spec - u_sign) difference direction. We keep that difference, take its
cosine against the incoming gradient (norm-free, so the rate is a fixed
model-agnostic constant), damp by the sigmoid Jacobian 4w(1-w), and clamp the
logit to +/- LOGIT_CLAMP - a floor that keeps both geometries alive (shares
stay within ~[0.12, 0.88]) so either can recover, the same floored-mixture
rule as the memory bandit and the residual SMEAR.

State per matrix: exp_avg (the shared momentum; named so the optimizer
dynamics suite reads it), geo_diff (previous u_spec - u_sign), geo_logit
(scalar). Two tensors per param, the same footprint as Adam, plus one
Newton-Schulz per matrix per step, the same compute as Muon. No syncs in the
step path; ``get_smear_shares`` syncs only when the metrics interval reads it.

Intended for interior >=2D matrices only (the MuonGeo split): embeddings, the
head, norms and biases route to a plain Lion secondary via CompositeOptimizer.
"""

import math

import torch
from torch.optim import Optimizer

from pytorch_optimizer.optimizer.muon import zero_power_via_newton_schulz_5

# Hypergradient nudge per step, applied to a cosine in [-1, 1] damped by the
# sigmoid Jacobian: ~50 consistently-aligned steps traverse the clamp range.
ADAPT_RATE = 0.05
# Logit clamp = the mixture floor: sigmoid(+/-2) keeps shares in [0.119, 0.881].
LOGIT_CLAMP = 2.0


class LionGeo(Optimizer):
    """SMEAR blend of sign (infinity-norm) and Newton-Schulz (spectral-norm)
    updates over a shared Lion momentum, with hypergradient-adapted weights."""

    def __init__(self, params, lr=3e-4, betas=(0.95, 0.98), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "LionGeo"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            wd = float(group.get("weight_decay", 0.0) or 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["geo_logit"] = torch.zeros((), device=p.device)
                m = state["exp_avg"]
                logit = state["geo_logit"]

                # Hypergradient on the mixture: if the previous step's
                # spectral-minus-sign direction still correlates with the new
                # gradient, the spectral branch was the better descent
                # direction there - raise its share (and vice versa).
                d_prev = state.get("geo_diff")
                if d_prev is not None:
                    denom = (g.norm() * d_prev.norm()).clamp_min(1e-12)
                    cos = (g * d_prev).sum() / denom
                    w_now = torch.sigmoid(logit)
                    jac = 4.0 * w_now * (1.0 - w_now)
                    logit.add_(ADAPT_RATE * jac * cos).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)

                c = m.lerp(g, 1.0 - beta1)
                u_sign = torch.sign(c)
                flat = c.reshape(c.size(0), -1)
                u_spec = (
                    zero_power_via_newton_schulz_5(flat)
                    .to(c.dtype)
                    .reshape_as(c)
                    .mul_(math.sqrt(max(flat.shape)))
                )
                update = torch.lerp(u_sign, u_spec, torch.sigmoid(logit))

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                state["geo_diff"] = u_spec - u_sign
                m.lerp_(g, 1.0 - beta2)
        return loss

    @torch.no_grad()
    def get_smear_shares(self):
        """Per-matrix spectral shares (sigmoid of the logits). Syncs to host;
        call from the metrics interval, never the step path."""
        shares = []
        for group in self.param_groups:
            for p in group["params"]:
                logit = self.state.get(p, {}).get("geo_logit")
                if logit is not None:
                    shares.append(float(torch.sigmoid(logit)))
        return shares
