"""Low-rank second-moment tracker (telemetry only).

A passthrough wrapper that estimates a factored second moment of the gradient
without changing the update - so the second-moment dashboard cards light up even
under Lion (sign momentum, no ``exp_avg_sq``), which is the praxis default.

For a 2D weight ``W in [out, in]`` it keeps Adafactor-style rank-1 factors
(Shazeer & Stern 2018): a row EMA ``vr in [out]`` (per-output variance) and a
column EMA ``vc in [in]`` (per-input variance) of ``g**2``. These are the
input/output geometry of the gradient; the full second moment is reconstructed
on demand as ``v[i,j] ~= vr[i]*vc[j]/mean(vr)`` (exact for a rank-1 separable
``g**2``). Storage is ``O(out+in)``, not ``O(out*in)``. Non-2D params keep a full
vector second moment (already cheap).

It does NOT precondition: Lion's ``sign`` update is per-coordinate scale-
invariant, so dividing the gradient by ``sqrt(v)`` changes nothing. This only
tracks. ``get_second_moment`` exposes the reconstructed ``v`` so the optimizer
metrics can compute second-moment RMS, the implied Adam update, etc. Multiple
factors at lower ranks (a second-moment pyramid) are the natural extension.
"""

from collections import defaultdict

import torch
from pytorch_optimizer.base.optimizer import BaseOptimizer

EPS = 1e-12
BETA2 = 0.999  # frozen second-moment decay (no per-experiment tuning)


class LowRankSecondMoment(BaseOptimizer):
    """Track a factored second moment of the gradient; pass the step through."""

    def __init__(self, optimizer, beta2: float = BETA2, **kwargs):
        self.validate_range(beta2, "beta2", 0.0, 1.0, "[)")
        self.beta2 = float(beta2)
        self.optimizer = self.load_optimizer(optimizer, **kwargs)
        self.state = defaultdict(dict)
        self.defaults = self.optimizer.defaults

    def __str__(self) -> str:
        return "LowRankSecondMoment"

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)

    def train(self):
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

    def eval(self):
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

    def state_dict(self):
        return {
            "low_rank_moment_state": self.state,
            "beta2": self.beta2,
            "base_optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state) -> None:
        self.state = state["low_rank_moment_state"]
        self.beta2 = state.get("beta2", self.beta2)
        self.optimizer.load_state_dict(state["base_optimizer"])

    def get_second_moment(self, p):
        """Reconstruct the (bias-corrected) per-coordinate second moment for
        ``p``, or None if it has no statistics yet. Transient - not stored."""
        st = self.state.get(p)
        if not st or "t" not in st:
            return None
        bc = 1.0 - self.beta2 ** st["t"]
        if "v" in st:  # full vector (non-2D params)
            return st["v"] / bc
        vr, vc = st["vr"], st["vc"]  # rank-1 factors (2D params)
        v = torch.outer(vr, vc) / vr.mean().clamp_min(EPS)
        return (v / bc).reshape(p.shape)

    @torch.no_grad()
    def reset(self) -> None:
        pass

    def init_group(self, *args, **kwargs) -> None:
        pass  # wrapper overrides step entirely; satisfies the BaseOptimizer ABC

    @torch.no_grad()
    def step(self, closure=None):
        b = self.beta2
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g2 = p.grad * p.grad
                st = self.state[p]
                st["t"] = st.get("t", 0) + 1
                if g2.ndim == 2:  # factor the input/output geometry (rank 1)
                    if "vr" not in st:
                        st["vr"] = g2.new_zeros(g2.shape[0])
                        st["vc"] = g2.new_zeros(g2.shape[1])
                    st["vr"].mul_(b).add_(g2.mean(dim=1), alpha=1.0 - b)
                    st["vc"].mul_(b).add_(g2.mean(dim=0), alpha=1.0 - b)
                else:  # full vector second moment (cheap for non-2D params)
                    if "v" not in st:
                        st["v"] = torch.zeros_like(g2)
                    st["v"].mul_(b).add_(g2, alpha=1.0 - b)
        return self.optimizer.step(closure)
