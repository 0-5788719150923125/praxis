"""VEAR: Variance-driven Experts with Adaptive Routing.

The repo's own variant on SMEAR (praxis/routers/smear.py), and unlike the
modular reorganization of SMEAR itself this one IS a departure from the paper
rather than a return to it. Two additions, both aimed at making the merged
geometry discrete and the deviations distinct:

  * SHARPEN. The coefficients are raised to a power and renormalized before the
    merge, so a target selects a near-single deviation rather than blending.
  * REPULSION. A penalty on the mean off-diagonal cosine between a target's own
    expert affinity directions, so its deviations occupy distinct niches instead
    of converging on one.

Read the sharpening sceptically. Applied on top of a BATCH reduction it cannot
add per-input discreteness - there is one merged geometry either way - and all
it does is drive the losing deviations to zero gradient. That is the documented
mechanism behind abstractinator-g's dead experts, where three of four merge
weights sat at ~1e-21. Under the default per-example reduction it is a more
honest proposition, because a sharpened per-example coefficient really does
select a different geometry for different inputs, and because base-plus-deviation
means a starved deviation now costs its rank rather than a whole block. That is
the experiment this class exists to run, not a claim it already won.

Constants are baked, per the project's tuning-free stance.
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.routers.smear import SMEAR

# How hard to peak a target's coefficients before merging: ``p**GAMMA``
# renormalized. Carried over unchanged from the whole-block VEAR so the two are
# directly comparable.
VEAR_SHARPEN: float = 4.0

# Weight on the inter-expert repulsion. Applied WITHIN each target's own
# coefficient rows: the goal is that a given module's deviations point in
# distinct directions, not that unrelated modules disagree with each other.
VEAR_REPULSION: float = 0.01


class VEAR(SMEAR):
    """SMEAR with sharpened routing and repelled experts."""

    SHARPEN: float = VEAR_SHARPEN

    def router_aux_loss(self) -> Dict[str, Tensor]:
        """Inter-expert repulsion, collected ONCE per step outside the forward.

        Parameter-only, so computing it inside the gradient-checkpointed
        recurrent forward would both add it once per pass and let it escape the
        checkpointed region - the two hazards the whole-block VEAR documented
        and the reason this rides ``decoder.router_aux_losses`` instead.
        """
        if not self.training or self.num_experts < 2:
            return {}
        w = self.router.weight.view(len(self.targets), self.num_experts, -1)
        w = F.normalize(w, dim=-1)
        sim = torch.bmm(w, w.transpose(1, 2))  # [T, N, N]
        eye = torch.eye(self.num_experts, device=w.device, dtype=w.dtype)
        off = (sim - eye).abs().sum() / (
            len(self.targets) * self.num_experts * (self.num_experts - 1)
        )
        return {"vear_repulsion": VEAR_REPULSION * off}
