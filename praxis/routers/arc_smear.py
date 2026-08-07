"""Depth-aware SMEAR/VEAR: one merged geometry per recurrent pass.

Base SMEAR has a single router serving every recurrent depth. The routing input
is ``inputs.mean(dim=1)``, so the probabilities do vary with the hidden state as
it evolves down the loop - but nothing in the router *knows* which pass it is
on, and every depth shares one set of affinity directions. The merge is
therefore the same function of the hidden state at pass 1 and pass 6, and a
model that wants early passes to look different from late ones has to encode
that entirely in the hidden state.

These subclasses add the ArcAttention idiom (praxis/attention/arc.py:115): a
zero-initialized ``nn.Embedding(depth, num_experts)`` added to the router logits
and keyed by ``current_depth``. Zero-init makes this exactly the parent router
at step 0 - the bias is a learned departure from shared routing, not a new
initialization - and each pass can then pull the merge toward its own expert
mixture. Same parameter count in the experts, one ``depth * num_experts`` table
added (24 numbers at depth 6, 4 experts).

The bias sits on the LOGITS, before the softmax and before VEAR's ``p**4``
sharpening, so it moves which expert a pass selects without touching how hard it
selects. That ordering is deliberate: sharpening after the bias keeps VEAR's
discreteness argument intact.

``ArcVEAR`` is the same mixin over VEAR, so it keeps the sharpening, the
inter-expert repulsion and the load-balancing floor.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from praxis.routers.smear import SMEAR
from praxis.routers.vear import VEAR


class DepthBiasedRouting:
    """Per-recurrent-pass additive bias on the router logits.

    Mixin, not a base class: it has to sit in front of SMEAR/VEAR in the MRO so
    ``_route_logits`` resolves here first while everything else - the merge, the
    diagnostics, VEAR's repulsion - keeps coming from the parent.
    """

    def _init_depth_bias(self, config: Any) -> None:
        self.depth = getattr(config, "depth", 1) or 1
        self.depth_bias = nn.Embedding(self.depth, len(self.experts))
        # Identity at init: with a zero bias these routers are bit-identical to
        # their parents, so a config swap is a clean A/B rather than a reroll.
        nn.init.zeros_(self.depth_bias.weight)

    def _route_logits(self, router_input: Tensor, current_depth: int) -> Tensor:
        logits = super()._route_logits(router_input, current_depth)
        # The decoder loops past `depth` when halting samples deeper than the
        # table; wrap rather than raise, so a pass reuses an existing row.
        idx = int(current_depth) % self.depth
        depth_idx = torch.tensor(idx, device=logits.device)
        return logits + self.depth_bias(depth_idx)

    def get_metrics(self) -> dict:
        """Router metrics plus: are the per-depth biases specializing?

        Overrides ``get_metrics`` rather than declaring ``training_metrics``
        because ``get_metrics`` is the channel routers actually report on -
        ``BaseDecoder.get_metrics`` collects it (praxis/decoders/base.py:318)
        and nothing walks routers for ``training_metrics``.

        ``specialization`` is 0 both when the passes have collapsed onto one
        shared routing AND at zero-init, so read it as "has per-pass routing
        earned anything yet", not as a health check. Charted by the
        ``router_depth_bias`` entry in praxis/metrics/training_metrics.py.
        """
        from praxis.metrics.specialization import depth_dispersion

        out = super().get_metrics()
        stats = depth_dispersion(self.depth_bias.weight)
        if stats is not None:
            out["router_depth_specialization"] = stats["specialization"]
            out["router_depth_similarity"] = stats["similarity"]
        return out


class ArcSMEAR(DepthBiasedRouting, SMEAR):
    """SMEAR with a per-recurrent-pass routing bias."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self._init_depth_bias(config)


class ArcVEAR(DepthBiasedRouting, VEAR):
    """VEAR with a per-recurrent-pass routing bias.

    Keeps VEAR's sharpening, repulsion and balancing; the bias is applied to the
    logits beforehand, so it changes which expert a pass concentrates on rather
    than how sharply it concentrates.
    """

    def __init__(self, config: Any, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self._init_depth_bias(config)
