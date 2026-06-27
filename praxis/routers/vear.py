"""VEAR: Variance-driven Experts with Adaptive Routing.

A SMEAR variant. Where SMEAR soft-merges expert parameters by the batch-mean
routing - pulling toward a single blended geometry (the convex-hull mean) - VEAR
biases toward DISCRETE, UNIQUE geometries:

  * Discrete: it SHARPENS the per-sequence routing before the merge, so the merged
    expert approximates a *selection* rather than an average.
  * Unique: a repulsion penalty on the router's per-expert affinity directions
    drives the experts to own distinct input niches and specialize into distinct
    geometries - high between-expert variance, the "variance-driven" part.

Honest limit inherited from SMEAR: the merge reduces to ONE expert per batch
(``routing_probs.mean(dim=0)`` -> one ``functional_call``), so VEAR's discreteness
is sharpened-selection + unique-experts, not per-sequence routing. Truly per-input
discrete geometries would require leaving the single-merge design (a future
router). The repulsion surfaces as a ``LossContainer`` aux - the channel Taxus
uses and that provably reaches the loss.

Constants are baked (fixed, model-agnostic), per the project's tuning-free stance.
Companion: next/roadmap.md (geometry banks + voting), praxis/routers/smear.py.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from praxis.routers.smear import SMEAR

# How hard to peak the per-sequence routing before merging: ``p**GAMMA`` renormalized.
# >1 concentrates mass onto the dominant expert(s) -> a more discrete geometry.
VEAR_SHARPEN: float = 4.0
# Weight on the inter-expert repulsion (mean off-diagonal |cosine| of the router's
# per-expert affinity rows). Drives experts to distinct niches -> unique geometries.
VEAR_REPULSION: float = 0.01
# Upper bound on the current_depth sweep that materializes per-depth lazy
# ModuleList entries (e.g. ArcGLU.act) before the first merge. Early-exits as soon
# as no uninitialized params remain, so this is just a safety ceiling.
_MAX_INIT_DEPTH: int = 128


class VEAR(SMEAR):
    """Variance-driven Experts with Adaptive Routing (sharpened, repelled SMEAR)."""

    # --- discrete: sharpen routing before SMEAR's batch-mean merge -------------
    def _merge_expert_parameters(self, routing_probs: Tensor, current_depth: int = 0):
        # Sharpen so the batch-mean merge SELECTS a near-single expert rather than
        # averaging. The variance-driven repulsion is NOT here: it's a parameter-
        # only loss surfaced via aux_loss() so it can't escape a checkpointed
        # forward (double-backward) or get added once per recurrent depth.
        sharp = routing_probs.pow(VEAR_SHARPEN)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return super()._merge_expert_parameters(sharp, current_depth)

    def router_aux_loss(self) -> dict:
        """Variance-driven repulsion as a named loss, collected ONCE per step
        OUTSIDE the forward (``decoder.router_aux_losses`` ->
        ``modeling._collect_aux_losses``). It depends only on ``router.weight``;
        computing it inside the gradient-checkpointed, recurrent forward would add
        it once per depth AND let a parameter-only loss escape the checkpoint
        (double-backward). Training-only."""
        if not self.training or len(self.experts) < 2:
            return {}
        return {"vear_repulsion": VEAR_REPULSION * self._inter_expert_repulsion()}

    def _inter_expert_repulsion(self) -> Tensor:
        """Mean off-diagonal |cosine| between the router's per-expert affinity
        directions. 0 = orthogonal experts (distinct niches); high = redundant."""
        n = self.router.weight.shape[0]
        if n < 2:
            return self.router.weight.new_zeros(())
        W = F.normalize(self.router.weight, dim=1)  # [N, H]
        sim = W @ W.t()  # [N, N]
        off = sim - torch.eye(n, device=W.device, dtype=W.dtype)
        return off.abs().sum() / (n * (n - 1))

    def _ensure_experts_initialized(self, expert_args: tuple) -> None:
        """Lazy blocks only materialize their params on a real forward, but SMEAR
        merges expert params *without* running each expert (it ``functional_call``s
        only the merged ``experts[0]``), so the merge would hit an
        ``UninitializedParameter``. Materialize every expert before the first merge.

        The hard case: a per-depth ``ModuleList`` (e.g. ``ArcGLU.act``, holding
        custom lazy Serpent activations) runs only ``act[current_depth]`` per
        forward, so one call leaves the siblings lazy. We **sweep current_depth**
        so each entry runs through the block's OWN forward (correct shapes, any
        lazy type), then fall back to a sibling-borrow probe for anything not
        indexed by depth. Gated on uninitialized params -> a one-time no-op after."""

        def any_uninit() -> bool:
            return any(
                isinstance(p, UninitializedParameter)
                for e in self.experts
                for p in e.parameters()
            )

        if not any_uninit():
            return
        args = list(expert_args)
        # Router-mode expert args: (inputs, attention_mask, past_key_values,
        # current_state, current_depth, block_ids) - current_depth at index 4.
        depth_idx = 4 if len(args) >= 6 and isinstance(args[4], int) else None
        with torch.no_grad():
            if depth_idx is None:
                for e in self.experts:
                    e(*args)
            else:
                for d in range(_MAX_INIT_DEPTH):
                    if not any_uninit():
                        break
                    args[depth_idx] = d
                    for e in self.experts:
                        e(*args)
            if any_uninit():  # stragglers not reached by the depth sweep
                self._materialize_lazy_siblings(expert_args[0].device)

    def _materialize_lazy_siblings(self, device) -> None:
        """Probe still-lazy entries of any per-depth ``ModuleList`` (only one
        entry runs per forward). Borrows a feature dim from ANY sibling that has
        real params - including one that already graduated to its concrete class
        (``LazyLinear`` swaps itself to ``Linear`` on init) - which the model
        bootstrap's helper misses because it searches only still-lazy entries."""
        from torch.nn import ModuleList
        from torch.nn.modules.lazy import LazyModuleMixin

        for module in self.modules():
            if not isinstance(module, ModuleList):
                continue
            lazy = [
                m
                for m in module
                if isinstance(m, LazyModuleMixin) and m.has_uninitialized_params()
            ]
            if not lazy:
                continue
            feat_dim = None
            for sib in module:
                for p in sib.parameters(recurse=False):
                    if not isinstance(p, UninitializedParameter) and p.dim() >= 1:
                        feat_dim = p.shape[-1]
                        break
                if feat_dim is not None:
                    break
            if feat_dim is None:
                continue
            dummy = torch.zeros(1, feat_dim, device=device)
            for entry in lazy:
                entry(dummy)

    def _router_forward(self, *args):
        # args = (layer, inputs, attention_mask, past_key_values, current_state,
        #         current_depth, block_ids); experts are called with args[1:].
        self._ensure_experts_initialized(tuple(args[1:]))
        return super()._router_forward(*args)

    def _direct_forward(self, inputs: Tensor, current_state: Optional[Tensor]):
        self._ensure_experts_initialized((inputs, current_state))
        return super()._direct_forward(inputs, current_state)
