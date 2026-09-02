import math
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2CLS, ACT2FN
from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# The retrieval bank is budgeted against the dense FFN it replaces, so it tracks
# the model instead of pinning an absolute expert count. Bank params are
# ROWS_PER_EXPERT * num_experts * hidden while a GLU FFN is ~4 * hidden^2, so
# num_experts = BANK_WIDTH_MULTIPLE * hidden * 2 / ROWS_PER_EXPERT holds the
# ratio at (BANK_WIDTH_MULTIPLE / 2)x the dense FFN at EVERY width - the fixed
# count could not, being linear in hidden against the dense FFN's quadratic (12x
# the GLU at hidden=64, 0.68x at hidden=1024). 4 reproduces this module's
# historical 32^2 = 1024 experts at the config's default hidden_size of 256 in
# the two-row case: the existing choice generalized, not a new tuning.
BANK_WIDTH_MULTIPLE: int = 4

# Rows a single expert occupies in the banks. An ungated expert is rank-1:
# ``up_e * act(x . down_e)``, so two. A ``glu`` expert adds a gate vector, so
# three - and the budget above divides by this, which is what keeps the two
# variants comparable at equal parameter count rather than making `glu` a 1.5x
# capacity increase wearing an architecture change's clothes.
ROWS_PER_EXPERT: int = 2
ROWS_PER_GLU_EXPERT: int = 3

# Floor on the product-key sub-query width, below which the query is too narrow
# to discriminate the key set. Fixed and model-agnostic.
MIN_KEY_DIMS: int = 16

# Experts retrieved per head. A granularity, not a width, so it does not scale
# with the model; clamped to num_keys, since topk cannot outrun the key set.
TOP_K: int = 8


class ParameterEfficientExpertRetrieval(BaseDense):
    """
    This class implements the Parameter-Efficient Expert Retrieval (PEER) mechanism:
    https://arxiv.org/abs/2407.04153v1

    PEER combines aspects of product key memory and mixture of experts,
    using factorized keys for efficient expert retrieval. It enables each token
    to select its own set of experts for processing.

    Every dimension is derived from ``config`` unless explicitly overridden, so
    the module fits whatever model it is dropped into (see the two invariants
    on ``BANK_WIDTH_MULTIPLE`` and ``key_dims`` below). Overrides are for
    registry profiles; nothing here needs a per-experiment knob.
    """

    # Opt out of parameter-merging routers (praxis/routers/targeting.py). PEER
    # already routes per TOKEN, over its own bank of thousands of tiny experts;
    # wrapping a per-batch merge around that adds strictly coarser routing at
    # the cost of replicating the largest module in the model. Nothing to gain,
    # everything to pay.
    MERGE_OPAQUE: bool = True

    def __init__(
        self,
        config: ConfigType,
        key_dims: Optional[int] = None,
        num_experts: Optional[int] = None,
        num_heads: Optional[int] = None,
        k: Optional[int] = None,
        offset_heads: bool = False,
        sparse: bool = False,
        act_value: Optional[str] = None,
        act_alt: Optional[str] = None,
        glu: bool = False,
    ):
        """
        Initialize the PEER module.

        Args:
            config: Configuration object containing PEER parameters
            key_dims: product-key sub-query width. Default: hidden_size //
                (2 * num_heads), which makes the query net exactly one
                attention-sized projection (hidden -> hidden), floored at
                MIN_KEY_DIMS.
            num_experts: retrieval bank size, rounded to a perfect square.
                Default: BANK_WIDTH_MULTIPLE * hidden_size * 2 /
                rows_per_expert, so the bank keeps a constant PARAMETER ratio to
                the dense FFN at any width and under either expert form. NOTE: this is
                PEER's own bank, unrelated to `config.num_experts` (the router's
                expert count) - the names collide but the quantities do not.
            num_heads: independent retrieval heads. Default: config.num_heads.
            k: experts retrieved per head. Default: TOP_K, clamped to num_keys.
            act_alt: a SECOND activation for the branch ``self.act`` drives,
                carried by the back half of the expert BANK while the front half
                keeps ``config.activation``. Not a width change and not an extra
                nonlinearity - the same function slot, filled two different ways
                depending on which expert was retrieved.
            glu: if True, every expert is a gated unit rather than a rank-1
                projection: ``up_e * (act(x . gate_e) * (x . down_e))`` instead
                of ``up_e * act(x . down_e)``. This is the same change SwiGLU
                makes to a dense FFN, applied per retrieved expert - the
                multiplicative term lets one expert suppress its own
                contribution on a per-token basis, which a single activation
                cannot. Costs a third bank row per expert, so the auto-sized
                expert count shrinks by 2/3 to hold the parameter budget: the
                comparison against ``peer`` is capacity-matched, trading expert
                COUNT for per-expert expressiveness.
            sparse: if True, the expert banks emit sparse gradients (only the
                selected rows get a grad/optimizer update), which is what lets
                `num_experts` scale without paying dense grad + optimizer state
                on every untouched expert. Requires a sparse-aware optimizer
                (e.g. torch.optim.SparseAdam); Lion/Muon and the schedule-free
                optimizers here reject sparse grads, so it is off by default.
        """
        super().__init__()

        hidden_size = config.hidden_size
        self.num_heads: int = num_heads if num_heads is not None else config.num_heads
        self.offset_heads: bool = offset_heads
        self.num_sets: int = 1 if not self.offset_heads else self.num_heads
        self.glu: bool = glu
        self.rows_per_expert: int = ROWS_PER_GLU_EXPERT if glu else ROWS_PER_EXPERT

        # Product-Key retrieval factorizes the expert index into two key lookups,
        # so the bank is num_keys^2 by construction. Auto-sizing rounds the
        # budgeted row count to the nearest square and splits it across the
        # per-head sets, so offset_heads redistributes the bank rather than
        # multiplying it.
        if num_experts is None:
            # The 2 / rows_per_expert factor is what makes `glu` capacity-matched
            # against `peer` instead of 1.5x larger.
            budgeted_rows = (
                BANK_WIDTH_MULTIPLE
                * hidden_size
                * ROWS_PER_EXPERT
                / self.rows_per_expert
                / self.num_sets
            )
            self.num_keys: int = max(2, round(math.sqrt(budgeted_rows)))
        else:
            assert (
                num_experts**0.5
            ).is_integer(), "`num_experts` needs to be a perfect square"
            self.num_keys = int(math.sqrt(num_experts))
        self.num_experts: int = self.num_keys**2

        # The query net emits 2 (product-key halves) * num_heads * key_dims, so
        # this default makes retrieval cost exactly one attention-sized
        # projection. The floor wins when the head count would starve the
        # sub-query, widening the projection past hidden_size rather than
        # degenerating the retrieval.
        if key_dims is None:
            key_dims = max(MIN_KEY_DIMS, hidden_size // (2 * self.num_heads))
        self.key_dims: int = key_dims

        # A narrow model can budget fewer keys than the default granularity asks
        # for; topk would raise rather than clamp on its own.
        self.k: int = min(k if k is not None else TOP_K, self.num_keys)

        self.hidden_size: int = hidden_size
        self.sparse: bool = sparse
        # Second activation for the GLU value branch (default: identity,
        # i.e. unchanged behaviour). Named by a profile as `act_value`.
        self.act_value: nn.Module = ACT2CLS[act_value]() if act_value else nn.Identity()

        # No parity constraint on hidden_size. Every use of it here is a
        # projection width, never a split: the `2` throughout is the
        # product-key half count, which the query net *emits*
        # (`key_dims * num_heads * 2`) rather than carving out of the model
        # width. The one arithmetic contact, `hidden_size // (2 * num_heads)`,
        # is a floor already guarded by MIN_KEY_DIMS. Reference product-key
        # implementations assert on evenness because they project to `dim` and
        # `.chunk(2)` it; this one does not, so the assert that used to live
        # here was inherited, not earned. Verified at odd widths (111, 65, 257,
        # 33, 3): correct output shape, finite grads, and the same expert
        # count and key_dims as the neighbouring even width.

        class Permute(nn.Module):
            """Permute dimensions of tensor for product key memory."""

            def __init__(self):
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                """
                Permute dimensions [p, b, n, h, d] → [p, b, n, h, d]

                Args:
                    x: Input tensor

                Returns:
                    Permuted tensor
                """
                return x.permute(2, 0, 1, 3, 4).contiguous()

        # BatchNorm for combined partitions and heads
        class BatchNorm1d(nn.BatchNorm1d):
            """BatchNorm1d that handles sequence dimension."""

            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)

            def forward(self, x: Tensor) -> Tensor:
                """
                Apply batch norm across batch and sequence dimensions.

                Args:
                    x: Input tensor of shape [batch_size, seq_len, dim]

                Returns:
                    Normalized tensor of same shape
                """
                b, s, d = x.shape
                x = x.view(b * s, d)
                x = super().forward(x)
                return x.view(b, s, d)

        self.queries = nn.Sequential(
            BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, key_dims * self.num_heads * 2, bias=False),
            nn.Unflatten(-1, (2, self.num_heads, key_dims)),
            Permute(),
        )

        self.keys = nn.Parameter(
            torch.randn(self.num_heads, self.num_keys, 2, key_dims)
        )
        self.down = nn.Embedding(
            self.num_experts * self.num_sets, hidden_size, sparse=sparse
        )
        # The gate half of a GLU expert. Kept as its own bank rather than a
        # double-width `down` so the two projections can share `_project` and
        # the sparse-gradient path unchanged.
        self.gate = (
            nn.Embedding(self.num_experts * self.num_sets, hidden_size, sparse=sparse)
            if glu
            else None
        )
        self.act = ACT2FN[config.activation]
        # A second activation for the SAME function slot, carried by the back
        # half of the bank. The split is keyed on the EXPERT INDEX, and the two
        # alternatives are both worse:
        #
        #   the k axis is RANK. `topk` returns descending, so index j is the
        #   j-th best-scoring expert for that token. Splitting there would hand
        #   one activation the high-scoring experts systematically.
        #   the head axis is not a property of the expert. With `offset_heads`
        #   False - the default, and what every config here uses - all heads
        #   share ONE bank, so expert 17 would be periodic when head 0 retrieved
        #   it and non-periodic when head 2 did. The same row would be trained
        #   under two different functions. (It is also unavailable at
        #   `num_heads: 1`, which is this line's actual setting.)
        #
        # Keyed on the index, the function class is a persistent property of a
        # bank row: an expert trains under one activation for the whole run and
        # specializes into it. `% num_experts` makes that hold per-set under
        # `offset_heads` too, rather than giving set 0 one class and set 1 the
        # other.
        #
        # WHY THIS IS NOT `act_value`. `act_value` fills the GLU's empty LINEAR
        # slot, adding a nonlinearity every channel then has to pass through
        # (`peer_dual`). This one replaces `self.act` for half the experts and
        # adds nothing: total nonlinear depth is unchanged, and the GLU's linear
        # branch survives. Cheaper hypothesis, and separable from that one.
        self.act_alt = ACT2FN[act_alt] if act_alt else None
        # Front half keeps `config.activation`; the test is `index >= split`, so
        # an odd bank hands the alternate the extra expert (289 -> 144/145). The
        # split is a ratio, not a contract, and one row out of hundreds is not
        # worth an assert or a rounding rule.
        self.act_split: int = self.num_experts // 2
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.EmbeddingBag(
            self.num_experts * self.num_sets, hidden_size, mode="sum", sparse=sparse
        )
        self.init_weights()

    def init_weights(self, keys_std: float = 0.02) -> None:
        """Init the product keys and the expert banks by their TRUE fan-in.

        Neither PEER (arXiv 2407.04153) nor the product-key memory it builds on
        (arXiv 1907.05242) specifies how to initialize experts; PKM says only
        that keys are "randomly initialized". So this is derived rather than
        cited, and the derivation is the point.

        The banks are lookup tables, not weight matrices. Each row is one
        expert's vector, and only ``num_heads * k`` of them participate in any
        token - the other rows are untouched. That makes ``num_experts`` a
        LOOKUP dimension, not a fan, and it must not appear in a variance
        formula.

        Xavier was the previous choice and does exactly that: it reads
        ``fan_out = num_experts`` off the tensor shape, so the init scale falls
        as the bank grows, for no reason connected to the computation. At a
        289-expert bank that already attenuates the module's output to 0.065x
        its input at init; the factor is ~32x at a paper-scale 2^20 bank. A
        near-silent FFN that has to climb its way back up is a poor starting
        point, and it made bank size and init scale impossible to vary
        independently.

        The fan-in that each bank actually has:

        * ``down`` and ``gate`` project the input onto one expert vector, so
          their fan-in is ``hidden_size``.
        * ``up`` is summed over the retrieved experts (an EmbeddingBag in
          ``sum`` mode weighted by SIGMOID scores, which do not normalize to 1
          the way a softmax would), so its fan-in is the retrieval fan-out
          ``num_heads * k``.

        Measured at hidden_size=111, k=8: output/input std goes 0.065 -> 0.59,
        against 185 for the reference implementation's ``nn.Embedding`` default
        of N(0, 1). Both alternatives are independent of bank size; only this
        one is also scale-preserving.
        """
        nn.init.normal_(self.keys, std=keys_std)

        projection_std = self.hidden_size**-0.5
        nn.init.normal_(self.down.weight, std=projection_std)
        if self.gate is not None:
            nn.init.normal_(self.gate.weight, std=projection_std)

        # Summed over the retrieved experts, so the fan-in is that fan-out.
        nn.init.normal_(self.up.weight, std=(self.num_heads * self.k) ** -0.5)

    def extra_repr(self) -> str:
        # Fields only. ``num_experts`` is ``num_keys ** 2`` by construction -
        # the product-key grid is two-dimensional - but a parenthetical inside
        # a value is not a field, and it broke the key=value listing that the
        # rest of the module tree prints.
        return (
            f"num_experts={self.num_experts}, num_keys={self.num_keys}, "
            f"key_dims={self.key_dims}, num_heads={self.num_heads}, k={self.k}, "
            f"expert={'glu' if self.glu else 'rank1'}, "
            + (
                f"act_split={self.act_split}/{self.num_experts - self.act_split} experts, "
                if self.act_alt is not None
                else ""
            )
            + f"projection={'gather' if self._gathers() else 'dense'}"
        )

    def _gathers(self) -> bool:
        """Whether to gather expert rows before projecting, or project against
        the whole bank and gather after. The two compute the same thing; they
        differ only in which intermediate is materialized and retained for
        backward - ``[b, n, h, k, d]`` for the gather, ``[b, n, N]`` for the
        dense path - so the smaller one wins. The paper's Algorithm 1 gathers,
        which is right at its N >= 10^6, and it notes the fusion there "may
        require specialized hardware kernels". Our banks are budgeted to
        ~4 * hidden_size, orders of magnitude below h*k*d, so the dense path is
        the correct end of that trade: measured 8.5x less activation memory at
        a 256-wide model (2.38 -> 0.28 GB per call), where gathering OOMs a
        16GB card outright at depth 12. Structural, so it re-decides itself if
        the bank ever outgrows the retrieval fan-out.

        Sparse banks always gather: sparse gradients come from the embedding
        lookup, and projecting against ``.weight`` would densify them.

        A ``glu`` expert projects twice (gate and value), so the dense path's
        ``[b, n, N]`` intermediate is built twice - still far below the gather's
        ``[b, n, h, k, d]`` at our bank sizes, and the smaller bank a gated
        expert budgets shrinks it further.
        """
        return self.sparse or (
            self.num_experts * self.num_sets
            > self.num_heads * self.k * self.hidden_size
        )

    def _project(self, inputs: Tensor, bank: nn.Embedding, indices: Tensor) -> Tensor:
        """``x . w_e`` for each selected expert -> [b, n, h, k]."""
        if self._gathers():
            return torch.einsum("b n d, b n h k d -> b n h k", inputs, bank(indices))
        b, n = indices.shape[:2]
        projected = inputs @ bank.weight.T  # [b, n, num_experts * num_sets]
        return projected.gather(-1, indices.reshape(b, n, -1)).view_as(indices)

    def _activate(self, projected: Tensor, indices: Tensor) -> Tensor:
        """Apply the expert activation to ``[b, n, h, k]``, split by expert.

        With no ``act_alt`` this is just ``self.act``. With one, an element takes
        the alternate iff the expert it came from lives in the back half of the
        bank, so the shape and the ordering are untouched and everything
        downstream is unaware a split happened.

        BOTH activations are evaluated on the whole tensor and one is then
        selected, because the mask is data-dependent and ragged - a token's
        eight retrieved experts are an arbitrary mix of the two halves, so there
        is no contiguous slice to hand each branch. The cost is one extra
        elementwise pass over ``[b, n, h, k]``, which at ``h*k`` in the tens is
        nothing; gradients still reach only the selected elements.

        The consequence worth naming is for STATEFUL activations. ``Servant``
        standardizes against running statistics of live token energy, and here
        that energy is reduced over the full retrieved set rather than over its
        own experts. That is a consistent, population-level reference rather
        than a mismatched one - which is the failure that actually bites
        ([[project_energy_signal_saturation]]) - but it does mean the two
        branches share a view of "how much is going on in this token".
        """
        if self.act_alt is None:
            return self.act(projected)
        alt = (indices % self.num_experts) >= self.act_split
        return torch.where(alt, self.act_alt(projected), self.act(projected))

    def forward(
        self,
        inputs: Tensor,
        current_depth: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Forward pass through the PEER module.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            current_depth: unused - retrieval is depth-agnostic, but the
                BaseDense contract passes it to every FFN.

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Generate queries
        queries = self.queries(
            inputs
        )  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # Compute similarities using Einstein summation
        sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, self.keys)

        # For each partition, get top-k indices and scores
        scores_parts, indices_parts = sim.topk(self.k, dim=-1)
        scores_x, scores_y = scores_parts
        indices_x, indices_y = indices_parts

        # Compute Cartesian product of top-k indices and scores
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        # Flatten last two dimensions
        all_scores = all_scores.view(
            *all_scores.shape[:-2], math.prod(all_scores.shape[-2:])
        )
        all_indices = all_indices.view(
            *all_indices.shape[:-2], math.prod(all_indices.shape[-2:])
        )

        # Get top expert keys from the Cartesian product
        scores, pk_indices = all_scores.topk(self.k, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.offset_heads:
            head_expert_offsets = (
                torch.arange(self.num_heads, device=inputs.device) * self.num_experts
            )
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        # Project the input onto each retrieved expert's down vector
        outputs = self._project(inputs, self.down, indices)

        # A GLU expert multiplies its activated gate by a linear branch, so the
        # activation gates the value instead of merely shaping it.
        if self.gate is not None:
            # The value branch carries `act_value` when a profile names one -
            # otherwise it stays linear and this is the ordinary GLU expert. Two
            # multiplied function classes, rather than one steering a linear
            # half; see praxis/dense/dual_act.py for the argument.
            outputs = self.act_value(outputs)
            outputs = (
                self._activate(self._project(inputs, self.gate, indices), indices)
                * outputs
            )
        else:
            outputs = self._activate(outputs, indices)

        # Apply sigmoid retrieval scores, then drop whole experts
        outputs = F.sigmoid(scores) * outputs
        outputs = self.dropout(outputs)

        # Aggregate via EmbeddingBag: the score-weighted sum over (heads, k) is
        # fused in the kernel, so the [b, n, h, k, d] up tensor is never built.
        b, n = indices.shape[:2]
        flat_indices = indices.reshape(b * n, -1)
        flat_weights = outputs.reshape(b * n, -1).to(self.up.weight.dtype)
        outputs = self.up(flat_indices, per_sample_weights=flat_weights)

        return outputs.view(b, n, -1)


if __name__ == "__main__":
    # Exercises the config-derived sizing across widths and head counts: every
    # dimension below is derived, not passed. Prints the two invariants the
    # defaults are built on (bank/dense ratio, query projection == hidden) so a
    # regression in either is visible rather than silent.
    from praxis import PraxisConfig
    from praxis.dense.glu import GatedLinearMLP

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    print(
        f"{'hidden':>7} {'heads':>6} {'expert':>7} {'experts':>9} {'key_dims':>9} "
        f"{'k':>3} {'GLU':>10} {'PEER':>10} {'ratio':>7} {'q_out/hidden':>13}"
    )
    for hidden_size in (32, 64, 128, 256, 512, 1024):
        for num_heads in (4, 16):
            for glu in (False, True):
                config = PraxisConfig()
                config.hidden_size = hidden_size
                config.num_heads = num_heads
                config.activation = "gelu"
                config.dropout = 0.1

                peer = ParameterEfficientExpertRetrieval(config, glu=glu)
                dense = GatedLinearMLP(config)
                q_out = 2 * peer.num_heads * peer.key_dims

                inputs = torch.randn(2, 16, hidden_size)
                outputs = peer(inputs, current_depth=0)
                assert outputs.shape == inputs.shape, (outputs.shape, inputs.shape)
                assert peer.k <= peer.num_keys, "topk cannot outrun the key set"

                print(
                    f"{hidden_size:>7} {num_heads:>6} "
                    f"{'glu' if glu else 'rank1':>7} {peer.num_experts:>9} "
                    f"{peer.key_dims:>9} {peer.k:>3} {count(dense):>10} "
                    f"{count(peer):>10} {count(peer)/count(dense):>6.2f}x "
                    f"{q_out/hidden_size:>12.2f}x"
                )
