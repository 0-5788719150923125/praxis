import math
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# The retrieval bank is budgeted against the dense FFN it replaces, so it tracks
# the model instead of pinning an absolute expert count. Bank params are
# 2 * num_experts * hidden (a down row + an up row per expert) while a GLU FFN is
# ~4 * hidden^2, so num_experts = BANK_WIDTH_MULTIPLE * hidden holds the ratio at
# (BANK_WIDTH_MULTIPLE / 2)x the dense FFN at EVERY width - the fixed count could
# not, being linear in hidden against the dense FFN's quadratic (12x the GLU at
# hidden=64, 0.68x at hidden=1024). 4 reproduces this module's historical
# 32^2 = 1024 experts at the config's default hidden_size of 256: the existing
# choice generalized, not a new tuning.
BANK_WIDTH_MULTIPLE: int = 4

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

    def __init__(
        self,
        config: ConfigType,
        key_dims: Optional[int] = None,
        num_experts: Optional[int] = None,
        num_heads: Optional[int] = None,
        k: Optional[int] = None,
        offset_heads: bool = False,
        sparse: bool = False,
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
                Default: BANK_WIDTH_MULTIPLE * hidden_size, so the bank keeps a
                constant ratio to the dense FFN at any width. NOTE: this is
                PEER's own bank, unrelated to `config.num_experts` (the router's
                expert count) - the names collide but the quantities do not.
            num_heads: independent retrieval heads. Default: config.num_heads.
            k: experts retrieved per head. Default: TOP_K, clamped to num_keys.
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

        # Product-Key retrieval factorizes the expert index into two key lookups,
        # so the bank is num_keys^2 by construction. Auto-sizing rounds the
        # budgeted row count to the nearest square and splits it across the
        # per-head sets, so offset_heads redistributes the bank rather than
        # multiplying it.
        if num_experts is None:
            budgeted_rows = BANK_WIDTH_MULTIPLE * hidden_size / self.num_sets
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

        assert (hidden_size % 2) == 0, "`hidden_size` should be divisible by 2"

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
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.EmbeddingBag(
            self.num_experts * self.num_sets, hidden_size, mode="sum", sparse=sparse
        )
        self.init_weights()

    def init_weights(self, keys_std: float = 0.02) -> None:
        """Init the product keys (normal) and both expert banks (Xavier)."""
        nn.init.normal_(self.keys, std=keys_std)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts} ({self.num_keys}^2), "
            f"key_dims={self.key_dims}, num_heads={self.num_heads}, k={self.k}, "
            f"projection={'gather' if self._gathers() else 'dense'}"
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
        """
        return self.sparse or (
            self.num_experts * self.num_sets > self.num_heads * self.k * self.hidden_size
        )

    def _project(self, inputs: Tensor, bank: nn.Embedding, indices: Tensor) -> Tensor:
        """``x . w_e`` for each selected expert -> [b, n, h, k]."""
        if self._gathers():
            return torch.einsum("b n d, b n h k d -> b n h k", inputs, bank(indices))
        b, n = indices.shape[:2]
        projected = inputs @ bank.weight.T  # [b, n, num_experts * num_sets]
        return projected.gather(-1, indices.reshape(b, n, -1)).view_as(indices)

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

        # Apply sigmoid scores to activated outputs, then drop whole experts
        outputs = F.sigmoid(scores) * self.act(outputs)
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

    print(f"{'hidden':>7} {'heads':>6} {'experts':>9} {'key_dims':>9} "
          f"{'k':>3} {'GLU':>10} {'PEER':>10} {'ratio':>7} {'q_out/hidden':>13}")
    for hidden_size in (32, 64, 128, 256, 512, 1024):
        for num_heads in (4, 16):
            config = PraxisConfig()
            config.hidden_size = hidden_size
            config.num_heads = num_heads
            config.activation = "gelu"
            config.dropout = 0.1

            peer = ParameterEfficientExpertRetrieval(config)
            dense = GatedLinearMLP(config)
            q_out = 2 * peer.num_heads * peer.key_dims

            inputs = torch.randn(2, 16, hidden_size)
            outputs = peer(inputs, current_depth=0)
            assert outputs.shape == inputs.shape, (outputs.shape, inputs.shape)
            assert peer.k <= peer.num_keys, "topk cannot outrun the key set"

            print(f"{hidden_size:>7} {num_heads:>6} {peer.num_experts:>9} "
                  f"{peer.key_dims:>9} {peer.k:>3} {count(dense):>10} "
                  f"{count(peer):>10} {count(peer)/count(dense):>6.2f}x "
                  f"{q_out/hidden_size:>12.2f}x")
