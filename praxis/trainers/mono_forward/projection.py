"""Salvaged ``ProjectionMatrix`` from the deleted ``mono_forward_pipeline.py``.

This is the paper-faithful D2a projection: each layer gets its own
``[num_classes, hidden_size]`` matrix, and the goodness score is
``G = activations @ M^T``. The Phase 2-3 build uses D2b (shared output head
as the per-layer projection) instead, so this class is dead code at Phase 1
- it's kept in the tree as a hedge in case the D2a variant is needed later
for comparison or because D2b underperforms on a workload.

Per ``PROJECT_PLAN.md`` Phase 1 step 1: "Salvage the ``ProjectionMatrix``
class into ``praxis/trainers/mono_forward/projection.py`` in case we
need it later for the D2a variant; it will not be used in the initial D2b
build."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMatrix(nn.Module):
    """Per-layer projection matrix for D2a-style Mono-Forward.

    Computes ``G = activations @ M^T`` where ``M`` maps from ``hidden_size``
    to ``num_classes`` (typically the full vocabulary for a causal LM).
    """

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=(2.0 / hidden_size) ** 0.5)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute goodness scores: ``G = activations @ M^T``."""
        return F.linear(activations, self.weight)
