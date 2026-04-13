"""Per-layer projection matrix for Mono-Forward training.

From "Mono-Forward: Backpropagation-Free Algorithm for Efficient
Neural Network Training" (https://arxiv.org/abs/2501.09238):

    Each layer in MF contains a projection matrix M_i whose
    dimensions are m x n, where m represents the number of possible
    categories, and n is the number of neurons in layer i.

    The goodness score G_i for the ith layer is computed as:
        G_i = a_i x M_i^T

    where a_i are the activations at layer i.

Each layer's projection matrix is an independent learnable parameter.
There is no weight sharing or synchronisation between layers'
projections - each layer optimises its own projection as part of its
local loss ``L_i = CE(softmax(G_i), labels)``.

This module exposes a ``.classifier`` property so it can be used as
a drop-in for the ``head`` parameter in
:func:`praxis.losses.compute_layer_wise_loss`, which reads
``head(hidden_states)`` for logits and ``head.classifier.weight``
for the cut-cross-entropy fast path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMatrix(nn.Module):
    """Per-layer projection matrix M_i for Mono-Forward.

    Computes goodness scores: ``G_i = a_i @ M_i^T``.

    Shape:
        - Input: ``[batch, seq_len, hidden_size]``
        - Output: ``[batch, seq_len, vocab_size]`` (goodness scores)
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=(2.0 / hidden_size) ** 0.5)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute goodness scores: ``G_i = a_i @ M_i^T``."""
        return F.linear(activations, self.weight)

    @property
    def classifier(self) -> "ProjectionMatrix":
        """Return self as the classifier for cut-CE compatibility.

        :func:`praxis.losses.compute_layer_wise_loss` reads
        ``head.classifier.weight`` for the cut-cross-entropy fast
        path. Since ``ProjectionMatrix`` stores its weight directly
        as ``self.weight``, returning ``self`` satisfies
        ``classifier.weight`` without an extra indirection.
        """
        return self
