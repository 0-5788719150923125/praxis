from typing import List, Optional

import torch
import torch.nn as nn


class AdditiveEmbedding(nn.Module):
    """Sum several embedding primitives applied to the same token stream.

    Used to compose byte-latent inputs (e.g. a base byte table plus an
    additive n-gram hash embedding) while keeping each primitive independent.
    """

    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.embeddings = nn.ModuleList(modules)

    def tie_source(self) -> Optional[nn.Module]:
        """The sub-embedding whose table is the weight-tying target, if any."""
        for module in self.embeddings:
            if hasattr(module, "weight"):
                return module
        return None

    @property
    def weight(self) -> Optional[torch.Tensor]:
        source = self.tie_source()
        return source.weight if source is not None else None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = None
        for module in self.embeddings:
            embed = module(tokens)
            out = embed if out is None else out + embed
        return out

    def __repr__(self) -> str:
        parts = " + ".join(repr(m) for m in self.embeddings)
        return f"{self.__class__.__name__}({parts})"
