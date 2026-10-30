"""Multiscale residual quantization over fixed-length chunks of latents.

VAR's next-scale tokenizer (Tian et al. 2024), as Next-Concept Modeling applies
it to text (arXiv:2610.14162). Each chunk of ``C`` latents is described coarse to
fine: level ``k`` pools the residual to ``2**k`` vectors (area pooling over equal
spans), quantizes each against its own codebook, and spreads the selected codes
back over their spans. Level 0 is one code per chunk; the last level is one per
latent. Doubling is the structural schedule: every level halves the span.

The codebooks learn by EMA only. Nothing here takes a gradient, so the ladder
runs under ``no_grad`` and the caller passes gradient around it (straight-through).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from praxis.encoders.quantization.vector_quantizer import VectorQuantizer


class MultiScaleResidualVQ(nn.Module):
    """Coarse-to-fine residual codes, ``log2(chunk) + 1`` levels per chunk."""

    def __init__(
        self,
        dim: int,
        chunk: int,
        codebook_size: int,
        decay: float = 0.99,
        reset_interval: int = 250,
        stale_after: int = 1000,
    ) -> None:
        super().__init__()
        if chunk < 2 or chunk & (chunk - 1):
            raise ValueError(f"chunk must be a power of two >= 2, got {chunk}")
        self.dim = int(dim)
        self.chunk = int(chunk)
        self.codebook_size = int(codebook_size)
        self.lengths: Tuple[int, ...] = tuple(
            2**k for k in range(int(math.log2(chunk)) + 1)
        )
        self.levels = nn.ModuleList(
            VectorQuantizer(
                K=codebook_size,
                D=dim,
                ema=True,
                decay=decay,
                reset_codes=True,
                reset_interval=reset_interval,
                stale_after=stale_after,
            )
            for _ in self.lengths
        )
        # EMA owns the codebooks; an optimizer step would fight it.
        for level in self.levels:
            level.codebook.requires_grad_(False)
        self.register_buffer("seeded", torch.zeros((), dtype=torch.bool))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dim={self.dim}, chunk={self.chunk}, "
            f"codebook_size={self.codebook_size}, levels={len(self.lengths)})"
        )

    @staticmethod
    def pool(x: Tensor, valid: Tensor, length: int) -> Tuple[Tensor, Tensor]:
        """Masked mean of ``x`` [B, n, C, D] over ``length`` equal spans per chunk.

        Returns the pooled ``[B, n, length, D]`` and which spans hold any valid
        latent ``[B, n, length]``.
        """
        B, n, C, D = x.shape
        span = C // length
        weight = valid.to(x.dtype).view(B, n, length, span, 1)
        total = (x.view(B, n, length, span, D) * weight).sum(dim=3)
        count = weight.sum(dim=3)
        return total / count.clamp_min(1.0), count.squeeze(-1) > 0

    def spread(self, codes: Tensor, length: int) -> Tensor:
        """``[B, n, length, D]`` back to one vector per latent, ``[B, n, C, D]``."""
        return codes.repeat_interleave(self.chunk // length, dim=2)

    def lookup(self, level: int, codes: Tensor) -> Tensor:
        """Codebook vectors for ``codes`` of one level."""
        return self.levels[level].codebook[codes]

    @torch.no_grad()
    def reconstructions(self, codes: List[Tensor], valid: Tensor) -> List[Tensor]:
        """Cumulative reconstructions after each level, from given codes."""
        weight = valid.unsqueeze(-1).to(self.levels[0].codebook.dtype)
        total = None
        out = []
        for k, (level_codes, length) in enumerate(zip(codes, self.lengths)):
            part = self.spread(self.lookup(k, level_codes), length) * weight
            total = part if total is None else total + part
            out.append(total)
        return out

    @torch.compiler.disable
    @torch.no_grad()
    def forward(
        self, z: Tensor, valid: Tensor, learn: bool = True
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Quantize ``z`` [B, n, C, D] coarse to fine.

        Returns per level: codes ``[B, n, l]``, which codes are real
        ``[B, n, l]``, and the cumulative reconstruction ``[B, n, C, D]``.
        Codebooks seed and move (EMA, resets) only when ``learn`` and in
        training mode. Eager for the reason ``MultiStageResidualVQ.forward`` is.
        """
        learn = learn and self.training
        if learn and not bool(self.seeded):
            self._seed(z, valid)
        weight = valid.unsqueeze(-1).to(z.dtype)
        residual = z * weight
        total = torch.zeros_like(z)
        codes, masks, sums = [], [], []
        for level, length in zip(self.levels, self.lengths):
            pooled, mask = self.pool(residual, valid, length)
            level_codes = torch.zeros(mask.shape, dtype=torch.long, device=z.device)
            chosen = torch.zeros_like(pooled)
            if mask.any():
                rows = pooled[mask]
                if learn:
                    quantized, _, found, _ = level(rows.unsqueeze(0))
                    found, vectors = found.view(-1), quantized.squeeze(0)
                else:
                    found = level.nearest(rows)
                    vectors = level.codebook[found]
                level_codes[mask] = found
                chosen[mask] = vectors.to(chosen.dtype)
            part = self.spread(chosen, length) * weight
            total = total + part
            residual = residual - part
            codes.append(level_codes)
            masks.append(mask)
            sums.append(total)
        return codes, masks, sums

    @torch.no_grad()
    def _seed(self, z: Tensor, valid: Tensor) -> None:
        """Seed each level's codebook from the first training batch, level k from
        the residuals the seeded levels before it leave (SoundStream's RVQ
        seeding, arXiv:2107.03312). See ``MultiStageResidualVQ._seed``."""
        weight = valid.unsqueeze(-1).to(z.dtype)
        residual = z * weight
        for level, length in zip(self.levels, self.lengths):
            pooled, mask = self.pool(residual, valid, length)
            rows = pooled[mask]
            if rows.shape[0] == 0:
                return
            order = torch.randperm(rows.shape[0], device=rows.device)
            picked = rows[
                order[torch.arange(level.K, device=rows.device) % rows.shape[0]]
            ]
            scale = picked.float().pow(2).mean().sqrt().clamp_min(1e-12)
            level.seed(picked + 1e-3 * scale * torch.randn_like(picked))
            chosen = torch.zeros_like(pooled)
            chosen[mask] = level.codebook[level.nearest(rows)].to(chosen.dtype)
            residual = residual - self.spread(chosen, length) * weight
        self.seeded.fill_(True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint without the flag holds trained codebooks; never reseed.
        state_dict.setdefault(prefix + "seeded", torch.ones((), dtype=torch.bool))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def marginal_entropy(self, level: int) -> Optional[float]:
        """Entropy (nats) of a level's EMA code usage: the loss of a predictor
        that knows only how often each code appears."""
        counts = self.levels[level].ema_cluster_size.float()
        total = counts.sum()
        if float(total) <= 0:
            return None
        p = counts / total
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    @torch.no_grad()
    def telemetry(self) -> Dict[str, float]:
        """Per-level codebook health. Syncs to host; metrics interval only."""
        out: Dict[str, float] = {}
        log_k = math.log(max(self.codebook_size, 2))
        for k, level in enumerate(self.levels):
            out[f"vq_dead_frac_l{k}"] = float(level.dead_code_count) / level.K
            out[f"vq_resets_l{k}"] = float(level.reset_count_total)
            entropy = self.marginal_entropy(k)
            if entropy is not None:
                out[f"vq_usage_entropy_l{k}"] = entropy / log_k
        return out
