"""Absorbing-state (masked) diffusion: the corruption, the ELBO term, and the
iterative-unmask decode loop.

This is the D3PM-absorbing -> MDLM -> LLaDA line, not Gaussian denoising.
Corruption REPLACES a token with a mask symbol rather than adding noise to it,
which is what makes diffusion work on discrete data at all. Three facts from the
reference that shape everything here:

1. The ELBO collapses to a weighted masked cross-entropy. For the linear
   schedule the per-row term is ``(1/t) * mean_over_positions(CE * masked)``,
   which is an unbiased estimator of the NLL bound. That is the whole training
   objective - there is no separate score network.

2. No timestep conditioning. MDLM and LLaDA both drop it: the fraction of mask
   symbols in the input already tells the network where on the schedule it is,
   so an AdaLN time embedding is a parameter that learns to reproduce something
   the input states outright. One fewer thing to debug on a first attempt.

3. The model is NOT causal. Every position attends to every other, which is
   sound here precisely because the positions being scored are the ones that
   were replaced - the answer is not in the input. See
   next/temporal_mesh_audit.md for the failure mode this avoids.

``t`` is sampled per ROW, not per batch, so one step sees many corruption levels
and the 1/t weighting stays finite (``eps`` floors it).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedDiffusion(nn.Module):
    """Corruption, objective and sampling for absorbing-state text diffusion.

    Holds no trainable parameters. It owns the mask id, the schedule and the
    diagnostics; the model owns the denoiser.
    """

    metric_descriptions = {
        "diffusion_ce": {
            "description": (
                "Mean cross-entropy over corrupted positions, in nats. Not "
                "comparable to an autoregressive loss, which measures a "
                "likelihood rather than reconstruction at a random level."
            ),
            "chart": {
                "title": "Diffusion CE",
                "group": "diffusion",
                "group_order": 12,
                "order": 0,
                "series_group": "diffusion_ce",
                "series_label": "model",
            },
        },
        "diffusion_unigram_ce": {
            "description": (
                "The same positions scored by the running unigram marginal. "
                "The floor a model that has learned nothing about structure "
                "still reaches."
            ),
            "chart": {
                "title": "Diffusion CE",
                "group": "diffusion",
                "order": 1,
                "series_group": "diffusion_ce",
                "series_label": "unigram",
            },
        },
        "diffusion_unigram_gap": {
            "description": (
                "diffusion_unigram_ce minus diffusion_ce, in nats: how far the "
                "model is ahead of the marginal. At or below zero it has learned "
                "no structure, whatever the loss curve does."
            ),
            "chart": {
                "title": "Diffusion Unigram Gap",
                "group": "diffusion",
                "order": 2,
            },
        },
        "diffusion_ce_low": {
            "description": (
                "Mean CE on rows corrupted below 34%. The easy end: most "
                "context is intact."
            ),
            "chart": {
                "title": "Diffusion CE by Corruption Level",
                "group": "diffusion",
                "order": 3,
                "series_group": "diffusion_bucket",
                "series_label": "low",
            },
        },
        "diffusion_ce_mid": {
            "description": "Mean CE on rows corrupted between 34% and 67%.",
            "chart": {
                "title": "Diffusion CE by Corruption Level",
                "group": "diffusion",
                "order": 4,
                "series_group": "diffusion_bucket",
                "series_label": "mid",
            },
        },
        "diffusion_ce_high": {
            "description": (
                "Mean CE on rows corrupted above 67%. A model that only learns "
                "the easy end shows a healthy total while this stays at "
                "chance - the split is the diagnostic, not the total."
            ),
            "chart": {
                "title": "Diffusion CE by Corruption Level",
                "group": "diffusion",
                "order": 5,
                "series_group": "diffusion_bucket",
                "series_label": "high",
            },
        },
        "diffusion_mask_rate": {
            "description": (
                "Mean sampled corruption level t. Uniform over [eps, 1], so "
                "this should sit near 0.5 and is a sanity check on the "
                "sampler rather than a result."
            ),
            "chart": {
                "title": "Diffusion Corruption",
                "group": "diffusion",
                "order": 6,
                "series_group": "diffusion_rate",
                "series_label": "sampled t",
            },
        },
        "diffusion_masked_frac": {
            "description": (
                "Fraction of valid positions actually replaced. Tracks "
                "diffusion_mask_rate; a persistent gap means padding or the "
                "at-least-one-mask fallback is distorting the schedule."
            ),
            "chart": {
                "title": "Diffusion Corruption",
                "group": "diffusion",
                "order": 7,
                "series_group": "diffusion_rate",
                "series_label": "realized",
            },
        },
        "diffusion_bits": {
            "description": (
                "diffusion_ce in bits. Per corrupted token, not per byte of "
                "text - it does not line up with val_byte_nll_bits."
            ),
            "chart": {
                "title": "Diffusion Bits",
                "group": "diffusion",
                "order": 8,
            },
        },
    }

    def __init__(
        self,
        mask_token_id: int,
        eps: float = 1e-3,
        vocab_size: Optional[int] = None,
        unigram_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.mask_token_id = int(mask_token_id)
        # Floors the sampled corruption level. 1/t is the ELBO weight, so t=0
        # is an infinity and t near 0 is a row where almost nothing is masked
        # paying an enormous coefficient. The reference uses the same guard.
        self.eps = float(eps)
        self._metrics: Dict[str, float] = {}

        # Running unigram distribution over targets, used ONLY as a diagnostic
        # baseline: a masked-diffusion model that has learned nothing still
        # scores the per-position marginal, and that degenerate solution looks
        # like a healthy falling loss. `diffusion_unigram_gap` is how far the
        # model is ahead of it, in nats. At or below zero means nothing has
        # been learned about structure, whatever the loss curve says.
        self.unigram_momentum = float(unigram_momentum)
        if vocab_size:
            self.register_buffer(
                "unigram_counts", torch.ones(int(vocab_size)), persistent=True
            )
        else:
            self.unigram_counts = None

    # ------------------------------------------------------------------
    # corruption
    # ------------------------------------------------------------------

    def corrupt(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Replace a random subset of positions with the mask id.

        Args:
            input_ids: ``(B, L)`` clean ids.
            attention_mask: ``(B, L)`` 1 for real positions. Padding is never
                corrupted and never scored - masking a pad would make the
                1/t weight count positions that carry no information.

        Returns:
            ``(noisy_ids, masked, t)`` - ids with the mask id written in, the
            boolean ``(B, L)`` of what was replaced, and the ``(B, 1)`` level.
        """
        B, L = input_ids.shape
        device = input_ids.device

        # One level per row. Uniform over [eps, 1]: the objective integrates
        # over the whole schedule, so every step should see easy and hard rows.
        t = torch.rand(B, 1, device=device, generator=generator)
        t = t * (1.0 - self.eps) + self.eps

        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        draw = torch.rand(B, L, device=device, generator=generator)
        masked = (draw < t) & valid

        # A row with nothing masked contributes no gradient but still divides
        # by t, so it is pure variance. Force one position on those rows.
        empty = ~masked.any(dim=-1)
        if empty.any():
            # The lowest draw among that row's valid positions is the one the
            # threshold came closest to selecting.
            fallback = draw.masked_fill(~valid, float("inf")).argmin(dim=-1)
            masked[empty, fallback[empty]] = True
            # Rows with no valid position at all (fully padded) stay empty.
            masked &= valid

        noisy = torch.where(masked, torch.full_like(input_ids, self.mask_token_id), input_ids)
        return noisy, masked, t

    # ------------------------------------------------------------------
    # objective
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        masked: Tensor,
        t: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """The MDLM/LLaDA ELBO term, in nats per token.

        ``sum(CE over masked) / (t * L_valid)``, averaged over rows. Since
        ``E[#masked] = t * L_valid`` this has the magnitude of a mean CE over
        masked positions, but it is the unbiased bound rather than a
        convenience - which matters because the number gets reported.
        """
        B, L, V = logits.shape
        if targets.shape != (B, L):
            raise ValueError(
                "diffusion labels must be UNSHIFTED and the same length as the "
                f"input (got labels {tuple(targets.shape)} against logits "
                f"{(B, L)}). A caller that built labels as input_ids[..., 1:] "
                "has to ask the model for `outputs_are_aligned` first."
            )
        ce = F.cross_entropy(
            logits.reshape(B * L, V).float(),
            targets.reshape(B * L),
            reduction="none",
            ignore_index=-100,
        ).view(B, L)

        masked_f = masked.to(ce.dtype)
        per_row_ce = (ce * masked_f).sum(dim=-1)

        valid = (
            attention_mask.to(ce.dtype)
            if attention_mask is not None
            else torch.ones_like(ce)
        )
        n_valid = valid.sum(dim=-1).clamp_min(1.0)
        loss = (per_row_ce / (t.squeeze(-1) * n_valid)).mean()

        self._record(ce, masked, t, targets, n_valid)
        return loss

    @torch.no_grad()
    def _record(
        self,
        ce: Tensor,
        masked: Tensor,
        t: Tensor,
        targets: Tensor,
        n_valid: Tensor,
    ) -> None:
        """Diagnostics. Every one of these answers a question from
        next/text_diffusion.md's failure-mode list."""
        masked_f = masked.to(ce.dtype)
        n_masked = masked_f.sum().clamp_min(1.0)
        mean_ce = float((ce * masked_f).sum() / n_masked)

        metrics = {
            # The plain mean CE over corrupted positions, in nats. Comparable
            # only to itself and to the unigram baseline below - NOT to any
            # autoregressive run's loss, which measures a different quantity.
            "diffusion_ce": mean_ce,
            "diffusion_bits": mean_ce / math.log(2),
            "diffusion_mask_rate": float(t.mean()),
            "diffusion_masked_frac": float(n_masked / n_valid.sum().clamp_min(1.0)),
        }

        # CE split by corruption level. A model that only learns the easy end
        # (few masks, copy the neighbours) shows a healthy total while the
        # hard bucket stays at chance - the single most useful early read.
        tt = t.squeeze(-1)
        per_row_masked = masked_f.sum(dim=-1).clamp_min(1.0)
        per_row_ce = (ce * masked_f).sum(dim=-1) / per_row_masked
        for name, lo, hi in (("low", 0.0, 0.34), ("mid", 0.34, 0.67), ("high", 0.67, 1.01)):
            sel = (tt >= lo) & (tt < hi)
            if sel.any():
                metrics[f"diffusion_ce_{name}"] = float(per_row_ce[sel].mean())

        if self.unigram_counts is not None:
            flat = targets.reshape(-1)
            flat = flat[flat >= 0]
            if flat.numel():
                counts = torch.bincount(flat, minlength=self.unigram_counts.numel()).to(
                    self.unigram_counts.dtype
                )
                self.unigram_counts.mul_(self.unigram_momentum).add_(
                    counts * (1.0 - self.unigram_momentum)
                )
            probs = self.unigram_counts / self.unigram_counts.sum().clamp_min(1e-9)
            logp = probs.clamp_min(1e-9).log()
            unigram_ce = float(-(logp[targets.reshape(-1)] * masked_f.reshape(-1)).sum() / n_masked)
            metrics["diffusion_unigram_ce"] = unigram_ce
            # Positive = the model beats the marginal. At or below zero, the
            # degenerate solution has won regardless of what the loss does.
            metrics["diffusion_unigram_gap"] = unigram_ce - mean_ce

        self._metrics = metrics

    def training_metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        denoise: Callable[[Tensor], Tensor],
        prompt_ids: Optional[Tensor],
        length: int,
        steps: int = 16,
        temperature: float = 0.0,
        device: Optional[torch.device] = None,
        batch_size: int = 1,
    ) -> Tensor:
        """Iterative unmasking, without transformers in the way.

        The same loop ``praxis.diffusion.decoding.unmask_decoding`` runs - it
        delegates here - so a probe and a real generation cannot drift apart.
        ``denoise`` maps ids to logits over the full alphabet.
        """
        from praxis.diffusion.decoding import refine

        if prompt_ids is not None and prompt_ids.numel():
            batch_size = prompt_ids.shape[0]
            device = prompt_ids.device
        return refine(
            denoise,
            prompt_ids,
            length,
            mask_id=self.mask_token_id,
            steps=steps,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
        )

    def extra_repr(self) -> str:
        return f"mask_token_id={self.mask_token_id}, eps={self.eps}"
