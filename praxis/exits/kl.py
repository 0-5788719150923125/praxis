from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.exits.base import BaseExit

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class KLDivergenceExit(BaseExit):
    """Exit when the output distribution stabilizes between recurrence loops.

    Monitors KL-divergence between the model's output distribution at
    successive loop boundaries (every num_layers steps). Rather than
    comparing against a static threshold, this tracks the rate of change
    between consecutive KL measurements. When the divergence shrinks
    sufficiently relative to the previous boundary's divergence, the
    model has converged and remaining loops are skipped.

    This requires no training changes and no learned parameters - the
    convergence behavior emerges naturally from recurrent architectures.

    Reference: Geiping et al., "Scaling up Test-Time Compute with Latent
    Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171)
    """

    def __init__(self, config: ConfigType, convergence_ratio: float = 0.1) -> None:
        super().__init__(config)
        self.convergence_ratio = convergence_ratio
        self._prev_log_probs: Optional[Tensor] = None
        self._prev_kl: Optional[float] = None

    def reset(self) -> None:
        self._prev_log_probs = None
        self._prev_kl = None

    @torch.no_grad()
    def seed(
        self,
        hidden_states: Tensor,
        head: Optional[nn.Module] = None,
    ) -> None:
        if head is None:
            return
        logits = head(hidden_states)
        self._prev_log_probs = F.log_softmax(logits, dim=-1)

    def _is_loop_boundary(self, current_depth: int) -> bool:
        """True after completing a full pass through all blocks."""
        return (current_depth + 1) % self.num_layers == 0

    @torch.no_grad()
    def check(
        self,
        hidden_states: Tensor,
        current_depth: int,
        head: Optional[nn.Module] = None,
    ) -> bool:
        if not self._is_loop_boundary(current_depth):
            return False

        # Need the head to compute output distributions
        if head is None:
            return False

        logits = head(hidden_states)
        current_log_probs = F.log_softmax(logits, dim=-1)

        # First loop boundary - store baseline, can't compute KL yet
        if self._prev_log_probs is None:
            self._prev_log_probs = current_log_probs
            return False

        # Last step - no point exiting, we'd run it anyway
        if current_depth >= self.depth - 1:
            return False

        # KL(current || previous) averaged over batch and sequence
        kl = F.kl_div(
            self._prev_log_probs,
            current_log_probs.exp(),
            reduction="batchmean",
            log_target=False,
        ).item()

        self._prev_log_probs = current_log_probs

        # Second boundary - we have a KL but no previous KL to compare against
        if self._prev_kl is None:
            self._prev_kl = kl
            return False

        # Exit when KL has dropped to a small fraction of the previous KL,
        # meaning the model's outputs are barely changing anymore
        converged = self._prev_kl > 0 and kl / self._prev_kl < self.convergence_ratio
        self._prev_kl = kl

        return converged
