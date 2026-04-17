import math
from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.exits.base import BaseExit

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class KLDivergenceExit(BaseExit):
    """Randomized depth during training, KL-based early exit at inference.

    Training: each forward pass gets a random number of recurrence loops
    sampled from a log-normal Poisson distribution. This forces the model
    to front-load useful computation, since it never knows how many loops
    it will receive.

    Inference: runs full depth but monitors KL-divergence between output
    distributions at successive loop boundaries. When the divergence
    drops below a threshold, the model has converged and remaining loops
    are skipped.

    Reference: Geiping et al., "Scaling up Test-Time Compute with Latent
    Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171)
    """

    def __init__(
        self,
        config: ConfigType,
        threshold: float = 5e-4,
        sigma: float = 0.5,
    ) -> None:
        super().__init__(config)
        self.threshold = threshold
        self.sigma = sigma
        self.max_loops = self.depth // self.num_layers
        self._prev_log_probs: Optional[Tensor] = None

    def _sample_loop_count(self) -> int:
        """Sample from log-normal Poisson distribution (paper eqs. 1-2).

        tau ~ N(log(r_bar) - 0.5 * sigma^2, sigma)
        r ~ Poisson(e^tau) + 1
        """
        r_bar = self.max_loops
        tau = torch.distributions.Normal(
            math.log(r_bar) - 0.5 * self.sigma**2,
            self.sigma,
        ).sample()
        r = torch.distributions.Poisson(tau.exp()).sample().int().item() + 1
        return max(1, min(r, self.max_loops))

    def get_depth(self) -> int:
        self._prev_log_probs = None
        if self.training:
            loops = self._sample_loop_count()
            return loops * self.num_layers
        return self.depth

    @torch.no_grad()
    def seed(
        self,
        hidden_states: Tensor,
        head: Optional[nn.Module] = None,
    ) -> None:
        if self.training or head is None:
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
        if self.training:
            return False

        if not self._is_loop_boundary(current_depth):
            return False

        if head is None:
            return False

        logits = head(hidden_states)
        current_log_probs = F.log_softmax(logits, dim=-1)

        # First boundary without a baseline - store and continue
        if self._prev_log_probs is None:
            self._prev_log_probs = current_log_probs
            return False

        # Last step - no point exiting
        if current_depth >= self.depth - 1:
            return False

        kl = F.kl_div(
            self._prev_log_probs,
            current_log_probs.exp(),
            reduction="batchmean",
            log_target=False,
        ).item()

        self._prev_log_probs = current_log_probs

        return kl < self.threshold
