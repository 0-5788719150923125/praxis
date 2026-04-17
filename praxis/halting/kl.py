import math
from typing import Any, Dict, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.halting.base import BaseHalting

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class KLDivergenceHalting(BaseHalting):
    """Randomized depth during training, KL-based halting at inference.

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
        r_bar: Optional[float] = None,
    ) -> None:
        super().__init__(config)
        self.threshold = threshold
        self.sigma = sigma
        self.max_loops = self.depth // self.num_layers
        # Poisson rate target. The sampler returns Poisson(e^tau) + 1, so
        # the mean of an uncensored draw is r_bar + 1. Default picks the
        # middle of [1, max_loops] to keep both tails off the clamps.
        self.r_bar = (
            float(r_bar) if r_bar is not None else max(1.0, (self.max_loops - 1) / 2)
        )
        self._prev_log_probs: Optional[Tensor] = None

        # Running stats for dashboards.
        self._train_calls = 0
        self._train_loops_sum = 0
        self._last_train_loops: Optional[int] = None
        self._eval_checks = 0
        self._eval_halts = 0
        self._eval_kl_sum = 0.0

        # Per-r histograms of loops used. Training = sampled distribution;
        # eval = the r at which halting actually fired (or max_loops when
        # the pass ran to full depth). Buckets are 1-indexed.
        self._train_hist: Dict[int, int] = {i: 0 for i in range(1, self.max_loops + 1)}
        self._eval_hist: Dict[int, int] = {i: 0 for i in range(1, self.max_loops + 1)}
        self._inflight_halt_r: Optional[int] = None

    def _sample_loop_count(self) -> int:
        """Sample from log-normal Poisson distribution (paper eqs. 1-2).

        tau ~ N(log(r_bar) - 0.5 * sigma^2, sigma)
        r ~ Poisson(e^tau) + 1, clamped to [1, max_loops]
        """
        tau = torch.distributions.Normal(
            math.log(self.r_bar) - 0.5 * self.sigma**2,
            self.sigma,
        ).sample()
        r = torch.distributions.Poisson(tau.exp()).sample().int().item() + 1
        return max(1, min(r, self.max_loops))

    def get_depth(self) -> int:
        self._prev_log_probs = None
        self._inflight_halt_r = None
        if self.training:
            loops = self._sample_loop_count()
            self._train_calls += 1
            self._train_loops_sum += loops
            self._last_train_loops = loops
            self._train_hist[loops] = self._train_hist.get(loops, 0) + 1
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

    def _record_eval_r(self, r: int) -> None:
        if self._inflight_halt_r is not None:
            return
        self._inflight_halt_r = r
        self._eval_hist[r] = self._eval_hist.get(r, 0) + 1

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

        loop_r = (current_depth + 1) // self.num_layers

        logits = head(hidden_states)
        current_log_probs = F.log_softmax(logits, dim=-1)

        # First boundary without a baseline - store and continue
        if self._prev_log_probs is None:
            self._prev_log_probs = current_log_probs
            return False

        # Last step - no point halting, but record that this pass went full depth
        if current_depth >= self.depth - 1:
            self._record_eval_r(self.max_loops)
            return False

        kl = F.kl_div(
            self._prev_log_probs,
            current_log_probs.exp(),
            reduction="batchmean",
            log_target=False,
        ).item()

        self._prev_log_probs = current_log_probs

        self._eval_checks += 1
        self._eval_kl_sum += kl
        halted = kl < self.threshold
        if halted:
            self._eval_halts += 1
            self._record_eval_r(loop_r)
        return halted

    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "halting/max_loops": self.max_loops,
            "halting/r_bar": self.r_bar,
        }
        if self._train_calls > 0:
            metrics["halting/train_calls"] = self._train_calls
            metrics["halting/mean_loops"] = self._train_loops_sum / self._train_calls
            if self._last_train_loops is not None:
                metrics["halting/last_loops"] = self._last_train_loops
            for r, count in self._train_hist.items():
                metrics[f"halting/train_r_{r}"] = count
        if self._eval_checks > 0 or any(self._eval_hist.values()):
            metrics["halting/eval_checks"] = self._eval_checks
            metrics["halting/eval_halts"] = self._eval_halts
            if self._eval_checks > 0:
                metrics["halting/halt_rate"] = self._eval_halts / self._eval_checks
                metrics["halting/eval_mean_kl"] = self._eval_kl_sum / self._eval_checks
            for r, count in self._eval_hist.items():
                metrics[f"halting/eval_r_{r}"] = count
        return metrics
