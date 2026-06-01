import math
from typing import Any, Dict, Optional, TypeVar

import torch
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

    Inference: runs up to full depth but monitors KL-divergence between
    hidden states at successive loop boundaries. When the divergence
    drops below a threshold, the latent representation has converged and
    remaining loops are skipped. KL is computed directly on the hidden
    states (softmax over the hidden dim) - no LM head required, so this
    works uniformly for head-based and encoder-based models.

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
        self.r_bar = (
            float(r_bar) if r_bar is not None else max(1.0, (self.max_loops - 1) / 2)
        )
        self._prev_log_probs: Optional[Tensor] = None

        self._train_calls = 0
        self._train_loops_sum = 0
        self._last_train_loops: Optional[int] = None
        self._eval_checks = 0
        self._eval_halts = 0
        self._eval_kl_sum = 0.0

        self._train_hist: Dict[int, int] = {i: 0 for i in range(1, self.max_loops + 1)}
        self._eval_hist: Dict[int, int] = {i: 0 for i in range(1, self.max_loops + 1)}
        self._inflight_halt_r: Optional[int] = None

    def _sample_loop_count(self) -> int:
        """Sample from log-normal Poisson distribution (paper eqs. 1-2).

        tau ~ N(log(r_bar) - 0.5 * sigma^2, sigma)
        r ~ Poisson(e^tau) + 1, truncated to [1, max_loops] via
        rejection (clamping would pile tail mass onto max_loops).
        """
        mu = math.log(self.r_bar) - 0.5 * self.sigma**2
        for _ in range(32):
            tau = torch.distributions.Normal(mu, self.sigma).sample()
            r = torch.distributions.Poisson(tau.exp()).sample().int().item() + 1
            if r <= self.max_loops:
                return r
        return self.max_loops

    def get_depth(self) -> int:
        self._prev_log_probs = None
        self._inflight_halt_r = None
        if self.training:
            loops = self._sample_loop_count()
            # Skip recording while an encoder is in codec preflight: the decoder
            # still loops (so the model trains end to end), but its loop count is
            # not a meaningful early-exit signal yet, and folding it in pollutes
            # the Halting Distribution with pre-decoder-training noise.
            if self.record_metrics:
                self._train_calls += 1
                self._train_loops_sum += loops
                self._last_train_loops = loops
                self._train_hist[loops] = self._train_hist.get(loops, 0) + 1
            return loops * self.num_layers
        return self.depth

    @torch.no_grad()
    def seed(self, hidden_states: Tensor) -> None:
        if self.training:
            return
        self._prev_log_probs = F.log_softmax(hidden_states, dim=-1)

    def _is_loop_boundary(self, current_depth: int) -> bool:
        return (current_depth + 1) % self.num_layers == 0

    def _record_eval_r(self, r: int) -> None:
        if self._inflight_halt_r is not None:
            return
        self._inflight_halt_r = r
        self._eval_hist[r] = self._eval_hist.get(r, 0) + 1

    @torch.no_grad()
    def check(self, hidden_states: Tensor, current_depth: int) -> bool:
        if self.training:
            return False

        if not self._is_loop_boundary(current_depth):
            return False

        loop_r = (current_depth + 1) // self.num_layers

        if current_depth >= self.depth - 1:
            self._record_eval_r(self.max_loops)
            return False

        current_log_probs = F.log_softmax(hidden_states, dim=-1)

        if self._prev_log_probs is None:
            self._prev_log_probs = current_log_probs
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
