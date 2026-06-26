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
    hidden states at successive loop boundaries, halting once the latent
    has stopped moving. The floor it must drop below is ``convergence_ratio``
    times a *global* scale - a slow EMA of the per-pass peak KL across passes
    - not this pass's own peak.

    That distinction is the whole design. A trained recurrent block acts like
    a contraction toward a fixed point, so within one pass the per-position KL
    decays geometrically at a rate that is a property of the learned operator,
    roughly *independent of the input* (kl_r ~= kl_1 * lambda^(r-1)).
    Normalizing by this pass's own peak cancels kl_1 - the only input-dependent
    term - so ``kl_r < ratio * peak`` reduces to ``lambda^(r-1) < ratio``, a
    constant: every input exits at the same depth and the halting distribution
    collapses to a single route. Anchoring instead to a fixed global scale
    keeps the absolute magnitude of the early movement, so harder inputs (which
    move more) cross the floor later and the exit depth spreads into a curve.
    The scale is learned endogenously (the EMA), so there is still no
    per-experiment threshold to tune.

    Two details keep the signal honest across models and training: the
    hidden state is standardized (shift/scale invariant) before the
    softmax, so the measure does not drift as residual norms grow (this is
    what made an absolute floor flaky before, and the EMA tracks any slow
    residual drift on top of it); and KL is averaged per position, so it does
    not scale with sequence length. No LM head required - works uniformly for
    head- and encoder-based models.

    Reference: Geiping et al., "Scaling up Test-Time Compute with Latent
    Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171)
    """

    def __init__(
        self,
        config: ConfigType,
        convergence_ratio: float = 0.1,
        sigma: float = 0.5,
        r_bar: Optional[float] = None,
        peak_ema_decay: float = 0.95,
    ) -> None:
        super().__init__(config)
        self.convergence_ratio = convergence_ratio
        self.sigma = sigma
        self.peak_ema_decay = peak_ema_decay
        self.max_loops = self.depth // self.num_layers
        self.r_bar = (
            float(r_bar) if r_bar is not None else max(1.0, (self.max_loops - 1) / 2)
        )
        self._prev_log_probs: Optional[Tensor] = None
        # Global scale the inference floor is measured against: a slow EMA of
        # each pass's peak KL. Persists across passes (unlike the per-pass
        # state below) so the floor is a fixed absolute level within any one
        # pass, which is what lets the exit depth vary with input difficulty.
        self._peak_ema: Optional[float] = None
        self._pass_peak: float = 0.0  # largest KL this pass; folded into the EMA next pass
        self._pass_anchor: Optional[float] = None  # EMA snapshot frozen for this pass

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
        # Fold the just-finished eval pass's peak KL into the slow global EMA,
        # then freeze that EMA as this pass's floor anchor. There is no
        # pass-end callback, so we settle the previous pass here at the start of
        # the next. _pass_peak is only ever > 0 after an eval pass (training
        # never computes KL), so training passes leave the EMA untouched.
        if self._pass_peak > 0:
            if self._peak_ema is None:
                self._peak_ema = self._pass_peak
            else:
                d = self.peak_ema_decay
                self._peak_ema = d * self._peak_ema + (1.0 - d) * self._pass_peak
        self._pass_peak = 0.0
        self._pass_anchor = self._peak_ema

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

    def _to_log_probs(self, hidden_states: Tensor) -> Tensor:
        """Standardize over the hidden dim, then log-softmax.

        Subtracting the mean and dividing by the std makes the measure
        invariant to the residual stream's shift and scale, so KL reflects
        a change in the latent's *shape*, not its growing magnitude.
        """
        mean = hidden_states.mean(dim=-1, keepdim=True)
        std = hidden_states.std(dim=-1, keepdim=True)
        normed = (hidden_states - mean) / (std + 1e-5)
        return F.log_softmax(normed, dim=-1)

    @torch.no_grad()
    def seed(self, hidden_states: Tensor) -> None:
        if self.training:
            return
        self._prev_log_probs = self._to_log_probs(hidden_states)

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

        current_log_probs = self._to_log_probs(hidden_states)

        if self._prev_log_probs is None:
            self._prev_log_probs = current_log_probs
            return False

        # Per-position mean KL(current || prev): summed over the hidden dim,
        # averaged over batch and sequence so it does not scale with length.
        kl = (
            F.kl_div(
                self._prev_log_probs,
                current_log_probs,
                reduction="none",
                log_target=True,
            )
            .sum(dim=-1)
            .mean()
            .item()
        )

        self._prev_log_probs = current_log_probs

        self._eval_checks += 1
        self._eval_kl_sum += kl
        self._pass_peak = max(self._pass_peak, kl)

        # Absolute convergence on a self-calibrating scale: halt once the
        # per-position movement drops below ``convergence_ratio`` of the model's
        # typical peak movement (the global EMA), frozen for this pass. Unlike a
        # this-pass-peak anchor, the global scale keeps the input-dependent
        # magnitude, so harder inputs cross the floor later (see class docstring
        # for the contraction-mapping argument).
        anchor = self._pass_anchor
        if anchor is None:
            # First eval pass, EMA not warmed yet: fall back to this pass's
            # running peak so the very first pass still behaves sanely while the
            # global scale calibrates. From the next pass on, the EMA takes over.
            anchor = self._pass_peak
        halted = anchor > 0 and kl < self.convergence_ratio * anchor
        if halted:
            self._eval_halts += 1
            self._record_eval_r(loop_r)
        return halted

    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "halting/max_loops": self.max_loops,
            "halting/r_bar": self.r_bar,
        }
        if self._peak_ema is not None:
            metrics["halting/peak_ema"] = self._peak_ema
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
