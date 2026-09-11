"""Recurrent-depth halting over a loop that re-reads its input at every step.

Geiping et al. (arXiv:2502.05171) and the reference Huginn code: the loop state
starts from truncated-normal noise, every step's block input is an adapter over
the concatenated state and embedded input, and at inference each position exits
on its own once successive steps stop moving it. Without the re-read, the paper
notes, "the iterative process would not be stable": a block that sees the input
only at the first step cannot depend on it anywhere else.

Three translations, each forced by a structural difference from Huginn:

  * Huginn's blocks normalize the residual SUM, so the state it re-injects is
    unit-scale. Blocks here normalize only the branch, so both halves of the
    adapter's input are RMS-normalized instead, which makes the noise scale
    irrelevant: normalized noise is unit-scale whatever it was drawn at.
  * Huginn exits on the KL between successive OUTPUT distributions. A
    byte-latent trunk has none - its output feeds a local decoder - so this reads
    the parent's standardized-state KL, per position.
  * Huginn's threshold is an absolute constant. Here it is ``convergence_ratio``
    times a slow EMA of each pass's mean FIRST-REFINEMENT KL (step 2 against
    step 1). The step out of noise is kept out of the scale: it measures a jump
    from a random state, and a floor anchored on it is cleared by the first
    refinement at every position.

Inference freezes each position at its exit - later steps leave it unchanged
while later positions still attend to it - and the pass ends once every position
has exited or the budget runs out. The Halting Distribution counts POSITIONS.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.halting.kl import KLDivergenceHalting

# Huginn's truncation bounds on the initial state, in standard deviations.
NOISE_BOUND: float = 3.0
# Inference draws the initial state from a fixed seed, so validation is
# deterministic while still starting from the distribution training saw.
EVAL_NOISE_SEED: int = 0


class ReinjectedKLHalting(KLDivergenceHalting):
    """KL halting over a noise-initialized loop that re-reads its input."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        hidden = config.hidden_size
        eps = getattr(config, "epsilon", 1e-6)
        self.state_norm = nn.RMSNorm(hidden, eps=eps)
        self.input_norm = nn.RMSNorm(hidden, eps=eps)
        # Huginn's adapter: Linear(2h -> h) over [state; input], no bias.
        self.adapter = nn.Linear(2 * hidden, hidden, bias=False)
        self._input: Optional[Tensor] = None
        self._frozen: Optional[Tensor] = None  # [B, T] positions that exited
        self._frozen_state: Optional[Tensor] = None  # their states at exit

    def extra_repr(self) -> str:
        return f"prior={self.prior}, max_loops={self.max_loops}"

    def seed(self, hidden_states: Tensor) -> None:
        # Unlike the parent's KL baseline, the re-read input is needed in
        # training too.
        self._input = hidden_states
        self._prev_log_probs = None
        self._frozen = None
        self._frozen_state = None

    def initial_state(self, hidden_states: Tensor) -> Tensor:
        generator = None
        if not self.training:
            generator = torch.Generator(device=hidden_states.device)
            generator.manual_seed(EVAL_NOISE_SEED)
        noise = torch.randn(
            hidden_states.shape,
            generator=generator,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return noise.clamp_(-NOISE_BOUND, NOISE_BOUND)

    def inject(self, hidden_states: Tensor, current_depth: int) -> Tensor:
        if self._input is None or self._input.shape != hidden_states.shape:
            raise ValueError(
                "kl_log_reinject re-reads the decoder input at every step, so the "
                "sequence cannot change length inside the loop (got state "
                f"{tuple(hidden_states.shape)} against input "
                f"{None if self._input is None else tuple(self._input.shape)})."
            )
        state = self.state_norm(hidden_states)
        read = self.input_norm(self._input.to(hidden_states.dtype))
        return self.adapter(torch.cat([state, read], dim=-1))

    def settle(self, hidden_states: Tensor) -> Tensor:
        if self._frozen is None:
            return hidden_states
        return torch.where(
            self._frozen.unsqueeze(-1), self._frozen_state, hidden_states
        )

    def release(self) -> None:
        # The re-read input carries the encoder's graph; held past the loop it
        # would keep that graph alive until the next forward.
        self._input = None
        self._frozen = None
        self._frozen_state = None

    @torch.no_grad()
    def check(self, hidden_states: Tensor, current_depth: int) -> bool:
        if self.training or not self._is_loop_boundary(current_depth):
            return False

        loop_r = (current_depth + 1) // self.num_layers
        last = current_depth >= self.depth - 1
        log_probs = self._to_log_probs(hidden_states)

        if self._prev_log_probs is None:
            # Step 1 is the move out of noise; there is no refinement to read.
            self._prev_log_probs = log_probs
            self._frozen = torch.zeros(
                hidden_states.shape[:-1], dtype=torch.bool, device=hidden_states.device
            )
            self._frozen_state = hidden_states.clone()
            if last:
                self._record_exits(self._frozen.logical_not(), loop_r)
                return True
            return False

        # Per-position KL(current || previous), summed over the hidden dim.
        kl = F.kl_div(
            self._prev_log_probs, log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
        self._prev_log_probs = log_probs

        active = self._frozen.logical_not()
        if not bool(active.any()):
            return True
        kl_active = kl[active]
        if self._pass_peak == 0.0:
            # The first refinement of this pass; folded into the EMA next pass.
            self._pass_peak = float(kl_active.mean())
        anchor = self._pass_anchor if self._pass_anchor is not None else self._pass_peak

        if last:
            exits = active
        else:
            exits = active & (kl < self.convergence_ratio * anchor)
        if self.record_metrics:
            self._eval_checks += int(active.sum())
            self._eval_kl_sum += float(kl_active.sum())
            if not last:
                self._eval_halts += int(exits.sum())
        self._record_exits(exits, loop_r)

        self._frozen_state = torch.where(
            exits.unsqueeze(-1), hidden_states, self._frozen_state
        )
        self._frozen = self._frozen | exits
        return bool(self._frozen.all())

    def _record_exits(self, exits: Tensor, loop_r: int) -> None:
        count = int(exits.sum())
        if self.record_metrics and count:
            self._eval_hist[loop_r] = self._eval_hist.get(loop_r, 0) + count
