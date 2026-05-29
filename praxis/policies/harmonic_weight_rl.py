"""REINFORCE controller that proposes harmonic edits to weight chunks.

A deliberate, minimal RL exercise (see calm-c). The controller observes a
small summary of training dynamics (state), proposes a sinusoidal
modulation of one weight row (action), and is rewarded by the loss
improvement that follows (reward). Trained by vanilla policy gradient
(REINFORCE) with an EMA baseline - the same estimator as
``policies/reinforce.py``, but acting on parameters rather than activations.

This is on purpose the hard case for RL: the reward is delayed, noisy, and
non-stationary (the model's own optimizer is moving the loss too). The
value of the exercise is watching those dynamics directly.
``HarmonicWeightRLCallback`` drives this from the training loop.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def build_gate_mask(
    selector: str,
    n: int,
    threshold: float,
    omega: float,
    phi: float,
    seed: int,
    device,
) -> torch.Tensor:
    """Boolean mask over a length-``n`` weight row: True = replace with anchor.

    Pluggable so the structure of the selection is the ablation (see
    next/hash_gated_anchor.md). All selectors are deterministic given their
    inputs; ``uniform_hash`` is the controllable baseline that isolates whether
    *structure* in the gate helps versus random selection at the same density.
    """
    if selector == "sinusoidal":
        idx = torch.arange(n, device=device, dtype=torch.float32)
        return torch.sin(omega * idx + phi) > threshold
    if selector == "uniform_hash":
        # threshold in (-1, 1) -> density in (0, 1); seeded PRNG marks a
        # density fraction. CPU generator keeps it reproducible across devices.
        density = (1.0 - threshold) / 2.0
        gen = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
        r = torch.rand(n, generator=gen)
        return (r < density).to(device)
    # precision_hash: derive the mask from where float precision breaks down
    # as a scalar is driven around the representable grid (see the doc). Not
    # built yet - slots in here as one more branch once the baselines are
    # validated.
    raise ValueError(f"unknown rl_selector: {selector!r}")


class HarmonicWeightPolicy(nn.Module):
    """Tiny Gaussian policy over harmonic edit parameters (alpha, omega, phi)."""

    # Driven by a training callback, not the forward pass: modeling.py must
    # NOT instantiate this as self.policy.
    is_weight_controller = True

    STATE_DIM = 3  # [loss, loss_slope, target-row grad-norm], each normalized
    ACTION_DIM = 3  # raw Gaussian dims -> (alpha, omega, phi)

    def __init__(self, config) -> None:
        super().__init__()
        h = int(getattr(config, "rl_hidden", 32))
        # alpha is the modulation depth; keep it small so edits stay "minimally
        # invasive". omega is the spatial frequency across the row.
        self.alpha_scale = float(getattr(config, "rl_alpha_scale", 0.1))
        self.omega_max = float(getattr(config, "rl_omega_max", math.pi))
        self.entropy_coef = float(getattr(config, "rl_entropy_coef", 1e-2))
        self.baseline_decay = float(getattr(config, "rl_baseline_decay", 0.99))
        lr = float(getattr(config, "rl_lr", 3e-3))

        self.net = nn.Sequential(
            nn.Linear(self.STATE_DIM, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(h, self.ACTION_DIM)
        # State-independent log-std: a learnable exploration width.
        self.log_std = nn.Parameter(torch.zeros(self.ACTION_DIM))
        # Start near zero mean so early edits are tiny and exploration drives.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        # EMA baseline b; advantage = reward - b. Buffer so it checkpoints.
        self.register_buffer("baseline", torch.zeros(()))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _dist(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_head(self.net(state))
        std = self.log_std.clamp(-5.0, 2.0).exp()
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
        """Sample a raw action; return (raw_action, mapped (alpha, omega, phi))."""
        raw = self._dist(state).sample()
        alpha, omega, phi = self.map_action(raw)
        return raw, (float(alpha), float(omega), float(phi))

    def map_action(self, raw: torch.Tensor):
        """Squash raw Gaussian samples to bounded harmonic params.

        We deliberately skip the tanh log-prob Jacobian correction (the policy
        is Gaussian over ``raw``; the environment applies the squash). That is
        standard in intro REINFORCE - it biases the gradient slightly but keeps
        the estimator legible.
        """
        alpha = self.alpha_scale * torch.tanh(raw[..., 0])
        omega = self.omega_max * torch.sigmoid(raw[..., 1])
        phi = math.pi * torch.tanh(raw[..., 2])
        return alpha, omega, phi

    def map_gate_action(self, raw: torch.Tensor):
        """Map the same raw action to anchor-gate params: (threshold, omega, phi).

        ``threshold`` in (-1, 1) sets the gate density (how much of the row is
        pulled back to the frozen anchor); ``omega``/``phi`` shape the mask for
        the structured selectors. Same 3-D action and policy as harmonic mode -
        only the interpretation differs.
        """
        threshold = torch.tanh(raw[..., 0])
        omega = self.omega_max * torch.sigmoid(raw[..., 1])
        phi = math.pi * torch.tanh(raw[..., 2])
        return threshold, omega, phi

    def update(
        self, state: torch.Tensor, raw_action: torch.Tensor, reward: float
    ) -> Dict[str, float]:
        """One REINFORCE step on a single (state, action, reward) sample."""
        reward_t = torch.as_tensor(float(reward))
        advantage = reward_t - self.baseline
        dist = self._dist(state)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        # Maximize E[log_prob * advantage] + entropy bonus -> minimize negative.
        loss = -(log_prob * advantage.detach()) - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.baseline.mul_(self.baseline_decay).add_(
                (1.0 - self.baseline_decay) * reward_t
            )
        return {
            "rl_reward": float(reward_t),
            "rl_baseline": float(self.baseline),
            "rl_advantage": float(advantage),
            "rl_policy_loss": float(loss.detach()),
            "rl_entropy": float(entropy.detach()),
            "rl_log_std_mean": float(self.log_std.detach().mean()),
        }
