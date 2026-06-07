"""Energy generative head for CALM.

MLP-based generative head following the reference implementation
(github.com/shaochenze/calm). Maps a conditioning hidden state ``h``
plus uniform noise ``ε`` to a latent proposal ``z_hat``. Trained by
the energy-score loss (section 3.3 of arXiv 2510.27688).

Faithful to the reference on a few points that were biasing training in
earlier versions of this file:
- Conditioning ``y`` is re-injected at *every* ``MLPBlock`` (via concat
  with the LayerNorm'd residual stream), not just at the input.
- Both noise and conditioning paths run through a Linear → LayerNorm
  before any block sees them.
- The final projection is a small (LayerNorm → Linear → SiLU → Linear)
  head whose last Linear is zero-initialised, so the head starts
  predicting near-zero latents and learns to expand from there.
- Noise is uniform [-0.5, 0.5].
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# Streaming-ridge sufficient statistics: EMA decay and the scale-free ridge
# coefficient (lambda = RIDGE_LAMBDA * mean diagonal of A). Fixed and
# model-agnostic per the no-per-experiment-tuning rule.
PRIOR_STATS_DECAY = 0.999
PRIOR_RIDGE_LAMBDA = 1e-3
# Post-freeze re-solve gate (all fixed, model-agnostic): a re-solve is
# triggered when cond_gap has gained GAP_DELTA since the last one, applied as
# a damped blend, and kept only if the energy-loss EMA has not worsened by
# the verify window's end - otherwise W restores. Protects against the copy
# attractor: a solve that buys R² by dragging proposals toward repetition
# won't improve the energy loss on true continuations, and gets rejected.
PRIOR_RESOLVE_GAP_DELTA = 0.5
PRIOR_RESOLVE_BLEND = 0.25
PRIOR_RESOLVE_VERIFY_STEPS = 200  # optimizer steps
PRIOR_LOSS_EMA_DECAY = 0.99  # ~100-step horizon, matches the repo convention
PRIOR_RESOLVE_TOLERANCE = 1.002  # relative slack so EMA noise can't veto
# Harmonic mode: integer frequencies 1..F over the patch period, sin+cos.
PRIOR_HARMONIC_FREQS = 8


class LinearPrior(nn.Module):
    """Closed-form linear predictor of the next latent: ``z ~ W phi(h, t)``.

    Credit where it's due: this exists because of a 16-year-old who spent
    months on Reddit insisting ML could be cracked with a linear-solve
    algorithm. He never published code and his arguments were shaky, but he
    was writing ML equations at 16 and he believed it - and at small scale,
    on the linear part of the problem, he was right. This module is his idea
    grown up: solve what is solvable, learn only the residual.

    Reservoir-style readout: never trained by gradient. (h, z_next) pairs
    accumulate into EMA sufficient statistics ``A = E[phi phi^T]``,
    ``B = E[phi z^T]`` and ``W`` is re-solved by ridge regression - the prior
    is computed, not learned, and drifts only as the data statistics drift.
    The energy head adds ``W phi`` to its proposal and spends its gradient
    budget on the residual the solve can't capture.

    ``mode="harmonic"`` augments features with sin/cos of integer frequencies
    over the patch period, so quasi-periodic latent structure is absorbed as
    Fourier coefficients - also a linear solve.

    All state is persistent buffers (survives checkpoint/resume). ``frozen``
    latches at the end of the solve window; afterwards W is a fixed prior.
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        mode: str = "linear",
        period: int = 256,
    ) -> None:
        super().__init__()
        if mode not in ("linear", "harmonic"):
            raise ValueError(f"LinearPrior mode must be linear|harmonic, got {mode!r}")
        self.mode = mode
        self.period = max(2, int(period))
        self.in_dim = feature_dim
        F_h = 2 * PRIOR_HARMONIC_FREQS if mode == "harmonic" else 0
        D = feature_dim + F_h
        self.register_buffer("A", torch.zeros(D, D), persistent=True)
        self.register_buffer("B", torch.zeros(D, latent_dim), persistent=True)
        self.register_buffer("W", torch.zeros(D, latent_dim), persistent=True)
        self.register_buffer(
            "frozen", torch.zeros((), dtype=torch.bool), persistent=True
        )
        self.register_buffer("seen", torch.zeros(()), persistent=True)
        # Post-freeze re-solve state machine (see update_resolve).
        self.register_buffer("W_prev", torch.zeros(D, latent_dim), persistent=True)
        self.register_buffer(
            "pending", torch.zeros((), dtype=torch.bool), persistent=True
        )
        self.register_buffer("pending_step", torch.zeros(()), persistent=True)
        self.register_buffer(
            "gap_anchor", torch.full((), float("nan")), persistent=True
        )
        self.register_buffer("loss_ema", torch.full((), float("nan")), persistent=True)
        self.register_buffer("ema_at_apply", torch.zeros(()), persistent=True)
        self.register_buffer("resolves_kept", torch.zeros(()), persistent=True)
        self.register_buffer("resolves_rejected", torch.zeros(()), persistent=True)
        # Transient diagnostics (consumed by the encoder's training_metrics).
        self.last_r2 = float("nan")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before the re-solve machinery lack its buffers;
        # seed them with their init values so resume keeps working.
        for name in (
            "W_prev",
            "pending",
            "pending_step",
            "gap_anchor",
            "loss_ema",
            "ema_at_apply",
            "resolves_kept",
            "resolves_rejected",
        ):
            state_dict.setdefault(prefix + name, getattr(self, name).clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def features(self, h: torch.Tensor, t: Optional[torch.Tensor]) -> torch.Tensor:
        """``[..., in_dim]`` -> ``[..., D]``; harmonic mode appends the basis."""
        if self.mode != "harmonic":
            return h
        if t is None:
            # Positions default to the trailing sequence axis indices.
            t = torch.arange(h.shape[-2], device=h.device)
        t = t.to(h.dtype).reshape(*t.shape, 1)
        f = torch.arange(1, PRIOR_HARMONIC_FREQS + 1, device=h.device, dtype=h.dtype)
        ang = 2 * math.pi * t * f / self.period
        basis = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return torch.cat([h, basis.expand(*h.shape[:-1], basis.shape[-1])], dim=-1)

    @torch.no_grad()
    def observe(
        self, h: torch.Tensor, z: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> None:
        """Fold a batch of (conditioning, next-latent) pairs into the stats.

        Keeps accumulating after the freeze: the EMA decay washes out the
        immature-trunk statistics, so a later re-solve (update_resolve) reads
        the current features rather than the freeze-time snapshot.
        """
        phi = self.features(h.detach(), t).reshape(-1, self.A.shape[0]).float()
        zt = z.detach().reshape(-1, self.B.shape[1]).float()
        n = phi.shape[0]
        if n == 0:
            return
        d = PRIOR_STATS_DECAY
        self.A.mul_(d).add_((phi.T @ phi) / n, alpha=1.0 - d)
        self.B.mul_(d).add_((phi.T @ zt) / n, alpha=1.0 - d)
        self.seen.add_(1.0)
        # R^2 of the CURRENT W on this batch: the fraction of next-latent
        # variance the linear solve already explains. The decisive metric -
        # high means CALM's grind was mostly the linear map, ~0 means the
        # backbone doesn't linearize the sequence at all.
        pred = phi @ self.W
        ss_res = (zt - pred).pow(2).sum()
        ss_tot = (zt - zt.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
        self.last_r2 = float(1.0 - ss_res / ss_tot)

    @torch.no_grad()
    def _ridge(self) -> torch.Tensor:
        """Ridge solve ``(A + lambda I) W = B``; scale-free lambda."""
        D = self.A.shape[0]
        lam = PRIOR_RIDGE_LAMBDA * (self.A.diagonal().mean().clamp_min(1e-12))
        eye = torch.eye(D, device=self.A.device, dtype=self.A.dtype)
        return torch.linalg.solve(self.A + lam * eye, self.B)

    @torch.no_grad()
    def solve(self) -> None:
        """Initial solve window: re-solve W in place each step until frozen."""
        if bool(self.frozen.item()) or float(self.seen.item()) <= 0:
            return
        self.W.copy_(self._ridge())

    @torch.no_grad()
    def freeze(self) -> None:
        self.frozen.fill_(True)

    @torch.no_grad()
    def update_resolve(
        self, cond_gap: float, energy_loss: float, opt_step: int
    ) -> None:
        """Milestone-gated, damped, accept-if-not-worse re-solve.

        Called once per training step after the freeze. A re-solve fires when
        cond_gap has gained PRIOR_RESOLVE_GAP_DELTA since the last one (the
        trunk matured enough to be worth re-reading), blends the fresh ridge
        solution in at PRIOR_RESOLVE_BLEND, then watches the energy-loss EMA
        for PRIOR_RESOLVE_VERIFY_STEPS: not-worse keeps the new W, worse
        restores the old one. Endogenous - no per-experiment knobs.
        """
        if not bool(self.frozen.item()):
            return
        if math.isnan(energy_loss):
            return
        if math.isnan(float(self.loss_ema.item())):
            self.loss_ema.fill_(energy_loss)
        else:
            self.loss_ema.mul_(PRIOR_LOSS_EMA_DECAY).add_(
                (1.0 - PRIOR_LOSS_EMA_DECAY) * energy_loss
            )
        if bool(self.pending.item()):
            if opt_step - int(self.pending_step.item()) >= PRIOR_RESOLVE_VERIFY_STEPS:
                worse = (
                    float(self.loss_ema.item())
                    > float(self.ema_at_apply.item()) * PRIOR_RESOLVE_TOLERANCE
                )
                if worse:
                    # REPLACE the buffer, never mutate in place: this runs
                    # after the loss is computed, so the current W tensor is
                    # saved inside the live autograd graph for backward.
                    self.W = self.W_prev.clone()
                    self.resolves_rejected.add_(1.0)
                else:
                    self.resolves_kept.add_(1.0)
                self.pending.fill_(False)
            return
        if math.isnan(cond_gap):
            return
        if math.isnan(float(self.gap_anchor.item())):
            self.gap_anchor.fill_(cond_gap)  # baseline at freeze
            return
        if cond_gap - float(self.gap_anchor.item()) < PRIOR_RESOLVE_GAP_DELTA:
            return
        self.W_prev.copy_(self.W)
        # Out-of-place blend + buffer REPLACEMENT (see above): the old W must
        # survive untouched until the in-flight backward has consumed it.
        self.W = (
            (1.0 - PRIOR_RESOLVE_BLEND) * self.W + PRIOR_RESOLVE_BLEND * self._ridge()
        ).detach()
        self.gap_anchor.fill_(cond_gap)
        self.ema_at_apply.copy_(self.loss_ema)
        self.pending.fill_(True)
        self.pending_step.fill_(float(opt_step))

    def forward(
        self, h: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The prior's contribution: ``W phi(h, t)``. W carries no gradient;
        gradient flows through h into the backbone, which is desirable."""
        phi = self.features(h, t)
        return phi @ self.W.to(phi.dtype)


# Options for the energy head's closed-form prior. "linear" is the default
# wherever the energy head is used; "none" is the paper-pure ablation.
ENERGY_PRIOR_REGISTRY = {
    "none": None,
    "linear": partial(LinearPrior, mode="linear"),
    "harmonic": partial(LinearPrior, mode="harmonic"),
}


class MLPBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True),
        )
        self.down_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
        gate, up = h.chunk(2, dim=-1)
        return x + self.down_proj(F.silu(gate) * up)


class FinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, out_channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(self.in_ln(x))


class EnergyHead(nn.Module):
    """Conditioned implicit generator of continuous latents.

    ``hidden_dim`` is the residual width carrying the noise/cond stream
    through the blocks; ``cond_dim`` / ``noise_dim`` are the input
    widths (typically the LM hidden state width and the noise vector
    width respectively).
    """

    def __init__(
        self,
        cond_dim: int,
        noise_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Optional closed-form linear prior (see LinearPrior): set by the
        # owner via set_prior(); its solved W phi(h) adds to every proposal.
        self.prior: Optional[LinearPrior] = None

        self.cond_embd = nn.Linear(cond_dim, hidden_dim, bias=True)
        self.noise_embd = nn.Linear(noise_dim, hidden_dim, bias=True)
        self.norm_cond = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm_noise = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_layer = FinalLayer(hidden_dim, latent_dim)

        # Zero-init the last projection: the head starts predicting
        # near-zero latents and grows from there. Without this the head
        # emits arbitrarily-scaled outputs from random init and has to
        # claw its way back through the energy score.
        nn.init.zeros_(self.final_layer.linears[-1].weight)
        nn.init.zeros_(self.final_layer.linears[-1].bias)

    def set_prior(self, prior: Optional[LinearPrior]) -> None:
        self.prior = prior

    def forward(
        self,
        h: torch.Tensor,
        noise: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a latent proposal.

        Args:
            h: ``[..., cond_dim]`` conditioning.
            noise: ``[..., noise_dim]`` noise drawn by the caller.
            t: optional patch positions (harmonic prior only).

        Returns:
            ``[..., latent_dim]`` latent sample: the linear prior's solved
            contribution (when set) plus the MLP's learned residual.
        """
        cond = self.norm_cond(self.cond_embd(h))
        x = self.norm_noise(self.noise_embd(noise))
        for block in self.blocks:
            x = block(x, cond)
        out = self.final_layer(x)
        if self.prior is not None:
            out = out + self.prior(h, t)
        return out

    def sample(
        self,
        h: torch.Tensor,
        num_samples: int,
        noise_dtype: torch.dtype = torch.float32,
        noise_scale: float = 1.0,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draw ``num_samples`` latents per conditioning row.

        Noise is uniform on [-0.5, 0.5]; the energy score is a proper
        scoring rule, so the head learns to map this noise to whatever
        target distribution the loss is fed. ``noise_scale`` shrinks the
        noise toward 0 (the conditional-mean prediction) - generation uses
        it as a temperature, so low T concentrates on the head's best guess.

        Returns ``[num_samples, ..., latent_dim]``.
        """
        shape = h.shape[:-1] + (self.noise_dim,)
        expanded = h.unsqueeze(0).expand(num_samples, *h.shape)
        noise = (
            torch.rand(num_samples, *shape, device=h.device, dtype=noise_dtype) - 0.5
        ) * noise_scale
        return self(expanded, noise, t=t)
