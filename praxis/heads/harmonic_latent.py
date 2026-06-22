"""Harmonic latent generative head for CALM.

A drop-in sibling to ``FlowHead`` (same ``flow_loss`` / ``forward`` /
``sample`` / ``set_prior`` surface) selectable via ``LATENT_HEAD_REGISTRY``.
It trains by flow matching like ``FlowHead``, but the flow lives in a small
harmonic *coefficient* space: each next-latent is synthesized as a
superposition of low-frequency harmonics over the latent-index axis,
``z = c @ Q^T``, where ``Q`` is a fixed orthonormal harmonic basis and the
flow predicts the ``coeff_dim``-vector ``c``.

The bet (research/main.tex's log-scaling conjecture: interference-based
composition reaches expressivity logarithmically where generic approximation
scales linearly): if the codec's latent is smooth/low-frequency, the head
needs only ``coeff_dim < latent_dim`` numbers to describe it, so its effective
output dimension - and thus its variance - shrinks. That is the same
"buy structural bias to spend less capacity" move that K=4 made on the codec,
applied to the head. ``num_freqs`` is the knob: fewer = stronger smoothness
prior + more compression (the real bet); ``2*num_freqs == latent_dim`` is the
lossless-reparameterization baseline (~equivalent to ``FlowHead``).

Reuses ``FlowHead``'s velocity net and integrator. The projection trick keeps
the encoder loss path identical: ``flow_loss`` accepts latent-space targets and
a latent-space ``x0`` (what the encoder builds for the flow head) and projects
both into coefficient space internally, so the encoder routes ``head_kind
"harmonic"`` through the unchanged ``_register_flow_loss``.

Future direction (not done here): a 2D harmonic field over (patch position,
latent dim) - like ``HarmonicField`` - would model the whole latent sequence's
temporal-frequency structure, but breaks the per-position sample() interface.
"""

import math
from typing import Optional

import torch
from torch import nn

from praxis.heads.flow import FLOW_SAMPLE_STEPS, FLOW_SOLVER, FlowMLP


def _harmonic_basis(latent_dim: int, num_freqs: int) -> torch.Tensor:
    """Orthonormal basis ``[latent_dim, coeff_dim]`` spanning the lowest
    ``num_freqs`` harmonics (DC + cos/sin) over the latent-index axis.

    QR-orthonormalized so projection and synthesis are exact transposes;
    ``coeff_dim = min(latent_dim, 1 + 2*num_freqs)`` after the reduced QR.
    """
    idx = torch.arange(latent_dim, dtype=torch.float32)
    cols = [torch.ones(latent_dim)]  # DC (the latent mean)
    for k in range(1, num_freqs + 1):
        ang = math.pi * (idx + 0.5) * k / latent_dim
        cols.append(torch.cos(ang))
        cols.append(torch.sin(ang))
    feats = torch.stack(cols[: min(len(cols), latent_dim)], dim=1)
    q, _ = torch.linalg.qr(feats)  # [latent_dim, coeff_dim], orthonormal columns
    return q


class HarmonicLatentHead(nn.Module):
    """Flow-matching latent generator over a harmonic coefficient space.

    Same constructor kwargs as ``FlowHead``/``EnergyHead`` for registry
    interchange. ``noise_dim`` is reported as ``latent_dim`` (the caller builds
    a latent-width start state, projected internally), matching ``FlowHead``.
    """

    def __init__(
        self,
        cond_dim: int,
        noise_dim: int,  # accepted for parity; unused (start state is latent-width)
        latent_dim: int,
        hidden_dim: int,
        num_blocks: int = 4,
        num_freqs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = latent_dim  # caller builds a latent-dim start state
        self.prior = None  # no linear prior; kept for interface parity

        # Default: keep the lowest ~quarter of frequencies (a 2:1-ish coefficient
        # compression - the smoothness bet). Set num_freqs so 2*num_freqs ==
        # latent_dim for the lossless-reparameterization baseline.
        if num_freqs is None:
            num_freqs = max(1, latent_dim // 4)
        basis = _harmonic_basis(latent_dim, num_freqs)
        self.register_buffer("basis", basis, persistent=False)  # [L, coeff_dim]
        self.coeff_dim = basis.shape[1]
        self.net = FlowMLP(
            in_channels=self.coeff_dim,
            model_channels=hidden_dim,
            z_channels=cond_dim,
            num_blocks=num_blocks,
        )

    def set_prior(self, prior) -> None:  # no-op: harmonic head uses no prior
        self.prior = None

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Latent ``[..., L]`` -> coefficients ``[..., coeff_dim]``."""
        return z @ self.basis.to(z.dtype)

    def synthesize(self, c: torch.Tensor) -> torch.Tensor:
        """Coefficients ``[..., coeff_dim]`` -> latent ``[..., L]`` (smooth)."""
        return c @ self.basis.to(c.dtype).T

    def flow_loss(
        self,
        target: torch.Tensor,
        cond: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-element flow-matching loss in coefficient space.

        ``target`` and ``x0`` are latent-space (what ``_register_flow_loss``
        builds); both are projected to coefficients here, so the encoder path is
        shared with ``FlowHead`` unchanged. The orthonormal projection of a
        standard-normal latent ``x0`` is itself standard-normal in coefficient
        space, so the flow's start distribution is unchanged.
        """
        lead = target.shape[:-1]
        if x0 is None:
            x0 = torch.randn_like(target)
        if t is None:
            t = torch.rand(lead, device=target.device, dtype=target.dtype)
        c_target = self.project(target)
        c0 = self.project(x0)
        ct = (1 - t[..., None]) * c0 + t[..., None] * c_target
        flat = math.prod(lead) if lead else 1
        v_pred = self.net(
            ct.reshape(flat, -1), t.reshape(flat), cond.reshape(flat, -1)
        ).reshape(*lead, -1)
        return (v_pred - (c_target - c0)).pow(2).mean(dim=-1)

    @torch.no_grad()
    def _integrate(self, c: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ODE-integrate the coefficient-space velocity field from ``c``."""
        lead = cond.shape[:-1]
        flat = math.prod(lead) if lead else 1
        ctx = cond.reshape(flat, -1)
        c = c.reshape(flat, -1)
        steps = (
            FLOW_SAMPLE_STEPS // 2 if FLOW_SOLVER == "midpoint" else FLOW_SAMPLE_STEPS
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((flat,), step / steps, device=c.device, dtype=c.dtype)
            if FLOW_SOLVER == "midpoint":
                v1 = self.net(c, t, ctx)
                v_mid = self.net(c + 0.5 * dt * v1, t + 0.5 * dt, ctx)
                c = c + dt * v_mid
            else:
                c = c + dt * self.net(c, t, ctx)
        return c.reshape(*lead, -1)

    def forward(
        self, h: torch.Tensor, noise: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Deterministic readout: project the latent-width ``noise`` start into
        coefficient space, integrate, and synthesize. A zero start gives the
        conditional best-guess. ``t`` is accepted for parity and unused."""
        c = self._integrate(self.project(noise), h)
        return self.synthesize(c)

    def sample(
        self,
        h: torch.Tensor,
        num_samples: int,
        noise_dtype: torch.dtype = torch.float32,
        noise_scale: float = 1.0,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draw ``num_samples`` latents per conditioning row: integrate the
        coefficient flow from independent Gaussian starts and synthesize.
        ``noise_scale`` shrinks the start spread toward the mean (a temperature).
        Returns ``[num_samples, ..., latent_dim]``."""
        cond = h.unsqueeze(0).expand(num_samples, *h.shape)
        c0 = (
            torch.randn(
                num_samples,
                *h.shape[:-1],
                self.coeff_dim,
                device=h.device,
                dtype=noise_dtype,
            )
            * noise_scale
        )
        return self.synthesize(self._integrate(c0, cond))
