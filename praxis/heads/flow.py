"""Flow-matching generative head for CALM.

A drop-in sibling to ``EnergyHead`` (same constructor signature, same
``sample`` / ``forward`` / ``set_prior`` surface) selectable via
``LATENT_HEAD_REGISTRY``. Where the energy head trains an implicit
generator with a high-variance sample-based score, this trains a velocity
field by flow matching - a dense, low-variance regression target at every
noise level. Faithful to the reference (github.com/shaochenze/calm,
modeling_flow.py): a SimpleMLPAdaLN velocity net conditioned on the AR
hidden state and the flow timestep via AdaLN, zero-initialised so the
field is identity at init.

The "noise" the energy head maps from is replaced by the flow's own start
state, which lives in latent space - so ``noise_dim == latent_dim`` here,
and ``forward(h, x0, t)`` integrates the ODE from ``x0`` (a zero start
gives the deterministic best-guess the diagnostic reads).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# ODE integration steps at sampling time. Midpoint halves the count (two net
# calls per step). Fixed, model-agnostic (matches the reference's 20/midpoint).
FLOW_SAMPLE_STEPS = 20
FLOW_SOLVER = "midpoint"


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of a scalar flow timestep -> hidden vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class FlowResBlock(nn.Module):
    """AdaLN-modulated residual MLP block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = self.mlp(modulate(self.in_ln(x), shift, scale))
        return x + gate * h


class FlowFinalLayer(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 2 * channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class FlowMLP(nn.Module):
    """Velocity field v(x_t, t, c). Operates on flattened rows."""

    def __init__(
        self, in_channels: int, model_channels: int, z_channels: int, num_blocks: int
    ) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [FlowResBlock(model_channels) for _ in range(num_blocks)]
        )
        self.final_layer = FlowFinalLayer(model_channels, in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        # Zero the AdaLN + output so the field is identity at init.
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_proj(x)
        y = self.time_embed(t * 1000) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


class FlowHead(nn.Module):
    """Flow-matching latent generator. Sibling of ``EnergyHead``.

    Same constructor kwargs as ``EnergyHead`` for registry interchange;
    ``noise_dim`` is ignored (the flow starts in latent space) and is
    reported as ``latent_dim`` so callers size their start state correctly.
    """

    def __init__(
        self,
        cond_dim: int,
        noise_dim: int,  # accepted for parity; unused (flow starts in latent space)
        latent_dim: int,
        hidden_dim: int,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = latent_dim  # callers build a latent-dim start state
        self.prior = None  # flow has no linear prior; kept for interface parity
        self.net = FlowMLP(
            in_channels=latent_dim,
            model_channels=hidden_dim,
            z_channels=cond_dim,
            num_blocks=num_blocks,
        )

    def set_prior(self, prior) -> None:  # no-op: flow does not use a prior
        self.prior = None

    def flow_loss(
        self,
        target: torch.Tensor,
        cond: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-element flow-matching loss ``||v_pred - (target - x0)||^2``.

        ``target``/``cond`` carry arbitrary leading dims. ``x0``/``t`` may be
        passed so matched and mismatched scorings share the same draw (for a
        clean cond-gap comparison); otherwise they are sampled fresh.
        """
        lead = target.shape[:-1]
        if x0 is None:
            x0 = torch.randn_like(target)
        if t is None:
            t = torch.rand(lead, device=target.device, dtype=target.dtype)
        xt = (1 - t[..., None]) * x0 + t[..., None] * target
        flat = math.prod(lead) if lead else 1
        v_pred = self.net(
            xt.reshape(flat, -1), t.reshape(flat), cond.reshape(flat, -1)
        ).reshape(*lead, -1)
        return (v_pred - (target - x0)).pow(2).mean(dim=-1)

    @torch.no_grad()
    def _integrate(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ODE-integrate the velocity field from start ``x`` toward t=1."""
        lead = cond.shape[:-1]
        flat = math.prod(lead) if lead else 1
        c = cond.reshape(flat, -1)
        x = x.reshape(flat, -1)
        steps = (
            FLOW_SAMPLE_STEPS // 2 if FLOW_SOLVER == "midpoint" else FLOW_SAMPLE_STEPS
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((flat,), step / steps, device=x.device, dtype=x.dtype)
            if FLOW_SOLVER == "midpoint":
                v1 = self.net(x, t, c)
                v_mid = self.net(x + 0.5 * dt * v1, t + 0.5 * dt, c)
                x = x + dt * v_mid
            else:
                x = x + dt * self.net(x, t, c)
        return x.reshape(*lead, -1)

    def forward(
        self, h: torch.Tensor, noise: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Deterministic readout: integrate the ODE from ``noise`` as the start
        state (a zero start gives the conditional best-guess). ``t`` (patch
        position) is accepted for interface parity and unused."""
        return self._integrate(noise, h)

    def sample(
        self,
        h: torch.Tensor,
        num_samples: int,
        noise_dtype: torch.dtype = torch.float32,
        noise_scale: float = 1.0,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draw ``num_samples`` latents per conditioning row by integrating the
        flow from independent Gaussian starts. ``noise_scale`` shrinks the start
        spread toward the conditional mean (a temperature). Returns
        ``[num_samples, ..., latent_dim]``."""
        cond = h.unsqueeze(0).expand(num_samples, *h.shape)
        x0 = (
            torch.randn(
                num_samples,
                *h.shape[:-1],
                self.latent_dim,
                device=h.device,
                dtype=noise_dtype,
            )
            * noise_scale
        )
        return self._integrate(x0, cond)


# Energy and flow share the encoder's head slot; the encoder picks via the
# head_kind kwarg baked into its profile partial.
from praxis.heads.energy import EnergyHead  # noqa: E402

LATENT_HEAD_REGISTRY = {
    "energy": EnergyHead,
    "flow": FlowHead,
}
