"""SSOG attention: a query-steered Gaussian field over relative lag, no Q, no K.

Port of Sum of Separable Gaussians (Pisoni, https://github.com/4rtemi5/ssog,
vision only) to a causal 1D sequence. Content never scores content. Each head
owns ``NUM_ATOMS`` Gaussian atoms over the lag ``d = q_idx - kv_idx`` - three
numbers each: centre ``mu`` (how far back to look), width ``sigma`` and mixture
weight ``lambda`` - and the attention logit from a query to a key is the log of
that mixture at their lag, sharpened by one learned temperature:

    logit(q, k) = logsumexp_r( log lambda_r + log N(q - k; mu_r, sigma_r) ) / tau

softmaxed over the causal keys and applied to V. Only V is projected; the QK
projections and their d^2 parameters are gone.

Steering ("lookat"): a zero-initialised probe on the QUERY token predicts bounded
residuals on mu, sigma and lambda behind cold softplus(-8) gates, so the field
starts frozen and the model opens the content taps itself. The residual on mu
lives in softplus's raw space so a steered atom can never point into the
future - under a causal mask an atom with ``mu < 0`` collapses onto lag 0 and
its gradient dies, so non-negativity is enforced rather than hoped for.

Differences from the reference on purpose:

* 1D and causal. The 2D separability trick has nothing to factorise here, so
  the cost is that of ordinary attention: FlexAttention with a ``score_mod``
  that evaluates the mixture from per-query captured tensors (each atom its
  own tensor - indexing ``mu[b, h, q, r]`` with a python int does not trace),
  a materialised ``[B, H, T, T]`` path on CPU.
* No ghost token. Softmax1's always-visible zero-logit ghost would take
  roughly half of a Gaussian field's mass; a field wants an explicit learned
  null atom for that, which is deliberately left out of this first version.
* Mixture-then-softmax (the reference's stated maths) rather than the code's
  per-atom softmax then lambda-mix; the reference reports the two within noise
  and the former is one kernel call.

Positional encoding is irrelevant (there is no Q/K to rotate); ``pos_type``
reports ``"ssog"``. ``num_queries`` is corrected to 1 by ``patch_config`` since
no query heads are built. See ``next/`` for the language-modeling expectation:
position-addressed, never content-addressed - a hybrid arm, not a replacement.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_ATOMS: int = 4  # Gaussian atoms per head (the reference's sweet spot)
SIGMA_FLOOR: float = 0.25  # minimum atom width, in tokens
MAX_OFFSET: float = 8.0  # bound on per-query mu travel, in tokens (raw space)
COLD_GATE_INIT: float = -8.0  # softplus(-8) ~ 3e-4: steering starts closed
SIGMA_RAW_INIT: float = -0.5  # softplus(-0.5) + floor ~ 0.72 tokens
TEMPERATURE_RAW_INIT: float = -1.0  # softplus(-1) + 0.5 ~ 0.81, slightly sharp
QK_DUMMY_DIM: int = 16  # flex needs Q/K tensors; theirs are zeros
_EPS: float = 1e-4
_LOG_2PI: float = math.log(2.0 * math.pi)


def _inv_softplus(y: Tensor) -> Tensor:
    return torch.log(torch.expm1(y))


def _log_kernel(d: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """log N(d; mu, sigma^2)."""
    return -0.5 * _LOG_2PI - torch.log(sigma) - (d - mu) ** 2 / (2.0 * sigma**2)


class SSOGAttention(nn.Module):
    """Sum-of-Gaussians attention field over causal lag (see module docstring)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.patch_config(config)
        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = getattr(config, "head_size", None) or hidden_size // self.num_heads
        self.num_atoms = NUM_ATOMS
        self.causal = config.causal
        self.window_size = getattr(config, "window_size", None)
        self.dropout_p = config.dropout
        self.pos_type = "ssog"

        H, R = self.num_heads, self.num_atoms
        self.value = nn.Linear(hidden_size, H * self.head_dim, bias=False)
        self.output = nn.Linear(H * self.head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(self.dropout_p)

        # The field. Atoms start staggered one token apart (lag r + 0.5, jittered)
        # so the heads break symmetry from step 0 instead of all staring at the
        # same lag; mu is softplus-parametrised so it is never negative.
        init_mu = torch.arange(R, dtype=torch.float32).add(0.5).repeat(H, 1)
        init_mu = init_mu + 0.1 * torch.randn(H, R)
        self.raw_mu = nn.Parameter(_inv_softplus(init_mu.clamp_min(0.05)))
        self.raw_sigma = nn.Parameter(torch.full((H, R), SIGMA_RAW_INIT))
        self.log_lambda = nn.Parameter(torch.zeros(H, R))
        self.raw_temperature = nn.Parameter(torch.tensor(TEMPERATURE_RAW_INIT))

        # Steering: one zero-init probe per token -> (mu, sigma, lambda) residuals
        # for every atom of every head, each family behind its own cold gate.
        self.steer = nn.Linear(hidden_size, H * R * 3, bias=True)
        nn.init.zeros_(self.steer.weight)
        nn.init.zeros_(self.steer.bias)
        self.raw_gate = nn.Parameter(torch.full((3,), COLD_GATE_INIT))

        self.flex_attention = None
        self.create_block_mask = None
        self.and_masks = None
        try:
            from torch.nn.attention.flex_attention import (
                and_masks,
                create_block_mask,
                flex_attention,
            )

            # Compiled on purpose: eager flex_attention cannot backprop through
            # captured score_mod tensors and V together (a vmap limitation in
            # its dense backward), and the steering probes ARE captured tensors
            # that need gradient. Under an outer torch.compile dynamo simply
            # inlines this, so compiled runs pay nothing extra.
            self.flex_attention = torch.compile(flex_attention)
            self.create_block_mask = create_block_mask
            self.and_masks = and_masks
        except ImportError:
            pass
        self.block_mask_cache = {}

    @classmethod
    def patch_config(cls, config) -> None:
        """No query heads are built, so ``num_queries`` must read 1. Idempotent."""
        if getattr(config, "num_queries", 1) != 1:
            config.num_queries = 1

    # ------------------------------------------------------------------ field
    def _field(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-query atom parameters, all ``[B, H, T, R]`` float32, plus tau."""
        B, T, _ = x.shape
        H, R = self.num_heads, self.num_atoms
        steer = self.steer(x).float().view(B, T, H, R, 3).permute(0, 2, 1, 3, 4)
        gate = F.softplus(self.raw_gate.float())  # (3,)
        raw_mu = self.raw_mu.float()[None, :, None, :]
        mu = F.softplus(raw_mu + gate[0] * MAX_OFFSET * torch.tanh(steer[..., 0]))
        sigma0 = F.softplus(self.raw_sigma.float()) + _EPS + SIGMA_FLOOR
        sigma = sigma0[None, :, None, :] * torch.exp(gate[1] * torch.tanh(steer[..., 1]))
        loglam = torch.log_softmax(
            self.log_lambda.float()[None, :, None, :] + gate[2] * torch.tanh(steer[..., 2]),
            dim=-1,
        )
        tau = F.softplus(self.raw_temperature.float()) + 0.5
        return mu, sigma, loglam, tau

    # -------------------------------------------------------------- flex path
    def _mask_mod(self):
        window = self.window_size

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        if window is None:
            return causal

        def within(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= window

        return self.and_masks(causal, within)

    def _block_mask(self, T: int, device: torch.device):
        key = (T, str(device))
        if key not in self.block_mask_cache:
            self.block_mask_cache[key] = self.create_block_mask(
                self._mask_mod(), B=None, H=None, Q_LEN=T, KV_LEN=T, device=device
            )
        return self.block_mask_cache[key]

    def _flex(self, v: Tensor, mu, sigma, loglam, tau) -> Tensor:
        B, H, T, _ = v.shape
        # One captured tensor per atom: the closure indexes them by (b, h, q_idx).
        mus = [mu[..., r].contiguous() for r in range(self.num_atoms)]
        sigs = [sigma[..., r].contiguous() for r in range(self.num_atoms)]
        lams = [loglam[..., r].contiguous() for r in range(self.num_atoms)]
        # Captured as a full [B, H, T] tensor, not a 0-dim scalar: inductor's
        # flex backward cannot allocate a grad buffer for a scalar capture.
        inv_tau = (1.0 / tau).expand(B, H, T).contiguous()

        def score_mod(score, b, h, q_idx, kv_idx):
            d = (q_idx - kv_idx).to(torch.float32)
            acc = None
            for m, s, l in zip(mus, sigs, lams):
                term = l[b, h, q_idx] + _log_kernel(d, m[b, h, q_idx], s[b, h, q_idx])
                acc = term if acc is None else torch.logaddexp(acc, term)
            return score + acc * inv_tau[b, h, q_idx]

        q_dummy = v.new_zeros(B, H, T, QK_DUMMY_DIM)
        k_dummy = v.new_zeros(B, H, T, QK_DUMMY_DIM)
        block_mask = self._block_mask(T, v.device) if self.causal else None
        return self.flex_attention(
            q_dummy, k_dummy, v, score_mod=score_mod, block_mask=block_mask
        )

    # ------------------------------------------------------- materialised path
    def _materialised(self, v: Tensor, mu, sigma, loglam, tau) -> Tensor:
        B, H, T, _ = v.shape
        pos = torch.arange(T, device=v.device, dtype=torch.float32)
        d = (pos[:, None] - pos[None, :])[None, None]  # [1, 1, T, T] = q - k
        logits = None
        for r in range(self.num_atoms):
            term = loglam[..., r, None] + _log_kernel(
                d, mu[..., r, None], sigma[..., r, None]
            )  # [B, H, T, T]
            logits = term if logits is None else torch.logaddexp(logits, term)
        logits = logits / tau
        if self.causal:
            allowed = d >= 0
            if self.window_size is not None:
                allowed = allowed & (d <= self.window_size)
            logits = logits.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(logits, dim=-1).to(v.dtype)
        return weights @ v

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        B, T, _ = inputs.shape
        v = self.value(inputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        mu, sigma, loglam, tau = self._field(inputs)
        if self.flex_attention is not None and inputs.device.type != "cpu":
            out = self._flex(v, mu, sigma, loglam, tau)
        else:
            out = self._materialised(v, mu, sigma, loglam, tau)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.dropout(self.output(out)), past_key_values, 0.0

    # ---------------------------------------------------------------- metrics
    def training_metrics(self) -> dict:
        """How open the steering taps are and where the field is looking."""
        gate = F.softplus(self.raw_gate.detach().float())
        out = {
            "ssog_gate_mu": gate[0].item(),
            "ssog_gate_sigma": gate[1].item(),
            "ssog_gate_lambda": gate[2].item(),
            "ssog_temperature": (F.softplus(self.raw_temperature.detach().float()) + 0.5).item(),
            "ssog_mu_mean": F.softplus(self.raw_mu.detach().float()).mean().item(),
            "ssog_sigma_mean": (
                F.softplus(self.raw_sigma.detach().float()) + _EPS + SIGMA_FLOOR
            )
            .mean()
            .item(),
        }
        return out

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, head_dim={self.head_dim}, atoms={self.num_atoms}, "
            f"causal={self.causal}, window={self.window_size}"
        )
