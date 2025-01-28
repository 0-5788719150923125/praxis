import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConnection(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def connect_width(self, h: torch.Tensor):
        return h, None

    def connect_depth(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor):
        return mix_h + h_o

    def format_state(self, h: torch.Tensor):
        return h[..., 0, :] if h.dim() == 4 else h


class HyperConnection(ResidualConnection):
    """
    This module implements static hyper-connections, which are a replacement
    to residual connections.
    https://arxiv.org/abs/2409.19606
    """

    def __init__(
        self, dim: int, rate: int = 2, current_depth: int = None, dynamic: bool = True
    ):
        super().__init__()
        if current_depth is None:
            current_depth = random.randint(0, rate - 1)

        self.current_depth = current_depth
        self.rate = rate
        self.dynamic = dynamic

        # alpha/beta for static
        self.static_beta = nn.Parameter(torch.ones(rate))  # shape [rate]
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[self.current_depth % rate, 0] = 1.0
        eye_alpha = torch.eye(rate)
        # shape => (rate, rate+1)
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, eye_alpha], dim=1))

        # If dynamic, add extra transforms
        if self.dynamic:
            # shape => (dim, rate+1) => for alpha offsets
            self.alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.alpha_scale = nn.Parameter(torch.tensor([0.01]))
            # shape => (dim,) => for beta offsets
            self.beta_fn = nn.Parameter(torch.zeros(dim))
            self.beta_scale = nn.Parameter(torch.tensor([0.01]))
            self.norm = nn.RMSNorm(dim, eps=1e-5)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def connect_width(self, h: torch.Tensor):
        B, L, D = h.shape
        alpha, beta = self._compute_alpha_beta(h)  # alpha => (B,L,rate, rate+1)

        # Flatten (B,L) => BL
        alpha = alpha.view(-1, self.rate, self.rate + 1)  # (BL,rate, rate+1)

        # We'll treat h_2d as (BL, rate, D) by repeating the second dimension
        h_3d = h.view(-1, 1, D).expand(-1, self.rate, D)  # => (BL,rate,D)
        mix_h_2d = torch.bmm(alpha.transpose(1, 2), h_3d)  # => (BL, (rate+1), D)
        mix_h = mix_h_2d.view(B, L, self.rate + 1, D)
        return mix_h, beta

    def connect_depth(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor):
        B, L, D = h_o.shape

        # We'll do an outer product => shape (BL, rate, D)
        part_new_3d = torch.bmm(
            beta.reshape(-1, self.rate, 1), h_o.view(-1, 1, D)
        )  # => (BL, rate, D)
        part_new = part_new_3d.view(B, L, self.rate, D)

        # columns 1..rate => shape (B,L, rate, D)
        parked = mix_h[:, :, 1:, :]
        updated = parked + part_new

        col0 = mix_h[:, :, 0, :].unsqueeze(2)  # keep the merger col
        final = torch.cat([col0, updated], dim=2)  # => (B,L, rate+1, D)
        return final

    def _compute_alpha_beta(self, h: torch.Tensor):
        B, L, D = h.shape
        if self.dynamic:
            # dynamic alpha offsets
            h_norm = self.norm(h)
            alpha_offset = torch.tanh(h_norm @ self.alpha_fn) * self.alpha_scale
            alpha = alpha_offset.unsqueeze(2) + self.static_alpha.expand(
                B, L, self.rate, self.rate + 1
            )
            # similarly for beta
            beta_offset = torch.tanh(h_norm @ self.beta_fn) * self.beta_scale
            beta = beta_offset.unsqueeze(-1) + self.static_beta.expand(B, L, self.rate)
        else:
            alpha = self.static_alpha.expand(B, L, self.rate, self.rate + 1)
            beta = self.static_beta.expand(B, L, self.rate)

        return alpha, beta
