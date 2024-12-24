import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConnection(nn.Module):
    def connect_width(self, h: torch.Tensor):
        return h, None

    def connect_depth(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor):
        return mix_h + h_o


class HyperConnection(nn.Module):
    """
    This module implements static hyper-connections, which are a replacement to
    residual connections.
    https://arxiv.org/abs/2409.19606
    """

    def __init__(
        self, dim: int, rate: int = 2, layer_id: int = None, dynamic: bool = False
    ):
        super().__init__()
        if layer_id is None:
            layer_id = random.randint(0, rate - 1)
        self.layer_id = layer_id
        self.rate = rate
        self.dynamic = dynamic

        self.static_beta = nn.Parameter(torch.ones(rate))
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[self.layer_id % rate, 0] = 1.0
        eye_alpha = torch.eye(rate)
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, eye_alpha], dim=1))

        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.tensor([0.01]))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.tensor([0.01]))
            self.layer_norm = nn.RMSNorm(dim, eps=1e-5)

    def connect_width(self, h: torch.Tensor):
        """
        h: shape (B,L,D) -> returns
          mix_h: shape (B,L, rate+1, D)
          beta:  shape (B,L, rate)
        The first column in mix_h is the 'merger' column, the rest are parked expansions.
        """
        B, L, D = h.shape
        BL = B * L
        alpha, beta = self._compute_alpha_beta(h)
        alpha_2d = alpha.reshape(BL, self.rate, self.rate + 1)
        alpha_bmm = alpha_2d.transpose(1, 2)
        h_2d = h.reshape(BL, D)
        h_3d = h_2d.unsqueeze(1).expand(BL, self.rate, D)
        mix_h_2d = torch.bmm(alpha_bmm, h_3d)
        mix_h = mix_h_2d.reshape(B, L, self.rate + 1, D)
        return mix_h, beta

    def connect_depth(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor):
        """
        mix_h: (B,L, rate+1, D)
        h_o:   (B,L,D)
        beta:  (B,L, rate)
        -> returns (B,L, rate+1, D)
        We merge h_o into columns 1..rate, keep col 0 as the "merger" column.
        """
        B, L, D = h_o.shape
        BL = B * L
        h_o_2d = h_o.reshape(BL, D)
        beta_2d = beta.reshape(BL, self.rate)
        beta_3d = beta_2d.unsqueeze(2)  # (BL, rate, 1)
        h_o_3d = h_o_2d.unsqueeze(1)  # (BL, 1,   D)
        part_new_3d = torch.bmm(beta_3d, h_o_3d)  # (BL, rate, D)
        part_new = part_new_3d.reshape(B, L, self.rate, D)

        # columns 1..rate of mix_h, shape => (B,L, rate, D)
        parked = mix_h[:, :, 1:, :]
        updated = parked + part_new

        # Reinsert column 0 unchanged
        # final => (B,L, rate+1, D)
        col0 = mix_h[:, :, 0, :].unsqueeze(2)
        final = torch.cat([col0, updated], dim=2)
        return final

    def _compute_alpha_beta(self, h: torch.Tensor):
        B, L, D = h.shape
        if self.dynamic:
            h_norm = self.layer_norm(h)
            alpha_offset = (
                torch.tanh(h_norm @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
            )
            alpha_offset_4d = alpha_offset.unsqueeze(2)
            static_alpha_4d = self.static_alpha.view(
                1, 1, self.rate, self.rate + 1
            ).expand(B, L, self.rate, self.rate + 1)
            alpha = alpha_offset_4d + static_alpha_4d

            beta_offset = torch.tanh(
                h_norm @ self.dynamic_beta_fn.unsqueeze(-1)
            ).squeeze(-1)
            beta_offset = beta_offset * self.dynamic_beta_scale
            beta_offset_3d = beta_offset.unsqueeze(-1)
            static_beta_3d = self.static_beta.view(1, 1, self.rate).expand(
                B, L, self.rate
            )
            beta = beta_offset_3d + static_beta_3d
        else:
            alpha_4d = self.static_alpha.view(1, 1, self.rate, self.rate + 1).expand(
                B, L, self.rate, self.rate + 1
            )
            beta_3d = self.static_beta.view(1, 1, self.rate).expand(B, L, self.rate)
            alpha, beta = alpha_4d, beta_3d
        return alpha, beta
