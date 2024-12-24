import random

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Static alpha/beta
        self.static_beta = nn.Parameter(torch.ones(rate))
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[self.layer_id % rate, 0] = 1.0
        eye_alpha = torch.eye(rate)
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, eye_alpha], dim=1))

        if self.dynamic:
            # Dynamic offsets
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.tensor([0.01]))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.tensor([0.01]))
            self.layer_norm = nn.RMSNorm(dim, eps=1e-5)

    def connect_width(self, h: torch.Tensor):
        # Flatten (B,L) to (BL) and do one batched matmul to form mix_h
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
        # Combine new layer output with previously merged expansions, then sum
        B, L, D = h_o.shape
        BL = B * L
        h_o_2d = h_o.reshape(BL, D)
        beta_2d = beta.reshape(BL, self.rate)
        beta_3d = beta_2d.unsqueeze(2)
        h_o_3d = h_o_2d.unsqueeze(1)
        part_new_3d = torch.bmm(beta_3d, h_o_3d)
        part_new = part_new_3d.reshape(B, L, self.rate, D)
        part_old = mix_h[:, :, 1:, :]
        final_exp = part_new + part_old
        return final_exp.sum(dim=2)

    def _compute_alpha_beta(self, h: torch.Tensor):
        B, L, D = h.shape
        if self.dynamic:
            h_norm = self.layer_norm(h)
            alpha_offset = torch.tanh(h_norm @ self.dynamic_alpha_fn)
            alpha_offset = alpha_offset * self.dynamic_alpha_scale
            alpha_offset_4d = alpha_offset.unsqueeze(2)
            static_alpha_4d = self.static_alpha.view(1, 1, self.rate, self.rate + 1)
            static_alpha_4d = static_alpha_4d.expand(B, L, self.rate, self.rate + 1)
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
            alpha_4d = self.static_alpha.view(1, 1, self.rate, self.rate + 1)
            alpha = alpha_4d.expand(B, L, self.rate, self.rate + 1)
            beta_3d = self.static_beta.view(1, 1, self.rate)
            beta = beta_3d.expand(B, L, self.rate)
        return alpha, beta
