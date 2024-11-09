import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoConfig
from praxis.activations import ACT2FN
from typing import Optional


class PraxisNano(nn.Module):
    """
    A special kind of block that omits all self-attention mechanisms, in favor
    of dense layers with sine activations. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    def __init__(self, config: AutoConfig):
        super().__init__()
        hidden_dim = config.num_dims
        embed_dim = config.num_embeds
        self.base_seq_len = 256
        self.stride = 128

        # Core weight matrices at fixed size
        self.fft = nn.ParameterDict(
            {
                "w1": nn.Parameter(torch.Tensor(self.base_seq_len, self.base_seq_len)),
                "w2": nn.Parameter(torch.Tensor(self.base_seq_len, self.base_seq_len)),
            }
        )

        # Initialize weights with triangular structure
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fft["w1"])
            nn.init.xavier_uniform_(self.fft["w2"])
            self.fft["w1"].copy_(torch.tril(self.fft["w1"]))
            self.fft["w2"].copy_(torch.tril(self.fft["w2"]))

        # Create fixed mask for the base sequence length
        mask = torch.tril(torch.ones(self.base_seq_len, self.base_seq_len))
        row_sums = mask.sum(dim=1, keepdim=True)
        self.register_buffer("base_mask", mask / row_sums)

        # Register gradient hooks to maintain triangular structure
        self.fft["w1"].register_hook(lambda grad: grad * self.base_mask)
        self.fft["w2"].register_hook(lambda grad: grad * self.base_mask)

        # Layer norms and FFN
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            ACT2FN["sin"],
            nn.Linear(embed_dim, hidden_dim),
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, T, E = x.shape
        x = self.ln1(x)

        if T <= self.base_seq_len:
            # For shorter sequences, use standard processing
            window = x
            if T < self.base_seq_len:
                window = F.pad(x, (0, 0, 0, self.base_seq_len - T))

            x_fft = window.transpose(1, 2).reshape(-1, self.base_seq_len)
            x_fft = x_fft @ self.fft["w1"]
            x_fft = x_fft @ self.fft["w2"]
            x_fft = x_fft.view(B, E, self.base_seq_len).transpose(1, 2)
            x_fft = x_fft[:, :T, :]  # Trim padding if any

        else:
            # For longer sequences, use strided windows
            x_fft = torch.zeros_like(x)
            count = torch.zeros(B, T, 1, device=x.device)

            for start_idx in range(0, T, self.stride):
                window_output = self.process_window(x, start_idx)
                end_idx = min(start_idx + self.base_seq_len, T)

                # Add window contribution
                x_fft[:, start_idx:end_idx, :] += window_output[
                    :, : (end_idx - start_idx), :
                ]
                count[:, start_idx:end_idx, :] += 1

            # Average overlapping regions
            x_fft = x_fft / count.clamp(min=1)

        # Apply residual connections and FFN
        x = x + x_fft
        x = self.ln2(x)
        x = x + self.ffw(x)

        return x

    def process_window(self, x: Tensor, start_idx: int) -> Tensor:
        """Process a single window while maintaining causality."""
        # Extract window
        end_idx = min(start_idx + self.base_seq_len, x.size(1))
        effective_len = end_idx - start_idx

        # If we need padding, add it
        window = x[:, start_idx:end_idx, :]
        if effective_len < self.base_seq_len:
            padding = self.base_seq_len - effective_len
            window = F.pad(window, (0, 0, 0, padding))

        B, T, E = window.shape

        # Apply FFT transformation with fixed weights
        x_fft = window.transpose(1, 2).reshape(-1, T)
        x_fft = x_fft @ self.fft["w1"]
        x_fft = x_fft @ self.fft["w2"]
        x_fft = x_fft.view(B, E, T).transpose(1, 2)

        # Return only the valid part
        return x_fft[:, :effective_len, :]


# class PraxisNano(nn.Module):
#     """
#     A special kind of block that omits all self-attention mechanisms, in favor
#     of dense layers with sine activations. Inspired by NanoFFT:
#     https://github.com/timurgepard/nanoFFT
#     """

#     def __init__(self, config: AutoConfig):
#         super().__init__()
#         max_seq_len = config.context_length // 2
#         embed_dim = config.num_embeds
#         hidden_dim = config.num_dims

#         # Define the weight matrices with maximum sequence length
#         self.ln1 = nn.LayerNorm(hidden_dim)
#         self.fft = nn.ParameterDict(
#             {
#                 "w1": nn.Parameter(torch.Tensor(max_seq_len, max_seq_len)),
#                 "w2": nn.Parameter(torch.Tensor(max_seq_len, max_seq_len)),
#             }
#         )

#         # Initialize weights
#         nn.init.xavier_uniform_(self.fft["w1"])
#         nn.init.xavier_uniform_(self.fft["w2"])

#         # Define the mask
#         mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
#         # Normalize per row
#         row_sums = mask.sum(dim=1, keepdim=True)
#         mask = mask / row_sums
#         self.register_buffer("mask", mask)

#         class SineActivation(nn.Module):
#             def forward(self, x):
#                 return torch.sin(x)

#         # Feed-forward network with sine activation
#         self.ln2 = nn.LayerNorm(hidden_dim)
#         self.ffw = nn.Sequential(
#             nn.Linear(hidden_dim, embed_dim),
#             SineActivation(),
#             # ACT2FN["sinlu"],
#             nn.Linear(embed_dim, hidden_dim),
#         )

#     def forward(self, x: Tensor, attention_mask: Tensor):
#         # x: [batch_size, seq_len, embed_dim]
#         B, T, E = x.shape
#         if T > self.fft["w1"].size(0):
#             raise ValueError(
#                 f"Sequence length {T} exceeds maximum supported length {self.W1.size(0)}."
#             )

#         x = self.ln1(x)

#         # Get the relevant slices of weights and mask based on the actual sequence length
#         W1 = self.fft["w1"][:T, :T] * self.mask[:T, :T]
#         W2 = self.fft["w2"][:T, :T] * self.mask[:T, :T]

#         # Reshape x for matrix multiplication
#         x_fft = x.transpose(1, 2).reshape(-1, T)  # [B * embed_dim, T]

#         # Apply the masked and normalized weight matrices
#         x_fft = x_fft @ W1  # [B * embed_dim, T]
#         x_fft = x_fft @ W2  # [B * embed_dim, T]

#         # Reshape back to original dimensions
#         x_fft = x_fft.view(B, E, T).transpose(1, 2)  # [B, T, E]

#         x = x + x_fft  # Residual connection
#         x = self.ln2(x)
#         x = x + self.ffw(x)  # Residual connection
#         return x
