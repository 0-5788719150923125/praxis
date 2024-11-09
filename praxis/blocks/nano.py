import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoConfig


class PraxisNano(nn.Module):
    """
    A special kind of block that omits all self-attention mechanisms, in favor
    of dense layers with sine activations. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    def __init__(self, config: AutoConfig):
        super().__init__()
        max_seq_len = config.context_length // 2
        embed_dim = config.num_embeds
        hidden_dim = config.num_dims

        # Define the weight matrices with maximum sequence length
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fft = nn.ParameterDict(
            {
                "w1": nn.Parameter(torch.Tensor(max_seq_len, max_seq_len)),
                "w2": nn.Parameter(torch.Tensor(max_seq_len, max_seq_len)),
            }
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.fft["w1"])
        nn.init.xavier_uniform_(self.fft["w2"])

        # Define the mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # Normalize per row
        row_sums = mask.sum(dim=1, keepdim=True)
        epsilon = 1e-8
        row_sums = row_sums + epsilon  # Add epsilon to avoid division by zero
        mask = mask / row_sums
        self.register_buffer("mask", mask)

        class SineActivation(nn.Module):
            def forward(self, x):
                return torch.sin(x)

        # Feed-forward network with sine activation
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            SineActivation(),
            nn.Linear(embed_dim, hidden_dim),
        )

    def forward(self, x: Tensor, attention_mask: Tensor):
        # x: [batch_size, seq_len, embed_dim]
        B, T, E = x.shape
        if T > self.fft["w1"].size(0):
            raise ValueError(
                f"Sequence length {T} exceeds maximum supported length {self.W1.size(0)}."
            )

        x = self.ln1(x)

        # Get the relevant slices of weights and mask based on the actual sequence length
        W1 = self.fft["w1"][:T, :T] * self.mask[:T, :T]
        W2 = self.fft["w2"][:T, :T] * self.mask[:T, :T]

        # Reshape x for matrix multiplication
        x_fft = x.transpose(1, 2).reshape(-1, T)  # [B * embed_dim, T]

        # Apply the masked and normalized weight matrices
        x_fft = x_fft @ W1  # [B * embed_dim, T]
        x_fft = x_fft @ W2  # [B * embed_dim, T]

        # Reshape back to original dimensions
        x_fft = x_fft.view(B, E, T).transpose(1, 2)  # [B, T, E]

        x = x + x_fft  # Residual connection
        x = self.ln2(x)
        x = x + self.ffw(x)  # Residual connection
        return x
