import torch
import torch.nn as nn
import torch.nn.functional as F


class PraxisAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal = config.causal
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.query = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.key = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.out = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.register_buffer("m", self._get_alibi_slope(self.num_heads))

    def _get_relative_positions(self, seq_len: int) -> torch.tensor:
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return x - y

    def _get_alibi_slope(self, num_heads):
        x = (2**8) ** (1 / num_heads)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )

        # Apply ALiBi bias
        bias = (self.m * self._get_relative_positions(seq_len).to(x.device)).unsqueeze(
            0
        )
        scores = scores - bias

        # Apply the causal mask
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.hidden_size
        )
        attn_output = self.out(attn_output)
        return attn_output
