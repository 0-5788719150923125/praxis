from math import ceil, floor, sqrt
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module, ModuleList


def exists(v: Any) -> bool:
    return v is not None


def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d


class ProductKeyAttention(Module):
    """
    Basically stolen from here:
    https://github.com/lucidrains/PEER-pytorch/blob/main/PEER_pytorch/PK.py
    """

    def __init__(
        self,
        config: "AutoConfig" = None,
        dim: Optional[int] = None,
        causal: Optional[bool] = None,
        heads: Optional[int] = None,
        num_key_values: Optional[int] = None,
        key_value_pk_topk: int = 8,
        product_keys: int = 2,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.causal = causal or config.causal
        self.heads = heads or config.num_heads
        self.dim = dim or config.hidden_size
        self.num_key_values = num_key_values or divide_and_round_to_square(
            self.dim * 4, 4
        )
        dropout = dropout or getattr(config, "dropout", 0)

        # queries projection
        self.to_queries = nn.Linear(self.dim, self.dim * self.heads, bias=False)

        # keys and values selected using product-key
        self.keys = nn.EmbeddingBag(
            self.num_key_values * self.heads, self.dim, mode="sum"
        )
        self.values = nn.EmbeddingBag(
            self.num_key_values * self.heads, self.dim, mode="sum"
        )

        assert sqrt(
            self.num_key_values
        ).is_integer(), "`num_key_values` needs to be a square"
        assert (self.dim % 2) == 0, "feature dimension should be divisible by 2"

        self.to_kv_pk_indices = PK(
            dim=self.dim,
            num_keys=int(sqrt(self.num_key_values)),
            final_topk=key_value_pk_topk,
            product_keys=product_keys,
            heads=self.heads,
            dim_key=self.dim // 16,
        )

        self.dropout = nn.Dropout(dropout)

        # output projection
        self.to_out = nn.Linear(self.dim * self.heads, self.dim, bias=False)

    def forward(
        self, 
        inputs: Tensor, 
        attention_mask: Optional[Tensor] = None, 
        past_key_values: Optional[Any] = None, 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[Tensor, Optional[Any], float]:
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # queries
        q = self.to_queries(inputs)
        q = q.view(batch_size, seq_len, self.heads, -1)
        q = q.permute(0, 2, 1, 3)

        q = q * (q.shape[-1] ** -0.5)

        # keys and values
        kv_scores, indices = self.to_kv_pk_indices(inputs, softmax_scores=True)

        # add head offsets to indices
        offsets = torch.arange(self.heads, device=device) * self.num_key_values
        indices = indices.unsqueeze(2) + offsets.view(1, 1, -1, 1)

        # flatten batch and sequence dimensions for embedding lookup
        flat_indices = indices.view(-1, indices.size(-1))
        flat_scores = kv_scores.view(-1, kv_scores.size(-1))

        # get keys and values
        k = self.keys(flat_indices, per_sample_weights=flat_scores)
        v = self.values(flat_indices, per_sample_weights=flat_scores)

        # reshape back to [batch, seq_len, heads, dim]
        k = k.view(batch_size, seq_len, self.heads, -1)
        v = v.view(batch_size, seq_len, self.heads, -1)

        # move heads to dim 1: [batch, heads, seq_len, dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # attention
        sim = torch.matmul(q, k.transpose(-2, -1))

        if self.causal:
            # assert not exists(attention_mask)
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        elif exists(attention_mask):
            mask = attention_mask.unsqueeze(1)
            sim = sim.masked_fill(~mask.unsqueeze(2), -torch.finfo(sim.dtype).max)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # aggregate
        out = torch.matmul(attn, v)

        # reshape back to [batch, seq_len, heads * dim]
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, -1)

        aux_loss = 0.0
        layer_kv = None
        return self.to_out(out), layer_kv, aux_loss


class PK(Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int = 8,
        dim_key: Optional[int] = None,
        num_keys: int = 1_000,
        product_keys: int = 2,
        product_key_topk: Optional[int] = None,
        final_topk: int = 16,
    ) -> None:
        super().__init__()
        assert (dim % 2) == 0
        dim_key = default(dim_key, dim // 2)

        self.dim_key = dim_key
        self.heads = heads
        self.product_keys = product_keys

        # Query projection
        self.to_queries = nn.Linear(dim, dim_key * product_keys * heads, bias=False)

        self.num_keys = num_keys
        self.product_keys = product_keys

        # Initialize keys
        self.keys = nn.Parameter(torch.zeros(product_keys, num_keys, heads, dim_key))
        nn.init.normal_(self.keys, std=0.02)

        product_key_topk = default(product_key_topk, final_topk)
        assert final_topk <= (product_key_topk**product_keys)

        self.topk = product_key_topk
        self.final_topk = final_topk
        # the maximum index, or the total space being indexed into
        self.max_index = int(num_keys**product_keys)

    def forward(
        self, 
        x: Tensor, 
        softmax_scores: bool = False
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project and reshape queries: [b, n, (p h d)] -> [p, b, n, h, d]
        queries = self.to_queries(x)
        queries = queries.view(
            batch_size, seq_len, self.product_keys, self.heads, self.dim_key
        )
        queries = queries.permute(2, 0, 1, 3, 4)

        # Transpose keys for matmul: [p, k, h, d] -> [p, h, k, d]
        keys = self.keys.permute(0, 2, 1, 3)

        # Compute similarities using einsum (equivalent to the working implementation)
        sim = torch.einsum("p b n h d, p h k d -> p b n h k", queries, keys)

        # Get top-k scores and indices for each product key
        scores, indices = sim.topk(self.topk, dim=-1)

        # Compute cartesian products
        strides = self.num_keys ** torch.arange(self.product_keys, device=device)
        indices = indices * strides.view(-1, 1, 1, 1, 1)

        # Combine indices through cartesian product
        final_indices = indices[0]
        for i in range(1, self.product_keys):
            final_indices = final_indices.unsqueeze(-1) + indices[i].unsqueeze(-2)
            final_indices = final_indices.view(*final_indices.shape[:-2], -1)

        # Combine scores through addition
        final_scores = scores[0]
        for i in range(1, self.product_keys):
            final_scores = final_scores.unsqueeze(-1) + scores[i].unsqueeze(-2)
            final_scores = final_scores.view(*final_scores.shape[:-2], -1)

        # Get final top-k
        final_scores, pk_indices = final_scores.topk(self.final_topk, dim=-1)
        final_indices = torch.gather(final_indices, -1, pk_indices)

        if softmax_scores:
            final_scores = F.softmax(final_scores, dim=-1)

        return final_scores, final_indices


def is_perfect_square(n: Union[int, float]) -> bool:
    """
    Check if a number is a perfect square.
    
    Args:
        n: Number to check
        
    Returns:
        True if the number is a perfect square, False otherwise
    """
    if n < 0:
        return False
    root = int(sqrt(n))
    return root * root == n


def find_nearest_square(n: Union[int, float]) -> int:
    """
    Find the nearest perfect square to a given number.
    
    Args:
        n: Input number
        
    Returns:
        Nearest perfect square to the input
        
    Raises:
        ValueError: If the input is negative
    """
    if n < 0:
        raise ValueError("Input must be non-negative")

    # If it's already a perfect square, return it
    if is_perfect_square(n):
        return int(n)

    # Find the floor and ceiling square numbers
    floor_root = floor(sqrt(n))
    ceil_root = ceil(sqrt(n))
    floor_square = floor_root * floor_root
    ceil_square = ceil_root * ceil_root

    # Return the closer one
    if abs(n - floor_square) <= abs(n - ceil_square):
        return floor_square
    else:
        return ceil_square


def divide_and_round_to_square(number: Union[int, float], divisor: Union[int, float]) -> int:
    """
    Divide a number by a divisor and round to the nearest perfect square.

    Args:
        number: The number to divide
        divisor: The number to divide by

    Returns:
        The nearest perfect square to the division result
        
    Raises:
        ValueError: If divisor is zero
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero")

    # Perform the division
    divided = number / divisor

    # Find the nearest perfect square
    return int(find_nearest_square(divided))


if __name__ == "__main__":
    peer_attn = ProductKeyAttention(
        dim=256, causal=True, heads=4, num_key_values=int(1e4)
    )

    x = torch.randn(2, 512, 256)
    out, _ = peer_attn(x)
    assert x.shape == out.shape
    print("success!")
