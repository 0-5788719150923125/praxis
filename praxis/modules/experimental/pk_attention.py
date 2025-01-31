from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class PKAttention(Module):
    """
    Basically stolen from here:
    https://github.com/lucidrains/PEER-pytorch/blob/main/PEER_pytorch/PK.py
    """

    def __init__(
        self,
        dim,
        *,
        causal=True,
        heads=8,
        num_key_values=1_000_000,
        key_value_pk_topk=16,
        dim_key=None,
        product_keys=2,
        pre_rmsnorm=False,
        dropout=0.0,
    ):
        super().__init__()
        self.causal = causal
        self.heads = heads
        self.num_key_values = num_key_values
        self.dim = dim

        self.norm = nn.LayerNorm(dim) if pre_rmsnorm else nn.Identity()

        # queries projection
        self.to_queries = nn.Linear(dim, dim * heads, bias=False)

        # keys and values selected using product-key
        self.keys = nn.EmbeddingBag(num_key_values * heads, dim, mode="sum")
        self.values = nn.EmbeddingBag(num_key_values * heads, dim, mode="sum")

        assert sqrt(
            num_key_values
        ).is_integer(), "`num_key_values` needs to be a square"
        assert (dim % 2) == 0, "feature dimension should be divisible by 2"

        self.to_kv_pk_indices = PK(
            dim=dim,
            num_keys=int(sqrt(num_key_values)),
            final_topk=key_value_pk_topk,
            product_keys=product_keys,
        )

        self.dropout = nn.Dropout(dropout)

        # output projection
        self.to_out = nn.Linear(dim * heads, dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        x = self.norm(x)

        # queries
        q = self.to_queries(x)
        q = q.view(batch_size, seq_len, self.heads, -1)
        q = q.permute(0, 2, 1, 3)

        q = q * (q.shape[-1] ** -0.5)

        # keys and values
        kv_scores, indices = self.to_kv_pk_indices(x, softmax_scores=True)

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
            assert not exists(mask)
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        elif exists(mask):
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(~mask.unsqueeze(2), -torch.finfo(sim.dtype).max)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # aggregate
        out = torch.matmul(attn, v)

        # reshape back to [batch, seq_len, heads * dim]
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, -1)

        return self.to_out(out)


class PK(Module):
    def __init__(
        self,
        dim,
        *,
        heads=8,
        dim_key=None,
        num_keys=1_000,
        product_keys=2,
        product_key_topk=None,
        final_topk=16,
    ):
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
        self.max_index = int(num_keys**product_keys)

    def forward(self, x, softmax_scores=False):
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


if __name__ == "__main__":
    peer_attn = PKAttention(
        dim=256, causal=True, heads=8, num_key_values=int(1e4), pre_rmsnorm=True
    )

    x = torch.randn(2, 512, 256)
    out = peer_attn(x) + x
    assert x.shape == out.shape
    print("success!")
