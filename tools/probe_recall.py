#!/usr/bin/env python3
"""Associative recall on one attention layer: can it find a value by its key?

A row is n (key, value) pairs and then a query key; the target is that key's
value. Every input also carries its previous token's embedding, standing in for
a previous-token head, so one layer suffices: a value's position holds its key.
Runs on CPU in minutes. The evidence behind next/cracked_mirror.md.

    python tools/probe_recall.py                                  # all variants
    python tools/probe_recall.py --variants drop,crack --pairs 48
    python tools/probe_recall.py --repeat-keys 8 --nth 2          # second-latest

Variants:
    qk        single-head softmax q.k, no position encoding
    split     kaleidoscope, ratio + lag mirrors, no crack
    count     the crack's count alone (offset frozen at 0), count attached
    drop      the crack's offset alone (crack mirrors never weighted)
    attached  offset + count, count sends the classifier gradient
    crack     offset + count as built (count detached)
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from praxis.attention.kaleidoscope import KaleidoscopeAttention  # noqa: E402

N_KEYS, N_VALUES, D = 64, 32, 64


def batch(bs, n_pairs, gen, repeat=0, nth=1):
    """Rows and targets. ``repeat > 0`` draws keys from that many so they recur,
    and the target is the value at the ``nth``-latest occurrence."""
    vals = torch.randint(0, N_VALUES, (bs, n_pairs), generator=gen) + N_KEYS
    if not repeat:
        keys = torch.stack(
            [torch.randperm(N_KEYS, generator=gen)[:n_pairs] for _ in range(bs)]
        )
        pick = torch.randint(0, n_pairs, (bs,), generator=gen)
        q = keys[torch.arange(bs), pick]
        tgt = vals[torch.arange(bs), pick] - N_KEYS
        seq = torch.stack([keys, vals], -1).reshape(bs, -1)
        return torch.cat([seq, q[:, None]], 1), tgt
    keys = torch.randint(0, repeat, (bs, n_pairs), generator=gen)
    q = torch.empty(bs, dtype=torch.long)
    tgt = torch.empty(bs, dtype=torch.long)
    for b in range(bs):
        while True:
            counts = torch.bincount(keys[b], minlength=repeat)
            pool = (counts >= max(2, nth)).nonzero().flatten()
            if len(pool):
                break
            keys[b] = torch.randint(0, repeat, (n_pairs,), generator=gen)
        k = pool[torch.randint(0, len(pool), (1,), generator=gen)].item()
        at = (keys[b] == k).nonzero().flatten()[-nth]
        q[b], tgt[b] = k, vals[b, at] - N_KEYS
    seq = torch.stack([keys, vals], -1).reshape(bs, -1)
    return torch.cat([seq, q[:, None]], 1), tgt


class Attached(KaleidoscopeAttention):
    """``KaleidoscopeAttention._crack`` without the detach on the count."""

    def _crack(self, inputs, allowed):
        T = inputs.shape[1]
        plane = self.crack_plane(inputs).float()
        normal, threshold, drop = plane[..., :-2], plane[..., -2], plane[..., -1]
        point = self.crack_point(inputs).float()
        logits = normal @ point.transpose(1, 2) * self.crack_dim**-0.5
        logits = logits + threshold.unsqueeze(-1)
        pos = torch.arange(T, device=inputs.device)
        seen = (pos[None, :] < pos[:, None]) & allowed
        side = torch.sigmoid(logits)
        gate = side * seen
        if self.training:
            self._note_crack(logits, gate, normal, point, seen.expand_as(gate), drop)
        lag = gate.flip(-1).cumsum(-1).flip(-1) - 0.5 * gate
        return lag.clamp_min(0.0), drop.unsqueeze(-1) * (side - 0.5) * seen


class QK(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self._metrics = {}

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out), None, 0.0


def _freeze(param, index, dim=0):
    with torch.no_grad():
        param.index_fill_(dim, torch.as_tensor(index), 0.0)
    param.register_hook(lambda g: g.index_fill(dim, torch.as_tensor(index), 0.0))


class Model(nn.Module):
    def __init__(self, variant, T):
        super().__init__()
        V = N_KEYS + N_VALUES + 1
        self.emb = nn.Embedding(V, D)
        self.prev = nn.Embedding(V, D)
        if variant == "qk":
            self.attn = QK(D)
        else:
            cfg = SimpleNamespace(
                hidden_size=D,
                num_heads=1,
                head_size=D,
                num_queries=1,
                causal=True,
                dropout=0.0,
                depth=1,
                num_layers=1,
                window_size=None,
                max_position_embeddings=T,
            )
            cls = (
                Attached if variant in ("attached", "count") else KaleidoscopeAttention
            )
            coords = "split" if variant == "split" else "crack"
            self.attn = cls(cfg, num_mirrors=12, coords=coords)
        a = self.attn
        if variant == "count":
            _freeze(a.crack_plane.weight, [a.crack_dim + 1])
            _freeze(a.crack_plane.bias, [a.crack_dim + 1])
        if variant == "drop":
            rows = list(range(a.n_shared, a.num_mirrors))
            _freeze(a.turn.weight, rows)
            _freeze(a.turn_static.weight, rows, dim=1)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, N_VALUES)

    def forward(self, tok):
        pad = torch.full_like(tok[:, :1], N_KEYS + N_VALUES)
        x = self.emb(tok) + self.prev(torch.cat([pad, tok[:, :-1]], 1))
        h, _, _ = self.attn(x)
        return self.head(self.norm(x + h)[:, -1])


def accuracy(model, args):
    model.eval()
    with torch.no_grad():
        gen = torch.Generator().manual_seed(999)
        tok, tgt = batch(1024, args.pairs, gen, args.repeat_keys, args.nth)
        acc = (model(tok).argmax(-1) == tgt).float().mean().item()
    model.train()
    return acc


def run(variant, args):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = Model(variant, 2 * args.pairs + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start, curve = time.time(), []
    for step in range(args.steps):
        if step and step % 250 == 0:
            curve.append(round(accuracy(model, args), 2))
        tok, tgt = batch(64, args.pairs, gen, args.repeat_keys, args.nth)
        loss = F.cross_entropy(model(tok), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    metrics = {
        k: round(v, 3)
        for k, v in model.attn._metrics.items()
        if k.startswith("kaleido_crack") or k.endswith("_share")
    }
    print(
        f"{variant:9s} acc={accuracy(model, args):.3f} (chance {1 / N_VALUES:.3f}) "
        f"every 250 steps={curve} {time.time() - start:.0f}s {metrics}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default="qk,split,count,drop,attached,crack")
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--repeat-keys", type=int, default=0)
    ap.add_argument("--nth", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(
        f"pairs={args.pairs} T={2 * args.pairs + 1} repeat_keys={args.repeat_keys} "
        f"nth={args.nth} lr={args.lr} seed={args.seed}"
    )
    for variant in args.variants.split(","):
        run(variant, args)


if __name__ == "__main__":
    main()
