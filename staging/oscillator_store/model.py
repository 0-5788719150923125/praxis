"""A character model whose knowledge is a text store, addressed by a learnable oscillator.

  Oscillator   64 damped rotations, h_t = lam * h_{t-1} + E[x_t] with lam = exp(-exp(nu) + i theta).
               Its state is the address: 128 numbers saying what the recent history looks like.
  TextStore    every position of the training text, keyed by the oscillator, with the character
               that followed it. An inverted-file index (Jegou, Douze & Schmid 2011) over k-means
               or s-means centroids limits each search to the nearest few lists.
  TinyExperts  SMEAR (Muqeeth, Liu & Raffel 2023): many tiny MLPs (hidden 4) merged per position by
               a router reading the address, then run once. The one parametric reader.
  Gate         weighs the readers per position from their own confidence (about 50 numbers).

Readers: the store's nearest neighbors (soft match), Kneser-Ney over the same text (exact match,
ngram.py), the experts, and a GRU trained on the same text (the kNN-LM pairing, for comparison with the
experts). run.py trains the pieces in order and measures each one.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V = 34
WARMUP = 32  # leading positions of each stream that are neither stored nor scored
NEIGHBORS = 64


def entries(keys, text):
    """Keys at positions WARMUP..T-2 of each stream, and the characters that followed them."""
    k = keys[:, WARMUP:-1].reshape(-1, keys.shape[-1])
    y = torch.as_tensor(np.ascontiguousarray(text[:, WARMUP + 1 :]).reshape(-1), device=keys.device)
    return k, y


def scan(u, lam, chunk=32):
    """h_t = lam * h_{t-1} + u_t along dim 1 from h = 0, one chunk of steps per sequential pass."""
    B, T, n = u.shape
    pad = (-T) % chunk
    if pad:
        u = torch.cat([u, u.new_zeros(B, pad, n)], 1)
    u = u.view(B, -1, chunk, n)
    i = torch.arange(chunk, device=u.device)
    gap = i[:, None] - i[None, :]
    log_lam = torch.log(lam)
    powers = torch.exp(log_lam[:, None, None] * gap.clamp(min=0))
    within = torch.where(gap >= 0, powers, torch.zeros((), dtype=powers.dtype, device=u.device))
    carry = torch.exp(log_lam[None, :] * (i[:, None] + 1))  # lam^(j+1) for step j of a chunk
    local = torch.einsum("nij,bcjn->bcin", within, u)
    h = u.new_zeros(B, n)
    out = []
    for c in range(u.shape[1]):
        block = local[:, c] + carry * h[:, None, :]
        out.append(block)
        h = block[:, -1]
    return torch.stack(out, 1).reshape(B, -1, n)[:, :T]


class Oscillator(nn.Module):
    """64 complex units; starts as Test 6's fixed bank (8 decay rates x 8 phases)."""

    def __init__(self, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        r = 1 - 2.0 ** -np.linspace(1, 6, 8)  # timescales of 2 to 64 characters
        theta = np.pi * np.arange(8) / 8
        lam = (r[:, None] * np.exp(1j * theta[None, :])).reshape(-1)
        emb = rng.normal(size=(V, 64)) + 1j * rng.normal(size=(V, 64))
        param = lambda a: nn.Parameter(torch.tensor(np.asarray(a), dtype=torch.float32))
        self.nu = param(np.log(-np.log(np.abs(lam))))
        self.theta = param(np.angle(lam))
        self.emb_re, self.emb_im = param(emb.real), param(emb.imag)

    def lam(self):
        return torch.polar(torch.exp(-torch.exp(self.nu)), self.theta)

    def embedding(self):
        return torch.complex(self.emb_re, self.emb_im)

    def forward(self, text):
        """(B, T) characters -> (B, T, 128) states as real and imaginary halves."""
        h = scan(self.embedding()[text], self.lam())
        return torch.cat([h.real, h.imag], -1)

    @torch.no_grad()
    def stats(self, text, streams=250):
        """Per-dimension mean and spread of the stored keys, from the first streams of `text`."""
        keys = torch.cat([entries(self(torch.from_numpy(text[i : i + streams]).to(self.nu.device)), text[i : i + streams])[0] for i in range(0, min(len(text), 1000), streams)])
        return keys.mean(0), keys.std(0).clamp_min(1e-6)

    @torch.no_grad()
    def keys(self, text, mu, sd, dtype=torch.float32, device=None, streams=250):
        """Standardized keys and next characters for every stored position of `text`."""
        device = device or self.nu.device
        ks, ys = [], []
        for i in range(0, len(text), streams):
            part = text[i : i + streams]
            k, y = entries(self(torch.from_numpy(part).to(self.nu.device)), part)
            ks.append(((k - mu) / sd).to(device=device, dtype=dtype))
            ys.append(y)
        return torch.cat(ks), torch.cat(ys)


def nearest(points, table, m, budget=2**29):
    """Squared distances and indices of the m nearest rows of `table` for each row of `points`."""
    tn = (table * table).sum(1)
    rows = max(16, budget // (4 * len(table)))
    dist = torch.empty(len(points), m, device=points.device)
    index = torch.empty(len(points), m, dtype=torch.long, device=points.device)
    for i in range(0, len(points), rows):
        q = points[i : i + rows]
        d = torch.addmm(tn[None, :], q, table.T, beta=1, alpha=-2)
        dv, di = torch.topk(d, m, dim=1, largest=False)
        dist[i : i + rows] = (dv + (q * q).sum(1, keepdim=True)).clamp_min(0)
        index[i : i + rows] = di
    return dist, index


def lloyd(x, lists, iters=25, seed=0):
    """Plain k-means from random rows."""
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:lists].to(x.device)].clone()
    for _ in range(iters):
        lab = nearest(x, c, 1)[1][:, 0]
        cnt = torch.bincount(lab, minlength=lists).float()
        s = torch.zeros_like(c).index_add_(0, lab, x)
        keep = cnt > 0
        c[keep] = s[keep] / cnt[keep, None]
    return c


class TextStore:
    """Keys and next characters for every stored position, searched exactly or by inverted file.

    Keys may live on the CPU (float16) and stream to the GPU a block or a list at a time.
    """

    def __init__(self, keys, nexts, device):
        self.keys, self.nexts, self.device = keys, nexts.to(device), device
        self.centroids = None

    def __len__(self):
        return len(self.keys)

    def exact(self, q, k=NEIGHBORS, block=1 << 20):
        best_d = torch.full((len(q), k), float("inf"), device=self.device)
        best_i = torch.zeros((len(q), k), dtype=torch.long, device=self.device)
        for start in range(0, len(self), block):
            kb = self.keys[start : start + block].to(self.device, non_blocking=True).float()
            d, i = nearest(q, kb, min(k, len(kb)))
            cd, ci = torch.cat([best_d, d], 1), torch.cat([best_i, i + start], 1)
            best_d, j = torch.topk(cd, k, dim=1, largest=False)
            best_i = torch.gather(ci, 1, j)
        return best_d, best_i

    def build_index(self, lists, cluster, sample=1_000_000, seed=0):
        """Cluster a sample of keys into `lists` centroids, then file every key under its nearest."""
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(len(self), generator=g)[: min(sample, len(self))]
        self.centroids = cluster(self.keys[pick].to(self.device).float(), lists).float().to(self.device)
        assign = torch.cat([nearest(self.keys[s : s + (1 << 20)].to(self.device).float(), self.centroids, 1)[1][:, 0] for s in range(0, len(self), 1 << 20)])
        order = torch.argsort(assign)
        self.keys = self.keys[order.to(self.keys.device)]
        self.nexts = self.nexts[order]
        counts = torch.bincount(assign, minlength=lists)
        self.offsets = [0] + counts.cumsum(0).tolist()
        self.nprobe = max(1, lists // 128)  # search about 1/128 of the store
        self.mse = float(nearest(self.keys[: min(len(self), 1 << 20)].to(self.device).float(), self.centroids, 1)[0].mean())

    def search(self, q, k=NEIGHBORS):
        if self.centroids is None:
            return self.exact(q, k)
        lists = len(self.centroids)
        probes = nearest(q, self.centroids, self.nprobe)[1].reshape(-1)
        owner = torch.arange(len(q), device=self.device).repeat_interleave(self.nprobe)
        order = torch.argsort(probes)
        probes, owner = probes[order], owner[order]
        bounds = torch.searchsorted(probes, torch.arange(lists + 1, device=self.device)).tolist()
        best_d = torch.full((len(q), k), float("inf"), device=self.device)
        best_i = torch.zeros((len(q), k), dtype=torch.long, device=self.device)
        qn = (q * q).sum(1)
        for l in torch.unique(probes).tolist():
            a, b = bounds[l], bounds[l + 1]
            s, e = self.offsets[l], self.offsets[l + 1]
            if a == b or s == e:
                continue
            who = owner[a:b]
            kb = self.keys[s:e].to(self.device, non_blocking=True).float()
            d = (qn[who, None] - 2 * q[who] @ kb.T + (kb * kb).sum(1)[None]).clamp_min(0)
            dv, di = torch.topk(d, min(k, e - s), dim=1, largest=False)
            cd, ci = torch.cat([best_d[who], dv], 1), torch.cat([best_i[who], di + s], 1)
            top, j = torch.topk(cd, k, dim=1, largest=False)
            best_d[who], best_i[who] = top, torch.gather(ci, 1, j)
        return best_d, best_i


def knn_probs(dist, chars, T):
    """Next-character distribution from neighbors at squared distance `dist` that were followed by `chars`."""
    w = torch.softmax(-dist / T, 1).nan_to_num(0.0)
    return torch.zeros(len(dist), V, device=dist.device).scatter_add_(1, chars, w)


class TinyExperts(nn.Module):
    """SMEAR over tiny MLPs (key -> hidden 4 -> 34 logits): merge the experts' weights per position, run once.

    Base plus deviations, as in praxis/routers/smear.py: merged = base + sum_e w_e delta_e, deltas start
    at zero so every expert starts as the base. Expert dropout 0.1, as in the SMEAR paper.
    """

    def __init__(self, dim=128, hidden=4, experts=1024, dropout=0.1):
        super().__init__()
        self.dim, self.hidden, self.dropout = dim, hidden, dropout
        first, second = nn.Linear(dim, hidden), nn.Linear(hidden, V)
        base = torch.cat([first.weight.flatten(), first.bias, second.weight.flatten(), second.bias]).detach()
        self.base = nn.Parameter(base)
        self.deltas = nn.Parameter(torch.zeros(experts, base.numel()))
        self.router = nn.Linear(dim, experts)

    def forward(self, key):
        w = torch.softmax(self.router(key), -1)
        if self.training and self.dropout:
            w = w * (torch.rand_like(w) > self.dropout)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        p = self.base + w @ self.deltas
        d, h = self.dim, self.hidden
        w1, b1 = p[:, : h * d].view(-1, h, d), p[:, h * d : h * d + h]
        w2, b2 = p[:, h * d + h : h * d + h + V * h].view(-1, V, h), p[:, h * d + h + V * h :]
        z = F.gelu(torch.bmm(w1, key[:, :, None]).squeeze(-1) + b1)
        return torch.bmm(w2, z[:, :, None]).squeeze(-1) + b2


def entropy_bits(p):
    return -(p * torch.log2(p.clamp_min(1e-12))).sum(1)


def confidence(nearest_d2, dists):
    """What the gate sees: how close the nearest stored moment is, and how sure each reader is."""
    columns = [torch.log(nearest_d2 + 1e-6)]
    for p in dists:
        columns += [p.max(1).values, entropy_bits(p)]
    return torch.stack(columns, 1)


class Gate(nn.Module):
    """Mixing weights over the readers per position, from their confidence: (1 + 2R) -> 4 -> R."""

    def __init__(self, readers):
        features = 1 + 2 * readers
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 4), nn.GELU(), nn.Linear(4, readers))
        self.register_buffer("mu", torch.zeros(features))
        self.register_buffer("sd", torch.ones(features))

    def forward(self, f):
        return torch.softmax(self.net((f - self.mu) / self.sd), -1)


class GRU(nn.Module):
    """The parametric baseline: everything in trained weights."""

    def __init__(self, hidden):
        super().__init__()
        self.rnn = nn.GRU(V, hidden, batch_first=True)
        self.out = nn.Linear(hidden, V)

    def forward(self, text):
        y, _ = self.rnn(F.one_hot(text, V).float())
        return self.out(y)


def count(module):
    return sum(p.numel() for p in module.parameters())


def bits(p):
    return float(-torch.log2(p.clamp_min(1e-12)).mean())
