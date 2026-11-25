"""Interpolated Kneser-Ney character n-grams, vectorized with numpy: the exact-match reader."""
import numpy as np


def _codes(data, n, V):
    T = len(data) - n + 1
    code = np.zeros(T, dtype=np.int64)
    for j in range(n):
        code = code * V + data[j : j + T]
    return code


def _lookup(table, counts, query):
    pos = np.searchsorted(table, query).clip(0, len(table) - 1)
    return np.where(table[pos] == query, counts[pos], 0)


class KneserNey:
    """P(w | last order-1 symbols); lower orders use continuation counts."""

    def __init__(self, data, V, order):
        self.V, self.order = V, order
        data = np.concatenate([np.zeros(order - 1, dtype=np.int64), data.astype(np.int64)])
        raw = [None] + [np.unique(_codes(data, n, V), return_counts=True) for n in range(1, order + 1)]
        self.levels = []
        for n in range(1, order + 1):
            if n == order:
                grams, counts = raw[n]
            else:
                grams, counts = np.unique(raw[n + 1][0] % V**n, return_counts=True)
            n1, n2 = (counts == 1).sum(), (counts == 2).sum()
            discount = n1 / max(n1 + 2 * n2, 1)
            ctx, inverse = np.unique(grams // V, return_inverse=True)
            totals = np.bincount(inverse, weights=counts).astype(np.int64)
            types = np.bincount(inverse).astype(np.int64)
            self.levels.append((grams, counts, ctx, totals, types, discount))
        self.stored = sum(len(level[0]) for level in self.levels)

    def _prob(self, history, w):
        """history[n]: code of the last n symbols (n = 0..order-1); w: candidate symbols."""
        p = np.full(np.shape(w), 1.0 / self.V)
        for n, (grams, counts, ctx, totals, types, d) in enumerate(self.levels):
            h = history[n]
            c = _lookup(grams, counts, h * self.V + w)
            total = _lookup(ctx, totals, h)
            kinds = _lookup(ctx, types, h)
            safe = np.maximum(total, 1)
            mixed = np.maximum(c - d, 0) / safe + d * kinds / safe * p
            p = np.where(total > 0, mixed, p)
        return p

    def _history(self, data):
        data = np.concatenate([np.zeros(self.order - 1, dtype=np.int64), data.astype(np.int64)])
        T = len(data) - self.order + 1
        hist = [np.zeros(T, dtype=np.int64)]
        for n in range(1, self.order):
            hist.append(_codes(data[self.order - 1 - n : len(data) - 1], n, self.V)[:T])
        return hist, data[self.order - 1 :]

    def prob(self, data):
        """P(data[t] | data[:t]) for every t (history padded with symbol 0)."""
        hist, targets = self._history(data)
        return self._prob(hist, targets)

    def table(self, text):
        """(B, T) streams -> (B, T, V): row t is P(. | text[b, :t])."""
        hist, _ = self._history(text.reshape(-1))
        out = np.stack([self._prob(hist, np.full(len(hist[0]), w)) for w in range(self.V)], -1)
        return (out / out.sum(-1, keepdims=True)).reshape(*text.shape, self.V)

    def dist(self, context):
        """Next-symbol distribution after one context (a sequence of ints)."""
        context = [0] * self.order + [int(c) for c in context]
        hist = []
        for n in range(self.order):
            code = 0
            for s in context[len(context) - n :] if n else []:
                code = code * self.V + s
            hist.append(np.full(self.V, code, dtype=np.int64))
        p = self._prob(hist, np.arange(self.V))
        return p / p.sum()
