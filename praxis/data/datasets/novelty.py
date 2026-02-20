"""Novelty-based data sampling using Count-Min Sketches for bigram frequency tracking.

Implements an online variant of importance resampling (cf. DSIR, Xie et al. NeurIPS 2023)
using streaming approximate frequency counting. Documents with rare token bigrams relative
to the global distribution are scored as more "novel", and their source datasets are
upsampled accordingly.
"""

from typing import List, Optional, Set

import numpy as np


class CountMinSketch:
    """Approximate frequency counter using the Count-Min Sketch data structure.

    Uses multiple hash functions to maintain frequency estimates with
    bounded overcount error. Memory usage: width * depth * 4 bytes (int32).
    """

    def __init__(self, width: int = 65536, depth: int = 4, seed: int = 0):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)
        # Per-depth hash seeds derived from base seed
        self._seeds = np.array(
            [seed + i * 0x9E3779B9 for i in range(depth)], dtype=np.uint64
        )

    def _hash(self, key: int, depth_idx: int) -> int:
        """Fast integer hash combining key with per-depth seed.

        Uses splitmix64-style mixing. Overflow on uint64 multiply is
        intentional and produces valid hash bits, so we suppress the warning.
        """
        with np.errstate(over="ignore"):
            # Wrap to uint64 range (handles negative keys from numeric sentinel)
            h = np.uint64(key % (2**64)) ^ self._seeds[depth_idx]
            h = (h ^ (h >> np.uint64(16))) * np.uint64(0x45D9F3B)
            h = (h ^ (h >> np.uint64(16))) * np.uint64(0x45D9F3B)
            h = h ^ (h >> np.uint64(16))
        return int(h % self.width)

    def add(self, key: int, count: int = 1):
        """Increment counters at hashed positions for the given key."""
        for d in range(self.depth):
            idx = self._hash(key, d)
            self.table[d, idx] += count

    def query(self, key: int) -> int:
        """Return minimum count across all depth rows (standard CMS query)."""
        min_val = self.table[0, self._hash(key, 0)]
        for d in range(1, self.depth):
            val = self.table[d, self._hash(key, d)]
            if val < min_val:
                min_val = val
        return int(min_val)

    def add_batch(self, keys: np.ndarray, count: int = 1):
        """Batch-insert multiple keys."""
        for key in keys:
            self.add(int(key), count)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        """Batch-query multiple keys, returning array of counts."""
        counts = np.empty(len(keys), dtype=np.int32)
        for i, key in enumerate(keys):
            counts[i] = self.query(int(key))
        return counts

    def decay(self, factor: float):
        """Multiply entire table by factor to forget old patterns."""
        self.table = (self.table * factor).astype(np.int32)


class NoveltyTracker:
    """Orchestrates novelty scoring and weight computation across datasets.

    Maintains a global Count-Min Sketch tracking all bigrams seen across
    all datasets, plus per-dataset sketches. Documents are scored by how
    rare their bigrams are in the global distribution. Per-dataset novelty
    scores are EMA-smoothed and used to adjust sampling weights.
    """

    # Sentinel value that replaces all numeric tokens before bigram extraction.
    # Collapsing digits prevents random numbers from inflating novelty scores.
    _NUM_SENTINEL = -1

    def __init__(
        self,
        num_datasets: int,
        cms_width: int = 65536,
        cms_depth: int = 4,
        ema_alpha: float = 0.3,
        novelty_exponent: float = 0.5,
        decay_factor: float = 0.95,
        decay_interval: int = 1000,
        warmup_samples: int = 50,
        numeric_token_ids: Optional[Set[int]] = None,
    ):
        self.num_datasets = num_datasets
        self.ema_alpha = ema_alpha
        self.novelty_exponent = novelty_exponent
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.warmup_samples = warmup_samples
        self._numeric_ids = numeric_token_ids or set()

        # Global CMS tracking all bigrams across all datasets
        self.global_cms = CountMinSketch(width=cms_width, depth=cms_depth, seed=42)

        # Per-dataset CMS
        self.dataset_cms = [
            CountMinSketch(width=cms_width, depth=cms_depth, seed=42 + i + 1)
            for i in range(num_datasets)
        ]

        # Per-dataset EMA novelty scores (start at 1.0 = maximally novel)
        self.dataset_novelty = np.ones(num_datasets, dtype=np.float64)

        # Total documents processed (for decay scheduling and warmup)
        self.total_docs = 0

    def _normalize_tokens(self, token_ids: List[int]) -> List[int]:
        """Replace numeric tokens with a sentinel value.

        This prevents random digit sequences (e.g. tool-calling arguments,
        code literals) from inflating novelty scores.  After normalization
        "What is 922583 + 622541" and "What is 47 + 13" produce identical
        bigrams, correctly identifying the template as repetitive.
        """
        if not self._numeric_ids:
            return token_ids
        sentinel = self._NUM_SENTINEL
        return [sentinel if t in self._numeric_ids else t for t in token_ids]

    def _extract_bigram_keys(self, token_ids: List[int]) -> np.ndarray:
        """Extract bigram keys from token IDs.

        Packs two token IDs into one integer: key = token_a * 131072 + token_b.
        131072 = 2^17 covers vocab sizes up to 128K.
        """
        if len(token_ids) < 2:
            return np.array([], dtype=np.int64)
        ids = np.array(token_ids, dtype=np.int64)
        keys = ids[:-1] * 131072 + ids[1:]
        return keys

    def score_and_update(self, dataset_idx: int, token_ids: List[int]) -> float:
        """Score a document's novelty and update all sketches.

        Args:
            dataset_idx: Index of the source dataset.
            token_ids: Token IDs of the document.

        Returns:
            The novelty score for this document.
        """
        # Normalize: collapse numeric tokens so random digits don't dominate
        token_ids = self._normalize_tokens(token_ids)
        bigram_keys = self._extract_bigram_keys(token_ids)

        if len(bigram_keys) == 0:
            return 0.0

        # Score: median(1 / (1 + global_count(bigram))) across all bigrams.
        # Median is robust against outlier bigrams (e.g. residual high-entropy
        # tokens like base64 or UUIDs that slip past numeric normalization).
        counts = self.global_cms.query_batch(bigram_keys)
        doc_score = float(np.median(1.0 / (1.0 + counts.astype(np.float64))))

        # EMA-update dataset novelty
        self.dataset_novelty[dataset_idx] = (
            self.ema_alpha * doc_score
            + (1.0 - self.ema_alpha) * self.dataset_novelty[dataset_idx]
        )

        # Insert bigrams into global and per-dataset CMS
        self.global_cms.add_batch(bigram_keys)
        self.dataset_cms[dataset_idx].add_batch(bigram_keys)

        # Periodic decay to forget old patterns
        self.total_docs += 1
        if self.total_docs % self.decay_interval == 0:
            self.global_cms.decay(self.decay_factor)
            for cms in self.dataset_cms:
                cms.decay(self.decay_factor)

        return doc_score

    def get_target_weights(self, static_weights: List[float]) -> List[float]:
        """Compute target sampling weights from novelty scores.

        Args:
            static_weights: Base/static weights for each dataset.

        Returns:
            Normalized target weights adjusted by novelty.
        """
        n = len(static_weights)
        raw = np.array(static_weights, dtype=np.float64)

        # Scale by novelty^exponent (sqrt by default to dampen extremes)
        novelty_factor = self.dataset_novelty[:n] ** self.novelty_exponent
        raw = raw * novelty_factor

        # Warmup blend: linearly blend from static to novelty weights
        if self.total_docs < self.warmup_samples:
            blend = self.total_docs / self.warmup_samples
            static = np.array(static_weights, dtype=np.float64)
            static = static / static.sum()
            raw = (1.0 - blend) * static + blend * raw

        # Floor clamp: no dataset drops below 1% of its static weight
        floor = 0.01 * np.array(static_weights, dtype=np.float64)
        raw = np.maximum(raw, floor)

        # Normalize to sum=1.0
        total = raw.sum()
        if total > 0:
            raw = raw / total

        return raw.tolist()
