import threading
from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn


class RealtimeNgramProcessor(nn.Module):
    def __init__(
        self, ngram_to_size: Dict[int, int], min_freq: int = 1, reset_every: int = 100
    ):
        """
        Initialize processor with vocabulary size limits per ngram length

        Args:
            ngram_to_size: Dict mapping ngram length to max vocab size
            min_freq: Minimum frequency required to include ngram in vocab
            reset_every: Flush ngram counts to make room for new discoveries
        """
        super().__init__()
        self.device = torch.device("cpu")
        self.ngram_to_size = ngram_to_size
        self.min_freq = min_freq
        self.ngram_sizes = sorted(ngram_to_size.keys())
        self.reset_every = reset_every  # Number of batches between resets
        assert min(self.ngram_sizes) >= 2, "Minimum ngram size must be 2"

        # Frequency tracking
        self.ngram_counters: Dict[int, Counter] = {
            n: Counter() for n in self.ngram_sizes
        }

        # Vocabulary mappings
        self.ngram_to_idx_tables: Dict[int, Dict[Tuple, int]] = {
            n: {} for n in self.ngram_sizes
        }

        # Track vocab sizes
        self.vocab_sizes: Dict[int, int] = {n: 0 for n in self.ngram_sizes}

        # Batch counter for resets
        self.batch_count = 0

        # Lock for thread safety
        self._lock = threading.Lock()

    def _truncate_counter(self, n: int):
        """Truncate counter to keep only top-k most frequent items"""
        counter = self.ngram_counters[n]
        max_size = self.ngram_to_size[n] * 2  # Keep 2x vocab size for buffer

        if len(counter) > max_size:
            most_common = counter.most_common(max_size)
            self.ngram_counters[n] = Counter(dict(most_common))

    def _maybe_reset_counters(self):
        """Periodically reset counters while preserving vocabulary information"""
        self.batch_count += 1
        if self.batch_count % self.reset_every == 0:
            with self._lock:
                # Store current vocabs
                old_vocabs = {
                    n: set(self.ngram_to_idx_tables[n].keys()) for n in self.ngram_sizes
                }

                # Reset counters
                self.ngram_counters = {n: Counter() for n in self.ngram_sizes}

                # Initialize new counters with small counts for existing vocab
                # This helps maintain stability while allowing competition
                for n in self.ngram_sizes:
                    self.ngram_counters[n].update(
                        {ngram: self.min_freq for ngram in old_vocabs[n]}
                    )

    def _update_frequencies(self, data: Union[np.ndarray, torch.Tensor]):
        """Update ngram frequencies preserving original ordering"""
        data_np = self._to_numpy(data)
        batch_size, seq_len = data_np.shape

        # Check if we should reset counters
        self._maybe_reset_counters()

        # Pre-allocate a buffer for all n-grams to avoid repeated padding
        max_n = max(self.ngram_sizes)
        padded = np.pad(
            data_np,
            ((0, 0), (max_n - 1, 0)),
            mode="constant",
            constant_values=0,
        )

        for n in self.ngram_sizes:
            # Create view into padded array for current n-gram size
            offset = max_n - n
            windows = np.lib.stride_tricks.as_strided(
                padded[:, offset:],
                shape=(batch_size, seq_len, n),
                strides=padded.strides[:2] + (padded.strides[1],),
            )

            # Process windows maintaining original order
            # Use contiguous array for better performance
            flat_windows = np.ascontiguousarray(windows.reshape(-1, n))

            # Update frequencies preserving order of discovery
            batch_counts = Counter(tuple(window) for window in flat_windows)

            with self._lock:
                self.ngram_counters[n].update(batch_counts)
                self._truncate_counter(n)

    def _rebuild_vocabulary(self, n: int):
        """Rebuild vocabulary for n-grams based on current frequencies"""
        counter = self.ngram_counters[n]
        max_size = self.ngram_to_size[n]

        # Get most common ngrams meeting minimum frequency
        most_common = [
            (ngram, count)
            for ngram, count in counter.most_common()
            if count >= self.min_freq
        ][:max_size]

        # Build new mapping
        new_table = {
            ngram: idx + 4  # Keep original offset
            for idx, (ngram, _) in enumerate(most_common)
        }

        with self._lock:
            self.ngram_to_idx_tables[n] = new_table
            self.vocab_sizes[n] = len(new_table) + 4

    # Rest of the methods remain unchanged...
    def update_from_batch(self, data: np.ndarray):
        self._update_frequencies(data)
        for n in self.ngram_sizes:
            self._rebuild_vocabulary(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        ngram_tensors = self.encode_token_ngrams(x)
        return self.stack_ngrams(ngram_tensors)

    def _compute_ngram_ids(
        self, data: Union[np.ndarray, torch.Tensor], n: int
    ) -> torch.Tensor:
        data_np = self._to_numpy(data)
        batch_size, seq_len = data_np.shape
        padded = np.pad(
            data_np, ((0, 0), (n - 1, 0)), mode="constant", constant_values=0
        )
        windows = np.lib.stride_tricks.as_strided(
            padded,
            shape=(batch_size, seq_len, n),
            strides=padded.strides[:2] + (padded.strides[1],),
        )

        lookup_table = self.ngram_to_idx_tables[n]
        ids_np = np.array(
            [
                [lookup_table.get(tuple(window), 0) for window in batch]
                for batch in windows
            ]
        )

        return self._to_torch(ids_np)

    def encode_token_ngrams(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> List[torch.Tensor]:
        return [self._compute_ngram_ids(data, n) for n in self.ngram_sizes]

    def stack_ngrams(self, ngram_tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(ngram_tensors, dim=0)

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _to_torch(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).to(self.device)


if __name__ == "__main__":
    # Initialize with same vocabulary size constraints
    ngram_to_size = {
        2: 38396,
        3: 50000,
        4: 50000,
        5: 50000,
        6: 50000,
        7: 50000,
        8: 50000,
    }

    # Test both CPU and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = torch.tensor(torch.randn(4, 64)).to(device)

    processor = RealtimeNgramProcessor(ngram_to_size)
    processor.update_from_batch(input_data)
    ngram_ids = processor(input_data[0].unsqueeze(0))
    print(ngram_ids)
    print(f"Input device: {input_data.device}")
    print(f"Output device: {ngram_ids.device}")
