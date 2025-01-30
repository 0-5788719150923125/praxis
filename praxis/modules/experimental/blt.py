import threading
from collections import Counter
from heapq import heappop, heappush
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


class RealtimeNgramProcessor:
    def __init__(self, ngram_to_size: Dict[int, int], min_freq: int = 1):
        """
        Initialize processor with vocabulary size limits per ngram length

        Args:
            ngram_to_size: Dict mapping ngram length to max vocab size
            min_freq: Minimum frequency required to include ngram in vocab
        """
        self.ngram_to_size = ngram_to_size
        self.min_freq = min_freq
        self.ngram_sizes = sorted(ngram_to_size.keys())
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

        # Lock for thread safety during updates
        self._lock = threading.Lock()

    def _update_frequencies(self, data: np.ndarray):
        """Update ngram frequencies from new batch"""

        data_np = self._to_numpy(data)

        batch_size, seq_len = data.shape

        for n in self.ngram_sizes:
            # Create sliding windows
            padded = np.pad(
                data, ((0, 0), (n - 1, 0)), mode="constant", constant_values=0
            )
            windows = np.lib.stride_tricks.as_strided(
                padded,
                shape=(batch_size, seq_len, n),
                strides=padded.strides[:2] + (padded.strides[1],),
            )

            # Update frequencies
            batch_counts = Counter(
                tuple(window) for batch in windows for window in batch
            )

            with self._lock:
                self.ngram_counters[n].update(batch_counts)

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
            self.vocab_sizes[n] = len(new_table) + 4  # Include offset

    def update_from_batch(self, data: np.ndarray):
        """Update statistics from new batch and rebuild vocabs if needed"""
        self._update_frequencies(data)

        # Rebuild vocabularies
        for n in self.ngram_sizes:
            self._rebuild_vocabulary(n)

    def _compute_ngram_ids(
        self, data: Union[np.ndarray, torch.Tensor], n: int
    ) -> torch.Tensor:
        """Compute ngram IDs and return torch tensor"""
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
        """Encode and return list of torch tensors"""
        return [self._compute_ngram_ids(data, n) for n in self.ngram_sizes]

    def stack_ngrams(self, ngram_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Stack ngram tensors along first dimension"""
        return torch.stack(ngram_tensors, dim=0)

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input data to numpy if needed"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _to_torch(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor"""
        return torch.from_numpy(data)


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

    # Can use either numpy or torch input
    input_data = torch.tensor([[65, 66, 67, 68], [69, 70, 71, 72]])
    # Or: input_data = np.array([[65, 66, 67, 68], [69, 70, 71, 72]])

    processor = RealtimeNgramProcessor(ngram_to_size)
    processor.update_from_batch(input_data)

    # Get tensor outputs
    ngram_tensors = processor.encode_token_ngrams(input_data)
    # Stack into single tensor
    ngram_ids = processor.stack_ngrams(ngram_tensors)
    print(ngram_ids)
