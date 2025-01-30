import random
import threading
import time
from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn


class RealtimeNgramProcessor(nn.Module):
    def __init__(
        self,
        ngram_to_size: Dict[int, int],
        min_freq: int = 1,
        reset_every: int = 100,
        debug: bool = False,
    ):
        """
        Initialize processor with vocabulary size limits per ngram length

        Args:
            ngram_to_size: Dict mapping ngram length to max vocab size
            min_freq: Minimum frequency required to include ngram in vocab
            reset_every: Flush ngram counts to make room for new discoveries
        """
        super().__init__()
        self.debug = debug
        self.print_frequency = 0.1
        self.device = torch.device("cpu")
        self.ngram_to_size = ngram_to_size
        self.min_freq = min_freq
        self.ngram_sizes = sorted(ngram_to_size.keys())
        self.reset_every = reset_every
        assert min(self.ngram_sizes) >= 2, "Minimum ngram size must be 2"

        # Frequency tracking
        self.ngram_counters = {n: Counter() for n in self.ngram_sizes}

        # Vocabulary mappings
        self.ngram_to_idx_tables = {n: {} for n in self.ngram_sizes}
        self.vocab_sizes = {n: 0 for n in self.ngram_sizes}
        self.batch_count = 0
        self._lock = threading.Lock()

    def _update_frequencies(self, data: Union[np.ndarray, torch.Tensor]):
        timing_stats = {}
        total_start = time.perf_counter()

        data_np = self._to_numpy(data)
        batch_size, seq_len = data_np.shape

        self._maybe_reset_counters()

        max_n = max(self.ngram_sizes)
        padded = np.pad(
            data_np,
            ((0, 0), (max_n - 1, 0)),
            mode="constant",
            constant_values=0,
        )

        for n in self.ngram_sizes:
            t0 = time.perf_counter()

            # Create windows efficiently
            windows = np.lib.stride_tricks.as_strided(
                padded[:, max_n - n :],
                shape=(batch_size, seq_len, n),
                strides=padded.strides[:2] + (padded.strides[1],),
            )

            # Fast counting using Counter directly
            batch_counts = Counter(tuple(window) for window in windows.reshape(-1, n))

            with self._lock:
                self.ngram_counters[n].update(batch_counts)
                self._truncate_counter(n)

            t1 = time.perf_counter()
            timing_stats[f"ngram_{n}"] = (t1 - t0) * 1000

        total_time = (time.perf_counter() - total_start) * 1000
        timing_stats["total_time"] = total_time

        # Optional: print timing stats
        # print(f"\nTiming Statistics (ms):")
        # print(f"Total Time: {timing_stats['total_time']:.2f}ms")
        # for n in self.ngram_sizes:
        #     print(f"N-gram {n}: {timing_stats[f'ngram_{n}']:.2f}ms")

    def _scale_frequencies(self, counter: Counter, scale_factor: float) -> Counter:
        """Scale frequencies in counter by factor while maintaining min_freq floor"""
        return Counter(
            {
                ngram: max(int(count * scale_factor), self.min_freq)
                for ngram, count in counter.items()
            }
        )

    def _truncate_counter(self, n: int):
        counter = self.ngram_counters[n]
        max_size = self.ngram_to_size[n] * 2

        if len(counter) > max_size:
            most_common = counter.most_common(max_size)
            self.ngram_counters[n] = Counter(dict(most_common))

    def _maybe_reset_counters(self):
        self.batch_count += 1
        if self.batch_count % self.reset_every == 0:
            with self._lock:
                # Save existing vocabulary
                old_vocabs = {
                    n: set(self.ngram_to_idx_tables[n].keys()) for n in self.ngram_sizes
                }

                # Scale down each counter
                for n in self.ngram_sizes:
                    counter = self.ngram_counters[n]
                    if not counter:
                        continue

                    # Find maximum frequency
                    max_freq = max(counter.values())
                    if max_freq <= self.min_freq:
                        continue

                    # Calculate scale factor to bring max_freq down to target
                    target_max = max(
                        self.min_freq * 10, 100
                    )  # Arbitrary but reasonable target
                    scale_factor = target_max / max_freq

                    # Scale frequencies while preserving min_freq floor
                    self.ngram_counters[n] = self._scale_frequencies(
                        counter, scale_factor
                    )

                    # Ensure vocabulary items are present with at least min_freq
                    for ngram in old_vocabs[n]:
                        if ngram not in self.ngram_counters[n]:
                            self.ngram_counters[n][ngram] = self.min_freq

            if self.debug and random.random() < self.print_frequency:
                self.print_frequency_distribution(num_samples=5)

    def _rebuild_vocabulary(self, n: int):
        counter = self.ngram_counters[n]
        max_size = self.ngram_to_size[n]
        most_common = [
            (ngram, count)
            for ngram, count in counter.most_common()
            if count >= self.min_freq
        ][:max_size]
        new_table = {ngram: idx + 4 for idx, (ngram, _) in enumerate(most_common)}
        with self._lock:
            self.ngram_to_idx_tables[n] = new_table
            self.vocab_sizes[n] = len(new_table) + 4

    def update_from_batch(self, data: np.ndarray):
        """Update statistics and rebuild vocabularies"""
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

    def get_frequency_stats(self) -> Dict[int, List[Tuple[Tuple[int, ...], int]]]:
        """
        Returns frequency statistics for each n-gram size.

        Returns:
            Dict mapping n-gram size to list of (n-gram, frequency) pairs,
            sorted by frequency in descending order
        """
        stats = {}
        with self._lock:  # Thread-safe access
            for n in self.ngram_sizes:
                # Get all n-grams and their frequencies, sorted by frequency
                ngram_freqs = self.ngram_counters[n].most_common()
                stats[n] = ngram_freqs
        return stats

    def print_frequency_distribution(self, num_samples: int = 10):
        """
        Prints a sample of frequency distributions for each n-gram size.

        Args:
            num_samples: Number of samples to print from start and end of distribution
        """
        stats = self.get_frequency_stats()

        for n in self.ngram_sizes:
            freqs = stats[n]
            if not freqs:
                continue

            print(f"\nN-gram size {n}:")
            print(f"Total unique n-grams: {len(freqs)}")

            # Print highest frequencies
            print(f"\nTop {num_samples} frequencies:")
            for ngram, freq in freqs[:num_samples]:
                # Convert numpy values to plain integers
                clean_ngram = tuple(int(x) for x in ngram)
                print(f"N-gram: {clean_ngram}, Frequency: {freq}")

            # Print lowest frequencies (if enough samples exist)
            if len(freqs) > num_samples * 2:
                print(f"\nBottom {num_samples} frequencies:")
                for ngram, freq in freqs[-num_samples:]:
                    # Convert numpy values to plain integers
                    clean_ngram = tuple(int(x) for x in ngram)
                    print(f"N-gram: {clean_ngram}, Frequency: {freq}")

    def get_frequency_distribution_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Returns statistical measures of the frequency distribution for each n-gram size.

        Returns:
            Dict mapping n-gram size to statistics including:
            - max_freq: Maximum frequency
            - min_freq: Minimum frequency
            - median_freq: Median frequency
            - mean_freq: Mean frequency
            - std_freq: Standard deviation of frequencies
        """
        stats = {}
        with self._lock:
            for n in self.ngram_sizes:
                freqs = list(self.ngram_counters[n].values())
                if not freqs:
                    continue

                freqs_array = np.array(freqs)
                stats[n] = {
                    "max_freq": float(np.max(freqs_array)),
                    "min_freq": float(np.min(freqs_array)),
                    "median_freq": float(np.median(freqs_array)),
                    "mean_freq": float(np.mean(freqs_array)),
                    "std_freq": float(np.std(freqs_array)),
                }
        return stats


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
    processor.print_frequency_distribution(num_samples=3)
