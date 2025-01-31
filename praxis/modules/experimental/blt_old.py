import random
import threading
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
        debug: bool = False,
    ):
        """
        Initialize processor with vocabulary size limits per ngram length

        Args:
            ngram_to_size: Dict mapping ngram length to max vocab size
            min_freq: Minimum frequency required to include ngram in vocab
        """

    def __init__(
        self,
        ngram_to_size: Dict[int, int],
        min_freq: int = 1,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.print_frequency = 0.002
        self.device = torch.device("cpu")
        self.ngram_to_size = ngram_to_size
        self.min_freq = min_freq
        self.ngram_sizes = sorted(ngram_to_size.keys())
        assert min(self.ngram_sizes) >= 2, "Minimum ngram size must be 2"

        # Just maintain counters - no vocabulary tables needed
        self.ngram_counters = {n: Counter() for n in self.ngram_sizes}
        self._lock = threading.Lock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        ngram_tensors = self.encode_token_ngrams(x)
        return self.stack_ngrams(ngram_tensors)

    def _update_frequencies(self, data: Union[np.ndarray, torch.Tensor]):
        data_np = self._to_numpy(data)
        batch_size, seq_len = data_np.shape

        max_n = max(self.ngram_sizes)
        padded = np.pad(
            data_np,
            ((0, 0), (max_n - 1, 0)),
            mode="constant",
            constant_values=0,
        )

        for n in self.ngram_sizes:
            windows = np.lib.stride_tricks.as_strided(
                padded[:, max_n - n :],
                shape=(batch_size, seq_len, n),
                strides=padded.strides[:2] + (padded.strides[1],),
            )

            batch_counts = Counter(tuple(window) for window in windows.reshape(-1, n))

            with self._lock:
                self.ngram_counters[n].update(batch_counts)

    def _get_top_ngrams(self, n: int) -> Dict[Tuple[int, ...], int]:
        """Get vocabulary mapping for top ngrams by frequency"""
        counter = self.ngram_counters[n]
        max_size = self.ngram_to_size[n]

        # Get most common ngrams that meet minimum frequency
        most_common = [
            (ngram, count)
            for ngram, count in counter.most_common()
            if count >= self.min_freq
        ][:max_size]

        # Create mapping starting at index 4
        return {ngram: idx + 4 for idx, (ngram, _) in enumerate(most_common)}

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

        # Get current top ngrams
        lookup_table = self._get_top_ngrams(n)
        ids_np = np.array(
            [
                [lookup_table.get(tuple(window), 0) for window in batch]
                for batch in windows
            ]
        )

        return self._to_torch(ids_np)

    def update_from_batch(self, data: np.ndarray):
        """Just update frequency counts"""
        self._update_frequencies(data)
        if self.debug and random.random() < self.print_frequency:
            self.print_frequency_distribution(num_samples=3)

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
        Prints frequency distributions showing:
        - Top frequencies
        - Bottom frequencies within vocabulary size
        - Bottom frequencies in entire distribution

        Args:
            num_samples: Number of samples to print from each section
        """
        for n in self.ngram_sizes:
            counter = self.ngram_counters[n]
            if not counter:
                continue

            max_size = self.ngram_to_size[n]
            total_ngrams = len(counter)

            # Get all ngrams that meet minimum frequency
            valid_ngrams = [
                (ngram, count)
                for ngram, count in counter.most_common()
                if count >= self.min_freq
            ]
            valid_size = len(valid_ngrams)

            print(f"\nN-gram size {n}:")
            print(f"Total unique n-grams: {total_ngrams}")
            print(f"Valid n-grams (freq >= {self.min_freq}): {valid_size}")
            print(f"Vocabulary size limit: {max_size}")

            # Print top frequencies
            print(f"\nTop {num_samples} frequencies:")
            for ngram, freq in valid_ngrams[:num_samples]:
                clean_ngram = tuple(int(x) for x in ngram)
                print(f"N-gram: {clean_ngram}, Frequency: {freq}")

            # Print bottom frequencies within vocabulary size
            if max_size > num_samples * 2:
                print(f"\nBottom {num_samples} frequencies within vocabulary size:")
                vocab_bottom = valid_ngrams[
                    max(0, min(max_size, valid_size) - num_samples) : min(
                        max_size, valid_size
                    )
                ]
                for ngram, freq in vocab_bottom:
                    clean_ngram = tuple(int(x) for x in ngram)
                    print(f"N-gram: {clean_ngram}, Frequency: {freq}")

            # Print bottom frequencies from entire distribution
            if total_ngrams > num_samples * 2:
                print(f"\nBottom {num_samples} frequencies in entire distribution:")
                distribution_bottom = counter.most_common()[: -num_samples - 1 : -1]
                for ngram, freq in distribution_bottom:
                    clean_ngram = tuple(int(x) for x in ngram)
                    print(f"N-gram: {clean_ngram}, Frequency: {freq}")


# if __name__ == "__main__":
#     # Initialize with same vocabulary size constraints
#     ngram_to_size = {
#         2: 38396,
#         3: 50000,
#         4: 50000,
#         5: 50000,
#         6: 50000,
#         7: 50000,
#         8: 50000,
#     }

#     # Test both CPU and GPU
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
#     input_data = torch.tensor(torch.randn(4, 64)).to(device)

#     processor = RealtimeNgramProcessor(ngram_to_size)
#     processor.update_from_batch(input_data)
#     ngram_ids = processor(input_data[0].unsqueeze(0))
#     print(ngram_ids)
#     print(f"Input device: {input_data.device}")
#     print(f"Output device: {ngram_ids.device}")
#     processor.print_frequency_distribution(num_samples=3)
import time
from statistics import mean, stdev


def benchmark_forward(
    processor, batch_sizes=[1, 4, 16, 64], seq_lengths=[64, 256, 1024], num_runs=100
):
    results = {}

    print("\nBenchmarking Forward Method:")
    print("-" * 50)
    print(f"{'Batch Size':>10} {'Seq Length':>12} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("-" * 50)

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            times = []

            # Warm-up run
            input_data = torch.randint(0, 256, (batch_size, seq_len)).to(
                processor.device
            )
            _ = processor(input_data)

            # Timed runs
            for _ in range(num_runs):
                input_data = torch.randint(0, 256, (batch_size, seq_len)).to(
                    processor.device
                )

                start_time = time.perf_counter()
                _ = processor(input_data)
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)  # Convert to milliseconds

            mean_time = mean(times)
            std_time = stdev(times)

            print(
                f"{batch_size:>10} {seq_len:>12} {mean_time:>12.2f} {std_time:>12.2f}"
            )

            results[(batch_size, seq_len)] = {"mean": mean_time, "std": std_time}

    return results


if __name__ == "__main__":
    # Initialize processor
    ngram_to_size = {
        2: 38396,
        3: 50000,
        4: 50000,
        5: 50000,
        6: 50000,
        7: 50000,
        8: 50000,
    }

    device = torch.device("cpu")
    processor = RealtimeNgramProcessor(ngram_to_size)

    # Run benchmarks
    results = benchmark_forward(processor)
