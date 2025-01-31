import random
import threading
import time
from collections import Counter
from statistics import mean, stdev
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RealtimeNgramProcessor(nn.Module):
    def __init__(self, ngram_to_size: Dict[int, int]):
        super().__init__()
        self.ngram_to_size = ngram_to_size
        self.ngram_sizes = sorted(ngram_to_size.keys())
        assert min(self.ngram_sizes) >= 2, "Minimum ngram size must be 2"

        # Pre-compute prime multipliers
        self.prime = 31
        self.prime_powers_tensor = {
            n: torch.tensor([self.prime**i for i in range(n)], dtype=torch.int64)
            for n in self.ngram_sizes
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is int64 for large number operations
        x = x.to(torch.int64)

        # Process each n-gram size
        ngram_tensors = [self._compute_ngram_hashes(x, n) for n in self.ngram_sizes]

        return torch.stack(ngram_tensors, dim=0)

    def _compute_ngram_hashes(self, data: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, seq_len = data.shape

        # Ensure prime powers are on same device as input
        if self.prime_powers_tensor[n].device != data.device:
            self.prime_powers_tensor[n] = self.prime_powers_tensor[n].to(data.device)

        # Pad the input
        padded = F.pad(data, (n - 1, 0), mode="constant", value=0)

        # Create windows using unfold
        windows = padded.unfold(dimension=1, size=n, step=1)

        # Multiply by prime powers and sum
        prime_powers = self.prime_powers_tensor[n].view(1, 1, n)
        products = windows * prime_powers
        sums = torch.sum(products, dim=2)

        # Apply modulo and offset
        vocab_size = self.ngram_to_size[n]
        hashed_values = (sums % (vocab_size - 4)) + 4

        return hashed_values

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _to_torch(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).to(self.device)


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
            input_data = torch.randint(0, 256, (batch_size, seq_len))
            _ = processor(input_data)

            # Timed runs
            for _ in range(num_runs):
                input_data = torch.randint(0, 256, (batch_size, seq_len))

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

    processor = RealtimeNgramProcessor(ngram_to_size)

    input_data = torch.tensor(torch.randn(4, 64))
    ngram_ids = processor(input_data[0].unsqueeze(0))
    print(ngram_ids)

    # Run benchmarks
    results = benchmark_forward(processor)
