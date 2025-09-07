"""Seed utilities for reproducibility."""

import random
import os
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: int, workers: bool = False) -> int:
    """Set seed for all random number generators.

    Args:
        seed: The seed value to use
        workers: Whether to also seed worker processes

    Returns:
        The seed value used
    """
    # Python built-in
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic algorithms
    torch.use_deterministic_algorithms(
        False
    )  # Some ops don't have deterministic implementations

    # Worker seed
    if workers:
        # This will be used by DataLoader workers
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Store for later use
        torch.utils.data.dataloader._worker_init_fn = seed_worker

    # Environment variables for better reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


def reset_seed(seed: Optional[int] = None) -> None:
    """Reset the random seed.

    Args:
        seed: Optional seed value. If None, uses current time.
    """
    if seed is None:
        import time

        seed = int(time.time() * 1000) % 2**32

    seed_everything(seed)
