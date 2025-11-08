"""Weighted iterable dataset implementation."""

import random
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler
from praxis.data.datasets.manager import InterleaveDataManager
from praxis.data.formatters import _rl_logger


class WeightedIterableDataset(IterableDataset):
    """Dataset that samples from multiple datasets with weights."""

    def __init__(
        self,
        datasets: List[PraxisSampler],
        weights: List[float],
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        batch_size: int,
        oversample_chance: float = 0,
        supersample_chance: float = 0,
        hypersample_chance: float = 0,
        rl_type: Optional[str] = None,
        run_dir: Optional[str] = None,
        data_metrics_log_interval: int = 50,
        enable_chat_validation: bool = True,
        strict_chat_validation: bool = False,
    ):
        # Always use the new message queue system
        self.data_manager = InterleaveDataManager(
            datasets,
            weights,
            tokenizer,
            block_size,
            rl_type=rl_type,
            run_dir=run_dir,
            data_metrics_log_interval=data_metrics_log_interval,
            enable_chat_validation=enable_chat_validation,
            strict_chat_validation=strict_chat_validation,
        )

        self.batch_size = batch_size
        self.oversample_chance = oversample_chance
        self.supersample_chance = supersample_chance
        self.hypersample_chance = hypersample_chance
        self.rl_type = rl_type
        self.tokenizer = tokenizer  # Store tokenizer for reward extraction
        self._first_batch = True  # Track if this is the first batch

    def __iter__(self):
        while True:
            # Force first batch to use maximum sequence length to trigger
            # torch.compile for worst-case scenario, avoiding lazy recompilation
            if self._first_batch:
                self._first_batch = False
                # Use maximum available sampling mode based on batch size
                if self.batch_size >= 64:
                    oversample, supersample, hypersample = False, False, True
                    print(
                        f"[DATA] First batch: forcing hypersample (8x) for batch_size={self.batch_size}"
                    )
                elif self.batch_size >= 16:
                    oversample, supersample, hypersample = False, True, False
                    print(
                        f"[DATA] First batch: forcing supersample (4x) for batch_size={self.batch_size}"
                    )
                elif self.batch_size >= 4:
                    oversample, supersample, hypersample = True, False, False
                    print(
                        f"[DATA] First batch: forcing oversample (2x) for batch_size={self.batch_size}"
                    )
                else:
                    # Batch size too small for any sampling mode
                    oversample, supersample, hypersample = False, False, False
                    print(
                        f"[DATA] First batch: batch_size={self.batch_size} too small for oversampling"
                    )
            else:
                # Normal random sampling after first batch
                oversample = random.random() < self.oversample_chance
                supersample = random.random() < self.supersample_chance
                hypersample = random.random() < self.hypersample_chance

            result = self.data_manager.get_batch(
                self.batch_size, oversample, supersample, hypersample
            )

            # Extract batch and rewards
            batch = result["batch"]
            rewards = result.get("rewards")
            metadata = result.get("metadata", [])
            token_weights = result.get("token_weights")
            sampler_weights = result.get("sampler_weights")  # Get the current weights

            # Stack batch tensors
            batch_tensor = torch.stack(batch)

            # Handle rewards if RL is enabled
            if self.rl_type and rewards:
                # Convert rewards to tensor
                reward_tensor = torch.tensor(rewards, dtype=torch.float32)

                # Check if this batch needs generation (rewards == -1)
                needs_generation = (reward_tensor == -1).any()
                generation_count = (reward_tensor == -1).sum().item()

                # Only log when we actually have generation flags
                if needs_generation:
                    # Return special format for generation with proper metadata
                    result_dict = {
                        "input_ids": batch_tensor,
                        "rewards": reward_tensor,
                        "needs_generation": True,
                        "metadata": metadata,  # Now properly tracked from data manager
                    }
                    if token_weights is not None:
                        result_dict["token_weights"] = torch.stack(token_weights)
                    if sampler_weights is not None:
                        result_dict["sampler_weights"] = sampler_weights
                    yield result_dict
                else:
                    # Log batch statistics
                    _rl_logger.log_batch(reward_tensor)

                    # Return regular RL format
                    result_dict = {"input_ids": batch_tensor, "rewards": reward_tensor}
                    if token_weights is not None:
                        result_dict["token_weights"] = torch.stack(token_weights)
                    if sampler_weights is not None:
                        result_dict["sampler_weights"] = sampler_weights
                    yield result_dict
            else:
                # No reinforcement learning
                # If we have sampler weights, return dict format
                if sampler_weights is not None:
                    yield {
                        "input_ids": batch_tensor,
                        "sampler_weights": sampler_weights,
                    }
                else:
                    # Return regular tensor for backward compatibility
                    yield batch_tensor
                    yield batch_tensor
