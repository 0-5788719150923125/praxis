"""Reinforcement learning data formatting."""

import json
import torch
from collections import defaultdict
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS
from praxis.data.formatters.base import text_formatter


class RLLogger:
    """Centralized logging for RL training metrics."""

    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(int))
        self.batch_count = 0
        self.last_log_batch = 0
        self.log_interval = 50  # Log every N batches

    def log_batch(self, rewards, source="unknown"):
        """Log statistics for a batch of rewards."""
        self.batch_count += 1

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.tolist()

        # Update statistics
        self.stats["total"]["sequences"] += len(rewards)
        non_zero = [
            r for r in rewards if r != 0
        ]  # Include both positive rewards and -1 generation flags
        self.stats["total"]["rl_sequences"] += len(non_zero)

        # Handle positive rewards separately for statistics (exclude -1 generation flags)
        positive_rewards = [r for r in rewards if r > 0]
        if positive_rewards:
            self.stats["rewards"]["count"] += len(positive_rewards)
            self.stats["rewards"]["sum"] += sum(positive_rewards)
            if "min" not in self.stats["rewards"]:
                self.stats["rewards"]["min"] = min(positive_rewards)
            else:
                self.stats["rewards"]["min"] = min(
                    self.stats["rewards"]["min"], min(positive_rewards)
                )

            if "max" not in self.stats["rewards"]:
                self.stats["rewards"]["max"] = max(positive_rewards)
            else:
                self.stats["rewards"]["max"] = max(
                    self.stats["rewards"]["max"], max(positive_rewards)
                )

            # Track reward distribution for positive rewards only
            for r in positive_rewards:
                bucket = f"{int(r * 10) / 10:.1f}"
                self.stats["distribution"][bucket] += 1

        # Count generation flags separately
        generation_flags = [r for r in rewards if r == -1]
        if generation_flags:
            self.stats["generation_flags"]["count"] += len(generation_flags)

        # Log periodically
        if self.batch_count - self.last_log_batch >= self.log_interval:
            self._print_summary()
            self.last_log_batch = self.batch_count

    def log_dataset_sample(self, dataset_name, has_reward):
        """Log when a dataset is sampled."""
        self.stats["dataset_samples"][dataset_name] += 1
        if has_reward:
            self.stats["dataset_rl_samples"][dataset_name] += 1

    def log_reward_found(self, reward, dataset_name):
        """Log when a reward is found during sequence creation."""
        self.stats["rewards_by_dataset"][dataset_name] += 1
        if "reward_values" not in self.stats:
            self.stats["reward_values"] = defaultdict(list)
        self.stats["reward_values"][dataset_name].append(reward)

    def _print_summary(self):
        """Print a summary of RL statistics."""
        total_seq = self.stats["total"]["sequences"]
        rl_seq = self.stats["total"]["rl_sequences"]

        if total_seq == 0:
            return

        print(f"\n[RL Stats] After {self.batch_count} batches:")
        print(f"  Total sequences: {total_seq:,}")
        print(f"  RL sequences: {rl_seq:,} ({100.0 * rl_seq / total_seq:.1f}%)")

        # Show generation flags
        gen_flags = self.stats["generation_flags"].get("count", 0)
        if gen_flags > 0:
            print(f"  Generation flags: {gen_flags:,} sequences awaiting generation")

        if self.stats["rewards"]["count"] > 0:
            avg_reward = self.stats["rewards"]["sum"] / self.stats["rewards"]["count"]
            print(
                f"  Rewards: avg={avg_reward:.3f}, min={self.stats['rewards']['min']:.3f}, max={self.stats['rewards']['max']:.3f}"
            )

            # Show reward distribution
            if self.stats["distribution"]:
                print("  Distribution:")
                for bucket in sorted(self.stats["distribution"].keys()):
                    count = self.stats["distribution"][bucket]
                    pct = 100.0 * count / self.stats["rewards"]["count"]
                    print(f"    [{bucket}]: {count:4d} ({pct:5.1f}%)")

        # Show dataset sampling
        if self.stats["dataset_samples"]:
            print("  Dataset sampling:")
            for dataset, count in self.stats["dataset_samples"].items():
                rl_count = self.stats["dataset_rl_samples"].get(dataset, 0)
                print(f"    {dataset}: {count:,} samples, {rl_count:,} with rewards")

        print()


# Global RL logger instance
_rl_logger = RLLogger()


def format_rl(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Format RL dataset for generation-based reinforcement learning.

    For proper RL, we format the prompt for generation and store metadata
    for evaluation. The actual response will be generated during training.

    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document (must be exactly 3)
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Dict with special formatting for RL including text, reward, and metadata
    """
    assert len(keys) == 3, "RL format requires exactly 3 keys"

    prompt = text_formatter(document.get(keys[0], ""))
    verification_info = document.get(keys[1], "{}") 
    solve_rate = document.get(keys[2], 0.0)

    # Parse the ground truth from verification_info
    try:
        verification_data = json.loads(verification_info)
        ground_truth = verification_data.get("ground_truth", "")
    except:
        ground_truth = ""

    # Format with unified system/developer prompts for RL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["answer_question"]},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template with generation prompt
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Adds the assistant prefix
    )

    # Return special format for RL - ALWAYS use -1 for generation
    return {
        "text": prompt_text,
        "reward": -1.0,  # Special flag indicating this needs generation
        "ground_truth": ground_truth,
        "original_difficulty": solve_rate,
    }