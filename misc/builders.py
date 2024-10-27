import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional

import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer


@dataclass
class BatchConfig:
    """Configuration for batch sampling behavior"""

    base_batch_size: int
    target_batch_size: int
    base_sequence_length: int

    # Cache configuration
    buffer_size: int = 100  # Number of texts to accumulate before tokenizing
    cache_size: int = 1000  # Number of tokenized sequences to keep in cache

    # Variation probabilities
    # Format: (length_mult, batch_div, probability)
    variation_config: List[tuple] = (
        (1.0, 1, 0.90),  # Normal: 1x length, 1x batch
        (2.0, 4, 0.09),  # Medium: 2x length, 1/4 batch
        (4.0, 16, 0.01),  # Large: 4x length, 1/16 batch
    )

    def get_variation(self) -> tuple:
        """Get length multiplier and batch divisor based on probabilities"""
        rand = random.random()
        cumsum = 0
        for length_mult, batch_div, prob in self.variation_config:
            cumsum += prob
            if rand < cumsum:
                return length_mult, batch_div
        return self.variation_config[0][:2]  # Default to normal config


class DataFormat(Enum):
    """Supported data formatting strategies"""

    SIMPLE = "simple"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    QA = "qa"
    CUSTOM = "custom"


class DataFormatter:
    """Data formatting utilities"""

    @staticmethod
    def format_text(
        format_type: DataFormat,
        sample: Dict[str, str],
        keys: List[str],
        tokenizer: PreTrainedTokenizer,
        custom_formatter=None,
    ) -> str:
        if format_type == DataFormat.SIMPLE:
            return f"{sample[keys[0]]}{tokenizer.eos_token}"

        if format_type == DataFormat.INSTRUCTION:
            instruction = sample.get(keys[0], "")
            inputs = sample.get(keys[1], "")
            output = sample.get(keys[2], "")
            return f"Instruction: {instruction}\nInput: {inputs}\nOutput: {output}{tokenizer.eos_token}"

        if format_type == DataFormat.CONVERSATION:
            system = sample.get(keys[0], "")
            user = sample.get(keys[1], "")
            assistant = sample.get(keys[2], "")
            return f"System: {system}\nUser: {user}\nAssistant: {assistant}{tokenizer.eos_token}"

        if format_type == DataFormat.QA:
            question = sample.get(keys[0], "")
            answer = sample.get(keys[1], "")
            return f"Question: {question}\nAnswer: {answer}{tokenizer.eos_token}"

        if format_type == DataFormat.CUSTOM and custom_formatter:
            return custom_formatter(sample, keys, tokenizer)

        raise ValueError(f"Unsupported format type: {format_type}")


class Dataset(IterableDataset):
    """Base dataset implementation"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_config: BatchConfig,
        format_type: DataFormat,
        keys: List[str],
        custom_formatter=None,
    ):
        self.tokenizer = tokenizer
        self.batch_config = batch_config
        self.format_type = format_type
        self.keys = keys
        self.custom_formatter = custom_formatter
        self.token_cache = []

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield batches from the token cache"""
        while True:
            # Get the current variation config
            length_mult, _ = self.batch_config.get_variation()
            sequence_length = int(self.batch_config.base_sequence_length * length_mult)

            # Get next batch
            try:
                yield self.get_next_batch(sequence_length)
            except RuntimeError:
                # If cache is empty and can't be filled, try again
                continue

    def get_formatted_text(self, sample: Dict[str, str]) -> str:
        """Format a sample according to the specified format type"""
        return DataFormatter.format_text(
            self.format_type, sample, self.keys, self.tokenizer, self.custom_formatter
        )

    def fill_cache(self):
        """Fill token cache - implement in subclasses"""
        raise NotImplementedError

    def get_next_batch(self, sequence_length: int) -> torch.Tensor:
        """Get next batch from cache, refilling if needed"""
        if len(self.token_cache) == 0:
            self.fill_cache()

        if len(self.token_cache) == 0:
            raise RuntimeError("Failed to fill token cache")

        # Get and truncate/pad to desired sequence length
        tokens = self.token_cache.pop(0)
        return self.tokenizer.pad(
            {"input_ids": tokens},
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]


class HuggingFaceDataset(Dataset):
    """Dataset implementation for HuggingFace datasets"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_config: BatchConfig,
        dataset_path: str,
        format_type: DataFormat,
        keys: List[str],
        dataset_name: Optional[str] = None,
        custom_formatter=None,
    ):
        super().__init__(tokenizer, batch_config, format_type, keys, custom_formatter)

        dataset_args = dict(
            path=dataset_path, split="train", streaming=True, trust_remote_code=True
        )
        if dataset_name:
            dataset_args["name"] = dataset_name

        self.dataset = load_dataset(**dataset_args)
        self.shuffled_dataset = self.dataset.shuffle(seed=42, buffer_size=1000)
        self.iterator = iter(self.shuffled_dataset)
        self.token_cache = []

    def fill_cache(self):
        """Fill cache with tokenized sequences"""
        # Only fill if cache is nearly empty
        if len(self.token_cache) > self.batch_config.cache_size // 4:
            return

        # Collect a large batch of samples at once
        samples = []
        try:
            for _ in range(1000):  # Process 1000 samples at once
                samples.append(next(self.iterator))
        except StopIteration:
            self.iterator = iter(self.shuffled_dataset)
            if not samples:
                return

        # Format all samples at once
        texts = [self.get_formatted_text(sample) for sample in samples]

        # Tokenize all texts in one batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.batch_config.base_sequence_length * 4,
            return_tensors="pt",
        )

        # Add to cache
        self.token_cache.extend(encodings["input_ids"])

        # Maintain maximum cache size but keep plenty of tokens available
        if len(self.token_cache) > self.batch_config.cache_size:
            self.token_cache = self.token_cache[: self.batch_config.cache_size]

    def get_next_batch(self, sequence_length: int) -> torch.Tensor:
        """Get next batch from cache, refilling if needed"""
        if len(self.token_cache) < self.batch_config.base_batch_size:
            self.fill_cache()

        if len(self.token_cache) == 0:
            raise RuntimeError("Failed to fill token cache")

        # Get and resize to desired sequence length
        tokens = self.token_cache.pop(0)

        # Handle sequence length adjustment
        if tokens.size(0) > sequence_length:
            tokens = tokens[:sequence_length]
        elif tokens.size(0) < sequence_length:
            padding = torch.full(
                (sequence_length - tokens.size(0),),
                self.tokenizer.pad_token_id,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat([tokens, padding])

        return tokens.unsqueeze(0)


class BatchSampler(IterableDataset):
    """Handles batching across multiple datasets"""

    def __init__(
        self, datasets: List[Dataset], weights: List[float], batch_config: BatchConfig
    ):
        self.datasets = datasets
        self.weights = weights
        self.batch_config = batch_config

    def get_batch(self) -> torch.Tensor:
        """Get next batch of tokens"""
        # Get variation config for this batch
        length_mult, batch_div = self.batch_config.get_variation()
        sequence_length = int(self.batch_config.base_sequence_length * length_mult)
        batch_size = max(1, self.batch_config.base_batch_size // batch_div)

        # Sample from datasets
        batch = []
        for _ in range(batch_size):
            dataset_idx = random.choices(
                range(len(self.datasets)), weights=self.weights
            )[0]
            tokens = self.datasets[dataset_idx].get_next_batch(sequence_length)
            batch.append(tokens)

        return torch.cat(batch, dim=0)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield self.get_batch()


class DataModule(LightningDataModule):
    """PyTorch Lightning data module"""

    def __init__(
        self,
        datasets: List[Dataset],
        batch_config: BatchConfig,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_config = batch_config

        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        self.weights = weights

        self.sampler = BatchSampler(datasets, weights, batch_config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.sampler, batch_size=None, num_workers=1, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.train_dataloader()


def test_datasets():
    """Comprehensive dataset tests"""
    tokenizer = AutoTokenizer.from_pretrained("UNSAFE/praxis-8192")

    # Test configurations
    configs = [
        # Standard config
        BatchConfig(
            base_batch_size=16,
            target_batch_size=64,
            base_sequence_length=128,
            cache_size=2000,  # Increased cache size
            variation_config=[(1.0, 1, 1.0)],  # Only use normal length/batch
        ),
        # Small batches
        BatchConfig(
            base_batch_size=4,
            target_batch_size=16,
            base_sequence_length=128,
            cache_size=2000,
            variation_config=[(1.0, 1, 1.0)],  # Only use normal length/batch
        ),
        # Variable length and batch size
        BatchConfig(
            base_batch_size=16,
            target_batch_size=64,
            base_sequence_length=128,
            cache_size=2000,
            variation_config=[
                (1.0, 1, 0.90),  # Normal: 1x length, 1x batch
                (2.0, 4, 0.09),  # Medium: 2x length, 1/4 batch
                (4.0, 16, 0.01),  # Large:  4x length, 1/16 batch
            ],
        ),
    ]

    for config in configs:
        print(f"\nTesting config: {config}")

        # Create datasets
        datasets = [
            HuggingFaceDataset(
                tokenizer=tokenizer,
                batch_config=config,
                dataset_path="HuggingFaceTB/smollm-corpus",
                dataset_name="cosmopedia-v2",
                format_type=DataFormat.SIMPLE,
                keys=["text"],
            ),
            HuggingFaceDataset(
                tokenizer=tokenizer,
                batch_config=config,
                dataset_path="Muennighoff/natural-instructions",
                format_type=DataFormat.INSTRUCTION,
                keys=["definition", "inputs", "targets"],
            ),
        ]

        # Create data module
        data_module = DataModule(
            datasets=datasets, batch_config=config, weights=[0.7, 0.3]
        )

        # Test batches
        dataloader = data_module.train_dataloader()

        # Get multiple batches to verify distribution
        batches = [next(iter(dataloader)) for _ in range(100)]

        # Count distribution of batch sizes
        size_counts = {}
        for batch in batches:
            size = batch.size(0)
            size_counts[size] = size_counts.get(size, 0) + 1

        print(f"Batch size distribution: {size_counts}")

        # Verify some basic properties
        for batch in batches:
            assert batch.dim() == 2, f"Expected 2D tensor, got {batch.dim()}D"
            assert batch.size(0) > 0, "Batch size should be positive"
            assert batch.size(1) > 0, "Sequence length should be positive"


if __name__ == "__main__":
    test_datasets()
