import os
import random
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

HUGGINGFACE_PROBS = [0, 0, 0, 0, 2.3, 0.666666, 0.333, 0.1]
HUGGINGFACE_DATASETS = [
    dict(path="open-phi/textbooks", keys=["markdown"]),
    dict(
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        keys=["prompt", "text"],
    ),
    dict(
        path="Muennighoff/natural-instructions",
        name="default",
        keys=["definition", "inputs", "targets"],
    ),
    dict(
        path="togethercomputer/RedPajama-Data-V2",
        name="sample-10B",
        snapshots=["2023-14"],
        keys=["raw_content"],
    ),
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", keys=["text"]),
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-100BT", keys=["text"]),
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-350BT", keys=["text"]),
    dict(path="HuggingFaceFW/fineweb", name="default", keys=["text"]),
]


def get_datamodules(
    dev: bool,
    phi: bool,
    instruct: bool,
    gun: bool,
    tokenizer,
    hparams,
    seed,
    data_path,
    *args,
):
    train_data = []
    config = get_dataset_configs(dev, phi, instruct)
    for c in config["primary"]:
        train_data.append(
            get_dataset("huggingface", tokenizer, hparams["block_size"], seed, c, *args)
        )

    if data_path:
        train_data.append(
            get_dataset("directory", tokenizer, hparams["block_size"], seed, *args)
        )

    if gun:
        train_data.append(get_dataset("gun", tokenizer, hparams["block_size"], seed))

    validation_data = []
    if len(config["validation"]) > 0:
        for c in config["validation"]:
            validation_data.append(
                get_dataset(
                    "huggingface", tokenizer, hparams["block_size"], seed, c, *args
                )
            )

    train_dataloader = PraxisDataModule(
        train_data,
        tokenizer,
        hparams["batch_size"],
        hparams["block_size"],
        hparams["oversample_chance"],
        hparams["supersample_chance"],
    )

    validation_dataloader = None
    if len(validation_data) > 0:
        validation_dataloader = PraxisDataModule(
            validation_data, tokenizer, hparams["batch_size"], hparams["block_size"]
        )

    return train_dataloader, validation_dataloader


def get_dataset(format, tokenizer, block_size, seed, *args):
    if format == "huggingface":
        return HuggingfaceDataset(tokenizer, block_size, seed, *args)
    elif format == "directory":
        return MultiDirectoryDataset(tokenizer, block_size, *args)
    elif format == "gun":
        return GunChatDataset(tokenizer, block_size)


def get_dataset_configs(dev: bool, phi: bool, instruct: bool):
    config = {"primary": [], "validation": []}
    config["primary"].append(
        random.choices(HUGGINGFACE_DATASETS, HUGGINGFACE_PROBS, k=1)[0]
    )
    if phi:
        config["primary"].append(HUGGINGFACE_DATASETS[0])
        config["primary"].append(HUGGINGFACE_DATASETS[1])
    if instruct:
        config["primary"].append(HUGGINGFACE_DATASETS[2])
    if dev:
        # Overwrite with simpler dataset
        config["primary"] = [HUGGINGFACE_DATASETS[0]]
    else:
        config["validation"].append(HUGGINGFACE_DATASETS[3])

    return config


class PraxisSampler:
    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sequence_cache = []  # Store raw text sequences
        self.token_cache = []  # Store tokenized batches

    @property
    def can_sample(self):
        return True

    def fill_sequence_cache(self):
        """Each dataset implementation should override this"""
        raise NotImplementedError

    def get_sequences(self, count: int = 1) -> List[str]:
        """Get raw sequences from this dataset"""
        while len(self.sequence_cache) < count:
            self.fill_sequence_cache()
        return [self.sequence_cache.pop(0) for _ in range(count)]

    def get_batch(
        self, oversample: bool = False, supersample: bool = False
    ) -> torch.Tensor:
        if supersample and oversample:
            raise ValueError("Cannot both oversample and supersample simultaneously.")

        seq_factor = 4 if supersample else (2 if oversample else 1)

        while len(self.token_cache) < seq_factor:
            self.fill_token_cache()

        batch = torch.cat([self.token_cache.pop(0) for _ in range(seq_factor)], dim=0)
        return batch


class InterleaveDataManager:
    def __init__(
        self, samplers, weights, tokenizer, block_size, text_cache_size=100_000
    ):
        self.samplers = samplers
        self.weights = weights
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_cache_size = text_cache_size
        self.token_stream = torch.tensor(
            [], dtype=torch.long
        )  # Single continuous stream
        self.debug = True

    def extend_token_stream(self):
        """Add more tokens to our stream when needed"""
        interleaved = self.create_interleaved_sequence()
        tokens = self.tokenizer(
            text=interleaved,
            padding=False,  # No padding needed since we're building a stream
            return_tensors="pt",
        )["input_ids"].squeeze(
            0
        )  # Get flat token sequence
        self.token_stream = torch.cat([self.token_stream, tokens])

    def get_batch(
        self, batch_size: int, oversample: bool = False, supersample: bool = False
    ) -> List[torch.Tensor]:
        sequence_length = self.block_size
        current_batch_size = batch_size

        # Check if batch size supports the requested sampling mode
        if supersample and batch_size >= 16:
            current_batch_size = batch_size // 16
            sequence_length = self.block_size * 4
        elif oversample and batch_size >= 4:
            current_batch_size = batch_size // 4
            sequence_length = self.block_size * 2
        else:
            # If batch size isn't sufficient, fall back to normal sampling
            oversample = False
            supersample = False

        # Calculate how many total tokens we need
        tokens_needed = current_batch_size * sequence_length

        # Make sure we have enough tokens
        while len(self.token_stream) < tokens_needed:
            self.extend_token_stream()

        # Extract batch
        batch = []
        for i in range(current_batch_size):
            start = i * sequence_length
            end = start + sequence_length
            batch.append(self.token_stream[start:end])

        # Remove used tokens from the stream
        self.token_stream = self.token_stream[tokens_needed:]

        if self.debug:
            batch_tensor = torch.stack(batch)
            # print(f"\nBatch shape: {batch_tensor.shape}")

            if random.random() < 0.05:
                first_seq = batch_tensor[0].tolist()
                decoded = self.tokenizer.decode(first_seq, skip_special_tokens=False)
                print("\nSample sequence (with special tokens):")
                print("-" * 50)
                print(decoded)
                print("-" * 50)

        return batch

    def create_interleaved_sequence(self) -> str:
        """Create a single interleaved sequence from multiple samplers"""
        sequence = ""
        while len(sequence) < self.text_cache_size:
            # Pick a sampler based on weights
            sampler = random.choices(self.samplers, weights=self.weights, k=1)[0]
            # Get a sequence from that sampler
            new_sequences = sampler.get_sequences(1)
            sequence += new_sequences[0] + self.tokenizer.eos_token
        return sequence

    def fill_token_cache(self):
        """Fill token cache with interleaved sequences"""
        interleaved = self.create_interleaved_sequence()

        tokens = self.tokenizer(
            text=interleaved,
            max_length=self.block_size,
            stride=0,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )["input_ids"]

        self.token_cache.extend(
            [batch for batch in tokens if len(batch) == self.block_size]
        )


class HuggingfaceDataset(PraxisSampler):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, block_size: int, seed: int, config: Dict
    ):
        super().__init__(tokenizer, block_size)
        self.keys = config.get("keys", ["text"])
        dataset_args = dict(
            path=config.get("path", "HuggingFaceFW/fineweb"),
            split="train",
            streaming=True,
            cache_dir=os.path.join(config.get("cache_dir", "data"), "datasets"),
            trust_remote_code=True,
        )
        if "name" in config:
            dataset_args["name"] = config["name"]
        self.dataset = load_dataset(**dataset_args)
        self.buffer_size = 1_000
        self.shuffled_dataset = self.dataset.shuffle(
            seed=seed, buffer_size=self.buffer_size
        )
        self.dataset_iterator = iter(self.shuffled_dataset)

    def fill_sequence_cache(self):
        try:
            document = next(self.dataset_iterator)
            formatted = self._format_document(document)
            self.sequence_cache.append(formatted)
        except StopIteration:
            self.dataset_iterator = iter(self.shuffled_dataset)
            self.fill_sequence_cache()

    def _format_document(self, document):
        formatted = ""
        for i, key in enumerate(self.keys):
            content = document.get(key)
            if len(self.keys) == 3:
                formats = [
                    ["SYSTEM", "INPUT", "OUTPUT"],
                    ["SYSTEM", "USER", "ASSISTANT"],
                ]
                fmt = random.choice(formats)
                formatted += f"\n{fmt[i]}: {content}"
            elif len(self.keys) == 2:
                formats = [["INPUT", "OUTPUT"], ["USER", "ASSISTANT"]]
                fmt = random.choice(formats)
                formatted += f"\n{fmt[i%2]}: {content}"
            else:
                formatted += content
        return formatted


class MultiDirectoryDataset(PraxisSampler):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, block_size: int, directories: List[str]
    ):
        super().__init__(tokenizer, block_size)
        self.directories = directories
        self.file_list = self._get_file_list()
        random.shuffle(self.file_list)
        self.file_iterator = iter(self.file_list)

    def fill_sequence_cache(self):
        try:
            file_path = next(self.file_iterator)
            content = self._read_file(file_path)
            self.sequence_cache.append(content)
        except StopIteration:
            random.shuffle(self.file_list)
            self.file_iterator = iter(self.file_list)
            self.fill_sequence_cache()


class GunChatDataset(PraxisSampler):
    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int):
        super().__init__(tokenizer, block_size)
        from adapters import GunAdapter as Gun

        self.gun = Gun()

    def fill_sequence_cache(self):
        text_list = self.gun.get_sample(250)
        for text in text_list:
            formatted = random.choice(["INPUT: ", "OUTPUT: "]) + text
            self.sequence_cache.append(formatted)


class WeightedIterableDataset(IterableDataset):
    def __init__(
        self,
        datasets: List[PraxisSampler],
        weights: List[float],
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        batch_size: int,
        oversample_chance: float = 0,
        supersample_chance: float = 0,
    ):
        self.data_manager = InterleaveDataManager(
            datasets, weights, tokenizer, block_size
        )
        self.batch_size = batch_size
        self.oversample_chance = oversample_chance
        self.supersample_chance = supersample_chance

    def __iter__(self):
        while True:
            oversample = random.random() < self.oversample_chance
            supersample = random.random() < self.supersample_chance

            batch = self.data_manager.get_batch(
                self.batch_size, oversample, supersample
            )
            yield torch.stack(batch)


class PraxisDataModule(LightningDataModule):
    def __init__(
        self,
        train_datasets: List[Dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        block_size: int = 512,
        oversample_chance: float = 0,
        supersample_chance: float = 0,
    ):
        super().__init__()

        weights = []
        # TODO: This is awful
        if len(train_datasets) == 1:
            weights.append(1.0)
        elif len(train_datasets) == 2:
            weights.extend([0.9, 0.1])
        elif len(train_datasets) == 3:
            weights.extend([0.8, 0.1, 0.1])
        elif len(train_datasets) == 4:
            weights.extend([0.79, 0.1, 0.1, 0.01])
        elif len(train_datasets) >= 5:
            weights.extend([0.78, 0.1, 0.1, 0.01, 0.01])

        self.weighted_dataset = WeightedIterableDataset(
            train_datasets,
            weights,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.weighted_dataset,
            batch_size=None,
            num_workers=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.weighted_dataset,
            batch_size=None,
            num_workers=1,
            pin_memory=True,
        )
