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
        validation_dataloader = PraxisDataModule(validation_data, hparams["batch_size"])

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


class PraxisDataSampler:
    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size

    @property
    def can_sample(self):
        return True

    def fill_cache(self):
        AssertionError("This method should be implemented by a child class.")

    def get_batch(
        self, oversample: bool = False, supersample: bool = False
    ) -> torch.Tensor:
        if supersample and oversample:
            raise ValueError("Cannot both oversample and supersample simultaneously.")

        seq_factor = 4 if supersample else (2 if oversample else 1)

        while len(self.token_cache) < seq_factor:
            self.fill_cache()

        batch = torch.cat([self.token_cache.pop(0) for _ in range(seq_factor)], dim=0)
        return batch


class HuggingfaceDataset(PraxisDataSampler):
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
        self.text_cache_size = 100 * self.buffer_size
        self.token_cache = []
        self.shuffled_dataset = self.dataset.shuffle(
            seed=seed, buffer_size=self.buffer_size
        )
        self.dataset_iterator = iter(self.shuffled_dataset)

    def fill_cache(self):
        cache_text = ""
        while len(cache_text) < self.text_cache_size:
            try:
                if len(self.keys) == 3:
                    formats = [
                        ["SYSTEM", "INPUT", "OUTPUT"],
                        ["SYSTEM", "USER", "ASSISTANT"],
                    ]
                    fmt = random.choice(formats)
                elif len(self.keys) == 2:
                    formats = [
                        ["INPUT", "OUTPUT"],
                        ["USER", "ASSISTANT"],
                    ]
                    fmt = random.choice(formats)
                document = next(self.dataset_iterator)
                for i, key in enumerate(self.keys):
                    content = document.get(key)
                    if len(self.keys) == 3:
                        if i % 3 == 0:
                            content = f"\n{fmt[0]}: " + content
                        elif i % 3 == 1:
                            content = f"\n{fmt[1]}: " + content
                        elif i % 3 == 2:
                            content = (
                                f"\n{fmt[2]}: " + content + self.tokenizer.eos_token
                            )
                    elif len(self.keys) == 2:
                        if i % 2 == 0:
                            content = f"\n{fmt[0]}: " + content
                        else:
                            content = (
                                f"\n{fmt[1]}: " + content + self.tokenizer.eos_token
                            )
                    else:
                        content += self.tokenizer.eos_token
                    cache_text += content
            except StopIteration:
                self.dataset_iterator = iter(self.shuffled_dataset)

        tokens = self.tokenizer(
            text=cache_text,
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


class MultiDirectoryDataset(PraxisDataSampler):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, block_size: int, directories: List[str]
    ):
        super().__init__(tokenizer, block_size)
        self.directories = directories
        self.cached_text = ""
        self.token_cache = []
        self.file_list = self._get_file_list()
        self.buffer_size = 10_000
        self.text_cache_size = 10 * self.buffer_size
        random.shuffle(self.file_list)
        self.file_iterator = iter(self.file_list)

    def _get_file_list(self) -> List[str]:
        """Recursively get all files in all directories."""
        file_list = []
        for directory in self.directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_list.append(os.path.join(root, file))
        return file_list

    def _read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def fill_cache(self):
        while len(self.cached_text) < self.text_cache_size:
            try:
                file_path = next(self.file_iterator)
                self.cached_text += (
                    self._read_file(file_path) + self.tokenizer.eos_token
                )
            except StopIteration:
                random.shuffle(self.file_list)
                self.file_iterator = iter(self.file_list)

        tokens = self.tokenizer(
            text=self.cached_text,
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
        self.cached_text = ""


class GunChatDataset(PraxisDataSampler):
    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int):
        super().__init__(tokenizer, block_size)

        from adapters import GunAdapter as Gun

        self.gun = Gun()
        self.token_cache = []
        self._next_batch = []

    @property
    def can_sample(self):
        self._next_batch = self._tokenize_text()
        if len(self._next_batch) < 4:
            return False
        else:
            return True

    def _tokenize_text(self):
        text_list = self.gun.get_sample(250)
        formatted = "\n".join(
            [random.choice(["INPUT: ", "OUTPUT: "]) + entry for entry in text_list]
        )

        tokens = self.tokenizer(
            text=formatted,
            max_length=self.block_size,
            stride=0,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )["input_ids"]

        return tokens

    def fill_cache(self):
        self.token_cache = []
        self.token_cache.extend(
            [batch for batch in self._next_batch if len(batch) == self.block_size]
        )


class WeightedIterableDataset(IterableDataset):
    def __init__(
        self,
        datasets: List[HuggingfaceDataset],
        weights: List[float],
        batch_size: int,
        oversample_chance: float = 0,
        supersample_chance: float = 0,
    ):
        assert len(datasets) == len(
            weights
        ), "Number of datasets and weights must match"
        assert sum(weights) == 1, "Weights must sum to 1"

        self.datasets = datasets
        self.weights = weights
        self.batch_size = batch_size
        self.oversample_chance = oversample_chance
        self.supersample_chance = supersample_chance

    def __iter__(self):
        while True:
            oversample = False
            supersample = False
            rand = random.random()
            current_batch_size = self.batch_size
            if rand < self.supersample_chance:
                if self.batch_size // 16 > 0:
                    supersample = True
                    current_batch_size = self.batch_size // 16
            elif rand < self.oversample_chance:
                if self.batch_size // 4 > 0:
                    oversample = True
                    current_batch_size = self.batch_size // 4

            batch = []

            available_datasets = []
            available_weights = []
            for i, dataset in enumerate(self.datasets):
                if dataset.can_sample:
                    available_datasets.append(dataset)
                    available_weights.append(self.weights[i])

            for _ in range(current_batch_size):
                dataset_index = random.choices(
                    range(len(available_datasets)), weights=available_weights
                )[0]
                item = available_datasets[dataset_index].get_batch(
                    oversample,
                    supersample,
                )
                batch.append(item)

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
            train_datasets, weights, batch_size, oversample_chance, supersample_chance
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
