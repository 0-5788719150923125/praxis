import os
import random
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

debug = False


class DataFormat(Enum):
    SIMPLE = "simple"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PERSONACHAT = "persona_chat"
    CUSTOM = "custom"


HUGGINGFACE_PROBS = [0, 0, 0, 0, 0, 2.3, 0.666666, 0.333, 0.1]
HUGGINGFACE_DATASETS = [
    dict(
        path="open-phi/textbooks",
        keys=["markdown"],
        format=DataFormat.SIMPLE,
        weight=0.001,
    ),
    dict(
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        keys=["prompt", "text"],
        format=DataFormat.INSTRUCTION,
        weight=0.01,
    ),
    dict(
        path="Muennighoff/natural-instructions",
        name="default",
        keys=["definition", "inputs", "targets"],
        format=DataFormat.CONVERSATION,
        weight=0.01,
    ),
    dict(
        path="google/Synthetic-Persona-Chat",
        keys=["user 1 personas", "user 2 personas", "Best Generated Conversation"],
        format=DataFormat.PERSONACHAT,
        weight=0.01,
    ),
    dict(
        path="togethercomputer/RedPajama-Data-V2",
        name="sample-10B",
        snapshots=["2023-14"],
        keys=["raw_content"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-350BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    dict(
        path="HuggingFaceFW/fineweb",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
]


def format_simple(document: Dict, keys: List[str]) -> str:
    """Just concatenate content with spaces"""
    return " ".join(document.get(key, "") for key in keys)


def format_instruction(document: Dict, keys: List[str]) -> str:
    """Format as instruction/output pairs in ChatML format."""
    assert len(keys) == 2, "Instruction format requires exactly 2 keys"
    instruction = document.get(keys[0], "")
    output = document.get(keys[1], "")
    return (
        f"<|im_start|>user\n{instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{output}\n<|im_end|>\n"
    )


def format_conversation(document: Dict, keys: List[str]) -> str:
    """Format as a conversation in ChatML format."""
    assert len(keys) == 3, "Conversation format requires exactly 3 keys"
    parts = []
    for i, key in enumerate(keys):
        if i == 0:
            role = "system"
        elif i == 1:
            role = "user"
        elif i == 2:
            role = "assistant"
        message = document.get(key, "")
        parts.append(f"<|im_start|>{role}\n{message}\n<|im_end|>\n")
    return "".join(parts)


def format_personachat(document: Dict, keys: List[str]) -> str:
    """Format persona chat conversations into ChatML format."""
    # Extract personas
    user_personas = document.get("user 1 personas", "").split("\n")
    assistant_personas = document.get("user 2 personas", "").split("\n")
    conversation = document.get("Best Generated Conversation", "").split("\n")

    # Include personas in system message
    system_message = ""
    if user_personas:
        system_message += "".join(
            f"- {p.strip()}\n" for p in user_personas if p.strip()
        )
    if assistant_personas:
        system_message += "".join(
            f"- {p.strip()}\n" for p in assistant_personas if p.strip()
        )

    # Initialize the formatted text with system message
    formatted = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n"

    # Map speaker labels to ChatML roles
    speaker_map = {
        "A": "user",
        "B": "assistant",
        "USER 1": "user",
        "USER1": "user",
        "USER 2": "assistant",
        "USER2": "assistant",
    }

    # Format the conversation using "user" and "assistant"
    for i, utterance in enumerate(conversation):
        if ": " in utterance:
            speaker_label, text = utterance.split(": ", 1)
            role = speaker_map.get(speaker_label.strip().upper(), "user")
        else:
            # Alternate speakers if no prefix is present
            role = "user" if i % 2 == 0 else "assistant"
            text = utterance
        formatted += f"<|im_start|>{role}\n{text.strip()}\n<|im_end|>\n"

    return formatted


FORMAT_HANDLERS = {
    DataFormat.SIMPLE: format_simple,
    DataFormat.INSTRUCTION: format_instruction,
    DataFormat.CONVERSATION: format_conversation,
    DataFormat.PERSONACHAT: format_personachat,
}


def get_datamodules(
    seed: int,
    dev: bool,
    phi: bool,
    gun: bool,
    tokenizer,
    hparams,
    data_path,
    *args,
):
    train_data = []
    config = get_dataset_configs(dev, phi)
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
            validation_data,
            tokenizer,
            hparams["batch_size"],
            hparams["block_size"],
        )

    return train_dataloader, validation_dataloader


def get_dataset(format, tokenizer, block_size, seed, *args):
    if format == "huggingface":
        dataset = HuggingfaceDataset(tokenizer, block_size, seed, *args)
        # First arg is config dict for huggingface
        dataset.weight = args[0].get("weight", 1.0)
        return dataset
    elif format == "directory":
        dataset = MultiDirectoryDataset(tokenizer, block_size, *args)
        dataset.weight = 0.1  # Default weight for directory datasets
        return dataset
    elif format == "gun":
        dataset = GunChatDataset(tokenizer, block_size)
        dataset.weight = 0.001  # Default weight for gun dataset
        return dataset


def get_dataset_configs(dev: bool, phi: bool):
    config = {"primary": [], "validation": []}
    config["primary"].append(
        random.choices(HUGGINGFACE_DATASETS, HUGGINGFACE_PROBS, k=1)[0]
    )
    if phi:
        config["primary"].append(HUGGINGFACE_DATASETS[0])
        config["primary"].append(HUGGINGFACE_DATASETS[1])
        config["primary"].append(HUGGINGFACE_DATASETS[2])
        config["primary"].append(HUGGINGFACE_DATASETS[3])
    if dev:
        # Overwrite with simpler dataset
        config["primary"] = [HUGGINGFACE_DATASETS[0]]
    else:
        config["validation"].append(HUGGINGFACE_DATASETS[4])

    return config


class PraxisSampler:
    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sequence_cache = []  # Store raw text sequences
        self.token_cache = []  # Store tokenized batches
        self.weight = 1.0

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
        self.debug = debug

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
        self.format = config.get("format", DataFormat.SIMPLE)
        if isinstance(self.format, str):
            self.format = DataFormat(self.format)
        # For custom formats, config should provide format_handler
        self.format_handler = (
            config.get("format_handler")
            if self.format == DataFormat.CUSTOM
            else FORMAT_HANDLERS[self.format]
        )
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
        return self.format_handler(document, self.keys)


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
        # Get a list of text samples
        text_list = self.gun.get_sample(250)

        # Prepare the system prompt with arbitrary descriptions
        user_description = random.choice(
            [
                "The user is interested in technology and gadgets.",
                "The user loves discussing philosophy and life.",
                "The user is curious about the latest news.",
                "The user enjoys learning about history.",
                "The user is seeking advice on personal development.",
                "The user is passionate about art and creativity.",
                "The user is looking for travel recommendations.",
                "The user is studying computer science.",
                "The user wants to learn new cooking recipes.",
                "The user is enthusiastic about sports and fitness.",
            ]
        )
        assistant_description = random.choice(
            [
                "The assistant is a knowledgeable and helpful AI.",
                "The assistant provides clear and concise answers.",
                "The assistant is friendly and supportive.",
                "The assistant offers detailed explanations.",
                "The assistant helps users understand complex topics.",
                "The assistant is skilled in problem-solving.",
                "The assistant is patient and understanding.",
                "The assistant excels in educational guidance.",
                "The assistant is adept at providing creative ideas.",
                "The assistant is resourceful and informative.",
            ]
        )

        system_prompt = f"{user_description}\n{assistant_description}"

        # Initialize the formatted text with system message
        formatted = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"

        # Build the conversation by randomly assigning text to user or assistant
        for text in text_list:
            role = random.choice(["user", "assistant"])
            formatted += f"<|im_start|>{role}\n{text.strip()}\n<|im_end|>\n"

        # Add the conversation to the sequence cache
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

        # Get weights and normalize them while preserving relative magnitudes
        raw_weights = [dataset.weight for dataset in train_datasets]
        weights = [w / sum(raw_weights) for w in raw_weights]

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
