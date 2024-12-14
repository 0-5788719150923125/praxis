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

start_token = "<|im_start|> "
end_token = "<|im_end|> "


class DataFormat(Enum):
    SIMPLE = "simple"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PERSONACHAT = "persona_chat"
    CUSTOM = "custom"
    SMOLTALK = "smoltalk"


HUGGINGFACE_DATASETS = {
    "textbooks": dict(
        path="open-phi/textbooks",
        keys=["markdown"],
        format=DataFormat.SIMPLE,
        weight=0.001,
    ),
    "smollm-corpus": dict(
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        keys=["prompt", "text"],
        format=DataFormat.INSTRUCTION,
        weight=0.001,
    ),
    "natural-instructions": dict(
        path="Muennighoff/natural-instructions",
        name="default",
        keys=["definition", "inputs", "targets"],
        format=DataFormat.CONVERSATION,
        weight=0.001,
    ),
    "persona-chat": dict(
        path="google/Synthetic-Persona-Chat",
        keys=["user 1 personas", "user 2 personas", "Best Generated Conversation"],
        format=DataFormat.PERSONACHAT,
        weight=0.001,
    ),
    "smoltalk": dict(
        path="HuggingFaceTB/smoltalk",
        name="all",
        keys=["messages"],
        format=DataFormat.SMOLTALK,
        weight=0.001,
    ),
    "github-code": dict(
        path="codeparrot/github-code",
        name="all-all",
        keys=["code"],
        format=DataFormat.SIMPLE,
        weight=0.001,
    ),
    "tinystories": dict(
        path="roneneldan/TinyStories",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=0.001,
    ),
    "redpajama": dict(
        path="togethercomputer/RedPajama-Data-V2",
        name="sample-10B",
        snapshots=["2023-14"],
        keys=["raw_content"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    "fineweb-edu-10bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    "fineweb-edu-100bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    "fineweb-edu-350bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-350BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=1.0,
    ),
    "fineweb": dict(
        path="HuggingFaceFW/fineweb",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
        weight=0.1,
    ),
}


def format_simple(document: Dict, keys: List[str]) -> str:
    """Just concatenate content with spaces"""
    return " ".join(document.get(key, "") for key in keys)


def format_instruction(document: Dict, keys: List[str]) -> str:
    """Format as instruction/output pairs in ChatML format."""
    assert len(keys) == 2, "Instruction format requires exactly 2 keys"
    instruction = document.get(keys[0], "")
    output = document.get(keys[1], "")
    return (
        f"{start_token}user\n{instruction}\n{end_token}\n"
        f"{start_token}assistant\n{output}\n{end_token}\n"
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
        parts.append(f"{start_token}{role}\n{message}\n{end_token}\n")
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
    formatted = f"{start_token}system\n{system_message.strip()}\n{end_token}\n"

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
        formatted += f"{start_token}{role}\n{text.strip()}\n{end_token}\n"

    return formatted


def format_smoltalk(document: Dict, keys: List[str]) -> str:
    """Format Smoltalk-style message arrays into ChatML format."""
    assert (
        len(keys) == 1 and keys[0] == "messages"
    ), "Smoltalk format requires 'messages' key"

    # Get messages array
    messages = document.get(keys[0], [])

    # Format each message in original order
    formatted_messages = []
    for message in messages:
        role = message.get("role", "user")  # Default to user if role missing
        content = message.get("content", "").strip()
        if content:  # Only add non-empty messages
            formatted_messages.append(f"{start_token}{role}\n{content}\n{end_token}\n")

    # Join all messages together
    return "".join(formatted_messages)


FORMAT_HANDLERS = {
    DataFormat.SIMPLE: format_simple,
    DataFormat.INSTRUCTION: format_instruction,
    DataFormat.CONVERSATION: format_conversation,
    DataFormat.PERSONACHAT: format_personachat,
    DataFormat.SMOLTALK: format_smoltalk,
}


def get_datamodules(
    seed: int,
    dev: bool,
    phi: bool,
    gun: bool,
    source: bool,
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
            get_dataset(
                "directory",
                tokenizer,
                hparams["block_size"],
                seed,
                data_path=data_path,
                *args,
            )
        )
    if source:
        train_data.append(
            get_dataset(
                "self",
                tokenizer,
                hparams["block_size"],
                seed,
                *args,
            )
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
        validation_data,
        tokenizer,
        hparams["batch_size"],
        hparams["block_size"],
        hparams["oversample_chance"],
        hparams["supersample_chance"],
    )

    return train_dataloader


def get_dataset(format, tokenizer, block_size, seed, *args, **kwargs):
    if format == "huggingface":
        dataset = HuggingfaceDataset(tokenizer, block_size, seed, *args)
        # First arg is config dict for huggingface
        dataset.weight = args[0].get("weight", 1.0)
        return dataset
    elif format == "directory":
        dataset = MultiDirectoryDataset(
            tokenizer, block_size, directories=kwargs.get("data_path")
        )
        dataset.weight = 0.1
        return dataset
    elif format == "self":
        dataset = MultiDirectoryDataset(
            tokenizer,
            block_size,
            directories="./",
            allowed_extensions=[
                ".py",
                ".js",
                ".mjs",
                ".gd",
                ".tscn",
                ".cfg",
                ".godot",
                ".html",
                ".css",
                ".txt",
                ".md",
                ".sh",
            ],
        )
        dataset.weight = 0.001
        return dataset
    elif format == "gun":
        dataset = GunChatDataset(tokenizer, block_size)
        dataset.weight = 0.001
        return dataset


def get_dataset_configs(dev: bool, phi: bool):
    config = {"primary": [], "validation": []}
    config["primary"].append(HUGGINGFACE_DATASETS.get("fineweb-edu-10bt"))
    config["primary"].append(HUGGINGFACE_DATASETS.get("fineweb"))
    if phi:
        config["primary"].append(HUGGINGFACE_DATASETS.get("textbooks"))
        config["primary"].append(HUGGINGFACE_DATASETS.get("smollm-corpus"))
        config["primary"].append(HUGGINGFACE_DATASETS.get("natural-instructions"))
        config["primary"].append(HUGGINGFACE_DATASETS.get("persona-chat"))
        config["primary"].append(HUGGINGFACE_DATASETS.get("smoltalk"))
        config["primary"].append(HUGGINGFACE_DATASETS.get("github-code"))
        config["primary"].append(HUGGINGFACE_DATASETS.get("tinystories"))
    if dev:
        # Overwrite with simpler dataset
        config["primary"] = [HUGGINGFACE_DATASETS.get("textbooks")]
    else:
        config["validation"].append(HUGGINGFACE_DATASETS.get("redpajama"))
    print("training on:")
    [
        print(f"dataset: {entry['path']}, weight: {entry['weight']}")
        for entry in config["primary"]
    ]
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
            sequence += (
                self.tokenizer.bos_token + new_sequences[0] + self.tokenizer.eos_token
            )
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

    def state_dict(self):
        # Get the internal state of the shuffled dataset
        return self.shuffled_dataset.state_dict()

    def load_state_dict(self, state_dict):
        # Restore the internal state so iteration picks up where we left off
        self.shuffled_dataset.load_state_dict(state_dict)
        # Recreate the iterator from the restored state
        self.dataset_iterator = iter(self.shuffled_dataset)


class MultiDirectoryDataset(PraxisSampler):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        directories: List[str],
        allowed_extensions: Optional[List[str]] = [],
        excluded_dirs: Optional[List[str]] = None,
    ):
        super().__init__(tokenizer, block_size)
        # Normalize and resolve all directory paths relative to current working directory
        self.cwd = os.getcwd()
        self.directories = [
            os.path.normpath(os.path.join(self.cwd, d)) for d in directories
        ]
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]

        # Default exclusions for common development directories
        default_exclusions = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
        }
        user_exclusions = set(excluded_dirs) if excluded_dirs else set()
        self.excluded_dirs = default_exclusions.union(user_exclusions)

        print(f"Working directory: {self.cwd}")
        print(f"Scanning directories: {self.directories}")
        print(f"File extensions filter: {self.allowed_extensions}")
        print(f"Excluding directories: {self.excluded_dirs}")

        self.file_list = self._get_file_list()
        print(f"Found {len(self.file_list)} files")
        random.shuffle(self.file_list)
        self.file_iterator = iter(self.file_list)

    def _should_skip_directory(self, dirpath: str) -> bool:
        """
        Check if directory should be skipped based on exclusion rules
        and ensure we don't leave the working directory context.
        """
        dir_name = os.path.basename(dirpath)

        # Check if directory is in excluded list
        if dir_name in self.excluded_dirs:
            return True

        # Ensure directory is within working directory context
        try:
            # Resolve the real path, following symlinks
            real_path = os.path.realpath(dirpath)
            # Check if this path is within our working directory
            return not real_path.startswith(self.cwd)
        except:
            # If there's any error resolving the path, skip it to be safe
            return True

    def _get_file_list(self) -> List[str]:
        """
        Recursively traverse directories and return a flat list of fully-qualified file paths,
        staying within the working directory context.
        """
        all_files = []

        for directory in self.directories:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist, skipping...")
                continue

            try:
                # Walk through directory recursively
                for root, dirs, files in os.walk(
                    directory, topdown=True, followlinks=False
                ):
                    # Modify dirs in-place to prevent walking into excluded directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if not self._should_skip_directory(os.path.join(root, d))
                    ]

                    for filename in files:
                        # Get full path
                        full_path = os.path.join(root, filename)

                        # Verify file is within working directory
                        real_path = os.path.realpath(full_path)
                        if not real_path.startswith(self.cwd):
                            continue

                        # Check if file extension is allowed
                        file_ext = os.path.splitext(filename)[1].lower()
                        if len(self.allowed_extensions) > 0:
                            if file_ext in self.allowed_extensions:
                                all_files.append(full_path)
                        else:
                            all_files.append(full_path)
            except Exception as e:
                print(f"Error scanning directory {directory}: {str(e)}")
                continue

        return all_files

    def _read_file(self, file_path: str) -> str:
        """Read and return the contents of a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                return ""
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return ""

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
        formatted = f"{start_token}system\n{system_prompt}\n{end_token}\n"

        # Build the conversation by randomly assigning text to user or assistant
        for text in text_list:
            role = random.choice(["user", "assistant"])
            formatted += f"{start_token}{role}\n{text.strip()}\n{end_token}\n"

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
        val_datasets: List[Dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        block_size: int = 512,
        oversample_chance: float = 0,
        supersample_chance: float = 0,
    ):
        super().__init__()
        self.train_datasets = self.create_datasets(
            train_datasets,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
        )
        self.val_datasets = False
        if len(val_datasets) > 0:
            self.val_datasets = self.create_datasets(
                val_datasets, tokenizer, block_size, batch_size, 0, 0
            )

    def create_datasets(
        self,
        datasets,
        tokenizer,
        block_size,
        batch_size,
        oversample_chance=0,
        supersample_chance=0,
    ):
        # Get weights and normalize them while preserving relative magnitudes
        raw_weights = [dataset.weight for dataset in datasets]
        weights = [w / sum(raw_weights) for w in raw_weights]
        return WeightedIterableDataset(
            datasets,
            weights,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_datasets,
            batch_size=2,
            num_workers=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_datasets:
            return DataLoader(
                dataset=self.val_datasets,
                batch_size=2,
                num_workers=1,
                pin_memory=True,
            )
        else:
            return {}

    def state_dict(self) -> Dict[str, Any]:
        sampler_states = []
        for sampler in self.train_datasets.data_manager.samplers:
            if hasattr(sampler, "state_dict") and callable(sampler.state_dict):
                sampler_states.append(sampler.state_dict())
            else:
                # If sampler doesn't support state, append None
                sampler_states.append(None)

        return {"sampler_states": sampler_states}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        sampler_states = state_dict.get("sampler_states", [])
        samplers = self.train_datasets.data_manager.samplers
        for sampler, s_state in zip(samplers, sampler_states):
            if (
                s_state is not None
                and hasattr(sampler, "load_state_dict")
                and callable(sampler.load_state_dict)
            ):
                sampler.load_state_dict(s_state)
