import os
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer


class DataFormat(Enum):
    SIMPLE = "simple"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PERSONACHAT = "persona_chat"
    CUSTOM = "custom"
    SMOLTALK = "smoltalk"
    SODA = "soda"
    WIKI = "wiki"


HUGGINGFACE_DATASETS = {
    "minipile-train": dict(
        path="JeanKaddour/minipile",
        split="train",
        keys=["text"],
        format=DataFormat.SIMPLE,
        streaming=False,
    ),
    "minipile-validation": dict(
        path="JeanKaddour/minipile",
        split="validation",
        keys=["text"],
        format=DataFormat.SIMPLE,
        streaming=False,
    ),
    "textbooks": dict(
        path="open-phi/textbooks",
        keys=["markdown"],
        format=DataFormat.SIMPLE,
    ),
    "cosmopedia-v2": dict(
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        keys=["prompt", "text"],
        format=DataFormat.INSTRUCTION,
    ),
    "natural-instructions": dict(
        path="Muennighoff/natural-instructions",
        name="default",
        keys=["definition", "inputs", "targets"],
        format=DataFormat.CONVERSATION,
    ),
    "persona-chat": dict(
        path="google/Synthetic-Persona-Chat",
        keys=["user 1 personas", "user 2 personas", "Best Generated Conversation"],
        format=DataFormat.PERSONACHAT,
    ),
    "smoltalk": dict(
        path="HuggingFaceTB/smoltalk",
        name="all",
        keys=["messages"],
        format=DataFormat.SMOLTALK,
    ),
    "soda": dict(
        path="allenai/soda",
        keys=[
            "speakers",
            "narrative",
            "literal",
            "dialogue",
            "head",
            "relation",
            "tail",
        ],
        format=DataFormat.SODA,
    ),
    "github-code": dict(
        path="codeparrot/github-code",
        name="all-all",
        keys=["code"],
        format=DataFormat.SIMPLE,
        trust_remote_code=True,
    ),
    "tinystories": dict(
        path="roneneldan/TinyStories",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "legal": dict(
        path="pile-of-law/pile-of-law",
        name="all",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "wikipedia": dict(
        path="wikimedia/wikipedia",
        name="20231101.en",
        keys=["title", "text"],
        format=DataFormat.WIKI,
    ),
    "redpajama": dict(
        path="togethercomputer/RedPajama-Data-V2",
        name="sample-10B",
        snapshots=["2023-14"],
        keys=["raw_content"],
        format=DataFormat.SIMPLE,
    ),
    "slimpajama": dict(
        path="cerebras/SlimPajama-627B",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb-edu-10bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb-edu-100bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb-edu-350bt": dict(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-350BT",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
    "fineweb": dict(
        path="HuggingFaceFW/fineweb",
        name="default",
        keys=["text"],
        format=DataFormat.SIMPLE,
    ),
}

DEFAULT_WEIGHT = 1.0
DATASET_COLLECTIONS = dict(
    base={
        "fineweb-edu-350bt": DEFAULT_WEIGHT,
    },
    phi={
        "fineweb": 0.5,
        "textbooks": 0.005,
        "soda": 0.01,
        "cosmopedia-v2": 0.01,
        "natural-instructions": 0.02,
        "github-code": 0.01,
        # "smoltalk": 0.005,
        # "persona-chat": 0.002,
        # "wikipedia": 0.001,
        # "tinystories": 0.05,
        # "legal": 0.001,
    },
    pile={
        "minipile-train": DEFAULT_WEIGHT,
    },
    validation={
        "minipile-validation": DEFAULT_WEIGHT,
    },
    dev={
        "textbooks": DEFAULT_WEIGHT,
    },
    redpajama={
        "redpajama": DEFAULT_WEIGHT,
    },
)


def format_simple(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Just concatenate content with spaces"""
    return document.get(keys[0], "") + "\n"


def format_instruction(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format as instruction/output pairs using the tokenizer's chat template."""
    assert len(keys) == 2, "Instruction format requires exactly 2 keys"
    instruction = document.get(keys[0], "")
    output = document.get(keys[1], "")

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_conversation(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format as a conversation using the tokenizer's chat template."""
    assert len(keys) == 3, "Conversation format requires exactly 3 keys"

    messages = [
        {"role": "system", "content": document.get(keys[0], "")},
        {"role": "user", "content": document.get(keys[1], "")},
        {"role": "assistant", "content": document.get(keys[2], "")},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_personachat(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format persona chat conversations using the tokenizer's chat template."""
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

    # Map speaker labels to ChatML roles
    speaker_map = {
        "A": "user",
        "B": "assistant",
        "USER 1": "user",
        "USER1": "user",
        "USER 2": "assistant",
        "USER2": "assistant",
    }

    # Build messages list
    messages = [{"role": "system", "content": system_message.strip()}]

    for i, utterance in enumerate(conversation):
        if ": " in utterance:
            speaker_label, text = utterance.split(": ", 1)
            role = speaker_map.get(speaker_label.strip().upper(), "user")
        else:
            # Alternate speakers if no prefix is present
            role = "user" if i % 2 == 0 else "assistant"
            text = utterance

        messages.append({"role": role, "content": text.strip()})

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_smoltalk(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format Smoltalk-style message arrays using the tokenizer's chat template."""
    assert (
        len(keys) == 1 and keys[0] == "messages"
    ), "Smoltalk format requires 'messages' key"

    # Get messages array
    messages = document.get(keys[0], [])

    # Filter out any empty messages
    filtered_messages = [
        message for message in messages if message.get("content", "").strip()
    ]

    return tokenizer.apply_chat_template(filtered_messages, tokenize=False) + "\n"


def format_wiki(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> str:
    """Format wiki text."""
    assert len(keys) == 2, "Wiki format requires exactly 2 keys"
    title = document.get(keys[0], "")
    body = document.get(keys[1], "")

    messages = [
        {"role": "user", "content": title},
        {"role": "assistant", "content": body},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_soda(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> str:
    """Formats a single SODA example using the tokenizer's chat template."""
    speakers = document[keys[0]]
    narrative = document[keys[1]]
    literal = document[keys[2]]
    dialogue = document[keys[3]]
    head = document[keys[4]]
    relation = document[keys[5]]
    tail = document[keys[6]]

    # Create person mapping first
    person_mapping = create_person_mapping(document)

    # Get speaker roles
    unique_speakers = list(dict.fromkeys(speakers))  # preserve order, remove duplicates
    speaker_roles = {}

    # Always map first two speakers to user/assistant
    if len(unique_speakers) >= 1:
        speaker_roles[unique_speakers[0]] = "user"
    if len(unique_speakers) >= 2:
        speaker_roles[unique_speakers[1]] = "assistant"

    # Map any additional speakers to "other"
    for speaker in unique_speakers[2:]:
        speaker_roles[speaker] = "other"

    # Create system message content
    system_content = ""

    # Add role mappings to system context
    for speaker, role in speaker_roles.items():
        system_content += f"{role}: {speaker}\n"

    # Add knowledge structure
    system_content += f"cause: {replace_person_references(head, person_mapping)}\n"
    system_content += f"relation: {relation[1:]}\n"
    system_content += f"effect: {replace_person_references(tail, person_mapping)}\n"

    # Add context from literal and narrative
    system_content += f"context: {narrative}\n"
    system_content += f"thought: ({literal})\n"

    # Create messages array
    messages = [{"role": "system", "content": system_content}]

    # Add conversation turns
    for speaker, message in zip(speakers, dialogue):
        role = speaker_roles[speaker]
        messages.append({"role": role, "content": message})

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def create_person_mapping(example: Dict) -> Dict[str, str]:
    """Creates a mapping from PersonX/Y/Z to actual names."""
    mapping = {}
    # Only add non-empty mappings
    if example["PersonX"]:
        mapping["PersonX"] = example["PersonX"]
    if example["PersonY"]:
        mapping["PersonY"] = example["PersonY"]
    if example["PersonZ"]:
        mapping["PersonZ"] = example["PersonZ"]
    return mapping


def replace_person_references(text: str, mapping: Dict[str, str]) -> str:
    """Replaces PersonX/Y/Z references with actual names."""
    if not text:
        return text

    result = text
    for person, name in mapping.items():
        if name:  # Only replace if we have a name
            result = result.replace(person, name)
    return result


FORMAT_HANDLERS = {
    DataFormat.SIMPLE: format_simple,
    DataFormat.INSTRUCTION: format_instruction,
    DataFormat.CONVERSATION: format_conversation,
    DataFormat.PERSONACHAT: format_personachat,
    DataFormat.SMOLTALK: format_smoltalk,
    DataFormat.SODA: format_soda,
    DataFormat.WIKI: format_wiki,
}


def get_datamodules(
    seed: int,
    dev: bool,
    pile: bool,
    phi: bool,
    gun: bool,
    source: bool,
    tokenizer,
    hparams,
    data_path,
    *args,
):

    # An important warning
    if gun and seed and not dev:
        print(
            "WARNING: GUN chats are never deterministic, and cannot be reproduced when using a `seed`. You should omit the `--gun` argument for experiments."
        )
        time.sleep(5)

    train_data = []
    config = get_dataset_configs(dev, pile, phi)
    for c in config["primary"]:
        # load configs for huggingface datasets
        train_data.append(get_dataset("huggingface", tokenizer, seed, c, *args))

    if not pile:
        if source:
            # load configs for training on praxis source code
            train_data.append(
                get_dataset(
                    "self",
                    tokenizer,
                    seed,
                    *args,
                )
            )
        # load configs for local file datasets
        if data_path:
            train_data.append(
                get_dataset(
                    "directory",
                    tokenizer,
                    seed,
                    data_path=data_path,
                    *args,
                )
            )
        if gun:
            # load configs for training on live chat data from https://src.eco
            train_data.append(get_dataset("gun", tokenizer, seed))

    validation_data = []
    if len(config["validation"]) > 0:
        for c in config["validation"]:
            validation_data.append(
                get_dataset("huggingface", tokenizer, seed, c, *args)
            )

    train_dataloader = PraxisDataModule(
        train_data,
        validation_data,
        tokenizer,
        hparams["batch_size"],
        hparams["block_size"],
        hparams["oversample_chance"],
        hparams["supersample_chance"],
        hparams["hypersample_chance"],
    )

    return train_dataloader


def get_dataset(format, tokenizer, seed, *args, **kwargs):
    if format == "huggingface":
        dataset = HuggingfaceDataset(tokenizer, seed, *args)
        dataset.weight = args[0].get("weight", 1.0)
        return dataset
    elif format == "directory":
        dataset = MultiDirectoryDataset(tokenizer, directories=kwargs.get("data_path"))
        dataset.weight = 0.1
        return dataset
    elif format == "self":
        dataset = MultiDirectoryDataset(
            tokenizer,
            directories="./",
            allowed_extensions=[
                ".cfg",
                ".css",
                ".gd",
                ".godot",
                ".html",
                ".ini",
                ".js",
                ".md",
                ".mjs",
                ".py",
                ".sh",
                ".ts",
                ".tscn",
                ".txt",
            ],
        )
        dataset.weight = 0.01
        return dataset
    elif format == "gun":
        dataset = GunChatDataset(tokenizer)
        dataset.weight = 0.01
        return dataset


def add_collection(config, collection_name, target_key):
    """Add datasets from a collection to the config with their weights"""
    if collection_name in DATASET_COLLECTIONS:
        for dataset_name, weight in DATASET_COLLECTIONS[collection_name].items():
            dataset_config = HUGGINGFACE_DATASETS.get(dataset_name).copy()
            dataset_config["weight"] = weight
            config[target_key].append(dataset_config)
    return config


def get_dataset_configs(dev: bool, pile: bool, phi: bool):
    config = {"primary": [], "validation": []}
    if pile:
        config = add_collection(config, "pile", "primary")
        config = add_collection(config, "validation", "validation")
    else:
        config = add_collection(config, "base", "primary")
        if phi:
            config = add_collection(config, "phi", "primary")
        if dev:
            config["primary"] = []
            config = add_collection(config, "dev", "primary")
        else:
            config = add_collection(config, "redpajama", "validation")
    print("training on:")
    [
        print(f"dataset: {entry['path']}, weight: {entry['weight']}")
        for entry in config["primary"]
    ]
    return config


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

    def get_batch(
        self,
        batch_size: int,
        oversample: bool = False,
        supersample: bool = False,
        hypersample: bool = False,
    ) -> List[torch.Tensor]:
        sequence_length = self.block_size
        current_batch_size = batch_size

        # Check if batch size supports the requested sampling mode
        if hypersample and batch_size >= 64:
            current_batch_size = batch_size // 64
            sequence_length = self.block_size * 8
        elif supersample and batch_size >= 16:
            current_batch_size = batch_size // 16
            sequence_length = self.block_size * 4
        elif oversample and batch_size >= 4:
            current_batch_size = batch_size // 4
            sequence_length = self.block_size * 2

        # Calculate how many total tokens we need
        tokens_needed = current_batch_size * sequence_length

        # Make sure we have enough tokens
        while len(self.token_stream) < tokens_needed:
            self._extend_token_stream()

        # Extract batch
        batch = []
        for i in range(current_batch_size):
            start = i * sequence_length
            end = start + sequence_length
            batch.append(self.token_stream[start:end])

        # Remove used tokens from the stream
        self.token_stream = self.token_stream[tokens_needed:]

        return batch

    def _create_interleaved_sequence(self) -> str:
        """Create a single interleaved sequence from multiple samplers"""
        sequence = ""
        while len(sequence) < self.text_cache_size:
            # Pick a sampler based on weights
            sampler = random.choices(self.samplers, weights=self.weights, k=1)[0]
            # Get a sequence from that sampler
            new_sequences = sampler.get_sequences(1)
            # Use a separator token between sequences
            sequence += new_sequences[0] + self.tokenizer.eos_token + "\n"
        return sequence

    def _extend_token_stream(self):
        """Add more tokens to our stream when needed"""
        interleaved = self._create_interleaved_sequence()
        tokens = self.tokenizer(
            text=interleaved,
            padding=False,  # No padding needed since we're building a stream
            return_tensors="pt",
        )["input_ids"].squeeze(
            0
        )  # Get flat token sequence
        self.token_stream = torch.cat([self.token_stream, tokens])


class PraxisSampler:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.weight = 1.0
        self.tokenizer = tokenizer
        self.sequence_cache = []  # Store raw text sequences

    def fill_sequence_cache(self):
        """Each dataset implementation should override this"""
        raise NotImplementedError

    def get_sequences(self, count: int = 1) -> List[str]:
        """Get raw sequences from this dataset"""
        while len(self.sequence_cache) < count:
            self.fill_sequence_cache()
        return [self.sequence_cache.pop(0) for _ in range(count)]


class HuggingfaceDataset(PraxisSampler):
    counts = {}

    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int, config: Dict):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
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
        self.dataset_path = config.get("path", "HuggingFaceFW/fineweb")
        dataset_args = dict(
            path=self.dataset_path,
            split=config.get("split", "train"),
            streaming=config.get("streaming", True),
            trust_remote_code=config.get("trust_remote_code", False),
        )
        if "name" in config:
            dataset_args["name"] = config["name"]
        self.dataset = load_dataset(**dataset_args)
        shuffle_args = {"seed": seed}
        if dataset_args["streaming"]:
            shuffle_args["buffer_size"] = 1000
        self.shuffled_dataset = self.dataset.shuffle(**shuffle_args)
        self.dataset_iterator = iter(self.shuffled_dataset)

        # Initialize the count for this dataset path if not exists
        if self.dataset_path not in HuggingfaceDataset.counts:
            HuggingfaceDataset.counts[self.dataset_path] = 0

    def fill_sequence_cache(self):
        try:
            document = next(self.dataset_iterator)
            formatted = self._format_document(document)
            self.sequence_cache.append(formatted)
        except StopIteration:
            HuggingfaceDataset.counts[self.dataset_path] += 1
            print(
                f"INFO: Reached the last batch of '{self.dataset_path}' dataset. Starting over. ({HuggingfaceDataset.counts[self.dataset_path]}x)"
            )
            self.dataset_iterator = iter(self.shuffled_dataset)

    def _format_document(self, document):
        return self.format_handler(document, self.keys, self.tokenizer)

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
        directories: List[str],
        allowed_extensions: Optional[List[str]] = [],
        excluded_dirs: Optional[List[str]] = None,
    ):
        super().__init__(tokenizer)
        # Normalize and resolve all directory paths relative to current working directory
        self.cwd = os.getcwd()
        self.directories = [
            os.path.normpath(os.path.join(self.cwd, d)) for d in directories
        ]
        self.directories.remove("/")
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
        # print(f"File extensions filter: {self.allowed_extensions}")
        # print(f"Excluding directories: {self.excluded_dirs}")

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
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)
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

        # Build conversation in ChatML format using the tokenizer's template
        messages = [{"role": "system", "content": system_prompt}]

        # Add the conversation messages
        for text in text_list:
            role = random.choice(["user", "assistant"])
            messages.append({"role": role, "content": text.strip()})

        # Apply the chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

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
        hypersample_chance: float = 0,
    ):
        self.data_manager = InterleaveDataManager(
            datasets, weights, tokenizer, block_size
        )
        self.batch_size = batch_size
        self.oversample_chance = oversample_chance
        self.supersample_chance = supersample_chance
        self.hypersample_chance = hypersample_chance

    def __iter__(self):
        while True:
            oversample = random.random() < self.oversample_chance
            supersample = random.random() < self.supersample_chance
            hypersample = random.random() < self.hypersample_chance

            batch = self.data_manager.get_batch(
                self.batch_size, oversample, supersample, hypersample
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
        hypersample_chance: float = 0,
    ):
        super().__init__()
        self.train_datasets = self.create_datasets(
            train_datasets,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
            hypersample_chance,
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
        hypersample_chance=0,
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
            hypersample_chance,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_datasets,
            batch_size=None,
            num_workers=1,
            pin_memory=False,
        )

    def val_dataloader(self):
        if self.val_datasets:
            return DataLoader(
                dataset=self.val_datasets,
                batch_size=None,
                num_workers=1,
                pin_memory=False,
            )
        else:
            return []

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
