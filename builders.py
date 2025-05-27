import os
import random
import re
import sys
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer


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
        non_zero = [r for r in rewards if r > 0]
        self.stats["total"]["rl_sequences"] += len(non_zero)
        
        if non_zero:
            self.stats["rewards"]["count"] += len(non_zero)
            self.stats["rewards"]["sum"] += sum(non_zero)
            if "min" not in self.stats["rewards"]:
                self.stats["rewards"]["min"] = min(non_zero)
            else:
                self.stats["rewards"]["min"] = min(self.stats["rewards"]["min"], min(non_zero))
            
            if "max" not in self.stats["rewards"]:
                self.stats["rewards"]["max"] = max(non_zero)
            else:
                self.stats["rewards"]["max"] = max(self.stats["rewards"]["max"], max(non_zero))
            
            # Track reward distribution
            for r in non_zero:
                bucket = f"{int(r * 10) / 10:.1f}"
                self.stats["distribution"][bucket] += 1
        
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
        
        if self.stats["rewards"]["count"] > 0:
            avg_reward = self.stats["rewards"]["sum"] / self.stats["rewards"]["count"]
            print(f"  Rewards: avg={avg_reward:.3f}, min={self.stats['rewards']['min']:.3f}, max={self.stats['rewards']['max']:.3f}")
            
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


class DataFormat(Enum):
    SIMPLE = "simple"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PERSONACHAT = "persona_chat"
    CUSTOM = "custom"
    SMOLTALK = "smoltalk"
    SODA = "soda"
    WIKI = "wiki"
    RL = "rl"


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
    "intellect-rl": dict(
        path="PrimeIntellect/INTELLECT-2-RL-Dataset",
        split="train",
        keys=["prompt", "verification_info", "solve_rate_qwen_r1_distill_7b"],
        format=DataFormat.RL,
        streaming=True,
    ),
}

DEFAULT_WEIGHT = 1.0
SOURCE_WEIGHT = 0.025
DIRECTORY_WEIGHT = 2.0
GUN_WEIGHT = 0.01

DATASET_COLLECTIONS = dict(
    base={
        "fineweb-edu-350bt": DEFAULT_WEIGHT,
    },
    phi={
        "fineweb": 0.5,
        "textbooks": 0.002,
        "soda": 0.01,
        "cosmopedia-v2": 0.01,
        "natural-instructions": 0.05,
        "github-code": 0.01,
        "smoltalk": 0.01,
        "tinystories": 0.05,
        "persona-chat": 0.005,
        # "wikipedia": 0.001,
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
    rl={
        "intellect-rl": DEFAULT_WEIGHT,
    },
)


def text_formatter(text):
    """
    Convert single newlines to double newlines between paragraphs while preserving
    existing formatting with multiple newlines.

    A paragraph boundary is identified by:
    1. End of line is a letter, number, punctuation, or quote
    2. Start of next line is a capital letter (possibly preceded by quotes)
    3. Start of next line is NOT a list marker, indentation, or code-like content

    Args:
        text (str): The input text to reformat

    Returns:
        str: Reformatted text with appropriate double newlines
    """

    text = add_newline_before_lists(text)

    # First, preserve existing multiple newlines (2 or more)
    # Use regex to match and replace sequences of 2 or more newlines
    text = re.sub(
        r"\n{2,}",
        lambda m: "\n" + "__NEWLINE_" + str(len(m.group()) - 1) + "__" + "\n",
        text,
    )

    # Special case for lines ending with triple backticks
    # This specifically handles code block endings
    backtick_pattern = r"(```)\n(?![ \t]|[-*•+] |[0-9]+[\.\)] )([\"\']*[A-Z])"
    backtick_replacement = r"\1\n\n\2"
    text = re.sub(backtick_pattern, backtick_replacement, text)

    # Define the pattern for paragraph boundaries
    # Look for:
    # 1. One of these characters at the end: letter, number, common punctuation, quote, parenthesis
    # 2. Followed by a single newline
    # 3. NOT followed by indentation, list markers, or code keywords
    # 4. Followed by an optional quotation mark and then an uppercase letter
    pattern = r'([a-zA-Z0-9.,;:!?"\')])(\n)(?![ \t]|[-*•+] |[0-9]+[\.\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await )([\"\']*[A-Z])'

    # Replace with the same characters but with double newline
    replacement = r"\1\n\n\3"

    # Perform the replacement
    reformatted_text = re.sub(pattern, replacement, text)

    # Restore original multiple newlines, but collapse 3+ newlines to 2
    reformatted_text = re.sub(
        r"\n__NEWLINE_(\d+)__\n",
        lambda m: "\n\n" if int(m.group(1)) >= 2 else "\n" * (int(m.group(1)) + 1),
        reformatted_text,
    )

    return reformatted_text


def add_newline_before_lists(text):
    # Define patterns for list items
    list_patterns = [
        r"^\s*- ",  # Bullet points
        r"^\s*\d+\.\s",  # Numbered lists with dot
        r"^\s*\d+\)\s*",  # Numbered lists with parenthesis
        r"^\s*\d+#\s*",  # Hash notation as mentioned in example
    ]

    # Function to check if a line is a list item
    def is_list_item(line):
        return any(re.match(pattern, line) for pattern in list_patterns)

    # Split the text into lines
    lines = text.split("\n")
    if not lines:  # Handle empty text
        return ""

    # Process the text
    result = []
    i = 0
    while i < len(lines):
        result.append(lines[i])

        # Check if we need to add a newline before a list item
        if i < len(lines) - 1:
            current = lines[i]
            next_line = lines[i + 1]

            # If current line has content and is not a list item,
            # and next line is a list item
            if (
                current.strip()
                and not is_list_item(current)
                and is_list_item(next_line)
            ):

                # Check if there's already a blank line between them
                if next_line.strip():  # This means there's only one newline separator
                    result.append("")  # Add a blank line

        i += 1

    return "\n".join(result)


def format_simple(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Just concatenate content with spaces"""
    return tokenizer.bos_token + "\n" + text_formatter(document.get(keys[0])) + "\n"


def format_instruction(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format as instruction/output pairs using the tokenizer's chat template."""
    assert len(keys) == 2, "Instruction format requires exactly 2 keys"
    instruction = text_formatter(document.get(keys[0], ""))
    output = text_formatter(document.get(keys[1], ""))

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


def format_rl(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Format RL dataset for chain-of-thought training.

    The format includes:
    - prompt: The mathematical problem
    - verification_info: JSON with ground truth answer
    - solve_rate: Performance score (0-1) used as reward signal
    
    Returns a dict with 'text' and 'reward' keys.
    """
    assert len(keys) == 3, "RL format requires exactly 3 keys"

    prompt = document.get(keys[0], "")
    verification_info = document.get(keys[1], "{}")
    solve_rate = document.get(keys[2], 0.0)

    # Log every Nth document for monitoring
    if not hasattr(format_rl, "_doc_count"):
        format_rl._doc_count = 0
        format_rl._doc_log_interval = 100
    
    format_rl._doc_count += 1
    
    # Log first few documents and then periodically
    if format_rl._doc_count <= 5 or format_rl._doc_count % format_rl._doc_log_interval == 0:
        print(f"[RL Dataset] Document {format_rl._doc_count} - solve_rate: {solve_rate:.3f}")

    # Parse the ground truth from verification_info
    import json

    try:
        verification_data = json.loads(verification_info)
        ground_truth = verification_data.get("ground_truth", "")
    except:
        ground_truth = ""

    # Create a chain-of-thought style conversation
    # The assistant should generate reasoning steps before the answer
    messages = [
        {
            "role": "system",
            "content": "You are a mathematical problem solver. Solve the problem step by step.",
        },
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": f"Let me solve this step by step.\n\n[Chain of thought reasoning would go here]\n\nTherefore, the answer is: {ground_truth}",
        },
    ]

    # Apply chat template WITHOUT embedding rewards
    text = tokenizer.apply_chat_template(messages, tokenize=False) + "\n"
    
    # Return both text and reward separately
    return {"text": text, "reward": solve_rate}


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
    DataFormat.RL: format_rl,
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
    reinforce: bool = False,
    *args,
):
    print(f"[RL] get_datamodules called with reinforce={reinforce}")

    # An important warning
    if gun and seed and not dev:
        print(
            "WARNING: GUN chats are never deterministic, and cannot be reproduced when using a `seed`. You should omit the `--gun` argument for experiments."
        )
        time.sleep(5)

    train_data = []
    config = get_dataset_configs(dev, pile, phi, reinforce)
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
        reinforce=reinforce,
    )

    return train_dataloader


def get_dataset(format, tokenizer, seed, *args, **kwargs):
    if format == "huggingface":
        dataset = HuggingfaceDataset(tokenizer, seed, *args)
        dataset.weight = args[0].get("weight", 1.0)
        return dataset
    elif format == "directory":
        dataset = MultiDirectoryDataset(tokenizer, directories=kwargs.get("data_path"))
        dataset.weight = DIRECTORY_WEIGHT
        return dataset
    elif format == "self":
        dataset = MultiDirectoryDataset(
            tokenizer,
            directories="./",
            allowed_extensions=[
                ".bib",
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
                ".tex",
                ".toml",
                ".ts",
                ".tscn",
                ".txt",
                ".yaml",
                ".yml",
                "LICENSE",
            ],
        )
        dataset.weight = SOURCE_WEIGHT
        return dataset
    elif format == "gun":
        dataset = GunChatDataset(tokenizer)
        dataset.weight = GUN_WEIGHT
        return dataset


def add_collection(config, collection_name, target_key):
    """Add datasets from a collection to the config with their weights"""
    if collection_name in DATASET_COLLECTIONS:
        for dataset_name, weight in DATASET_COLLECTIONS[collection_name].items():
            dataset_config = HUGGINGFACE_DATASETS.get(dataset_name).copy()
            dataset_config["weight"] = weight
            config[target_key].append(dataset_config)
    return config


def get_dataset_configs(dev: bool, pile: bool, phi: bool, reinforce: bool = False):
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
            # Add RL datasets even in dev mode if reinforce is enabled
            if reinforce:
                config = add_collection(config, "rl", "primary")
        else:
            if reinforce:
                config = add_collection(config, "rl", "primary")
            config = add_collection(config, "redpajama", "validation")
    print("training on:")
    [
        print(f"dataset: {entry['path']}, weight: {entry['weight']}")
        for entry in config["primary"]
    ]
    
    # Debug: print reinforce status
    if reinforce:
        print(f"[RL] Reinforce enabled, {len([e for e in config['primary'] if 'RL' in e.get('path', '')])} RL datasets in config")
    
    return config


class InterleaveDataManager:
    def __init__(
        self, samplers, weights, tokenizer, block_size, text_cache_size=100_000, reinforce=False
    ):
        self.samplers = samplers
        self.weights = weights
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_cache_size = text_cache_size
        self.reinforce = reinforce
        self.token_stream = torch.tensor(
            [], dtype=torch.long
        )  # Single continuous stream
        # Track sequences and their boundaries
        self.sequence_boundaries = []  # List of (start_idx, end_idx, reward) tuples
        self.current_stream_offset = 0  # Track position in token stream

    def get_batch(
        self,
        batch_size: int,
        oversample: bool = False,
        supersample: bool = False,
        hypersample: bool = False,
    ) -> Dict[str, Any]:
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
        rewards = [] if self.reinforce else None
        
        for i in range(current_batch_size):
            start = i * sequence_length
            end = start + sequence_length
            batch.append(self.token_stream[start:end])
            
            # Find the reward for this sequence chunk if reinforce is enabled
            if self.reinforce:
                sequence_reward = self._get_reward_for_range(start, end)
                rewards.append(sequence_reward)

        # Remove used tokens from the stream and update boundaries
        self.token_stream = self.token_stream[tokens_needed:]
        self._update_boundaries_after_removal(tokens_needed)

        return {"batch": batch, "rewards": rewards}
    
    def _get_reward_for_range(self, start: int, end: int) -> float:
        """Get the reward for a token range, handling sequence boundaries properly."""
        # Adjust indices to account for stream offset
        abs_start = self.current_stream_offset + start
        abs_end = self.current_stream_offset + end
        
        # Find all sequences that overlap with this range
        overlapping_rewards = []
        overlap_weights = []
        
        for seq_start, seq_end, reward in self.sequence_boundaries:
            # Calculate overlap
            overlap_start = max(abs_start, seq_start)
            overlap_end = min(abs_end, seq_end)
            
            if overlap_start < overlap_end:
                # This sequence overlaps with our range
                overlap_size = overlap_end - overlap_start
                overlapping_rewards.append(reward)
                overlap_weights.append(overlap_size)
        
        # If we have overlapping sequences, return weighted average
        if overlapping_rewards:
            # If a single sequence fully contains this range, just return its reward
            for seq_start, seq_end, reward in self.sequence_boundaries:
                if seq_start <= abs_start and seq_end >= abs_end:
                    return reward
            
            # Otherwise, return weighted average
            total_weight = sum(overlap_weights)
            if total_weight > 0:
                weighted_reward = sum(r * w for r, w in zip(overlapping_rewards, overlap_weights))
                return weighted_reward / total_weight
        
        # No overlap found, return 0
        return 0.0
    
    def _update_boundaries_after_removal(self, tokens_removed: int):
        """Update sequence boundaries after removing tokens from the stream."""
        self.current_stream_offset += tokens_removed
        
        # Remove boundaries that are now completely before the current stream
        self.sequence_boundaries = [
            (start, end, reward) 
            for start, end, reward in self.sequence_boundaries 
            if end > self.current_stream_offset
        ]

    def _extend_token_stream(self):
        """Add more tokens to our stream when needed, tracking sequence boundaries."""
        sequences_to_add = []
        total_text = ""
        
        # Collect sequences until we have enough text
        while len(total_text) < self.text_cache_size:
            # Pick a sampler based on weights
            sampler = random.choices(self.samplers, weights=self.weights, k=1)[0]
            # Get a sequence from that sampler
            new_sequences = sampler.get_sequences(1)
            text = new_sequences[0]
            
            # Track dataset sampling
            dataset_name = getattr(sampler, 'dataset_path', 'unknown')
            
            # Get reward for this sequence if applicable
            reward = 0.0
            has_reward = False
            if self.reinforce and hasattr(sampler, 'reward_cache'):
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                reward = sampler.reward_cache.get(text_hash, 0.0)
                has_reward = reward > 0
                
                if has_reward:
                    _rl_logger.log_reward_found(reward, dataset_name)
            
            _rl_logger.log_dataset_sample(dataset_name, has_reward)
            
            # Add separator
            text_with_sep = text + self.tokenizer.eos_token + "\n"
            sequences_to_add.append((len(total_text), len(total_text) + len(text_with_sep), reward))
            total_text += text_with_sep
        
        # Tokenize the entire text at once
        tokens = self.tokenizer(
            text=total_text,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        
        # Convert character positions to token positions
        # This is approximate but should work well enough
        chars_per_token = len(total_text) / len(tokens) if len(tokens) > 0 else 1.0
        current_pos = self.current_stream_offset + len(self.token_stream)
        
        for char_start, char_end, reward in sequences_to_add:
            # Estimate token positions based on character positions
            token_start = current_pos + int(char_start / chars_per_token)
            token_end = current_pos + int(char_end / chars_per_token)
            
            # Ensure we have at least one token per sequence
            if token_end <= token_start:
                token_end = token_start + 1
                
            # Store the boundary information
            self.sequence_boundaries.append((token_start, token_end, reward))
        
        # Add tokens to stream
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

        # Debug log RL datasets
        if self.format == DataFormat.RL:
            print(f"[RL] Initializing RL dataset: {self.dataset_path}")
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
            
        # Storage for rewards when using RL format
        self.reward_cache = {}

    def fill_sequence_cache(self):
        try:
            document = next(self.dataset_iterator)
            formatted = self._format_document(document)
            
            # Handle RL format which returns dict
            if self.format == DataFormat.RL and isinstance(formatted, dict):
                text = formatted["text"]
                reward = formatted["reward"]
                # Store reward indexed by text hash
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.reward_cache[text_hash] = reward
                self.sequence_cache.append(text)
                
            else:
                # Regular format, just text
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
            "archive",
            "build",
            "dist",
            "node_modules",
            "praxis.egg-info",
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
            content = self.tokenizer.bos_token + self._read_file(file_path)
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
        reinforce: bool = False,
    ):
        self.data_manager = InterleaveDataManager(
            datasets, weights, tokenizer, block_size, reinforce=reinforce
        )
        self.batch_size = batch_size
        self.oversample_chance = oversample_chance
        self.supersample_chance = supersample_chance
        self.hypersample_chance = hypersample_chance
        self.reinforce = reinforce
        self.tokenizer = tokenizer  # Store tokenizer for reward extraction

        # Debug log
        print(f"[RL] WeightedIterableDataset initialized with reinforce={reinforce}")

    def __iter__(self):
        while True:
            oversample = random.random() < self.oversample_chance
            supersample = random.random() < self.supersample_chance
            hypersample = random.random() < self.hypersample_chance

            result = self.data_manager.get_batch(
                self.batch_size, oversample, supersample, hypersample
            )

            # Extract batch and rewards
            batch = result["batch"]
            rewards = result.get("rewards")

            # Stack batch tensors
            batch_tensor = torch.stack(batch)

            # Handle rewards if reinforce is enabled
            if self.reinforce and rewards:
                # Convert rewards to tensor
                reward_tensor = torch.tensor(rewards, dtype=torch.float32)
                
                # Log batch statistics
                _rl_logger.log_batch(reward_tensor)
                
                # Return dict format with rewards
                yield {"input_ids": batch_tensor, "rewards": reward_tensor}
            else:
                # No reinforcement learning, return regular tensor
                yield batch_tensor



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
        reinforce: bool = False,
    ):
        super().__init__()
        self.reinforce = reinforce
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

        # Debug log
        print(f"[RL] Creating WeightedIterableDataset with reinforce={self.reinforce}")

        return WeightedIterableDataset(
            datasets,
            weights,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
            hypersample_chance,
            reinforce=self.reinforce,
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
