import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

DEFAULT_WEIGHT = 1.0
SRC_WEIGHT = 1.0
DIR_WEIGHT = 1.0
TOOLS_WEIGHT = 1.0

DATASET_COLLECTIONS = dict(
    base={
        "fineweb-edu-350bt": DEFAULT_WEIGHT,
    },
    phi={
        "fineweb": 0.75,
        "textbooks": 0.002,
        "tinystories": 0.01,
        "wikipedia": 0.02,
        "persona-chat": 0.1,
        "soda": 0.1,
        "wildchat": 0.1,
        "natural-instructions": 0.5,
        "cosmopedia-v2": 0.2,
        "smoltalk": 0.1,
        "nextcoder": 0.05,
        "nextcoder-conversational": 0.1,
        "hermes-3-dataset": 0.1,
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
    slimpajama={
        "slimpajama": DEFAULT_WEIGHT,
    },
    rl={
        "intellect-rl": DEFAULT_WEIGHT,
    },
    cot={
        "chain-of-thought": DEFAULT_WEIGHT * 0.1,
    },
)


class DataFormat(Enum):
    SIMPLE = "simple"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    PERSONACHAT = "persona_chat"
    CUSTOM = "custom"
    MESSAGES = "messages"
    SODA = "soda"
    WIKI = "wiki"
    RL = "rl"
    COT = "cot"
    TOOL_CALLING = "tool_calling"


# Unified prompts for all data types
SYSTEM_PROMPT = "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation."

DEVELOPER_PROMPTS = {
    "continue_text": "Continue or complete the provided text, maintaining style and coherence.",
    "follow_instruction": "Follow the user's instructions precisely and provide a complete response.",
    "engage_conversation": "Engage naturally in this conversation, being helpful and appropriate.",
    "answer_question": "Answer the question accurately based on your knowledge.",
    "think_step_by_step": "Think step-by-step through this problem before providing your answer.",
    "use_tools": "Use the available tools when appropriate to help the user.",
    "write_article": "Write a comprehensive article or explanation on the given topic.",
    "persona_chat": "Engage in conversation while maintaining the specified personas.",
    "soda_dialogue": "Continue this dialogue naturally based on the context provided.",
}


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
    # "persona-chat": dict(
    #     path="google/Synthetic-Persona-Chat",
    #     keys=["user 1 personas", "user 2 personas", "Best Generated Conversation"],
    #     format=DataFormat.PERSONACHAT,
    # ),
    "persona-chat": dict(
        path="AlekseyKorshuk/persona-chat",
        keys=["personality", "utterances"],
        format=DataFormat.PERSONACHAT,
    ),
    "smoltalk": dict(
        path="HuggingFaceTB/smoltalk",
        name="all",
        keys=["messages"],
        format=DataFormat.MESSAGES,
    ),
    "nextcoder": dict(
        path="microsoft/NextCoderDataset",
        split="train",
        keys=["prompt", "completion"],
        format=DataFormat.INSTRUCTION,
    ),
    "nextcoder-conversational": dict(
        path="microsoft/NextCoderDataset-Conversational",
        keys=["messages"],
        format=DataFormat.MESSAGES,
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
    "hermes-3-dataset": dict(
        path="NousResearch/Hermes-3-Dataset",
        keys=["conversations"],
        format=DataFormat.MESSAGES,
    ),
    "wildchat": dict(
        path="allenai/WildChat",
        keys=["conversation"],
        format=DataFormat.MESSAGES,
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
    # "redpajama": dict(
    #     path="togethercomputer/RedPajama-Data-V2",
    #     name="sample-10B",
    #     snapshots=["2023-14"],
    #     keys=["raw_content"],
    #     format=DataFormat.SIMPLE,
    # ),
    # "slimpajama": dict(
    #     path="cerebras/SlimPajama-627B",
    #     # name="default",
    #     keys=["text"],
    #     format=DataFormat.SIMPLE,
    # ),
    "validation": dict(
        path="allenai/c4",
        name="en",
        split="validation",
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
        mix_simple_math=True,  # Mix in simple problems for untrained models
        # DEBUG: Enable to see what we're loading
        trust_remote_code=False,
    ),
    "chain-of-thought": dict(
        path="isaiahbjork/chain-of-thought",
        split="train",
        keys=["prompt", "response", "category", "topic"],
        format=DataFormat.COT,
        streaming=True,
        trust_remote_code=False,
    ),
}


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

# Chain of Thought tag definitions
COT_TAGS = {
    "wrapper": {
        "thinking": ("<thinking>", "</thinking>"),
        "output": ("<output>", "</output>"),
    },
    "thinking_components": {
        "initial_analysis": ("<initial_analysis>", "</initial_analysis>"),
        "conscious_thought": ("<conscious_thought>", "</conscious_thought>"),
        "step_by_step": ("<step_by_step>", "</step_by_step>"),
        "reflection": ("<reflection>", "</reflection>"),
        "feeling": ("<feeling>", "</feeling>"),
        "self_improvement": ("<self_improvement>", "</self_improvement>"),
        "subcomponent_analysis": (
            "<subcomponent_analysis>",
            "</subcomponent_analysis>",
        ),
    },
    # Tag weights for training
    "tag_weights": {
        "thinking": 1.5,
        "output": 1.0,
        "initial_analysis": 1.3,
        "conscious_thought": 1.3,
        "step_by_step": 1.6,  # Highest weight for step-by-step reasoning
        "reflection": 1.3,
        "feeling": 1.1,
        "self_improvement": 1.2,
        "subcomponent_analysis": 1.4,
    },
    # Reward values for REINFORCE
    "tag_rewards": {
        "thinking": 0.3,
        "output": 0.1,
        "initial_analysis": 0.2,
        "conscious_thought": 0.2,
        "step_by_step": 0.3,
        "reflection": 0.2,
        "feeling": 0.1,
        "self_improvement": 0.1,
        "subcomponent_analysis": 0.2,
    },
}


def text_formatter(text):
    """
    Convert single newlines to double newlines between paragraphs while preserving
    existing formatting with multiple newlines.

    A paragraph boundary is identified by:
    1. End of line is a letter, number, punctuation, or quote
    2. Start of next line is a capital letter (possibly preceded by quotes)
    3. Start of next line is NOT a list marker, indentation, or code-like content

    Special handling for tags:
    - Tags should "squeeze" their content (no double newlines between tags and content)
    - Pattern: <tag>\n becomes <tag> and \n</tag> becomes </tag>

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
    backtick_pattern = r"(```)\n(?![ \t]|[-*•+] |[0-9]+[.\)] )([\"\'" "'']*[A-Z])"
    backtick_replacement = r"\1\n\n\2"
    text = re.sub(backtick_pattern, backtick_replacement, text)

    # Define the pattern for paragraph boundaries
    # Look for:
    # 1. Lines ending with sentence-ending punctuation (., !, ?) - these are likely complete thoughts
    # 2. Followed by a single newline
    # 3. NOT followed by indentation, list markers, or code keywords
    # 4. Followed by an optional quotation mark and then an uppercase letter
    pattern_basic = (
        r"([.!?][\"\'"
        "'']*[)\\]]*)(\n)(?![ \t]|[-*•+] |[0-9]+[.\\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await )([\"'"
        "'']*[A-Z])"
    )

    # Separate pattern for colons: Include them but exclude structured data patterns (word: value)
    pattern_colon = (
        r"(:[\"\'"
        "'']*[)\\]]*)(\n)(?![ \t]|[-*•+] |[0-9]+[.\\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await |[A-Za-z][^:\n]*: )([\"'"
        "'']*[A-Z])"
    )

    # Replace with the same characters but with double newline
    replacement = r"\1\n\n\3"

    # Perform the replacements - first basic punctuation, then colons
    reformatted_text = re.sub(pattern_basic, replacement, text)
    reformatted_text = re.sub(pattern_colon, replacement, reformatted_text)

    # Restore original multiple newlines, but collapse 3+ newlines to 2
    reformatted_text = re.sub(
        r"\n__NEWLINE_(\d+)__\n",
        lambda m: "\n\n" if int(m.group(1)) >= 2 else "\n" * (int(m.group(1)) + 1),
        reformatted_text,
    )

    # Handle tag squeezing AFTER paragraph formatting: remove blank lines between tags and content
    # Tags remain on their own lines, but extra spacing is removed

    # Remove blank lines after opening tags: <tag>\n\ncontent becomes <tag>\ncontent
    tag_squeeze_pattern = r"(<[^/>]+>)\n\n+"  # Opening tag followed by blank lines
    reformatted_text = re.sub(tag_squeeze_pattern, r"\1\n", reformatted_text)

    # Remove blank lines before closing tags: content\n\n</tag> becomes content\n</tag>
    tag_squeeze_pattern_close = r"\n\n+(</[^>]+>)"  # Blank lines before closing tag
    reformatted_text = re.sub(tag_squeeze_pattern_close, r"\n\1", reformatted_text)

    # Ensure closing tags are followed by double newlines when there's content after them
    # </tag>\ncontent becomes </tag>\n\ncontent (but preserve existing double newlines)
    # BUT: Keep consecutive tags together (don't add space between </tag> and <tag>)
    tag_after_pattern = r"(</[^>]+>)\n(?!\n|<|$)"  # Closing tag followed by single newline and content (not another tag or end)
    reformatted_text = re.sub(tag_after_pattern, r"\1\n\n", reformatted_text)

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
    """Convert raw text to unified format with system/developer prompts."""
    text = document.get(keys[0], "")
    if not text:
        return ""

    # For simple/raw text, we treat it as a direct completion task
    # The model should learn to continue/complete texts naturally
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["continue_text"]},
        {"role": "assistant", "content": text_formatter(text)},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_instruction(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format as instruction/output pairs with unified system/developer prompts."""
    assert len(keys) == 2, "Instruction format requires exactly 2 keys"
    instruction = text_formatter(document.get(keys[0], ""))
    output = text_formatter(document.get(keys[1], ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["follow_instruction"]},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_conversation(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format as a conversation with unified system/developer prompts."""
    assert len(keys) == 3, "Conversation format requires exactly 3 keys"

    # Original system message becomes developer message
    original_system = text_formatter(document.get(keys[0], ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": original_system or DEVELOPER_PROMPTS["engage_conversation"],
        },
        {"role": "user", "content": text_formatter(document.get(keys[1], ""))},
        {"role": "assistant", "content": text_formatter(document.get(keys[2], ""))},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def repair_text_punctuation(text: str) -> str:
    """First pass: Fix punctuation and spacing issues."""
    import re

    # Fix common typos/mangled words from the dataset
    # "ll" at start of sentence is often "lol" that got mangled
    text = re.sub(r"^ll\b", "lol", text, flags=re.IGNORECASE)
    text = re.sub(r"([.!?]\s+)ll\b", r"\1lol", text, flags=re.IGNORECASE)

    # Special case: Fix ". . . ? d" pattern (common in dataset)
    # This becomes "...? :D"
    text = re.sub(r"\.\s+\.\s+\.\s+\?\s+d$", r"...? :D", text, flags=re.IGNORECASE)

    # Pass 1: Fix broken emoticons (before any other punctuation fixes)
    # Common patterns where emoticons got split with spaces
    text = re.sub(r":\s+([dDpPsS)])", r":\1", text)  # ": d" -> ":D"
    text = re.sub(r";\s+([dDpPsS)])", r";\1", text)  # "; d" -> ";D"
    text = re.sub(r"x\s+d\b", r"xD", text, flags=re.IGNORECASE)  # "x d" -> "xD"
    # Fix standalone "? d" or ". d" at end
    text = re.sub(r"\?\s+d$", r"? :D", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s+d$", r". :D", text, flags=re.IGNORECASE)

    # Pass 2: Collapse spaced punctuation sequences
    text = re.sub(r"([.!?])\s+([.!?])", r"\1\2", text)  # ". ." -> ".."
    text = re.sub(r"([.!?])\1{3,}", r"\1\1\1", text)  # Limit to max 3 repetitions

    # Pass 3: Fix spacing around punctuation (but preserve emoticons)
    # Don't collapse spaces before : or ; that are part of emoticons
    text = re.sub(r"\s+([,.])", r"\1", text)  # Remove spaces before comma and period
    text = re.sub(
        r"\s+([!?])(?!\s*:)", r"\1", text
    )  # Remove before ! and ? unless followed by :
    text = re.sub(
        r"\s+([:;])(?![DPSdps)(/])", r"\1", text
    )  # Remove before : and ; unless emoticon
    text = re.sub(r"([,;:])(?=[^\s])", r"\1 ", text)  # Add space after punctuation
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces

    return text


def repair_broken_emoticons(text: str) -> str:
    """Second pass: Fix emoticons that got mangled by first pass."""
    import re

    # Fix patterns where emoticons got merged with punctuation
    # "...?:D" should be "...? :D"
    text = re.sub(r"([.!?]):([DPSdps)])", r"\1 :\2", text)
    text = re.sub(r"([.!?]);([DPSdps)])", r"\1 ;\2", text)

    # Uppercase emoticon letters (with optional space before)
    text = re.sub(r":\s*d\b", ":D", text, flags=re.IGNORECASE)
    text = re.sub(r":\s*p\b", ":P", text, flags=re.IGNORECASE)
    text = re.sub(r":\s*s\b", ":S", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*d\b", ";D", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*p\b", ";P", text, flags=re.IGNORECASE)
    text = re.sub(r"x\s*d\b", "xD", text, flags=re.IGNORECASE)

    # Fix isolated ": d" patterns that might remain
    text = re.sub(
        r":\s+([dps])\b", lambda m: ":" + m.group(1).upper(), text, flags=re.IGNORECASE
    )
    text = re.sub(
        r";\s+([dps])\b", lambda m: ";" + m.group(1).upper(), text, flags=re.IGNORECASE
    )

    return text


def simple_truecase(text: str) -> str:
    """Apply simple truecasing heuristics to text with multi-pass repair."""
    if not text:
        return text

    # Strip whitespace
    text = text.strip()
    if not text:
        return text

    import re

    # Multi-pass text repair
    text = repair_text_punctuation(text)
    text = repair_broken_emoticons(text)

    # Split on sentence boundaries while preserving punctuation
    sentences = re.split(r"([.!?]+\s*)", text)

    result = []
    for i, part in enumerate(sentences):
        if not part or part.isspace():
            continue

        # If this is punctuation, just add it
        if re.match(r"^[.!?]+\s*$", part):
            result.append(part)
            continue

        # Process the sentence
        words = part.split()
        if not words:
            continue

        processed_words = []
        for j, word in enumerate(words):
            # Preserve emoticons as-is (common ones)
            if word in [
                ":D",
                ":P",
                ":S",
                ";D",
                ";P",
                "xD",
                "XD",
                ":)",
                ":(",
                ";)",
                ":/",
            ]:
                processed_words.append(word)
                continue

            # Handle words with punctuation
            prefix = ""
            suffix = ""
            core_word = word

            # Extract leading punctuation
            while core_word and core_word[0] in "\"'-":
                prefix += core_word[0]
                core_word = core_word[1:]

            # Extract trailing punctuation
            while core_word and core_word[-1] in ".,!?;:'\"":
                suffix = core_word[-1] + suffix
                core_word = core_word[:-1]

            # Process the core word
            if not core_word:
                processed_words.append(word)
                continue

            # Special cases for I and contractions
            if core_word.lower() == "i" or core_word.lower().startswith("i'"):
                processed_word = "I" + core_word[1:] if len(core_word) > 1 else "I"
            # Special case: Internet slang should stay lowercase even at start of sentence
            elif core_word.lower() in [
                "lol",
                "lmao",
                "rofl",
                "omg",
                "wtf",
                "brb",
                "btw",
                "fyi",
                "imo",
                "imho",
                "afaik",
            ]:
                processed_word = core_word.lower()
            # First word of sentence
            elif j == 0:
                processed_word = (
                    core_word[0].upper() + core_word[1:].lower()
                    if len(core_word) > 1
                    else core_word.upper()
                )
            # Everything else lowercase
            else:
                processed_word = core_word.lower()

            processed_words.append(prefix + processed_word + suffix)

        result.append(" ".join(processed_words))

    # Join the result
    final = "".join(result)

    # Final repair pass to fix any emoticons that got broken during truecasing
    final = repair_broken_emoticons(final)

    # Ensure the text ends with proper punctuation (but not after emoticons)
    if final:
        # Check if it ends with an emoticon
        emoticon_endings = [
            ":D",
            ":P",
            ":S",
            ";D",
            ";P",
            "xD",
            "XD",
            ":)",
            ":(",
            ";)",
            ":/",
        ]
        ends_with_emoticon = any(final.endswith(em) for em in emoticon_endings)

        # Only add period if it doesn't end with punctuation or emoticon
        if not ends_with_emoticon and final[-1] not in ".!?":
            final += "."

    return final


def format_personachat(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format AlekseyKorshuk persona-chat dataset with JSON structure for better randomization."""
    # Extract personality traits and utterances
    personality = document.get("personality", [])
    utterances = document.get("utterances", [])

    if not utterances:
        return ""

    # Randomly select how much of the conversation to include (1 to all utterances)
    num_utterances = random.randint(1, len(utterances))
    selected_utterance = utterances[num_utterances - 1]  # Get the selected utterance

    # Decide whether to apply truecasing (50% chance, applies to entire conversation)
    apply_truecase = random.random() < 0.5

    # Build personality description for developer message
    # Always include personality when available (it's the assistant's personality)
    developer_message = ""
    if personality:
        developer_message = "You have the following personality traits:\n"
        # Randomly select subset of personality traits (or all)
        if random.random() < 0.5:
            # 50% chance to use all traits
            selected_traits = personality
        else:
            # 50% chance to use a random subset
            num_traits = random.randint(1, len(personality))
            selected_traits = random.sample(personality, num_traits)

        for trait in selected_traits:
            # Always truecase personality traits (they're for the assistant)
            trait_text = simple_truecase(trait) if trait else trait
            developer_message += f"- {trait_text}\n"
        developer_message = developer_message.strip()
    else:
        # Use default persona chat prompt if no personality provided
        developer_message = DEVELOPER_PROMPTS["persona_chat"]

    # Build messages list
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": developer_message},
    ]

    # Add the conversation history from the selected utterance
    history = selected_utterance.get("history", [])
    candidates = selected_utterance.get("candidates", [])

    # Add history messages with proper role alternation (user starts)
    for i, hist_msg in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        # Apply truecasing based on role and setting
        if role == "assistant":
            # Assistant always uses proper casing
            content = simple_truecase(hist_msg) if hist_msg else hist_msg
        else:
            # User uses truecasing only if apply_truecase is True
            content = (
                simple_truecase(hist_msg) if (apply_truecase and hist_msg) else hist_msg
            )
        messages.append({"role": role, "content": text_formatter(content)})

    # Pick one candidate as the final assistant response
    if candidates:
        # Filter out candidates that look like conversation starters
        # (these sometimes appear incorrectly in the dataset)
        filtered_candidates = []
        for cand in candidates:
            cand_lower = cand.lower().strip()
            # Skip greetings that don't make sense in context
            if len(history) > 2:  # If we're deep in conversation
                if any(
                    phrase in cand_lower
                    for phrase in [
                        "hi there",
                        "hello",
                        "how are you today",
                        "hey there",
                        "hi, how are you",
                        "hello, how are you",
                    ]
                ):
                    continue  # Skip this candidate
            filtered_candidates.append(cand)

        # Use filtered candidates if any remain, otherwise fall back to all
        candidates_to_use = filtered_candidates if filtered_candidates else candidates
        response = random.choice(candidates_to_use)

        # Assistant always uses proper casing
        response = simple_truecase(response) if response else response
        messages.append({"role": "assistant", "content": text_formatter(response)})

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_messages(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """Format message arrays with unified system/developer prompts."""
    assert len(keys) == 1, "'keys' should have a length of 1"

    # Get messages array
    messages = document.get(keys[0], [])

    # Start with unified system prompt
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Track if we've seen a system message to convert to developer
    developer_content = None

    # Filter out any empty messages and apply text formatting
    for message in messages:
        # We just hardcode the processing of NousResearch/Hermes-3-Dataset here
        if message.get("from"):
            KEY_MAP = {"human": "user", "gpt": "assistant"}
            message["role"] = KEY_MAP.get(message["from"])
        if message.get("value"):
            message["content"] = message["value"]

        content = message.get("content", "").strip()
        if content:
            role = message.get("role", "user")

            # Convert first system message to developer
            if role == "system" and developer_content is None:
                developer_content = text_formatter(content)
                formatted_messages.append(
                    {"role": "developer", "content": developer_content}
                )
            elif role != "system":  # Skip additional system messages
                formatted_message = message.copy()
                formatted_message["content"] = text_formatter(content)
                formatted_messages.append(formatted_message)

    # Add default developer message if none was found
    if developer_content is None:
        formatted_messages.insert(
            1,
            {"role": "developer", "content": DEVELOPER_PROMPTS["engage_conversation"]},
        )

    return tokenizer.apply_chat_template(formatted_messages, tokenize=False) + "\n"


def format_wiki(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> str:
    """Format wiki text with unified system/developer prompts."""
    assert len(keys) == 2, "Wiki format requires exactly 2 keys"
    title = document.get(keys[0], "")
    body = document.get(keys[1], "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["write_article"]},
        {"role": "user", "content": f"Write an article about: {title}"},
        {"role": "assistant", "content": body},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_tool_calling(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> str:
    """
    Format synthetic tool-calling examples for training.
    Generates math problems that require the calc tool.
    """

    # Choose a random operation
    operation = random.choice(["add", "sub", "mul", "div", "sqrt", "exp"])

    if operation == "add":
        # Generate 2-4 numbers for addition
        num_values = random.randint(2, 4)
        values = [random.randint(1, 100_000_000) for _ in range(num_values)]
        result = sum(values)

        if len(values) == 2:
            problem_templates = [
                f"What is {values[0]} + {values[1]}?",
                f"Calculate {values[0]} plus {values[1]}",
                f"Can you add {values[0]} and {values[1]} for me?",
                f"What's the sum of {values[0]} and {values[1]}?",
            ]
            result_phrase = f"The sum of {values[0]} and {values[1]} is {result}."
        else:
            values_str = " + ".join(map(str, values))
            problem_templates = [
                f"What is {values_str}?",
                f"Calculate the sum: {values_str}",
                f"Add these numbers: {', '.join(map(str, values))}",
            ]
            result_phrase = f"The sum of {', '.join(map(str, values))} is {result}."

    elif operation == "sub":
        # Generate 2-3 numbers for subtraction
        num_values = random.randint(2, 3)
        values = [random.randint(1, 100_000_000) for _ in range(num_values)]
        result = values[0]
        for v in values[1:]:
            result -= v

        if len(values) == 2:
            problem_templates = [
                f"What is {values[0]} - {values[1]}?",
                f"Subtract {values[1]} from {values[0]}",
                f"What's {values[0]} minus {values[1]}?",
            ]
            result_phrase = f"{values[0]} minus {values[1]} equals {result}."
        else:
            values_str = " - ".join(map(str, values))
            problem_templates = [
                f"What is {values_str}?",
                f"Calculate: {values_str}",
            ]
            result_phrase = f"The result of {values_str} is {result}."

    elif operation == "mul":
        # Generate 2-3 numbers for multiplication
        num_values = random.randint(2, 3)
        values = [random.randint(1, 10000) for _ in range(num_values)]
        result = 1
        for v in values:
            result *= v

        if len(values) == 2:
            problem_templates = [
                f"What is {values[0]} × {values[1]}?",
                f"Multiply {values[0]} by {values[1]}",
                f"What's {values[0]} times {values[1]}?",
            ]
            result_phrase = f"{values[0]} times {values[1]} equals {result}."
        else:
            values_str = " × ".join(map(str, values))
            problem_templates = [
                f"What is {values_str}?",
                f"Calculate the product: {', '.join(map(str, values))}",
            ]
            result_phrase = f"The product of {', '.join(map(str, values))} is {result}."

    elif operation == "div":
        # Generate 2 numbers for division
        b = random.randint(2, 1000)
        a = b * random.randint(1, 10000)
        values = [a, b]
        result = a / b

        problem_templates = [
            f"What is {a} ÷ {b}?",
            f"Divide {a} by {b}",
            f"What's {a} divided by {b}?",
        ]
        result_phrase = f"{a} divided by {b} equals {result}."

    elif operation == "sqrt":
        # Generate a perfect square for nice results
        base = random.randint(1, 1000)
        values = [base * base]
        result = base

        problem_templates = [
            f"What is the square root of {values[0]}?",
            f"Calculate √{values[0]}",
            f"Find the square root of {values[0]}",
        ]
        result_phrase = f"The square root of {values[0]} is {result}."

    else:  # exp
        # Generate base and exponent
        base = random.randint(2, 20)
        exp = random.randint(2, 5)
        values = [base, exp]
        result = math.pow(base, exp)

        problem_templates = [
            f"What is {base}^{exp}?",
            f"Calculate {base} to the power of {exp}",
            f"What's {base} raised to the {exp}th power?",
        ]
        result_phrase = f"{base} to the power of {exp} equals {result:.0f}."

    user_prompt = random.choice(problem_templates)

    # Build the conversation with unified system/developer prompts and tool usage
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["use_tools"]},
        {"role": "user", "content": user_prompt},
    ]

    # 50% chance to call get_tools() first before using calc
    if random.random() < 0.5:
        from praxis.tools import get_tools_json_schema

        tools_json = json.dumps(get_tools_json_schema(), indent=2)

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"<tool_call>\n{json.dumps({'name': 'get_tools', 'arguments': {}}, indent=2)}\n</tool_call>",
                },
                {"role": "tool", "content": tools_json},
            ]
        )

    # Always call calc tool
    messages.extend(
        [
            {
                "role": "assistant",
                "content": f"<tool_call>\n{json.dumps({'name': 'calc', 'arguments': {'values': values, 'op': operation}}, indent=2)}\n</tool_call>",
            },
            {"role": "tool", "content": str(float(result))},
            {"role": "assistant", "content": result_phrase},
        ]
    )

    # Apply chat template without tools parameter
    return tokenizer.apply_chat_template(messages, tokenize=False) + "\n"


def format_cot(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Format Chain of Thought dataset for training using chat templates.

    Uses the tokenizer's chat template to properly format the conversation
    with user prompt and assistant response (containing thinking tags).

    Returns a dict with text and metadata for token-level reward computation.
    """
    assert len(keys) >= 2, "CoT format requires at least 2 keys (prompt, response)"

    prompt = text_formatter(document.get(keys[0], ""))
    response = text_formatter(document.get(keys[1], ""))
    category = document.get(keys[2], "unknown") if len(keys) > 2 else "unknown"
    topic = document.get(keys[3], "unknown") if len(keys) > 3 else "unknown"

    # Use chat template with unified system/developer prompts
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["think_step_by_step"]},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Detect which CoT tags are present in the response
    tags_present = []
    for tag_type in ["wrapper", "thinking_components"]:
        for tag_name, (open_tag, close_tag) in COT_TAGS[tag_type].items():
            if open_tag in response and close_tag in response:
                tags_present.append(tag_name)

    # Return dict format similar to format_rl for metadata pipeline
    return {
        "text": formatted_text,
        "reward": 0.0,  # Sequence-level reward (not used for CoT)
        "cot_metadata": {
            "tags_present": tags_present,
            "category": category,
            "topic": topic,
            "has_cot": len(tags_present) > 0,
        },
    }


def format_rl(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Format RL dataset for generation-based reinforcement learning.

    For proper RL, we format the prompt for generation and store metadata
    for evaluation. The actual response will be generated during training.

    Returns a dict with special formatting for RL.
    """
    assert len(keys) == 3, "RL format requires exactly 3 keys"

    prompt = text_formatter(document.get(keys[0], ""))
    verification_info = document.get(keys[1], "{}")
    solve_rate = document.get(keys[2], 0.0)

    # Parse the ground truth from verification_info
    import json

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


def format_soda(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> str:
    """Formats a single SODA example using the tokenizer's chat template."""
    # Example: ["Veda", "Priest", "Veda", "Priest", "Veda", "Priest"]
    turns = document[keys[0]]

    # Example: "Veda thought about going to church because she was interested in the religion..."
    narrative = document[keys[1]]

    # Example: "Veda was interested in going to church. Veda thought about going to church."
    literal = document[keys[2]]

    # Example: ["Hi, Father. I'm Veda. I'm new to the area...", "Of course, Veda. Our church is based on..."]
    dialogue = document[keys[3]]

    # Example: "PersonX thought about going to church"
    head = document[keys[4]]

    # Example: "xNeed" (relationship type between cause and effect)
    relation = document[keys[5]]

    # Example: "to be interested in going to church"
    tail = document[keys[6]]

    # Create person mapping first
    person_mapping = create_person_mapping(document)

    # Get speaker roles
    unique_speakers = list(dict.fromkeys(turns))  # preserve order, remove duplicates
    speaker_roles = {}

    # Always map first two speakers to user/assistant
    if len(unique_speakers) >= 1:
        speaker_roles[unique_speakers[0]] = "user"
    if len(unique_speakers) >= 2:
        speaker_roles[unique_speakers[1]] = "assistant"

    # Map any additional speakers to "other"
    for speaker in unique_speakers[2:]:
        speaker_roles[speaker] = "other"

    # Create system message content with optional corruption
    system_content = ""

    # Random corruption encourages partial system context prompts
    corruption_chance = 0.5

    # Always add role mappings
    for speaker, role in speaker_roles.items():
        if random.random() < corruption_chance:
            system_content += f"{role}: {speaker}\n"

    # Randomly include each context element (50% chance each)
    if random.random() < corruption_chance:
        system_content += f"cause: {replace_person_references(head, person_mapping)}\n"
    if random.random() < corruption_chance:
        system_content += f"relation: {relation[1:]}\n"
    if random.random() < corruption_chance:
        system_content += f"effect: {replace_person_references(tail, person_mapping)}\n"
    if random.random() < corruption_chance:
        system_content += f"context: {narrative}\n"
    if random.random() < corruption_chance:
        system_content += f"thought: ({literal})\n"

    # Create messages array with unified system and soda context as developer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": system_content.strip() or DEVELOPER_PROMPTS["soda_dialogue"],
        },
    ]

    # Add conversation turns
    for speaker, message in zip(turns, dialogue):
        role = speaker_roles[speaker]
        messages.append({"role": role, "content": text_formatter(message)})

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
    DataFormat.MESSAGES: format_messages,
    DataFormat.SODA: format_soda,
    DataFormat.WIKI: format_wiki,
    DataFormat.RL: format_rl,
    DataFormat.COT: format_cot,
    DataFormat.TOOL_CALLING: format_tool_calling,
}


def get_datamodules(
    seed: int,
    dev: bool,
    pile: bool,
    phi: bool,
    source: bool,
    tokenizer,
    hparams,
    data_path,
    rl_type: Optional[str] = None,
    *args,
):
    print(f"[RL] get_dataintegrations called with rl_type={rl_type}")

    print("Training datasets:")
    train_data = []
    config = get_dataset_configs(dev, pile, phi, rl_type)
    for c in config["primary"]:
        # load configs for huggingface datasets
        print(dict(path=c["path"], weight=c["weight"]))
        train_data.append(get_dataset("huggingface", tokenizer, seed, c, *args))

    # Add synthetic tool-calling dataset if phi is enabled
    if phi:
        train_data.append(get_dataset("synthetic-tool-calling", tokenizer, seed))
        print("[CMD] Initialized SyntheticToolCallingDataset.")

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
        # Load any module-provided datasets
        try:
            from cli import integration_loader_with_conditions

            available_datasets = (
                integration_loader_with_conditions.integration_registry.get(
                    "datasets", {}
                )
            )
            # Process all available integration datasets
            # The integrations themselves will check if they're properly initialized
            for dataset_name in available_datasets:
                print(f"[Integrations] Checking dataset: {dataset_name}")
                dataset = get_dataset(dataset_name, tokenizer, seed)
                if dataset is not None:
                    print(f"[Integrations] Adding dataset: {dataset_name}")
                    train_data.append(dataset)
                else:
                    print(
                        f"[Integrations] Skipping dataset: {dataset_name} (not available)"
                    )
        except ImportError:
            pass  # Integration loader not available

    print("Validation data:")

    validation_data = []
    if len(config["validation"]) > 0:
        for c in config["validation"]:
            print(dict(path=c["path"], weight=c["weight"]))
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
        rl_type=rl_type,
    )

    return train_dataloader


def get_dataset(format, tokenizer, seed, *args, **kwargs):
    # Check if this is a module-provided dataset
    try:
        from cli import integration_loader_with_conditions

        dataset_provider = integration_loader_with_conditions.get_dataset(format)
        if dataset_provider:
            # Call the provider function with standard arguments
            dataset = dataset_provider(tokenizer, seed, *args, **kwargs)
            # Check if the provider returned a valid dataset
            if dataset is not None:
                # Set default weight if not already set
                if not hasattr(dataset, "weight"):
                    dataset.weight = 1.0
                return dataset
            else:
                # Provider returned None (e.g., integration not properly initialized)
                return None
    except ImportError:
        pass

    if format == "huggingface":
        dataset = HuggingfaceDataset(tokenizer, seed, *args)
        dataset.weight = args[0].get("weight", 1.0)
        return dataset
    elif format == "directory":
        dataset = MultiDirectoryDataset(tokenizer, directories=kwargs.get("data_path"))
        dataset.weight = DIR_WEIGHT
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
        dataset.weight = SRC_WEIGHT
        return dataset
    elif format == "synthetic-tool-calling":
        dataset = SyntheticToolCallingDataset(tokenizer, seed, {})
        dataset.weight = TOOLS_WEIGHT
        return dataset


def add_collection(config, collection_name, target_key):
    """Add datasets from a collection to the config with their weights"""
    if collection_name in DATASET_COLLECTIONS:
        for dataset_name, weight in DATASET_COLLECTIONS[collection_name].items():
            dataset_config = HUGGINGFACE_DATASETS.get(dataset_name).copy()
            dataset_config["weight"] = weight
            config[target_key].append(dataset_config)
    return config


def get_dataset_configs(
    dev: bool, pile: bool, phi: bool, rl_type: Optional[str] = None
):
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
            # Add RL datasets even in dev mode if RL is enabled
            if rl_type:
                if rl_type in ["cot", "cot-reinforce"]:
                    config = add_collection(config, "cot", "primary")
                else:
                    config = add_collection(config, "rl", "primary")
        else:
            if rl_type:
                # Use different dataset collections based on RL type
                if rl_type in ["cot", "cot-reinforce"]:
                    config = add_collection(config, "cot", "primary")
                else:
                    config = add_collection(config, "rl", "primary")
            config = add_collection(config, "validation", "validation")

    # Debug: print RL status
    if rl_type:
        print(
            f"[RL] RL enabled with algorithm '{rl_type}', {len([e for e in config['primary'] if 'RL' in e.get('path', '')])} RL datasets in config"
        )

    return config


class InterleaveDataManager:
    # Dynamic weighting control (hardcoded switch)
    use_dynamic_weights = True  # Set to False to use static weights
    ema_alpha = 0.1  # EMA smoothing factor (lower for more conservative updates)

    # Class variable to store shared weights across all instances
    # This is needed because DataLoader workers create separate instances
    shared_weights = None
    shared_weights_initialized = False

    def __init__(
        self,
        samplers,
        weights,
        tokenizer,
        block_size,
        text_cache_size=100_000,
        rl_type=None,
    ):
        self.samplers = samplers
        self.static_weights = weights.copy()  # Store original weights
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_cache_size = text_cache_size
        self.rl_type = rl_type
        self.token_stream = torch.tensor(
            [], dtype=torch.long
        )  # Single continuous stream
        # Track sequences and their boundaries
        self.sequence_boundaries = (
            []
        )  # List of (start_idx, end_idx, reward, metadata) tuples
        self.current_stream_offset = 0  # Track position in token stream

        # Dynamic weighting metrics
        if self.use_dynamic_weights:
            self.sampling_count = 0  # Total number of samplings
            self.sampler_metrics = {}
            for i, sampler in enumerate(self.samplers):
                dataset_name = getattr(sampler, "dataset_path", f"sampler_{i}")
                self.sampler_metrics[i] = {
                    "name": dataset_name,
                    "avg_doc_length": None,  # Will be initialized on first sample
                    "total_samples": 0,  # Total times sampled
                    "total_tokens": 0,  # Total tokens consumed
                }
            # Initialize dynamic weights for this instance
            # Each dataset (train/val) maintains its own weights based on its sampler count
            self.dynamic_weights = self.static_weights.copy()

            # Only share weights between workers of the same dataset type (train OR val)
            # Check if this is training by looking at the number of samplers
            num_samplers = len(self.samplers)
            if (
                InterleaveDataManager.shared_weights_initialized
                and InterleaveDataManager.shared_weights is not None
                and len(InterleaveDataManager.shared_weights) == num_samplers
            ):
                # Use shared weights only if they match our sampler count
                self.dynamic_weights = InterleaveDataManager.shared_weights.copy()
            elif not InterleaveDataManager.shared_weights_initialized:
                # First instance - initialize shared weights for training
                InterleaveDataManager.shared_weights = self.dynamic_weights.copy()
                InterleaveDataManager.shared_weights_initialized = True

            # Always use dynamic weights when enabled
            self.weights = self.dynamic_weights
        else:
            # Static weights mode
            self.weights = weights

    def get_batch(
        self,
        batch_size: int,
        oversample: bool = False,
        supersample: bool = False,
        hypersample: bool = False,
    ) -> Dict[str, Any]:
        sequence_length = self.block_size
        current_batch_size = batch_size

        # Update weights if using dynamic weighting and they match our sampler count
        if self.use_dynamic_weights:
            # Only use shared weights if they match our sampler count
            if InterleaveDataManager.shared_weights is not None and len(
                InterleaveDataManager.shared_weights
            ) == len(self.samplers):
                self.weights = InterleaveDataManager.shared_weights
            else:
                # Use our own dynamic weights (validation or mismatched sampler count)
                self.weights = self.dynamic_weights
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
        rewards = [] if self.rl_type else None
        metadata = [] if self.rl_type else None
        token_weights = [] if self.rl_type and self.rl_type == "cot" else None

        for i in range(current_batch_size):
            start = i * sequence_length
            end = start + sequence_length
            batch.append(self.token_stream[start:end])

            # Find the reward and metadata for this sequence chunk if RL is enabled
            if self.rl_type:
                sequence_reward, sequence_metadata = (
                    self._get_reward_and_metadata_for_range(start, end)
                )
                rewards.append(sequence_reward)
                if metadata is not None:
                    metadata.append(sequence_metadata)

                # Extract token weights for CoT
                if (
                    token_weights is not None
                    and sequence_metadata.get("token_weights") is not None
                ):
                    weights = sequence_metadata["token_weights"]
                    # Ensure it's a tensor and matches the sequence length
                    if isinstance(weights, torch.Tensor):
                        original_length = weights.shape[0]
                        if weights.shape[0] > sequence_length:
                            weights = weights[:sequence_length]
                            print(
                                f"[Builder] Truncating token weights from {original_length} to {sequence_length}"
                            )
                        elif weights.shape[0] < sequence_length:
                            padding = torch.ones(sequence_length - weights.shape[0])
                            weights = torch.cat([weights, padding])
                            print(
                                f"[Builder] Padding token weights from {original_length} to {sequence_length}"
                            )

                        # Log when we're adding CoT weights to a batch
                        non_default = (weights != 1.0).sum().item()
                        if non_default > 0:
                            print(
                                f"[Builder] Adding CoT weights to batch: {non_default}/{len(weights)} non-default tokens"
                            )

                        token_weights.append(weights)
                    else:
                        # Default weights if no token weights available
                        token_weights.append(torch.ones(sequence_length))
                elif token_weights is not None:
                    # Default weights for non-CoT sequences
                    token_weights.append(torch.ones(sequence_length))

        # Remove used tokens from the stream and update boundaries
        self.token_stream = self.token_stream[tokens_needed:]
        self._update_boundaries_after_removal(tokens_needed)

        # Include current weights in the return value for logging
        result = {
            "batch": batch,
            "rewards": rewards,
            "metadata": metadata,
            "token_weights": token_weights,
        }

        # Add the current sampler weights if dynamic weighting is enabled
        if self.use_dynamic_weights:
            result["sampler_weights"] = (
                self.weights.copy() if hasattr(self, "weights") else None
            )

        return result

    def _get_reward_and_metadata_for_range(
        self, start: int, end: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Get the reward and metadata for a token range, handling sequence boundaries properly.

        This implements sequence-level reward assignment where:
        - If a chunk is entirely within one sequence, it gets that sequence's full reward and metadata
        - If a chunk spans multiple sequences, it gets a weighted average reward and metadata from the dominant sequence
        - This ensures that long sequences split across batches are rewarded consistently

        Args:
            start: Start index in current token stream
            end: End index in current token stream

        Returns:
            Tuple of (reward value, metadata dict) for this range
        """
        # Adjust indices to account for stream offset
        abs_start = self.current_stream_offset + start
        abs_end = self.current_stream_offset + end

        # Find all sequences that overlap with this range
        overlapping_data = []  # List of (reward, metadata, weight) tuples

        for seq_start, seq_end, reward, metadata in self.sequence_boundaries:
            # Calculate overlap
            overlap_start = max(abs_start, seq_start)
            overlap_end = min(abs_end, seq_end)

            if overlap_start < overlap_end:
                # This sequence overlaps with our range
                overlap_size = overlap_end - overlap_start
                overlapping_data.append((reward, metadata, overlap_size))

        # If we have overlapping sequences, return weighted average reward and dominant metadata
        if overlapping_data:
            # COMMON CASE: If a single sequence fully contains this chunk,
            # give it the full reward and metadata (most chunks will be fully within one sequence)
            for seq_start, seq_end, reward, metadata in self.sequence_boundaries:
                if seq_start <= abs_start and seq_end >= abs_end:
                    return reward, metadata

            # EDGE CASE: Chunk spans multiple sequences (rare)
            # Use weighted average for reward, metadata from dominant sequence
            total_weight = sum(weight for _, _, weight in overlapping_data)
            if total_weight > 0:
                # Weighted average reward
                weighted_reward = (
                    sum(r * w for r, _, w in overlapping_data) / total_weight
                )
                # Metadata from the sequence with most overlap
                dominant_metadata = max(overlapping_data, key=lambda x: x[2])[1]
                return weighted_reward, dominant_metadata

        # No overlap found, return defaults
        return 0.0, {}

    def _update_dynamic_weights_after_sample(self, sampler_idx: int, doc_length: int):
        """Update metrics and weights with EMA after each sample."""
        if not self.use_dynamic_weights:
            return

        metrics = self.sampler_metrics[sampler_idx]

        # Update total counts
        metrics["total_samples"] += 1
        metrics["total_tokens"] += doc_length

        # Update average document length with EMA
        if metrics["avg_doc_length"] is None:
            metrics["avg_doc_length"] = float(doc_length)
        else:
            metrics["avg_doc_length"] = (
                self.ema_alpha * doc_length
                + (1 - self.ema_alpha) * metrics["avg_doc_length"]
            )

        # Calculate target weights based on current metrics
        target_weights = self._calculate_target_weights()

        # Update dynamic weights with EMA towards target
        old_weights = self.dynamic_weights.copy()
        for i in range(len(self.dynamic_weights)):
            self.dynamic_weights[i] = (
                self.ema_alpha * target_weights[i]
                + (1 - self.ema_alpha) * self.dynamic_weights[i]
            )

        # Normalize to ensure weights sum to 1
        total = sum(self.dynamic_weights)
        if total > 0:
            self.dynamic_weights = [w / total for w in self.dynamic_weights]

        # Update the shared class variable only if we're the training dataset
        # (validation datasets maintain their own weights)
        if (
            len(self.samplers) == len(InterleaveDataManager.shared_weights)
            if InterleaveDataManager.shared_weights
            else True
        ):
            InterleaveDataManager.shared_weights = self.dynamic_weights.copy()

    def _calculate_target_weights(self):
        """Calculate target weights based on current metrics."""
        if not self.sampler_metrics:
            return self.static_weights

        # Skip if we don't have enough data yet
        if all(m["avg_doc_length"] is None for m in self.sampler_metrics.values()):
            return self.static_weights

        target_weights = []

        # Calculate average document length across all samplers
        avg_length = sum(
            m["avg_doc_length"]
            for m in self.sampler_metrics.values()
            if m["avg_doc_length"] is not None
        ) / len(self.sampler_metrics)

        # Calculate target based on balancing token consumption
        total_tokens = sum(m["total_tokens"] for m in self.sampler_metrics.values())
        avg_tokens_per_sampler = (
            total_tokens / len(self.sampler_metrics) if total_tokens > 0 else 1
        )

        for i in range(len(self.samplers)):
            metrics = self.sampler_metrics[i]

            # Start with static weight
            weight = self.static_weights[i]

            if metrics["avg_doc_length"] is not None and metrics["total_samples"] > 0:
                # Factor 1: Inverse document length (shorter docs get higher weight)
                length_factor = avg_length / max(metrics["avg_doc_length"], 1.0)

                # Factor 2: Balance token consumption (underrepresented gets boost)
                if metrics["total_tokens"] > 0:
                    token_balance_factor = avg_tokens_per_sampler / max(
                        metrics["total_tokens"], 1.0
                    )
                else:
                    token_balance_factor = 2.0  # Strong boost for never sampled

                # Combine factors - geometric mean for balance
                weight = weight * (length_factor * token_balance_factor) ** 0.5

            target_weights.append(weight)

        # Normalize weights to sum to 1
        total = sum(target_weights)
        if total > 0:
            return [w / total for w in target_weights]
        else:
            return self.static_weights

    def _update_boundaries_after_removal(self, tokens_removed: int):
        """Update sequence boundaries after removing tokens from the stream."""
        self.current_stream_offset += tokens_removed

        # Remove boundaries that are now completely before the current stream
        self.sequence_boundaries = [
            (start, end, reward, metadata)
            for start, end, reward, metadata in self.sequence_boundaries
            if end > self.current_stream_offset
        ]

    def _extend_token_stream(self):
        """Add more tokens to our stream when needed, tracking sequence boundaries."""
        sequences_to_add = []
        total_text = ""

        # Collect sequences until we have enough text
        while len(total_text) < self.text_cache_size:
            # Pick a sampler based on current weights
            sampler_idx = random.choices(
                range(len(self.samplers)), weights=self.weights, k=1
            )[0]
            sampler = self.samplers[sampler_idx]
            # Get a sequence from that sampler
            new_sequences = sampler.get_sequences(1)
            text = new_sequences[0]

            # Update dynamic weights after each sample
            if self.use_dynamic_weights:
                self.sampling_count += 1
                doc_length = len(text)
                self._update_dynamic_weights_after_sample(sampler_idx, doc_length)

            # Track dataset sampling
            dataset_name = getattr(sampler, "dataset_path", "unknown")

            # Get reward and metadata for this sequence if applicable
            reward = 0.0
            metadata = {}
            has_reward = False

            if self.rl_type and hasattr(sampler, "reward_cache"):
                import hashlib

                text_hash = hashlib.md5(text.encode()).hexdigest()
                cache_data = sampler.reward_cache.get(text_hash, None)

                if cache_data is None:
                    reward = 0.0
                    metadata = {}
                elif isinstance(cache_data, dict):
                    reward = cache_data.get("reward", 0.0)
                    metadata = cache_data  # Store the full metadata
                else:
                    # Legacy format
                    reward = cache_data if isinstance(cache_data, (int, float)) else 0.0
                    metadata = {"reward": reward}

                # Only log interesting reward events
                if reward == -1:
                    print(f"[RL] Found generation sequence from {dataset_name}")
                elif reward > 0:
                    print(f"[RL] Found static reward {reward} from {dataset_name}")

                has_reward = (
                    reward != 0
                )  # Any non-zero reward (including -1 generation flag)

                if has_reward and reward != -1:
                    _rl_logger.log_reward_found(
                        reward, dataset_name
                    )  # Only log static rewards

            _rl_logger.log_dataset_sample(dataset_name, has_reward)

            # Add separator
            text_with_sep = text.rstrip() + "\n"
            sequences_to_add.append(
                (
                    len(total_text),
                    len(total_text) + len(text_with_sep),
                    reward,
                    metadata,
                )
            )
            total_text += text_with_sep

        # Tokenize the entire text at once
        tokens = self.tokenizer(
            text=total_text,
            padding=False,
            return_tensors="pt",
        )[
            "input_ids"
        ].squeeze(0)

        # Convert character positions to token positions
        # This is approximate but should work well enough
        chars_per_token = len(total_text) / len(tokens) if len(tokens) > 0 else 1.0
        current_pos = self.current_stream_offset + len(self.token_stream)

        for char_start, char_end, reward, metadata in sequences_to_add:
            # Estimate token positions based on character positions
            token_start = current_pos + int(char_start / chars_per_token)
            token_end = current_pos + int(char_end / chars_per_token)

            # Ensure we have at least one token per sequence
            if token_end <= token_start:
                token_end = token_start + 1

            # For CoT sequences, compute token-level rewards/weights
            if metadata.get("cot_metadata") is not None:
                # Extract the sequence text
                sequence_text = total_text[char_start:char_end]
                # Compute token-level weights for this sequence
                token_weights = self._compute_cot_token_weights(
                    sequence_text,
                    token_start - current_pos,  # Local start position
                    token_end - current_pos,  # Local end position
                    tokens,
                    metadata["cot_metadata"],
                )
                metadata["token_weights"] = token_weights

                # Validation logging
                if metadata["cot_metadata"].get("has_cot", False):
                    non_default = (token_weights != 1.0).sum().item()
                    print(f"[Builder] CoT sequence detected:")
                    print(f"  Text length: {len(sequence_text)} chars")
                    print(
                        f"  Token range: [{token_start - current_pos}, {token_end - current_pos})"
                    )
                    print(f"  Tags present: {metadata['cot_metadata']['tags_present']}")
                    print(f"  Non-default weights: {non_default}/{len(token_weights)}")
                    print(
                        f"  Weight range: [{token_weights.min():.3f}, {token_weights.max():.3f}]"
                    )

                    # Check for potential splitting issues
                    thinking_start = sequence_text.find("<thinking>")
                    thinking_end = sequence_text.find("</thinking>")
                    if thinking_start != -1 and thinking_end == -1:
                        print(
                            f"  ⚠️  WARNING: <thinking> tag opened but not closed - sequence may be split"
                        )
                    elif thinking_start == -1 and thinking_end != -1:
                        print(
                            f"  ⚠️  WARNING: </thinking> tag found but no opening - sequence may be split"
                        )

                    # Show first few chars for context
                    preview = sequence_text[:100].replace("\n", "\\n")
                    print(f"  Preview: {preview}...")

            # Store the boundary information with metadata
            self.sequence_boundaries.append((token_start, token_end, reward, metadata))

        # Add tokens to stream
        self.token_stream = torch.cat([self.token_stream, tokens])

    def _compute_cot_token_weights(
        self, text, local_start, local_end, tokens, cot_metadata
    ):
        """
        Compute token-level weights for CoT sequences based on tag positions.

        Returns a tensor of weights for each token in the sequence.
        """
        seq_length = local_end - local_start
        token_weights = torch.ones(seq_length)

        # If no CoT tags present, return default weights
        if not cot_metadata.get("has_cot", False):
            return token_weights

        # Extract just the tokens for this sequence
        seq_tokens = tokens[local_start:local_end]

        # Decode tokens to get exact character positions
        # We need to map tag regions to token positions
        char_to_token = {}
        current_char = 0

        for i, token_id in enumerate(seq_tokens):
            # Decode single token to get its text
            token_text = self.tokenizer.decode(
                [token_id.item()], skip_special_tokens=False
            )
            char_to_token[current_char] = i
            current_char += len(token_text)

        # Find tag regions in the text and map to tokens
        regions_found = []
        incomplete_tags = []

        for tag_type in ["wrapper", "thinking_components"]:
            for tag_name, (open_tag, close_tag) in COT_TAGS[tag_type].items():
                if tag_name in cot_metadata.get("tags_present", []):
                    weight = COT_TAGS["tag_weights"].get(tag_name, 1.0)

                    # Find all occurrences of this tag pair
                    start_pos = 0
                    while True:
                        open_pos = text.find(open_tag, start_pos)
                        if open_pos == -1:
                            break
                        close_pos = text.find(close_tag, open_pos + len(open_tag))

                        if close_pos == -1:
                            # Handle incomplete tag (split sequence) - apply weight from open tag to end
                            print(
                                f"[Builder] Incomplete {tag_name} tag detected - applying weight to end of sequence"
                            )
                            token_start = 0
                            for char_pos, token_pos in sorted(char_to_token.items()):
                                if char_pos <= open_pos:
                                    token_start = token_pos
                            token_weights[token_start:] = weight
                            incomplete_tags.append(tag_name)
                            break
                        else:
                            # Complete tag pair found
                            # Map character positions to token positions
                            token_start = 0
                            token_end = seq_length - 1

                            for char_pos, token_pos in sorted(char_to_token.items()):
                                if char_pos <= open_pos:
                                    token_start = token_pos
                                if char_pos <= close_pos + len(close_tag):
                                    token_end = min(token_pos + 1, seq_length)

                            # Apply weight to tokens in this region
                            token_weights[token_start:token_end] = weight
                            regions_found.append((tag_name, token_start, token_end))

                        start_pos = (
                            close_pos + len(close_tag) if close_pos != -1 else len(text)
                        )

                # Also check for orphaned closing tags (from split sequences)
                close_pos = text.find(close_tag)
                if close_pos != -1 and text.find(open_tag) == -1:
                    print(
                        f"[Builder] Orphaned closing {tag_name} tag detected - applying weight from start"
                    )
                    token_end = seq_length - 1
                    for char_pos, token_pos in sorted(char_to_token.items()):
                        if char_pos <= close_pos + len(close_tag):
                            token_end = min(token_pos + 1, seq_length)
                    token_weights[:token_end] = weight
                    incomplete_tags.append(f"{tag_name}_orphaned")

        # Log findings
        if regions_found or incomplete_tags:
            print(f"[Builder] Token weight mapping for sequence:")
            for tag_name, start, end in regions_found:
                print(
                    f"  Complete {tag_name}: tokens [{start}:{end}] = {COT_TAGS['tag_weights'][tag_name]}"
                )
            for tag in incomplete_tags:
                print(f"  Incomplete/orphaned: {tag}")

        return token_weights


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


def load_dataset_smart(dataset_args: Dict) -> Any:
    """
    Load a dataset, handling cases where metadata files interfere.

    Some datasets (e.g., microsoft/NextCoderDataset-Conversational) have
    state.json or other metadata that gets loaded instead of the actual data.
    This function detects and fixes that by adding data_files="*.arrow".

    Args:
        dataset_args: Arguments to pass to load_dataset

    Returns:
        The loaded dataset
    """

    # First try normal loading
    dataset = load_dataset(**dataset_args)

    # Check if we got metadata instead of real data
    # This is a simple heuristic: real data won't have '_data_files' as a column
    needs_fix = False

    if hasattr(dataset, "column_names"):
        # Single dataset
        if "_data_files" in dataset.column_names:
            needs_fix = True
    elif hasattr(dataset, "keys"):
        # DatasetDict - check first split
        for split in dataset:
            if (
                hasattr(dataset[split], "column_names")
                and "_data_files" in dataset[split].column_names
            ):
                needs_fix = True
                break

    if needs_fix:
        # Reload with data_files to skip metadata
        print(f"Note: Fixing metadata issue for {dataset_args.get('path', 'dataset')}")
        fixed_args = dataset_args.copy()
        fixed_args["data_files"] = "*.arrow"
        return load_dataset(**fixed_args)

    return dataset


class SyntheticToolCallingDataset(PraxisSampler):
    """Generates synthetic tool-calling examples for training."""

    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int, config: Dict):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.format_handler = format_tool_calling
        self.dataset_path = "synthetic-tool-calling"

    def fill_sequence_cache(self):
        # Generate a synthetic document (empty since we generate everything in the formatter)
        document = {}
        formatted = self.format_handler(document, [], self.tokenizer)
        self.sequence_cache.append(formatted)


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
        self.dataset = load_dataset_smart(dataset_args)
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

        # Mix simple math for RL datasets
        self.mix_simple_math = config.get("mix_simple_math", False)
        if self.mix_simple_math:
            from praxis.datasets.simple_math import SimpleMathDataset

            self.simple_math = SimpleMathDataset(
                mix_ratio=0.95
            )  # 95% simple problems to force generation

    def fill_sequence_cache(self):
        try:
            # Mix in simple math problems for RL
            if (
                self.mix_simple_math
                and hasattr(self, "simple_math")
                and self.simple_math.should_use_simple()
            ):
                # Generate a simple problem
                simple_problem = self.simple_math.generate()
                document = self.simple_math.format_for_rl(simple_problem)
                # Log when we use simple math
                if not hasattr(self, "_simple_count"):
                    self._simple_count = 0
                self._simple_count += 1
                if (
                    self._simple_count % 10 == 1
                ):  # More frequent logging to see if it's working
                    print(
                        f"[RL] Using simple math #{self._simple_count}: {simple_problem['prompt']} = {simple_problem['ground_truth']}"
                    )
            else:
                if self.mix_simple_math:
                    print(
                        f"[RL DEBUG] Not using simple math (should_use={getattr(self, 'simple_math', None) and self.simple_math.should_use_simple() if hasattr(self, 'simple_math') else 'no simple_math'})"
                    )
                document = next(self.dataset_iterator)

            formatted = self._format_document(document)

            # Handle formats that return dicts (RL and CoT)
            if isinstance(formatted, dict):
                text = formatted["text"]

                # Store metadata in reward cache
                import hashlib

                text_hash = hashlib.md5(text.encode()).hexdigest()

                if self.format == DataFormat.RL:
                    # RL format with reward and ground truth
                    reward = formatted["reward"]
                    self.reward_cache[text_hash] = {
                        "reward": reward,
                        "ground_truth": formatted.get("ground_truth", ""),
                        "original_difficulty": formatted.get(
                            "original_difficulty", 0.0
                        ),
                    }
                elif self.format == DataFormat.COT:
                    # CoT format with tag metadata
                    self.reward_cache[text_hash] = {
                        "reward": formatted.get("reward", 0.0),
                        "cot_metadata": formatted.get("cot_metadata", {}),
                    }
                else:
                    # Generic dict format
                    self.reward_cache[text_hash] = formatted

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
            "data",
            "__pycache__",
            "staging",
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
        rl_type: Optional[str] = None,
    ):
        self.data_manager = InterleaveDataManager(
            datasets, weights, tokenizer, block_size, rl_type=rl_type
        )
        self.batch_size = batch_size
        self.oversample_chance = oversample_chance
        self.supersample_chance = supersample_chance
        self.hypersample_chance = hypersample_chance
        self.rl_type = rl_type
        self.tokenizer = tokenizer  # Store tokenizer for reward extraction

        # Debug log
        print(f"[RL] WeightedIterableDataset initialized with rl_type={rl_type}")

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

                # Debug batch reward composition
                if generation_count > 0:
                    print(
                        f"[RL DEBUG] Batch has {generation_count} generation flags out of {len(reward_tensor)} total"
                    )

                # Only log when we actually have generation flags
                if needs_generation:
                    print(
                        f"[RL] Batch ready for generation: {generation_count} sequences need responses"
                    )
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
        rl_type: Optional[str] = None,
    ):
        super().__init__()
        self.rl_type = rl_type
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
        print(f"[RL] Creating WeightedIterableDataset with rl_type={self.rl_type}")

        return WeightedIterableDataset(
            datasets,
            weights,
            tokenizer,
            block_size,
            batch_size,
            oversample_chance,
            supersample_chance,
            hypersample_chance,
            rl_type=self.rl_type,
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
