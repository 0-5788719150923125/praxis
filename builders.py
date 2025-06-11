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
    COT = "cot"


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

DEFAULT_WEIGHT = 1.0
SRC_WEIGHT = 0.1
DIR_WEIGHT = 2.0
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
    cot={
        "chain-of-thought": DEFAULT_WEIGHT * 0.1,
    },
)

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
    backtick_pattern = r"(```)\n(?![ \t]|[-*•+] |[0-9]+[\.\)] )([\"\']*[A-Z])"
    backtick_replacement = r"\1\n\n\2"
    text = re.sub(backtick_pattern, backtick_replacement, text)

    # Define the pattern for paragraph boundaries
    # Look for:
    # 1. Lines ending with sentence-ending punctuation (., !, ?) - these are likely complete thoughts
    # 2. Followed by a single newline
    # 3. NOT followed by indentation, list markers, or code keywords
    # 4. Followed by an optional quotation mark and then an uppercase letter
    #
    # EXCLUDE lines ending with:
    # - Colons (:) - these are typically labels or keys
    # - Commas, semicolons - these are mid-sentence
    # - Letters/numbers without punctuation - these might be labels
    pattern = r"([.!?][\"\']*[)\]]*)(\n)(?![ \t]|[-*•+] |[0-9]+[\.\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await )([\"\']*[A-Z])"

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

    # Use chat template for proper formatting
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Add EOS token if not already added by the template
    if tokenizer.eos_token and not formatted_text.endswith(tokenizer.eos_token):
        formatted_text += tokenizer.eos_token

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

    # Format just the prompt (no answer) with generation prompt
    messages = [{"role": "user", "content": prompt}]

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
    DataFormat.COT: format_cot,
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
    rl_type: Optional[str] = None,
    *args,
):
    print(f"[RL] get_datamodules called with rl_type={rl_type}")

    # An important warning
    if gun and seed and not dev:
        print(
            "WARNING: GUN chats are never deterministic, and cannot be reproduced when using a `seed`. You should omit the `--gun` argument for experiments."
        )
        time.sleep(5)

    train_data = []
    config = get_dataset_configs(dev, pile, phi, rl_type)
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
        rl_type=rl_type,
    )

    return train_dataloader


def get_dataset(format, tokenizer, seed, *args, **kwargs):
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
            config = add_collection(config, "redpajama", "validation")
    print("training on:")
    [
        print(f"dataset: {entry['path']}, weight: {entry['weight']}")
        for entry in config["primary"]
    ]

    # Debug: print RL status
    if rl_type:
        print(
            f"[RL] RL enabled with algorithm '{rl_type}', {len([e for e in config['primary'] if 'RL' in e.get('path', '')])} RL datasets in config"
        )

    return config


class InterleaveDataManager:
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
        self.weights = weights
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

        return {
            "batch": batch,
            "rewards": rewards,
            "metadata": metadata,
            "token_weights": token_weights,
        }

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
            # Pick a sampler based on weights
            sampler = random.choices(self.samplers, weights=self.weights, k=1)[0]
            # Get a sequence from that sampler
            new_sequences = sampler.get_sequences(1)
            text = new_sequences[0]

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
            text_with_sep = text + self.tokenizer.eos_token + "\n"
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
                    yield result_dict
                else:
                    # Log batch statistics
                    _rl_logger.log_batch(reward_tensor)

                    # Return regular RL format
                    result_dict = {"input_ids": batch_tensor, "rewards": reward_tensor}
                    if token_weights is not None:
                        result_dict["token_weights"] = torch.stack(token_weights)
                    yield result_dict
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
