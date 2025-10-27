"""HuggingFace dataset implementation."""

import hashlib
from typing import Dict

from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler, load_dataset_smart
from praxis.data.formats import DataFormat
from praxis.data.formatters import (
    _rl_logger,
    format_conversation,
    format_cot,
    format_instruction,
    format_messages,
    format_personachat,
    format_rl,
    format_simple,
    format_soda,
    format_tool_calling,
    format_wiki,
)

# Format handlers mapping
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


class HuggingfaceDataset(PraxisSampler):
    """Dataset that loads from HuggingFace datasets library."""

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

        # Store base seed and restart counter
        self.base_seed = seed
        self.restart_count = 0
        self.is_streaming = config.get("streaming", True)

        dataset_args = dict(
            path=self.dataset_path,
            split=config.get("split", "train"),
            streaming=self.is_streaming,
            trust_remote_code=config.get("trust_remote_code", False),
        )
        if "name" in config:
            dataset_args["name"] = config["name"]
        self.dataset = load_dataset_smart(dataset_args)

        # Initial shuffle with base seed
        shuffle_args = {"seed": self.base_seed}
        if self.is_streaming:
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
            from praxis.data.datasets.simple_math import SimpleMathDataset

            self.simple_math = SimpleMathDataset(
                mix_ratio=0.95
            )  # 95% simple problems to force generation

    def get_document(self) -> Dict:
        """Get a formatted document with messages and metadata."""
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
                try:
                    document = next(self.dataset_iterator)
                except StopIteration:
                    # Restart the dataset
                    self.restart_count += 1
                    new_seed = self.base_seed + self.restart_count
                    shuffle_args = {"seed": new_seed}
                    if self.is_streaming:
                        shuffle_args["buffer_size"] = 1000
                    self.shuffled_dataset = self.dataset.shuffle(**shuffle_args)
                    self.dataset_iterator = iter(self.shuffled_dataset)
                    document = next(self.dataset_iterator)

            # Debug what keys the document has
            if not hasattr(self, "_debug_printed"):
                self._debug_printed = True

            formatted = self._format_document(document)

            # Handle different formatter return types
            if isinstance(formatted, dict):
                # Check if it's the new format (with messages)
                if "messages" in formatted:
                    # New format - return as-is
                    return formatted
                elif "text" in formatted:
                    # Old RL/CoT format - needs conversion
                    # For now, return empty to avoid breaking
                    return {"messages": [], "metadata": {}}
            else:
                # Legacy text format - skip
                return {"messages": [], "metadata": {}}

        except Exception as e:
            print(f"[ERROR] HuggingfaceDataset.get_document failed: {e}")
            import traceback

            traceback.print_exc()
            return {"messages": [], "metadata": {}}

    def fill_sequence_cache(self):
        """Legacy method for compatibility - converts to old text format."""
        try:
            document_data = self.get_document()

            # Convert back to text for legacy compatibility
            if document_data and document_data.get("messages"):
                text = (
                    self.tokenizer.apply_chat_template(
                        document_data["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    + "\n"
                )

                # Store metadata if needed
                if self.format == DataFormat.RL and "reward" in document_data.get(
                    "metadata", {}
                ):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    self.reward_cache[text_hash] = document_data["metadata"]

                self.sequence_cache.append(text)
            else:
                # Empty document, try again
                self.fill_sequence_cache()
        except StopIteration:
            HuggingfaceDataset.counts[self.dataset_path] += 1
            self.restart_count += 1

            # Log every restart so we know when datasets are consumed too quickly
            print(
                f"INFO: Reached the last batch of '{self.dataset_path}' dataset. Reshuffling with new seed. ({HuggingfaceDataset.counts[self.dataset_path]}x)"
            )

            # Reshuffle with a new seed to avoid repeating the same pattern
            new_seed = self.base_seed + self.restart_count
            shuffle_args = {"seed": new_seed}
            if self.is_streaming:
                shuffle_args["buffer_size"] = 1000
            self.shuffled_dataset = self.dataset.shuffle(**shuffle_args)
            self.dataset_iterator = iter(self.shuffled_dataset)

            # Try again with the new iterator
            try:
                document_data = self.get_document()

                # Convert back to text for legacy compatibility
                if document_data and document_data.get("messages"):
                    text = (
                        self.tokenizer.apply_chat_template(
                            document_data["messages"],
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        + "\n"
                    )

                    # Store metadata if needed
                    if self.format == DataFormat.RL and "reward" in document_data.get(
                        "metadata", {}
                    ):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        self.reward_cache[text_hash] = document_data["metadata"]

                    self.sequence_cache.append(text)
                else:
                    # Empty document, add placeholder
                    self.sequence_cache.append("")
            except StopIteration:
                # Dataset is empty or has issues, add a placeholder to prevent infinite loop
                print(
                    f"WARNING: Dataset '{self.dataset_path}' appears to be empty or has issues. Adding placeholder."
                )
                self.sequence_cache.append("")

    def _format_document(self, document):
        return self.format_handler(document, self.keys, self.tokenizer)

    def state_dict(self):
        # Get the internal state of the shuffled dataset and restart counter
        return {
            "dataset_state": self.shuffled_dataset.state_dict(),
            "restart_count": self.restart_count,
        }

    def load_state_dict(self, state_dict):
        # Restore the restart counter
        self.restart_count = state_dict.get("restart_count", 0)
        # Restore the internal state so iteration picks up where we left off
        if "dataset_state" in state_dict:
            self.shuffled_dataset.load_state_dict(state_dict["dataset_state"])
        else:
            # Old format compatibility
            self.shuffled_dataset.load_state_dict(state_dict)
        # Recreate the iterator from the restored state
        self.dataset_iterator = iter(self.shuffled_dataset)
