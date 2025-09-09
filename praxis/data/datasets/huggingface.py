"""HuggingFace dataset implementation."""

import hashlib
from typing import Dict
from transformers import PreTrainedTokenizer

from praxis.data.formats import DataFormat
from praxis.data.datasets.base import PraxisSampler, load_dataset_smart
from praxis.data.formatters import (
    format_simple,
    format_instruction,
    format_conversation,
    format_personachat,
    format_messages,
    format_soda,
    format_wiki,
    format_rl,
    format_cot,
    format_tool_calling,
    _rl_logger,
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
            from praxis.data.datasets.simple_math import SimpleMathDataset

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
            # Try again with the new iterator
            try:
                document = next(self.dataset_iterator)
                formatted = self._format_document(document)
                
                # Handle formats that return dicts (RL and CoT)
                if isinstance(formatted, dict):
                    text = formatted["text"]
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    
                    if self.format == DataFormat.RL:
                        # RL format with reward and ground truth
                        reward = formatted["reward"]
                        self.reward_cache[text_hash] = {
                            "reward": reward,
                            "ground_truth": formatted.get("ground_truth", ""),
                            "original_difficulty": formatted.get("original_difficulty", 0.0),
                        }
                        
                        # Add generation flag if needed
                        if formatted.get("reward") == -1:
                            _rl_logger.log_dataset_sample(self.dataset_path, True)
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
                # Dataset is empty or has issues, add a placeholder to prevent infinite loop
                print(f"WARNING: Dataset '{self.dataset_path}' appears to be empty or has issues. Adding placeholder.")
                self.sequence_cache.append("")

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