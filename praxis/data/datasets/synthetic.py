"""Synthetic dataset generation."""

from typing import Dict
from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler
from praxis.data.formatters import format_tool_calling


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