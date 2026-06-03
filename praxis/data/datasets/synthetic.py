"""Synthetic dataset generation."""

from typing import Dict

from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler
from praxis.data.formatters import format_print, format_tool_calling


class SyntheticPrintDataset(PraxisSampler):
    """Generates synthetic `print` examples: the model leads with a question
    and predicts the user's short answer."""

    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int, config: Dict):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.format_handler = format_print
        self.dataset_path = "synthetic-print"

    def get_document(self) -> Dict:
        return self.format_handler({}, [], self.tokenizer)

    def fill_sequence_cache(self):
        document_data = self.get_document()
        if document_data and document_data.get("messages"):
            text = self.tokenizer.apply_chat_template(
                document_data["messages"], tokenize=False, add_generation_prompt=False
            )
            self.sequence_cache.append(text)
        else:
            self.sequence_cache.append("")


class SyntheticToolCallingDataset(PraxisSampler):
    """Generates synthetic tool-calling examples for training."""

    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int, config: Dict):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.format_handler = format_tool_calling
        self.dataset_path = "synthetic-tool-calling"

    def get_document(self) -> Dict:
        """Get a synthetic tool-calling document.

        Returns:
            Dictionary with messages and metadata
        """
        # Generate a synthetic document (empty since we generate everything in the formatter)
        document = {}
        return self.format_handler(document, [], self.tokenizer)

    def fill_sequence_cache(self):
        # Legacy method for compatibility - converts to old text format
        document_data = self.get_document()

        # Convert back to text for legacy compatibility
        if document_data and document_data.get("messages"):
            text = self.tokenizer.apply_chat_template(
                document_data["messages"], tokenize=False, add_generation_prompt=False
            )
            self.sequence_cache.append(text)
        else:
            self.sequence_cache.append("")
