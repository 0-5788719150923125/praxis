"""HuggingFace dataset implementation."""

import hashlib
from typing import Dict

from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler, load_dataset_smart
from praxis.data.datasets.network_retry import (
    enter_offline_mode,
    hf_offline,
    hub_reachable,
    is_network_error,
    retry_on_network_error,
)
from praxis.data.formats import DataFormat
from praxis.data.formatters import (
    _rl_logger,
    format_conversation,
    format_cot,
    format_instruction,
    format_joke,
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
    DataFormat.JOKE: format_joke,
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
        if hf_offline() and self.is_streaming:
            # Streams read over HTTP and cache nothing reusable; a full
            # (non-streaming) load CAN resolve from the local datasets cache.
            # Uncached datasets fail fast here and get skipped upstream.
            self.is_streaming = False

        dataset_args = dict(
            path=self.dataset_path,
            split=config.get("split", "train"),
            streaming=self.is_streaming,
            trust_remote_code=config.get("trust_remote_code", False),
        )
        if "name" in config:
            dataset_args["name"] = config["name"]
        # Pin a commit when provided: resolves from cache harder and keeps runs reproducible.
        if "revision" in config:
            dataset_args["revision"] = config["revision"]
        self._dataset_args = dict(dataset_args)  # kept for mid-run cache fallback
        try:
            self.dataset = self._load_frugal(dataset_args)
        except Exception as e:
            if hf_offline() or not is_network_error(e):
                raise
            if hub_reachable():
                # Hub is up, so this failure is specific to this dataset.
                # Raise to skip it upstream; the rest stay online.
                print(
                    f"[DATA] hub reachable but {self.dataset_path} failed "
                    f"({type(e).__name__}); skipping this dataset only."
                )
                raise
            # Hub unreachable: latch the process offline and retry this
            # same dataset from the local cache (non-streaming is the only
            # mode that can read it). Uncached -> raises -> skipped upstream.
            enter_offline_mode(f"{type(e).__name__} loading {self.dataset_path}")
            self.is_streaming = False
            dataset_args["streaming"] = False
            self.dataset = self._load_frugal(dataset_args)

        # Initial shuffle with base seed
        self.buffer_size = config.get("buffer_size", 32)
        shuffle_args = {"seed": self.base_seed}
        if self.is_streaming:
            shuffle_args["buffer_size"] = self.buffer_size
        self.shuffled_dataset = self.dataset.shuffle(**shuffle_args)
        self.dataset_iterator = iter(self.shuffled_dataset)

        # Initialize the count for this dataset path if not exists
        if self.dataset_path not in HuggingfaceDataset.counts:
            HuggingfaceDataset.counts[self.dataset_path] = 0

        # Storage for rewards when using RL format
        self.reward_cache = {}

        # A stream that can no longer fetch (offline mode latched mid-run)
        # retires: logs once, then yields empty documents quietly.
        self._retired = False

    def _load_frugal(self, dataset_args: Dict):
        """Load the stream with small fetches: prune to the columns we actually
        read, decode small record batches (default is a whole row group), and
        shrink the per-request network buffer (default 32MiB). These are parquet
        builder options; datasets that reject them fall back to a plain load.
        """
        if not self.is_streaming:
            return load_dataset_smart(dataset_args)
        frugal = dict(dataset_args)
        if self.keys:
            frugal["columns"] = list(self.keys)
        frugal["batch_size"] = 64  # rows per decoded RecordBatch
        try:
            import pyarrow as pa
            import pyarrow.dataset as pads

            frugal["fragment_scan_options"] = pads.ParquetFragmentScanOptions(
                cache_options=pa.CacheOptions(range_size_limit=4 << 20)
            )
        except Exception:
            pass
        try:
            return load_dataset_smart(frugal)
        except Exception:
            return load_dataset_smart(dataset_args)

    def get_document(self) -> Dict:
        """Get a formatted document with messages and metadata."""
        if self._retired:
            return {"messages": [], "metadata": {}}
        try:
            try:
                document = retry_on_network_error(
                    lambda: next(self.dataset_iterator),
                    label=f"stream next from {self.dataset_path}",
                )
            except StopIteration:
                # Restart the dataset
                self.restart_count += 1
                new_seed = self.base_seed + self.restart_count
                shuffle_args = {"seed": new_seed}
                if self.is_streaming:
                    shuffle_args["buffer_size"] = self.buffer_size
                self.shuffled_dataset = self.dataset.shuffle(**shuffle_args)
                self.dataset_iterator = iter(self.shuffled_dataset)
                document = retry_on_network_error(
                    lambda: next(self.dataset_iterator),
                    label=f"stream next from {self.dataset_path} (post-reshuffle)",
                )

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
            if hf_offline() and is_network_error(e):
                # Offline mode latched while this stream was live. Try to
                # carry on from the local cache (non-streaming is the only
                # mode that can read it); loop over stale data rather than
                # lose the source. No cache -> retire quietly: one line, no
                # traceback, no per-fetch repetition.
                if getattr(self, "_cache_fallback_tried", False):
                    self._retired = True
                    print(
                        f"[DATA] OFFLINE: retiring {self.dataset_path} "
                        "(cache fallback also failed)."
                    )
                    return {"messages": [], "metadata": {}}
                self._cache_fallback_tried = True
                try:
                    self.is_streaming = False
                    cached_args = dict(self._dataset_args, streaming=False)
                    self.dataset = self._load_frugal(cached_args)
                    self.shuffled_dataset = self.dataset.shuffle(seed=self.base_seed)
                    self.dataset_iterator = iter(self.shuffled_dataset)
                    print(
                        f"[DATA] OFFLINE: {self.dataset_path} now looping "
                        "over the local cache."
                    )
                    return self.get_document()
                except Exception:
                    self._retired = True
                    print(
                        f"[DATA] OFFLINE: retiring stream {self.dataset_path} "
                        "for this session (no local cache)."
                    )
                return {"messages": [], "metadata": {}}
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
                shuffle_args["buffer_size"] = self.buffer_size
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
                    # Empty document, try recursively instead of adding placeholder
                    self.fill_sequence_cache()
            except StopIteration:
                # Dataset is empty or has issues - create properly formatted minimal message
                print(
                    f"[ERROR] Dataset '{self.dataset_path}' appears to be empty or has issues."
                )
                # Create minimal valid message instead of empty string
                from praxis.data.config import SYSTEM_PROMPT

                error_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "assistant",
                        "content": f"Dataset {self.dataset_path} unavailable.",
                    },
                ]
                formatted_error = self.tokenizer.apply_chat_template(
                    error_messages, tokenize=False, add_generation_prompt=False
                )
                self.sequence_cache.append(formatted_error)

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
