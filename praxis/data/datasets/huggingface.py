"""HuggingFace dataset implementation."""

import hashlib
from typing import Dict

from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler, load_dataset_smart
from praxis.data.datasets.network_retry import (
    force_offline,
    hf_offline,
    is_network_error,
    is_skippable_load_error,
    is_unrecoverable,
    reset_hub_session,
    retry_on_network_error,
)
from praxis.data.formats import DataFormat
from praxis.data.formatters import (
    _rl_logger,
    format_conversation,
    format_human_assistant,
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
    DataFormat.HUMAN_ASSISTANT: format_human_assistant,
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
    # The httpx client is shared process-wide; once one sampler proves it
    # dead and unrebuildable, the rest skip straight to their caches.
    transport_broken = False

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
        # True once we've dropped to non-streaming as a *fallback* (offline or
        # dead transport); gates _load_frugal to cache-only so it never pulls a
        # full download. Configured streaming=False sets is_streaming without
        # this, so those deliberate one-time downloads still work.
        self._cache_only = False
        self.is_streaming = config.get("streaming", True)
        if hf_offline() and self.is_streaming:
            # Streams read over HTTP and cache nothing reusable; a full
            # (non-streaming) load CAN resolve from the local datasets cache.
            # Uncached datasets fail fast here and get skipped upstream.
            self.is_streaming = False
            self._cache_only = True

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
            try:
                self.dataset = self._load_frugal(dataset_args)
            except Exception as e:
                if not is_unrecoverable(e):
                    raise
                # Dead shared client at boot (e.g. closed by a resume fork):
                # actually reset the hub client/filesystem so the retry is fresh.
                # The bare retry used to reuse the same dead client and refail.
                print(
                    f"[DATA] {self.dataset_path}: resetting stale HF client "
                    f"({type(e).__name__}: {str(e)[:120]}) and retrying."
                )
                reset_hub_session()
                self.dataset = self._load_frugal(dataset_args)
        except Exception as e:
            if hf_offline() or not is_skippable_load_error(e):
                raise
            # A load failure on one dataset must not disable the rest. We do NOT
            # latch the whole process offline (a startup DNS blip on the first
            # dataset would otherwise skip every later one): instead try THIS
            # dataset from the local cache, and if it isn't there (uncached or a
            # corrupt partial download), skip just it. Every other dataset
            # retries independently, so a connection that recovers a moment
            # later still loads them. The cache read is forced offline (see
            # _load_frugal) so it never downloads. Instance-local, not global.
            self.is_streaming = False
            self._cache_only = True
            dataset_args["streaming"] = False
            try:
                self.dataset = self._load_frugal(dataset_args)
                print(
                    f"[DATA] {self.dataset_path} streaming failed "
                    f"({type(e).__name__}: {str(e)[:120]}); using local cache."
                )
            except Exception:
                print(
                    f"[DATA] skipping {self.dataset_path} (not cached); other "
                    f"datasets keep streaming. streaming error was "
                    f"{type(e).__name__}: {str(e)[:160]}"
                )
                raise

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
            # Two ways to be non-streaming: the config asked for it (a small
            # set we deliberately download once) or we fell back here because
            # streaming died / we're offline. The fallback must never download
            # the whole set (it wedges on a flaky network) - it exists only to
            # read the cache. local_files_only is necessary but NOT sufficient
            # (a builder's download_and_prepare can ignore it), so we also run
            # the load under force_offline(): the runtime flags it does honor.
            # Uncached -> raises -> skipped upstream. Configured non-streaming
            # downloads normally.
            if self._cache_only or hf_offline():
                from datasets import DownloadConfig

                dataset_args = dict(dataset_args)
                dataset_args.setdefault(
                    "download_config", DownloadConfig(local_files_only=True)
                )
                with force_offline():
                    return load_dataset_smart(dataset_args)
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

            self._stream_rebuilds = 0  # healthy fetch resets the budget

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
            # StopIteration after a reshuffle means the stream is broken (a
            # dead transport yields zero documents), not exhausted.
            if not hf_offline() and (
                is_network_error(e) or isinstance(e, StopIteration)
            ):
                # A dead transport (e.g. closed shared httpx client after a
                # checkpoint-resume fork) cannot heal by retrying; rebuild the
                # stream so it gets a fresh client. Bounded so a true poison
                # state degrades to retirement, not a rebuild loop.
                self._stream_rebuilds = getattr(self, "_stream_rebuilds", 0) + 1
                if (
                    not HuggingfaceDataset.transport_broken
                    and self._stream_rebuilds <= 1
                ):
                    try:
                        # Actually get a fresh client (the dead transport is
                        # usually a closed shared client); reusing it refails.
                        reset_hub_session()
                        self.dataset = self._load_frugal(dict(self._dataset_args))
                        shuffle_args = {"seed": self.base_seed + self.restart_count}
                        if self.is_streaming:
                            shuffle_args["buffer_size"] = self.buffer_size
                        self.shuffled_dataset = self.dataset.shuffle(**shuffle_args)
                        self.dataset_iterator = iter(self.shuffled_dataset)
                        print(
                            f"[DATA] rebuilt stream {self.dataset_path} "
                            "after a dead transport."
                        )
                        return self.get_document()
                    except Exception:
                        # One failed rebuild condemns the shared client for
                        # everyone - no per-sampler relearning at boot.
                        HuggingfaceDataset.transport_broken = True
                return self._fall_back_to_cache("dead transport")
            if hf_offline() and is_network_error(e):
                # Offline mode latched while this stream was live.
                return self._fall_back_to_cache("offline")
            print(f"[ERROR] HuggingfaceDataset.get_document failed: {e}")
            import traceback

            traceback.print_exc()
            return {"messages": [], "metadata": {}}

    def _fall_back_to_cache(self, reason: str) -> Dict:
        """Last resort for a stream that cannot fetch: loop over the local
        cache (non-streaming is the only mode that can read it). No cache ->
        retire quietly: one line, no traceback, no per-fetch repetition."""
        if getattr(self, "_cache_fallback_tried", False):
            self._retired = True
            print(
                f"[DATA] retiring {self.dataset_path} "
                f"({reason}; cache fallback also failed)."
            )
            return {"messages": [], "metadata": {}}
        self._cache_fallback_tried = True
        try:
            # Non-streaming is the only mode that reads the cache; _cache_only
            # makes _load_frugal forbid downloads, so this never pulls the set.
            self.is_streaming = False
            self._cache_only = True
            cached_args = dict(self._dataset_args, streaming=False)
            self.dataset = self._load_frugal(cached_args)
            self.shuffled_dataset = self.dataset.shuffle(seed=self.base_seed)
            self.dataset_iterator = iter(self.shuffled_dataset)
            print(
                f"[DATA] {self.dataset_path} now looping over the local "
                f"cache ({reason})."
            )
            return self.get_document()
        except Exception:
            self._retired = True
            print(
                f"[DATA] retiring stream {self.dataset_path} for this "
                f"session ({reason}; no local cache)."
            )
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
