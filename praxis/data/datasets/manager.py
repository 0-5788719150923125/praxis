"""Data interleaving and management with message queue for efficient deduplication."""

import random
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer

from praxis.data.datasets.message_queue import MessageQueueManager
from praxis.data.datasets.novelty import NoveltyTracker
from praxis.data.formatters import _rl_logger
from praxis.logging.data_metrics_logger import DataMetricsLogger

# Valid weighting modes
WEIGHTING_MODES = ("static", "dynamic", "novelty")


class InterleaveDataManager:
    """
    Manages interleaving of multiple datasets using message queue for efficient tokenization.

    This version uses a MessageQueueManager to handle tokenization at the batch level,
    ensuring proper system prompt deduplication.

    Weighting modes:
        "static"  — use dataset weights as given, no adaptation
        "dynamic" — adjust weights based on document length and token balance
        "novelty" — adjust weights based on bigram novelty (Count-Min Sketch)
    """

    # Weighting mode: "static", "dynamic", or "novelty"
    weighting_mode = "novelty"
    ema_alpha = 0.3  # EMA smoothing factor (used by dynamic and novelty modes)

    # Class variable to store shared weights across all instances
    shared_weights = None
    shared_weights_initialized = False

    def __init__(
        self,
        samplers,
        weights,
        tokenizer,
        block_size,
        rl_type=None,
        run_dir=None,
        data_metrics_log_interval=50,
        enable_chat_validation=True,
        strict_chat_validation=False,
    ):
        """
        Initialize the data manager with message queue.

        Args:
            samplers: List of dataset samplers
            weights: Initial weights for each sampler
            tokenizer: Tokenizer to use
            block_size: Sequence length for training
            rl_type: Type of RL training if applicable
            run_dir: Directory for logging data metrics (optional)
            data_metrics_log_interval: Log data metrics every N samples (default: 50)
            enable_chat_validation: Enable BOS token validation (default: True)
            strict_chat_validation: Raise exception on validation failure (default: False)
        """
        self.samplers = samplers
        self.static_weights = weights.copy()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rl_type = rl_type

        # Initialize message queue manager with validation settings
        self.message_queue = MessageQueueManager(
            tokenizer,
            block_size,
            enable_chat_validation=enable_chat_validation,
            strict_chat_validation=strict_chat_validation,
        )

        # Initialize data metrics logging
        self.data_metrics_logger = None
        self.data_metrics_log_interval = data_metrics_log_interval
        self.samples_since_last_log = 0

        adaptive = self.weighting_mode in ("dynamic", "novelty")

        if run_dir is not None and adaptive:
            try:
                self.data_metrics_logger = DataMetricsLogger(run_dir=run_dir)
                print(f"[DATA METRICS] Initialized logger for run: {run_dir}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize data metrics logger: {e}")
        else:
            print(
                f"[DATA METRICS] Logger not initialized: run_dir={run_dir}, weighting_mode={self.weighting_mode}"
            )

        # Adaptive weighting setup (shared by dynamic and novelty modes)
        if adaptive:
            self.sampling_count = 0
            self.sampler_metrics = {}
            for i, sampler in enumerate(self.samplers):
                dataset_name = getattr(sampler, "dataset_path", f"sampler_{i}")
                metrics = {"name": dataset_name, "total_samples": 0}
                if self.weighting_mode == "dynamic":
                    metrics["avg_doc_length"] = None
                    metrics["total_tokens"] = 0
                self.sampler_metrics[i] = metrics

            # Novelty tracker (only for novelty mode)
            if self.weighting_mode == "novelty":
                # Build set of token IDs that decode to pure digits so the
                # tracker can collapse them before bigram extraction.
                numeric_ids = set()
                for token_id in range(len(self.tokenizer)):
                    try:
                        token_str = self.tokenizer.convert_ids_to_tokens(
                            token_id
                        )
                        if token_str is not None:
                            cleaned = token_str.replace("\u0120", "").replace(
                                "\u2581", ""
                            )
                            if cleaned and cleaned.isdigit():
                                numeric_ids.add(token_id)
                    except Exception:
                        pass
                self.novelty_tracker = NoveltyTracker(
                    num_datasets=len(self.samplers),
                    numeric_token_ids=numeric_ids,
                )

            # Initialize dynamic weights
            self.dynamic_weights = self.static_weights.copy()

            # Share weights between workers
            num_samplers = len(self.samplers)
            if (
                InterleaveDataManager.shared_weights_initialized
                and InterleaveDataManager.shared_weights is not None
                and len(InterleaveDataManager.shared_weights) == num_samplers
            ):
                self.dynamic_weights = InterleaveDataManager.shared_weights.copy()
            elif not InterleaveDataManager.shared_weights_initialized:
                InterleaveDataManager.shared_weights = self.dynamic_weights.copy()
                InterleaveDataManager.shared_weights_initialized = True

            self.weights = self.dynamic_weights
        else:
            self.weights = weights

    @property
    def _adaptive(self):
        return self.weighting_mode in ("dynamic", "novelty")

    def get_batch(
        self,
        batch_size: int,
        oversample: bool = False,
        supersample: bool = False,
        hypersample: bool = False,
    ) -> Dict[str, Any]:
        """
        Get a batch of sequences using the message queue.

        Args:
            batch_size: Number of sequences in the batch
            oversample: Whether to use 2x sequence length
            supersample: Whether to use 4x sequence length
            hypersample: Whether to use 8x sequence length

        Returns:
            Dictionary with batch data and metadata
        """
        # Adjust batch size and sequence length for oversampling
        current_batch_size = batch_size
        sequence_multiplier = 1

        if hypersample and batch_size >= 64:
            current_batch_size = batch_size // 64
            sequence_multiplier = 8  # 8x sequence length
        elif supersample and batch_size >= 16:
            current_batch_size = batch_size // 16
            sequence_multiplier = 4  # 4x sequence length
        elif oversample and batch_size >= 4:
            current_batch_size = batch_size // 4
            sequence_multiplier = 2  # 2x sequence length

        # Update weights if using adaptive weighting
        if self._adaptive:
            if InterleaveDataManager.shared_weights is not None and len(
                InterleaveDataManager.shared_weights
            ) == len(self.samplers):
                self.weights = InterleaveDataManager.shared_weights
            else:
                self.weights = self.dynamic_weights

        # Ensure message queue has enough documents
        self._refill_message_queue()

        # Get batch from message queue with adjusted sequence length
        if self.rl_type:
            batch = self.message_queue.get_batch_with_rewards(
                current_batch_size, sequence_multiplier=sequence_multiplier
            )
        else:
            batch = self.message_queue.get_batch(
                current_batch_size, sequence_multiplier=sequence_multiplier
            )

        # Add sampler weights to result if using adaptive weighting
        if self._adaptive:
            batch["sampler_weights"] = self.weights.copy()

        return batch

    def _refill_message_queue(self, min_documents: int = 100):
        """
        Refill the message queue with documents from samplers.

        Args:
            min_documents: Minimum number of documents to maintain in queue
        """
        # Check how many documents we need
        queue_size = len(self.message_queue.message_queue)
        if queue_size >= min_documents:
            return

        documents_to_add = min_documents - queue_size

        for i in range(documents_to_add):
            # Pick a sampler based on weights
            sampler_idx = random.choices(
                range(len(self.samplers)), weights=self.weights, k=1
            )[0]
            sampler = self.samplers[sampler_idx]

            # Get a formatted document (now returns dict with messages)
            document_data = sampler.get_document()

            # Handle both old (text) and new (dict) formats for compatibility
            if isinstance(document_data, str):
                # Legacy format - skip for now
                continue
            elif isinstance(document_data, dict):
                # New format with messages and metadata
                if "messages" in document_data:
                    # Check if messages is empty
                    if not document_data["messages"]:
                        continue

                    # Add dataset info to metadata
                    dataset_name = getattr(
                        sampler, "dataset_path", f"sampler_{sampler_idx}"
                    )
                    document_data["metadata"]["dataset"] = dataset_name

                    # Handle RL rewards if present
                    if self.rl_type and "reward" in document_data.get("metadata", {}):
                        reward = document_data["metadata"]["reward"]
                        if reward == -1:
                            pass  # Generation sequence
                        elif reward > 0:
                            _rl_logger.log_reward_found(reward, dataset_name)

                    # Add to message queue
                    self.message_queue.add_document(document_data)

                    # Update adaptive weights if enabled
                    if self.weighting_mode == "dynamic":
                        doc_length = (
                            len(document_data.get("messages", [])) * 50
                        )  # Rough estimate
                        self._update_weights_after_sample(
                            sampler_idx, doc_length=doc_length
                        )
                    elif self.weighting_mode == "novelty":
                        try:
                            messages = document_data["messages"]
                            text = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                            token_ids = self.tokenizer.encode(
                                text, add_special_tokens=False
                            )
                            self.novelty_tracker.score_and_update(
                                sampler_idx, token_ids
                            )
                        except Exception:
                            pass  # Never break the data pipeline
                        self._update_weights_after_sample(sampler_idx)

    def _update_weights_after_sample(self, sampler_idx: int, doc_length: int = 0):
        """Update metrics and weights with EMA after each sample."""
        if not self._adaptive:
            return

        metrics = self.sampler_metrics[sampler_idx]
        metrics["total_samples"] += 1
        self.sampling_count += 1

        # Dynamic mode: track document length metrics
        if self.weighting_mode == "dynamic":
            metrics["total_tokens"] += doc_length
            if metrics["avg_doc_length"] is None:
                metrics["avg_doc_length"] = float(doc_length)
            else:
                metrics["avg_doc_length"] = (
                    self.ema_alpha * doc_length
                    + (1 - self.ema_alpha) * metrics["avg_doc_length"]
                )

        # Calculate target weights based on current mode
        target_weights = self._calculate_target_weights()

        # Update dynamic weights with EMA towards target
        for i in range(len(self.dynamic_weights)):
            self.dynamic_weights[i] = (
                self.ema_alpha * target_weights[i]
                + (1 - self.ema_alpha) * self.dynamic_weights[i]
            )

        # Normalize to ensure weights sum to 1
        total = sum(self.dynamic_weights)
        if total > 0:
            self.dynamic_weights = [w / total for w in self.dynamic_weights]

        # Update shared weights
        if (
            len(self.samplers) == len(InterleaveDataManager.shared_weights)
            if InterleaveDataManager.shared_weights
            else True
        ):
            InterleaveDataManager.shared_weights = self.dynamic_weights.copy()

        # Log data metrics periodically
        if self.data_metrics_logger is not None:
            self.samples_since_last_log += 1
            if self.samples_since_last_log >= self.data_metrics_log_interval:
                self._log_data_metrics()
                self.samples_since_last_log = 0

    def _calculate_target_weights(self):
        """Calculate target weights based on current weighting mode."""
        if self.weighting_mode == "novelty":
            return self.novelty_tracker.get_target_weights(self.static_weights)

        # Dynamic mode: balance by document length and token consumption
        if not self.sampler_metrics:
            return self.static_weights

        if all(m["avg_doc_length"] is None for m in self.sampler_metrics.values()):
            return self.static_weights

        valid_lengths = [
            m["avg_doc_length"]
            for m in self.sampler_metrics.values()
            if m["avg_doc_length"] is not None
        ]
        if not valid_lengths:
            return self.static_weights

        avg_length = sum(valid_lengths) / len(valid_lengths)
        total_tokens = sum(m["total_tokens"] for m in self.sampler_metrics.values())
        avg_tokens_per_sampler = (
            total_tokens / len(self.sampler_metrics) if total_tokens > 0 else 1
        )

        target_weights = []
        for i in range(len(self.samplers)):
            metrics = self.sampler_metrics[i]
            weight = self.static_weights[i]

            if metrics["avg_doc_length"] is not None and metrics["total_samples"] > 0:
                length_factor = avg_length / max(metrics["avg_doc_length"], 1.0)
                if metrics["total_tokens"] > 0:
                    token_balance_factor = avg_tokens_per_sampler / max(
                        metrics["total_tokens"], 1.0
                    )
                else:
                    token_balance_factor = 2.0
                weight = weight * (length_factor * token_balance_factor) ** 0.5

            target_weights.append(weight)

        total = sum(target_weights)
        if total > 0:
            return [w / total for w in target_weights]
        return self.static_weights

    def _log_data_metrics(self):
        """Log current sampling weights and metrics to data metrics file."""
        if self.data_metrics_logger is None:
            return

        # Build sampling weights dictionary with dataset names
        sampling_weights = {}
        dataset_stats = {}
        for i, weight in enumerate(self.dynamic_weights):
            dataset_name = self.sampler_metrics[i]["name"]
            sampling_weights[dataset_name] = round(weight, 6)
            stats = {"total_samples": self.sampler_metrics[i]["total_samples"]}
            if self.weighting_mode == "novelty":
                stats["novelty"] = round(
                    float(self.novelty_tracker.dataset_novelty[i]), 6
                )
            dataset_stats[dataset_name] = stats

        # Log to file
        try:
            self.data_metrics_logger.log(
                step=self.sampling_count,
                sampling_weights=sampling_weights,
                dataset_stats=dataset_stats,
            )
        except Exception as e:
            print(f"[WARNING] Failed to log data metrics: {e}")
