"""Data interleaving and management with message queue for efficient deduplication."""

import random
from typing import Dict, List, Any, Optional
import torch
from transformers import PreTrainedTokenizer

from praxis.data.datasets.message_queue import MessageQueueManager
from praxis.data.formatters import _rl_logger


class InterleaveDataManagerV2:
    """
    Manages interleaving of multiple datasets using message queue for efficient tokenization.
    
    This version uses a MessageQueueManager to handle tokenization at the batch level,
    ensuring proper system prompt deduplication.
    """
    
    # Dynamic weighting control (hardcoded switch)
    use_dynamic_weights = True  # Set to False to use static weights
    ema_alpha = 0.1  # EMA smoothing factor
    
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
    ):
        """
        Initialize the data manager with message queue.
        
        Args:
            samplers: List of dataset samplers
            weights: Initial weights for each sampler
            tokenizer: Tokenizer to use
            block_size: Sequence length for training
            rl_type: Type of RL training if applicable
        """
        self.samplers = samplers
        self.static_weights = weights.copy()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rl_type = rl_type
        
        # Initialize message queue manager
        self.message_queue = MessageQueueManager(tokenizer, block_size)
        
        # Dynamic weighting setup
        if self.use_dynamic_weights:
            self.sampling_count = 0
            self.sampler_metrics = {}
            for i, sampler in enumerate(self.samplers):
                dataset_name = getattr(sampler, "dataset_path", f"sampler_{i}")
                self.sampler_metrics[i] = {
                    "name": dataset_name,
                    "avg_doc_length": None,
                    "total_samples": 0,
                    "total_tokens": 0,
                }
            
            # Initialize dynamic weights
            self.dynamic_weights = self.static_weights.copy()
            
            # Share weights between workers
            num_samplers = len(self.samplers)
            if (
                InterleaveDataManagerV2.shared_weights_initialized
                and InterleaveDataManagerV2.shared_weights is not None
                and len(InterleaveDataManagerV2.shared_weights) == num_samplers
            ):
                self.dynamic_weights = InterleaveDataManagerV2.shared_weights.copy()
            elif not InterleaveDataManagerV2.shared_weights_initialized:
                InterleaveDataManagerV2.shared_weights = self.dynamic_weights.copy()
                InterleaveDataManagerV2.shared_weights_initialized = True
            
            self.weights = self.dynamic_weights
        else:
            self.weights = weights
            
    
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
        if hypersample and batch_size >= 64:
            current_batch_size = batch_size // 64
            # Note: The sequence length adjustment is handled by using multiple sequences
        elif supersample and batch_size >= 16:
            current_batch_size = batch_size // 16
        elif oversample and batch_size >= 4:
            current_batch_size = batch_size // 4
        
        # Update weights if using dynamic weighting
        if self.use_dynamic_weights:
            if InterleaveDataManagerV2.shared_weights is not None and len(
                InterleaveDataManagerV2.shared_weights
            ) == len(self.samplers):
                self.weights = InterleaveDataManagerV2.shared_weights
            else:
                self.weights = self.dynamic_weights
        
        # Ensure message queue has enough documents
        self._refill_message_queue()
        
        # Get batch from message queue
        if self.rl_type:
            batch = self.message_queue.get_batch_with_rewards(current_batch_size)
        else:
            batch = self.message_queue.get_batch(current_batch_size)
        
        # Add sampler weights to result if using dynamic weighting
        if self.use_dynamic_weights:
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
                    dataset_name = getattr(sampler, "dataset_path", f"sampler_{sampler_idx}")
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
                    
                    # Update dynamic weights if enabled
                    if self.use_dynamic_weights:
                        # Estimate document length (number of messages * average tokens per message)
                        doc_length = len(document_data.get("messages", [])) * 50  # Rough estimate
                        self._update_dynamic_weights_after_sample(sampler_idx, doc_length)
    
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
            len(self.samplers) == len(InterleaveDataManagerV2.shared_weights)
            if InterleaveDataManagerV2.shared_weights
            else True
        ):
            InterleaveDataManagerV2.shared_weights = self.dynamic_weights.copy()
    
    def _calculate_target_weights(self):
        """Calculate target weights based on current metrics."""
        if not self.sampler_metrics:
            return self.static_weights
        
        # Skip if we don't have enough data yet
        if all(m["avg_doc_length"] is None for m in self.sampler_metrics.values()):
            return self.static_weights
        
        target_weights = []
        
        # Calculate average document length across all samplers
        valid_lengths = [
            m["avg_doc_length"]
            for m in self.sampler_metrics.values()
            if m["avg_doc_length"] is not None
        ]
        if not valid_lengths:
            return self.static_weights
            
        avg_length = sum(valid_lengths) / len(valid_lengths)
        
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
                
                # Factor 2: Balance token consumption
                if metrics["total_tokens"] > 0:
                    token_balance_factor = avg_tokens_per_sampler / max(
                        metrics["total_tokens"], 1.0
                    )
                else:
                    token_balance_factor = 2.0  # Boost for never sampled
                
                # Combine factors
                weight = weight * (length_factor * token_balance_factor) ** 0.5
            
            target_weights.append(weight)
        
        # Normalize weights
        total = sum(target_weights)
        if total > 0:
            return [w / total for w in target_weights]
        else:
            return self.static_weights