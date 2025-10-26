"""Message queue manager for efficient batching."""

from collections import deque
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


class MessageQueueManager:
    """
    Manages a queue of messages and efficiently batches them.

    Simply queues and tokenizes messages without modifying their structure.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, block_size: int):
        """
        Initialize the message queue manager.

        Args:
            tokenizer: The tokenizer to use for converting messages to tokens
            block_size: The sequence length for each training example
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Message queue stores structured message data
        self.message_queue = deque()

        # Token buffer stores already tokenized content
        self.token_buffer = torch.tensor([], dtype=torch.long)

        # Metadata buffer parallel to token buffer
        self.metadata_buffer = []

    def add_document(self, document_data: Dict[str, Any]):
        """
        Add a document (with messages and metadata) to the queue.

        Args:
            document_data: Dict with 'messages' and 'metadata' keys
        """
        messages = document_data.get("messages", [])
        metadata = document_data.get("metadata", {})

        if not messages:
            return

        # Simply add the messages as-is, no filtering or modification
        self.message_queue.append({"messages": messages, "metadata": metadata})

    def _refill_token_buffer(self):
        """Refill the token buffer from the message queue."""
        # Process documents individually to preserve BOS -> role constraints
        documents_to_process = []

        # Collect up to 50 documents or until queue is empty
        while len(documents_to_process) < 50 and self.message_queue:
            doc = self.message_queue.popleft()
            documents_to_process.append(doc)

        if not documents_to_process:
            return

        # Tokenize each document separately, then concatenate
        all_tokens = []
        all_metadata = []

        for doc in documents_to_process:
            messages = doc["messages"]
            metadata = doc["metadata"]

            if not messages:
                continue

            # Apply chat template to this document only
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception as e:
                print(
                    f"[ERROR] MessageQueue._refill_token_buffer failed to apply chat template: {e}"
                )
                print(f"[ERROR] Messages: {messages}")
                import traceback

                traceback.print_exc()
                continue

            # Tokenize this document
            doc_tokens = self.tokenizer(
                text, return_tensors="pt", padding=False, truncation=False
            )["input_ids"].squeeze(0)

            all_tokens.append(doc_tokens)

            # Add metadata for each token in this document
            for _ in range(len(doc_tokens)):
                all_metadata.append(metadata)

        if not all_tokens:
            return

        # Concatenate all document tokens
        new_tokens = torch.cat(all_tokens)

        # Append to buffers
        self.token_buffer = torch.cat([self.token_buffer, new_tokens])
        self.metadata_buffer.extend(all_metadata)

    def get_batch(
        self, batch_size: int, sequence_multiplier: int = 1
    ) -> Dict[str, Any]:
        """
        Get a batch of sequences.

        Args:
            batch_size: Number of sequences in the batch
            sequence_multiplier: Factor to multiply the sequence length by (for oversampling)

        Returns:
            Dictionary with 'batch' tensor and metadata
        """
        effective_block_size = self.block_size * sequence_multiplier
        tokens_needed = batch_size * effective_block_size

        # Ensure we have enough tokens
        refill_attempts = 0
        max_refill_attempts = 100
        while len(self.token_buffer) < tokens_needed:
            prev_buffer_size = len(self.token_buffer)
            self._refill_token_buffer()

            # Check if buffer didn't grow (no new tokens added)
            if len(self.token_buffer) == prev_buffer_size:
                refill_attempts += 1
                if refill_attempts > max_refill_attempts:
                    print(
                        f"[ERROR] MessageQueue.get_batch: Unable to refill token buffer after {max_refill_attempts} attempts"
                    )
                    print(
                        f"[ERROR] Buffer size: {len(self.token_buffer)}, needed: {tokens_needed}"
                    )
                    print(f"[ERROR] Message queue size: {len(self.message_queue)}")
                    raise RuntimeError(
                        "Failed to refill token buffer - possible data loading issue"
                    )

            # Check if we've run out of data
            if len(self.message_queue) == 0 and len(self.token_buffer) < tokens_needed:
                # Pad with zeros if we don't have enough data
                padding_needed = tokens_needed - len(self.token_buffer)
                self.token_buffer = torch.cat(
                    [self.token_buffer, torch.zeros(padding_needed, dtype=torch.long)]
                )
                # Add empty metadata for padding
                self.metadata_buffer.extend([{}] * padding_needed)
                break

        sequences = []
        batch_metadata = []

        for i in range(batch_size):
            # Extract tokens for this sequence
            start_idx = i * effective_block_size
            end_idx = start_idx + effective_block_size
            sequence = self.token_buffer[start_idx:end_idx]

            # Ensure exactly effective_block_size tokens
            if len(sequence) < effective_block_size:
                padding = torch.zeros(
                    effective_block_size - len(sequence), dtype=torch.long
                )
                sequence = torch.cat([sequence, padding])

            sequences.append(sequence)

            # Collect metadata for this sequence (from the dominant source)
            if start_idx < len(self.metadata_buffer):
                batch_metadata.append(self.metadata_buffer[start_idx])
            else:
                batch_metadata.append({})

        # Remove consumed tokens and metadata
        self.token_buffer = self.token_buffer[tokens_needed:]
        self.metadata_buffer = self.metadata_buffer[tokens_needed:]

        return {
            "batch": sequences,  # Return as list for WeightedIterableDataset to stack
            "metadata": batch_metadata,
        }

    def get_batch_with_rewards(
        self, batch_size: int, sequence_multiplier: int = 1
    ) -> Dict[str, Any]:
        """
        Get a batch with reward information preserved.

        Args:
            batch_size: Number of sequences in the batch
            sequence_multiplier: Factor to multiply the sequence length by (for oversampling)

        Returns:
            Dictionary with batch, rewards, and metadata
        """
        result = self.get_batch(batch_size, sequence_multiplier)

        # Extract rewards from metadata if available
        rewards = []
        for meta in result["metadata"]:
            reward = meta.get("reward", 0.0)
            rewards.append(reward)

        result["rewards"] = (
            torch.tensor(rewards, dtype=torch.float32) if rewards else None
        )

        return result
