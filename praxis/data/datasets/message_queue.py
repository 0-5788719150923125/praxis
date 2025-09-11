"""Message queue manager for efficient batching with forced system prompts."""

import torch
from typing import Dict, List, Any, Optional
from collections import deque
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT


class MessageQueueManager:
    """
    Manages a queue of messages and efficiently batches them with forced system prompts.
    
    Every training sequence of block_size tokens MUST start with the system prompt,
    with content flowing continuously after the system prompt tokens.
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
        
        # Token buffer stores already tokenized content (without system prompt)
        self.token_buffer = torch.tensor([], dtype=torch.long)
        
        # Metadata buffer parallel to token buffer
        self.metadata_buffer = []
        
        # Pre-tokenize and cache the system prompt
        self.system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        system_text = self.tokenizer.apply_chat_template(
            self.system_messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        self.system_prompt_tokens = self.tokenizer(
            system_text, 
            return_tensors="pt",
            padding=False,
            truncation=False
        )["input_ids"].squeeze(0)
        self.system_prompt_length = len(self.system_prompt_tokens)
        
        
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
            
        # Filter out system messages - we'll use our own consistent system prompt
        filtered_messages = []
        for msg in messages:
            if msg["role"] != "system":
                filtered_messages.append(msg)
                
        if filtered_messages:
            self.message_queue.append({
                "messages": filtered_messages,
                "metadata": metadata
            })
    
    def _refill_token_buffer(self):
        """Refill the token buffer from the message queue."""
        # Process messages in chunks for efficiency
        messages_to_process = []
        metadata_to_process = []
        
        initial_queue_size = len(self.message_queue)
        initial_buffer_size = len(self.token_buffer)
        
        # Collect up to 50 documents or until queue is empty
        while len(messages_to_process) < 50 and self.message_queue:
            doc = self.message_queue.popleft()
            messages_to_process.extend(doc["messages"])
            # Extend metadata for each message
            for _ in doc["messages"]:
                metadata_to_process.append(doc["metadata"])
        
        if not messages_to_process:
            return
            
        # Tokenize all messages at once (without system prompt)
        try:
            text = self.tokenizer.apply_chat_template(
                messages_to_process,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"[ERROR] MessageQueue._refill_token_buffer failed to apply chat template: {e}")
            print(f"[ERROR] Messages: {messages_to_process[:2]}...")  # Show first 2 messages
            import traceback
            traceback.print_exc()
            return
        
        new_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )["input_ids"].squeeze(0)
        
        # Append to buffers
        self.token_buffer = torch.cat([self.token_buffer, new_tokens])
        self.metadata_buffer.extend(metadata_to_process)
        
        # Trim metadata buffer to match token buffer size
        while len(self.metadata_buffer) > len(self.token_buffer):
            self.metadata_buffer.pop()
            
        docs_processed = initial_queue_size - len(self.message_queue)
    
    def get_batch(self, batch_size: int) -> Dict[str, Any]:
        """
        Get a batch of sequences, each starting with the system prompt.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            Dictionary with 'input_ids' tensor and metadata
        """
        content_length = self.block_size - self.system_prompt_length
        tokens_needed = batch_size * content_length
        
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
                    print(f"[ERROR] MessageQueue.get_batch: Unable to refill token buffer after {max_refill_attempts} attempts")
                    print(f"[ERROR] Buffer size: {len(self.token_buffer)}, needed: {tokens_needed}")
                    print(f"[ERROR] Message queue size: {len(self.message_queue)}")
                    raise RuntimeError("Failed to refill token buffer - possible data loading issue")
            
            # Check if we've run out of data
            if len(self.message_queue) == 0 and len(self.token_buffer) < tokens_needed:
                # Pad with zeros if we don't have enough data
                padding_needed = tokens_needed - len(self.token_buffer)
                self.token_buffer = torch.cat([
                    self.token_buffer,
                    torch.zeros(padding_needed, dtype=torch.long)
                ])
                # Add empty metadata for padding
                self.metadata_buffer.extend([{}] * padding_needed)
                break
        
        sequences = []
        batch_metadata = []
        
        for i in range(batch_size):
            # Start with system prompt
            sequence = self.system_prompt_tokens.clone()
            
            # Extract content tokens
            start_idx = i * content_length
            end_idx = start_idx + content_length
            content_tokens = self.token_buffer[start_idx:end_idx]
            
            # Concatenate system prompt + content
            full_sequence = torch.cat([sequence, content_tokens])
            
            # Ensure exactly block_size tokens
            if len(full_sequence) > self.block_size:
                full_sequence = full_sequence[:self.block_size]
            elif len(full_sequence) < self.block_size:
                padding = torch.zeros(
                    self.block_size - len(full_sequence), 
                    dtype=torch.long
                )
                full_sequence = torch.cat([full_sequence, padding])
            
            sequences.append(full_sequence)
            
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
            "metadata": batch_metadata
        }
    
    def get_batch_with_rewards(self, batch_size: int) -> Dict[str, Any]:
        """
        Get a batch with reward information preserved.
        
        Args:
            batch_size: Number of sequences in the batch
            
        Returns:
            Dictionary with batch, rewards, and metadata
        """
        result = self.get_batch(batch_size)
        
        # Extract rewards from metadata if available
        rewards = []
        for meta in result["metadata"]:
            reward = meta.get("reward", 0.0)
            rewards.append(reward)
        
        result["rewards"] = torch.tensor(rewards, dtype=torch.float32) if rewards else None
        
        return result