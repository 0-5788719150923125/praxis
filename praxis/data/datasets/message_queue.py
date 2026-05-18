"""Message queue manager for efficient batching."""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from praxis.data.validators import ChatTemplateValidator
from praxis.tasks import DEFAULT_TASK


class MessageQueueManager:
    """
    Manages a queue of messages and packs them into training sequences.

    Packs documents doc-by-doc into sequences of length `block_size`. The
    leading BOS of each document is kept only when that doc begins a fresh
    sequence; when a doc is appended mid-sequence its leading BOS is omitted
    (via the chat template's `omit_leading_bos` flag). This keeps every
    sequence's position 0 a real BOS -> role transition, matching inference,
    without discarding tokens. A doc that overflows the sequence boundary
    has its tail carried over to the next sequence.

    Per-token side channels travel with the tokens through packing:
        task_type_ids: int8 task ID per token, copied from doc metadata.
        assistant_mask: 1 where the token is part of an assistant turn,
                        0 elsewhere -- emitted by the chat template's
                        {% generation %} blocks. Used downstream for
                        prompt-loss masking during SFT.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        enable_chat_validation: bool = True,
        strict_chat_validation: bool = False,
    ):
        """
        Initialize the message queue manager.

        Args:
            tokenizer: The tokenizer to use for converting messages to tokens
            block_size: The sequence length for each training example
            enable_chat_validation: Enable BOS token validation (default: True)
            strict_chat_validation: If True, raise exception on validation failure.
                                   If False, log warning and skip document (default: False)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.enable_chat_validation = enable_chat_validation
        self.strict_chat_validation = strict_chat_validation

        # Structured message queue (docs not yet tokenized).
        self.message_queue = deque()

        # Overflow from the previous sequence's packing: the tail of a doc that
        # didn't fit. Kept so no tokens are discarded at sequence boundaries.
        self._carry_tokens: Optional[torch.Tensor] = None
        self._carry_task_ids: Optional[torch.Tensor] = None
        self._carry_assistant_mask: Optional[torch.Tensor] = None
        self._carry_metadata: List[Dict] = []

        self.chat_validator = None
        if self.enable_chat_validation:
            self.chat_validator = ChatTemplateValidator(
                tokenizer=tokenizer, strict_mode=strict_chat_validation
            )

        self.validation_stats = {
            "documents_validated": 0,
            "documents_failed": 0,
            "documents_skipped": 0,
            "template_application_errors": 0,
        }

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

        self.message_queue.append({"messages": messages, "metadata": metadata})

    def _tokenize_doc(
        self, doc: Dict[str, Any], omit_leading_bos: bool
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Tokenize one document, optionally dropping the leading BOS.

        Returns (tokens, assistant_mask) or None if chat-template
        application fails or per-doc validation rejects the document
        (and strict mode is off). assistant_mask is the same shape as
        tokens, with 1 marking assistant-generated positions.
        """
        messages = doc["messages"]
        metadata = doc["metadata"]

        if not messages:
            return None

        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                omit_leading_bos=omit_leading_bos,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
        except Exception as e:
            self.validation_stats["template_application_errors"] += 1

            print("=" * 80)
            print("[CRITICAL ERROR] Failed to apply chat template!")
            print(f"Error: {e}")
            print(f"Document metadata: {metadata}")
            print(f"Messages structure:")
            for i, msg in enumerate(messages):
                role = msg.get("role", "MISSING_ROLE")
                content_preview = str(msg.get("content", "MISSING_CONTENT"))[:200]
                print(f"  [{i}] role={role}, content={content_preview}...")
            print("=" * 80)

            import traceback

            traceback.print_exc()
            return None

        ids = encoded["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        doc_tokens = torch.as_tensor(ids, dtype=torch.long)

        masks = encoded.get("assistant_masks")
        if masks is None:
            assistant_mask = torch.zeros_like(doc_tokens, dtype=torch.uint8)
        else:
            if isinstance(masks, list) and masks and isinstance(masks[0], list):
                masks = masks[0]
            assistant_mask = torch.as_tensor(masks, dtype=torch.uint8)
            if assistant_mask.shape != doc_tokens.shape:
                # Defensive: skip the doc rather than emit a misaligned mask.
                print(
                    f"[WARNING] assistant_mask shape {tuple(assistant_mask.shape)} "
                    f"!= input_ids shape {tuple(doc_tokens.shape)}; skipping doc"
                )
                return None

        if self.chat_validator is not None:
            self.validation_stats["documents_validated"] += 1
            text = self.tokenizer.decode(doc_tokens, skip_special_tokens=False)
            is_valid, report = self.chat_validator.validate_and_report(
                doc_tokens, messages=messages, formatted_text=text
            )
            if not is_valid:
                self.validation_stats["documents_failed"] += 1
                if self.strict_chat_validation:
                    raise ValueError(f"Chat template validation failed:\n{report}")
                print("[WARNING] Chat template validation failed, skipping document:")
                print(report)
                self.validation_stats["documents_skipped"] += 1
                return None

        return doc_tokens, assistant_mask

    def get_batch(
        self, batch_size: int, sequence_multiplier: int = 1
    ) -> Dict[str, Any]:
        """
        Get a batch of sequences.

        Args:
            batch_size: Number of sequences in the batch
            sequence_multiplier: Factor to multiply the sequence length by

        Returns:
            Dictionary with 'batch' (list of tensors) and 'metadata' (list of dicts)
        """
        effective_block_size = self.block_size * sequence_multiplier

        sequences: List[torch.Tensor] = []
        task_id_seqs: List[torch.Tensor] = []
        assistant_mask_seqs: List[torch.Tensor] = []
        batch_metadata: List[Dict] = []

        for _ in range(batch_size):
            seq_parts: List[torch.Tensor] = []
            seq_task_parts: List[torch.Tensor] = []
            seq_mask_parts: List[torch.Tensor] = []
            seq_meta: List[Dict] = []
            seq_len = 0

            # Drain any carryover from the previous sequence first. Carryover
            # means we're continuing mid-doc, so this sequence does not start
            # with a fresh BOS -- the next appended doc must still strip its
            # leading BOS (it's mid-sequence).
            if self._carry_tokens is not None:
                seq_parts.append(self._carry_tokens)
                seq_task_parts.append(self._carry_task_ids)
                seq_mask_parts.append(self._carry_assistant_mask)
                seq_meta.extend(self._carry_metadata)
                seq_len += len(self._carry_tokens)
                self._carry_tokens = None
                self._carry_task_ids = None
                self._carry_assistant_mask = None
                self._carry_metadata = []
                first_doc_in_seq = False
            else:
                first_doc_in_seq = True

            # Safety cap: if every doc the queue hands us fails validation,
            # bail so we don't spin forever.
            failed_attempts = 0
            max_failed_attempts = 100

            while seq_len < effective_block_size:
                if not self.message_queue:
                    # Queue drained. The parent InterleaveDataManager only
                    # refills once per get_batch, so we fall through to
                    # zero-padding below. (This matches prior behavior.)
                    break

                doc = self.message_queue.popleft()
                tokenized = self._tokenize_doc(
                    doc, omit_leading_bos=not first_doc_in_seq
                )
                if tokenized is None:
                    failed_attempts += 1
                    if failed_attempts > max_failed_attempts:
                        print(
                            f"[WARN] MessageQueue.get_batch: {max_failed_attempts} "
                            f"consecutive doc tokenizations/validations failed"
                        )
                        break
                    continue

                doc_tokens, doc_mask = tokenized
                raw_task = doc["metadata"].get("task_type", DEFAULT_TASK)
                try:
                    task_id_val = int(raw_task)
                except (TypeError, ValueError):
                    task_id_val = int(DEFAULT_TASK)
                doc_task = torch.full(
                    doc_tokens.shape, task_id_val, dtype=torch.uint8
                )

                remaining = effective_block_size - seq_len
                if len(doc_tokens) <= remaining:
                    seq_parts.append(doc_tokens)
                    seq_task_parts.append(doc_task)
                    seq_mask_parts.append(doc_mask)
                    seq_meta.extend([doc["metadata"]] * len(doc_tokens))
                    seq_len += len(doc_tokens)
                    first_doc_in_seq = False
                else:
                    fitting_tokens = doc_tokens[:remaining]
                    fitting_task = doc_task[:remaining]
                    fitting_mask = doc_mask[:remaining]
                    overflow_tokens = doc_tokens[remaining:]
                    overflow_task = doc_task[remaining:]
                    overflow_mask = doc_mask[remaining:]
                    seq_parts.append(fitting_tokens)
                    seq_task_parts.append(fitting_task)
                    seq_mask_parts.append(fitting_mask)
                    seq_meta.extend([doc["metadata"]] * len(fitting_tokens))
                    seq_len += len(fitting_tokens)
                    # Preserve the tail for the next sequence rather than
                    # discarding it.
                    self._carry_tokens = overflow_tokens
                    self._carry_task_ids = overflow_task
                    self._carry_assistant_mask = overflow_mask
                    self._carry_metadata = [doc["metadata"]] * len(overflow_tokens)
                    break

            # Pad with zeros if starvation leaves us short. Matches prior
            # behavior; the underlying refill path is the place to fix this
            # for real.
            if seq_len < effective_block_size:
                pad = effective_block_size - seq_len
                seq_parts.append(torch.zeros(pad, dtype=torch.long))
                seq_task_parts.append(
                    torch.full((pad,), int(DEFAULT_TASK), dtype=torch.uint8)
                )
                seq_mask_parts.append(torch.zeros(pad, dtype=torch.uint8))
                seq_meta.extend([{}] * pad)

            sequence = torch.cat(seq_parts)[:effective_block_size]
            task_seq = torch.cat(seq_task_parts)[:effective_block_size]
            mask_seq = torch.cat(seq_mask_parts)[:effective_block_size]
            sequences.append(sequence)
            task_id_seqs.append(task_seq)
            assistant_mask_seqs.append(mask_seq)
            batch_metadata.append(seq_meta[0] if seq_meta else {})

        return {
            "batch": sequences,
            "task_type_ids": task_id_seqs,
            "assistant_mask": assistant_mask_seqs,
            "metadata": batch_metadata,
        }

    def get_batch_with_rewards(
        self, batch_size: int, sequence_multiplier: int = 1
    ) -> Dict[str, Any]:
        """
        Get a batch with reward information preserved.

        Args:
            batch_size: Number of sequences in the batch
            sequence_multiplier: Factor to multiply the sequence length by

        Returns:
            Dictionary with batch, rewards, and metadata
        """
        result = self.get_batch(batch_size, sequence_multiplier)

        rewards = []
        for meta in result["metadata"]:
            reward = meta.get("reward", 0.0)
            rewards.append(reward)

        result["rewards"] = (
            torch.tensor(rewards, dtype=torch.float32) if rewards else None
        )

        return result

    def get_validation_stats(self) -> Dict[str, int]:
        """Get chat template validation statistics."""
        return self.validation_stats.copy()
