"""Message queue manager for efficient batching."""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from praxis.data.validators import ChatTemplateValidator
from praxis.tasks import DEFAULT_TASK

# Sentinel for "separator id not looked up yet", so that a legitimate None
# (format declares no separator) is cached instead of re-resolved per document.
_UNRESOLVED = object()


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

    Document boundaries are reported out-of-band as `block_ids` rather than
    marked in the token stream, so packing works for a format with no control
    tokens at all. Formats that declare a `document_separator` still get one
    appended (see `_terminate_doc`), because it is part of their wire format -
    but nothing downstream depends on finding it any more.

    Per-token side channels travel with the tokens through packing:
        task_type_ids: int8 task ID per token, copied from doc metadata.
        assistant_mask: 1 where the token is part of an assistant turn,
                        0 elsewhere -- emitted by the chat template's
                        {% generation %} blocks. Used downstream for
                        prompt-loss masking during SFT.
        block_ids: 1-based document index per token, restarting at 1 in
                        every sequence. Gates attention so one packed
                        document cannot read another.
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
        self._sep_id: Any = _UNRESOLVED

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

        # Tokenizers whose character-to-token map is not 1:1 (byte, char) get the
        # mask built segment-wise. HuggingFace's return_assistant_tokens_mask
        # maps CHARACTER offsets to token spans, which slips on multi-byte
        # characters there and silently misaligns the prompt-loss mask - it
        # trained on some prompt tokens and skipped some assistant ones on any
        # text with a curly quote, accent, em dash or emoji. See
        # praxis/tokenizers/chat_templates.py::tokenize_with_mask.
        from praxis.tokenizers.chat_templates import tokenize_with_mask

        try:
            direct = tokenize_with_mask(
                self.tokenizer, messages, omit_leading_bos=omit_leading_bos
            )
        except Exception as e:
            print(f"[WARNING] segment-wise mask failed ({e}); using offsets")
            direct = None

        if direct is not None:
            ids, masks = direct
            encoded = {"input_ids": ids, "assistant_masks": masks}
        else:
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

        return self._terminate_doc(doc_tokens, assistant_mask)

    def _terminate_doc(
        self, doc_tokens: torch.Tensor, assistant_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append the format's document separator, if it declares one.

        A no-op for formats with ``document_separator=None`` (``prose``), and
        that is now the interesting case: document boundaries reach the model
        as ``block_ids`` rather than as a token, so a separator is no longer
        needed to make packing work. It remains for formats whose wire format
        includes one - ``default`` writes ``[BOS]``/``[SEP]`` per turn and
        ``[EOS]`` per document, and that layout is unchanged.

        Added AFTER validation, because the validator checks the template's own
        output.

        The mask copies the document's last position rather than being pinned
        to 0 or 1: the separator is supervised exactly when the document ends
        on a generated turn. Zeroing it unconditionally would make it a symbol
        the model conditions on at every document end but is never trained to
        produce, which is the asymmetry ``prose`` was built to remove.
        """
        sep_id = self._separator_id()
        if sep_id is None or doc_tokens.numel() == 0:
            return doc_tokens, assistant_mask
        sep = torch.tensor([sep_id], dtype=doc_tokens.dtype)
        tail = assistant_mask[-1:].clone()
        return torch.cat([doc_tokens, sep]), torch.cat([assistant_mask, tail])

    def _separator_id(self) -> Optional[int]:
        """The document-separator id for this tokenizer's chat format."""
        if self._sep_id is _UNRESOLVED:
            from praxis.tokenizers.chat_templates import chat_format_of

            self._sep_id = chat_format_of(self.tokenizer).document_separator_id(
                self.tokenizer
            )
        return self._sep_id

    def get_batch(
        self, batch_size: int, sequence_multiplier: int = 1
    ) -> Dict[str, Any]:
        """
        Get a batch of sequences.

        Args:
            batch_size: Number of sequences in the batch
            sequence_multiplier: Factor to multiply the sequence length by

        Returns:
            Dictionary with 'batch' (list of tensors) and 'metadata' (list of
            dicts), plus the per-token side channels 'task_type_ids',
            'assistant_mask' and 'block_ids'.

            ``block_ids`` labels which packed document each position belongs
            to, and is what gates attention so one document cannot read
            another (praxis/attention/core.py). It is emitted here, rather
            than recovered downstream by scanning ``input_ids`` for a
            separator id, because packing is the only step that actually knows
            where the seams are. Deriving it from a token had three failure
            modes this avoids: it was silently inert for any chat format whose
            template emits no separator, it forced a control token to exist in
            the vocabulary purely as an out-of-band signal, and it could not
            distinguish a separator the model generated from one the packer
            wrote.
        """
        effective_block_size = self.block_size * sequence_multiplier

        sequences: List[torch.Tensor] = []
        task_id_seqs: List[torch.Tensor] = []
        assistant_mask_seqs: List[torch.Tensor] = []
        block_id_seqs: List[torch.Tensor] = []
        batch_metadata: List[Dict] = []

        for _ in range(batch_size):
            seq_parts: List[torch.Tensor] = []
            seq_task_parts: List[torch.Tensor] = []
            seq_mask_parts: List[torch.Tensor] = []
            seq_block_parts: List[torch.Tensor] = []
            seq_meta: List[Dict] = []
            seq_len = 0
            # Block ids are per-SEQUENCE and 1-based, matching
            # praxis.utils.create_block_ids: a block never spans two rows.
            block_counter = 0

            # Drain any carryover from the previous sequence first. Carryover
            # means we're continuing mid-doc, so this sequence does not start
            # with a fresh BOS -- the next appended doc must still strip its
            # leading BOS (it's mid-sequence).
            if self._carry_tokens is not None:
                seq_parts.append(self._carry_tokens)
                seq_task_parts.append(self._carry_task_ids)
                seq_mask_parts.append(self._carry_assistant_mask)
                # The carryover is the tail of a document that was split across
                # sequences. It is a document in its own right here: attention
                # cannot reach back to the half that landed in the previous row.
                block_counter = 1
                seq_block_parts.append(
                    torch.full(self._carry_tokens.shape, 1, dtype=torch.long)
                )
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
                # A new document opens a new attention block. The packer is the
                # only place that KNOWS where documents meet, so it says so
                # directly instead of leaving the model to infer it from a
                # separator id in the stream (see the block_ids note on
                # get_batch).
                block_counter += 1
                doc_block = torch.full(
                    doc_tokens.shape, block_counter, dtype=torch.long
                )
                raw_task = doc["metadata"].get("task_type", DEFAULT_TASK)
                try:
                    task_id_val = int(raw_task)
                except (TypeError, ValueError):
                    task_id_val = int(DEFAULT_TASK)
                doc_task = torch.full(doc_tokens.shape, task_id_val, dtype=torch.uint8)

                remaining = effective_block_size - seq_len
                if len(doc_tokens) <= remaining:
                    seq_parts.append(doc_tokens)
                    seq_task_parts.append(doc_task)
                    seq_mask_parts.append(doc_mask)
                    seq_block_parts.append(doc_block)
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
                    seq_block_parts.append(doc_block[:remaining])
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
                # Padding is its own block, so it cannot attend into the last
                # real document (nor that document into it).
                seq_block_parts.append(
                    torch.full((pad,), block_counter + 1, dtype=torch.long)
                )
                seq_meta.extend([{}] * pad)

            sequence = torch.cat(seq_parts)[:effective_block_size]
            task_seq = torch.cat(seq_task_parts)[:effective_block_size]
            mask_seq = torch.cat(seq_mask_parts)[:effective_block_size]
            block_seq = torch.cat(seq_block_parts)[:effective_block_size]
            sequences.append(sequence)
            task_id_seqs.append(task_seq)
            assistant_mask_seqs.append(mask_seq)
            block_id_seqs.append(block_seq)
            batch_metadata.append(seq_meta[0] if seq_meta else {})

        return {
            "batch": sequences,
            "task_type_ids": task_id_seqs,
            "assistant_mask": assistant_mask_seqs,
            "block_ids": block_id_seqs,
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
