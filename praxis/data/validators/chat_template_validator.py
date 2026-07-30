"""Chat template validation: turn boundaries must name a real role.

What a turn boundary IS depends on the active chat format, so this validator
has two modes:

- Control-token formats (``chat_format: default``): a structural ``[BOS]`` must
  be followed by a valid role name, catching data corruption where random text
  lands after a BOS.
- Text-boundary formats (``chat_format: prose``): there is no BOS to anchor on,
  so instead every message's ``\\n\\n<role>\\n\\n`` boundary must appear in the
  rendered text, in message order. That catches the failure that actually
  matters here - a renderer that dropped or reordered a boundary, which would
  train the model on turns it cannot delimit.

Running the BOS check against a prose document would fail every document and
silently drain the training stream, which is why the mode is chosen from the
format rather than assumed.
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from praxis.tokenizers.chat_templates import chat_format_of


class ChatTemplateValidator:
    """Validates that a rendered document's turn boundaries are well-formed."""

    # Fallback role set for tokenizers with no format attached.
    ALLOWED_ROLES = ["system", "developer", "user", "assistant", "tool"]

    def __init__(self, tokenizer: PreTrainedTokenizer, strict_mode: bool = False):
        """Initialize the validator.

        Args:
            tokenizer: The tokenizer to use for decoding
            strict_mode: If True, raise exceptions on violations.
                        If False, return validation status.
        """
        self.tokenizer = tokenizer
        self.strict_mode = strict_mode
        self.chat_format = chat_format_of(tokenizer)
        if self.chat_format.roles:
            self.ALLOWED_ROLES = list(self.chat_format.roles)

        # Get BOS token ID
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.bos_token_id = tokenizer.bos_token_id
        else:
            # Fallback: try to encode the BOS token string
            bos_token = tokenizer.special_tokens_map.get("bos_token", "[BOS]")
            self.bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)

        # Get SEP token ID for structural validation
        if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None:
            self.sep_token_id = tokenizer.sep_token_id
        else:
            sep_token = tokenizer.special_tokens_map.get("sep_token", "[SEP]")
            self.sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

        # Pre-tokenize role names for faster checking
        self._role_token_prefixes = {}
        for role in self.ALLOWED_ROLES:
            # Tokenize just the role name to see what tokens it produces
            role_tokens = tokenizer.encode(role, add_special_tokens=False)
            self._role_token_prefixes[role] = role_tokens

    def _expected_boundary(self, index: int, role: str) -> str:
        """The text opening message ``index``'s turn under a text format.

        The first turn has no leading blank line - it opens the document - so
        its boundary is the bare role line.
        """
        boundary = self.chat_format.boundary(role)
        return boundary.lstrip("\n") if index == 0 else boundary

    def _validate_text_boundaries(
        self, token_ids: torch.Tensor, messages: Optional[List[Dict]]
    ) -> Tuple[bool, List[Dict]]:
        """Check that each message's boundary appears, in order, in the render.

        Without ``messages`` there is nothing to check against: a bare token
        stream has no ground truth for where its boundaries belong, and role
        names are ordinary words, so scanning for unexpected boundaries would
        flag any prose containing a blank-line-delimited word.
        """
        if not messages:
            return True, []

        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        violations: List[Dict] = []
        cursor = 0
        for index, message in enumerate(messages):
            role = message.get("role", "")
            expected = self._expected_boundary(index, role)
            found = text.find(expected, cursor)
            if found == -1:
                violations.append(
                    {
                        "message_index": index,
                        "role": role,
                        "expected_boundary": expected,
                        "search_from": cursor,
                        "full_sequence_length": len(token_ids),
                    }
                )
                break
            cursor = found + len(expected)
        return len(violations) == 0, violations

    def validate_token_sequence(
        self, token_ids: torch.Tensor, messages: List[Dict] = None
    ) -> Tuple[bool, List[Dict]]:
        """Validate the document's turn boundaries.

        Token-boundary formats: structural BOS tokens (at position 0 or right
        after a SEP) must be followed by a role name; BOS tokens occurring
        naturally in content (e.g. inside a code string) are ignored.

        Text-boundary formats: every message's role boundary must appear in the
        rendered text, in order.

        Args:
            token_ids: Tensor of token IDs to validate
            messages: Optional original messages for debugging context

        Returns:
            Tuple of (is_valid, violations_list)
        """
        if self.chat_format.text_boundaries:
            return self._validate_text_boundaries(token_ids, messages)

        violations = []

        # Find all BOS token positions
        bos_positions = (token_ids == self.bos_token_id).nonzero(as_tuple=True)[0]

        for pos in bos_positions:
            # Only validate structural BOS tokens:
            # 1. At position 0 (start of document)
            # 2. Immediately after a SEP token
            is_structural = pos == 0 or (
                pos > 0 and token_ids[pos - 1] == self.sep_token_id
            )

            if not is_structural:
                # This BOS is in content, skip validation
                continue

            # Skip if BOS is the last token
            if pos >= len(token_ids) - 1:
                continue

            # Get tokens after BOS (up to 15 tokens to capture full role name)
            # Most role names are 1-3 tokens, but we check more to be safe
            next_token_count = min(15, len(token_ids) - pos - 1)
            next_tokens = token_ids[pos + 1 : pos + 1 + next_token_count]

            # Decode to text
            next_text = self.tokenizer.decode(
                next_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()

            # Check if starts with valid role
            valid = self._check_role_prefix(next_tokens, next_text)

            if not valid:
                violation = {
                    "position": pos.item(),
                    "bos_token_id": self.bos_token_id,
                    "next_tokens": next_tokens.tolist(),
                    "decoded_text": next_text[:100],  # First 100 chars
                    "full_sequence_length": len(token_ids),
                    "is_structural": is_structural,
                }
                violations.append(violation)

        is_valid = len(violations) == 0
        return is_valid, violations

    def _check_role_prefix(self, next_tokens: torch.Tensor, decoded_text: str) -> bool:
        """Check if tokens/text starts with a valid role name.

        Args:
            next_tokens: Token IDs following BOS
            decoded_text: Decoded text of those tokens

        Returns:
            True if starts with valid role, False otherwise
        """
        # Method 1: Check decoded text
        for role in self.ALLOWED_ROLES:
            if decoded_text.startswith(role):
                return True

        # Method 2: Check token sequence directly
        # This handles cases where decoding might have issues
        for role, role_tokens in self._role_token_prefixes.items():
            if len(next_tokens) >= len(role_tokens):
                # Check if first N tokens match the role tokens
                if torch.all(
                    next_tokens[: len(role_tokens)] == torch.tensor(role_tokens)
                ):
                    return True

        return False

    def format_violation_report(
        self,
        violations: List[Dict],
        messages: List[Dict] = None,
        formatted_text: str = None,
        token_ids: torch.Tensor = None,
    ) -> str:
        """Format a detailed violation report for debugging.

        Args:
            violations: List of violation dictionaries
            messages: Optional original messages
            formatted_text: Optional formatted template output
            token_ids: Optional full token sequence

        Returns:
            Formatted error message string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CHAT TEMPLATE VALIDATION FAILURE")
        lines.append("=" * 80)

        # Show original messages if available
        if messages:
            lines.append("\nOriginal Messages:")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_preview = content[:100] + ("..." if len(content) > 100 else "")
                lines.append(f"  [{i}] {role}: {content_preview}")

        # Show formatted template output if available
        if formatted_text:
            lines.append("\nFormatted Template Output:")
            preview = formatted_text[:500] + (
                "..." if len(formatted_text) > 500 else ""
            )
            lines.append(preview)

        # Show violations. The two boundary styles produce different violation
        # shapes, so report whichever keys are present.
        lines.append(f"\nFound {len(violations)} violation(s):")
        for i, v in enumerate(violations):
            lines.append(f"\n  Violation {i + 1}:")
            if "expected_boundary" in v:
                lines.append(f"    Message index: {v['message_index']}")
                lines.append(f"    Role: {v['role']}")
                lines.append(f"    Missing boundary: {v['expected_boundary']!r}")
                lines.append(f"    Searched from char: {v['search_from']}")
            else:
                lines.append(f"    Position: {v['position']}")
                lines.append(f"    BOS token ID: {v['bos_token_id']}")
                lines.append(f"    Next tokens: {v['next_tokens'][:10]}...")
                lines.append(f"    Decoded text: '{v['decoded_text']}'")
                lines.append(f"    Expected one of: {', '.join(self.ALLOWED_ROLES)}")

        # Show token sequence preview if available
        if token_ids is not None:
            lines.append("\nToken Sequence (first 100 tokens):")
            token_preview = token_ids[:100].tolist()
            lines.append(str(token_preview))

            # Decode first 100 tokens for context
            lines.append("\nDecoded Sequence (first 100 tokens):")
            decoded_preview = self.tokenizer.decode(
                token_ids[:100], skip_special_tokens=False
            )
            lines.append(
                decoded_preview[:300] + ("..." if len(decoded_preview) > 300 else "")
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def validate_and_report(
        self,
        token_ids: torch.Tensor,
        messages: List[Dict] = None,
        formatted_text: str = None,
    ) -> Tuple[bool, str]:
        """Validate token sequence and generate report if invalid.

        Convenience method that combines validation and reporting.

        Args:
            token_ids: Token sequence to validate
            messages: Optional original messages
            formatted_text: Optional formatted template output

        Returns:
            Tuple of (is_valid, report_string)
        """
        is_valid, violations = self.validate_token_sequence(token_ids, messages)

        if not is_valid:
            report = self.format_violation_report(
                violations,
                messages=messages,
                formatted_text=formatted_text,
                token_ids=token_ids,
            )
            return False, report

        return True, ""
