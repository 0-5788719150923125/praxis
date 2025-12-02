# Tool Calling Fix: Implementation Plan

## Problem Summary

When generating tool calls, the `[SEP]` token (used as an EOS token) can interrupt generation mid-tag, causing:
1. Truncated `</tin>` tags (e.g., just `<` instead of `</tin>`)
2. Spurious `[SEP][BOS]assistant` blocks inserted mid-stream
3. Malformed output like `/tin><tout>result</tout>` appearing after the new assistant block

## Design Goal

- Regular prompts continue to stop on `[SEP]` as expected
- Tool call generation completes the `</tin>` tag before stopping
- Eliminate the complex parse-fix-rebuild-continue cycle
- Single inline injection of `<tout>result</tout>` without re-encoding

---

## Implementation Plan

### Step 1: Create a Tag-Aware Stopping Criteria

**File**: `praxis/generation/stopping.py` (new file)

Create a custom `StoppingCriteria` that tracks whether we're inside a `<tin>` block:

```python
class ToolTagStoppingCriteria(StoppingCriteria):
    """
    Prevents stopping on [SEP] while inside an unclosed <tin> tag.

    Behavior:
    - If no <tin> has been generated: allow normal [SEP] stopping
    - If <tin> opened but </tin> not closed: suppress [SEP] stopping
    - Once </tin> is complete: allow normal stopping again
    """

    def __init__(self, tokenizer, sep_token_id):
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id
        # Pre-encode the tag tokens for fast checking
        self.tin_open_ids = tokenizer.encode("<tin>", add_special_tokens=False)
        self.tin_close_ids = tokenizer.encode("</tin>", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        # Decode recent tokens to check tag state
        # Only decode the tail (optimization)
        text = self.tokenizer.decode(input_ids[0, -50:], skip_special_tokens=False)

        # Count open vs closed tags
        open_count = text.count("<tin>")
        close_count = text.count("</tin>")

        inside_tag = open_count > close_count

        if inside_tag:
            # Don't stop - we need to complete the tag
            return False

        # Outside tag - allow normal stopping (let other criteria decide)
        return False  # Let the default EOS logic handle it
```

**Note**: The actual implementation may need refinement - the stopping criteria returns `True` to STOP, `False` to CONTINUE. We need to intercept the EOS logic, not just add another criterion.

### Step 2: Modify Generator to Use Conditional EOS Tokens

**File**: `praxis/generation/generator.py`

Instead of a stopping criteria, a cleaner approach is to dynamically set `eos_token_id` based on generation state:

```python
def _process_single_request(self, request, ...):
    # ... existing setup ...

    # Start with only true EOS, not SEP
    # SEP will be added back once tool tags are closed
    generation_eos_ids = [self.tokenizer.eos_token_id]

    # Track if we're in tool-calling mode
    tool_mode = False

    while attempts < max_attempts:
        # Check current state for open tool tags
        if tool_mode or self._has_unclosed_tool_tag(return_text):
            # Only use EOS, not SEP
            combined["eos_token_id"] = generation_eos_ids
            tool_mode = True
        else:
            # Normal mode: use both EOS and SEP
            combined["eos_token_id"] = [
                self.tokenizer.eos_token_id,
                self.tokenizer.sep_token_id,
            ]

        # ... rest of generation loop ...
```

Add helper method:

```python
def _has_unclosed_tool_tag(self, text: str) -> bool:
    """Check if there's an open <tin> tag without matching </tin>."""
    open_count = text.count("<tin>")
    close_count = text.count("</tin>")
    return open_count > close_count
```

### Step 3: Simplify Tool Result Injection

**File**: `praxis/generation/generator.py`

Replace the complex message-rebuilding continuation with direct token injection:

**Current flow** (lines 358-433):
1. Find unprocessed tool call
2. Execute tool
3. Rebuild messages list with assistant content
4. Create new GenerationRequest
5. Recursively call `_process_single_request`
6. Chat template re-applied, adding `[BOS]assistant`

**New flow**:
1. Find unprocessed tool call
2. Execute tool
3. Inject `<tout>result</tout>` directly into the text
4. Re-encode ONLY the current text (no chat template reapplication)
5. Continue generation from that position

```python
# After tool execution (around line 358)
tool_output_tag = format_tool_output(tool_result)
text_with_result = (
    return_text[:tin_end_pos]
    + tool_output_tag
    + return_text[tin_end_pos:]
)

# Direct continuation - NO message rebuilding
new_input_ids = self.tokenizer.encode(text_with_result, add_special_tokens=False)
new_input_ids = torch.tensor([new_input_ids], dtype=torch.long, device=model_device)

# Update tracking
generated_tokens = new_input_ids
original_prompt_length = new_input_ids.shape[1]  # Reset baseline

# Continue the while loop - don't recurse
# The loop will generate more tokens from this position
```

This eliminates the recursive call and the chat template reapplication that was adding the spurious `[BOS]assistant`.

### Step 4: Remove Tag Fixing as Primary Defense

**File**: `praxis/tools/tags.py`

Keep `fix_truncated_tags()` as a safety net, but it should rarely be needed after Steps 2-3. Consider:

1. Add logging when fixes are applied (to detect if the new approach has gaps)
2. Simplify the function since fewer edge cases should occur

```python
def fix_truncated_tags(text: str) -> str:
    """
    Safety net for malformed tags. Should rarely trigger after
    the stopping criteria fix.
    """
    original = text
    # ... existing fixes ...

    if text != original:
        print(f"[WARN] fix_truncated_tags applied: {repr(original[-50:])} -> {repr(text[-50:])}")

    return text
```

---

## Testing Plan

### Unit Tests

1. **Test normal prompts still stop on SEP**:
   - Generate a simple response with no tool calls
   - Verify it stops at `[SEP]` as expected

2. **Test tool call tag completion**:
   - Generate a response that includes a `<tin>` block
   - Verify `</tin>` is always complete before stopping

3. **Test inline result injection**:
   - Generate a tool call
   - Verify `<tout>result</tout>` appears immediately after `</tin>`
   - Verify no spurious `[BOS]assistant` blocks appear

4. **Test multi-tool sequences**:
   - Generate responses with 2-3 tool calls
   - Verify each completes correctly

### Integration Tests

1. Run the dashboard with the calc tool (as shown in screenshot)
2. Verify the context panel shows clean sequences:
   ```
   [BOS]assistant
   <tin>
   {"name": "calc", ...}
   </tin><tout>result</tout>
   The answer is...
   [SEP]
   ```

---

## Files to Modify

| File | Changes |
|------|---------|
| `praxis/generation/generator.py` | Add `_has_unclosed_tool_tag()`, modify EOS token logic, simplify continuation |
| `praxis/tools/tags.py` | Add warning logging to `fix_truncated_tags()` |
| `tests/test_tools.py` | Add new test cases for the stopping behavior |

## Files to Create

| File | Purpose |
|------|---------|
| `praxis/generation/stopping.py` | (Optional) Custom StoppingCriteria if the simpler approach doesn't work |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking regular prompts | Conditional EOS logic only activates when `<tin>` is detected |
| Infinite generation if `</tin>` never produced | Keep `max_new_tokens` as hard limit; add `max_tool_tag_tokens` safety |
| Tool execution errors mid-generation | Existing error handling remains; inject error message as `<tout>` content |

---

## Rollback Plan

If issues arise:
1. Revert generator.py changes
2. The existing `fix_truncated_tags()` approach remains as fallback
3. No data format changes required

---

## Estimated Scope

- ~50-80 lines of code changes in generator.py
- ~10 lines of logging additions in tags.py
- ~30-50 lines of new tests
- No changes to training data format or tokenizer
