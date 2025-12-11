"""Text generation with request queuing and inline tool calling support.

Tool calls are handled inline using <tin>...</tin> and <tout>...</tout> tags,
allowing the model to execute tools and continue generation in a single pass.
"""

import contextlib
import json
import re
import time
import uuid
from queue import Queue
from typing import Any, Dict, Optional

import torch
from transformers import GenerationConfig

from praxis.generation.request import GenerationRequest
from praxis.tools import (
    fix_truncated_tags,
    format_tool_output,
    get_unprocessed_tool_call,
    parse_tool_call,
)


class Generator:
    """
    Wraps a model in a simplified generation API with request queuing.
    """

    # Maximum number of results to keep in memory to prevent VRAM leaks
    MAX_RESULTS = 100

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.request_queue = Queue()
        self.results = {}
        self._result_order = []  # Track insertion order for cleanup

        from praxis.tools import call_tool, get_tools_json_schema

        self.tools = get_tools_json_schema()
        self.call_tool = call_tool

        # Tool calling safety limits
        self.max_tool_calls_per_request = 3  # Maximum recursive tool calls
        self.max_tool_call_time = 10.0  # Maximum time (seconds) for all tool calls

        print(f"[TOOLS]: Loaded {len(self.tools)} tools.")

    @contextlib.contextmanager
    def _eval_mode(self):
        # Simple eval mode handling - no Lightning-specific branching needed
        training = self.model.training
        self.model.eval()
        try:
            yield
        except Exception as e:
            import traceback

            print(f"[ERROR] Exception during generation: {e}")
            print(traceback.format_exc())
            raise
        finally:
            self.model.train(training)

    def _has_unclosed_tool_tag(self, text: str) -> bool:
        """Check if there's an open <tin> tag without matching </tin>.

        This is used to determine if we should suppress [SEP] as a stop token
        to allow the model to complete the tool call tag.

        Args:
            text: The generated text to check

        Returns:
            True if there's an unclosed <tin> tag
        """
        open_count = text.count("<tin>")
        close_count = text.count("</tin>")
        return open_count > close_count

    def request_generation(self, prompt, kwargs={}) -> str:
        """
        Submit a generation request and return a request ID.

        Args:
            prompt: Either a string prompt or a list of message dicts with 'role' and 'content'
            kwargs: Additional generation parameters

        Returns:
            Request ID string
        """
        request_id = str(uuid.uuid4())
        request = GenerationRequest(id=request_id, prompt=prompt, kwargs=kwargs)
        self.request_queue.put(request)
        return request_id

    def get_result(self, request_id: str) -> Optional[str]:
        """
        Check if a result is ready for a given request ID.
        Returns None if the result isn't ready yet.
        """
        result = self.results.get(request_id)
        if result is not None:
            del self.results[request_id]
            if request_id in self._result_order:
                self._result_order.remove(request_id)
        return result

    def generate_with_messages(self, messages: list, **kwargs) -> str:
        """
        Convenience method for generating with a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        request = GenerationRequest(
            id=str(uuid.uuid4()), prompt=messages, kwargs=kwargs
        )
        return self._process_single_request(request)

    def _process_single_request(
        self,
        request: GenerationRequest,
        tool_call_depth: int = 0,
        tool_call_history: list = None,
        start_time: float = None,
    ):
        """
        Process a single generation request, automatically handling tool calls if detected.
        Returns the generated text with proper message structure for tool calls.

        Args:
            request: The generation request to process
            tool_call_depth: Current recursion depth for tool calls (safety limit)
            tool_call_history: List of (tool_name, tool_args_json) tuples to detect duplicates
            start_time: Start time for timeout detection
        """
        # Initialize tracking on first call
        if tool_call_history is None:
            tool_call_history = []
        if start_time is None:
            start_time = time.time()
        # Check if the prompt is already a list of messages
        if isinstance(request.prompt, list):
            # Apply chat template to messages
            # print(f"[DEBUG] Applying chat template to messages: {request.prompt}")
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    request.prompt, tokenize=False, add_generation_prompt=True
                )
                # print(f"[DEBUG] Chat template output: {prompt_text[:200]}...")
            except Exception as e:
                print(f"[ERROR] Failed to apply chat template: {e}")
                # Fallback: convert messages to simple string
                prompt_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in request.prompt]
                )
                print(f"[DEBUG] Using fallback prompt: {prompt_text}")

            # Encode without return_tensors to get a list first
            input_ids = self.tokenizer.encode(prompt_text)
            # print(f"[DEBUG] Encoded to {len(input_ids)} tokens")
        else:
            # Legacy string prompt
            prompt_text = request.prompt
            # Encode without return_tensors to get a list first
            input_ids = self.tokenizer.encode(request.prompt)

        if isinstance(input_ids, list):
            # Get device from model parameters directly
            model_device = next(self.model.parameters()).device
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)

        # Ensure input_ids are on the same device as the model
        model_device = next(self.model.parameters()).device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)

        defaults = dict(
            do_sample=True,
            renormalize_logits=True,
            remove_invalid_values=True,
            # token_healing=True,
            # Stop generation at </tin> to allow tool execution
            stop_strings=["</tin>"],
        )

        # Add stop tokens if tokenizer has them
        # Store both variants for conditional stopping during tool calls
        eos_only = None
        eos_with_sep = None
        if (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            eos_only = [self.tokenizer.eos_token_id]
            if (
                hasattr(self.tokenizer, "sep_token_id")
                and self.tokenizer.sep_token_id is not None
            ):
                # Use both EOS and SEP as stop tokens (normal mode)
                eos_with_sep = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.sep_token_id,
                ]
                defaults["eos_token_id"] = eos_with_sep
            else:
                # Use only EOS as stop token
                defaults["eos_token_id"] = eos_only
        combined = {**defaults, **request.kwargs}

        # These values are largely an extension of the Huggingface `generate()` method, and
        # not supported by that API directly.
        if "prompt" in combined:
            del combined["prompt"]
        skip_special_tokens = True
        if "skip_special_tokens" in combined:
            if combined["skip_special_tokens"] == False:
                skip_special_tokens = False
            del combined["skip_special_tokens"]
        if "truncate_to" in combined:
            truncate_to = combined["truncate_to"]
            if input_ids.size(1) > truncate_to:
                input_ids = input_ids[:, -truncate_to:]
            del combined["truncate_to"]

        # Remove use_cache if set to False - it causes device placement issues
        # during inference with certain model architectures
        # if "use_cache" in combined and combined["use_cache"] == False:
        #     del combined["use_cache"]

        generated_tokens = input_ids

        max_attempts = 3
        attempts = 0
        # Initialize return_text based on prompt type
        if isinstance(request.prompt, list):
            return_text = prompt_text  # Use the text after applying chat template
        else:
            return_text = request.prompt  # String prompt

        # Store the original prompt length for extracting only generated text
        original_prompt_length = input_ids.shape[1]

        with self._eval_mode():
            # Track if we're in tool-tag completion mode
            completing_tool_tag = False
            max_tool_tag_continuation = 50  # Safety limit for tag completion

            while attempts < max_attempts:
                # Create GenerationConfig to avoid repeated warnings
                generation_config = GenerationConfig(**combined)

                # Direct model access - no branching needed
                outputs = self.model.generate(
                    generated_tokens,
                    generation_config=generation_config,
                    tokenizer=self.tokenizer,
                    return_dict_in_generate=True,
                )

                # Update generated_tokens with the new token
                generated_tokens = outputs.sequences

                # Check if we only generated stop tokens (SEP/EOS) AFTER a tool output
                # This specifically prevents the oscillation where SEP follows </tout>
                # But we allow SEP in normal chat flow (it's needed between messages)
                new_token_count = len(generated_tokens[0]) - original_prompt_length
                if new_token_count > 0 and new_token_count <= 2:
                    # Only check for the tool output case - SEP after </tout>
                    # Decode to check if we're in that specific scenario
                    prompt_text = return_text if isinstance(request.prompt, list) else request.prompt
                    if prompt_text.rstrip().endswith("</tout>"):
                        new_tokens = generated_tokens[0][-new_token_count:]
                        # Check if ALL new tokens are stop tokens
                        stop_token_ids = set()
                        if self.tokenizer.eos_token_id is not None:
                            stop_token_ids.add(self.tokenizer.eos_token_id)
                        if self.tokenizer.sep_token_id is not None:
                            stop_token_ids.add(self.tokenizer.sep_token_id)

                        all_stop_tokens = all(
                            tok.item() in stop_token_ids for tok in new_tokens
                        )
                        if all_stop_tokens:
                            # Only generated stop tokens after </tout>, return unchanged
                            # This prevents the oscillation where SEP is added then removed
                            return prompt_text

                # Always decode the full sequence (matching original behavior)
                # This ensures context accumulation works properly
                decoded_new = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=skip_special_tokens
                )

                # Check if the decoded text contains the replacement character
                if "�" not in decoded_new:
                    # Check if we actually generated something new
                    generated_token_count = len(generated_tokens[0])

                    # If we have more tokens than the prompt, something was generated
                    if generated_token_count > original_prompt_length:
                        return_text = decoded_new

                        # Check if we stopped with an unclosed tool tag
                        # If so, continue generating with only EOS (not SEP) to complete the tag
                        if self._has_unclosed_tool_tag(return_text) and eos_only:
                            if not completing_tool_tag:
                                completing_tool_tag = True
                                # Switch to EOS-only mode to complete the </tin> tag
                                combined["eos_token_id"] = eos_only
                                combined["max_new_tokens"] = max_tool_tag_continuation
                                # Continue from current position
                                continue
                            else:
                                # Already in completion mode, keep trying
                                max_tool_tag_continuation -= 1
                                if max_tool_tag_continuation > 0:
                                    continue
                                # Safety limit reached, exit with what we have
                                break
                        else:
                            # Tag is closed or no tool tag, we're done
                            # Restore normal stopping for future iterations if needed
                            if completing_tool_tag and eos_with_sep:
                                combined["eos_token_id"] = eos_with_sep
                                completing_tool_tag = False
                            break
                    # For string prompts, also check if decoded differs (handles edge cases)
                    elif (
                        isinstance(request.prompt, str)
                        and decoded_new != request.prompt
                    ):
                        return_text = decoded_new
                        break
                    else:
                        # No new tokens were generated, try again with more tokens
                        attempts += 1
                        combined["max_new_tokens"] = min(
                            combined.get("max_new_tokens", 1) + 1, 10
                        )
                        continue
                else:
                    # The decoded text contains '�', so we need to generate more tokens
                    attempts += 1
                    generated_tokens = input_ids
                    combined["max_new_tokens"] += 1
            else:
                # If we exhausted all attempts, return what we have
                return_text = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=skip_special_tokens
                )

        # Clean up any malformed tool call closing tags before processing
        return_text = fix_truncated_tags(return_text)

        # Check if the generated text contains an unprocessed tool call
        unprocessed_call = get_unprocessed_tool_call(return_text)

        if unprocessed_call and self.tools and self.call_tool:
            tool_call, _ = unprocessed_call

            # === SAFETY CHECK 1: Maximum recursion depth ===
            if tool_call_depth >= self.max_tool_calls_per_request:
                print(
                    f"[TOOL_SAFETY] Maximum tool call depth ({self.max_tool_calls_per_request}) reached. Stopping recursion."
                )
                print(f"[TOOL_SAFETY] Tool call history: {tool_call_history}")
                return return_text

            # === SAFETY CHECK 2: Timeout protection ===
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_tool_call_time:
                print(
                    f"[TOOL_SAFETY] Tool calling timeout ({self.max_tool_call_time}s) exceeded after {elapsed_time:.2f}s."
                )
                print(
                    f"[TOOL_SAFETY] Completed {tool_call_depth} tool calls before timeout."
                )
                return return_text

            # Execute the tool
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            # Debug: Log the actual tool call structure
            if tool_name is None:
                print(
                    f"Warning: Tool call missing 'name' field. Full tool_call: {tool_call}"
                )
                # Try alternative field names
                tool_name = tool_call.get("tool") or tool_call.get("function", {}).get(
                    "name"
                )
                if tool_name is None:
                    print(
                        f"Error: Could not extract tool name from tool call: {tool_call}"
                    )
                    return return_text

            # === SAFETY CHECK 3: Duplicate tool call detection ===
            # Create a signature for this tool call (name + sorted args)
            tool_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
            if tool_signature in tool_call_history:
                print(
                    f"[TOOL_SAFETY] Duplicate tool call detected: {tool_name}({tool_args})"
                )
                print(
                    f"[TOOL_SAFETY] This tool was already called with identical arguments."
                )
                print(f"[TOOL_SAFETY] Stopping to prevent infinite loop.")
                print(
                    f"[TOOL_SAFETY] Tool call history: {[sig[0] for sig in tool_call_history]}"
                )
                return return_text

            try:
                tool_result = self.call_tool(tool_name, tool_args)
                print(f"Called tool: {tool_name} with args: {tool_args}")
                print(f"Tool result: {tool_result}")

                # Add this tool call to history (after successful execution)
                tool_call_history.append(tool_signature)

                # Inline tool execution: inject <tout>result</tout> and continue generation
                # Find the end position of the tool call (after </tin>)
                _, tin_end_pos = unprocessed_call

                # Inject the tool output tag immediately after </tin>
                tool_output_tag = format_tool_output(tool_result)
                text_with_result = (
                    return_text[:tin_end_pos]
                    + tool_output_tag
                    + return_text[tin_end_pos:]
                )

                # Direct continuation: encode the text with result and continue generating
                # This avoids rebuilding messages and reapplying chat template (which adds [BOS]assistant)
                new_input_ids = self.tokenizer.encode(text_with_result)

                if isinstance(new_input_ids, list):
                    model_device = next(self.model.parameters()).device
                    new_input_ids = torch.tensor(
                        [new_input_ids], dtype=torch.long, device=model_device
                    )

                # Ensure input_ids are on the same device as the model
                model_device = next(self.model.parameters()).device
                if new_input_ids.device != model_device:
                    new_input_ids = new_input_ids.to(model_device)

                # Calculate remaining tokens to generate
                original_max_tokens = request.kwargs.get("max_new_tokens", 100)
                tokens_used = new_input_ids.shape[1] - original_prompt_length
                remaining_tokens = original_max_tokens - tokens_used

                # If we've exhausted our token budget, return the text with tool result
                if remaining_tokens <= 0:
                    return text_with_result

                # Continue generation directly from the injected position
                # No chat template reapplication - just generate more tokens
                continuation_config = GenerationConfig(
                    max_new_tokens=remaining_tokens,
                    do_sample=combined.get("do_sample", True),
                    renormalize_logits=combined.get("renormalize_logits", True),
                    remove_invalid_values=combined.get("remove_invalid_values", True),
                    eos_token_id=eos_with_sep if eos_with_sep else eos_only,
                    # Stop at tool calls in continuation (for chained tool calls)
                    stop_strings=["</tin>"],
                )

                with self._eval_mode():
                    continuation_output = self.model.generate(
                        new_input_ids,
                        generation_config=continuation_config,
                        tokenizer=self.tokenizer,
                        return_dict_in_generate=True,
                    )

                # Decode the continuation
                continuation_text = self.tokenizer.decode(
                    continuation_output.sequences[0],
                    skip_special_tokens=skip_special_tokens,
                )

                # Clean up any malformed tags in the continuation
                continuation_text = fix_truncated_tags(continuation_text)

                # Check if continuation generated only noise/separators
                # Decode ONLY the newly generated tokens (after tool injection)
                new_tokens_only = continuation_output.sequences[0][new_input_ids.shape[1]:]
                if len(new_tokens_only) > 0:
                    continuation_only = self.tokenizer.decode(
                        new_tokens_only,
                        skip_special_tokens=skip_special_tokens,
                    )
                    # If continuation is just whitespace or common separators, don't use it
                    if continuation_only.strip() in ['', '[SEP]', '[BOS]', 'assistant', '>']:
                        # Model generated nothing meaningful, return the tool result
                        return text_with_result

                # Check for additional tool calls in the continuation (recursive handling)
                # Use the same safety checks (depth, timeout, duplicates)
                if tool_call_depth + 1 < self.max_tool_calls_per_request:
                    elapsed = time.time() - start_time
                    if elapsed < self.max_tool_call_time:
                        next_call = get_unprocessed_tool_call(continuation_text)
                        if next_call:
                            # Create a simple request for recursive processing
                            # Use the raw text as prompt to avoid chat template reapplication
                            continuation_request = GenerationRequest(
                                id=request.id + "_tool_continuation",
                                prompt=continuation_text,  # Raw text, not messages
                                kwargs={"max_new_tokens": remaining_tokens},
                            )
                            return self._process_single_request(
                                continuation_request,
                                tool_call_depth=tool_call_depth + 1,
                                tool_call_history=tool_call_history,
                                start_time=start_time,
                            )

                return continuation_text

            except Exception as e:
                print(f"Error calling tool {tool_name}: {e}")
                return return_text

        return return_text

    def fulfill_requests(self, max_requests: int = None) -> int:
        """
        Process pending generation requests. Should be called from inside the training loop.
        Returns the number of requests processed.
        """
        processed = 0
        while not self.request_queue.empty():
            if max_requests is not None and processed >= max_requests:
                break

            request = self.request_queue.get()
            result = self._process_single_request(request)
            self.results[request.id] = result
            self._result_order.append(request.id)

            # Prevent memory leaks by limiting results dictionary size
            while len(self.results) > self.MAX_RESULTS:
                oldest_id = self._result_order.pop(0)
                if oldest_id in self.results:
                    del self.results[oldest_id]

            processed += 1

        return processed
