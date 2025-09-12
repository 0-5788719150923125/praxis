"""Text generation with request queuing and tool calling support."""

import contextlib
import json
import re
import uuid
from queue import Queue
from typing import Any, Dict, Optional

import torch

from praxis.generation.request import GenerationRequest


class Generator:
    """
    Wraps a model in a simplified generation API with request queuing.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.request_queue = Queue()
        self.results = {}

        from praxis.tools import call_tool, get_tools_json_schema

        self.tools = get_tools_json_schema()
        self.call_tool = call_tool
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

    def _process_single_request(self, request: GenerationRequest):
        """
        Process a single generation request, automatically handling tool calls if detected.
        Returns the generated text with proper message structure for tool calls.
        """
        # Check if the prompt is already a list of messages
        if isinstance(request.prompt, list):
            # Apply chat template to messages
            print(f"[DEBUG] Applying chat template to messages: {request.prompt}")
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    request.prompt, tokenize=False, add_generation_prompt=True
                )
                print(f"[DEBUG] Chat template output: {prompt_text[:200]}...")
            except Exception as e:
                print(f"[ERROR] Failed to apply chat template: {e}")
                # Fallback: convert messages to simple string
                prompt_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in request.prompt]
                )
                print(f"[DEBUG] Using fallback prompt: {prompt_text}")

            # Encode without return_tensors to get a list first
            input_ids = self.tokenizer.encode(prompt_text)
            print(f"[DEBUG] Encoded to {len(input_ids)} tokens")
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
        )
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
            while attempts < max_attempts:
                # Direct model access - no branching needed
                outputs = self.model.generate(
                    generated_tokens,
                    **combined,
                    tokenizer=self.tokenizer,
                    return_dict_in_generate=True,
                )

                # Update generated_tokens with the new token
                generated_tokens = outputs.sequences

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

        # Check if the generated text contains an unprocessed tool call
        unprocessed_call = self._get_unprocessed_tool_call(return_text)

        if unprocessed_call and self.tools and self.call_tool:
            tool_call, _ = unprocessed_call

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

            try:
                tool_result = self.call_tool(tool_name, tool_args)
                print(f"Called tool: {tool_name} with args: {tool_args}")
                print(f"Tool result: {tool_result}")

                # Build a new messages list with the tool result
                if isinstance(request.prompt, list):
                    messages = request.prompt.copy()
                else:
                    # Convert string prompt to messages format
                    messages = [{"role": "user", "content": request.prompt}]

                # Extract assistant's content (without the original prompt)
                assistant_content = return_text
                if prompt_text in assistant_content:
                    assistant_content = assistant_content.replace(
                        prompt_text, ""
                    ).strip()

                # Add assistant message if there's content
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

                # Add tool result as a proper tool message
                messages.append({"role": "tool", "content": str(tool_result)})

                # Create new request with proper messages
                tool_response_request = GenerationRequest(
                    id=request.id + "_tool_response",
                    prompt=messages,
                    kwargs=request.kwargs,
                )

                # Recursively process to get the model's response after tool execution
                # This will handle any additional tool calls the model might make
                final_response = self._process_single_request(tool_response_request)
                return final_response

            except Exception as e:
                print(f"Error calling tool {tool_name}: {e}")
                return return_text

        return return_text

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from generated text, returning the LAST complete tool call."""

        # Look for ALL tool call patterns
        tool_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(tool_pattern, text, re.DOTALL)

        # Process from last to first, returning the first valid JSON
        for match in reversed(matches):
            try:
                tool_data = json.loads(match)
                return tool_data
            except json.JSONDecodeError:
                continue

        return None

    def _get_unprocessed_tool_call(
        self, text: str
    ) -> Optional[tuple[Dict[str, Any], int]]:
        """
        Find the last tool call in the text.
        Returns a tuple of (tool_data, match_end_position) or None.
        Since we use message-based tool responses, we simply return the last valid tool call.
        """
        tool_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = list(re.finditer(tool_pattern, text, re.DOTALL))

        if not matches:
            return None

        # Return the last valid tool call
        for match in reversed(matches):
            try:
                tool_data = json.loads(match.group(1))
                return (tool_data, match.end())
            except json.JSONDecodeError:
                continue

        return None

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
            processed += 1

        return processed
