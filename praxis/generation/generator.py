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

        # Try to import tools, but don't fail if they're not available
        try:
            from praxis.tools import call_tool, get_tools_json_schema

            self.tools = get_tools_json_schema()
            self.call_tool = call_tool
            print(f"[TOOLS]: Loaded {len(self.tools)} tools.")
        except ImportError:
            self.tools = []
            self.call_tool = None
            print("Tools module not available, function calling disabled")

    @contextlib.contextmanager
    def _eval_mode(self):
        training = self.model.training
        self.model.eval()
        try:
            yield
        except Exception as e:
            import traceback

            print(traceback.format_exc())
        finally:
            self.model.train(training)

    def request_generation(self, prompt: str, kwargs={}) -> str:
        """
        Submit a generation request and return a request ID.
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

    def _process_single_request(self, request: GenerationRequest):
        """
        Process a single generation request, automatically handling tool calls if detected.
        """
        input_ids = self.tokenizer.encode(request.prompt, return_tensors="pt")

        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)

        if self.device.startswith("cuda"):
            input_ids = input_ids.to(self.device)

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

        generated_tokens = input_ids

        max_attempts = 3
        attempts = 0
        return_text = request.prompt  # Initialize with the prompt as default

        with self._eval_mode():
            while attempts < max_attempts:
                outputs = self.model.generate(
                    generated_tokens,
                    **combined,
                    tokenizer=self.tokenizer,
                    return_dict_in_generate=True,
                )

                # Update generated_tokens with the new token
                generated_tokens = outputs.sequences

                # Decode the tokens generated so far
                decoded_new = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=skip_special_tokens
                )

                # Check if the decoded text contains the replacement character
                if "�" not in decoded_new:
                    # Check if we actually generated something new
                    # Compare token lengths to detect if model generated whitespace
                    prompt_tokens = len(self.tokenizer.encode(request.prompt))
                    generated_token_count = len(generated_tokens[0])

                    # If we have more tokens than the prompt, something was generated
                    if generated_token_count > prompt_tokens:
                        return_text = decoded_new
                        break
                    # If the decoded text is different, use it
                    elif decoded_new != request.prompt:
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

                # Append the tool result directly as a simple tag
                # This preserves the exact format and allows for multiple tool calls
                tool_result_tag = f"\n<tool_result>{str(tool_result)}</tool_result>\n"

                # Build the complete prompt with tool result for continuation
                complete_prompt = return_text + tool_result_tag

                # Create a new generation request with the tool result included
                # This allows the model to continue generating (possibly more tool calls)
                tool_response_request = GenerationRequest(
                    id=request.id + "_tool_response",
                    prompt=complete_prompt,
                    kwargs=request.kwargs,  # Use same generation parameters
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
        Find the last unprocessed tool call in the text.
        Returns a tuple of (tool_data, match_end_position) or None.
        A tool call is considered processed if there's a <tool_result> tag after it.
        """
        tool_pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = list(re.finditer(tool_pattern, text, re.DOTALL))

        if not matches:
            return None

        # Check each tool call from last to first
        for match in reversed(matches):
            # Check if there's a tool_result tag after THIS specific tool call
            text_after_this_call = text[match.end() :]

            # Look for the next tool_result tag after this specific tool call
            # If there's no tool_result or there's another tool_call before the tool_result,
            # then this tool call is unprocessed
            next_tool_call_pos = text_after_this_call.find("<tool_call>")
            next_tool_result_pos = text_after_this_call.find("<tool_result>")

            # This tool is unprocessed if:
            # 1. There's no tool_result after it, OR
            # 2. There's another tool_call before the next tool_result
            is_unprocessed = next_tool_result_pos == -1 or (
                next_tool_call_pos != -1 and next_tool_call_pos < next_tool_result_pos
            )

            if is_unprocessed:
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
