"""Text generation with request queuing and inline tool calling support.

Tool calls are marked by atomic special tokens ``[TOOL_CALL]`` /
``[/TOOL_CALL]``. The generator halts on ``[/TOOL_CALL]`` via the
tokenizer's eos_token_id set, executes the tool, and splices
``[TOOL_RESULT] result [/TOOL_RESULT]`` as tokens (not text) before
continuing.
"""

import contextlib
import json
import time
import uuid
from queue import Queue
from typing import Any, Dict, Optional

import torch
from transformers import GenerationConfig

from praxis.generation.request import GenerationRequest
from praxis.tools import (
    build_result_splice_ids,
    find_unprocessed_tool_call_ids,
    tool_token_ids,
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

    def _eos_token_id_list(self) -> Optional[list]:
        """Build the eos_token_id list for a chat-style generation.

        The model halts on any of these token ids. The tool-call close
        token is always included when the tokenizer knows about it -
        that way generation halts cleanly at ``[/TOOL_CALL]`` so the
        tool can be executed.
        """
        ids = []
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.append(int(eos))
        sep = getattr(self.tokenizer, "sep_token_id", None)
        if sep is not None:
            ids.append(int(sep))
        tool_close = tool_token_ids(self.tokenizer).get("call_close")
        if tool_close is not None and tool_close not in ids:
            ids.append(int(tool_close))
        return ids or None

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

    def _process_single_request(self, request: GenerationRequest):
        """Process a generation request, executing any tool calls in line.

        The flow is a single loop:
          1. Generate up to the next stop token (eos / sep / ``[/TOOL_CALL]``).
          2. Look for an unprocessed tool call past ``processed_position``.
          3. If found, execute it, splice the real ``[TOOL_RESULT]`` ids,
             advance ``processed_position`` past the splice, and continue
             generating.
          4. If not found, return the decoded text.

        ``processed_position`` is the only way the loop knows a call has
        been handled - the model's own ``[TOOL_RESULT]`` output is treated
        as text, not as proof of execution. This stops the model from
        fooling us with a hallucinated result block.
        """
        start_time = time.time()
        tool_call_history: list = []

        if isinstance(request.prompt, list):
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    request.prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to apply chat template: {e}")
                prompt_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in request.prompt]
                )
            input_ids = self.tokenizer.encode(prompt_text)
            return_text = prompt_text
        else:
            prompt_text = request.prompt
            input_ids = self.tokenizer.encode(request.prompt)
            return_text = request.prompt

        model_device = next(self.model.parameters()).device
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)
        elif input_ids.device != model_device:
            input_ids = input_ids.to(model_device)

        defaults = dict(
            do_sample=True,
            renormalize_logits=True,
            remove_invalid_values=True,
        )
        eos_list = self._eos_token_id_list()
        if eos_list:
            defaults["eos_token_id"] = eos_list
        combined = {**defaults, **request.kwargs}

        if "prompt" in combined:
            del combined["prompt"]
        skip_special_tokens = True
        if "skip_special_tokens" in combined:
            if combined["skip_special_tokens"] is False:
                skip_special_tokens = False
            del combined["skip_special_tokens"]
        if "truncate_to" in combined:
            truncate_to = combined["truncate_to"]
            if input_ids.size(1) > truncate_to:
                input_ids = input_ids[:, -truncate_to:]
            del combined["truncate_to"]

        original_max_tokens = int(combined.get("max_new_tokens", 100))
        original_prompt_length = input_ids.shape[1]
        generated_tokens = input_ids
        tool_call_depth = 0

        with self._eval_mode():
            while True:
                tokens_used = generated_tokens.shape[1] - original_prompt_length
                remaining_tokens = max(0, original_max_tokens - tokens_used)
                if remaining_tokens <= 0:
                    break

                step_kwargs = dict(combined)
                step_kwargs["max_new_tokens"] = remaining_tokens
                generation_config = GenerationConfig(**step_kwargs)
                outputs = self.model.generate(
                    generated_tokens,
                    generation_config=generation_config,
                    tokenizer=self.tokenizer,
                    return_dict_in_generate=True,
                )
                new_generated = outputs.sequences
                if new_generated.shape[1] <= generated_tokens.shape[1]:
                    generated_tokens = new_generated
                    break
                generated_tokens = new_generated

                # Scan from position 0: in streaming mode the [TOOL_CALL]
                # open may live in the prompt (added on a prior tick) and
                # only [/TOOL_CALL] is fresh. The parser's complete-result
                # heuristic skips calls we've already spliced for.
                token_list = generated_tokens[0].tolist()
                unprocessed = (
                    find_unprocessed_tool_call_ids(token_list, self.tokenizer)
                    if self.tools and self.call_tool is not None
                    else None
                )
                if unprocessed is None:
                    break

                if tool_call_depth >= self.max_tool_calls_per_request:
                    print(
                        f"[TOOL_SAFETY] Maximum tool call depth "
                        f"({self.max_tool_calls_per_request}) reached."
                    )
                    break
                if time.time() - start_time > self.max_tool_call_time:
                    print(
                        f"[TOOL_SAFETY] Tool calling timeout "
                        f"({self.max_tool_call_time}s) exceeded."
                    )
                    break

                tool_call, call_end_index = unprocessed

                if tool_call.get("_malformed"):
                    err = tool_call.get("_error", "malformed tool call")
                    print(f"Error: {err}")
                    result_payload = f"Error: {err}"
                else:
                    tool_name = tool_call.get("name") or tool_call.get("tool") or (
                        tool_call.get("function", {}) or {}
                    ).get("name")
                    tool_args = tool_call.get("arguments", {})

                    if tool_name is None:
                        print(f"Error: Could not extract tool name from: {tool_call}")
                        result_payload = "Error: tool call is missing the 'name' field."
                    else:
                        signature = (tool_name, json.dumps(tool_args, sort_keys=True))
                        if signature in tool_call_history:
                            print(
                                f"[TOOL_SAFETY] Duplicate tool call detected: "
                                f"{tool_name}({tool_args}); stopping."
                            )
                            break
                        tool_call_history.append(signature)

                        try:
                            tool_result = self.call_tool(tool_name, tool_args)
                            print(f"Called tool: {tool_name} with args: {tool_args}")
                            print(f"Tool result: {tool_result}")
                            result_payload = tool_result
                        except Exception as e:
                            print(f"Error calling tool {tool_name}: {e}")
                            result_payload = f"Error: {e}"

                tool_call_depth += 1

                result_ids = build_result_splice_ids(self.tokenizer, result_payload)
                spliced = (
                    list(token_list[:call_end_index])
                    + list(result_ids)
                    + list(token_list[call_end_index:])
                )
                generated_tokens = torch.tensor(
                    [spliced], dtype=torch.long, device=model_device
                )

        return_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=skip_special_tokens
        )
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
