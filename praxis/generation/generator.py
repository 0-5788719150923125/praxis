"""Text generation with request queuing and inline tool calling support.

Tool calls are marked by atomic special tokens ``[TOOL_CALL]`` /
``[/TOOL_CALL]``. Both boundaries sit in the tokenizer's
``eos_token_id`` set so generation halts at each one:

- On ``[TOOL_CALL]`` open, we switch to deterministic decoding (greedy,
  no temperature/top-k/top-p) for the JSON body - any sampling noise in
  there breaks the downstream parse.
- On ``[/TOOL_CALL]`` close, we execute the tool, splice the real
  ``[TOOL_RESULT]`` block (with role transitions matching the chat
  template), restore the caller's sampling params, and resume.
"""

import contextlib
import logging
import time
import uuid
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import GenerationConfig

from praxis.generation.request import GenerationRequest
from praxis.tools import (
    STOP_TOOL_LOOP,
    build_result_splice_ids,
    execute_tool_call,
    find_unprocessed_tool_call_ids,
    tool_token_ids,
)

_log = logging.getLogger("praxis.generation")


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

        Generation halts on any of these. Both tool-call boundaries are
        included when the tokenizer knows about them: halting on
        ``[TOOL_CALL]`` open lets us switch to deterministic decoding for
        the JSON body, halting on ``[/TOOL_CALL]`` close lets us execute
        the tool before resuming.
        """
        ids = []
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.append(int(eos))
        sep = getattr(self.tokenizer, "sep_token_id", None)
        if sep is not None:
            ids.append(int(sep))
        tt = tool_token_ids(self.tokenizer)
        for key in ("call_open", "call_close"):
            tid = tt.get(key)
            if tid is not None and tid not in ids:
                ids.append(int(tid))
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

    def _prepare_inputs(
        self, request: GenerationRequest
    ) -> Tuple[torch.Tensor, Dict[str, Any], int, bool]:
        """Resolve a request into model-ready inputs and generation kwargs.

        Returns ``(input_ids, gen_kwargs, max_new_tokens, skip_special_tokens)``.
        ``gen_kwargs`` is the per-step kwargs dict (already merged with
        defaults and stripped of our own non-HF keys like ``truncate_to``).
        """
        if isinstance(request.prompt, list):
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    request.prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                _log.error(f"Failed to apply chat template: {e}")
                prompt_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in request.prompt]
                )
        else:
            prompt_text = request.prompt

        ids = self.tokenizer.encode(prompt_text)
        model_device = next(self.model.parameters()).device
        if isinstance(ids, list):
            input_ids = torch.tensor([ids], dtype=torch.long, device=model_device)
        else:
            input_ids = ids.to(model_device) if ids.device != model_device else ids

        gen_kwargs: Dict[str, Any] = {
            "do_sample": True,
            "renormalize_logits": True,
            "remove_invalid_values": True,
        }
        eos_list = self._eos_token_id_list()
        if eos_list:
            gen_kwargs["eos_token_id"] = eos_list
        # Caller overrides win, except for our own keys handled below.
        gen_kwargs.update(request.kwargs)

        gen_kwargs.pop("prompt", None)
        skip_special_tokens = not (gen_kwargs.pop("skip_special_tokens", True) is False)
        truncate_to = gen_kwargs.pop("truncate_to", None)
        if truncate_to is not None and input_ids.size(1) > truncate_to:
            input_ids = input_ids[:, -truncate_to:]

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 100))
        return input_ids, gen_kwargs, max_new_tokens, skip_special_tokens

    def _process_single_request(self, request: GenerationRequest) -> str:
        """Process a request, halting at tool-call boundaries to (a) switch
        into deterministic decoding for the JSON body and (b) execute the
        tool when it closes.

        The state machine has three halt-token cases:
          - ``[TOOL_CALL]`` open  -> enter deterministic mode and keep going
          - ``[/TOOL_CALL]`` close -> execute, splice the real result, exit
            deterministic mode and keep going
          - ``[EOS]`` / ``[SEP]`` / max_new_tokens hit -> done

        ``[TOOL_RESULT]`` blocks the model emits itself are ignored: a call
        is only marked "processed" after we splice a real result for it.
        """
        input_ids, gen_kwargs, max_new_tokens, skip_special_tokens = (
            self._prepare_inputs(request)
        )

        tt = tool_token_ids(self.tokenizer)
        call_open_id = tt.get("call_open")
        call_close_id = tt.get("call_close")
        tools_enabled = bool(self.tools and self.call_tool and call_close_id is not None)

        start_time = time.time()
        history: list = []
        tokens = input_ids
        initial_len = tokens.shape[1]
        in_tool_call = False
        tool_call_depth = 0

        with self._eval_mode():
            while True:
                remaining = max_new_tokens - (tokens.shape[1] - initial_len)
                if remaining <= 0:
                    break

                step_kwargs = dict(gen_kwargs)
                step_kwargs["max_new_tokens"] = remaining
                if in_tool_call:
                    # Tool-call JSON is a parsing target, not creative
                    # output: any sampling noise breaks the downstream
                    # decode. Greedy is the right policy here.
                    step_kwargs["do_sample"] = False
                    for key in ("temperature", "top_k", "top_p", "renormalize_logits"):
                        step_kwargs.pop(key, None)

                outputs = self.model.generate(
                    tokens,
                    generation_config=GenerationConfig(**step_kwargs),
                    tokenizer=self.tokenizer,
                    return_dict_in_generate=True,
                )
                if outputs.sequences.shape[1] <= tokens.shape[1]:
                    tokens = outputs.sequences
                    break
                tokens = outputs.sequences
                last_token = int(tokens[0, -1].item())

                if tools_enabled and last_token == call_open_id:
                    in_tool_call = True
                    continue

                if tools_enabled and last_token == call_close_id:
                    if tool_call_depth >= self.max_tool_calls_per_request:
                        _log.info(
                            f"[TOOL_SAFETY] Max tool-call depth "
                            f"({self.max_tool_calls_per_request}) reached."
                        )
                        break
                    if time.time() - start_time > self.max_tool_call_time:
                        _log.info(
                            f"[TOOL_SAFETY] Tool-call timeout "
                            f"({self.max_tool_call_time}s) exceeded."
                        )
                        break

                    token_list = tokens[0].tolist()
                    unprocessed = find_unprocessed_tool_call_ids(
                        token_list, self.tokenizer
                    )
                    if unprocessed is None:
                        in_tool_call = False
                        break

                    tool_call, call_end_index = unprocessed
                    payload = execute_tool_call(
                        tool_call, history, self.call_tool, log=_log.info
                    )
                    if payload is STOP_TOOL_LOOP:
                        break

                    result_ids = build_result_splice_ids(self.tokenizer, payload)
                    spliced = (
                        token_list[:call_end_index]
                        + list(result_ids)
                        + token_list[call_end_index:]
                    )
                    tokens = torch.tensor(
                        [spliced], dtype=torch.long, device=tokens.device
                    )
                    tool_call_depth += 1
                    in_tool_call = False
                    continue

                # EOS / SEP / max-tokens halt: done.
                break

        return self.tokenizer.decode(tokens[0], skip_special_tokens=skip_special_tokens)

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
