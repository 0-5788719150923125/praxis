"""Generator adapter that routes inference through MonoForwardTrainer.

Under Mono-Forward the model is sharded across Ray actors, so inference
hops activations through the live actor chain rather than calling
``model.generate()`` directly. This adapter exposes the same
``request_generation`` / ``get_result`` / ``fulfill_requests`` surface
as :class:`Generator`, so API routes and integrations work unchanged.

Ray serializes method calls per actor, so inference requests submitted
during training queue behind any in-flight ``train_batch`` and see a
consistent weight snapshot without explicit locking.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from queue import Queue
from typing import Any, Dict, List, Optional, Union

import torch

from praxis.generation.request import GenerationRequest
from praxis.tools import (
    call_tool,
    fix_truncated_tags,
    format_tool_output,
    get_tools_json_schema,
    get_unprocessed_tool_call,
)

_api_logger = logging.getLogger("praxis.web")


class MonoForwardGenerator:
    """Drop-in replacement for :class:`Generator` that routes through
    :meth:`MonoForwardTrainer.generate`."""

    MAX_RESULTS = 100

    def __init__(self, trainer: Any, tokenizer: Any) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.device = "cpu"

        self.request_queue: "Queue[GenerationRequest]" = Queue()
        self.results: Dict[str, str] = {}
        self._result_order: List[str] = []

        self.tools = get_tools_json_schema()
        self.call_tool = call_tool
        self.max_tool_calls_per_request = 3
        self.max_tool_call_time = 10.0

    # ------------------------------------------------------------------
    # public API (matches Generator)
    # ------------------------------------------------------------------

    def request_generation(
        self, prompt: Union[str, List[Dict[str, str]]], kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a generation request synchronously, return a request id."""
        kwargs = dict(kwargs or {})
        request_id = str(uuid.uuid4())
        request = GenerationRequest(id=request_id, prompt=prompt, kwargs=kwargs)
        self.request_queue.put(request)

        try:
            result = self._run_sync(request)
            result = self._handle_tool_calls(result, request)
        except Exception as exc:
            _api_logger.error(f"MonoForwardGenerator request {request_id} failed: {exc}")
            result = prompt if isinstance(prompt, str) else str(prompt)

        try:
            self.request_queue.get_nowait()
        except Exception:
            pass

        self._store_result(request_id, result)
        return request_id

    def get_result(self, request_id: str) -> Optional[str]:
        """Pop and return a stored result, or None if not ready."""
        result = self.results.get(request_id)
        if result is not None:
            del self.results[request_id]
            if request_id in self._result_order:
                self._result_order.remove(request_id)
        return result

    def fulfill_requests(self, max_requests: Optional[int] = None) -> int:
        """No-op: requests are fulfilled synchronously in request_generation."""
        del max_requests
        return 0

    def generate_with_messages(self, messages: list, **kwargs: Any) -> str:
        """Sync one-shot generation from a chat message list."""
        request_id = self.request_generation(messages, kwargs)
        result = self.get_result(request_id)
        return result or ""

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_sync(self, request: GenerationRequest) -> str:
        """Encode prompt, generate through actor chain, decode."""
        prompt_text = self._prompt_as_text(request.prompt)
        if not prompt_text:
            return ""

        input_ids = self._encode_prompt(prompt_text)
        if input_ids is None:
            return prompt_text

        kwargs = request.kwargs or {}
        max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        do_sample = bool(kwargs.get("do_sample", True))
        temperature = float(kwargs.get("temperature", 0.4))
        top_k = kwargs.get("top_k", 50)
        top_k_val = int(top_k) if top_k is not None else None
        skip_special_tokens = bool(kwargs.get("skip_special_tokens", False))

        truncate_to = kwargs.get("truncate_to")
        if truncate_to is not None and input_ids.shape[1] > int(truncate_to):
            input_ids = input_ids[:, -int(truncate_to) :]

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        new_tokens: List[torch.Tensor] = []
        for tok in self.trainer.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k_val,
        ):
            new_tokens.append(tok)

        if not new_tokens:
            return prompt_text

        new_ids = torch.stack(new_tokens, dim=-1)
        full_ids = torch.cat([input_ids, new_ids], dim=-1)
        try:
            return self.tokenizer.decode(
                full_ids[0].tolist(), skip_special_tokens=skip_special_tokens
            )
        except Exception as exc:  # pragma: no cover
            _api_logger.error(f"MonoForwardGenerator decode failed: {exc}")
            return prompt_text

    def _handle_tool_calls(self, text: str, request: GenerationRequest) -> str:
        """Scan for <tin>...</tin> tags, execute tools, inject <tout>, continue."""
        if not self.tools or not self.call_tool:
            return text

        text = fix_truncated_tags(text)
        start_time = time.time()
        tool_history: List[tuple] = []
        kwargs = request.kwargs or {}
        skip_special_tokens = bool(kwargs.get("skip_special_tokens", False))

        for depth in range(self.max_tool_calls_per_request):
            unprocessed = get_unprocessed_tool_call(text)
            if not unprocessed:
                break

            tool_call, tin_end_pos = unprocessed
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            if tool_name is None:
                tool_name = tool_call.get("tool") or tool_call.get(
                    "function", {}
                ).get("name")
                if tool_name is None:
                    _api_logger.warning(
                        f"Could not extract tool name from: {tool_call}"
                    )
                    break

            if time.time() - start_time > self.max_tool_call_time:
                _api_logger.info(f"[TOOL_SAFETY] Timeout after {depth} tool call(s).")
                break

            sig = (tool_name, json.dumps(tool_args, sort_keys=True))
            if sig in tool_history:
                _api_logger.info(f"[TOOL_SAFETY] Duplicate tool call: {tool_name}")
                break
            tool_history.append(sig)

            try:
                result = self.call_tool(tool_name, tool_args)
                _api_logger.info(f"Tool {tool_name}({tool_args}) -> {result}")
            except Exception as exc:
                _api_logger.error(f"Tool {tool_name} failed: {exc}")
                break

            tool_output_tag = format_tool_output(result)
            text_with_result = (
                text[:tin_end_pos] + tool_output_tag + text[tin_end_pos:]
            )

            input_ids = self._encode_prompt(text_with_result)
            if input_ids is None:
                return text_with_result

            max_new_tokens = int(kwargs.get("max_new_tokens", 128))
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

            new_tokens: List[torch.Tensor] = []
            for tok in self.trainer.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                do_sample=bool(kwargs.get("do_sample", True)),
                temperature=float(kwargs.get("temperature", 0.4)),
                top_k=int(kwargs["top_k"]) if kwargs.get("top_k") else None,
            ):
                new_tokens.append(tok)

            if not new_tokens:
                return text_with_result

            new_ids = torch.stack(new_tokens, dim=-1)
            full_ids = torch.cat([input_ids, new_ids], dim=-1)
            try:
                text = self.tokenizer.decode(
                    full_ids[0].tolist(),
                    skip_special_tokens=skip_special_tokens,
                )
            except Exception:
                return text_with_result

            text = fix_truncated_tags(text)

        return text

    def _prompt_as_text(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Normalise prompt to a plain string, applying chat template if needed."""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            try:
                return self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc:
                _api_logger.warning(f"apply_chat_template failed: {exc}")
                return "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt
                )
        return str(prompt)

    def _encode_prompt(self, prompt_text: str) -> Optional[torch.Tensor]:
        """Encode a string prompt to a [1, seq] long tensor, or None on failure."""
        try:
            ids = self.tokenizer.encode(prompt_text)
        except Exception as exc:  # pragma: no cover
            _api_logger.error(f"MonoForwardGenerator encode failed: {exc}")
            return None
        if not ids:
            return None
        return torch.as_tensor([ids], dtype=torch.long)

    def _store_result(self, request_id: str, result: str) -> None:
        """LRU bookkeeping for the results cache."""
        self.results[request_id] = result
        self._result_order.append(request_id)
        while len(self.results) > self.MAX_RESULTS:
            oldest = self._result_order.pop(0)
            self.results.pop(oldest, None)
