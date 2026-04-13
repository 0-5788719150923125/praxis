"""Drop-in Generator adapter for :class:`MonoForwardTrainer`.

The backprop Praxis flow routes API-driven inference through
:class:`praxis.generation.Generator`, which wraps a monolithic
``model.generate()`` call. Under Mono-Forward, the model is sharded
across Ray actors and the driver has no full model of its own - so
``model.generate()`` on the driver's CPU copy would run against
untrained weights. Instead, inference must hop activations through
the live actor chain via :meth:`MonoForwardTrainer.generate`, which
Phase 5 built on top of ``LayerActor.infer_batch`` / ``project_logits``.

This adapter exposes the exact same surface the Flask API routes
already use - ``request_generation`` / ``get_result`` /
``fulfill_requests`` - so the existing ``/messages`` and ``/input``
endpoints can drive MF inference without any route-level changes.
The one-line swap happens in ``main.py`` after the trainer is
constructed; see ``_maybe_swap_inference_generator`` there.

What this class deliberately does NOT support (and why):

- **Tool calling** (``<tin>`` / ``<tout>`` tags, recursion, etc.):
  the backprop Generator's tool-call loop is tightly coupled to HF
  ``GenerationMixin.generate`` return shapes and to synchronous
  in-process model calls. Under MF those assumptions break. Live
  MF inference is meant for "type a prompt, get a reply" dashboard
  UX today; tool calling is deferred until it's genuinely needed
  and the Ray actor chain has a natural way to handle mid-sequence
  tool injection.

- **HF ``GenerationConfig``**: we accept a few well-known kwargs
  (``max_new_tokens``, ``temperature``, ``top_k``, ``do_sample``)
  and forward them to :meth:`MonoForwardTrainer.generate`. The full
  HF config surface (beam search, contrastive decoding, custom
  stopping criteria) isn't plumbed through.

- **``_eval_mode`` context / ``self.model.training`` flipping**:
  we never touch actor training mode. Under Ray each
  ``train_batch.remote`` and ``infer_batch.remote`` call is
  serialized per actor, so an inference request can't collide with
  a training step mid-mutation. No eval/train toggle needed.

Concurrency: the Flask server runs requests on worker threads.
``request_generation`` runs the actual generate synchronously on
the calling thread and stores the result before returning, so
``get_result`` finds it on the next poll. Ray serializes actor
method calls, so an inference request submitted while
``fit()`` is mid-batch queues behind any in-flight ``train_batch``
and sees a consistent snapshot of the actor weights - that
guarantee is the reason Phase 5 was able to make live inference
safe without any explicit locking.
"""

from __future__ import annotations

import logging
import uuid
from queue import Queue
from typing import Any, Dict, List, Optional, Union

import torch

from praxis.generation.request import GenerationRequest

_api_logger = logging.getLogger("praxis.api")


class MonoForwardGenerator:
    """Drop-in replacement for :class:`Generator` that routes through
    :meth:`MonoForwardTrainer.generate`.

    Holds no model, no tokenizer device. Everything it needs is the
    trainer (which exposes ``generate(...)``) and the tokenizer (which
    exposes ``encode`` / ``decode`` / ``apply_chat_template``).

    The Flask routes in ``praxis/api/routes/generation.py`` only call
    ``request_generation`` and ``get_result``, in a simple
    submit-then-poll loop. This class implements both methods with
    synchronous-under-the-hood semantics: the submit call actually
    does the work, the poll call just hands back the cached result.
    """

    # Match the backprop Generator's cap so late GC stays consistent.
    MAX_RESULTS = 100

    def __init__(self, trainer: Any, tokenizer: Any) -> None:
        """
        Args:
            trainer: A :class:`MonoForwardTrainer` instance whose
                ``fit`` method has either started running (the actor
                set is live) or is about to. Requests submitted before
                the actor set exists fail with an explicit error
                message rather than silently deadlocking.
            tokenizer: The Praxis tokenizer shared with the rest of
                main.py. Must expose ``encode``, ``decode``, and -
                for chat-style prompts - ``apply_chat_template``.
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        # Kept for compat with backprop callers that inspect ``.device``.
        # Under MF the driver holds no model parameters; CPU is correct.
        self.device = "cpu"

        # Match the backprop Generator's state containers exactly so
        # any integration (Discord, ngrok, etc.) that reads ``results``
        # or ``request_queue`` keeps working.
        self.request_queue: "Queue[GenerationRequest]" = Queue()
        self.results: Dict[str, str] = {}
        self._result_order: List[str] = []

        # Tools surface: the backprop Generator exposes a ``tools``
        # list + ``call_tool`` that integrations reference. Under MF
        # tool calling is unsupported (see module docstring), but we
        # still publish the fields so a `hasattr(generator, "tools")`
        # check doesn't raise on integrations that peek.
        self.tools: list = []
        self.call_tool = None

    # ------------------------------------------------------------------
    # public API (matches Generator)
    # ------------------------------------------------------------------

    def request_generation(
        self, prompt: Union[str, List[Dict[str, str]]], kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a generation request and store the decoded result.

        Unlike the backprop Generator (which enqueues and defers work
        until ``fulfill_requests`` is called on a background thread),
        this adapter runs the generation synchronously on the calling
        thread. The Flask worker thread that invoked us will therefore
        block inside :meth:`MonoForwardTrainer.generate` until Ray's
        per-actor queue lets our ``infer_batch`` calls through. That's
        fine: Phase 5 proved concurrent train+infer is safe, and the
        caller's ``while generator.get_result(id) is None: sleep(0.1)``
        loop will find the answer on its first iteration.

        Args:
            prompt: Either a plain string prompt or a list of chat
                messages in ``{"role": ..., "content": ...}`` form.
                Chat lists are run through
                ``tokenizer.apply_chat_template`` the same way the
                backprop ``generate_from_messages`` helper does.
            kwargs: Optional generation parameters. Recognised keys:
                ``max_new_tokens`` (default 128), ``temperature``
                (default 0.4, matches the backprop default for the
                dashboard), ``top_k`` (default 50), ``do_sample``
                (default True), ``skip_special_tokens`` (default
                False - we return the full decoded text the same
                way the backprop Generator does so that
                :func:`extract_assistant_reply` can still find the
                ``<s>assistant`` marker). Unknown keys are ignored
                rather than raising, so callers built against the
                backprop signature don't have to be updated.

        Returns:
            A UUID request id. The caller should poll
            :meth:`get_result` with this id until it returns a non-
            None string.
        """
        kwargs = dict(kwargs or {})
        request_id = str(uuid.uuid4())

        # Create a ``GenerationRequest`` for the tiny minority of
        # integrations (e.g. Discord) that peek at ``request_queue``
        # contents. We push it AND immediately process it - the queue
        # never really fills because ``_run_sync`` runs inline below.
        request = GenerationRequest(id=request_id, prompt=prompt, kwargs=kwargs)
        self.request_queue.put(request)

        try:
            result = self._run_sync(request)
        except Exception as exc:
            _api_logger.error(f"MonoForwardGenerator request {request_id} failed: {exc}")
            # Mirror the backprop Generator's "on error, return the
            # original prompt text" behaviour so the API route has
            # something non-None to return to the client.
            result = prompt if isinstance(prompt, str) else str(prompt)

        # Drain the queue entry we just put in (we processed it inline).
        try:
            self.request_queue.get_nowait()
        except Exception:
            pass

        self._store_result(request_id, result)
        return request_id

    def get_result(self, request_id: str) -> Optional[str]:
        """Return a stored result and pop it from the cache.

        Matches :meth:`Generator.get_result` exactly: non-destructive
        peek is not supported - the first caller to ask for a given
        id consumes the result, and subsequent lookups return ``None``.
        """
        result = self.results.get(request_id)
        if result is not None:
            del self.results[request_id]
            if request_id in self._result_order:
                self._result_order.remove(request_id)
        return result

    def fulfill_requests(self, max_requests: Optional[int] = None) -> int:
        """No-op fast path: requests are already fulfilled synchronously.

        The backprop Generator defers work to a background-thread
        ``fulfill_requests`` call. This adapter runs everything inline
        in ``request_generation``, so by the time anyone calls
        ``fulfill_requests`` the queue is already empty. We still
        expose the method for interface parity (the backprop
        TerminalInterface polls it in a tight loop, but that callback
        never runs under MF).
        """
        del max_requests
        return 0

    def generate_with_messages(self, messages: list, **kwargs: Any) -> str:
        """Convenience wrapper used by integrations that want a
        sync one-shot generation without the submit/poll dance."""
        request_id = self.request_generation(messages, kwargs)
        result = self.get_result(request_id)
        return result or ""

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_sync(self, request: GenerationRequest) -> str:
        """Core work: prompt → ids → trainer.generate → decoded string."""
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

        # Optional left-truncate for callers that pin context length.
        truncate_to = kwargs.get("truncate_to")
        if truncate_to is not None and input_ids.shape[1] > int(truncate_to):
            input_ids = input_ids[:, -int(truncate_to) :]

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Collect tokens from the trainer's generator. Each yield is a
        # ``[batch]`` 1-D tensor, one per decoded step; we stack them
        # back into a ``[batch, new_tokens]`` matrix and concatenate
        # with the prompt so the final decoded text contains both
        # the seed and the continuation - the shape the backprop
        # Generator's API contract produces.
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
        except Exception as exc:  # pragma: no cover - defensive
            _api_logger.error(f"MonoForwardGenerator decode failed: {exc}")
            return prompt_text

    def _prompt_as_text(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Normalise the API-accepted prompt shape to a single string.

        Chat message lists are folded via ``apply_chat_template`` (same
        behaviour the backprop path uses in
        :func:`praxis.api.utils.formatters.generate_from_messages`); a
        failing template call falls back to a simple ``role: content``
        join so the caller always gets *something*.
        """
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            try:
                return self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc:
                _api_logger.warning(
                    f"apply_chat_template failed, using role: content fallback: {exc}"
                )
                return "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt
                )
        return str(prompt)

    def _encode_prompt(self, prompt_text: str) -> Optional[torch.Tensor]:
        """Encode a plain-string prompt to a ``[1, seq]`` long tensor.

        Returns ``None`` only on a flat encode failure so the caller
        can fall back to echoing the prompt rather than crashing the
        Flask request thread.
        """
        try:
            ids = self.tokenizer.encode(prompt_text)
        except Exception as exc:  # pragma: no cover - defensive
            _api_logger.error(f"MonoForwardGenerator encode failed: {exc}")
            return None
        if not ids:
            return None
        return torch.as_tensor([ids], dtype=torch.long)

    def _store_result(self, request_id: str, result: str) -> None:
        """LRU-ish bookkeeping matching the backprop Generator."""
        self.results[request_id] = result
        self._result_order.append(request_id)
        while len(self.results) > self.MAX_RESULTS:
            oldest = self._result_order.pop(0)
            self.results.pop(oldest, None)
