"""Text generation with request queuing and inline tool calling support.

The active chat format (``CHAT_FORMAT_REGISTRY``) decides what a tool-call
boundary looks like; this loop's state machine is the same either way. Under
``tool_style="tokens"`` the boundaries are the atomic ``[TOOL_CALL]`` /
``[/TOOL_CALL]`` ids; under ``tool_style="roles"`` they are the role-name
boundaries opening the ``call`` and ``tool`` turns. Both sit in the step's halt
set - ``eos_token_id`` for ids, ``stop_strings`` for text - so generation stops
at each one:

- On the CALL boundary, we switch to deterministic decoding (greedy, no
  temperature/top-k/top-p) for the JSON body - any sampling noise in there
  breaks the downstream parse.
- On the RESULT boundary, we execute the tool, splice the real result (with
  role transitions matching the chat template), restore the caller's sampling
  params, and resume.
"""

import logging
import time
import uuid
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import torch

from praxis.generation.decode_backend import ModelBackend
from praxis.generation.request import GenerationRequest, GenerationResult
from praxis.tokenizers.chat_templates import chat_format_of
from praxis.tools import (
    STOP_TOOL_LOOP,
    build_result_splice_ids,
    classify_boundary_halt,
    execute_tool_call,
    find_pending_call_text,
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

    def __init__(
        self, model=None, tokenizer=None, device="cuda", backend=None, synchronous=False
    ):
        # One Generator, two ways to decode (a DecodeBackend) and two ways to
        # drive the queue. ``synchronous`` runs each request in-place in
        # request_generation (fulfill_requests is a no-op) for callers with no
        # training loop to drain the queue - the Mono-Forward web path. The
        # default queues and does the work in fulfill_requests.
        if backend is None:
            backend = ModelBackend(model, tokenizer)
        self.backend = backend
        self.model = getattr(backend, "model", None)
        self.tokenizer = tokenizer
        self.device = device
        self.synchronous = synchronous
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

    @property
    def chat_format(self):
        """The active chat format. Owns the turn boundaries, the halting
        contract and the tool-call layout as one unit."""
        return chat_format_of(self.tokenizer)

    def _eos_token_id_list(self) -> Optional[list]:
        """Build the eos_token_id list for a chat-style generation.

        Generation halts on any of these. Turn terminators come from the
        format's ``stop_token_names``; the tool-call boundaries are added when
        the format marks them with control tokens and the tokenizer knows the
        ids - halting on ``[TOOL_CALL]`` open lets us switch to deterministic
        decoding for the JSON body, halting on ``[/TOOL_CALL]`` close lets us
        execute the tool before resuming.

        Text-boundary formats declare no halt ids at all and return None here;
        ``_stop_strings`` carries their contract instead.
        """
        fmt = self.chat_format
        ids = fmt.stop_token_ids(self.tokenizer)
        if fmt.tool_style == "tokens":
            tt = tool_token_ids(self.tokenizer)
            for key in ("call_open", "call_close"):
                tid = tt.get(key)
                if tid is not None and tid not in ids:
                    ids.append(int(tid))
        return ids or None

    def _stop_strings(self) -> Optional[list]:
        """Text boundaries that halt a step, for formats that use them.

        These already include the tool-flow boundaries: the format lists the
        ``call`` and ``result`` roles among its ``stop_roles`` precisely so this
        loop gets control at each one.
        """
        stops = self.chat_format.stop_strings()
        return list(stops) or None

    def _halt_text(self, tokens: torch.Tensor, fmt) -> str:
        """Decode just enough of the tail to recognise a text boundary.

        Every token decodes to at least one character, so a window as long as
        the longest boundary always contains it; the slack covers a
        multi-character token straddling its start.
        """
        longest = max((len(fmt.boundary(r)) for r in fmt.roles), default=0)
        window = longest + 4
        tail = tokens[0, -window:].tolist()
        return self.tokenizer.decode(tail, skip_special_tokens=False)

    def _classify_halt(self, tokens: torch.Tensor, fmt) -> Optional[str]:
        """Name the boundary generation just halted on.

        Returns ``"call_open"``, ``"call_close"``, or None for a plain turn
        terminator (which the caller treats as done). Token-boundary formats
        read the last id; text-boundary formats read the decoded tail.
        """
        if fmt.tool_style == "roles":
            return classify_boundary_halt(self._halt_text(tokens, fmt), fmt)
        tt = tool_token_ids(self.tokenizer)
        last_token = int(tokens[0, -1].item())
        if tt.get("call_open") is not None and last_token == tt["call_open"]:
            return "call_open"
        if tt.get("call_close") is not None and last_token == tt["call_close"]:
            return "call_close"
        return None

    def _pending_call(self, tokens: torch.Tensor, fmt) -> Optional[Tuple[Dict, int]]:
        """The call awaiting execution, plus the index to splice its result at.

        Text-boundary formats halt with the result boundary at the very end of
        the sequence, so the splice point is the end; token-boundary formats
        scan for the first unanswered ``[TOOL_CALL]`` and splice just past its
        close, which may sit mid-sequence.
        """
        token_list = tokens[0].tolist()
        if fmt.tool_style == "roles":
            text = self.tokenizer.decode(token_list, skip_special_tokens=False)
            call = find_pending_call_text(text, fmt)
            return None if call is None else (call, len(token_list))
        return find_unprocessed_tool_call_ids(token_list, self.tokenizer)

    def _max_positions(self) -> Optional[int]:
        """The model's positional capacity (max_position_embeddings).

        Learned absolute positions only exist up to this many tokens, so the
        generation context (prompt + generated + any spliced tool results) must
        never exceed it - otherwise the position lookup overflows and crashes.
        """
        return self.backend.max_positions

    @property
    def draft_window(self) -> int:
        """Tokens worth requesting per step to actually exercise multi-token
        drafting when byte-latent MTP is live, else 1. A caller that asks for
        fewer spends its whole budget on the main forward's token and never
        uses the MTP drafts.

        This is the *achievable* window (``mtp.draft_width + 1``), not the
        trained depth. Asking for ``mtp_depth + 1`` bytes when acceptance only
        delivers a couple per commit does not make the step wider - it makes
        the step loop, paying another main forward and another verify batch per
        few bytes, which is how a large ``mtp_depth`` turns into a slow step."""
        model = self.model
        mtp = getattr(model, "mtp", None)
        if mtp is not None and getattr(mtp, "byte_level", False):
            width = getattr(mtp, "draft_width", None)
            if width is None:
                depth = getattr(getattr(model, "config", None), "mtp_depth", 1) or 1
                width = int(depth)
            return int(width) + 1
        return 1

    def request_generation(self, prompt, kwargs={}, deadline=None) -> str:
        """
        Submit a generation request and return a request ID.

        Args:
            prompt: Either a string prompt or a list of message dicts with 'role' and 'content'
            kwargs: Additional generation parameters
            deadline: Wall-clock time past which this request is not worth
                serving. Pass one whenever the caller intends to stop waiting:
                the queued path runs inside the training loop, so an abandoned
                request without a deadline stalls the run to completion for
                nobody. See ``GenerationRequest``.

        Returns:
            Request ID string
        """
        request_id = str(uuid.uuid4())
        request = GenerationRequest(
            id=request_id, prompt=prompt, kwargs=kwargs, deadline=deadline
        )
        # Synchronous backends (no training loop to drain the queue) run now;
        # the queued path defers the work to fulfill_requests.
        if self.synchronous:
            try:
                self.results[request_id] = self._process_single_request(request)
            except Exception as exc:
                _log.error(f"Generation request {request_id} failed: {exc}")
                self.results[request_id] = (
                    prompt if isinstance(prompt, str) else str(prompt)
                )
            self._result_order.append(request_id)
            self._evict_old_results()
        else:
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
        model_device = self.backend.device
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
        stop_strings = self._stop_strings()
        if stop_strings:
            gen_kwargs["stop_strings"] = stop_strings
        # Control ids the format's data never makes a target must not be
        # sampleable either - see ChatFormat.suppressed_token_ids.
        suppress = self.chat_format.suppressed_token_ids(self.tokenizer)
        if suppress:
            gen_kwargs["suppress_tokens"] = suppress
        # Caller overrides win, except for our own keys handled below.
        gen_kwargs.update(request.kwargs)

        # When the caller omits temperature, let the model pick its own
        # default. CALM needs T=0.5 (count-based sampling is near-random at
        # T=1); token models are happy at the transformers default.
        if "temperature" not in gen_kwargs:
            default_temp = self.backend.default_sampling_temperature
            if default_temp is not None:
                gen_kwargs["temperature"] = float(default_temp)

        gen_kwargs.pop("prompt", None)
        skip_special_tokens = not (gen_kwargs.pop("skip_special_tokens", True) is False)
        truncate_to = gen_kwargs.pop("truncate_to", None)
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 100))

        # The prompt plus what we generate must fit the model's positional
        # capacity (learned absolute positions can't extend past it). Cap the
        # prompt to leave room for new tokens - tighter than any caller's
        # truncate_to - so inference can never overflow and crash.
        mpe = self._max_positions()
        if mpe is not None:
            budget = max(1, mpe - max_new_tokens)
            truncate_to = budget if truncate_to is None else min(truncate_to, budget)
        if truncate_to is not None and input_ids.size(1) > truncate_to:
            start = input_ids.size(1) - truncate_to
            # A token index is not always a character boundary. On the
            # byte-level tokenizer a token IS a byte, so cutting here splits
            # multi-byte characters, and the severed continuation bytes come
            # back from decode as U+FFFD - which a rolling context then
            # re-encodes into three real bytes and carries forever. Tokenizers
            # that know their own layout move the cut to the next boundary;
            # the rest (one token = whole characters) have nothing to fix.
            align = getattr(self.tokenizer, "align_left_cut", None)
            if align is not None:
                start = align(input_ids[0].tolist(), start)
            input_ids = input_ids[:, start:]

        return input_ids, gen_kwargs, max_new_tokens, skip_special_tokens

    def _process_single_request(self, request: GenerationRequest) -> str:
        """Process a request, halting at tool-call boundaries to (a) switch
        into deterministic decoding for the JSON body and (b) execute the
        tool when it closes.

        The state machine has three halt cases, named by the format rather
        than by a literal token:
          - CALL boundary   -> enter deterministic mode and keep going
          - RESULT boundary -> execute, splice the real result, exit
            deterministic mode and keep going
          - any other turn terminator / max_new_tokens hit -> done

        Results the model emits itself are ignored: a call is only marked
        "processed" after we splice a real result for it.
        """
        input_ids, gen_kwargs, max_new_tokens, skip_special_tokens = (
            self._prepare_inputs(request)
        )

        fmt = self.chat_format
        if fmt.tool_style == "roles":
            # The loop can only intercept a call if the format actually halts on
            # both boundaries; a format that named them but left them out of
            # stop_roles would run the JSON straight into the reply.
            boundaries_detectable = (
                fmt.call_role in fmt.stop_roles and fmt.result_role in fmt.stop_roles
            )
        else:
            boundaries_detectable = (
                tool_token_ids(self.tokenizer).get("call_close") is not None
            )
        tools_enabled = bool(self.tools and self.call_tool and boundaries_detectable)

        start_time = time.time()
        history: list = []
        tokens = input_ids
        initial_len = tokens.shape[1]
        in_tool_call = False
        tool_call_depth = 0
        # Tokens the MODEL produced, counted directly rather than derived from
        # the sequence length. A spliced tool result grows the sequence without
        # the model having written any of it, so the derived form charged the
        # splice to the caller's budget: one ~800-byte `get_tools` result
        # against the web default of 256 drove `remaining` negative and broke
        # the loop before the model ever spoke.
        produced = 0
        # Where the model's current turn starts, in tokens. It begins after the
        # prompt and moves past each result we splice, so it always names text
        # the runtime wrote rather than a boundary the model may have written
        # itself. See GenerationResult.reply_start.
        reply_start_tokens = initial_len

        with self.backend.eval_mode():
            while True:
                # The caller's patience, enforced where it costs something.
                # The real bound is INSIDE the decode: the deadline goes to
                # generate_until_halt below and reaches both the transformers
                # loop and our speculative loop as a StoppingCriteria. A plain
                # turn with no tool call is ONE call to it, so this check on its
                # own would fire only before the first token; what it bounds is
                # the TOOL loop, where a request round-trips several decodes and
                # a tool execution. Whatever the model produced so far is
                # returned as normal: `reply_start` is the runtime's own offset
                # and does not depend on halting cleanly.
                if request.deadline is not None and time.time() >= request.deadline:
                    _log.info(
                        f"Generation request {request.id} hit its deadline after "
                        f"{produced} tokens; returning the partial turn."
                    )
                    break
                remaining = max_new_tokens - produced
                # Counting model output alone stops a splice from spending the
                # caller's budget, but it also drops the bound the derived form
                # gave for free: prompt + everything since has to stay inside
                # the positional capacity that `_prepare_inputs` sized the
                # prompt against. A splice cut back to `mpe` leaves `produced`
                # small and the context already full, so clamp here too.
                mpe = self._max_positions()
                if mpe is not None:
                    remaining = min(remaining, mpe - tokens.shape[1])
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

                extended = self.backend.generate_until_halt(
                    tokens, step_kwargs, deadline=request.deadline
                )
                if extended.shape[1] <= tokens.shape[1]:
                    tokens = extended
                    break
                produced += extended.shape[1] - tokens.shape[1]
                tokens = extended
                halt = self._classify_halt(tokens, fmt)

                if tools_enabled and halt == "call_open":
                    in_tool_call = True
                    continue

                if tools_enabled and halt == "call_close":
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
                    unprocessed = self._pending_call(tokens, fmt)
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
                    # Splicing grows the sequence; keep the most recent context
                    # within the model's positional capacity so the next forward
                    # can't overflow learned positions.
                    mpe = self._max_positions()
                    if mpe is not None and len(spliced) > mpe:
                        spliced = spliced[-mpe:]
                    tokens = torch.tensor(
                        [spliced], dtype=torch.long, device=tokens.device
                    )
                    # Both splice styles end by handing control back to the
                    # reply role, so the end of the splice opens a new turn and
                    # the anchor belongs there - but only when the splice went
                    # in at the END. `find_unprocessed_tool_call_ids` scans from
                    # the front, so a stale unanswered call in the PROMPT puts
                    # the insertion mid-sequence; taking the anchor to the end
                    # of that splice would drag it backwards over text the
                    # model had already written. Shift for the insertion, then
                    # advance only if the splice really is further along.
                    dropped = max(0, len(token_list) + len(result_ids) - len(spliced))
                    if call_end_index <= reply_start_tokens:
                        reply_start_tokens += len(result_ids)
                    reply_start_tokens = max(
                        reply_start_tokens, call_end_index + len(result_ids)
                    )
                    # `spliced[-mpe:]` above drops tokens off the front.
                    reply_start_tokens = max(0, reply_start_tokens - dropped)
                    tool_call_depth += 1
                    in_tool_call = False
                    continue

                # Turn terminator / max-tokens halt: done.
                break

            # Some vocabs (e.g. tokenmonster) can halt on a partial token -
            # a split multi-byte glyph or pending capcode modifier - which a
            # decode-then-reencode round trip (rolling contexts) silently
            # drops. Ask for a few more tokens until the tail is complete.
            # Deliberately NOT deadline-bounded: four tokens is a bounded
            # overshoot, and halting here would trade it for a corrupted tail.
            incomplete_tail = getattr(self.tokenizer, "incomplete_tail", None)
            if incomplete_tail is not None:
                for _ in range(4):
                    if not incomplete_tail(tokens[0].tolist()):
                        break
                    step_kwargs = dict(gen_kwargs)
                    step_kwargs["max_new_tokens"] = 1
                    extended = self.backend.generate_until_halt(tokens, step_kwargs)
                    if extended.shape[1] <= tokens.shape[1]:
                        break
                    tokens = extended
                strip = getattr(self.tokenizer, "strip_incomplete_tail", None)
                if strip is not None and incomplete_tail(tokens[0].tolist()):
                    tokens = tokens[:, : len(strip(tokens[0].tolist()))]

        ids = tokens[0]
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return GenerationResult(
            text,
            self._reply_start_char(ids, reply_start_tokens, text, skip_special_tokens),
        )

    def _reply_start_char(
        self,
        ids: torch.Tensor,
        reply_start_tokens: int,
        text: str,
        skip_special_tokens: bool,
    ) -> Optional[int]:
        """Character offset in ``text`` where the model's turn starts.

        Decoding the prefix and taking its length assumes piece-wise decoding
        concatenates, which holds for byte- and char-level vocabs but not for
        every BPE. Rather than trust it, check that the prefix really is one:
        ``None`` tells the reader to fall back to scanning for a boundary,
        which is what it did before this offset existed.
        """
        try:
            prefix = self.tokenizer.decode(
                ids[:reply_start_tokens], skip_special_tokens=skip_special_tokens
            )
        except Exception:
            return None
        return len(prefix) if text.startswith(prefix) else None

    def fulfill_requests(self, max_requests: int = None) -> int:
        """
        Process pending generation requests. Should be called from inside the training loop.
        Returns the number of requests processed.
        """
        if self.synchronous:
            return 0  # synchronous backends already ran the work in request_generation

        processed = 0
        while not self.request_queue.empty():
            if max_requests is not None and processed >= max_requests:
                break

            request = self.request_queue.get()
            if request.deadline is not None and time.time() >= request.deadline:
                # Nobody is listening any more, and this runs in the training
                # loop, so serving it would stall the run for a reply that gets
                # thrown away. Store an empty result rather than dropping the
                # id: a late poller then gets a falsy answer instead of waiting
                # out its own timeout on a request that will never run.
                _log.info(
                    f"Generation request {request.id} expired before it was served"
                )
                self.results[request.id] = GenerationResult("")
                self._result_order.append(request.id)
                self._evict_old_results()
                # Deliberately not counted against `max_requests`: that budget
                # exists to bound how much GENERATION runs per step, and a drop
                # runs none. Counting it would let a burst of stale requests
                # take a step each to clear.
                continue
            try:
                result = self._process_single_request(request)
            except Exception as exc:
                # This runs inside the training loop, so a request that blows
                # up must not take the run down with it. Store the prompt back
                # (what the synchronous path already does) so the waiting
                # client gets a response instead of hanging to its timeout.
                _log.error(f"Generation request {request.id} failed: {exc}")
                result = GenerationResult(
                    request.prompt
                    if isinstance(request.prompt, str)
                    else str(request.prompt)
                )
            self.results[request.id] = result
            self._result_order.append(request.id)
            self._evict_old_results()

            processed += 1

        return processed

    def _evict_old_results(self) -> None:
        """Cap the results cache to prevent unbounded memory growth."""
        while len(self.results) > self.MAX_RESULTS:
            oldest_id = self._result_order.pop(0)
            self.results.pop(oldest_id, None)
