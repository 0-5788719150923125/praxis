"""A generation backend with its output pinned, and a sink for what it streams.

``ScriptedBackend`` drives the real ``Generator`` - prompt construction, the halt
contract, the tool state machine, reply extraction - with the model's output fixed
instead of sampled.
"""

import contextlib

import torch
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)


class ScriptedBackend:
    """Emits a fixed script one token per step, halting as the real decode does.

    The script is consumed across calls, so a turn that halts at a tool boundary
    resumes where it stopped. The criteria are the ones transformers would build
    for the step's kwargs, so this halts on exactly what the real backend halts on.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, tokenizer, script):
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.max_positions = None
        self.pending = list(tokenizer.encode(script, add_special_tokens=False))

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    # StopStringCriteria precomputes tables over the vocab; share them.
    _stop_criteria = {}

    def _criteria(self, step_kwargs):
        criteria = StoppingCriteriaList()
        stops = step_kwargs.get("stop_strings")
        if stops:
            key = (id(self.tokenizer), tuple(stops))
            if key not in self._stop_criteria:
                self._stop_criteria[key] = StopStringCriteria(
                    stop_strings=list(stops), tokenizer=self.tokenizer
                )
            criteria.append(self._stop_criteria[key])
        eos = step_kwargs.get("eos_token_id")
        if eos:
            eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            criteria.append(EosTokenCriteria(eos_token_id=torch.tensor(eos)))
        return criteria

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        criteria = self._criteria(step_kwargs)
        budget = int(step_kwargs.get("max_new_tokens", 100))
        # transformers publishes the step's prompt first, then each new token.
        if streamer is not None:
            streamer.put(tokens)
        ids = tokens[0].tolist()
        produced = 0
        while self.pending and produced < budget:
            nxt = self.pending.pop(0)
            ids.append(nxt)
            produced += 1
            if streamer is not None:
                streamer.put(torch.tensor([nxt]))
            # Checked after every token, as transformers' own loop does.
            if criteria and criteria(torch.tensor([ids]), None).any():
                break
        if streamer is not None:
            streamer.end()
        return torch.tensor([ids], dtype=torch.long)


class Sink:
    """Collects ``on_text`` deltas; ``reset`` drops them, as a client would."""

    def __init__(self):
        self.chunks = []
        self.resets = 0

    def text(self, delta):
        self.chunks.append(delta)

    def reset(self):
        self.resets += 1
        self.chunks.clear()

    @property
    def joined(self):
        return "".join(self.chunks)
