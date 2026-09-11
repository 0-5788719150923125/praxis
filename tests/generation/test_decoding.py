"""The shared pieces every Praxis decoding method is built from.

Praxis registers its non-``_sample`` loops (MTP speculative decoding, CALM's patch
vote) as transformers decoding methods, so transformers hands them a prepared
``LogitsProcessorList`` and ``StoppingCriteriaList``. What such a loop still does for
itself is pick the next token and find where a multi-token commit halts; these pin
the properties that make ``praxis.generation.decoding`` safe to share.
"""

from __future__ import annotations

import time

import pytest
import torch
from transformers import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

from praxis.generation.decoding import (
    first_halt,
    is_halted,
    pick_next,
    split_positional_criteria,
    stream_end,
    stream_put,
)
from praxis.tokenizers.chat_templates import chat_format_of


# ---------------------------------------------------------------------------
# pick_next
# ---------------------------------------------------------------------------


def test_greedy_pick_is_the_argmax_of_the_processed_scores():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    ids = torch.tensor([[7]])
    assert int(pick_next(logits, ids, do_sample=False).item()) == 1


def test_processors_run_against_the_prefix_the_token_follows():
    """The repetition penalty is context-dependent, so a shared sampler that
    scored against the wrong prefix would penalize the wrong tokens."""
    from transformers import RepetitionPenaltyLogitsProcessor

    procs = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(penalty=100.0)])
    logits = torch.tensor([[1.0, 0.9, 0.0]])
    # Token 0 is in the prefix, so the penalty should knock it below token 1.
    picked = pick_next(logits, torch.tensor([[0]]), procs, do_sample=False)
    assert int(picked.item()) == 1
    # With a prefix that does NOT contain token 0, it wins on its own merit.
    picked = pick_next(logits, torch.tensor([[2]]), procs, do_sample=False)
    assert int(picked.item()) == 0
    # The processors score a copy; the caller's logits are untouched.
    assert torch.equal(logits, torch.tensor([[1.0, 0.9, 0.0]]))


def test_temperature_is_applied_exactly_once():
    """Transformers puts a TemperatureLogitsWarper in the prepared list, so
    pick_next must never divide by temperature itself - the hand-rolled loops
    this replaces did BOTH, which squares the effect once the list arrives
    prepared. Exact draw-for-draw equivalence against a single application is
    the only check that catches extra arithmetic anywhere in the body.
    """
    from transformers import TemperatureLogitsWarper

    procs = LogitsProcessorList([TemperatureLogitsWarper(2.0)])
    logits = torch.tensor([[0.0, 1.4, 2.9]])
    ids = torch.tensor([[0]])

    torch.manual_seed(1234)
    ours = [
        int(pick_next(logits, ids, procs, do_sample=True).item()) for _ in range(64)
    ]

    torch.manual_seed(1234)
    scores = procs(ids, logits.to(dtype=torch.float32))
    probs = torch.nn.functional.softmax(scores, dim=-1)
    expected = [
        int(torch.multinomial(probs, num_samples=1).squeeze(1).item())
        for _ in range(64)
    ]
    assert ours == expected


def test_pick_next_casts_to_float32():
    """_sample casts before the processors; a shared sampler that did not would
    quietly disagree with the standard path on a bf16 model."""
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.bfloat16)
    seen = {}

    class _Record(LogitsProcessorList):
        def __call__(self, ids, scores):
            seen["dtype"] = scores.dtype
            return scores

    pick_next(logits, torch.tensor([[0]]), _Record(), do_sample=False)
    assert seen["dtype"] is torch.float32


# ---------------------------------------------------------------------------
# criteria partitioning
# ---------------------------------------------------------------------------


def test_positional_partition(prose_tokenizer):
    criteria = StoppingCriteriaList(
        [
            StopStringCriteria(stop_strings=["\n\nuser\n\n"], tokenizer=prose_tokenizer),
            EosTokenCriteria(eos_token_id=torch.tensor([0])),
            MaxLengthCriteria(max_length=99),
            MaxTimeCriteria(max_time=1.0),
        ]
    )
    positional, whole = split_positional_criteria(criteria)
    assert {type(c).__name__ for c in positional} == {
        "StopStringCriteria",
        "EosTokenCriteria",
        "MaxLengthCriteria",
    }
    assert [type(c).__name__ for c in whole] == ["MaxTimeCriteria"]


def test_an_unknown_criterion_is_treated_as_non_positional():
    """We cannot know that evaluating someone else's criterion at a prefix
    means anything, so it halts the loop but never truncates it."""

    class _Custom(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.ones(input_ids.shape[0], dtype=torch.bool)

    positional, whole = split_positional_criteria(StoppingCriteriaList([_Custom()]))
    assert len(positional) == 0 and len(whole) == 1


# ---------------------------------------------------------------------------
# first_halt
# ---------------------------------------------------------------------------


def _stop_criteria(tokenizer):
    """The criteria transformers builds for this tokenizer's format."""
    stops = list(chat_format_of(tokenizer).stop_strings())
    return StoppingCriteriaList(
        [StopStringCriteria(stop_strings=stops, tokenizer=tokenizer)]
    )


def test_first_halt_cuts_a_multi_token_commit_at_the_boundary(prose_tokenizer):
    """The whole reason this exists: a speculative run commits several tokens
    at once, so the boundary can complete mid-commit and the criteria only
    report that it completed somewhere."""
    criteria = _stop_criteria(prose_tokenizer)
    text = "hello there\n\nuser\n\nXXXX"
    ids = torch.tensor([prose_tokenizer.encode(text)])
    keep = first_halt(ids, criteria, start_index=0)
    assert keep is not None
    assert prose_tokenizer.decode(ids[0, :keep].tolist()) == "hello there\n\nuser\n\n"


def test_first_halt_returns_the_earliest_completion(prose_tokenizer):
    """Anything drafted past the boundary belongs to a turn the model does not
    get to write, so the FIRST boundary wins, not the last."""
    criteria = _stop_criteria(prose_tokenizer)
    text = "a\n\nuser\n\nb\n\nuser\n\n"
    ids = torch.tensor([prose_tokenizer.encode(text)])
    keep = first_halt(ids, criteria, start_index=0)
    assert prose_tokenizer.decode(ids[0, :keep].tolist()) == "a\n\nuser\n\n"


@pytest.mark.parametrize(
    "resumed_from,continuation",
    [
        ("a\n\nuser\n\n", "still going"),
        # The tool loop's case: it halted on the call boundary and resumes to
        # write the body.
        ("assistant\n\nlet me look\n\ncall\n\n", "{"),
    ],
)
def test_first_halt_ignores_positions_at_or_before_start_index(
    prose_tokenizer, resumed_from, continuation
):
    """Halt-and-resume: the sequence we resume from already ENDS in a boundary,
    so re-halting on it would return the same position forever and the loop
    would never advance."""
    criteria = _stop_criteria(prose_tokenizer)
    ids = prose_tokenizer.encode(resumed_from)
    resumed = ids + prose_tokenizer.encode(continuation)
    assert first_halt(torch.tensor([ids]), criteria, len(ids)) is None
    assert first_halt(torch.tensor([resumed]), criteria, len(ids)) is None


def test_first_halt_finds_an_eos_inside_a_commit():
    criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=torch.tensor([9]))])
    ids = torch.tensor([[1, 2, 9, 3, 4]])
    assert first_halt(ids, criteria, start_index=0) == 3


def test_a_deadline_halts_but_never_truncates():
    """A deadline is a length halt: keep what was produced and stop. Walking it
    positionally would see it fire at the first position tested and throw the
    whole commit away."""
    criteria = StoppingCriteriaList([MaxTimeCriteria(max_time=0.0)])
    time.sleep(0.01)
    ids = torch.tensor([[1, 2, 3, 4]])
    assert is_halted(ids, criteria) is True
    assert first_halt(ids, criteria, start_index=0) is None


def test_first_halt_is_none_when_nothing_completed(prose_tokenizer):
    criteria = _stop_criteria(prose_tokenizer)
    ids = torch.tensor([prose_tokenizer.encode("nothing to see here")])
    assert first_halt(ids, criteria, start_index=0) is None


def test_first_halt_and_is_halted_tolerate_an_empty_list():
    ids = torch.tensor([[1, 2, 3]])
    assert first_halt(ids, None, 0) is None
    assert first_halt(ids, StoppingCriteriaList(), 0) is None
    assert is_halted(ids, None) is False


# ---------------------------------------------------------------------------
# streamer plumbing
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.puts = []
        self.ended = False

    def put(self, value):
        self.puts.append(value)

    def end(self):
        self.ended = True


def test_a_multi_token_commit_is_published_one_step_at_a_time():
    """BaseStreamer.put is documented against what _sample hands it - one
    step's [B] tensor - so consumers decoding incrementally never have to guess
    which shape they were given."""
    rec = _Recorder()
    stream_put(rec, torch.tensor([[4, 5, 6]]))
    assert len(rec.puts) == 3
    assert [int(p.item()) for p in rec.puts] == [4, 5, 6]
    assert all(p.dim() == 1 for p in rec.puts)


def test_a_single_step_tensor_is_published_as_is():
    rec = _Recorder()
    stream_put(rec, torch.tensor([4]))
    assert len(rec.puts) == 1 and int(rec.puts[0].item()) == 4


def test_stream_helpers_are_inert_without_a_streamer():
    stream_put(None, torch.tensor([[1]]))
    stream_end(None)


def test_stream_end_signals_the_consumer():
    rec = _Recorder()
    stream_end(rec)
    assert rec.ended is True
