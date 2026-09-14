"""Iterative-unmask decoding.

Generation here is not sampling one token after another: the block exists from
the first step and each forward replaces some of its mask symbols. These tests
pin the properties that a caller depends on - the prompt survives, no mask
symbol is ever emitted, the block is exactly the requested length - and the one
that decides whether the whole approach is worth anything, which is that the
step count actually reaches the loop so quality against steps can be measured.
"""

import pytest
import torch

from praxis import PraxisConfig
from praxis.diffusion.decoding import unmask_decoding
from praxis.diffusion.masked import MaskedDiffusion
from praxis.modeling import PraxisForCausalLM

MASK_ID = 256
VOCAB = 257


def _model(**over):
    base = dict(
        diffusion_type="masked",
        mask_token_id=MASK_ID,
        vocab_size=VOCAB,
        hidden_size=64,
        embed_size=64,
        num_heads=1,
        num_queries=1,
        head_size=32,
        depth=2,
        num_layers=1,
        block_size=64,
        dropout=0.0,
        diffusion_steps=8,
        encoder_type=None,
        attention_type="causal",
        classifier_type="forward",
        regularizers=[],
        mtp_type=None,
        rl_type=[],
    )
    base.update(over)
    torch.manual_seed(0)
    return PraxisForCausalLM(PraxisConfig(**base)).eval()


def test_generate_resolves_to_the_unmask_loop():
    model = _model()
    assert model._resolve_decoding_method(None, None) is unmask_decoding


def test_generate_is_not_the_unmask_loop_without_diffusion():
    model = _model(diffusion_type=None, vocab_size=256, mask_token_id=None)
    assert model._resolve_decoding_method(None, None) is not unmask_decoding


@pytest.mark.parametrize("steps", [1, 4, 16])
def test_block_is_exactly_the_requested_length(steps):
    model = _model(diffusion_steps=steps)
    prompt = torch.randint(0, 256, (1, 6))
    out = model.generate(inputs=prompt, max_new_tokens=20, do_sample=False)
    assert out.shape == (1, 26)


def test_prompt_survives_and_no_mask_is_emitted():
    model = _model()
    prompt = torch.randint(0, 256, (2, 6))
    out = model.generate(inputs=prompt, max_new_tokens=16, do_sample=False)

    assert torch.equal(out[:, :6], prompt), "the loop overwrote its own prompt"
    assert not (out == MASK_ID).any(), "a mask symbol was emitted as a token"
    assert int(out.max()) < MASK_ID


def test_every_position_is_committed_even_when_steps_are_few():
    """The schedule walks the masked count to zero; the final sweep catches any
    position a rounding step left behind."""
    model = _model(diffusion_steps=3)
    prompt = torch.randint(0, 256, (1, 4))
    out = model.generate(inputs=prompt, max_new_tokens=37, do_sample=False)
    assert not (out == MASK_ID).any()


def test_greedy_decoding_is_deterministic():
    model = _model()
    prompt = torch.randint(0, 256, (1, 8))
    a = model.generate(inputs=prompt, max_new_tokens=16, do_sample=False)
    b = model.generate(inputs=prompt, max_new_tokens=16, do_sample=False)
    assert torch.equal(a, b)


def test_step_count_reaches_the_loop():
    """Quality-against-steps is the measurement that decides whether diffusion
    buys anything over autoregression, so a step count that silently failed to
    apply would make every such reading meaningless. Counted as model forwards,
    which is also the thing the step count actually costs."""
    model = _model()
    prompt = torch.randint(0, 256, (1, 4))

    for steps in (2, 8, 16):
        model.config.diffusion_steps = steps
        calls = []
        handle = model.register_forward_pre_hook(lambda *a, **k: calls.append(1))
        try:
            model.generate(inputs=prompt, max_new_tokens=16, do_sample=False)
        finally:
            handle.remove()
        # One forward per refinement pass, plus at most one final sweep.
        assert steps <= len(calls) <= steps + 1, (
            f"{steps} steps asked for, {len(calls)} forwards taken"
        )


def test_module_level_generate_matches_the_decoding_method():
    """MaskedDiffusion.generate is the same loop without transformers in the
    way - used by tests and probes, so it must not drift from the real path."""
    model = _model(diffusion_steps=4)
    diffusion = model.criterion.main
    prompt = torch.randint(0, 256, (1, 6))

    direct = diffusion.generate(
        lambda x: model(input_ids=x).logits,
        prompt_ids=prompt,
        length=12,
        steps=4,
        temperature=0.0,
    )
    via_hf = model.generate(inputs=prompt, max_new_tokens=12, do_sample=False)
    assert torch.equal(direct, via_hf)


def test_learns_to_reconstruct_a_periodic_corpus():
    """End to end: the objective, the corruption and the decode loop together
    on a stream that is genuinely learnable. Held-out masked-position accuracy
    far above chance is the smallest honest claim that this works at all."""
    torch.manual_seed(0)
    text = (b"the quick brown fox jumps over the lazy dog. " * 24)
    data = torch.tensor(list(text), dtype=torch.long)

    model = _model(hidden_size=128, embed_size=128, num_heads=2, head_size=64, depth=4)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(220):
        starts = torch.randint(0, len(data) - 65, (16,))
        ids = torch.stack([data[s : s + 64] for s in starts])
        out = model(input_ids=ids, labels=ids)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    diffusion = model.criterion.main
    with torch.no_grad():
        starts = torch.randint(0, len(data) - 65, (16,))
        ids = torch.stack([data[s : s + 64] for s in starts])
        noisy, masked, _ = diffusion.corrupt(ids)
        pred = model(input_ids=noisy).logits[..., :MASK_ID].argmax(-1)
        accuracy = float((pred[masked] == ids[masked]).float().mean())

    assert accuracy > 0.5, f"masked-position accuracy {accuracy:.3f} (chance 0.004)"
    assert diffusion.training_metrics()["diffusion_unigram_gap"] > 0.5


def test_logits_processors_see_the_mask_column():
    """A processor indexes ``scores`` with ids read off ``input_ids``, and the
    block is full of mask ids until the last pass. Trimming the mask column
    before the processors run puts every one of those gathers out of bounds -
    silent on CPU in some builds, a device-side assert on CUDA with the
    traceback pointing at whatever ran next.
    """
    from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

    from praxis.diffusion.decoding import refine

    model = _model()
    mask_id = model.criterion.main.mask_token_id
    prompt = torch.randint(0, 256, (1, 6))

    out = refine(
        lambda ids: model(input_ids=ids).logits,
        prompt,
        16,
        mask_id=mask_id,
        steps=4,
        logits_processor=LogitsProcessorList(
            [RepetitionPenaltyLogitsProcessor(penalty=1.2)]
        ),
    )
    assert out.shape == (1, 22)
    assert not (out == mask_id).any()


def test_decoding_ignores_length_stopping_criteria():
    """The block is allocated at full length before the first forward, so a
    length criterion is already satisfied on pass 1. Consulting the prepared
    list ended refinement after a single pass and returned first guesses."""
    model = _model(diffusion_steps=8)
    prompt = torch.randint(0, 256, (1, 4))

    calls = []
    handle = model.register_forward_pre_hook(lambda *a, **k: calls.append(1))
    try:
        model.generate(inputs=prompt, max_new_tokens=16, do_sample=False)
    finally:
        handle.remove()
    assert len(calls) >= 8, f"refinement stopped after {len(calls)} passes"


@pytest.mark.parametrize("return_dict", [False, True])
def test_generate_honours_the_return_shape(return_dict):
    """transformers decides the return shape, not the decoding method. With
    ``return_dict_in_generate`` the caller reads ``.sequences``; handing back a
    bare tensor failed every inference request with "'Tensor' object has no
    attribute 'sequences'" - which is how the serving path calls it."""
    model = _model()
    prompt = torch.randint(0, 256, (1, 6))
    out = model.generate(
        inputs=prompt, max_new_tokens=12, do_sample=False,
        return_dict_in_generate=return_dict,
    )
    sequences = out.sequences if return_dict else out
    assert torch.is_tensor(sequences)
    assert sequences.shape == (1, 18)
    if return_dict:
        assert not torch.is_tensor(out), "a dict-returning call got a bare tensor"


@pytest.mark.parametrize("steps", [1, 3, 8, 16])
@pytest.mark.parametrize("batch", [1, 5])
def test_batched_commit_selects_the_same_positions(steps, batch):
    """The commit step is a batched top-k that replaced a Python loop over rows.

    It has to pick exactly what the loop picked: the schedule decides how many
    positions commit, confidence decides which, and neither may change with
    batch size. A row decoded alone and the same row decoded inside a batch must
    come out identical - if they differ, the top-k is leaking across rows.
    """
    from praxis.diffusion.decoding import refine

    mask_id = 50
    torch.manual_seed(0)
    # A fixed id -> logits table, so the denoiser is deterministic and depends
    # on the current ids the way a real one does.
    table = torch.randn(mask_id + 1, mask_id + 1)

    def denoise(ids):
        return table[ids]

    prompt = torch.randint(0, mask_id, (batch, 4))
    out = refine(denoise, prompt, 20, mask_id=mask_id, steps=steps)

    assert out.shape == (batch, 24)
    assert torch.equal(out[:, :4], prompt), "the prompt moved"
    assert not (out == mask_id).any(), "a position was left masked"

    solo = refine(denoise, prompt[:1], 20, mask_id=mask_id, steps=steps)
    assert torch.equal(out[:1], solo), "a row's decode depended on its batch"


def test_commit_budget_respects_the_schedule():
    """Each pass commits exactly the schedule's share, no more: committing early
    would silently turn a 16-step decode into a 1-step one."""
    from praxis.diffusion.decoding import refine

    mask_id = 50
    torch.manual_seed(0)
    table = torch.randn(mask_id + 1, mask_id + 1)
    seen = []

    def denoise(ids):
        seen.append(int((ids == mask_id).sum()))
        return table[ids]

    refine(denoise, torch.randint(0, mask_id, (2, 4)), 16, mask_id=mask_id, steps=4)
    # 32 masked across 2 rows, falling by a quarter each pass.
    assert seen == [32, 24, 16, 8], f"commit schedule was {seen}"


def test_streamer_receives_the_block_once():
    """A streamer's ``put`` APPENDS, and a diffusion block is revised in place
    rather than extended. Publishing each refinement pass appended the whole
    block again every pass - mask symbols included - so an 8-step decode of 16
    tokens streamed 128 ids, 56 of them the absorbing symbol. There is no
    token-by-token reveal to show: nothing is final until the pass that commits
    it, and positions commit by confidence rather than left to right.
    """
    model = _model(diffusion_steps=8)

    class _Streamer:
        def __init__(self):
            self.ids = []
            self.ended = False

        def put(self, value):
            row = value if value.dim() == 1 else value[0]
            self.ids.extend(int(i) for i in row.reshape(-1).tolist())

        def end(self):
            self.ended = True

    streamer = _Streamer()
    prompt = torch.randint(0, 256, (1, 5))
    out = model.generate(
        inputs=prompt, max_new_tokens=16, do_sample=False, streamer=streamer
    )

    # transformers publishes the prompt; this loop publishes the block.
    assert streamer.ids == out[0].tolist(), "the stream did not match the output"
    assert len(streamer.ids) == 21, f"streamed {len(streamer.ids)} ids, expected 21"
    assert MASK_ID not in streamer.ids, "an absorbing symbol reached the streamer"
    assert streamer.ended
