"""The coarse-to-fine encoder (byte-latent-j).

What makes its loss an honest bound is asserted here: the code loss trains only
the trunk and the reconstruction loss never reaches it, no chunk's codes or
predictions read a later chunk, and evaluation is deterministic. Generation is
checked against transformers' decoding-method contract.
"""

import pytest
import torch
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.attention.self_attention import SelfAttention
from praxis.containers import LossContainer
from praxis.metrics.copy_probe import copy_gain
from praxis.metrics.trunk_probe import trunk_swap_cost
from praxis.modeling import PraxisForCausalLM

TEXT = b"the quick brown fox jumps over the lazy dog, said nobody at all. " * 40


def _model(**over):
    base = dict(
        vocab_size=264,
        byte_vocab_size=256,
        byte_offset=0,
        hidden_size=32,
        embed_size=32,
        num_heads=2,
        num_queries=1,
        head_size=16,
        depth=2,
        num_layers=2,
        encoder_type="byte_latent_conv_coarse_to_fine",
        decoder_type="sequential",
        attention_type="self_attention",
        encoding="rope",
        block_size=512,
        max_position_embeddings=4096,
        device_map="cpu",
        dropout=0.0,
        regularizers=[],
        rl_type=[],
        mtp_type=None,
        pad_token_id=None,
    )
    base.update(over)
    torch.manual_seed(0)
    return PraxisForCausalLM(PraxisConfig(**base))


def _rows(n=512):
    return torch.tensor([list(TEXT[:n]), list(TEXT[9 : 9 + n])])


def _pieces(model, ids):
    """encode -> trunk -> decode by hand: (trunk output, code loss, decoder features)."""
    enc = model.encoder
    inputs, h_encoder, lengths, _, _, tokens = enc.encode(ids)
    SelfAttention.set_layout(model.decoder, enc.trunk_layout)
    try:
        hidden = model.decoder(inputs, None, None, None, None, LossContainer(), None)[0]
    finally:
        SelfAttention.set_layout(model.decoder, None)
    _, features = enc.decode(hidden, h_encoder, ids, lengths, tokens)
    return hidden, enc.consume_pending_losses()["code"], features


def _grads(module):
    return [
        p.grad
        for p in module.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]


def test_the_code_loss_trains_the_trunk_and_nothing_of_the_tokenizer():
    model = _model().train()
    _, code, _ = _pieces(model, _rows())
    code.backward()
    assert _grads(model.decoder), "the trunk got no gradient from its own loss"
    # The classifier holds the encoder as a submodule, so read its own scorer.
    tokenizer = (
        model.encoder.encoder,
        model.encoder.decoder,
        model.embeds,
        model.classifier.scorer,
    )
    for part in tokenizer:
        assert not _grads(part), f"the code loss reached {type(part).__name__}"


def test_the_reconstruction_loss_never_reaches_the_trunk():
    model = _model().train()
    ids = _rows()
    _, _, features = _pieces(model, ids)
    logits = model.classifier(features)[:, :-1]
    F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    ).backward()
    assert not _grads(model.decoder), "the trunk trained on reconstruction"
    assert _grads(model.encoder.encoder), "the tokenizer's encoder got no gradient"


def test_no_chunk_reads_a_later_one():
    model = _model().eval()
    enc = model.encoder
    ids = _rows()[:1]
    with torch.no_grad():
        base_hidden, _, _ = _pieces(model, ids)
        base_codes = [c.clone() for c in enc._state["codes"]]
        base_n = enc._state["n"]
        base_logits = model(input_ids=ids).logits
        lengths, _ = enc.patcher.patch(ids, include_next_token=True)
        start = int(torch.cumsum(lengths[0], 0)[enc.chunk - 1])  # chunk 1's first byte
        edited = ids.clone()
        edited[:, start:] = torch.randint(97, 123, edited[:, start:].shape)
        hidden, _, _ = _pieces(model, edited)
        n = enc._state["n"]
        logits = model(input_ids=edited).logits

    for before, after in zip(base_codes, enc._state["codes"]):
        assert torch.equal(before[:, 0], after[:, 0]), "chunk 0's codes moved"
    C, S = enc.chunk, sum(enc.ladder.lengths)
    # The edit changes how many chunks there are, so each run's blocks start
    # after its own context stream.
    torch.testing.assert_close(
        hidden[:, n * C : n * C + S], base_hidden[:, base_n * C : base_n * C + S]
    )
    torch.testing.assert_close(hidden[:, :C], base_hidden[:, :C])
    assert not torch.allclose(hidden[:, C : 2 * C], base_hidden[:, C : 2 * C])
    # Positions whose target byte is still in chunk 0.
    torch.testing.assert_close(logits[:, : start - 1], base_logits[:, : start - 1])


def test_evaluation_is_deterministic():
    model = _model().eval()
    ids = _rows()
    with torch.no_grad():
        model(input_ids=ids, labels=ids[:, 1:].contiguous())
        first = model.encoder.code_nats().clone()
        model(input_ids=ids, labels=ids[:, 1:].contiguous())
    assert torch.isfinite(first) and first > 0
    assert torch.equal(
        first, model.encoder.code_nats()
    ), "eval corrupted or cut the ladder"


def test_codebooks_learn_only_on_steps_with_gradient():
    model = _model().train()
    ids = _rows()
    with torch.no_grad():  # the lazy-init dummy forward's conditions
        model(input_ids=ids, labels=ids[:, 1:].contiguous())
    assert not bool(model.encoder.ladder.seeded)
    model(input_ids=ids, labels=ids[:, 1:].contiguous())
    assert bool(model.encoder.ladder.seeded)


@pytest.mark.parametrize("return_dict", [False, True])
def test_generation_returns_the_prompt_and_new_bytes(return_dict):
    model = _model().eval()
    prompt = torch.tensor([list(b"the quick brown")])
    out = model.generate(
        prompt, max_new_tokens=12, do_sample=False, return_dict_in_generate=return_dict
    )
    seq = out.sequences if return_dict else out
    assert seq.shape == (1, prompt.shape[1] + 12)
    assert torch.equal(seq[:, : prompt.shape[1]], prompt)


def test_generation_handles_a_batch_and_sampling():
    model = _model().eval()
    prompts = torch.tensor([list(b"the quick brown"), list(b"over the lazy d")])
    seq = model.generate(prompts, max_new_tokens=6, do_sample=True, temperature=0.8)
    assert seq.shape == (2, prompts.shape[1] + 6)


def test_context_probes_skip_reconstruction_logits():
    model = _model().eval()
    ids = _rows()
    with torch.no_grad():
        assert copy_gain(model, ids) is None
        assert trunk_swap_cost(model, ids) is None


@pytest.mark.parametrize(
    "over",
    [{"encoding": "alibi"}, {"attention_type": "kaleido"}, {"mtp_type": "linear"}],
)
def test_it_refuses_what_its_layout_cannot_steer(over):
    with pytest.raises(ValueError):
        _model(**over)
