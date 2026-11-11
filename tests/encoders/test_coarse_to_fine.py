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
from praxis.inference.decoding import trunk_hooks
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
    C, S = enc.chunk, enc._block_pattern("cpu")[0].numel()
    # The edit changes how many chunks there are, so each run's blocks start
    # after its own context stream.
    torch.testing.assert_close(
        hidden[:, n * C : n * C + S], base_hidden[:, base_n * C : base_n * C + S]
    )
    torch.testing.assert_close(hidden[:, :C], base_hidden[:, :C])
    assert not torch.allclose(hidden[:, C : 2 * C], base_hidden[:, C : 2 * C])
    # Positions whose target byte is still in chunk 0.
    torch.testing.assert_close(logits[:, : start - 1], base_logits[:, : start - 1])


def test_no_round_reads_its_own_or_a_later_rounds_codes(monkeypatch):
    """One finest code, drawn in round 1, changed: only the round blocks that
    read it as drawn move. No coarser level, its own round or an earlier one
    sees it, so each round's loss conditions on earlier rounds alone."""
    model = _model().eval()
    enc = model.encoder
    ids = _rows()[:1]
    finest = len(enc.ladder.lengths) - 1
    here = enc.fine_round == 1
    slot = int(enc.fine_index[here][~enc.fine_known[here]][0])
    quantize = enc.ladder.forward

    def edited(z, valid, learn=True):
        codes, masks, _ = quantize(z, valid, learn)
        codes = [c.clone() for c in codes]
        codes[-1][:, 0, slot] = (codes[-1][:, 0, slot] + 1) % enc.ladder.levels[-1].K
        return codes, masks, enc.ladder.reconstructions(codes, valid)

    with torch.no_grad():
        base, _, _ = _pieces(model, ids)
        monkeypatch.setattr(enc.ladder, "forward", edited)
        moved, _, _ = _pieces(model, ids)
    n, C = enc._state["n"], enc.chunk
    _, steps = enc._block_pattern("cpu")
    first = slice(n * C, n * C + steps.numel())
    changed = (moved[0, first] - base[0, first]).abs().amax(-1) > 1e-6
    assert not changed[steps <= finest + 1].any(), "a round read its own codes"
    assert changed[steps > finest + 1].any(), "later rounds never read the code"


@pytest.mark.parametrize(
    "encoder_type",
    ["byte_latent_conv_coarse_to_fine", "byte_latent_conv_coarse_to_fine_parallel"],
)
def test_generation_draws_codes_from_what_training_scores(encoder_type):
    """Handed a chunk's true codes, generation's code logits at every level and
    round are the ones training scores, and its reconstruction is the ladder's."""
    model = _model(encoder_type=encoder_type).eval()
    enc = model.encoder
    ids = _rows()[:1]
    c, finest = 1, len(enc.ladder.lengths) - 1
    with torch.no_grad():
        hidden, _, _ = _pieces(model, ids)
        inputs = enc.encode(ids)[0]
        st = enc._state
        n, C = st["n"], enc.chunk
        S = enc._block_pattern("cpu")[0].numel()
        widths = list(enc.ladder.lengths[:-1]) + [enc.fine_index.numel()]
        parts = hidden[0, n * C + c * S : n * C + (c + 1) * S].split(widths)
        expected = [enc.code_logits(k, parts[k]) for k in range(finest)]
        for r in range(enc.rounds):
            here = enc.fine_round == r
            drawn = parts[-1][here][~enc.fine_known[here]]
            expected.append(enc.code_logits(finest, drawn))

        seen = []

        def pick(level, logits):
            seen.append(logits)
            if level < finest:
                return st["codes"][level][0, c]
            here = enc.fine_round == len(seen) - 1 - finest
            return st["codes"][level][0, c][enc.fine_index[here][~enc.fine_known[here]]]

        trunk = {"root": model.decoder, "forward": trunk_hooks(model)["latent_forward"]}
        recon = enc._draw_chunk(trunk, inputs[:, : c * C], pick)
    assert len(seen) == finest + enc.rounds
    for got, want in zip(seen, expected):
        torch.testing.assert_close(got, want.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(recon[0], st["sums"][-1][0, c])


def test_the_decoder_reads_no_earlier_patch():
    """Bytes reach the decoder only within their own patch; anything about an
    earlier patch has to come through the codes. Rewriting one patch's bytes
    under fixed latents moves only the positions spelling that patch."""
    model = _model().eval()
    enc = model.encoder
    ids = _rows()[:1]
    with torch.no_grad():
        model(input_ids=ids)
        target, latents = enc._state["target_patch"], enc._state["latents"]
        base = enc._spell(ids, target, latents)
        patch = int(target[0, 40])
        inside = (target[0] == patch).nonzero().squeeze(-1)
        edited = ids.clone()
        edited[0, inside[1:]] = torch.randint(97, 123, (inside.numel() - 1,))
        moved = enc._spell(edited, target, latents)
    elsewhere = target[0] != patch
    torch.testing.assert_close(moved[0, elsewhere], base[0, elsewhere])
    assert not torch.allclose(moved[0, ~elsewhere], base[0, ~elsewhere])


def test_the_byte_loop_reads_what_training_reads():
    """Generation's next-byte logits, given the latents training decoded from,
    are the teacher-forced logits."""
    model = _model().eval()
    enc = model.encoder
    ids = _rows()[:1]
    with torch.no_grad():
        logits = model(input_ids=ids).logits
        table = enc._state["latents"]
        for n in (1, 40, 300, 511):
            prefix = ids[:, :n]
            cut, _ = enc.patcher.patch(prefix, include_next_token=True)
            cut = cut[:, cut[0] > 0]
            step = enc._next_byte_logits(model, prefix, cut, table)
            torch.testing.assert_close(step, logits[:, n - 1], atol=1e-4, rtol=1e-4)


def test_generation_continues_a_prompt_that_ends_mid_word(monkeypatch):
    """The first generated byte finishes the prompt's open patch, cut as
    training cuts it, so the decoder reads that patch's bytes so far. A fresh
    patch would hide every earlier byte from it, which byte-at-a-time callers
    (the dashboard asks for one byte per call) hit on every byte."""
    model = _model().eval()
    enc = model.encoder
    prompt = torch.tensor([list(b"the quick bro")])
    seen, step = [], enc._next_byte_logits

    def spy(model_, tokens, patch_lengths, table, origin=0):
        seen.append(patch_lengths.clone())
        return step(model_, tokens, patch_lengths, table, origin)

    monkeypatch.setattr(enc, "_next_byte_logits", spy)
    model.generate(prompt, max_new_tokens=1, do_sample=False)
    natural, _ = enc.patcher.patch(prompt, include_next_token=True)
    assert torch.equal(seen[0], natural[:, natural[0] > 0])
    assert int(seen[0][0, -1]) == len(b"bro") + 1  # its bytes and the next one


def test_every_ladder_cut_decodes_and_the_last_is_the_forward():
    model = _model().eval()
    enc = model.encoder
    ids = _rows()
    with torch.no_grad():
        logits = model(input_ids=ids).logits
        cuts = enc.ladder_cuts(model.classifier)
    assert len(cuts) == len(enc.ladder.lengths)
    torch.testing.assert_close(cuts[-1][0], logits)
    torch.testing.assert_close(cuts[-1][1], enc.code_nats())
    assert not torch.allclose(cuts[0][0], logits), "a cut decoded the whole ladder"


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
