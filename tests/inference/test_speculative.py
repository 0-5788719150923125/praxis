"""Lossless multi-token (speculative) decoding for the byte-latent stack.

The byte-latent core patches non-causally within a partial patch, so a single verify
forward over ``committed + drafts`` reads contaminated earlier positions. The fix reads
each truncated prefix at its OWN last real position (causal) and batches them behind
an attention mask. What makes that lossless, and what these pin:

1. padding invariance - a right-padded, mask-gated prefix predicts the same
   last-real-position token as its unpadded form (incl. the prismatic4
   CrystalVearClassifier router, which must route per-sequence and mask pads);
2. greedy speculative decoding reproduces byte-by-byte greedy exactly, up to
   floating-point argmax ties (batched-GEMM reduction order) where greedy is
   itself ill-defined - at every draft width;
3. a stop string that completes mid-commit cuts the commit exactly there.
"""

import pytest
import torch
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

from praxis import PraxisConfig
from praxis.inference import speculative
from praxis.modeling import PraxisForCausalLM
from praxis.tokenizers.chat_templates import chat_format_of

# ---------------------------------------------------------------------------
# the verifier, on a real drafting stack
# ---------------------------------------------------------------------------


def test_batched_verify_matches_single_row(spec_config):
    """The batched truncated-prefix verifier reads each prefix's last real
    position identically to running that prefix alone - up to float noise.

    This is the deterministic core invariant behind lossless multi-token
    decode. A genuine routing/contamination bug (e.g. the old batch-mean crystal
    merge) shifts these logits by O(0.1-1); float reordering (amplified by the
    crystal classifier's ``-n*log(dist^2)``) stays well under 5e-2. The gap cleanly
    separates the two.
    """
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    torch.manual_seed(500)
    max_diff = 0.0
    for length, k in ((24, 6), (12, 4)):
        gen = torch.randint(4, 260, (1, length))
        cand = torch.randint(4, 260, (1, k))
        batched = speculative.verify_prefixes_batched(model, gen, cand)  # [k, vocab]
        for j in range(1, k + 1):
            prefix = torch.cat([gen, cand[:, :j]], dim=1)
            with torch.no_grad():
                single = model(input_ids=prefix).logits[0, -1]
            max_diff = max(max_diff, (batched[j - 1] - single).abs().max().item())
    assert max_diff < 5e-2, f"batched verify diverges from single-row by {max_diff:.2e}"


def test_readout_is_causal_under_append(spec_config):
    """Appending bytes must not move ANY earlier logit.

    This is the property the one-forward speculative step rests on: a whole
    candidate block is verified by reading positions gen_len-1+k out of a
    single row, which is only lossless if each of those positions equals the
    prefix ending there run on its own. Every stage of the byte-latent stack
    is already causal (prefix-monotone space patching, causal conv local
    encoder/decoder, and decoder_patch_ids never gathering the open patch);
    the classifier is the piece that had to be fixed, because a SMEAR-style
    ``mean(dim=1)`` route let a draft byte reach back and re-route every
    earlier position. A classifier that reintroduces sequence pooling must set
    ``causal_readout = False`` rather than break this.
    """
    from praxis.encoders.byte_latent.constants import OFFSET

    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    assert (
        model.classifier.causal_readout
    ), "spec_config's classifier must declare causal readout"
    base = torch.randint(4, 260, (1, 24))

    # Information test: two rows of the SAME length differing only in the tail.
    # Both tails are alphanumeric bytes ('x' / 'y'), which the space patcher
    # never cuts on, so the two rows patch to the same number of patches and
    # every kernel sees identical shapes. Anything above zero here is then a
    # genuine future read rather than reduction-order noise.
    x_id, y_id = OFFSET + ord("x"), OFFSET + ord("y")
    with torch.no_grad():
        a = model(input_ids=torch.cat([base, torch.full((1, 3), x_id)], 1)).logits
        b = model(input_ids=torch.cat([base, torch.full((1, 3), y_id)], 1)).logits
    leak = (a[:, :24] - b[:, :24]).abs().max().item()
    assert leak == 0.0, f"tail bytes moved earlier logits by {leak:.2e}"

    # Length test: a longer row vs the prefix run alone. Shapes differ here, so
    # batched-GEMM reduction order moves the last bits; only float noise should
    # remain, and the argmax must not move at all.
    with torch.no_grad():
        short = model(input_ids=base).logits
    drift = (a[:, :24] - short).abs().max().item()
    assert drift < 1e-4, f"lengthening the row moved earlier logits by {drift:.2e}"
    flips = (a[0, :24].argmax(-1) != short[0].argmax(-1)).sum().item()
    assert flips == 0, f"{flips} argmax flips from lengthening the row"


def test_speculative_uses_one_forward_per_step(spec_config, monkeypatch):
    """A causal-readout classifier decodes with ONE model forward per step: the
    single verify row carries both the verification and the next step's
    drafting hidden, so total forwards must not exceed the number of committed
    bytes (plus the one that primes the loop)."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    ids = torch.randint(4, 260, (1, 16))

    seen = []
    original = speculative.spec_logits_and_hidden

    def counting(model_, generated, attention_mask=None):
        seen.append(generated.shape)
        return original(model_, generated, attention_mask)

    monkeypatch.setattr(speculative, "spec_logits_and_hidden", counting)
    out = model.generate(
        ids,
        generation_config=GenerationConfig(
            max_new_tokens=24, do_sample=False, num_beams=1
        ),
    )

    produced = out.shape[1] - ids.shape[1]
    assert produced > 0
    # Every forward is a single row: no per-candidate re-encode survives.
    assert all(s[0] == 1 for s in seen), f"batched verify rows leaked back in: {seen}"
    assert len(seen) <= produced + 1, f"{len(seen)} forwards for {produced} bytes"


@pytest.mark.parametrize("repetition_penalty", [None, 1.3])
def test_speculative_matches_byte_by_byte_greedy(spec_config, repetition_penalty):
    """Greedy speculative decoding == byte-by-byte greedy, up to float ties.

    The penalized case holds because the spec sampler applies
    ``repetition_penalty`` per prefix; the terminal passes one to keep rolling
    contexts from degenerating. Any divergence must sit at an argmax tie (top1/
    top2 gap below 3e-2 - generous because the crystal classifier's -n*log(dist^2)
    amplifies sub-1e-3 hidden noise into ~1e-2 logit noise); a real correctness
    bug shifts logits by O(0.1-1), far above it.
    """
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    assert model.mtp is not None and model.mtp.byte_level
    procs = LogitsProcessorList(
        [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)]
        if repetition_penalty
        else []
    )

    torch.manual_seed(321)
    length, n_new = 10, 20
    ids = torch.randint(4, 260, (1, length))

    ref, gaps = ids.clone(), []
    for _ in range(n_new):
        with torch.no_grad():
            raw = model(input_ids=ref).logits[0, -1:].clone()  # [1, vocab]
        scored = procs(ref, raw)[0]
        top2 = scored.topk(2).values
        gaps.append((top2[0] - top2[1]).item())
        ref = torch.cat([ref, scored.argmax().view(1, 1)], dim=1)

    extra = {"repetition_penalty": repetition_penalty} if repetition_penalty else {}
    spec = model.generate(
        ids,
        generation_config=GenerationConfig(
            max_new_tokens=n_new, do_sample=False, num_beams=1, **extra
        ),
    )
    ref_bytes = ref[0, length:].tolist()
    spec_bytes = spec[0, length : length + n_new].tolist()
    for i in range(min(len(ref_bytes), len(spec_bytes))):
        if ref_bytes[i] != spec_bytes[i]:
            assert gaps[i] < 3e-2, (
                f"speculative diverged from greedy at a non-tie "
                f"(pos={i}, gap={gaps[i]:.2e})"
            )
            break  # first divergence resyncs; downstream is a fresh context


def test_speculative_sampled_always_commits(spec_config):
    """Under sampling every step commits at least one byte, drawn from a REAL
    conditional, and the realized-throughput metrics stay in range.

    Candidate 0 is auto-accepted only when it was sampled from a MEASURED
    hidden: re-drawing from the same distribution adds no correctness, only
    spurious rejections. When it came from ``mtp.bridge_hidden`` instead (the
    common case - every step commits one byte past the block its forward read)
    it is an approximate draw, so it is verified like any other candidate and
    the step falls back to committing the verify's own sample. That byte is
    still exact, so progress is guaranteed and no committed byte ever comes
    from the bridge.

    The accept EMA therefore MAY sit below 1 under sampling: equality-based
    acceptance of a sampled draft succeeds with probability ~sum(p^2), so
    drafts genuinely rarely survive.
    """
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    ids = torch.randint(4, 260, (1, 12))
    gen_cfg = GenerationConfig(
        max_new_tokens=24,
        do_sample=True,
        temperature=1.0,
        num_beams=1,
        repetition_penalty=1.15,
    )
    torch.manual_seed(7)
    out = model.generate(ids, generation_config=gen_cfg)
    assert out.shape[1] >= ids.shape[1] + 24  # sampled steps still commit
    assert model.mtp._accept_seen > 0
    assert model.mtp._accept_ema >= 0.0

    metrics = model.mtp.training_metrics()
    assert metrics["mtp_accept_run"] >= 0.0
    assert 1 <= metrics["mtp_draft_width"] <= spec_config.mtp_depth


# ---------------------------------------------------------------------------
# the draft width
# ---------------------------------------------------------------------------
#
# Width is the one speculative knob chosen at DECODE time from the run's own
# accepted-run lengths. It is only safe to adapt because every committed byte
# is confirmed against a real forward, so it changes how much work is thrown
# away - never what is written.


@pytest.fixture(scope="module")
def mtp_model():
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        decoder_type="sequential",
        classifier_type="forward",
        encoder_type="abstractinator_v1",
        tokenizer_type="byte_level",
        byte_offset=0,
        byte_vocab_size=256,
        codebook_size=256,
        max_position_embeddings=512,
        mtp_depth=5,
        mtp_type="per_depth",
    )
    torch.manual_seed(0)
    return PraxisForCausalLM(cfg).eval()


def test_greedy_output_is_identical_at_every_width(mtp_model, monkeypatch):
    torch.manual_seed(1)
    ids = torch.randint(32, 127, (1, 48))
    cfg = GenerationConfig(max_new_tokens=20, do_sample=False, use_cache=True)

    def run():
        with torch.no_grad():
            return mtp_model.generate(ids, generation_config=cfg)

    adaptive = run()
    for width in (1, 2, 5):
        monkeypatch.setattr(
            type(mtp_model.mtp), "draft_width", property(lambda self, w=width: w)
        )
        assert torch.equal(adaptive, run()), f"width {width} wrote different bytes"


# ---------------------------------------------------------------------------
# halting on a text boundary mid-commit
# ---------------------------------------------------------------------------


class _ScriptedMTP:
    """MTP stub that drafts nothing, so each step commits the main forward's
    token plus the bonus token - a two-byte commit, which is the case where a
    boundary can complete mid-run."""

    byte_level = True

    def __init__(self):
        self.accepted = []

    def draft_next_tokens(self, hidden, token_0, embed_fn, classifier):
        return token_0.new_zeros((1, 0))

    def note_accepted(self, n):
        self.accepted.append(n)


class _ScriptedModel:
    """The minimum surface ``speculative_decoding`` touches, driven by a fixed
    list of byte ids so the loop's output is deterministic."""

    encoder = object()  # truthy: take the byte-latent branch
    embeds = None
    classifier = None

    def __init__(self, script, prompt_len, vocab_size=264):
        self.script = list(script)
        self.prompt_len = prompt_len
        self.vocab_size = vocab_size
        self.mtp = _ScriptedMTP()

    def _one_hot(self, index):
        logits = torch.full((1, self.vocab_size), -10.0)
        if 0 <= index < len(self.script):
            logits[0, self.script[index]] = 10.0
        else:
            logits[0, 0] = 10.0  # past the script: emit PAD
        return logits

    def spec_logits_and_hidden(self, generated, attention_mask=None):
        produced = generated.size(1) - self.prompt_len
        logits = torch.full((1, generated.size(1), self.vocab_size), -10.0)
        logits[:, -1, :] = self._one_hot(produced)
        return logits, torch.zeros(1, generated.size(1), 8)

    def verify_prefixes_batched(self, generated, candidates):
        produced = generated.size(1) - self.prompt_len
        return self._one_hot(produced + candidates.size(1))


@pytest.fixture
def run_scripted(monkeypatch):
    """Drive the real decoding method with criteria assembled the way
    ``generate`` assembles them.

    The scripted model is not a ``PreTrainedModel``, so it cannot go through
    ``generate`` itself - but the decoding method is a plain function taking
    ``model`` explicitly, so it can be called with the same inputs transformers
    would have handed it. The halt contract under test is then transformers'
    own ``StopStringCriteria``.
    """
    monkeypatch.setattr(
        speculative,
        "spec_logits_and_hidden",
        lambda m, generated, attention_mask=None: m.spec_logits_and_hidden(
            generated, attention_mask
        ),
    )
    monkeypatch.setattr(
        speculative,
        "verify_prefixes_batched",
        lambda m, generated, candidates: m.verify_prefixes_batched(
            generated, candidates
        ),
    )

    def run(tokenizer, prompt, continuation, stop_strings, max_new_tokens=200):
        prompt_ids = tokenizer.encode(prompt)
        model = _ScriptedModel(tokenizer.encode(continuation), len(prompt_ids))
        criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=len(prompt_ids) + max_new_tokens)]
        )
        if stop_strings:
            criteria.append(
                StopStringCriteria(stop_strings=list(stop_strings), tokenizer=tokenizer)
            )
        out = speculative.speculative_decoding(
            model,
            torch.tensor([prompt_ids], dtype=torch.long),
            stopping_criteria=criteria,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False
            ),
        )
        return tokenizer.decode(out[0], skip_special_tokens=False)

    return run


PROMPT = "user\n\nhi\n\nassistant\n\n"


@pytest.mark.parametrize("reply", ["ok", "yes", "Hello there!"])
def test_speculative_decode_halts_mid_commit(prose_tokenizer, run_scripted, reply):
    """The path abstractinator-g decodes through. Each step commits two bytes,
    so the boundary's last byte can land on the first of a pair (the replies
    differ in parity); ``first_halt`` must still cut exactly there, dropping
    everything drafted past it."""
    stops = chat_format_of(prose_tokenizer).stop_strings()
    text = run_scripted(
        prose_tokenizer, PROMPT, f"{reply}\n\nuser\n\nXXXXXX drafted past", stops
    )
    assert text == PROMPT + f"{reply}\n\nuser\n\n"


def test_speculative_decode_without_stop_strings_runs_on(prose_tokenizer, run_scripted):
    """Control: the halt comes from the stop strings, not from the script."""
    text = run_scripted(
        prose_tokenizer,
        PROMPT,
        "Hello there!\n\nuser\n\nkeeps going",
        (),
        max_new_tokens=40,
    )
    assert text.startswith(PROMPT + "Hello there!\n\nuser\n\nkeeps going")
    assert len(text) > len(PROMPT + "Hello there!\n\nuser\n\n")
