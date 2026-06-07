"""Tests for the tokenmonster integration's tokenizer wrapper."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenmonster")

REPO_ROOT = Path(__file__).parent.parent
VOCAB_NAME = "englishcode-8000-consistent-v1"


def _load_integration():
    if "staging_tokenmonster" in sys.modules:
        return sys.modules["staging_tokenmonster"]
    spec = importlib.util.spec_from_file_location(
        "staging_tokenmonster", REPO_ROOT / "integrations/tokenmonster/__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["staging_tokenmonster"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tm():
    module = _load_integration()
    # Avoid network in CI: only run when the vocab is already cached.
    if not (REPO_ROOT / "build/tokenmonster" / f"{VOCAB_NAME}.vocab").exists():
        pytest.skip("tokenmonster vocab not cached locally")
    return module


@pytest.fixture(scope="module")
def tokenizer(tm):
    return tm.main.TokenMonsterTokenizer(vocab_name=VOCAB_NAME)


def test_registry_keys(tm):
    from praxis.tokenizers import TOKENIZER_REGISTRY, VOCAB_SIZE_CHOICES

    assert "tokenmonster" in TOKENIZER_REGISTRY
    for key in tm.main.VARIANTS:
        assert key in TOKENIZER_REGISTRY
    assert 8000 in VOCAB_SIZE_CHOICES
    assert 50256 in VOCAB_SIZE_CHOICES


def test_resolve_vocab_name(tm):
    resolve = tm.main.resolve_vocab_name
    assert resolve(8000) == "englishcode-8000-consistent-v1"
    assert resolve(8192) == "englishcode-8000-consistent-v1"  # nearest below
    assert resolve(512) == "englishcode-1024-consistent-v1"  # clamp up
    assert resolve(2048, dataset="english", mode="clean") == "english-2048-clean-v1"


def test_special_token_layout(tokenizer):
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.sep_token_id == 3
    assert tokenizer.tool_call_token_id == 4
    assert tokenizer.vocab_size == 8000 + tokenizer.offset


def test_roundtrip(tokenizer):
    text = "Hello World! The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert min(ids) >= tokenizer.offset
    assert tokenizer.decode(ids) == text


def test_specials_inline(tokenizer):
    text = "[BOS]user\nhi[SEP]"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.sep_token_id
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, skip_special_tokens=True) == "user\nhi"


def test_chat_template(tokenizer):
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    assert "[BOS]" in text and "[SEP]" in text
    ids = tokenizer.apply_chat_template(msgs, tokenize=True)["input_ids"]
    assert tokenizer.bos_token_id in ids


def test_assistant_tokens_mask(tokenizer):
    """Slow-tokenizer mask path must not depend on char_to_token."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    out = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = out["input_ids"], out["assistant_masks"]
    assert len(ids) == len(mask)
    assert sum(mask) > 0
    assistant_ids = [i for i, f in zip(ids, mask) if f]
    decoded = tokenizer.decode(assistant_ids)
    assert decoded == "The answer is 4.\n[SEP]\n"
    # Whole-sequence decode matches the rendered template text.
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False)
    assert tokenizer.decode(ids) == rendered


def test_incomplete_tail(tokenizer):
    """Partial tails (split glyphs, pending capcode) must be detectable."""
    text = "CAPS and élève café 中文 emoji \U0001f600 end"
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Detection must agree exactly with decode->reencode lossiness.
    for k in range(1, len(ids) + 1):
        prefix = ids[:k]
        roundtrip = tokenizer.encode(
            tokenizer.decode(prefix), add_special_tokens=False
        )
        assert tokenizer.incomplete_tail(prefix) == (roundtrip != prefix), k
    assert not tokenizer.incomplete_tail(ids)  # full text is complete
    assert not tokenizer.incomplete_tail(ids + [tokenizer.eos_token_id])
    assert not tokenizer.incomplete_tail([])


def test_thread_safety(tokenizer):
    import threading

    errors = []

    def work(n):
        try:
            for i in range(10):
                t = f"thread {n} iteration {i} with CAPS and punctuation!"
                assert tokenizer.decode(
                    tokenizer.encode(t, add_special_tokens=False)
                ) == t
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_fork_safety(tokenizer):
    tokenizer.encode("warm up", add_special_tokens=False)
    pid = os.fork()
    if pid == 0:
        try:
            text = "child process text"
            ok = (
                tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
                == text
            )
            os._exit(0 if ok else 1)
        except Exception:  # pragma: no cover
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    text = "parent after fork"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def test_save_and_reload(tokenizer, tmp_path):
    tokenizer.save_vocabulary(str(tmp_path))
    reloaded = type(tokenizer).from_pretrained(tmp_path)
    assert reloaded.vocab_name == tokenizer.vocab_name
    text = "Reload then roundtrip."
    assert reloaded.decode(reloaded.encode(text, add_special_tokens=False)) == text
