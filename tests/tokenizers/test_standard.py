"""``StandardTokenizer``: a trained tokenizer loaded the normal way."""

import pytest

from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.byte_level import ByteLevelTokenizer


@pytest.fixture(scope="module")
def trained_tokenizer():
    """Loads through the HF hub cache; only the load may skip, never a result."""
    try:
        tokenizer = create_tokenizer(vocab_size=32768)
    except Exception as e:
        pytest.skip(f"StandardTokenizer not available: {e}")
    if isinstance(tokenizer, ByteLevelTokenizer):
        pytest.skip("No trained StandardTokenizer available")
    return tokenizer


@pytest.mark.network
def test_trained_tokenizer_round_trips(trained_tokenizer):
    text = "Hello, world!"
    ids = trained_tokenizer.encode(text, add_special_tokens=False)
    assert trained_tokenizer.decode(ids) == text


@pytest.mark.network
@pytest.mark.xfail(
    strict=True,
    reason="the published UNSAFE/praxis-32768 tokenizer has no post-processor, so "
    "add_special_tokens=True adds no BOS/EOS",
)
def test_trained_tokenizer_adds_specials_when_asked(trained_tokenizer):
    text = "Hello, world!"
    plain = trained_tokenizer.encode(text, add_special_tokens=False)
    assert len(trained_tokenizer.encode(text, add_special_tokens=True)) > len(plain)
