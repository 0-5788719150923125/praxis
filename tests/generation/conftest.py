import pytest

from praxis import PraxisConfig


@pytest.fixture
def spec_config():
    """Byte-latent + prismatic4 head + dual memory + VEAR MTP (drafting stack)."""
    return PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        head_type="prismatic4",
        memory_type="mal_energy_dual",
        mtp_type="vear",
        mtp_depth=4,
    )
