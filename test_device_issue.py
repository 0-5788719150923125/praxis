#!/usr/bin/env python
"""Simple test to reproduce the device management issue."""

import torch
from unittest.mock import Mock
from praxis import PraxisConfig, PraxisForCausalLM
from praxis.generation.generator import Generator
from praxis.tokenizers import create_tokenizer

print("[TEST] Creating mock Lightning module...")

# Create a small model for testing
config = PraxisConfig(
    depth=2,
    hidden_size=64,
    embed_size=64,
    num_heads=2,
    vocab_size=256,
    decoder_type="sequential",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TEST] Using device: {device}")

base_model = PraxisForCausalLM(config)
if device == "cuda":
    base_model = base_model.to(device)

# Create a mock Lightning module
mock_lightning = Mock()
mock_lightning.model = base_model
mock_lightning.training = False

print(f"[TEST] Model is on device: {next(base_model.parameters()).device}")

# Create tokenizer
tokenizer = create_tokenizer(vocab_size=256)

# Create generator with mock Lightning module
print("[TEST] Creating generator with mock Lightning module...")
generator = Generator(
    mock_lightning, tokenizer, device="cuda"
)  # Note: passing "cuda" string

# Test messages
messages = [{"role": "user", "content": "Hello"}]

print("[TEST] Testing generation...")
print(
    f"[TEST] Before generation - model device: {next(base_model.parameters()).device}"
)

try:
    result = generator.generate_with_messages(
        messages,
        max_new_tokens=5,
        temperature=0.4,
        do_sample=True,
    )
    print(f"[TEST] Generation successful")
    print(
        f"[TEST] After generation - model device: {next(base_model.parameters()).device}"
    )

except Exception as e:
    print(f"[TEST] Generation failed: {e}")
    import traceback

    traceback.print_exc()
    print(f"[TEST] After error - model device: {next(base_model.parameters()).device}")

print("[TEST] Test complete.")
