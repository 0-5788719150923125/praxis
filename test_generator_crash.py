#!/usr/bin/env python
"""Test script to reproduce the generator crash with messages."""

import torch
from praxis import PraxisConfig, PraxisForCausalLM
from praxis.generation.generator import Generator
from praxis.tokenizers import create_tokenizer

print("[TEST] Creating model and tokenizer...")

# Create a small model for testing
config = PraxisConfig(
    depth=2,
    hidden_size=64,
    embed_size=64,
    num_heads=2,
    vocab_size=256,
    decoder_type="sequential",
)

# Create model on CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TEST] Using device: {device}")

model = PraxisForCausalLM(config)
if device == "cuda":
    model = model.to(device)
    print(f"[TEST] Model moved to {device}")

# Create tokenizer
tokenizer = create_tokenizer(vocab_size=256)

# Create generator
generator = Generator(model, tokenizer, device=device)

# Test messages
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation.",
    },
    {"role": "developer", "content": "Write thy wrong."},
    {"role": "user", "content": "Is there anybody there?"},
]

print("[TEST] Testing generation with messages...")
try:
    result = generator.generate_with_messages(
        messages,
        max_new_tokens=10,
        temperature=0.4,
        repetition_penalty=1.15,
        do_sample=True,
        use_cache=False,
        skip_special_tokens=False,
    )
    print(f"[TEST] Generation successful: {result[:100]}...")
except Exception as e:
    print(f"[TEST] Generation failed: {e}")
    import traceback

    traceback.print_exc()

print("[TEST] Test complete.")
