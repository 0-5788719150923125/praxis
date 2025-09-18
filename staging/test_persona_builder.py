#!/usr/bin/env python3
"""Quick test script for persona-chat builder format."""

import argparse
import os
import random
import sys
import time

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from praxis import PraxisConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from praxis.data.formatters import format_personachat

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test persona-chat builder format")
parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
args = parser.parse_args()

# Use provided seed or generate from time
seed = args.seed if args.seed is not None else int(time.time())
random.seed(seed)

# Load tokenizer - use the 4096 vocab size version
print("Loading tokenizer...")
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
vocab_size = 4096

# Try the same paths as run.py
possible_paths = [
    os.path.join(cache_dir, "model"),
    f"UNSAFE/praxis-{vocab_size}",
]

tokenizer = None
for path in possible_paths:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
        print(f"Loaded tokenizer from: {path}")
        break
    except Exception as e:
        print(f"No tokenizer found at: {path}")

if tokenizer is None:
    raise ValueError("Could not load tokenizer from any of the expected paths")

# Load and shuffle dataset properly
print(f"Loading dataset with seed {seed}...")
ds = load_dataset("AlekseyKorshuk/persona-chat", split="train", streaming=True)

# Shuffle dataset with the random seed
ds_shuffled = ds.shuffle(seed=seed, buffer_size=1000)

# Get one sample
sample = next(iter(ds_shuffled))

# Format the sample
print("\n" + "=" * 80)
print("FORMATTED OUTPUT:")
print("=" * 80 + "\n")

formatted = format_personachat(sample, ["personality", "utterances"], tokenizer)
print(formatted)

# Optional: Enable debug mode to trace specific issues
DEBUG = False
if DEBUG:
    print("\n" + "=" * 80)
    print("DEBUG: TRUECASING PROCESS:")
    print("=" * 80)

    import re

    from praxis.data.formatters import simple_truecase

    test_text = "like most other humans . . . ? d"
    print(f"Raw text: {repr(test_text)}")
    print(f"After truecasing: {repr(simple_truecase(test_text))}")

print("\n" + "=" * 80)
print("SAMPLE INFO:")
print("=" * 80)
print(f"Personality traits: {len(sample['personality'])} traits")
for trait in sample["personality"]:
    print(f"  - {trait}")
print(f"\nUtterances in dataset: {len(sample['utterances'])}")
print(f"Output length: {len(formatted)} characters")
print(f"Output lines: {len(formatted.splitlines())}")
