#!/usr/bin/env python
"""
Tokenizer training module for Praxis.

This module provides the complete tokenizer training functionality,
including dataset loading, training, and HuggingFace Hub integration.
"""

import argparse
import json
import os
import tempfile
from itertools import islice
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from .standard import StandardTokenizer


def train_tokenizer_cli():
    """Command-line interface for training tokenizers."""
    parser = argparse.ArgumentParser(
        description="Train a Praxis tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["bpe", "unigram"],
        default="unigram",
        help="The type of tokenizer to train",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5_000_000,
        help="The number of examples to train",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
        default=16384,
        help="The absolute vocab size to use",
    )

    args = parser.parse_args()

    # Hardcoded dataset configuration
    dataset_name = "HuggingFaceFW/fineweb"
    dataset_config = "sample-350BT"

    # Train the tokenizer
    print(f"Training {args.type} tokenizer with vocab_size={args.vocab_size}...")
    print(f"Using {args.num_examples:,} examples from {dataset_name}")

    tokenizer = StandardTokenizer.train_from_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        num_examples=args.num_examples,
        vocab_size=args.vocab_size,
        tokenizer_type=args.type,
        dropout=0.1,
    )

    # Save the tokenizer to deterministic locations
    base_path = Path("data/tokenizers")

    # Main save path: data/tokenizers/praxis-{vocab_size}-{type}
    save_path = base_path / f"praxis-{args.vocab_size}-{args.type}"

    os.makedirs(save_path, exist_ok=True)

    tokenizer.save_pretrained(save_path)

    print(f"\n✓ Tokenizer saved to:")
    print(f"  - {save_path}")

    # Always test chat template
    test_chat_template(tokenizer)

    # Always attempt to upload to HuggingFace Hub (gated by auth and user confirmation)
    upload_to_hub(tokenizer, args.vocab_size, args.type)

    return tokenizer


def test_chat_template(tokenizer: PreTrainedTokenizerFast):
    """Test the chat template with a sample conversation."""
    print("\n" + "=" * 60)
    print("Testing Chat Template")
    print("=" * 60)

    # Import tool for testing
    try:
        from praxis.tools import call_tool

        result = call_tool("calc", {"values": [25, 17], "op": "add"})
    except ImportError:
        result = 42  # Fallback if tools not available

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation.",
        },
        {
            "role": "developer",
            "content": "Engage in a scientific discussion, using tools when needed for calculations.",
        },
        {
            "role": "user",
            "content": "I'm studying quantum mechanics. Can you explain superposition?",
        },
        {
            "role": "assistant",
            "content": "Superposition is a fundamental principle where a quantum system exists in multiple states simultaneously until measured. Think of Schrödinger's cat - before observation, it's both alive and dead. When we measure the system, it 'collapses' into one definite state.",
        },
        {
            "role": "user",
            "content": "Interesting! If I have a quantum computer with 25 qubits, and I add 17 more, how many total qubits would I have?",
        },
        {
            "role": "assistant",
            "content": f'Let me calculate that for you.\n<tool_call>\n{{"name": "calc", "arguments": {{"values": [25, 17], "op": "add"}}}}\n</tool_call>\n<tool_result>{result}</tool_result>',
        },
        {
            "role": "assistant",
            "content": f"You would have {result} qubits total. With 42 qubits, your quantum computer could theoretically represent 2^42 (about 4.4 trillion) different states simultaneously - that's the power of quantum superposition at scale!",
        },
    ]

    # Apply chat template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(chat_text)

    # Test tokenization
    tokens = tokenizer.encode(chat_text)
    decoded = tokenizer.decode(tokens)

    print(f"\n✓ Tokenization test:")
    print(f"  Token count: {len(tokens)}")
    print(f"  Roundtrip successful: {chat_text == decoded}")


def upload_to_hub(
    tokenizer: PreTrainedTokenizerFast,
    vocab_size: int,
    tokenizer_type: str,
):
    """Upload tokenizer and chat template to HuggingFace Hub."""
    print("\n" + "=" * 60)
    print("Chat Template Upload to HuggingFace")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi, upload_file
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        print("✗ huggingface_hub not installed")
        print("  Run 'pip install huggingface_hub' to enable auto-upload")
        return

    # Get HF API instance
    api = HfApi()

    # Check if user is authenticated
    try:
        user_info = api.whoami()
        print(f"✓ Authenticated as: {user_info['name']}")
    except Exception:
        print("✗ Not authenticated with HuggingFace")
        print("  Run 'huggingface-cli login' to authenticate")
        return

    # Define all praxis tokenizer repos for different vocab sizes
    vocab_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    repos = [f"UNSAFE/praxis-{size}" for size in vocab_sizes]

    # Check which repos the user has access to
    accessible_repos = []
    for repo_id in repos:
        try:
            repo_info = api.repo_info(repo_id, repo_type="model")
            accessible_repos.append(repo_id)
        except RepositoryNotFoundError:
            print(f"  Skipping {repo_id} - not found")
        except Exception:
            print(f"  Skipping {repo_id} - no access")

    if not accessible_repos:
        print("\n✗ No accessible praxis repos found")
        return

    print(f"\n✓ Found {len(accessible_repos)} accessible repos:")
    for repo in accessible_repos:
        print(f"  - {repo}")

    # Prompt user for confirmation
    response = input(
        "\nDo you want to upload the chat template to these repos? (y/n): "
    )

    if response.lower() != "y":
        print("Skipping upload.")
        return

    # Get chat template
    chat_template = tokenizer.chat_template

    # Upload chat template to each accessible repo as a .jinja file
    print("\nUploading chat_template.jinja files...")

    for repo_id in accessible_repos:
        try:
            # Create a temporary .jinja file with the chat template
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jinja", delete=False
            ) as f:
                f.write(chat_template)
                temp_path = f.name

            # Upload the chat_template.jinja to the repo
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="chat_template.jinja",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add chat_template.jinja with tool support",
            )

            print(f"  ✓ Uploaded chat_template.jinja to {repo_id}")

            # Clean up temp file
            os.remove(temp_path)

        except Exception as e:
            print(f"  ✗ Failed to upload to {repo_id}: {e}")

    print("\n✓ Chat template upload complete!")


def main():
    """Main entry point for the tokenizer training module."""
    train_tokenizer_cli()


if __name__ == "__main__":
    main()
