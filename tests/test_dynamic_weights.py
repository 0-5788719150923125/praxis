"""Test dynamic weighting with new per-document tokenization."""

from unittest.mock import Mock

from transformers import AutoTokenizer

from praxis.data.datasets.manager import InterleaveDataManager


def test_dynamic_weights_with_different_doc_sizes():
    """Verify datasets with huge documents get downweighted."""

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[BOS]', '[SEP]', '[PAD]']
    })
    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Create mock samplers
    # Sampler 1: Small documents (2 messages each)
    sampler1 = Mock()
    sampler1.dataset_path = "small-dataset"

    # Sampler 2: HUGE documents (50 messages each - like open-phi)
    sampler2 = Mock()
    sampler2.dataset_path = "open-phi-huge"

    # Configure what they return
    small_doc_call_count = [0]
    huge_doc_call_count = [0]

    def get_small_doc():
        small_doc_call_count[0] += 1
        return {
            "messages": [
                {"role": "user", "content": f"Question {small_doc_call_count[0]}"},
                {"role": "assistant", "content": f"Answer {small_doc_call_count[0]}"}
            ],
            "metadata": {"source": "small"}
        }

    def get_huge_doc():
        huge_doc_call_count[0] += 1
        # Simulate open-phi style huge documents with 50 messages
        messages = []
        for i in range(50):
            messages.append({"role": "user", "content": f"Part {i} of huge document {huge_doc_call_count[0]}"})
            messages.append({"role": "assistant", "content": f"Response to part {i}"})
        return {
            "messages": messages,
            "metadata": {"source": "huge"}
        }

    sampler1.get_document = get_small_doc
    sampler2.get_document = get_huge_doc

    # Create manager with equal initial weights
    manager = InterleaveDataManager(
        samplers=[sampler1, sampler2],
        weights=[0.5, 0.5],  # Start equal
        tokenizer=tokenizer,
        block_size=128,
        rl_type=None
    )

    # Verify dynamic weights are enabled
    assert manager.use_dynamic_weights, "Dynamic weights should be enabled"

    print(f"\nInitial weights: {manager.weights}")
    print(f"Initial metrics:")
    for idx, metrics in manager.sampler_metrics.items():
        print(f"  Sampler {idx} ({metrics['name']}): avg_doc_length={metrics['avg_doc_length']}")

    # Trigger some document sampling
    # This will call _refill_message_queue which updates weights
    for i in range(10):
        batch = manager.get_batch(batch_size=2)

        if i % 3 == 0:  # Print every 3rd iteration
            print(f"\nAfter {i+1} batches:")
            print(f"  Current weights: {manager.weights}")
            print(f"  Metrics:")
            for idx, metrics in manager.sampler_metrics.items():
                avg_len = f"{metrics['avg_doc_length']:.1f}" if metrics['avg_doc_length'] else 'None'
                print(f"    {metrics['name']}: avg_doc_length={avg_len}, total_samples={metrics['total_samples']}")

    # Check final state
    print(f"\n{'='*60}")
    print(f"FINAL STATE after 10 batches:")
    print(f"{'='*60}")
    print(f"Weights: {manager.weights}")

    for idx, metrics in manager.sampler_metrics.items():
        print(f"\nSampler {idx} ({metrics['name']}):")
        avg_len = f"{metrics['avg_doc_length']:.1f}" if metrics['avg_doc_length'] else 'None'
        print(f"  Average doc length (estimate): {avg_len}")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Total tokens (estimate): {metrics['total_tokens']}")

    # Verify: The huge-document dataset should have LOWER weight
    small_weight = manager.weights[0]
    huge_weight = manager.weights[1]

    print(f"\n{'='*60}")
    print(f"WEIGHT COMPARISON:")
    print(f"  Small docs weight: {small_weight:.4f}")
    print(f"  Huge docs weight:  {huge_weight:.4f}")
    print(f"  Ratio (small/huge): {small_weight/huge_weight:.2f}x")
    print(f"{'='*60}")

    # The small-doc dataset should have higher weight (it gets sampled more)
    assert small_weight > huge_weight, \
        f"Small docs should have higher weight! small={small_weight:.4f}, huge={huge_weight:.4f}"

    # Check that the ratio is significant (at least 1.5x)
    ratio = small_weight / huge_weight
    assert ratio >= 1.5, \
        f"Weight ratio should be significant (>=1.5x), got {ratio:.2f}x"

    print(f"\n✓ Dynamic weighting correctly downweights huge documents!")
    print(f"✓ Small docs get {ratio:.2f}x more weight than huge docs")


def test_current_estimate_accuracy():
    """Check how accurate the current estimate is vs actual token counts."""

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[BOS]', '[SEP]', '[PAD]']
    })
    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Test documents of varying sizes
    test_cases = [
        {
            "name": "Small (2 msgs, short content)",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]
        },
        {
            "name": "Medium (4 msgs, medium content)",
            "messages": [
                {"role": "user", "content": "Can you explain quantum mechanics?"},
                {"role": "assistant", "content": "Quantum mechanics is the study of matter at atomic scales."},
                {"role": "user", "content": "What about wave-particle duality?"},
                {"role": "assistant", "content": "It describes how particles exhibit both wave and particle properties."}
            ]
        },
        {
            "name": "Huge (100 msgs, typical open-phi)",
            "messages": [
                {"role": "user", "content": f"Question {i}"} if i % 2 == 0
                else {"role": "assistant", "content": f"Answer to question {i//2}"}
                for i in range(100)
            ]
        }
    ]

    print(f"\n{'='*70}")
    print(f"ESTIMATE ACCURACY TEST")
    print(f"{'='*70}")

    for test_case in test_cases:
        messages = test_case["messages"]

        # Current estimate: num_messages * 50
        estimate = len(messages) * 50

        # Actual token count
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=False)["input_ids"]
        actual = tokens.shape[1]

        error_pct = abs(actual - estimate) / actual * 100

        print(f"\n{test_case['name']}:")
        print(f"  Messages: {len(messages)}")
        print(f"  Estimate: {estimate} tokens")
        print(f"  Actual:   {actual} tokens")
        print(f"  Error:    {error_pct:.1f}%")

    print(f"\n{'='*70}")
    print(f"NOTE: Large errors suggest we should use actual token counts!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_dynamic_weights_with_different_doc_sizes()
    test_current_estimate_accuracy()
