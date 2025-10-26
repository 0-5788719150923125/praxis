# Cut Cross-Entropy Integration

This integration provides Apple's memory-efficient [cut-cross-entropy](https://github.com/apple/ml-cross-entropy) loss function for Praxis.

## Features

- **Memory Efficient**: Avoids materializing full logit matrices during training
- **Fused Kernels**: Uses Triton-based implementations for optimal GPU performance
- **Automatic Shifting**: Handles sequence shifting internally without allocating intermediate tensors
- **Tied Weights Support**: Works seamlessly with tied embeddings and different hidden/embed sizes
- **Smart Logits Computation**: Skips logits projection during training, computes only during validation/inference

## Installation

This integration is automatically installed with Praxis. To use it:

```yaml
# In your experiment config
loss_func: cut_cross_entropy
```

## Requirements

- CUDA-capable GPU (required for Triton kernels)
- Modern NVIDIA GPU with compute capability >= 7.0
- **Large vocabulary models** (32k-256k tokens) for significant benefits

## When to Use

✅ **Best for:**
- Large vocabulary models (GPT-3, LLaMA scale: 32k-256k vocab)
- Long sequence lengths (2048+ tokens)
- Memory-constrained training

❌ **NOT recommended for:**
- Small vocabulary models (<10k vocab size)
- Short sequences
- Models where vocabulary size is small relative to hidden size

## How It Works

### 1. Avoiding Shifted Tensor Allocation

```python
# Traditional approach (allocates shifted tensors):
shifted_embeddings = embeddings[..., :-1, :].contiguous()  # ❌ Extra memory
shifted_labels = labels[..., 1:].contiguous()              # ❌ Extra memory
loss = cross_entropy(shifted_logits, shifted_labels)

# cut_cross_entropy approach (zero-copy shifting):
loss = cut_cross_entropy(embeddings, labels, shift=1)     # ✅ No allocation
```

### 2. Avoiding Logits Materialization

**During Training:**
```python
# Traditional:
logits = hidden_states @ classifier.weight.T  # ❌ Materialize full matrix
loss = cross_entropy(logits, labels)

# cut_cross_entropy:
# Skip logits computation entirely!
loss = cut_cross_entropy(hidden_states, classifier.weight, labels)  # ✅ Fused kernel
```

**During Validation/Inference:**
```python
# Logits ARE computed for metrics (perplexity, generation, etc.)
logits = hidden_states @ classifier.weight.T
# But loss still uses cut_cross_entropy
```

This dual-mode operation is handled automatically by Praxis.

## Performance

For a sequence length of 2048, hidden_size 4096, batch_size 32, **vocab_size 50k**:

**Memory Savings:**
- Shifted tensors: ~150 MB per batch
- Logit matrix: **~12 GB per batch** (main benefit!)
- **Total: ~12 GB savings per batch**

**Time:**
- Fused Triton kernels are typically 10-20% faster than separate operations

**Note:** Benefits scale with vocabulary size. Small vocabs (<10k) see minimal gains and may actually use more memory due to Triton kernel overhead.

## References

- [Apple ML Cross-Entropy GitHub](https://github.com/apple/ml-cross-entropy)
- [Triton Language](https://triton-lang.org/)
