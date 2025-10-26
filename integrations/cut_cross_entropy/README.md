# Cut Cross-Entropy Integration

This integration provides Apple's memory-efficient [cut-cross-entropy](https://github.com/apple/ml-cross-entropy) loss function for Praxis.

## Features

- **Memory Efficient**: Avoids materializing full logit matrices during training
- **Fused Kernels**: Uses Triton-based implementations for optimal GPU performance
- **Automatic Shifting**: Handles sequence shifting internally without allocating intermediate tensors
- **Tied Weights Support**: Works seamlessly with tied embeddings and different hidden/embed sizes

## Installation

This integration is automatically installed with Praxis. To use it:

```yaml
# In your experiment config
loss_func: cut_cross_entropy
```

## Requirements

- CUDA-capable GPU (required for Triton kernels)
- Modern NVIDIA GPU with compute capability >= 7.0

## How It Works

The cut-cross-entropy loss uses the `shift=1` parameter to perform sequence shifting internally:

```python
# Traditional approach (allocates shifted tensors):
shifted_embeddings = embeddings[..., :-1, :].contiguous()  # ❌ Extra memory
shifted_labels = labels[..., 1:].contiguous()              # ❌ Extra memory

# cut_cross_entropy approach (zero-copy shifting):
loss = cut_cross_entropy(embeddings, labels, shift=1)     # ✅ No allocation
```

This saves significant memory during training, especially for long sequences.

## Performance

For a typical sequence length of 2048 with hidden_size 384 and batch_size 64:
- **Memory Savings**: ~150 MB per batch from avoided shifted tensor allocation
- **Additional Savings**: Avoids materializing full (batch × seq_len × vocab_size) logit matrix

## References

- [Apple ML Cross-Entropy GitHub](https://github.com/apple/ml-cross-entropy)
- [Triton Language](https://triton-lang.org/)
