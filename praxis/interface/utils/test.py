"""Test utilities for dashboard."""

import random

# Test text with various newline patterns
TEST_TEXT = """The implementation of artificial neural networks has revolutionized machine learning in recent years. Deep learning models have achieved unprecedented success in various tasks, from image recognition to natural language processing. The key to their success lies in their ability to learn hierarchical representations of data through multiple layers of processing.

This architectural approach allows for automatic feature extraction, eliminating the need for manual feature engineering that was previously required in traditional machine learning approaches.


Model training presents its own unique set of challenges. The optimization of neural network parameters requires careful consideration of learning rates, batch sizes, and initialization strategies. Additionally, the choice of activation functions can significantly impact model performance.

Sometimes small changes have big effects.



Gradient descent optimization remains a fundamental technique in deep learning. The process involves calculating partial derivatives with respect to each parameter in the network, enabling the model to adjust its weights in a direction that minimizes the loss function.

The backpropagation algorithm, essential for training deep neural networks, efficiently computes these gradients through the chain rule of calculus.


Regularization techniques play a crucial role in preventing overfitting:
1. Dropout randomly deactivates neurons during training
2. L1 and L2 regularization add penalty terms to the loss function
3. Batch normalization stabilizes the learning process

These methods help ensure the model generalizes well to unseen data.



The architecture of modern neural networks has grown increasingly complex. Transformer models, for instance, have revolutionized natural language processing through their self-attention mechanisms.

This innovation has led to breakthrough models like BERT, GPT, and their successors.


The computational requirements for training large models are substantial:
- High-performance GPUs or TPUs are often necessary
- Distributed training across multiple devices is common
- Memory optimization techniques are crucial

These requirements have driven advances in hardware acceleration and distributed computing.



Recent developments in few-shot learning and meta-learning have opened new possibilities. These approaches allow models to learn from limited examples, more closely mimicking human learning capabilities.

The field continues to evolve rapidly, with new architectures and training methods emerging regularly.


Ethical considerations in AI development have become increasingly important:
- Model bias and fairness
- Environmental impact of large-scale training
- Privacy concerns with data usage

These issues require careful consideration from researchers and practitioners.



The future of deep learning looks promising, with potential applications in:
1. Medical diagnosis and treatment
2. Climate change modeling
3. Autonomous systems
4. Scientific discovery

Each application brings its own unique challenges and opportunities.

The intersection of deep learning with other fields continues to yield interesting results. Quantum computing, for instance, may offer new approaches to optimization problems in neural network training.


This ongoing evolution of the field requires continuous learning and adaptation from practitioners. The rapid pace of development means that today's state-of-the-art might be outdated within months.

Best practices and methodologies must therefore remain flexible and adaptable.



The role of benchmarking and evaluation metrics cannot be overstated. Proper evaluation of model performance requires careful consideration of various metrics:
- Accuracy and precision
- Recall and F1 score
- Computational efficiency
- Model robustness

These metrics help guide development and deployment decisions."""


def get_random_chunks(text, min_size=1, max_size=3):
    """Split text into random-sized chunks."""
    chunks = []
    remaining = text
    while remaining:
        # Random chunk size
        size = random.randint(min_size, max_size)
        chunk = remaining[:size]
        remaining = remaining[size:]
        chunks.append(chunk)
    return chunks