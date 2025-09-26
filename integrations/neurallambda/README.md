# NeuralLambda Integration for Praxis

This integration provides differentiable stack-based memory and neural reasoning capabilities for advanced routing in Praxis models.

## Features

- **NeuralController**: A controller that uses NeuralLambda's differentiable stack for maintaining routing memory
- **Stack Memory**: Differentiable stack operations (push, pop, nop) for tracking layer visitation history
- **Tool Selection**: Dynamic activation function selection for processing hidden states
- **Trainable Tools**: Combination of fixed activation functions and trainable neural network tools

## Installation

The neurallambda integration is automatically installed when using the `--neurallambda` flag.

## Usage

To enable NeuralLambda-based routing in your model:

```bash
python main.py --neurallambda --controller-type neural
```

## Configuration

The NeuralController supports the following configuration options (inherited from model config):
- `hidden_size`: Model hidden dimension
- `depth`: Number of layers (determines stack depth)
- `num_experts`: Number of expert modules available for routing

## Architecture

The integration provides:

1. **NeuralController**: Advanced routing controller using stack memory
   - Maintains differentiable stack for routing history
   - Selects from multiple activation functions (tools)
   - Combines stack memory with tool outputs for routing decisions

2. **Stack Operations**:
   - Push: Add layer embedding to stack
   - Pop: Remove from stack
   - Nop: No operation
   - All operations are differentiable

3. **Tools**:
   - Fixed tools: ReLU, Tanh, Sigmoid, Sin, Identity
   - Trainable tools: Linear layers and MLPs