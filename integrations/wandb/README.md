# Weights & Biases Integration

Experiment tracking and visualization with Weights & Biases.

## Activation

Add the `--wandb` flag when running Praxis:
```bash
python main.py --wandb
```

## Features

- **Real-time Metrics**: Live training metrics streamed to W&B dashboard
- **Experiment Tracking**: Automatic logging of hyperparameters and configuration
- **Loss Visualization**: Track all loss components and training progress
- **Model Checkpointing**: Save and version model checkpoints
- **Comparison Tools**: Compare runs across different experiments
- **Team Collaboration**: Share results with team members

## Setup

1. Install W&B (automatically installed when integration is activated)
2. Login to your W&B account:
   ```bash
   wandb login
   ```
3. Run Praxis with `--wandb` flag

## Configuration

The integration automatically logs:
- Training and validation losses
- Learning rate schedules
- Model architecture details
- Hardware utilization
- Training configuration

## Dashboard

Access your runs at: https://wandb.ai/your-username/praxis

## Custom Logging

The integration provides a logger instance that can be accessed by other components for custom metric logging.