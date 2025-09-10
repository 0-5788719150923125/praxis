# Environments

This directory contains environment configurations that provide different presets for running Praxis.

## What are Environments?

Environments are high-priority configurations that override all other settings (defaults, experiments, and CLI args). They provide:

1. **Overrides**: Direct parameter overrides (e.g., `depth: 3`)
2. **Features**: Behavioral flags that control runtime features (e.g., `skip_compilation: true`)

## Usage

Each `.yml` file in this directory becomes a CLI flag:

```bash
# Development environment (fast iteration, small model)
./launch --dev
```

## Priority Order

Settings are applied in this order (later overwrites earlier):

1. Default values
2. Experiment configurations (`--alpha`, etc.)
3. CLI arguments
4. **Environment overrides** (highest priority)

## Creating Custom Environments

Create a new `.yml` file with your desired configuration:

```yaml
# environments/custom.yml
overrides:
  batch_size: 32
  device: cuda:0
  
features:
  skip_compilation: false
  detect_anomaly: true
```

Then use it: `./launch --custom`

## Environment Features

Features control runtime behavior without changing model architecture:

- `force_reset`: Skip checkpoint loading, always start fresh
- `detect_anomaly`: Enable PyTorch anomaly detection
- `skip_compilation`: Skip torch.compile optimization
- `minimal_data`: Use reduced datasets
- `cache_isolation`: Use separate cache directory
- `deterministic`: Enable deterministic algorithms (if needed)

## Mutual Exclusivity

Only one environment can be active at a time. Using multiple environment flags will raise an error:

```bash
./launch --dev --custom  # ERROR: Cannot use multiple environments
```

## Default Environment

- **dev**: Fast development iteration with a small model (3 layers), no compilation, minimal datasets, and isolated cache. This is the only environment shipped with Praxis.

## Difference from Experiments

- **Experiments** (`experiments/` directory): Applied before CLI args, can be overridden
- **Environments** (`environments/` directory): Applied after everything, cannot be overridden

## Custom Environments

You can create your own environments by adding new `.yml` files to this directory. Custom environment files are gitignored, so they won't be committed to the repository.