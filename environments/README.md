# Environments

Environment configurations override all other settings (defaults, experiments, and CLI args) to provide controlled presets for different use cases.

## Usage

Each `.yml` file in this directory becomes a CLI flag:

```bash
./launch --dev   # Uses dev.yml configuration
```

## Structure

```yaml
overrides:       # Parameter overrides
  depth: 3
  batch_size: 1
  
features:        # Behavioral flags  
  skip_compilation: true
  minimal_data: true
```

## Creating Custom Environments

Add a new `.yml` file (e.g., `custom.yml`) and it becomes available as `--custom`. Custom environments are gitignored.

Only one environment can be active at a time.