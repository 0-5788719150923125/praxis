# Experiments

This directory contains experiment configurations for Praxis. Each `.yml` file defines a preset combination of CLI arguments.

## Usage

Each experiment file automatically becomes a CLI flag:

- `alpha.yml` → `--alpha`
- `my-experiment.yml` → `--my-experiment`

Run an experiment:

```bash
./launch --alpha
# or
python main.py --alpha --batch-size 32  # Override individual settings
```

## Creating Experiments

Create a new `.yml` file with your desired settings:

```yaml
# experiments/beta.yml
batch_size: 4
depth: 2
device: cuda
hidden_size: 128
```

Then use it: `./launch --beta`

**Note:** Only `alpha.yml` is tracked in git. Other experiments are gitignored for local customization.
