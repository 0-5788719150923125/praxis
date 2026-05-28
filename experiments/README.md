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
./launch --alpha --batch-size 32  # Override individual settings
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

**Note:** Most experiments are gitignored to allow for local customization.

## Inheriting from another experiment

An experiment can inherit settings from one or more other experiments via the `extends` keyword:

```yaml
# experiments/delta-12.yml
extends: delta
depth: 12
num_layers: 4
```

`extends` accepts a single experiment name (by stem, matching its `.yml` filename) or a list:

```yaml
extends:
  - base
  - overrides
batch_size: 64
```

List entries merge left-to-right; the current file overrides all of them. Chains (A extends B extends C) are resolved recursively. The rendered config is flattened - the `extends` key is stripped before the config is applied or served over the API, so downstream consumers (like the web dashboard's `/api/config` endpoint) only ever see a single, complete config.
