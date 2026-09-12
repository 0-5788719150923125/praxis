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
  train_datasets: [dev]

features:        # Behavioral flags
  skip_compilation: true
```

## Creating Custom Environments

Add a new `.yml` file (e.g., `custom.yml`) and it becomes available as `--custom`. Custom environments are gitignored.

Only one environment can be active at a time. `PRAXIS_DEV=true` activates one the
same way `--dev` does, which is usually the easier half of this for a container.

## Environment variables

Every CLI flag is also an environment variable, `PRAXIS_` + its name uppercased:
`PRAXIS_DEPTH=9`, `PRAXIS_DEVICE=cuda:1`, `PRAXIS_NO_COMPILE=true`,
`PRAXIS_TRAIN_DATASETS=chat,cot` (lists take commas or a JSON array). No
registration needed - a new flag is a new variable. Unparseable values warn and
are skipped rather than failing the run.

Precedence, lowest first: argparse default, experiment YAML, `PRAXIS_*`, explicit
CLI flag, environment file. Experiment toggles are the exception - they are
applied before env vars are read, so `PRAXIS_SMOL=true` does nothing and an
experiment still has to be selected with `--smol`.