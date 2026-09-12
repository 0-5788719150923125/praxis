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

## The documented format

The experiments this repo commits (`alpha`, `beta`, `gpt2-1`, `smol`) are kept in
one generated format, the same one the web app's Download button produces: every
key carries the flag's description, what it accepts, what it defaults to and what
the chosen value means, then a blank line, then the key. Every other flag follows
at the bottom, commented out at its default, so a reader can see the whole
surface without `--help`.

`./launch` rewrites those files on startup, and `python tools/format_experiments.py`
does the same pass on demand (`--check` reports without writing). Only git-tracked
files are touched - your own experiments are gitignored and never read.

Two things to know when editing a committed experiment:

- The prose under each key is generated. An edit to it is overwritten.
- A comment you write by hand is kept. It is folded into a `Note:` paragraph
  above the key below it, which is where experiment-specific rationale belongs
  (why this arm pins `norm_type: sandwich_tied`, what a measurement was taken
  at). A comment with no key below it becomes the file's own note, in the header.

A file with a key that nothing in the checkout describes - no flag of that name
and no registry namespace either - is left alone rather than rewritten into a
shape that depends on how the process started. `--check` names the key.

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
