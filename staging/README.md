# staging/

Welcome to the junkyard! This is where we dump code that don't belong in core Praxis.

## What goes here?

Anything that adds functionality without cluttering the main codebase:

- **External integrations** (wandb, tensorboard, whatever)
- **Experimental features** that might break things
- **Third-party adapters** (like our funky GUN chat thing)
- **Plugin-style modules** that some people want but others don't
- **Random, orphaned code** can go here as well

## How it works

Each module is a folder with a `module.yaml` manifest. Praxis discovers and loads them automatically based on CLI flags or conditions.

```
staging/
├── wandb/
│   ├── module.yaml    # "I provide --wandb args and logging"
│   └── __init__.py    # The actual code
└── your_thing/
    ├── module.yaml    # "I do X when Y is enabled"
    └── __init__.py
```

Your module can hook into:

- **CLI args** - Add your own `--flags`
- **Loggers** - Custom logging implementations
- **Datasets** - New data sources
- **Lifecycle** - Init/cleanup when things start/stop
- **Cleanup** - Directories to nuke on `--reset`

## The deal

Keep your mess contained. If it's useful enough, maybe it graduates to core. If not, at least it's not breaking anyone else's stuff.

Questions? Check the wandb module as an example. It's probably the least broken thing in here.
