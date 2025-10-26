# Praxis Integrations

This directory contains optional integrations that extend Praxis with additional functionality. Each integration is self-contained and can be automatically loaded based on CLI flags or conditions.

## Available Integrations

Each subdirectory contains an integration with its own README.md describing its purpose and usage.

## Integration Structure

Each integration follows a standard structure:

```
integration-name/
├── __init__.py       # Main integration module
├── spec.yaml         # Integration specification
└── pyproject.toml    # Python dependencies (optional, preferred over spec.yaml dependencies)
```

## Creating a New Integration

1. Create a new directory in `/integrations/` with your integration name
2. Add a `spec.yaml` file defining the integration:

```yaml
name: your-integration
version: 1.0.0
description: Brief description of what your integration does
conditions:
  - "args.your_flag" # Activation condition
provides:
  - cli_args # If adding CLI arguments
  - lifecycle # If providing init/cleanup hooks
  - loggers # If providing logging functionality
  - datasets # If providing dataset loaders
# Optional: Dependencies can be specified here or in pyproject.toml
dependencies:
  python:
    - package-name>=1.0.0
```

3. (Recommended) Add a minimal `pyproject.toml` for dependencies:

```toml
[project]
name = "praxis-integration-your-integration"
version = "0.1.0"
dependencies = [
    "package-name>=1.0.0",
]
```

4. Implement the integration in `__init__.py`:

```python
from praxis.integrations.base import BaseIntegration

class YourIntegration(BaseIntegration):
    def add_cli_args(self, parser):
        """Add CLI arguments for this integration."""
        parser.add_argument('--your-flag', action='store_true',
                          help='Enable your integration')

    def initialize(self, args, cache_dir, **kwargs):
        """Initialize the integration."""
        if args.your_flag:
            # Setup code here
            pass

    def cleanup(self):
        """Clean up resources."""
        pass
```

## Integration Loading

Integrations are automatically discovered and loaded by the `IntegrationLoader` class:

- Discovery happens at startup by scanning this directory
- Integrations are loaded conditionally based on their `spec.yaml` conditions
- Dependencies are automatically installed if missing
- Multiple integrations can be active simultaneously

## Best Practices

1. **Self-contained**: Keep all integration code within its directory
2. **Conditional loading**: Use conditions to avoid loading when not needed
3. **Clean shutdown**: Implement cleanup methods for proper resource management
4. **Documentation**: Include clear documentation in your integration
5. **Dependencies**: Use pyproject.toml for dependencies (preferred) or spec.yaml as fallback
6. **Error handling**: Gracefully handle missing dependencies or initialization failures
7. **Package naming**: Use `praxis-integration-{name}` convention for pyproject.toml package names

## Testing Integrations

To test if integrations are loading correctly:

```bash
# List available integrations
python -c "from praxis.integrations.loader import IntegrationLoader; loader = IntegrationLoader(); specs = loader.discover_integrations(); print([s.name for s in specs])"

# Run with specific integration
python main.py --ngrok  # or --wandb, --gun, --quantum
```

## Troubleshooting

- If an integration isn't loading, check that:

  - The `spec.yaml` file is valid YAML
  - The conditions in spec.yaml match your CLI flags
  - Required dependencies are installed
  - The `__init__.py` follows the correct structure

- Integration logs appear in the console with `[INTEGRATIONS]` prefix
- Failed integrations are silently skipped to avoid breaking the main application
