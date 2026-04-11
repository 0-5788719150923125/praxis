# Praxis Roadmap

Tracked work items for the Praxis research framework. These range from near-term wins to longer-term architectural goals. Items are roughly ordered by priority, not dependency.

---

- [ ] **Environment variable support for CLI arguments**
  Every argparse flag exposed by `main.py` and the subcommand groups under `praxis/cli/` should also be configurable via environment variable, with the CLI flag taking precedence when both are set. This makes it easier to run experiments inside containers, launch scripts, and schedulers (where rewriting command lines is awkward) and to keep secrets and machine-specific paths out of shell history. The convention should be automatic - a single `PRAXIS_<ARG_NAME>` prefix derived from the flag name - so adding a new flag does not require remembering to wire up a matching env var. Downstream tooling (the `launch` script, experiment YAMLs, the web dashboard's run spawner) should be audited to make sure the precedence rules compose cleanly with whatever each layer already injects.
