# Praxis Roadmap

Tracked work items for the Praxis research framework. These range from near-term wins to longer-term architectural goals. Items are roughly ordered by priority, not dependency.

---

- [ ] **Environment variable support for CLI arguments**
  Every argparse flag exposed by `main.py` and the subcommand groups under `praxis/cli/` should also be configurable via environment variable, with the CLI flag taking precedence when both are set. This makes it easier to run experiments inside containers, launch scripts, and schedulers (where rewriting command lines is awkward) and to keep secrets and machine-specific paths out of shell history. The convention should be automatic - a single `PRAXIS_<ARG_NAME>` prefix derived from the flag name - so adding a new flag does not require remembering to wire up a matching env var. Downstream tooling (the `launch` script, experiment YAMLs, the web dashboard's run spawner) should be audited to make sure the precedence rules compose cleanly with whatever each layer already injects.

- [x] **Relocate web assets under `praxis/web/`**
  Moved `src/web/` (JS/CSS sources + build.py), `templates/`, and `static/` image assets into `praxis/web/`. Renamed the package from `praxis.api` to `praxis.web` since it serves both the API and the dashboard frontend. The build script now lives at `praxis/web/src/build.py` and outputs to `praxis/web/static/`.

- [ ] **Finish retraining tokenizers at varied sizes**
  Complete the in-progress work of retraining the project's tokenizers across a range of vocabulary sizes so each model scale has a tokenizer that actually fits it, rather than reusing a single vocab across small, medium, and large configurations. Varied sizes let us study the interaction between vocab size, embedding parameter count, and downstream loss without confounding it against the tokenizer's training corpus. Tracked in [0-5788719150923125/praxis#50](https://github.com/0-5788719150923125/praxis/issues/50).
