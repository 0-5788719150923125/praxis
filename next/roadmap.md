# Praxis Roadmap

Tracked work items for the Praxis research framework. These range from near-term wins to longer-term architectural goals. Items are roughly ordered by priority, not dependency.

---

- [ ] **Finish retraining tokenizers at varied sizes**
      Complete the in-progress work of retraining the project's tokenizers across a range of vocabulary sizes so each model scale has a tokenizer that actually fits it, rather than reusing a single vocab across small, medium, and large configurations. Varied sizes let us study the interaction between vocab size, embedding parameter count, and downstream loss without confounding it against the tokenizer's training corpus. Tracked in [0-5788719150923125/praxis#50](https://github.com/0-5788719150923125/praxis/issues/50).

- [x] **Environment variable support for CLI arguments**
      Every argparse flag (including dynamically added ones from integrations and experiment/environment discovery) is now settable via `PRAXIS_<DEST>` env vars. `EnvVarLoader` (`praxis/cli/loaders/env_vars.py`) iterates `parser._actions` after parsing, coerces the raw string to the action's declared type (bool-ish for store*true, type callable for scalars, comma or JSON list for `nargs`/`append`/`extend`, validated against `choices`), and applies it unless the CLI flag was given explicitly. Precedence top-down: explicit CLI > active environment YAML > `PRAXIS*\*` env var > experiment YAML > argparse default.

- [x] **Relocate web assets under `praxis/web/`**
      Moved `src/web/` (JS/CSS sources + build.py), `templates/`, and `static/` image assets into `praxis/web/`. Renamed the package from `praxis.api` to `praxis.web` since it serves both the API and the dashboard frontend. The build script now lives at `praxis/web/src/build.py` and outputs to `praxis/web/static/`.

- [ ] **Multi-path `--data-path` support in experiment YAMLs**
      Allow experiment YAML files to specify `data_path` as a list of strings (e.g. `data_path: [/data/a, /data/b]`). The CLI and `MultiDirectoryDataset` already handle multiple paths, but the experiment loader needs to normalize scalar values to a list, define merge-vs-replace semantics when both YAML and CLI provide paths, and the `launch` script needs to discover YAML-sourced paths so it can create the corresponding Docker volume mounts.

- [ ] **Fix metrics loading performance for long runs**
      As training progresses, the `/api/metrics` endpoint gets progressively slower because it re-reads and re-downsamples (LTTB) the full metrics history from SQLite on every request. Investigate server-side caching of the downsampled response (invalidated on new writes), pre-materialized summary tables, or incremental approaches where only new rows since the last request need processing. The frontend's ETag/304 caching helps with repeated identical requests but does nothing for the first fetch after new data arrives.

- [ ] **Harden integration error handling**
      The integration loader (`praxis/integrations/loader.py`) and individual integrations silently swallow exceptions at every level - spec discovery, module loading, hook execution, and runtime errors all land in bare `except Exception: pass` or quiet `print` calls. This hides real bugs (e.g. a stale `from praxis import api` import went unnoticed because the Discord integration caught the `ImportError` and printed a one-liner). Integrations should fail loudly by default, distinguishing between import/setup errors (which are always bugs and should raise) and runtime errors (which may warrant graceful degradation). Consider a `critical` flag in `spec.yaml` so truly optional integrations can degrade without stopping training, while misconfigured ones surface immediately.

- [ ] **Unify registry pattern with a lightweight `Registry` class**
      Replace the ~22 ad-hoc `*_REGISTRY` dicts with a thin `Registry` subclass of `dict` that standardizes error messages, supports optional decorator-based registration, and provides a consistent typed interface. Migration should be incremental - the class is still a dict, so existing `REGISTRY.keys()` and `REGISTRY[name](config)` call sites stay unchanged. Also audit `SAMPLER_REGISTRY` (maps strings to strings) and `EMBEDDING_REGISTRY` (maps every block type to the same class) as candidates for removal.

- [x] **Modularize dataset management with named collections**
      `--phi`, `--pile`, `--no-source`, and the `minimal_data` feature flag are gone; selection now flows through two CLI list flags, `--train-datasets` and `--validation-datasets`, each accepting space- or comma-separated collection names (e.g. `--train-datasets base phi`, `--train-datasets base,phi`). Collections live in `DATASET_COLLECTIONS` in `praxis/data/config.py`; individual entries in `DATASETS` carry a `type` field (`huggingface`, `directory`, `synthetic-tool-calling`) so that what used to be hard-coded branches - the `./` source tree via the old `"self"` format, the synthetic tool-calling gate keyed on `"phi" in datasets`, and the auto-`dev` swap under `minimal_data` - are now just typed dataset entries dropped into whichever collection you want. `environments/dev.yml` sets `train_datasets: [dev]` directly; the 12 experiment YAMLs that previously set `phi: true` now set `train_datasets: [base, phi]`. Individual datasets are addressable as collection members today; bare-dataset selection via the CLI (without a collection wrapper) remains a future-nice-to-have.
