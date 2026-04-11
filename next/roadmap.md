# Praxis Roadmap

Tracked work items for the Praxis research framework. These range from near-term wins to longer-term architectural goals. Items are roughly ordered by priority, not dependency.

---

- [ ] **Environment variable support for CLI arguments**
  Every argparse flag exposed by `main.py` and the subcommand groups under `praxis/cli/` should also be configurable via environment variable, with the CLI flag taking precedence when both are set. This makes it easier to run experiments inside containers, launch scripts, and schedulers (where rewriting command lines is awkward) and to keep secrets and machine-specific paths out of shell history. The convention should be automatic - a single `PRAXIS_<ARG_NAME>` prefix derived from the flag name - so adding a new flag does not require remembering to wire up a matching env var. Downstream tooling (the `launch` script, experiment YAMLs, the web dashboard's run spawner) should be audited to make sure the precedence rules compose cleanly with whatever each layer already injects.

- [ ] **Relocate web assets under `praxis/api/`**
  The web dashboard's source files and Flask templates currently live at the repo root (`src/web/` for JS/CSS sources and the `build.py` bundler, `templates/` for Flask's `index.html`, `static/` for built artifacts), separated from the Flask app that actually serves them in `praxis/api/`. This split makes the relationship between source, build output, and server non-obvious and clutters the project root. Move them into the package so the API owns its own assets:

  ```
  praxis/
    api/
      static/       # built JS, CSS
      templates/    # Flask templates (move from root templates/)
      src/          # JS source files (or just inline if small)
  ```

  The Flask app's `template_folder` and `static_folder` arguments need to be updated, `src/web/build.py` needs its input and output paths repointed, and any references in `launch`, `pyproject.toml` packaging, and `.gitignore` need to follow. Per existing project convention, edits still go through the source files and the dashboard is rebuilt manually with `python praxis/api/src/build.py` - the static directory is never edited by hand.
