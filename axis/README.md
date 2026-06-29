# axis

A home for random side projects that live in the Praxis repo but have little or no integration with it. Each subdirectory is a self-contained app with its own toolchain (Godot, Kotlin, ...); Praxis neither imports them nor depends on them. They live here because this is where the work happens, not because they plug into the framework.

## Projects

- **[`axis/`](axis/README.md)** - the original Axis mobile app: an archived Godot companion for controlling Praxis. Largely unmaintained; kept as a record of the idea.
- **[`nutube/`](nutube/README.md)** - a local-first YouTube explorer (Godot, mobile). An experiment in replacing the algorithm with a simple recommender that runs entirely on the device. In early development.
- **[`ghost/`](ghost/README.md)** - a spectral audio visualizer (Godot, desktop). Reads a `.wav` and draws seeded scene definitions modulated by audio features, looping like a recordable video. Procedural, deterministic, no generative AI in the render path. Early scaffold.

## Adding one

Drop a new directory in here with its own README and build setup. Keep it standalone - if a project starts depending on Praxis internals, it probably belongs somewhere else in the repo.
